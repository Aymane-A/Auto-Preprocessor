"""Isolated dataset inspection, transformation and export for the local web API."""

import json
import math
import csv
import os
import hashlib
import time
import shutil
import calendar
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from resampling import class_profile, rebalance

from app_preprocess import (
    Options, RunResult, _looks_like_classification, _with_target, analysis_recommendation_summary, build_config,
    build_preview_table, choices_to_overrides, label_action, label_step,
    preview_for_choice, read_table, run_preprocessing,
    parse_step_parameters, step_parameter_visible, validate_step_choice, recommendation_help,
)
from auto_preprocessor import AutoPreprocessor, ColumnOverride, _override_as_choices, _unique_integer_identifier, COMMON_ORDINAL_PATTERNS
from statistical_profile import association, adjusted_p_values, parse_dates, text_evidence
from web_contracts import WebOptions

MATRIX_LIMIT = 40
CATEGORY_LIMIT = 12
RELATION_LIMIT = 20_000
DOWNLOADS = {"train": "train.csv", "test": "test.csv", "full": "full.csv", "train_provenance": "train_provenance.csv", "report_xlsx": "rapport.xlsx",
             "report_json": "report.json", "full_xlsx": "dataset_transforme.xlsx", "preprocessor": "preprocesseur.joblib"}

PARAMETER_HINTS = {
    "KNN": "Nombre de voisins de 1 à 1 000, partagé par les variables KNN (facultatif)",
    "QUANTILE": "Nombre de quantiles de 2 à 50 000 (facultatif)",
    **{name: "Multiplicateur IQR strictement positif (facultatif)" for name in ("IQR", "MEDIAN_REPLACE", "TO_MISSING", "REPLACE_MEDIAN", "OUTLIER_TO_NAN")},
    **{name: "Seuil strictement positif (facultatif)" for name in ("ZSCORE", "Z_SCORE", "MODIFIED_Z", "MODIFIED_ZSCORE")},
    **{name: "N-grammes min, max et taille facultative, par exemple 1, 2, 16" for name in ("TFIDF_NGRAM", "COUNT", "HASH")},
    "SELECTED_TO_NAN": "Marqueurs séparés par des virgules, par exemple ?, inconnu",
    "CONSTANT": "Valeur de remplacement, numérique pour une variable numérique",
    "WINSORIZE": "Quantiles inférieur, supérieur, par exemple 0.01, 0.99",
    "ORDINAL": "Modalités ordonnées séparées par des virgules",
    "MANUAL_MAP": 'Objet JSON, par exemple {"faible": 0, "fort": 1}',
    "ONEHOT": "Fréquence minimale des modalités entre 0 et 1 (facultatif)",
    "HASH_ENCODE": "Nombre de colonnes de hachage (facultatif)",
    "LOO_ENCODE": "Bruit sigma positif ou nul (facultatif)",
    "TFIDF_SVD": "Nombre de composantes SVD, 0 pour désactiver",
    "TFIDF_NGRAMS": "N-grammes min, max et taille facultative, par exemple 1, 2, 15",
    "COUNT_VECTORIZER": "N-grammes min, max et taille facultative, par exemple 1, 2, 15",
    "HASH_TEXT": "N-grammes min, max et taille facultative, par exemple 1, 2, 32",
}


def processing_options(task, values=None):
    """Pass public options consistently into analysis, preview and final fitting."""
    values = WebOptions.model_validate(values or {}).model_dump(exclude={"imbalance_method", "imbalance_ratio", "imbalance_neighbors"})
    standard = {field.name for field in fields(Options)} - {"task", "engine_options"}
    return Options(task=task, **{key: value for key, value in values.items() if key in standard},
                   engine_options={key: value for key, value in values.items() if key not in standard})


class InputError(ValueError):
    """A dataset or selection error safe to show to the user."""


def json_safe(value, key=""):
    """Convert pandas and NumPy results to strict JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(name): json_safe(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item(), key)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "non applicable" if key == "score_fuite" else None
        # scipy may underflow a probability to zero. Keep internal arithmetic numeric;
        # disclose a bound at the JSON/report boundary rather than claim an exact zero.
        probability = key in {"p", "q", "p_value", "q_value"} or key.startswith(("p_", "q_")) or key.endswith("_p")
        return "< 0.001" if probability and value == 0 else value
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def load_frame(path, limit, *, keep_nonfinite=False):
    """Read a session-owned upload with the desktop parser and stable row IDs."""
    try:
        frame = read_table(str(path), max_table_bytes=limit)
    except MemoryError:
        raise
    except Exception as exc:
        raise InputError("Fichier tabulaire illisible ou limite du format dépassée.") from exc
    frame.columns = frame.columns.map(str)
    if frame.empty or not frame.columns.is_unique or any(not name for name in frame.columns):
        raise InputError("Le tableau doit contenir des lignes et des noms de colonnes distincts et non vides.")
    frame = frame.reset_index(drop=True)
    return frame if keep_nonfinite else frame.replace([np.inf, -np.inf], np.nan)


def upload_summary(path, limit):
    """Validate the upload and return its schema and first twenty rows."""
    frame = load_frame(path, limit)
    return json_safe({"shape": frame.shape, "columns": frame.columns, "dtypes": frame.dtypes.astype(str).to_dict(),
                      "preview": frame.head(20).to_dict("records")})


def validate_config(path, limit, target, task):
    """Reject missing, constant or incompatible supervised targets."""
    if target is None:
        return
    frame = load_frame(path, limit)
    if target not in frame or frame[target].nunique() < 2:
        raise InputError("Choisissez une cible existante avec au moins deux valeurs distinctes.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(frame[target]):
        raise InputError("La régression nécessite une cible numérique.")


def histogram(values):
    """Summarize finite values in bins suitable for interactive charts."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    if len(np.unique(values)) == 2:
        labels, counts = np.unique(values, return_counts=True)
        return [{"label": str(label), "count": int(count)} for label, count in zip(labels, counts)]
    counts, edges = np.histogram(values, bins=24 if values.min() != values.max() else 1)
    return [{"label": f"{low:.5g} à {high:.5g}", "count": int(count), "low": low, "high": high}
            for count, low, high in zip(counts, edges[:-1], edges[1:])]


def feature_stats(series):
    """Compute missingness, descriptive statistics and robust outlier diagnostics."""
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    values = series.dropna()
    binary = values.nunique() == 2
    top = [] if numeric and not binary else [{"label": str(k), "count": int(v)}
                              for k, v in values.astype(str).value_counts().head(15).items()]
    profile = {"column": series.name, "dtype": str(series.dtype), "type": "numérique" if numeric else
               "date" if pd.api.types.is_datetime64_any_dtype(series) else "catégoriel / texte",
               "missing_count": int(series.isna().sum()), "missing_pct": float(series.isna().mean() * 100),
               "unique": int(values.nunique()), "top": top, "binary": binary}
    if not numeric:
        lengths = values.astype(str).str.len()
        profile["text_length"] = {"mean": lengths.mean(), "min": lengths.min(), "max": lengths.max()}
    elif len(values):
        q1, median, q3 = values.quantile([.25, .5, .75])
        spread = q3 - q1
        zero_count_iqr = spread == 0 and q1 == 0 and values.mod(1).eq(0).all()
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        inside = values.between(lower, upper)
        mad = (values - median).abs().median()
        radius = 3.5 * mad / .67448975 if mad > 0 else None
        profile.update(mean=values.mean(), std=values.std(), min=values.min(), max=values.max(),
                       skew=values.skew(), kurtosis=values.kurt(), q1=q1, median=median, q3=q3,
                       boxplot={"low": values[inside].min(), "q1": q1, "median": median,
                                "q3": q3, "high": values[inside].max()},
                       iqr={"count": 0 if zero_count_iqr else int((~inside).sum()), "lower": lower, "upper": upper,
                            "note": "IQR nul sur comptage à zéros dominants : détection non applicable." if zero_count_iqr else ""},
                       robust_z={"count": int(((values - median).abs() > radius).sum()) if radius is not None else None,
                                 "lower": median - radius if radius is not None else None,
                                 "upper": median + radius if radius is not None else None})
    if binary:
        profile.pop("boxplot", None)
    return profile


def select_visualizations(series, target=None, task="auto", display_type="AUTO"):
    """Select EDA charts from observed values, with their data and explicit denominators.

    This is the sole selection policy for the feature EDA pane. It does not change
    preprocessing types or decisions. No feature or target name selects a rule.
    """
    if display_type not in {"AUTO", "FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT"}:
        raise InputError("Type d’affichage forcé inconnu.")
    if display_type == "FORCE_NUMERIC":
        try:
            series = pd.to_numeric(series, errors="raise")
        except (TypeError, ValueError) as exc:
            raise InputError("Le type numérique forcé nécessite des valeurs numériques convertibles.") from exc
    stats = feature_stats(series)
    observed = series.dropna()
    n, unique = len(observed), int(observed.nunique())
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    counts = observed.value_counts()
    counts = counts[counts > 0]  # Categorical dtype may retain unused, zero-count levels.
    dominant = float(counts.iloc[0] / n) if n else 0.
    examples = observed.drop_duplicates().head(5).astype(str).tolist()
    top_n, discrete_limit, category_limit = 10, 100, 15
    charts = []

    def bars(counts, denominator):
        return [{"label": str(label), "count": int(count), "percent": float(count * 100 / denominator) if denominator else 0.}
                for label, count in counts.items()]

    def other_label(labels):
        label, suffix = "Autres", 1
        while label in labels:
            label = f"Autres (regroupement {suffix})"
            suffix += 1
        return label

    def grouped_counts(values):
        frequency = values.value_counts()
        frequency = frequency[frequency > 0]
        keep = top_n if len(frequency) > category_limit else len(frequency)
        hidden = max(0, len(frequency) - keep)
        rows = bars(frequency.head(keep), len(values))
        if hidden:
            rows.append({"label": other_label([row["label"] for row in rows]), "count": int(frequency.iloc[keep:].sum()),
                         "percent": float(frequency.iloc[keep:].sum() * 100 / len(values)), "is_other": True})
        return rows, hidden

    def add(kind, reason, **values):
        charts.append({"kind": kind, "reason": reason, **values})

    def box(values):
        q1, median, q3 = values.quantile([.25, .5, .75])
        inside = values[values.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))]
        return {"low": float(inside.min()), "q1": float(q1), "median": float(median),
                "q3": float(q3), "high": float(inside.max())}

    # Reuse the engine's value-based date and prose recognition, without its DROP
    # decisions: a highly missing/near-constant column still has a physical type.
    dates = None
    text = None
    if n and not numeric and not pd.api.types.is_bool_dtype(series):
        if AutoPreprocessor()._is_date_column(series):
            dates = parse_dates(series)
        elif unique > 2:
            text = text_evidence(series, 42)
    integer = numeric and n > 0 and bool(observed.mod(1).eq(0).all())
    identifier = False
    if n >= 30 and unique / n > .9 and dates is None:
        if not numeric and (text is None or text["part_phrases"] < .6):
            identifier = float(observed.astype(str).str.len().mean()) < 30
        elif integer:
            identifier = _unique_integer_identifier(series)
    free_text = bool(text and text["part_phrases"] >= .6 and text["vocabulaire"] >= 8
                     and text["part_codes"] < .2 and unique >= 3
                     and (text["longueur_moyenne"] >= 24 or text["part_ponctuee"] >= .4))
    natural_order, order_reason = None, ""
    if not numeric and unique >= 2 and dates is None and not free_text:
        canonical = lambda value: AutoPreprocessor._canonical_variant(value).rstrip(".")
        months_fr = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
        months_short_fr = ["janv", "févr", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc"]
        days_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        ranks_month = {canonical(value): index for index in range(12)
                       for value in (calendar.month_name[index+1], calendar.month_abbr[index+1], months_fr[index], months_short_fr[index])}
        ranks_day = {canonical(value): index for index in range(7)
                     for value in (calendar.day_name[index], calendar.day_abbr[index], days_fr[index], days_fr[index][:3])}
        orders = [("mois reconnus par leurs valeurs : janvier → décembre", ranks_month),
                  ("jours reconnus par leurs valeurs : lundi → dimanche", ranks_day)]
        orders.extend(("ensemble ordinal reconnu par ses valeurs", {canonical(value): i for i, value in enumerate(pattern)})
                      for pattern in COMMON_ORDINAL_PATTERNS)
        for reason, ranks in orders:
            if all(canonical(value) in ranks for value in observed.unique()):
                natural_order = sorted(observed.unique(), key=lambda value: (ranks[canonical(value)], str(value)))
                order_reason = reason
                break
        if natural_order is None and isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
            natural_order = [value for value in series.cat.categories if value in counts.index]
            order_reason = "ordre déclaré dans le dtype catégoriel ordonné"

    # A possible code profile is a warning, not proof of nominal business semantics.
    # Mild skew, gaps and weak ordered target association avoid treating ordinary
    # counts/durations such as a zero-based, right-skewed measure as codes.
    code_evidence = {}
    possible_codes = False
    if integer and unique > discrete_limit:
        density = unique / (float(observed.max()) - float(observed.min()) + 1)
        ordered_target = None
        if target is not None and target.dropna().nunique() == 2:
            pairs = pd.DataFrame({"x": series, "y": target}).dropna()
            if len(pairs) >= 8 and pairs["y"].nunique() == 2:
                labels = sorted(pairs["y"].unique(), key=str)
                ordered_target = float(pairs["x"].corr(pairs["y"].map({labels[0]:0, labels[1]:1}), method="spearman"))
        skew = float(observed.skew())
        possible_codes = (not identifier and observed.min() > 0 and density < .8 and unique / n < .2
                         and np.isfinite(skew) and abs(skew) <= 2
                         and (ordered_target is None or (np.isfinite(ordered_target) and abs(ordered_target) <= .15)))
        code_evidence = {"integer_valued": True, "cardinality": unique, "unique_ratio": unique/n,
                         "repeated_ratio": 1-unique/n, "range_density": density, "skew": skew,
                         "ordered_target_spearman": ordered_target, "possible_codes": bool(possible_codes)}
    kind = ("EMPTY" if not n else "DATETIME" if dates is not None else
            "BOOLEAN" if pd.api.types.is_bool_dtype(series) else "ORDINAL" if natural_order is not None else "BINARY" if unique == 2 else
            "TEXT" if free_text else
            "ID_LIKE" if identifier else "NUMERIC_CODES" if possible_codes else
            "NUMERIC_DISCRETE" if integer and unique <= discrete_limit else
            "NUMERIC_CONTINUOUS" if numeric else
            "CATEGORICAL_HIGH_CARD" if unique > category_limit else "CATEGORICAL_LOW_CARD")
    if display_type == "FORCE_CATEGORICAL" and n:
        kind = "ORDINAL" if natural_order is not None else "CATEGORICAL_HIGH_CARD" if unique > category_limit else "CATEGORICAL_LOW_CARD"
    elif display_type == "FORCE_NUMERIC" and n:
        kind = "BINARY" if unique == 2 else "NUMERIC_DISCRETE" if integer and unique <= discrete_limit else "NUMERIC_CONTINUOUS"
        natural_order = None
    elif display_type == "FORCE_TEXT" and n:
        kind = "TEXT"
    elif display_type == "FORCE_DATE" and n:
        dates = parse_dates(series)
        if not dates.notna().any():
            raise InputError("Aucune date valide pour le type date forcé.")
        kind = "DATETIME"
    denominator = f"% des {n:,} valeurs observées ; manquants exclus."
    if stats["missing_pct"] >= 90:
        add("missingness", f"{stats['missing_pct']:.2f}% de manquants ≥ seuil 90% ; état des {len(series):,} lignes avant la distribution.",
            data=bars(pd.Series({"Manquant": stats["missing_count"], "Présent": n}), len(series)),
            denominator=len(series), horizontal=False)

    if not n:
        add("summary", "Aucune valeur observée : aucune distribution pertinente.", cardinality=unique, examples=examples)
    elif kind in {"TEXT", "ID_LIKE"}:
        reason = ("Texte libre reconnu par longueur, vocabulaire et séquences de mots. " if kind == "TEXT" else
                  f"Identifiant (valeurs uniques) : aucune distribution pertinente. {unique / n:.1%} de valeurs uniques, codes courts ou entiers sans répétition. ")
        if kind == "TEXT" and display_type == "FORCE_TEXT":
            reason = "Type texte forcé par le choix utilisateur existant. "
        elif kind == "TEXT":
            reason += ("Reconnaissance sur 2 000 valeurs, tirage aléatoire sans remise, graine 42, pour limiter le coût du typage ; cardinalité complète. "
                       if n > 2000 else f"Reconnaissance sur les {n} valeurs observées, sans échantillonnage. ")
        add("summary", reason + "Aucune distribution pertinente ; cardinalité et exemples uniquement.", cardinality=unique, examples=examples)
    elif kind == "ORDINAL":
        add("ordered_bar", f"{order_reason} ; toutes les {unique} modalités observées, ordre naturel, aucun regroupement. {denominator}",
            data=bars(counts.reindex(natural_order), n), denominator=n, hidden_categories=0,
            horizontal=unique > 8 or any(len(str(value)) > 18 for value in natural_order))
    elif kind in {"BINARY", "BOOLEAN"} and unique == 2:
        add("binary_bar", f"2 modalités observées : effectifs et pourcentages, sans histogramme ni boîte. {denominator}",
            data=bars(counts, n), denominator=n, horizontal=False)
    elif dominant >= .99 and not (kind == "CATEGORICAL_LOW_CARD" and unique <= category_limit):
        share = pd.Series({str(counts.index[0]): int(counts.iloc[0])})
        rows = bars(share, n)
        rows.append({"label": other_label([row["label"] for row in rows]), "count": n - int(counts.iloc[0]), "percent": (1 - dominant) * 100, "is_other": True})
        add("dominant_share", f"Valeur dominante « {counts.index[0]} » : {dominant:.2%} ≥ seuil 99%. {denominator}",
            data=rows, denominator=n, horizontal=any(len(row["label"]) > 18 for row in rows))
    elif kind == "DATETIME":
        valid = dates.dropna()
        monthly = valid.dt.to_period("M").value_counts().sort_index()
        monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"), fill_value=0)
        type_note = ("Type datetime natif, sans échantillonnage du typage." if pd.api.types.is_datetime64_any_dtype(series) else
                     "Reconnaissance du type sur 100 valeurs non manquantes, tirage aléatoire sans remise, graine 42, pour limiter le coût du typage."
                     if n > 100 else f"Reconnaissance du type sur les {n} valeurs observées, sans échantillonnage.")
        add("time_series", f"Dates reconnues : {len(valid):,}/{n:,} valeurs observées ; effectifs par mois, ordre chronologique. "
            f"{n-len(valid)} dates invalides exclues. {type_note} Agrégation sur toutes les dates valides.",
            data=bars(monthly, len(valid)), denominator=len(valid), invalid_count=n-len(valid), horizontal=False)
    elif kind == "NUMERIC_CODES":
        rows, hidden = grouped_counts(observed)
        rho = code_evidence["ordered_target_spearman"]
        relation = f"|ρ cible|={abs(rho):.3f}" if rho is not None and np.isfinite(rho) else "association ordonnée à la cible non disponible"
        add("top_categories", "Codes numériques possibles — codes numériques: distribution non interprétable comme mesure. "
            f"Entiers, cardinalité {unique} > {discrete_limit}, densité de plage {code_evidence['range_density']:.1%}, "
            f"ratio uniques/observés {unique/n:.1%}, asymétrie {code_evidence['skew']:.3g} ; {relation}. "
            f"Top {top_n} + Autres ({hidden} codes regroupés) ; les répétitions observées sont {(1-unique/n):.2%}. "
            "Heuristique à confirmer : la distance entre codes n’est pas interprétée. " + denominator,
            data=rows, denominator=n, hidden_categories=hidden, horizontal=False)
    elif kind == "NUMERIC_DISCRETE":
        add("value_counts", f"Valeurs entières, {unique} valeurs distinctes ≤ seuil {discrete_limit} : effectifs par valeur. {denominator}",
            data=bars(counts.sort_index(), n), denominator=n, horizontal=unique > 20)
    elif kind == "NUMERIC_CONTINUOUS":
        values = observed.to_numpy(dtype=float)
        skew = float(observed.skew()) if n > 2 else 0.
        scale = "symlog" if np.isfinite(skew) and abs(skew) >= 2 else "linear"
        axis = np.sign(values) * np.log1p(np.abs(values)) if scale == "symlog" else values
        edges = np.histogram_bin_edges(axis, bins=24)
        if scale == "symlog":
            edges = np.sign(edges) * np.expm1(np.abs(edges))
        edges[0], edges[-1] = values.min(), values.max()
        frequency, edges = np.histogram(values, bins=edges)
        bins = [{"label": f"{low:.5g} à {high:.5g}", "low": float(low), "high": float(high),
                 "count": int(count), "percent": float(count * 100 / n)}
                for low, high, count in zip(edges[:-1], edges[1:], frequency)]
        axis_note = ("Classes égales sur l’axe sign(x) × log(1+|x|) ; bornes en unités originales. "
                     "Aucune observation exclue." if scale == "symlog" else "Axe linéaire complet, aucune observation exclue.")
        add("histogram", f"Variable numérique, cardinalité {unique} ; asymétrie {skew:.3g}. {axis_note} {denominator}",
            data=bins, scale=scale, denominator=n, horizontal=False)
        add("boxplot", "Quartiles sur toutes les valeurs observées, en unités originales. Moustaches aux observations dans Q1/Q3 ± 1,5 IQR ; extrêmes non dessinés, non supprimés.",
            box=box(observed), sample_size=n, outlier_count=stats["iqr"]["count"])
    else:
        rows, hidden = grouped_counts(observed) if unique > category_limit else (bars(counts, n), 0)
        add("top_categories" if unique > category_limit else "frequency_bar",
            (f"Valeur dominante {dominant:.2%} ; toutes les modalités sont conservées. " if dominant >= .99 else "") +
            (f"Cardinalité {unique} > seuil {category_limit} : top {top_n} + Autres ; {hidden} modalités regroupées. "
             if unique > category_limit else f"{unique} modalités ≤ seuil {category_limit} : fréquences triées par effectif. ") + denominator,
            data=rows, hidden_categories=hidden, denominator=n, horizontal=any(len(row["label"]) > 18 for row in rows))

    target_type = "NONE"
    relation_note = "Aucune cible sélectionnée."
    if target is not None:
        classes = sorted(target.dropna().unique(), key=lambda value: str(value))
        regression = task == "regression" or (task == "auto" and pd.api.types.is_numeric_dtype(target) and len(classes) > 15)
        target_type = "CONTINUOUS" if regression else "BINARY" if len(classes) == 2 else "MULTICLASS"
        relation_note = ("La feature est la cible : pas de relation avec elle-même." if series.name == target.name else
                         "Pas de graphique par classe binaire : la cible sélectionnée n’est pas binaire." if target_type != "BINARY" else
                         "Identifiant : aucune relation graphique pertinente avec la cible." if kind == "ID_LIKE" else
                         "Aucune relation graphique pertinente pour une colonne vide, textuelle, datée ou quasi constante.")
        if target_type == "BINARY" and series.name != target.name and dominant < .99 and kind not in {"EMPTY", "TEXT", "ID_LIKE", "DATETIME"}:
            pairs = pd.DataFrame({"value": series, "target": target}).dropna()
            if pairs.empty:
                relation_note = "Aucune paire complète entre cette feature et la cible ; pas de graphique de relation."
            if kind == "NUMERIC_DISCRETE":
                labels = sorted(pairs["value"].unique())
                groups = []
                for label in classes:
                    values = pairs.loc[pairs["target"] == label, "value"]
                    if len(values):
                        frequency = values.value_counts().reindex(labels, fill_value=0)
                        groups.append({"label": str(label), "data": bars(frequency, len(values)), "sample_size": len(values)})
                if len(groups) == 2:
                    add("target_distribution", f"Valeurs discrètes par classe de « {target.name} » ; pourcentages normalisés dans chaque classe, mêmes valeurs sur les deux axes. "
                        f"{len(pairs):,} paires complètes, aucun échantillonnage ; une boîte pourrait avoir des quartiles confondus.", groups=groups,
                        target=str(target.name), horizontal=len(labels) > 20)
                    relation_note = ""
            elif numeric and kind not in {"BINARY", "BOOLEAN", "NUMERIC_CODES", "CATEGORICAL_HIGH_CARD", "CATEGORICAL_LOW_CARD", "ORDINAL"}:
                groups = []
                for label in classes:
                    values = pairs.loc[pairs["target"] == label, "value"]
                    if len(values):
                        groups.append({"label": str(label), "box": box(values), "sample_size": len(values)})
                if len(groups) == 2:
                    add("target_boxplot", f"Feature numérique par classe de « {target.name} » ; {len(pairs):,} paires complètes, sans échantillonnage. "
                        "Axe commun aux classes ; quartiles et moustaches IQR en unités originales ; extrêmes non dessinés.",
                        groups=groups, target=str(target.name),
                        domain={"low": min(group["box"]["low"] for group in groups), "high": max(group["box"]["high"] for group in groups)})
                    relation_note = ""
            else:
                positive = classes[-1]
                frequency = pairs["value"].value_counts()
                frequency = frequency[frequency > 0]
                selected = ([value for value in natural_order if value in frequency.index] if natural_order is not None else
                            list(frequency.head(top_n).index) if len(frequency) > category_limit else list(frequency.index))
                selected = [value for value in selected if frequency[value] >= 5]
                rows = []
                hidden = max(0, len(frequency) - len(selected))
                for label in selected + ([None] if hidden else []):
                    mask = ~pairs["value"].isin(selected) if label is None else pairs["value"].eq(label)
                    subset = pairs.loc[mask, "target"]
                    count = int(subset.eq(positive).sum())
                    rows.append({"label": other_label([str(v) for v in selected]) if label is None else str(label), "count": count,
                                 "total": len(subset), "percent": count * 100 / len(subset), "is_other": label is None})
                if rows:
                    add("target_rate", f"Taux de la classe « {positive} » de « {target.name} » par modalité ; dénominateur : paires complètes de chaque catégorie. "
                        f"{hidden} modalités regroupées dans Autres ; taux pondéré par leurs effectifs. {len(series)-len(pairs)} paires incomplètes exclues.",
                        data=rows, hidden_categories=hidden, target=str(target.name), positive_class=str(positive),
                        horizontal=any(len(row["label"]) > 18 for row in rows) or (natural_order is not None and len(rows) > 8))
                    relation_note = ""
    if display_type != "AUTO":
        for chart in charts:
            chart["reason"] = f"Type d’affichage forcé via le choix existant {display_type}. " + chart["reason"]
    return {"detected_type": kind, "display_type_override": display_type, "code_evidence": code_evidence,
            "cardinality": unique, "observed_count": n, "missing_pct": stats["missing_pct"],
            "dominant_share": dominant, "skew": stats.get("skew"), "target_type": target_type,
            "recommended_visualization": charts, "relationship_note": relation_note}


def feature_eda(path, limit, column, target=None, task="auto", display_type="AUTO"):
    """Compute the chart series for a single input column."""
    frame = load_frame(path, limit)
    if column not in frame:
        raise InputError("Colonne inconnue.")
    series = frame[column]
    profile = feature_stats(series)
    profile["distribution"] = histogram(series.dropna()) if profile["type"] == "numérique" else profile["top"]
    profile.update(select_visualizations(series, frame[target] if target is not None and target in frame else None, task, display_type))
    if profile["detected_type"] == "ID_LIKE":
        for key in ("distribution", "boxplot", "iqr", "robust_z"):
            profile.pop(key, None)
    return json_safe(profile)


def cramers_v(left, right):
    """Measure categorical association from a bounded contingency table."""
    table = pd.crosstab(left, right)
    if min(table.shape) < 2:
        return None
    chi2 = chi2_contingency(table, correction=False).statistic
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def dataset_eda(frame, target, task):
    """Profile all columns and bound the matrix widths and association sample."""
    profiles = [feature_stats(series) for _, series in frame.items()]
    identifier_columns = set()
    for row in profiles:
        name = row["column"]
        observed = len(frame) - row["missing_count"]
        numeric_id = _unique_integer_identifier(frame[name], unique_count=row["unique"],
                                                 observed_count=observed)
        text_candidate = (row["type"] != "numérique" and observed > 30
                          and row["unique"] / observed > .9)
        if numeric_id or (text_candidate and select_visualizations(
                frame[name], frame[target] if target is not None else None, task)["detected_type"] == "ID_LIKE"):
            row["detected_type"] = "ID_LIKE"
            identifier_columns.add(name)
    numeric = [row["column"] for row in profiles if row["type"] == "numérique" and row["column"] not in identifier_columns]
    categorical = [row["column"] for row in profiles if row["type"] != "numérique" and 2 <= row["unique"] <= 40]
    pearson = frame[numeric[:MATRIX_LIMIT]].corr()
    sample = frame.sample(RELATION_LIMIT, random_state=42) if len(frame) > RELATION_LIMIT else frame
    cat_names = categorical[:CATEGORY_LIMIT]
    cat_matrix = [[None] * len(cat_names) for _ in cat_names]
    for i, left in enumerate(cat_names):
        for j in range(i, len(cat_names)):
            cat_matrix[i][j] = cat_matrix[j][i] = cramers_v(sample[left], sample[cat_names[j]])
    relations = []
    if target is not None:
        regression = task == "regression" or (task == "auto" and target in numeric and frame[target].nunique() > 15)
        for name in frame.columns:
            if name == target or name in identifier_columns:
                continue
            evidence = association(sample[name], sample[target], name in numeric, regression, kendall=True)
            primary = next(iter(evidence["tests"]), None)
            method = {"spearman": "Spearman", "kruskal": "Kruskal · ε²", "khi_deux": "V de Cramér",
                      "fisher": "V de Cramér · Fisher"}.get(primary)
            relations.append({"column": name, "effect": evidence.get("effet_primaire"), "method": method,
                              "sample_size": evidence["effectif"], "note": evidence.get("limite"),
                              "p_value": evidence.get("p_primaire"), "tests": evidence["tests"]})
    corrected = adjusted_p_values({index: row["p_value"] for index, row in enumerate(relations)})
    for index, relation in enumerate(relations):
        relation["q_value"] = corrected.get(index)
    duplicates = frame.duplicated()
    imbalance = (class_profile(frame[target].dropna()) if target is not None and
                 _looks_like_classification(frame[target], task) else None)
    return {"shape": frame.shape, "features": profiles, "imbalance": imbalance,
            "missing_pct": sum(row["missing_count"] for row in profiles) * 100 / frame.size,
            "duplicate_count": int(duplicates.sum()), "duplicate_examples": frame[duplicates].head(5).to_dict("records"),
            "pearson": {"columns": pearson.columns, "values": pearson.to_numpy(), "limit": MATRIX_LIMIT,
                        "eligible": len(numeric)},
            "cramers_v": {"columns": cat_names, "values": cat_matrix, "limit": CATEGORY_LIMIT,
                          "eligible": len(categorical), "max_categories": 40},
            "relation_sample_size": len(sample),
            "sample_note": (f"V de Cramér et relations avec la cible : {len(sample):,} lignes, "
                            "tirage aléatoire sans remise, graine 42, pour limiter le coût des tests. "
                            "Statistiques descriptives et Pearson : toutes les lignes."
                            if len(sample) < len(frame) else "EDA : toutes les lignes, aucun échantillonnage."),
            "target_relations": relations}


def eda(path, limit, target, task):
    """Return the complete EDA payload as strict JSON."""
    return json_safe(dataset_eda(load_frame(path, limit), target, task))


def configured_config(frame, target, options):
    """Validate column settings and resolve the engine's normalized names."""
    config = build_config(options)
    names = AutoPreprocessor(config)._normalize_column_names(frame.drop(columns=[target] if target else []).iloc[:0]).columns
    mapping = dict(zip(frame.drop(columns=[target] if target else []).columns, names))
    config.column_overrides = {mapping.get(key, key): value for key, value in config.column_overrides.items()}
    for field in ("value_bounds", "ordinal_orders"):
        values = getattr(config, field) or {}
        if set(values) - set(frame.columns):
            raise InputError("Un réglage fait référence à une colonne inconnue.")
        setattr(config, field, {mapping.get(key, key): value for key, value in values.items()})
    if config.keep_columns:
        if set(config.keep_columns) - set(frame.columns):
            raise InputError("Une variable protégée est inconnue.")
        config.keep_columns = [mapping.get(name, name) for name in config.keep_columns]
    return config


def analyze_frame(frame, target, options):
    """Use the existing recommendation engine with the selected task and options."""
    return AutoPreprocessor(configured_config(frame, target, options), task=options.task).analyze(frame, target=target)


def leakage_review(row):
    """Association is a reason to ask about availability, never proof of temporal leakage."""
    values = [row.get(key) for key in ("score_fuite", "cramer_cible", "information_mutuelle_cible")]
    if (row.get("action") != "DROP" and not row.get("fuite_suspectee") and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            and values[0] >= .60 and values[1] >= .20 and values[2] >= .02):
        return (f"Forte association avec la cible — vérifier la disponibilité au moment de la prédiction · CV {values[0]:.3f}, V de Cramér corrigé {values[1]:.3f}, "
                f"information mutuelle {values[2]:.3f}. Seuils de revue : CV ≥ 0.60, V ≥ 0.20, MI ≥ 0.02. "
                "Aucune suppression automatique par cet avertissement.")
    return None


def recommendation(row, target):
    """Expose statistical evidence and applicable desktop actions to the browser."""
    if row.get("fuite_suspectee"):
        row = {**row, "action": "DROP", "raison": (
            f"ALERTE FUITE DE CIBLE : score de validation croisée {row['score_fuite']:.3f}. "
            "Suppression recommandée ; un choix explicite est nécessaire pour conserver cette variable."),
            "steps": {**row.get("steps", {}), "type": {**row.get("steps", {}).get("type", {}),
                "action": "DROP", "measure": f"score de fuite CV : {row['score_fuite']:.3f}",
                "threshold": "Fuite signalée par la validation croisée du moteur",
                "evidence": f"score de fuite CV : {row['score_fuite']:.3f} ; fuite détectée par le moteur",
                "reason": "Suppression automatique : fuite de cible détectée par validation croisée.",
                "explanation": "Fuite de cible détectée : supprimer cette variable avant l’entraînement."}}}
    anomalies = []
    review = leakage_review(row)
    if review:
        anomalies.append(review)
    if row.get("fuite_suspectee"):
        anomalies.append(f"ALERTE FUITE DE CIBLE · score CV {row['score_fuite']:.3f} · suppression recommandée")
    anomalies.extend("Comparaison d’alternative : " + warning for warning in row.get("avertissements_numeriques", []))
    for key, label in (("manquants_pct", "Manquants (%)"), ("outliers_pct", "Extrêmes avant nettoyage (%)"),
                       ("faux_manquants", "Faux manquants"), ("variantes_casse_espaces", "Variantes"),
                       ("date_formats", "Formats de date")):
        if row.get(key, 0) > 0:
            value = (f"{int(row[key]):,}".replace(",", " ") if key in {"faux_manquants", "variantes_casse_espaces", "date_formats"}
                     else f"{row[key]:.1f}")
            anomalies.append(f"{label} · {value}")
    if row.get("type", "").startswith("DROP"):
        anomalies.append(row["type"])
    if row.get("alerte_identifiant"):
        anomalies.append("Identifiant potentiel")
    if row.get("sparse_observation_pct", 0) > 50:
        anomalies.append(f"Risque de surapprentissage : {row['sparse_observation_pct']:.1f}% des observations dans des modalités d'effectif < 5 ; regroupement des rares, lissage de poids 10 en encodage supervisé.")
    if row.get("paires_correlees"):
        anomalies.append("Colinéarité")
    if abs(row.get("skew", 0)) > 1:
        anomalies.append(f"Asymétrie avant nettoyage · {row['skew']:.3g}")
    if "fuite" in row.get("raison", "").casefold():
        anomalies.append(row["raison"])
    unavailable = {"MANUAL_MAP", "SELECTED_TO_NAN", "SET_BOUNDS"}
    if row["type"] != "ORDINAL":
        unavailable.add("ORDINAL")
    if target is None:
        unavailable.update({"TARGET_ENCODE", "LOO_ENCODE"})
    alternatives = [{"value": action, "label": label_action(action)} for action in row["alternatives"]
                    if action != row["action"] and action not in unavailable]
    steps = {}
    for step, detail in row.get("steps", {}).items():
        steps[step] = {k: detail.get(k) for k in ("action", "measure", "threshold", "explanation", "confidence")}
        steps[step]["help"] = recommendation_help(step, detail.get("action", "AUTO"),
                                                   detail.get("measure", ""), detail.get("threshold", ""))
        steps[step]["label"] = label_step(step)
        steps[step]["choices"] = [{"value": action, "label": label_action(action),
                                    "parameters": PARAMETER_HINTS.get(action) if step_parameter_visible(step, action) else None,
                                    "help": recommendation_help(step, action, detail.get("measure", ""), detail.get("threshold", ""))}
                                   for action in detail["choices"]
                                   if not (target is None and action in {"TARGET_ENCODE", "LOO_ENCODE"})
                                   and detail.get("comparaison", {}).get(action, {}).get("applicable") is not False]
        steps[step]["comparison"] = detail.get("comparaison", {})
        for action in detail["choices"]:
            if action == "AUTO" or action in unavailable or step in {"target", "nettoyage"}:
                continue
            if detail.get("comparaison", {}).get(action, {}).get("applicable") is False:
                continue
            alternatives.append({"value": f"{step}:{action}", "label": f"{label_step(step)} · {label_action(action)}"})
    return {"column": row["colonne"], "type": row["type"], "is_target": row.get("est_cible", False),
            "anomalies": anomalies, "action": row["action"], "label": label_action(row["action"]),
            "justification": analysis_recommendation_summary(row), "alternatives": alternatives, "steps": steps,
            "diagnostics": {key: value for key, value in row.items() if key != "steps"}}


def recommendation_train_frame(frame, target, task, options):
    """Use the same deterministic training rows as execution for learned recommendations."""
    values = WebOptions.model_validate(options or {})
    data = frame.dropna(subset=[target]) if target is not None else frame
    if len(data) < (10 if target is not None else 2):
        return data
    stratify = data[target] if target is not None and _looks_like_classification(data[target], task) else None
    split = {"test_size": values.test_size_pct / 100, "random_state": values.random_state}
    try:
        train, _ = train_test_split(data, stratify=stratify, **split)
    except ValueError:
        train, _ = train_test_split(data, **split)
    return train


def cached_analysis(frame, path, target, task, options, choices, artifact=None):
    """Cache recommendations learned from the execution training split only."""
    options = WebOptions.model_validate(options or {}).model_dump(
        exclude={"imbalance_method", "imbalance_ratio", "imbalance_neighbors"})
    forced = {name: choice for name, choice in (choices or {}).items()
              if (isinstance(choice, str) and choice.startswith("type:FORCE_"))
              or (isinstance(choice, dict) and choice.get("type", {}).get("action", "").startswith("FORCE_"))}
    # Non-type choices do not affect analyze_choices; validate them separately on every request.
    forced = {name: choice if isinstance(choice, str) else {"type": choice["type"]}
              for name, choice in forced.items()}
    stat = Path(path).stat()
    signature = hashlib.sha256(json.dumps(
        [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, target, task, options, forced],
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if artifact is not None and Path(artifact).is_file():
        stored = joblib.load(artifact)  # Trusted, server-created session artifact, never a user-supplied path.
        if stored["signature"] == signature:
            return stored["analysis"], True
    result = analyze_choices(recommendation_train_frame(frame, target, task, options),
                             target, processing_options(task, options), choices or {})
    if artifact is not None:
        pending = Path(artifact).with_suffix(".tmp")
        joblib.dump({"signature": signature, "analysis": result}, pending)
        pending.replace(artifact)
    return result, False


def analyze(path, limit, target, task, options=None, choices=None, artifact=None):
    """Run analysis and return one recommendation per input column."""
    frame = load_frame(path, limit)
    result, _ = cached_analysis(frame, path, target, task, options, choices, artifact)
    if choices:
        selected_overrides(result, target, choices)
    note = result.sample_note or "Profil descriptif du train : toutes ses lignes, aucun échantillonnage."
    note += " Recommandations calculées sur le train du découpage configuré ; le test n'entre pas dans ces décisions."
    note += " CV de fuite : au plus 20 000 lignes (5 000 au-delà de 100 000 lignes), tirage sans remise, classe rare préservée ; mécanismes de manquants : au plus 5 000 lignes."
    imbalance = class_profile(frame[target].dropna()) if target is not None and _looks_like_classification(frame[target], task) else None
    return json_safe({"features": [recommendation(row, target) for row in result.columns], "sample_note": note,
                      "imbalance": imbalance})


def selection_preview(path, limit, target, task, options=None, choices=None):
    """Estimate selected output names by fitting on the proposed training split only."""
    frame = load_frame(path, limit)
    choices = choices or {}
    opts = processing_options(task, options)
    analysis = analyze_choices(recommendation_train_frame(frame, target, task, options), target, opts, choices)
    overrides, _ = selected_overrides(analysis, target, choices)
    passthrough = [name for name in frame if choices.get(name) == "none"]
    active = frame.drop(columns=passthrough)
    names = active.drop(columns=[target]) if target else active
    normalized = AutoPreprocessor(build_config(opts))._normalize_column_names(names.iloc[:0]).columns
    name_map = dict(zip(names.columns, normalized))
    opts.column_overrides = {name_map.get(name, name): value for name, value in overrides.items()}
    for key in ("value_bounds", "ordinal_orders"):
        opts.engine_options[key] = {name_map.get(name, name): value
                                    for name, value in (opts.engine_options.get(key) or {}).items()}
    if opts.engine_options.get("keep_columns"):
        opts.engine_options["keep_columns"] = [name_map.get(name, name) for name in opts.engine_options["keep_columns"]]
    for key in ("selection_include", "selection_exclude"):
        if opts.engine_options.get(key):
            if set(opts.engine_options[key]) - set(names.columns):
                raise InputError("La sélection manuelle contient une colonne source inconnue, brute ou la cible.")
            opts.engine_options[key] = [name_map[name] for name in opts.engine_options[key]]
    result = run_preprocessing(active, target, opts, analysis_result=analysis)
    prep = result.prep
    exported, _, renamed = export_frame(frame, result.X_train, passthrough,
                                  result.y_train, target, result.y_train_original)
    selected = [name for name in exported.columns if name not in
                ({target, f"{target}_original"} if target else set())]
    original = [name for name in frame.columns if name != target]
    details = getattr(prep, "feature_selection_report_", None)
    encoded_names = list(dict.fromkeys((details["selected"] + details["rejected"]) if details else
                                   list(result.X_train.columns)))
    origins = {renamed.get(output, output): source for source, normalized_name in name_map.items()
               for output in encoded_names
               if output == normalized_name or output.startswith(normalized_name + "_")}
    origins.update({name: name for name in passthrough})
    warnings = list(getattr(prep, "warnings_", []))
    if opts.engine_options.get("selection_method") in {"rfe", "rfecv", "forward", "backward"}:
        if details and details.get("cv_preprocessing") == "fold_isolated":
            warnings.append("Méthode wrapper coûteuse ; chaque pli réajuste son prétraitement. Le score interne de recherche ne remplace pas une évaluation externe indépendante.")
        elif opts.engine_options.get("selection_method") in {"rfecv", "forward", "backward"}:
            warnings.append("Méthode wrapper coûteuse ; prétraitement non isolé dans les plis. Le score interne peut être optimiste.")
    removed_before = list(dict.fromkeys(prep.dropped_variance_cols_ + prep.dropped_collinear_cols_))
    rejected = list(dict.fromkeys(removed_before + (details["rejected"] if details else [])))
    return json_safe({"status": "estimated_train_fit", "original": original, "selected": selected,
                      "rejected": rejected,
                      "encoded_count": len(result.X_train.columns) + len(rejected),
                      "selected_count": len(selected), "output_count": len(exported.columns),
                      "target_included": bool(target), "expected_rows": len(exported),
                      "method": opts.engine_options.get("selection_method", "auto"),
                       "warnings": warnings, "details": details, "feature_sources": origins,
                      "note": "Aperçu estimé sur le train du découpage demandé ; l'exécution finale réajustera le pipeline."})


def analyze_choices(frame, target, options, choices):
    """Refresh applicable operations after an explicit type override."""
    for column, choice in choices.items():
        forced = choice.get("type", {}).get("action") if isinstance(choice, dict) else (
            choice.split(":", 1)[1] if choice.startswith("type:") else None)
        if forced in {"FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT"}:
            if column not in frame or column == target:
                raise InputError("Le changement de type nécessite une variable explicative existante.")
            options.column_overrides[column] = ColumnOverride({"type": forced})
    return analyze_frame(frame, target, options)


def selected_overrides(result, target, choices):
    """Validate public choice tokens and convert alternatives to engine overrides."""
    rows = {row["colonne"]: row for row in result.columns}
    selected = {}
    for column, choice in choices.items():
        if column not in rows or column == target:
            raise InputError("Le choix doit porter sur une colonne explicative existante.")
        if isinstance(choice, dict):
            selected[column] = {}
            for step, value in choice.items():
                action = value["action"]
                validate_step_choice(result, column, step, action)
                if target is None and action in {"TARGET_ENCODE", "LOO_ENCODE"}:
                    raise InputError("Cet encodage nécessite une cible.")
                raw = value.get("parameters", "").strip()
                required = action in {"CONSTANT", "MANUAL_MAP", "SELECTED_TO_NAN"}
                if raw or required:
                    if not step_parameter_visible(step, action):
                        raise InputError("Cette action ne prend pas de paramètre textuel.")
                    params = parse_step_parameters(step, action, raw, rows[column]["type"])
                    selected[column][step] = (action, params) if params is not None else action
                else:
                    selected[column][step] = action
            continue
        if choice in {"recommended", "none"}:
            continue
        allowed = {item["value"] for item in recommendation(rows[column], target)["alternatives"]}
        allowed.add("DROP")
        if choice not in allowed:
            raise InputError("Transformation incompatible avec la colonne sélectionnée.")
        step, separator, action = choice.partition(":")
        selected[column] = {step: action} if separator else choice
    neighbour_counts = {value[1]["n_neighbors"] for choice in selected.values() if isinstance(choice, dict)
                        for value in [choice.get("missing")] if isinstance(value, tuple) and value[0] == "KNN"
                        and isinstance(value[1], dict) and "n_neighbors" in value[1]}
    if len(neighbour_counts) > 1:
        raise InputError("Le nombre de voisins doit être identique pour les variables partageant l’imputation KNN.")
    for column, row in rows.items():
        choice = choices.get(column, "recommended")
        if (row.get("fuite_suspectee") or row.get("action") == "DROP") and (choice == "recommended" or choice == {}):
            selected[column] = "DROP"
    return choices_to_overrides(result, selected), selected


def preview(path, limit, target, task, column, choice, options=None, artifact=None):
    """Render chart data and the desktop comparison table for a proposed choice."""
    frame = load_frame(path, limit)
    if column not in frame or column == target:
        raise InputError("Choisissez une colonne explicative existante.")
    series = frame[column]
    profile = feature_stats(series)
    chart = histogram(series.dropna()) if profile["type"] == "numérique" else profile["top"]
    if choice == "none":
        return json_safe({"status": "ok", "before_chart": chart, "after_charts": [{"title": column, "data": chart}],
                          "notes": ["Les valeurs brutes sont conservées sur les lignes exportées."], "table": [],
                          "rows": len(frame), "output_columns": [column], "applied": {"traitement": "NONE"},
                          "before": profile, "after_scaled": profile,
                          "samples": [{"row": index, "original": value, "values": {column: value}}
                                      for index, value in series.head(12).items()]})
    result, _ = cached_analysis(frame, path, target, task, options, {column: choice}, artifact)
    options = processing_options(task, options)
    _, selected = selected_overrides(result, target, {column: choice})
    config = build_config(options)
    config.clean_column_names = False
    comparison = preview_for_choice(frame, result, column, selected, config, target, task)
    data = comparison["data"]
    if "after_scaled" in data:
        before = histogram(data["before"])
        after = [{"title": column, "data": histogram(data["after_scaled"])}]
    else:
        before = chart
        after = [{"title": name, "data": histogram(values)} for name, values in data.get("after", {}).items()]
    return json_safe({"status": comparison["status"], "before_chart": before, "after_charts": after,
                      "notes": comparison["notes"], "table": build_preview_table(comparison).to_dict("records"),
                      **{key: comparison.get(key) for key in (
                          "rows", "output_columns", "applied", "before", "after", "after_scaled", "affected", "samples")}})


def fitted_values(estimator, column, names):
    """Extract the selected column's learned numeric estimator attributes."""
    if estimator is None:
        return {}
    learned = {}
    index = list(names).index(column) if column in names else 0
    for key in ("mean_", "var_", "scale_", "center_", "data_min_", "data_max_", "lambdas_", "n_quantiles_"):
        value = getattr(estimator, key, None)
        if isinstance(value, np.ndarray):
            value = value[index] if value.ndim == 1 and len(value) > index else None
        if value is not None:
            learned[key] = value
    return {"method": type(estimator).__name__, **learned,
            **({"fallback": estimator.fallback_, "warning": estimator.warning_} if getattr(estimator, "fallback_", None) else {})}


def trained_parameters(prep, column):
    """Summarize fitted per-column values without serializing model objects."""
    parameters = {"cleaning": prep.cleaning_decisions_.get(column)}
    if column in prep.num_cols_:
        winsorizer = prep.winsorizer_
        scaler = prep.column_scalers_.get(column, prep.num_scaler_)
        scaler_names = getattr(scaler, "feature_names_in_", [])
        parameters.update(imputation_method=prep.column_missing_.get(column),
                          fill_value=(prep.num_imputer_ or {}).get(column),
                          outlier_method=prep.column_outliers_.get(column),
                          lower=winsorizer.lower_bounds_.get(column), upper=winsorizer.upper_bounds_.get(column),
                          distribution=prep.column_distributions_.get(column),
                          distribution_fit=fitted_values((prep.num_power_transformer_ or {}).get(column), column, []),
                          scaling=fitted_values(scaler, column, scaler_names))
        capped = getattr(winsorizer, "capped_counts_", {}).get(column, 0) if winsorizer is not None else 0
        if capped:
            parameters["winsorization_note"] = (f"Bornes train [{parameters['lower']:.4g}, {parameters['upper']:.4g}] : "
                                                f"{capped} valeur(s) bornée(s) avant la mise à l’échelle.")
    if column in prep.categorical_imputers_:
        parameters["fill_value"] = prep.categorical_imputers_[column]
        if prep.column_missing_.get(column) == "MODE":
            parameters["imputation_method"] = "MODE"
            parameters["imputation_note"] = "Manquants du train < 5 : mode, sans modalité ni indicateur de manque."
    if column in prep.freq_encoding_maps_:
        mapping = prep.freq_encoding_maps_[column]
        parameters["frequency_encoding"] = {"cardinality": len(mapping), "first_20": dict(list(mapping.items())[:20])}
    if column in prep.high_card_cols_ and prep.target_cat_encoder_ is not None:
        encoder = prep.target_cat_encoder_
        parameters["target_encoding"] = {key: getattr(encoder, key) for key in ("target_mean_", "smooth", "cv")
                                         if hasattr(encoder, key)}
    if column in prep.low_card_cols_ and prep.ohe_encoder_ is not None:
        categories = prep.ohe_encoder_.categories_[prep.low_card_cols_.index(column)]
        parameters["onehot"] = {"cardinality": len(categories), "first_20": categories[:20]}
    if column in prep.ordinal_encoders_:
        parameters["ordinal"] = prep.ordinal_encoders_[column].categories_
    if column in prep.custom_encoders_:
        encoder = prep.custom_encoders_[column]
        parameters["custom_encoding"] = {key: value for key, value in encoder.items() if not isinstance(value, dict)}
        for key, value in encoder.items():
            if isinstance(value, dict):
                parameters["custom_encoding"][key] = {"cardinality": len(value), "first_20": dict(list(value.items())[:20])}
    for method, (imputer, columns, _, _) in prep.multivariate_imputers_.items():
        if column in columns:
            parameters["multivariate_imputation"] = {"method": method, "n_neighbors": getattr(imputer, "n_neighbors", None), "max_iter": getattr(imputer, "max_iter", None)}
    if prep.datetime_extractor_ is not None and column in prep.date_cols_:
        parameters["date_reference"] = prep.datetime_extractor_.reference_dates_.get(column)
    if column in prep.column_override_params_:
        parameters["step_parameters"] = prep.column_override_params_[column]
    transformer = prep.ngram_text_transformer_ if column in prep.ngram_text_cols_ else prep.text_transformer_
    if transformer is not None and column in transformer.vectorizers_:
        vectorizer = transformer.vectorizers_[column]
        vocabulary = getattr(vectorizer, "vocabulary_", {})
        parameters["text"] = {"method": type(vectorizer).__name__, "vocabulary_size": len(vocabulary),
                              "first_20_terms": list(vocabulary)[:20], "ngram_range": vectorizer.ngram_range}
        svd = getattr(transformer, "svd_models_", {}).get(column)
        if svd is not None:
            parameters["text"]["components"] = svd.n_components
            parameters["text"]["explained_variance_ratio"] = svd.explained_variance_ratio_
    return {key: value for key, value in parameters.items() if value is not None}


def xlsx_cell(value):
    """Sanitize a report value for Excel, including spreadsheet formula prefixes."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        if value.startswith(("=", "+", "-", "@")):
            value = "'" + value
        if len(value) > 32_767:
            value = value[:32_750] + " … [tronqué]"
    return value


def workbook_sheet(book, title, headers, rows):
    """Write a filterable report sheet with restrained column widths."""
    sheet = book.create_sheet(title)
    sheet.append([xlsx_cell(value) for value in headers])
    for row in rows:
        sheet.append([xlsx_cell(value) for value in row])
    sheet.freeze_panes = "B2" if title in {"Features", "Corrélations"} else "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cells in sheet.columns:
        head = cells[0]
        head.fill = PatternFill("solid", fgColor="164E49")
        head.font = Font(color="FFFFFF", bold=True)
        head.alignment = Alignment(wrap_text=True, vertical="center")
        width = max((len(str(cell.value or "")) for cell in cells), default=10)
        sheet.column_dimensions[head.column_letter].width = min(62, max(12, width + 3))
        for cell in cells[1:]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.row_dimensions[1].height = 28


def write_workbook(path, report):
    """Create the four report sheets from the same records exposed as JSON."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    summary = report["summary"]
    def readable_summary():
        for key, value in summary.items():
            if key == "Colonnes exportées":
                yield key, f"{len(value)} colonnes (voir « Colonnes de sortie »)"
            elif key.startswith("Dimensions ") and isinstance(value, list):
                yield key, " × ".join(str(part) for part in value)
            elif key == "Gestion du déséquilibre":
                if isinstance(value, dict):
                    description = (f"{value.get('method', 'aucun')} · "
                                   f"{value.get('synthetic_rows', 0)} lignes synthétiques")
                    provenance = value.get("provenance")
                    if isinstance(provenance, dict) and provenance.get("file"):
                        description += f" · provenance : {provenance['file']}"
                    yield key, description
                else:
                    yield key, "Aucun rééquilibrage"
            else:
                yield key, value
    workbook_sheet(workbook, "Résumé", ["Indicateur", "Valeur"], readable_summary())
    fields = [
        ("column", "Feature"), ("type", "Type détecté"), ("missing_pct", "Manquants (%)"),
        ("unique", "Uniques"), ("mean", "Moyenne"), ("std", "Écart-type"),
        ("min", "Minimum"), ("max", "Maximum"), ("skew", "Asymétrie"), ("top", "Modalité principale"),
        ("outliers", "Extrêmes IQR"), ("anomalies", "Anomalies"), ("recommended", "Recommandation"),
        ("justification", "Justification"), ("user_choice", "Choix utilisateur"),
        ("choice", "Transformation choisie"), ("applied", "Transformation appliquée"),
        ("parameters", "Paramètres appris sur train"), ("outputs", "Colonnes créées"),
        ("dropped", "Supprimée"), ("drop_reason", "Motif"), ("leakage", "Alerte fuite"),
        ("kurtosis", "Kurtosis"), ("steps", "Étapes et preuves"), ("diagnostics", "Diagnostics statistiques"),
        ("leakage_review", "Forte association : disponibilité à vérifier"), ("output_warnings", "Avertissements hérités par sortie"),
    ]
    workbook_sheet(workbook, "Features", [label for _, label in fields],
                   ([row.get(key) for key, _ in fields] for row in report["features"]))
    matrices = report["correlations"]
    rows = ((method, left, right, values[j]) for method, matrix in matrices.items()
            for i, (left, values) in enumerate(zip(matrix["columns"], matrix["values"]))
            for j, right in enumerate(matrix["columns"]) if j >= i)
    workbook_sheet(workbook, "Corrélations", ["Méthode", "Feature 1", "Feature 2", "Coefficient"], rows)
    workbook_sheet(workbook, "Colonnes de sortie", ["Position", "Colonne"],
                   enumerate(summary["Colonnes exportées"], 1))
    workbook_sheet(workbook, "Opérations", ["Variable", "Étape", "Action recommandée", "Mesure", "Seuil", "Explication", "Confiance", "Choix disponibles", "Comparaisons"],
                    ((feature["column"], "type", f"Supprimée ({feature['type']})", feature.get("drop_reason"), "", "", "", "", "")
                     if feature.get("dropped") else
                     (feature["column"], name, step["action"], step["measure"], step["threshold"], step["explanation"],
                      step["confidence"], step.get("choices"), step.get("comparison"))
                     for feature in report["features"]
                     for name, step in ([("type", {})] if feature.get("dropped") else feature.get("steps", {}).items())))
    def diagnostics():
        for feature in report["features"]:
            for key, value in feature.get("diagnostics", {}).items():
                content = json.dumps(value, ensure_ascii=False, allow_nan=False)
                for index in range(0, max(1, len(content)), 30000):
                    yield feature["column"], key, index // 30000 + 1, content[index:index + 30000]
    workbook_sheet(workbook, "Diagnostics", ["Variable", "Champ", "Partie", "Valeur JSON"], diagnostics())
    workbook.save(path)


def export_frame(frame, transformed, passthrough, y, target, original):
    """Restore unmodified features by row ID and append the aligned target."""
    out = transformed.copy()
    for column in passthrough:
        name = column
        while name in out or name == target:
            name += "_brut"
        out[name] = frame.loc[out.index, column]
    reserved = [target] if target else []
    if target and original is not None:
        reserved.append(f"{target}_original")
    renamed = {}
    for name in reserved:
        if name in out:
            replacement = name + "_feature"
            while replacement in out or replacement in reserved:
                replacement += "_feature"
            out = out.rename(columns={name: replacement})
            renamed[name] = replacement
    raw_names = list(out.columns[-len(passthrough):]) if passthrough else []
    return _with_target(out, y, target, original), dict(zip(passthrough, raw_names)), renamed


def fit_target_only(active, target, opts):
    """Fit the supervised target when every explanatory column is left raw."""
    data = active.dropna(subset=[target])
    if len(data) < 10:
        raise InputError("Le mode supervisé nécessite au moins dix lignes avec une cible renseignée.")
    notes = []
    if len(data) < len(active):
        notes.append(f"{len(active) - len(data)} ligne(s) sans valeur de cible supprimée(s) avant le traitement.")
    stratify = data[target] if _looks_like_classification(data[target], opts.task) else None
    split = {"test_size": opts.test_size_pct / 100, "random_state": opts.random_state}
    try:
        train, test = train_test_split(data, stratify=stratify, **split)
    except ValueError:
        train, test = train_test_split(data, **split)
        notes.append("Découpage non stratifié (une classe est trop rare pour être stratifiée).")
    prep = AutoPreprocessor(build_config(opts), task=opts.task)
    train_y = prep._clean_string_and_na(train)[target].dropna()
    test_y = prep._clean_string_and_na(test)[target]
    if train_y.empty:
        raise InputError("Aucune valeur de cible utilisable sur le train.")
    prep._fit_target(train_y)
    y_train, y_test = prep.transform_target(train_y), prep.transform_target(test_y)
    if prep.task_resolved_ == "classification" and np.any(np.asarray(y_test) == -1):
        notes.append("Le test contient des classes absentes du train : elles sont codées -1.")
    transformed = prep.target_stats_.get("transformed", False)
    if transformed:
        notes.append("La cible a été transformée. Sa valeur originale figure aussi dans les exports.")
    return RunResult(prep, target, train.loc[train_y.index].iloc[:, :0], y_train, test.iloc[:, :0], y_test,
                     train.loc[train_y.index, target] if transformed else None,
                     test[target] if transformed else None, notes=notes)


def execute(path, limit, target, task, options, choices, output, dataset_name, artifact=None):
    """Fit on train, transform test, and write an atomic generation of export files."""
    started = time.perf_counter()
    timings = {}
    raw_frame = load_frame(path, limit, keep_nonfinite=True)
    imbalance_options = WebOptions.model_validate(options or {})
    imbalance_method = imbalance_options.imbalance_method
    if imbalance_method != "none" and (target is None or not _looks_like_classification(raw_frame[target], task)):
        raise InputError("La gestion du déséquilibre exige une cible de classification.")
    frame = raw_frame.replace([np.inf, -np.inf], np.nan)
    opts = processing_options(task, options)
    timings["load"] = time.perf_counter() - started
    analysis_started = time.perf_counter()
    analysis, analysis_reused = cached_analysis(frame, path, target, task, options, choices, artifact)
    timings["analysis"] = time.perf_counter() - analysis_started
    timings["analysis_reused"] = analysis_reused
    timings["analysis_original_stages"] = analysis.timings
    configure_started = time.perf_counter()
    overrides, _ = selected_overrides(analysis, target, choices)
    passthrough = [column for column in frame if choices.get(column) == "none"]
    active = frame.drop(columns=passthrough)
    feature_names = active.drop(columns=[target]) if target is not None else active
    config = build_config(opts)
    normalized = AutoPreprocessor(config)._normalize_column_names(feature_names.iloc[:0]).columns
    name_map = dict(zip(feature_names.columns, normalized))
    for key in ("value_bounds", "ordinal_orders"):
        opts.engine_options[key] = {name_map.get(name, name): value
                                    for name, value in (opts.engine_options.get(key) or {}).items()}
    if opts.engine_options.get("keep_columns"):
        opts.engine_options["keep_columns"] = [name_map.get(name, name) for name in opts.engine_options["keep_columns"]]
    for key in ("selection_include", "selection_exclude"):
        if opts.engine_options.get(key):
            if set(opts.engine_options[key]) - set(feature_names.columns):
                raise InputError("La sélection manuelle contient une colonne source inconnue, brute ou la cible.")
            opts.engine_options[key] = [name_map[name] for name in opts.engine_options[key]]
    opts.column_overrides = {name_map.get(column, column): value for column, value in overrides.items()}
    if passthrough and len(feature_names.columns):
        # Rows equal on processed features can differ in a passthrough feature.
        # Leave those rows intact unless the user explicitly requests deduplication.
        if not any((isinstance(choice, str) and choice.startswith("duplicates:")) or
                   (isinstance(choice, dict) and "duplicates" in choice) for choice in choices.values()):
            first = name_map[feature_names.columns[0]]
            opts.column_overrides[first] = ColumnOverride({
                **_override_as_choices(opts.column_overrides.get(first)), "duplicates": "KEEP",
            })
    timings["configuration"] = time.perf_counter() - configure_started
    pipeline_started = time.perf_counter()
    if target is not None:
        if active[target].notna().sum() < 10:
            raise InputError("Le mode supervisé nécessite au moins dix lignes avec une cible renseignée.")
        result = run_preprocessing(active, target, opts, analysis_result=analysis) if name_map else fit_target_only(active, target, opts)
    else:
        if len(active) < 2:
            raise InputError("Le découpage nécessite au moins deux lignes.")
        train_data, test_data = train_test_split(active, test_size=opts.test_size_pct / 100,
                                                 random_state=opts.random_state)
        if not len(active.columns):
            result = RunResult(AutoPreprocessor(build_config(opts)), None, train_data, None, test_data)
        else:
            result = run_preprocessing(train_data, None, opts, analysis_result=analysis)
            result.X_test = result.prep.transform(test_data)
            result.notes = ["Mode sans cible : ajustement sur train et transformation du test."]
    timings["pipeline_total"] = time.perf_counter() - pipeline_started
    timings["pipeline_stages"] = getattr(result.prep, "stage_timings_", {})
    export_started = time.perf_counter()
    train, raw_outputs, renamed = export_frame(
        raw_frame, result.X_train, passthrough, result.y_train, target, result.y_train_original)
    test, _, _ = export_frame(raw_frame, result.X_test, passthrough, result.y_test, target, result.y_test_original)
    imbalance = None
    resampled_marker = None
    if target is not None and _looks_like_classification(frame[target], task):
        before_train = class_profile(train[target])
        before_test = class_profile(test[target])
        if len(before_test["counts"]) != len(before_train["counts"]):
            raise InputError("Le découpage ne contient pas toutes les classes dans le train et le test ; augmentez les effectifs minoritaires.")
        if imbalance_method != "none":
            if "__synthetic__" in train.columns:
                raise InputError("La colonne de provenance __synthetic__ existe déjà ; renommez-la avant le rééquilibrage.")
            counts = train[target].value_counts()
            no_change = (imbalance_method in {"smote", "smotenc", "random_over", "random_under"}
                         and len(counts) == 2 and counts.min() / counts.max() >= imbalance_options.imbalance_ratio)
            if no_change:
                imbalance = {"method": "none (ratio already reached)", "requested_method": imbalance_method,
                             "categorical_columns": [], "before": before_train, "after": before_train,
                             "synthetic_rows": 0, "duplicated_rows": 0, "warnings":
                             ["Ratio cible déjà atteint dans le train : aucune ligne créée ou retirée."], "seconds": 0.}
            else:
                original_rows = len(train)
                values, labels, imbalance = rebalance(train.drop(columns=target), train[target],
                    imbalance_method, imbalance_options.imbalance_ratio, imbalance_options.imbalance_neighbors,
                    imbalance_options.random_state)
                train = values
                train[target] = labels.to_numpy()
                train["__synthetic__"] = False
                if imbalance_method in {"smote", "smotenc", "random_over"}:
                    train.loc[original_rows:, "__synthetic__"] = True
                test["__synthetic__"] = False
        else:
            imbalance = {"method": "none", "before": before_train, "after": before_train,
                         "synthetic_rows": 0, "warnings": [], "seconds": 0.}
        imbalance["test"] = before_test
        imbalance["parameters"] = {"ratio": imbalance_options.imbalance_ratio,
                                   "k_neighbors": imbalance_options.imbalance_neighbors,
                                   "seed": imbalance_options.random_state}
        timings["resampling"] = imbalance["seconds"]
        if imbalance_method != "none":
            resampled_marker = (train.pop("__synthetic__").to_numpy(dtype=bool)
                                if "__synthetic__" in train else np.zeros(len(train), dtype=bool))
            test.drop(columns="__synthetic__", inplace=True, errors="ignore")
            imbalance["provenance"] = {
                "file": DOWNLOADS["train_provenance"], "row_index_base": 0,
                "marked_rows": int(resampled_marker.sum()),
                "note": "Positions du train rééchantillonné ; __synthetic__ est exclu des CSV train, test et complet. Les doublons du sur-échantillonnage aléatoire sont aussi marqués.",
            }
    if not train.columns.equals(test.columns):
        raise RuntimeError("Schémas train/test incompatibles.")
    core = result.prep.report()
    detail = core.pop("colonnes_detaillees").to_dict("records")
    fitted = {row["Colonne"]: row for row in detail}
    eda_result = dataset_eda(frame, target, task)
    features = []
    normalized_names = sorted((name for name in name_map.values()
                               if name not in result.prep.columns_to_drop_), key=len, reverse=True)
    owners = {name: [] for name in normalized_names}
    for name in result.X_train.columns:
        owner = next((candidate for candidate in normalized_names if name == candidate or name.startswith(candidate + "_")), None)
        if owner is not None:
            owners[owner].append(renamed.get(name, name))
    for stats, row in zip(eda_result["features"], analysis.columns):
        column = row["colonne"]
        choice = choices.get(column, "recommended")
        rec = recommendation(row, target)
        name = name_map.get(column)
        actual = fitted.get(name, {})
        outputs = owners.get(name, [])
        applied = actual.get("Action appliquée")
        parameters = trained_parameters(result.prep, name) if name is not None else {}
        if column == target:
            outputs = [target] + ([f"{target}_original"] if result.y_train_original is not None else [])
            applied = "CIBLE"
            parameters = result.prep.target_stats_
        elif column in raw_outputs:
            outputs = [raw_outputs[column]]
            applied = "NONE"
        elif not applied or pd.isna(applied):
            applied = "DROP" if not outputs else "KEEP"
        status = actual.get("Statut", "")
        dropped = column != target and not outputs
        leak = (row.get("score_fuite") if row.get("fuite_suspectee") else
                core["scores_fuite_par_colonne"].get(name) if name in core["fuites_suspectees"] else None)
        features.append({**stats, "type": row["type"], "top": stats["top"][0]["label"] if stats["top"] else None,
                         "outliers": stats.get("iqr", {}).get("count"), "anomalies": rec["anomalies"],
                         "recommended": rec["action"], "justification": rec["justification"], "choice": choice,
                         "steps": rec["steps"], "diagnostics": row,
                         "user_choice": "cible" if column == target else
                         ("personnalisé" if isinstance(choice, dict) else {"recommended": "acceptée", "none": "aucune"}.get(choice, "alternative")),
                         "applied": applied, "parameters": parameters, "outputs": outputs, "dropped": dropped,
                         "leakage_review": leakage_review(row) if not dropped else None,
                         "output_warnings": {output: {"source": column, "warning": leakage_review(row)}
                                             for output in outputs} if leakage_review(row) else {},
                         "drop_reason": ("Supprimée automatiquement : fuite de cible détectée (recommandation statistique acceptée)."
                                         if row.get("fuite_suspectee") and choice == "recommended" else
                                         status if status.startswith("SUPPRIM") else
                                         "Aucune colonne après sélection") if dropped else None,
                         "leakage": leak})
    warnings = list(dict.fromkeys(result.notes + result.prep.warnings_))
    if imbalance:
        warnings.extend(imbalance.get("warnings", []))
    warnings.extend("Comparaison d’alternative : " + warning for row in analysis.columns
                    for warning in row.get("avertissements_numeriques", []) if warning not in warnings)
    warnings.extend(f"ALERTE FUITE DE CIBLE : {row['colonne']!r}, score CV {row['score_fuite']:.3f}, "
                    + ("conservation explicite." if choices.get(row["colonne"], "recommended") not in ("recommended", {}) else "suppression recommandée appliquée.")
                    for row in analysis.columns if row.get("fuite_suspectee"))
    warnings.extend(f"{feature['column']} : {feature['leakage_review']} "
                    f"Avertissement hérité par les sorties : {', '.join(feature['outputs']) or 'aucune'}."
                    for feature in features if feature["leakage_review"])
    if passthrough:
        warnings.append("Les colonnes laissées brutes peuvent contenir du texte et des valeurs manquantes.")
    summary = {"Dataset": dataset_name, "Mode": "supervisé" if target else "non supervisé", "Cible": target,
               "Tâche": core["tache_detectee"], "Dimensions initiales": frame.shape,
               "Gestion du déséquilibre": imbalance,
               "Dimensions train": train.shape, "Dimensions test": test.shape,
               "Dimensions complètes": (len(train) + len(test), train.shape[1]),
               "Part test demandée (%)": opts.test_size_pct, "Graine": opts.random_state,
               "Colonnes exportées": train.columns, "Avertissements": warnings,
               "Nettoyage du train": core["nettoyage_dataset"],
               "Plafonds Pearson / V de Cramér": [MATRIX_LIMIT, CATEGORY_LIMIT],
               "Options appliquées": options, "Choix par variable": choices, "Analyse réutilisée": analysis_reused,
               "Échantillonnage de l’analyse": analysis.sample_note or "Aucun échantillonnage du profil descriptif",
               "Échantillonnage des diagnostics": "CV de fuite : au plus 20 000 lignes (5 000 au-delà de 100 000 lignes), tirage sans remise avec classe rare préservée, graine configurée ; mécanismes de manquants : au plus 5 000 lignes.",
               "Échantillonnage EDA": eda_result["sample_note"],
               "Politique de fuite": "drop_leaky_columns contrôle le filtre du fit. Indépendamment, accepter une recommandation DROP du détecteur statistique supprime la variable ; un choix explicite permet sa conservation.",
               "Avertissements par sortie": {name: warning for feature in features for name, warning in feature["output_warnings"].items()},
               "Types exportés": train.dtypes.astype(str).to_dict()}
    report = json_safe({"summary": summary, "features": features, "core": {**core, "colonnes_detaillees": detail},
                        "correlations": {name: eda_result[name] for name in ("pearson", "cramers_v")}})
    timings["export_preparation"] = time.perf_counter() - export_started
    csv_started = time.perf_counter()
    if resampled_marker is not None:
        pd.DataFrame({"row_index": np.flatnonzero(resampled_marker), "__synthetic__": True}).to_csv(
            output / DOWNLOADS["train_provenance"], index=False, encoding="utf-8-sig")
    for split, table in (("train", train), ("test", test)):
        table.to_csv(output / DOWNLOADS[split], index=False, encoding="utf-8-sig")
    full = output / DOWNLOADS["full"]
    # Reuse the serialized rows instead of formatting every train/test cell twice.
    # Count the encoded logical header, which may contain quotes or embedded newlines.
    header = train.iloc[:0].to_csv(index=False).encode("utf-8-sig")
    with (output / DOWNLOADS["train"]).open("rb") as first, (output / DOWNLOADS["test"]).open("rb") as second, full.open("wb") as combined:
        if first.read(len(header)) != header or second.read(len(header)) != header:
            raise RuntimeError("En-têtes CSV train/test incompatibles ; export complet non publié.")
        first.seek(0)
        shutil.copyfileobj(first, combined, length=1024 * 1024)
        shutil.copyfileobj(second, combined, length=1024 * 1024)
    timings["csv_export"] = time.perf_counter() - csv_started
    workbook_started = time.perf_counter()
    write_workbook(output / DOWNLOADS["report_xlsx"], report)
    timings["excel_report"] = time.perf_counter() - workbook_started
    model_started = time.perf_counter()
    if result.prep.is_fitted_:
        result.prep.save(output / DOWNLOADS["preprocessor"])
    timings["model_export"] = time.perf_counter() - model_started
    timings["total"] = time.perf_counter() - started
    report["summary"]["Durées réelles (secondes)"] = json_safe(timings)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return json_safe({"input_shape": frame.shape, "train_shape": train.shape, "test_shape": test.shape,
                      "imbalance": imbalance,
                      "full_shape": summary["Dimensions complètes"], "output_columns": train.columns,
                      "warnings": warnings, "timings": timings,
                      "downloads": [kind for kind, filename in DOWNLOADS.items() if (output / filename).is_file()]})


def evaluate(path, limit, target, task, options, family, folds, time_column, candidates):
    """Reuse the fold-safe comparison module within the existing worker limits."""
    from model_evaluation import compare_preprocessing
    if target is None:
        raise InputError("La validation prédictive nécessite une cible.")
    frame = load_frame(path, limit)
    if time_column is not None and time_column not in frame:
        raise InputError("Colonne temporelle inconnue.")
    try:
        return json_safe(compare_preprocessing(frame, target, family, config=configured_config(frame, target, processing_options(task, options)),
                                              task=task, cv=folds, time_column=time_column, candidates=tuple(candidates)))
    except ImportError as exc:
        raise InputError("La bibliothèque du modèle choisi n’est pas installée sur ce serveur.") from exc
    except ValueError as exc:
        raise InputError(str(exc)) from exc


def evaluate_nested(path, limit, target, task, options, family, outer_folds,
                    inner_folds, time_column, group_column, selection_counts,
                    scalers, model_params):
    """Evaluate raw data with an independent outer CV inside the worker limits."""
    from model_evaluation import nested_cross_validate
    if target is None:
        raise InputError("La validation imbriquée nécessite une cible.")
    frame = load_frame(path, limit)
    if group_column is not None and group_column not in frame:
        raise InputError("Colonne de groupe inconnue.")
    if group_column == target or time_column == target:
        raise InputError("La cible ne peut pas servir de groupe ou de date.")
    groups = frame[group_column].to_numpy() if group_column is not None else None
    features = frame.drop(columns=[group_column]) if group_column is not None else frame
    config = configured_config(features, target, processing_options(task, options))
    counts = selection_counts or [None]
    scales = scalers or [None]
    candidates = [{**({"selection_count": count} if count is not None else {}),
                   **({"scaler_type": scale} if scale is not None else {})}
                  for count in counts for scale in scales]
    try:
        result = nested_cross_validate(features, target, family, config=config,
                                       task=task, outer_cv=outer_folds, inner_cv=inner_folds,
                                       candidates=candidates, model_params=model_params,
                                       time_column=time_column, groups=groups)
        result.pop("fitted_outer_pipelines")
        return json_safe(result)
    except ImportError as exc:
        raise InputError("La bibliothèque du modèle choisi n’est pas installée sur ce serveur.") from exc
    except ValueError as exc:
        raise InputError(str(exc)) from exc


def result_eda(output):
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(output / DOWNLOADS["full"], low_memory=False).replace([np.inf, -np.inf], np.nan)
    result = dataset_eda(frame, report["summary"]["Cible"], report["summary"]["Tâche"])
    inherited = report["summary"].get("Avertissements par sortie", {})
    for feature in result["features"]:
        if feature["column"] in inherited:
            feature["leakage_review"] = inherited[feature["column"]]
    return json_safe(result)


def export_excel(output):
    """Stream the exported rows to an atomic workbook with formula-safe strings."""
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    rows, columns = report["summary"]["Dimensions complètes"]
    if rows > 1048575 or columns > 16384:
        raise InputError("Les dimensions dépassent les limites Excel. Utilisez les fichiers CSV.")
    destination = output / "dataset_transforme.xlsx"
    temporary = output / "dataset_transforme.tmp.xlsx"
    dtypes = report["summary"]["Types exportés"]
    book = Workbook(write_only=True)
    sheet = book.create_sheet("Données transformées")
    try:
        with (output / DOWNLOADS["full"]).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            sheet.append([xlsx_cell(name) for name in header])
            for row in reader:
                values = []
                for name, value in zip(header, row):
                    kind = dtypes[name].lower()
                    if value and ("float" in kind or "int" in kind):
                        values.append(float(value) if "float" in kind else int(value))
                    elif value and kind == "bool":
                        values.append(value.lower() == "true")
                    else:
                        values.append(xlsx_cell(value) if value else None)
                sheet.append(values)
        book.save(temporary)
        os.replace(temporary, destination)
    finally:
        book.close()
        temporary.unlink(missing_ok=True)
