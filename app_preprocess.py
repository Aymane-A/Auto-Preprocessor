"""
AutoPreprocessor GUI : nettoyage et transformation automatiques de n'importe quel dataset (Tkinter).

Utilisation :  py app_preprocess.py
Prérequis   :  auto_preprocessor.py dans le même dossier ; pandas, numpy, scikit-learn, scipy, joblib.
               (Excel : py -m pip install openpyxl)

Flux : ouvrir un fichier -> choisir la cible et les options -> lancer -> lire le rapport -> exporter.
La logique (sans interface) est regroupée dans la première moitié du fichier ; l'interface Tkinter
est dans la classe App, en seconde moitié.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from auto_preprocessor import (Action, AnalysisResult, AutoPreprocessor, ColumnOverride,
                               PreprocessConfig, recommendation_confidence)

NO_TARGET_LABEL = "(aucune : sans cible)"


# =============================================================================
# 1. LOGIQUE (SANS INTERFACE)
# =============================================================================

def _sniff_separator(sample: str) -> str:
    """Devine le séparateur d'un fichier texte (',', ';', tabulation, '|')."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(path: str) -> pd.DataFrame:
    """Lit un fichier tabulaire : CSV/TXT/TSV (séparateur et encodage détectés), Excel, JSON, Parquet."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError("Lecture Excel impossible : installez openpyxl (py -m pip install openpyxl).") from exc
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".parquet":
        return pd.read_parquet(path)

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                sample = handle.read(50000)
            sep = _sniff_separator(sample)
            return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Impossible de lire le fichier (encodage) : {last_error}")


@dataclass
class Options:
    """Options choisies dans l'interface (valeurs par défaut = celles d'AutoPreprocessor)."""
    task: str = "auto"                     # auto | classification | regression
    test_size_pct: int = 20                # 0 = pas de découpage train/test
    scaler_type: str = "standard"          # standard | robust | minmax | none
    outlier_method: str = "winsorize"      # winsorize | iqr | none
    cardinality_threshold: int = 10
    missing_threshold: float = 0.90
    correlation_threshold: float = 0.95
    rare_category_threshold: float = 0.01
    skew_threshold: float = 1.0
    transform_target: bool = True
    drop_leaky_columns: bool = False
    encode_discrete_as_categorical: bool = False
    random_state: int = 42
    column_overrides: Dict[str, ColumnOverride] = field(default_factory=dict)


def build_config(opts: Options) -> PreprocessConfig:
    """Construit un PreprocessConfig en ignorant les options inconnues de la version du module."""
    wanted = {
        "scaler_type": opts.scaler_type,
        "outlier_method": opts.outlier_method,
        "cardinality_threshold": opts.cardinality_threshold,
        "missing_threshold": opts.missing_threshold,
        "correlation_threshold": opts.correlation_threshold,
        "rare_category_threshold": opts.rare_category_threshold,
        "skew_threshold": opts.skew_threshold,
        "transform_target": opts.transform_target,
        "drop_leaky_columns": opts.drop_leaky_columns,
        "encode_discrete_as_categorical": opts.encode_discrete_as_categorical,
        "random_state": opts.random_state,
        "verbose": 0,
        "column_overrides": opts.column_overrides,
    }
    valid = {f.name for f in dataclasses.fields(PreprocessConfig)}
    return PreprocessConfig(**{k: v for k, v in wanted.items() if k in valid})


def build_analysis_table(result: AnalysisResult, choices: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Construit la table de présentation/export sans dépendre de Tkinter."""
    table = result.to_dataframe()
    choices = choices or {}
    def display(col: str, recommended: str) -> str:
        value = choices.get(col, recommended)
        return "; ".join(f"{k}={v[0] if isinstance(v, (tuple, list)) else v}" for k, v in value.items()) if isinstance(value, dict) else str(value)
    table["choix_final"] = ["cible" if row.get("est_cible") else display(row["colonne"], row["action"])
                            for row in result.columns]
    table["état_choix"] = ["cible" if row.get("est_cible") else
                           "✎ modifié" if row["colonne"] in choices and choices[row["colonne"]] != row["action"]
                           else "recommandation" for row in result.columns]
    summaries = [analysis_recommendation_summary(row) for row in result.columns]
    table["raison"] = [summary["raison"] for summary in summaries]
    table["preuve"] = [summary["preuve"] for summary in summaries]
    table["explication"] = [summary["explication"] for summary in summaries]
    for index, row in enumerate(result.columns):
        for step, detail in row.get("steps", {}).items():
            table.loc[index, f"recommandation_{step}"] = detail["action"]
            table.loc[index, f"preuve_{step}"] = f"{detail.get('measure', '')} ; seuil : {detail.get('threshold', '')}"
            table.loc[index, f"explication_{step}"] = pedagogical_explanation(
                step, detail.get("action", "AUTO"), detail.get("measure", ""),
                detail.get("threshold", ""), row.get("type", ""))
    return table


ANALYSIS_STEP_ORDER = ["nettoyage", "clean_missing_tokens", "clean_variants", "clean_numeric_text",
                       "clean_dates", "clean_impossible", "duplicates", "type", "missing", "outliers",
                       "distribution", "scaling", "encoding", "dates", "text", "selection"]


def pedagogical_explanation(step: str, action: str, measure: str, threshold: str,
                            column_type: str = "") -> str:
    """Explication courte de l'action, de son usage et du compromis principal."""
    action = str(action).upper()
    descriptions = {
        ("distribution", "YEO_JOHNSON"): "Yeo-Johnson réduit l'asymétrie sans exiger des valeurs positives; cela aide les modèles linéaires, tandis que les arbres y sont moins sensibles.",
        ("distribution", "LOG1P"): "Log1p compresse la longue queue droite et convient aux valeurs positives très asymétriques; elle ne convient pas aux valeurs inférieures à -1.",
        ("distribution", "BOX_COX"): "Box-Cox rapproche une variable positive d'une distribution symétrique; elle exige des valeurs strictement positives.",
        ("distribution", "SQRT"): "La racine carrée réduit modérément l'effet des grandes valeurs; elle convient aux comptes positifs et change l'échelle des coefficients.",
        ("distribution", "NONE"): "Aucune transformation conserve l'échelle observée; c'est adapté si la forme n'est pas assez éloignée du seuil, mais les modèles linéaires restent sensibles à l'asymétrie.",
        ("scaling", "ROBUST"): "RobustScaler centre par médiane et échelle par IQR; il convient aux outliers, mais ne rend pas la distribution normale.",
        ("scaling", "STANDARD"): "StandardScaler centre et réduit la variable; il convient aux distributions sans outliers marqués, mais reste sensible aux valeurs extrêmes.",
        ("scaling", "MINMAX"): "MinMax ramène les valeurs dans une plage bornée; il convient aux variables bornées, mais les extrêmes déterminent toute l'échelle.",
        ("scaling", "MAXABS"): "MaxAbs divise par la valeur absolue maximale et préserve les zéros; il convient aux données creuses, mais reste sensible aux extrêmes.",
        ("scaling", "NONE"): "Aucune mise à l'échelle garde les unités; c'est utile pour les arbres et rend les grandeurs incomparables pour certains modèles.",
        ("missing", "MEDIAN"): "La médiane remplace les manquants par une valeur centrale robuste; elle convient aux variables asymétriques, mais réduit leur dispersion.",
        ("missing", "MEAN"): "La moyenne remplace les manquants par le centre arithmétique; elle convient aux données symétriques, mais est sensible aux outliers.",
        ("missing", "MODE"): "Le mode remplace les manquants par la valeur la plus fréquente; il préserve une modalité existante, mais peut renforcer son déséquilibre.",
        ("missing", "MISSING_CATEGORY"): "Une modalité dédiée conserve l'information qu'une valeur manque; elle convient aux catégories, mais crée une modalité supplémentaire.",
        ("missing", "CONSTANT"): "Une constante explicite rend les manquants visibles; elle est utile si sa valeur a un sens métier, mais peut créer un point artificiel.",
        ("outliers", "WINSORIZE"): "La winsorisation borne les queues selon les quantiles; elle limite l'influence des extrêmes, mais écrase leurs différences.",
        ("outliers", "IQR"): "Le clipping IQR borne les valeurs au seuil de 1,5 IQR; il réduit l'effet des extrêmes, mais peut couper des valeurs rares légitimes.",
        ("outliers", "NONE"): "Aucun traitement conserve les extrêmes; c'est préférable s'ils sont réels, mais ils peuvent dominer les modèles sensibles à l'échelle.",
        ("encoding", "ONEHOT"): "One-Hot crée une variable par modalité; il convient aux catégories sans ordre, mais augmente la dimension quand la cardinalité est forte.",
        ("encoding", "TARGET_ENCODE"): "Target encoding remplace chaque modalité par une statistique de cible calculée hors pli; il réduit la dimension, mais exige une cible et un contrôle du surapprentissage.",
        ("encoding", "FREQUENCY_ENCODE"): "L'encodage fréquence remplace chaque modalité par sa fréquence; il compacte les catégories, mais confond celles de fréquence égale.",
        ("encoding", "ORDINAL"): "L'encodage ordinal associe un nombre selon l'ordre fourni; il convient aux niveaux ordonnés, mais invente un ordre s'il est mal défini.",
        ("encoding", "BINARY_MAP"): "Le mappage binaire code les deux modalités en 0/1; il est compact, mais le sens des codes doit rester stable.",
        ("dates", "DATE_FEATURES"): "Les composantes calendaires extraient année, mois et jour; elles rendent la date exploitable, mais n'expriment pas seules la saisonnalité cyclique.",
        ("text", "TFIDF_SVD"): "TF-IDF suivi de SVD résume le vocabulaire en composantes numériques; c'est utile pour le texte libre, mais les composantes sont moins interprétables.",
        ("type", "DROP"): "La suppression retire cette colonne des features; elle convient aux identifiants ou fuites détectés, mais élimine toute information qu'elle portait.",
        ("type", "AUTO"): "Le type détecté conserve le traitement automatique adapté au profil observé; cela évite une conversion manuelle, mais dépend des heuristiques de détection.",
        ("selection", "DROP"): "La sélection retire la variable redondante ou sans variance; elle simplifie le modèle, mais peut enlever un signal complémentaire.",
        ("selection", "KEEP"): "La sélection conserve la variable; c'est utile si elle porte un signal distinct, mais les redondances peuvent rendre le modèle instable.",
        ("clean_missing_tokens", "TO_NAN"): "Les marqueurs textuels connus deviennent des valeurs manquantes; cela permet leur imputation, mais suppose que ces chaînes ne sont pas des modalités légitimes.",
        ("clean_variants", "MERGE"): "La normalisation fusionne les variantes de casse, espaces et accents; elle évite des catégories artificiellement distinctes, mais peut confondre des libellés réellement différents.",
        ("clean_numeric_text", "CONVERT"): "La conversion extrait la valeur numérique et retire l'unité; elle rend la mesure exploitable, mais perd l'unité originale.",
        ("clean_dates", "PARSE"): "Le parsing convertit les chaînes reconnues en dates; il permet l'extraction calendaire, mais les formats ambigus peuvent être mal interprétés.",
        ("clean_impossible", "NAN"): "Les valeurs hors bornes métier deviennent manquantes; cela évite de traiter une valeur impossible comme réelle, mais demande une imputation ensuite.",
        ("duplicates", "DROP"): "La suppression retire les lignes strictement dupliquées; elle évite de surpondérer des répétitions accidentelles, mais peut enlever des répétitions légitimes.",
    }
    descriptions.update({
        ("type", "FORCE_NUMERIC"): "Forcer numérique convient à des nombres importés comme texte, mais les valeurs non convertibles deviennent manquantes.",
        ("type", "FORCE_CATEGORICAL"): "Forcer catégoriel convient aux codes numériques, mais une forte cardinalité augmente la dimension.",
        ("type", "FORCE_DATE"): "Forcer date convient aux dates encodées en texte, mais des formats ambigus peuvent être mal interprétés.",
        ("type", "FORCE_TEXT"): "Forcer texte convient aux descriptions libres, mais perd la structure numérique ou catégorielle.",
        ("type", "KEEP_RAW"): "Garder brut convient aux libellés dont la forme exacte compte, mais conserve les faux manquants et variantes.",
        ("missing", "MISSING_INDICATOR"): "L'indicateur convient si l'absence porte un signal; il augmente la dimension et doit accompagner une imputation.",
        ("missing", "AUTO"): "L'imputation automatique est ajustée sur le train; elle évite la fuite, mais peut masquer un mécanisme informatif.",
        ("outliers", "AUTO"): "Le traitement automatique applique la règle configurée sur le train; il limite l'influence des extrêmes, mais peut écraser des valeurs rares légitimes.",
        ("clean_missing_tokens", "KEEP"): "Garder les marqueurs convient s'ils sont des modalités réelles, mais ils ne seront pas traités comme manquants.",
        ("clean_missing_tokens", "SELECTED_TO_NAN"): "Convertir seulement les chaînes choisies convient aux marqueurs ambigus, mais les valeurs oubliées resteront des modalités.",
        ("clean_variants", "KEEP"): "Garder les variantes convient si casse ou accents distinguent des entités, mais multiplie les catégories équivalentes.",
        ("clean_numeric_text", "KEEP"): "Garder le texte convient aux codes et libellés avec unités, mais empêche les opérations numériques.",
        ("clean_dates", "KEEP"): "Garder le texte convient si la valeur n'est pas une date exploitable, mais bloque les features calendaires.",
        ("clean_dates", "DROP_COLUMN"): "Supprimer convient aux dates illisibles ou inutiles, mais fait perdre le signal temporel.",
        ("clean_impossible", "CLIP"): "Clipper convient aux erreurs proches des bornes, mais ramène plusieurs observations à la même valeur limite.",
        ("clean_impossible", "DROP_ROWS"): "Supprimer les lignes convient si les valeurs sont impossibles, mais réduit l'échantillon et peut introduire un biais.",
        ("clean_impossible", "IGNORE"): "Ignorer convient si la règle métier ne s'applique pas, mais conserve l'influence des anomalies.",
        ("duplicates", "KEEP"): "Garder convient si les répétitions sont des événements réels, mais surpondère les doublons accidentels.",
    })
    base = descriptions.get((step, action))
    if base is None:
        if action in {"AUTO", "NONE", "KEEP", "IGNORE"}:
            base = f"L'action conserve le comportement ou les valeurs actuelles; elle convient si la mesure reste compatible avec le seuil observé ({threshold}), mais laisse les limites de cette représentation en place."
        else:
            base = f"{action} applique le traitement prévu pour l'étape {step}; il est proposé au vu de {measure} et du seuil {threshold}. Vérifiez que cette hypothèse métier convient à la colonne."
    evidence = f"Preuve : {measure}; seuil : {threshold}." if measure or threshold else ""
    return f"{base} {evidence}".strip()


def recommendation_help(step: str, action: str, measure: str = "mesure observée",
                        threshold: str = "seuil indiqué") -> str:
    """Aide pour l'action sélectionnée, notamment quand elle n'est pas recommandée."""
    return pedagogical_explanation(step, action, measure, threshold)


def analysis_recommendation_summary(row: Dict[str, Any]) -> Dict[str, str]:
    """Construit raison courte, preuve et explication de la recommandation principale."""
    if row.get("est_cible"):
        return {"raison": "Colonne cible : exclue des transformations de features.", "preuve": "—", "explication": "La cible sert à mesurer les relations et à ajuster les transformations supervisées; elle n'est pas une feature transformée."}
    action = str(row.get("action", "AUTO"))
    kind = str(row.get("type", ""))
    steps = row.get("steps", {})
    if action == "DROP" and row.get("alerte_identifiant"):
        ratio = float(row.get("ratio_uniques_lignes", 0) or 0) * 100
        reason = f"{ratio:.1f} % d'uniques : profil d'identifiant, suppression recommandée."
        measure, threshold, step = f"ratio uniques={ratio:.1f}%", "> 90 % ou nom d'identifiant", "type"
    elif action == "DROP" and kind.startswith("DROP"):
        measure = str(row.get("raison", "Colonne signalée par le détecteur"))
        threshold = "règle de détection du module"
        step = "type"
        reason = f"{measure} : suppression de la colonne recommandée."
    elif action == "YEO_JOHNSON":
        skew = float(row.get("skew", np.nan))
        threshold = steps.get("distribution", {}).get("threshold", "|skew| > 1")
        measure = f"skew={skew:.2f}"
        reason = f"{measure} ({threshold}) : Yeo-Johnson réduit l'asymétrie."
        step = "distribution"
    elif action == "KEEP_NUMERIC":
        skew = float(row.get("skew", np.nan))
        threshold = steps.get("distribution", {}).get("threshold", "|skew| > 1")
        measure = f"skew={skew:.2f}"
        missing = float(row.get("manquants_pct", 0) or 0)
        reason = (f"{missing:.1f} % de manquants : imputation médiane; asymétrie faible ({measure})."
                  if missing > 0 else f"{measure} (seuil {threshold}) : asymétrie faible, conservation numérique.")
        step = "distribution"
    elif action == "DATE_FEATURES":
        measure = f"dates uniques={row.get('uniques', 0)}"; threshold = "dates parsables"; step = "dates"
        reason = f"{measure} : extraction des composantes calendaires recommandée."
    elif action == "TFIDF_SVD":
        measure = f"modalités uniques={row.get('uniques', 0)}"; threshold = "texte libre détecté"; step = "text"
        reason = f"{measure} : texte libre, résumé TF-IDF/SVD recommandé."
    else:
        measure = f"type={kind}; uniques={row.get('uniques', 0)}"
        threshold = steps.get("type", {}).get("threshold", "règle du type détecté")
        step = "encoding" if "encoding" in steps else "type"
        reason = f"{measure} : {action} selon la règle de type ({threshold})."
    explanation = pedagogical_explanation(step, action, measure, threshold, kind)
    return {"raison": reason, "preuve": f"{measure} ; seuil : {threshold}", "explication": explanation}


def build_analysis_detail(result: AnalysisResult, column: Optional[str],
                          choices: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Construit l'en-tête et les lignes de détail d'analyse, sans Tkinter."""
    if not column:
        return {"header": "Sélectionnez une colonne pour voir ses statistiques et les explications.", "rows": []}
    row = next((item for item in result.columns if item["colonne"] == column), None)
    if row is None:
        raise ValueError(f"Colonne inconnue : {column}")
    if row.get("est_cible"):
        return {"header": f"{column} | cible — aucune recommandation de transformation de feature.", "rows": []}
    stats = [f"manquants={row.get('manquants_pct', np.nan):.1f}%", f"uniques={row.get('uniques', 0)}"]
    for key, label in (("skew", "skew"), ("outliers_pct", "outliers"), ("relation_cible", "relation cible")):
        value = row.get(key, np.nan)
        if pd.notna(value):
            stats.append(f"{label}={float(value):.2f}" + ("%" if key == "outliers_pct" else ""))
    header = f"{column} | type={row.get('type')} | " + " | ".join(stats)
    selected = (choices or {}).get(column, {})
    if not isinstance(selected, dict):
        legacy_map = {"YEO_JOHNSON": ("distribution", "YEO_JOHNSON"), "LOG1P": ("distribution", "LOG1P"),
                      "WINSORIZE": ("outliers", "WINSORIZE"), "SCALE_STANDARD": ("scaling", "STANDARD"),
                      "SCALE_ROBUST": ("scaling", "ROBUST"), "SCALE_MINMAX": ("scaling", "MINMAX"),
                      "NO_SCALE": ("scaling", "NONE"), "ONEHOT": ("encoding", "ONEHOT"),
                      "TARGET_ENCODE": ("encoding", "TARGET_ENCODE"), "FREQUENCY_ENCODE": ("encoding", "FREQUENCY_ENCODE"),
                      "ORDINAL": ("encoding", "ORDINAL"), "BINARY_MAP": ("encoding", "BINARY_MAP"),
                      "DATE_FEATURES": ("dates", "DATE_FEATURES"), "TFIDF_SVD": ("text", "TFIDF_SVD"),
                      "DROP": ("type", "DROP")}
        selected = {}
        mapped = legacy_map.get(str((choices or {}).get(column, "")).upper())
        if mapped:
            selected[mapped[0]] = mapped[1]
    detail_rows = []
    for step in ANALYSIS_STEP_ORDER:
        detail = row.get("steps", {}).get(step)
        if not detail or len(detail.get("choices", [])) <= 1:
            continue
        recommended = str(detail.get("action", "AUTO"))
        saved = selected.get(step, recommended)
        action = saved[0] if isinstance(saved, (tuple, list)) and len(saved) == 2 else str(saved)
        measure, threshold = str(detail.get("measure", "")), str(detail.get("threshold", ""))
        detail_rows.append({"etape": step, "recommandation": recommended, "preuve": f"{measure} ; seuil : {threshold}",
                            "explication": pedagogical_explanation(step, recommended, measure, threshold, row.get("type", "")),
                            "choix": action, "modifie": action != recommended,
                            "aide_choix": recommendation_help(step, action), "detail_analyse": row})
    return {"header": header, "rows": detail_rows}


def format_analysis_value(column: str, value: Any) -> str:
    """Formate une cellule d'affichage sans altérer les données triables/exportées."""
    if value is None or value is pd.NA:
        return "—"
    if not isinstance(value, (dict, list, tuple)):
        if pd.isna(value):
            return "—"
        s_val = str(value).strip().lower()
        if s_val in {"nan", "none", "null", "—", "-", ""}:
            return "—"
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, bool):
        number = float(value)
        if not np.isfinite(number):
            return "—"
        if column in {"manquants_pct", "outliers_pct"}:
            return f"{number:.1f}%"
        return f"{number:.2f}" if isinstance(value, (float, np.floating)) else str(value)
    s = str(value).strip()
    return "—" if not s else s


def analysis_sort_key(value: Any) -> tuple:
    """Clé de tri numérique/texte; les valeurs absentes sont traitées à part par l'UI."""
    text = str(value).strip().replace("%", "")
    if text in {"—", "-", "nan", "NaN", ""}:
        return (2, 0.0)
    try:
        number = float(text)
        return (0, number)
    except (TypeError, ValueError):
        return (1, text.casefold())


def truncate_reason(text: str, limit: int = 92) -> str:
    """Tronque une raison au mot le plus proche, avec une ellipse visible."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = text[:max(1, limit - 1)].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def export_analysis_csv(result: AnalysisResult, path: str,
                        choices: Optional[Dict[str, Any]] = None) -> str:
    """Exporte les raisons, preuves et explications de chaque étape applicables."""
    build_analysis_table(result, choices).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def validate_column_choice(result: AnalysisResult, column: str, action: str) -> str:
    """Valide une action selon les alternatives proposées pour la colonne."""
    row = next((r for r in result.columns if r["colonne"] == column), None)
    if row is None:
        raise ValueError(f"Colonne inconnue : {column}")
    if action != row["action"] and action not in row["alternatives"]:
        raise ValueError(f"Action {action!r} incompatible avec la colonne {column!r}.")
    return Action(action).value


def validate_step_choice(result: AnalysisResult, column: str, step: str, action: str) -> str:
    """Valide un choix associé à une étape de transformation."""
    row = next((r for r in result.columns if r["colonne"] == column), None)
    if row is None or step not in row.get("steps", {}):
        raise ValueError(f"Étape ou colonne inconnue : {column}/{step}")
    allowed = row["steps"][step]["choices"]
    if action not in allowed:
        raise ValueError(f"Action {action!r} invalide pour l'étape {step!r} de {column!r}; choix valides : {allowed}.")
    return action


def parse_step_parameters(step: str, action: str, raw_value: str, column_type: str) -> Optional[Any]:
    """Valide et convertit les champs de paramètres de l'éditeur d'étapes."""
    value = (raw_value or "").strip()
    if step == "clean_missing_tokens" and action == "SELECTED_TO_NAN":
        selected = [part.strip() for part in value.split(",") if part.strip()]
        if not selected:
            raise ValueError("Saisissez au moins une valeur séparée par des virgules.")
        return {"values": selected}
    if step == "missing" and action == "CONSTANT":
        if column_type.startswith("NUMERIC"):
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError("La constante d'imputation numérique doit être un nombre.") from exc
        if not value:
            raise ValueError("La constante d'imputation ne peut pas être vide.")
        return value
    if step == "outliers" and action == "WINSORIZE":
        try:
            low, high = [float(part.strip()) for part in value.split(",", maxsplit=1)]
        except (ValueError, TypeError) as exc:
            raise ValueError("Saisissez deux percentiles entre 0 et 1, par exemple 0.01, 0.99.") from exc
        if not 0 <= low < high <= 1:
            raise ValueError("Les percentiles doivent vérifier 0 ≤ inférieur < supérieur ≤ 1.")
        return {"lower": low, "upper": high}
    if step == "encoding" and action == "ORDINAL":
        order = [part.strip() for part in value.split(",") if part.strip()]
        if len(order) < 2 or len(set(order)) != len(order):
            raise ValueError("L'ordre ordinal doit contenir au moins deux valeurs distinctes, séparées par des virgules.")
        return {"order": order}
    if step == "encoding" and action == "ONEHOT" and value:
        try:
            threshold = float(value)
        except ValueError as exc:
            raise ValueError("Le seuil de catégories rares doit être un nombre entre 0 et 1.") from exc
        if not 0 <= threshold <= 1:
            raise ValueError("Le seuil de catégories rares doit être compris entre 0 et 1.")
        return {"rare_threshold": threshold}
    if step == "text" and action == "TFIDF_SVD":
        try:
            components = int(value)
        except ValueError as exc:
            raise ValueError("Le nombre de composantes SVD doit être un entier positif ou nul.") from exc
        if components < 0:
            raise ValueError("Le nombre de composantes SVD ne peut pas être négatif.")
        return {"components": components}
    return None


def choices_to_overrides(result: AnalysisResult, choices: Dict[str, str]) -> Dict[str, ColumnOverride]:
    """Convertit des choix validés en configuration persistable."""
    out = {}
    for col, action in choices.items():
        if isinstance(action, dict):
            out[col] = ColumnOverride({step: (validate_step_choice(result, col, step, value[0]), value[1])
                                       if isinstance(value, (tuple, list)) and len(value) == 2 else
                                       validate_step_choice(result, col, step, value)
                                       for step, value in action.items()})
        else:
            out[col] = ColumnOverride(validate_column_choice(result, col, action))
    return out


def build_preprocess_summary(result: AnalysisResult, choices: Dict[str, Any]) -> Dict[str, Any]:
    """Prépare le résumé de confirmation, sans dépendre de Tkinter."""
    rows = {row["colonne"]: row for row in result.columns}
    user_columns = []
    dropped = []
    warnings = []
    for col, row in rows.items():
        selected = choices.get(col, {})
        if isinstance(selected, dict):
            modified = {step: action for step, action in selected.items()
                        if action != row.get("steps", {}).get(step, {}).get("action")}
            if modified:
                user_columns.append(col)
            if selected.get("type") == "DROP" or row.get("steps", {}).get("type", {}).get("action") == "DROP":
                dropped.append(col)
            if modified.get("outliers") == "NONE" and float(row.get("kurtosis", 0) or 0) > 3:
                warnings.append(f"{col}: outliers désactivés malgré kurtosis élevée ({row['kurtosis']:.2f}).")
            if modified.get("missing") == "MEAN" and abs(float(row.get("skew", 0) or 0)) > 1:
                warnings.append(f"{col}: imputation moyenne sur une distribution asymétrique (skew={row['skew']:.2f}).")
        elif selected != row["action"]:
            user_columns.append(col)
            if selected == "DROP":
                dropped.append(col)
        elif row["action"] == "DROP":
            dropped.append(col)
        if row.get("score_fuite", 0) and pd.notna(row.get("score_fuite")) and float(row["score_fuite"]) >= .95:
            warnings.append(f"{col}: score de fuite de cible élevé ({float(row['score_fuite']):.3f}).")
    return {"user_columns": sorted(set(user_columns)), "dropped_columns": sorted(set(dropped)),
            "warnings": list(dict.fromkeys(warnings))}


def export_user_choices(result: AnalysisResult, choices: Dict[str, str], path: str) -> None:
    """Exporte les choix avec les noms de colonnes pour rejouer l'analyse."""
    valid = {r["colonne"] for r in result.columns}
    unknown = sorted(set(choices) - valid)
    if unknown:
        raise ValueError("Colonnes absentes de l'analyse : " + ", ".join(unknown))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "choices": choices}, handle, ensure_ascii=False, indent=2)


def import_user_choices(result: AnalysisResult, path: str) -> Dict[str, str]:
    """Importe les choix; signale les colonnes manquantes ou devenues absentes."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    choices = payload.get("choices", payload)
    valid = {r["colonne"]: r for r in result.columns}
    missing = sorted(set(choices) - set(valid))
    if missing:
        raise ValueError("Colonnes du fichier de choix absentes du dataset : " + ", ".join(missing))
    return {col: ({step: (validate_step_choice(result, col, step, value[0]), value[1])
                            if isinstance(value, (tuple, list)) and len(value) == 2 else
                            validate_step_choice(result, col, step, value)
                   for step, value in action.items()}
                  if isinstance(action, dict) else validate_column_choice(result, col, action))
            for col, action in choices.items()}


@dataclass
class RunResult:
    """Résultat d'un prétraitement."""
    prep: AutoPreprocessor
    target: Optional[str]
    X_train: pd.DataFrame
    y_train: Optional[np.ndarray]
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[np.ndarray] = None
    y_train_original: Optional[pd.Series] = None
    y_test_original: Optional[pd.Series] = None
    notes: List[str] = field(default_factory=list)
    n_rows_in: int = 0
    n_cols_in: int = 0


def _looks_like_classification(y: pd.Series, task: str) -> bool:
    if task == "classification":
        return True
    if task == "regression":
        return False
    return (not pd.api.types.is_numeric_dtype(y)) or y.nunique(dropna=True) <= 15


def _original_target(df: pd.DataFrame, X: pd.DataFrame, target: str) -> Optional[pd.Series]:
    """Cible d'origine alignée sur les lignes de X (si l'index est retrouvable)."""
    try:
        if X.index.isin(df.index).all():
            return df.loc[X.index, target]
    except Exception:  # noqa: BLE001 - alignement facultatif
        pass
    return None


def run_preprocessing(df: pd.DataFrame, target: Optional[str], opts: Options) -> RunResult:
    """Ajuste AutoPreprocessor sur le train uniquement, puis transforme le test (aucune fuite)."""
    notes: List[str] = []
    config = build_config(opts)
    prep = AutoPreprocessor(config=config, task=opts.task)
    n_rows_in, n_cols_in = df.shape

    # Cas non supervisé : pas de cible
    if not target:
        X = prep.fit_transform(df)
        notes.append("Mode sans cible : aucun découpage train/test, encodages non supervisés.")
        return RunResult(prep, None, X, None, notes=notes, n_rows_in=n_rows_in, n_cols_in=n_cols_in)

    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' est absente du fichier.")

    n_missing = int(df[target].isna().sum())
    data = df.dropna(subset=[target]).copy()
    if n_missing:
        notes.append(f"{n_missing} ligne(s) sans valeur de cible supprimée(s) avant le traitement.")
    if len(data) < 10:
        raise ValueError("Moins de 10 lignes avec une cible renseignée : traitement impossible.")

    train_df, test_df = data, None
    if opts.test_size_pct > 0:
        stratify = data[target] if _looks_like_classification(data[target], opts.task) else None
        try:
            train_df, test_df = train_test_split(
                data, test_size=opts.test_size_pct / 100.0, random_state=opts.random_state, stratify=stratify
            )
        except ValueError:
            train_df, test_df = train_test_split(
                data, test_size=opts.test_size_pct / 100.0, random_state=opts.random_state
            )
            notes.append("Découpage non stratifié (une classe est trop rare pour être stratifiée).")

    X_train, y_train = prep.fit_transform_xy(train_df, target=target)
    y_train_original = None
    if prep.target_stats_.get("transformed", False):
        y_train_original = _original_target(train_df, X_train, target)
        notes.append("La cible a été transformée (asymétrie) : utilisez inverse_transform_target() pour revenir à l'échelle d'origine.")

    X_test = y_test = y_test_original = None
    if test_df is not None:
        X_test, y_test = prep.transform_xy(test_df, target=target)
        if prep.task_resolved_ == "classification" and np.any(np.asarray(y_test) == -1):
            notes.append("Le test contient des classes absentes du train : elles sont codées -1.")
        if prep.target_stats_.get("transformed", False):
            y_test_original = _original_target(test_df, X_test, target)

    return RunResult(
        prep, target, X_train, np.asarray(y_train), X_test,
        None if y_test is None else np.asarray(y_test),
        y_train_original, y_test_original, notes, n_rows_in, n_cols_in,
    )


def apply_to_new_data(prep: AutoPreprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """Applique un préprocesseur déjà ajusté à de nouvelles données (la colonne cible est ignorée si présente)."""
    target = getattr(prep, "target_col_", None)
    if target and target in df.columns:
        X, y = prep.transform_xy(df, target=target)
        out = X.copy()
        out[target if target not in out.columns else f"{target}_cible"] = np.asarray(y)
        return out
    return prep.transform(df)


def report_text(result: RunResult) -> str:
    """Rapport lisible : résumé, décisions colonne par colonne, fuites, classes de la cible."""
    prep = result.prep
    rep = prep.report()
    lines = [prep.summary(), ""]
    lines.append(f"Fichier : {result.n_rows_in} lignes x {result.n_cols_in} colonnes")
    lines.append(f"Train   : {result.X_train.shape[0]} lignes x {result.X_train.shape[1]} features numériques")
    if result.X_test is not None:
        lines.append(f"Test    : {result.X_test.shape[0]} lignes x {result.X_test.shape[1]} features numériques")
    if result.target:
        lines.append(f"Cible   : {result.target} ({rep['tache_detectee']})")
    classes = rep["analyse_cible"].get("classes")
    if classes:
        mapping = ", ".join(f"{i} = {c}" for i, c in enumerate(classes))
        lines.append(f"Classes encodées : {mapping}")
    for note in result.notes:
        lines.append(f"Note : {note}")
    scores = rep.get("scores_fuite_par_colonne") or {}
    if scores:
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("Test de fuite (score d'une colonne seule, 5 plus hauts) : "
                     + ", ".join(f"{c}={s:.3f}" for c, s in top))
    lines += ["", "Détail des transformations par colonne :",
              rep["colonnes_detaillees"].to_string(index=False)]
    return "\n".join(lines)


def _with_target(X: pd.DataFrame, y: Optional[np.ndarray], target: Optional[str],
                 y_original: Optional[pd.Series]) -> pd.DataFrame:
    out = X.copy()
    if y is not None and target:
        name = target if target not in out.columns else f"{target}_cible"
        out[name] = np.asarray(y)
        if y_original is not None:
            out[f"{target}_original"] = np.asarray(y_original)
    return out


def export_result(result: RunResult, folder: str) -> List[str]:
    """Écrit les CSV transformés, le rapport, le détail par colonne et le préprocesseur. Retourne les chemins."""
    os.makedirs(folder, exist_ok=True)
    written: List[str] = []

    train_path = os.path.join(folder, "train_transforme.csv")
    _with_target(result.X_train, result.y_train, result.target, result.y_train_original).to_csv(
        train_path, index=False, encoding="utf-8-sig")
    written.append(train_path)

    if result.X_test is not None:
        test_path = os.path.join(folder, "test_transforme.csv")
        _with_target(result.X_test, result.y_test, result.target, result.y_test_original).to_csv(
            test_path, index=False, encoding="utf-8-sig")
        written.append(test_path)

    detail_path = os.path.join(folder, "detail_colonnes.csv")
    result.prep.report()["colonnes_detaillees"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    written.append(detail_path)

    report_path = os.path.join(folder, "rapport_pretraitement.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"Généré le {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n{report_text(result)}\n")
    written.append(report_path)

    prep_path = os.path.join(folder, "preprocesseur.joblib")
    result.prep.save(prep_path)
    written.append(prep_path)
    return written


# =============================================================================
# 2. INTERFACE TKINTER
# =============================================================================

def _import_tk():
    """Import différé : la logique ci-dessus reste utilisable (et testable) sans Tkinter."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    return tk, ttk, filedialog, messagebox


class App:
    """Fenêtre principale."""

    PREVIEW_ROWS = 200
    PREVIEW_COLS = 60

    def __init__(self) -> None:
        tk, ttk, filedialog, messagebox = _import_tk()
        self.tk, self.ttk, self.filedialog, self.messagebox = tk, ttk, filedialog, messagebox

        self.root = tk.Tk()
        self.root.title("AutoPreprocessor : nettoyage et transformation automatiques")
        self.root.geometry("1150x720")
        self.root.minsize(900, 600)

        self.df: Optional[pd.DataFrame] = None
        self.path: Optional[str] = None
        self.result: Optional[RunResult] = None
        self.analysis: Optional[AnalysisResult] = None
        self.choices: Dict[str, str] = {}
        self.prep: Optional[AutoPreprocessor] = None   # préprocesseur actif (ajusté ou chargé)
        self._queue: "queue.Queue[tuple]" = queue.Queue()

        self._build_variables()
        self._build_layout()

    # ------------------------------------------------------------------ variables
    def _build_variables(self) -> None:
        tk = self.tk
        self.v_file = tk.StringVar(value="Aucun fichier chargé")
        self.v_target = tk.StringVar(value=NO_TARGET_LABEL)
        self.v_task = tk.StringVar(value="auto")
        self.v_test = tk.StringVar(value="20")
        self.v_scaler = tk.StringVar(value="standard")
        self.v_outlier = tk.StringVar(value="winsorize")
        self.v_card = tk.StringVar(value="10")
        self.v_missing = tk.StringVar(value="0.90")
        self.v_corr = tk.StringVar(value="0.95")
        self.v_rare = tk.StringVar(value="0.01")
        self.v_skew = tk.StringVar(value="1.0")
        self.v_transform_target = tk.BooleanVar(value=True)
        self.v_drop_leaky = tk.BooleanVar(value=False)
        self.v_discrete_cat = tk.BooleanVar(value=False)
        self.v_status = tk.StringVar(value="Ouvrez un fichier pour commencer.")

    # ------------------------------------------------------------------ mise en page
    def _build_layout(self) -> None:
        ttk = self.ttk
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Ouvrir un fichier (CSV, Excel, JSON...)", command=self._open_file).pack(side="left")
        ttk.Label(top, textvariable=self.v_file).pack(side="left", padx=10)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_params = ttk.Frame(self.notebook, padding=10)
        self.tab_report = ttk.Frame(self.notebook)
        self.tab_result = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="1. Données")
        self.notebook.add(self.tab_analysis, text="2. Analyse")
        self.notebook.add(self.tab_params, text="3. Paramètres")
        self.notebook.add(self.tab_report, text="4. Rapport")
        self.notebook.add(self.tab_result, text="5. Résultat")

        self._build_tab_data()
        self._build_tab_analysis()
        self._build_tab_params()
        self._build_tab_report()
        self._build_tab_result()

        bottom = ttk.Frame(self.root, padding=(8, 2))
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=160)
        self.progress.pack(side="left")
        ttk.Label(bottom, textvariable=self.v_status).pack(side="left", padx=10)

    def _make_tree(self, parent):
        ttk = self.ttk
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, show="headings")
        ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        return tree

    def _build_tab_data(self) -> None:
        self.lbl_info = self.ttk.Label(self.tab_data, text="", padding=6)
        self.lbl_info.pack(fill="x")
        self.tree_data = self._make_tree(self.tab_data)

    def _build_tab_analysis(self) -> None:
        ttk = self.ttk
        bar = ttk.Frame(self.tab_analysis, padding=6); bar.pack(fill="x")
        ttk.Button(bar, text="Analyser le dataset", command=self._start_analysis).pack(side="left")
        ttk.Button(bar, text="Tout accepter", command=self._accept_all).pack(side="left", padx=4)
        ttk.Button(bar, text="Tout réinitialiser", command=self._reset_all).pack(side="left", padx=4)
        ttk.Button(bar, text="Importer choix JSON", command=self._import_analysis_choices).pack(side="left", padx=4)
        ttk.Button(bar, text="Exporter analyse et choix", command=self._export_analysis_choices).pack(side="left", padx=4)
        ttk.Label(bar, text="Cible :").pack(side="left", padx=(8, 2))
        self.cb_analysis_target = ttk.Combobox(bar, textvariable=self.v_target, state="readonly", width=24,
                                               values=[NO_TARGET_LABEL])
        self.cb_analysis_target.pack(side="left")
        ttk.Label(bar, text="Tâche :").pack(side="left", padx=(8, 2))
        self.cb_analysis_task = ttk.Combobox(bar, textvariable=self.v_task, state="readonly", width=14,
                                             values=["auto", "classification", "regression"])
        self.cb_analysis_task.pack(side="left")
        self.v_analysis_warning = self.tk.StringVar(value="analyse sans cible : relations, fuites et encodage supervisé indisponibles")
        ttk.Label(self.tab_analysis, textvariable=self.v_analysis_warning, foreground="#9a5700").pack(fill="x", padx=8)
        self.v_target.trace_add("write", lambda *_: self._update_analysis_target_warning())
        self.analysis_paned = ttk.Panedwindow(self.tab_analysis, orient="vertical")
        self.analysis_paned.pack(fill="both", expand=True, padx=5, pady=3)
        table_panel = ttk.Frame(self.analysis_paned)
        detail_panel = ttk.Frame(self.analysis_paned)
        self.analysis_paned.add(table_panel, weight=3)
        self.analysis_paned.add(detail_panel, weight=5)
        self.tree_analysis = self._make_tree(table_panel)
        self.tree_analysis.configure(selectmode="extended")
        self.tree_analysis.bind("<<TreeviewSelect>>", self._analysis_selection)
        self.tree_analysis.bind("<Motion>", self._analysis_reason_hover)
        self.tree_analysis.bind("<Leave>", self._hide_reason_tooltip)
        self.detail_header = ttk.Label(detail_panel,
            text="Sélectionnez une colonne pour voir ses statistiques et les explications.",
            wraplength=1180, justify="left", padding=6)
        self.detail_header.pack(fill="x")
        self.step_editor = ttk.LabelFrame(detail_panel, text="Étapes applicables : recommandation, preuve, explication et choix")
        self.step_editor.pack(fill="both", expand=True, padx=5, pady=3)
        self.detail_canvas = self.tk.Canvas(self.step_editor, highlightthickness=0)
        detail_y = ttk.Scrollbar(self.step_editor, orient="vertical", command=self.detail_canvas.yview)
        detail_x = ttk.Scrollbar(self.step_editor, orient="horizontal", command=self.detail_canvas.xview)
        self.detail_canvas.configure(yscrollcommand=detail_y.set, xscrollcommand=detail_x.set)
        detail_y.pack(side="right", fill="y")
        detail_x.pack(side="bottom", fill="x")
        self.detail_canvas.pack(side="left", fill="both", expand=True)
        self.detail_rows_frame = ttk.Frame(self.detail_canvas)
        self.detail_canvas_window = self.detail_canvas.create_window((0, 0), window=self.detail_rows_frame, anchor="nw")
        self.detail_rows_frame.bind("<Configure>", lambda _e: self.detail_canvas.configure(
            scrollregion=self.detail_canvas.bbox("all")))
        edit = ttk.Frame(self.tab_analysis, padding=6); edit.pack(fill="x")
        self.v_analysis_action = self.tk.StringVar(value="")
        self.cb_analysis_action = ttk.Combobox(edit, textvariable=self.v_analysis_action, state="readonly", width=28)
        self.cb_analysis_action.pack(side="left")
        ttk.Button(edit, text="Appliquer aux colonnes sélectionnées", command=self._apply_analysis_choice).pack(side="left", padx=5)
        ttk.Button(edit, text="Réinitialiser la colonne", command=self._reset_analysis_selection).pack(side="left")

    def _analysis_selection(self, _event=None) -> None:
        if self.analysis is None:
            return
        selected = self.tree_analysis.selection()
        if not selected:
            return
        col = self.tree_analysis.item(selected[0], "values")[0]
        row = next((r for r in self.analysis.columns if r["colonne"] == col), None)
        if row:
            self._render_analysis_detail(col)
            if row.get("est_cible"):
                self.cb_analysis_action["values"] = []
                self.v_analysis_action.set("")
                return
            self.cb_analysis_action["values"] = list(dict.fromkeys([row["action"], *row["alternatives"]]))
            current_global = self.choices.get(col, row["action"])
            self.v_analysis_action.set(current_global if not isinstance(current_global, dict) else row["action"])
    def _render_analysis_detail(self, column: str) -> None:
        if self.analysis is None:
            return
        detail = build_analysis_detail(self.analysis, column, self.choices)
        self.detail_header.configure(text=detail["header"])
        for child in self.detail_rows_frame.winfo_children():
            child.destroy()
        if not detail["rows"]:
            self.ttk.Label(self.detail_rows_frame, text="Aucune étape de transformation applicable.",
                           padding=8).grid(row=0, column=0, sticky="w")
            return
        headings = [("Étape", 105), ("Recommandation", 170), ("Preuve chiffrée (mesure vs seuil)", 235),
                    ("Explication", 430), ("Choix / paramètres / aide", 340)]
        for col_index, (heading, width) in enumerate(headings):
            self.ttk.Label(self.detail_rows_frame, text=heading, font=("TkDefaultFont", 9, "bold"),
                           wraplength=width, padding=4).grid(row=0, column=col_index, sticky="nsew")
        source_row = next(row for row in self.analysis.columns if row["colonne"] == column)
        for index, item in enumerate(detail["rows"], start=1):
            step = item["etape"]
            rec_text = item["recommandation"]
            if item["modifie"]:
                rec_text = f"✎ modifié — recommandation : {rec_text}"
            self.ttk.Label(self.detail_rows_frame, text=step, wraplength=100, padding=4).grid(
                row=index, column=0, sticky="nw")
            self.ttk.Label(self.detail_rows_frame, text=rec_text, wraplength=165, padding=4).grid(
                row=index, column=1, sticky="nw")
            self.ttk.Label(self.detail_rows_frame, text=item["preuve"], wraplength=230,
                           justify="left", padding=4).grid(row=index, column=2, sticky="nw")
            self.ttk.Label(self.detail_rows_frame, text=item["explication"], wraplength=420,
                           justify="left", padding=4).grid(row=index, column=3, sticky="nw")
            detail_step = source_row["steps"][step]
            saved = self.choices.get(column, {})
            saved_value = saved.get(step, item["recommandation"]) if isinstance(saved, dict) else item["recommandation"]
            saved_action = saved_value[0] if isinstance(saved_value, (tuple, list)) and len(saved_value) == 2 else str(saved_value)
            saved_params = saved_value[1] if isinstance(saved_value, (tuple, list)) and len(saved_value) == 2 else None
            choice_frame = self.ttk.Frame(self.detail_rows_frame, padding=2)
            choice_frame.grid(row=index, column=4, sticky="nsew")
            variable = self.tk.StringVar(value=saved_action)
            combo = self.ttk.Combobox(choice_frame, textvariable=variable, values=detail_step["choices"],
                                      state="readonly", width=19)
            combo.grid(row=0, column=0, sticky="w")
            default_param = self._format_step_parameter(step, saved_action, saved_params, source_row)
            param_var = self.tk.StringVar(value=default_param)
            param_entry = self.ttk.Entry(choice_frame, textvariable=param_var, width=18)
            param_entry.configure(state="normal" if default_param else "disabled")
            param_entry.grid(row=0, column=1, sticky="w", padx=3)
            help_var = self.tk.StringVar(value=recommendation_help(
                step, saved_action, detail_step.get("measure", ""), detail_step.get("threshold", "")))
            help_label = self.ttk.Label(choice_frame, textvariable=help_var, wraplength=320,
                                        justify="left", foreground="#555")
            help_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
            combo.bind("<<ComboboxSelected>>", lambda _ev, c=column, st=step, v=variable, p=param_var,
                       e=param_entry, r=source_row: self._step_action_changed(c, st, v, p, e, r))
            combo.bind("<<ComboboxSelected>>", lambda _ev, v=variable, h=help_var, st=step,
                       d=detail_step: h.set(recommendation_help(st, v.get(), d.get("measure", ""),
                                                              d.get("threshold", ""))), add="+")
            param_entry.bind("<FocusOut>", lambda _ev, c=column, st=step, v=variable, p=param_var,
                             t=source_row["type"]: self._apply_step_value(c, st, v.get(), p.get(), t))
        self.detail_canvas.configure(scrollregion=self.detail_canvas.bbox("all"))

    def _analysis_reason_hover(self, event) -> None:
        if "raison" not in self.tree_analysis["columns"]:
            return
        row_id = self.tree_analysis.identify_row(event.y)
        column_id = self.tree_analysis.identify_column(event.x)
        try:
            col_name = self.tree_analysis["columns"][int(column_id.lstrip("#")) - 1]
        except (ValueError, IndexError):
            return
        if not row_id or col_name != "raison":
            self._hide_reason_tooltip()
            return
        column_name = self.tree_analysis.set(row_id, "colonne")
        row = next((item for item in self.analysis.columns if item["colonne"] == column_name), None) if self.analysis else None
        if row is None:
            return
        text = analysis_recommendation_summary(row)["raison"]
        if getattr(self, "_reason_tooltip_text", None) == text:
            return
        self._hide_reason_tooltip()
        self._reason_tooltip_text = text
        self._reason_tooltip = self.tk.Toplevel(self.root)
        self._reason_tooltip.wm_overrideredirect(True)
        self._reason_tooltip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
        self.ttk.Label(self._reason_tooltip, text=text, wraplength=520, justify="left",
                       padding=6, relief="solid").pack()

    def _hide_reason_tooltip(self, _event=None) -> None:
        tooltip = getattr(self, "_reason_tooltip", None)
        if tooltip is not None:
            tooltip.destroy()
            self._reason_tooltip = None
        self._reason_tooltip_text = None

    def _step_action_changed(self, column: str, step: str, action_var, param_var, param_entry, row) -> None:
        default = self._format_step_parameter(step, action_var.get(), None, row)
        param_var.set(default)
        param_entry.configure(state="normal" if default else "disabled")
        self._apply_step_value(column, step, action_var.get(), default, row["type"])

    @staticmethod
    def _format_step_parameter(step: str, action: str, params: Any, row: Dict[str, Any]) -> str:
        if step == "clean_missing_tokens" and action == "SELECTED_TO_NAN":
            values = params.get("values", []) if isinstance(params, dict) else []
            return ", ".join(map(str, values)) or "?, N/A, null"
        if step == "missing" and action == "CONSTANT":
            return str(params if params is not None else ("0" if str(row["type"]).startswith("NUMERIC") else "__CUSTOM__"))
        if step == "outliers" and action == "WINSORIZE":
            p = params if isinstance(params, dict) else {}
            return f"{p.get('lower', 0.01)}, {p.get('upper', 0.99)}"
        if step == "encoding" and action == "ORDINAL":
            p = params if isinstance(params, dict) else {}
            values = p.get("order", row.get("natural_order") or row.get("exemples", []))
            return ", ".join(map(str, values))
        if step == "encoding" and action == "ONEHOT":
            p = params if isinstance(params, dict) else {}
            return str(p.get("rare_threshold", 0.01))
        if step == "text" and action == "TFIDF_SVD":
            p = params if isinstance(params, dict) else {}
            return str(p.get("components", 4))
        return ""

    def _apply_step_value(self, column: str, step: str, action: str, raw_params: str = "", column_type: str = "") -> None:
        if self.analysis is None:
            return
        try:
            action = validate_step_choice(self.analysis, column, step, action)
            params = parse_step_parameters(step, action, raw_params, column_type)
        except ValueError as exc:
            self.messagebox.showerror("Choix invalide", str(exc)); return
        selected = dict(self.choices.get(column, {})) if isinstance(self.choices.get(column, {}), dict) else {}
        recommended = next(r["steps"][step]["action"] for r in self.analysis.columns if r["colonne"] == column)
        if action == recommended and params is None:
            selected.pop(step, None)
        else:
            selected[step] = (action, params) if params is not None else action
        if selected:
            self.choices[column] = selected
        else:
            self.choices.pop(column, None)
        self._refresh_analysis_tree()
        self._render_analysis_detail(column)

    def _refresh_analysis_tree(self) -> None:
        if self.analysis is None:
            return
        selected_columns = [self.tree_analysis.set(item, "colonne") for item in self.tree_analysis.selection()]
        view = build_analysis_table(self.analysis, self.choices)
        self.tree_analysis.delete(*self.tree_analysis.get_children())
        columns = list(self.tree_analysis["columns"])
        for _, row in view.iterrows():
            values = [truncate_reason(row.get(c, "")) if c == "raison" else format_analysis_value(c, row.get(c, ""))
                      for c in columns]
            tag = "cible" if bool(row.get("est_cible", False)) else ()
            item = self.tree_analysis.insert("", "end", values=values, tags=tag)
            if row["colonne"] in selected_columns:
                self.tree_analysis.selection_add(item)

    def _accept_all(self) -> None:
        if self.analysis:
            self.choices = {r["colonne"]: {step: info["action"] for step, info in r.get("steps", {}).items()}
                            for r in self.analysis.columns if not r.get("est_cible")}
            self._refresh_analysis_tree()

    def _reset_all(self) -> None:
        self.choices.clear(); self._refresh_analysis_tree()

    def _reset_analysis_selection(self) -> None:
        if self.analysis:
            for item in self.tree_analysis.selection():
                col = self.tree_analysis.item(item, "values")[0]
                self.choices.pop(col, None)
            self._refresh_analysis_tree()

    def _apply_analysis_choice(self) -> None:
        if self.analysis is None:
            return
        for item in self.tree_analysis.selection():
            col = self.tree_analysis.item(item, "values")[0]
            try:
                self.choices[col] = validate_column_choice(self.analysis, col, self.v_analysis_action.get())
            except ValueError as exc:
                self.messagebox.showerror("Choix invalide", str(exc)); return
        self._refresh_analysis_tree()

    def _import_analysis_choices(self) -> None:
        if self.analysis is None:
            self.messagebox.showinfo("Analyse requise", "Analysez d'abord le dataset."); return
        path = self.filedialog.askopenfilename(title="Importer les choix", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.choices.update(import_user_choices(self.analysis, path))
            self._refresh_analysis_tree()
        except Exception as exc:
            self.messagebox.showerror("Import impossible", str(exc))

    def _export_analysis_choices(self) -> None:
        if self.analysis is None:
            self.messagebox.showinfo("Analyse requise", "Analysez d'abord le dataset."); return
        folder = self.filedialog.askdirectory(title="Dossier d'export")
        if not folder:
            return
        try:
            export_analysis_csv(self.analysis, os.path.join(folder, "analyse_colonnes.csv"), self.choices)
            export_user_choices(self.analysis, self.choices, os.path.join(folder, "choix_utilisateur.json"))
            self.messagebox.showinfo("Export terminé", "analyse_colonnes.csv et choix_utilisateur.json exportés.")
        except Exception as exc:
            self.messagebox.showerror("Export impossible", str(exc))

    def _start_analysis(self) -> None:
        if self.df is None:
            self.messagebox.showinfo("Aucun fichier", "Ouvrez d'abord un fichier."); return
        target = self.v_target.get(); target = None if target == NO_TARGET_LABEL else target
        df = self.df.copy(deep=True)
        options = self._read_options()
        config = build_config(options)
        self.progress.start(12); self._set_status("Analyse en cours...")
        def work():
            try: self._queue.put(("analysis_ok", AutoPreprocessor(config, task=options.task).analyze(df, target)))
            except Exception as exc: self._queue.put(("analysis_err", f"{exc}\n\n{traceback.format_exc()}"))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._poll_analysis)

    def _poll_analysis(self) -> None:
        try: kind, payload = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll_analysis); return
        self.progress.stop()
        if kind == "analysis_err":
            self.messagebox.showerror("Analyse impossible", payload.split("\n\n")[0]); self._set_status(payload); return
        self.analysis = payload
        self.tree_analysis.delete(*self.tree_analysis.get_children())
        view = build_analysis_table(payload, self.choices)
        columns = ["colonne", "type", "manquants_pct", "uniques", "skew", "outliers_pct",
                   "relation_cible", "choix_final", "raison", "état_choix", "confiance"]
        columns = [c for c in columns if c in view.columns]
        self.tree_analysis["columns"] = columns
        headers = {
            "colonne": "colonne",
            "type": "type",
            "manquants_pct": "manquants (%)",
            "uniques": "uniques",
            "skew": "skew",
            "outliers_pct": "outliers (%)",
            "relation_cible": "relation cible",
            "choix_final": "choix final",
            "raison": "raison",
            "état_choix": "état du choix",
            "confiance": "confiance",
        }
        col_min_defaults = {
            "colonne": 140,
            "type": 190,
            "manquants_pct": 115,
            "uniques": 85,
            "skew": 75,
            "outliers_pct": 110,
            "relation_cible": 115,
            "choix_final": 170,
            "raison": 320,
            "état_choix": 115,
            "confiance": 95,
        }
        for col in columns:
            header = headers.get(col, col)
            self.tree_analysis.heading(col, text=header, command=lambda c=col: self._sort_analysis(c, False))
            display_values = [truncate_reason(value) if col == "raison" else format_analysis_value(col, value)
                              for value in view[col].tolist()]
            content_width = max((len(str(value)) for value in display_values), default=0)
            min_w = col_min_defaults.get(col, 95)
            calc_w = max(len(header), content_width) * 9 + 30
            width = max(min_w, min(500, calc_w))
            self.tree_analysis.column(col, width=width, minwidth=min_w, stretch=False)
        self.tree_analysis.tag_configure("risque", foreground="#b00020")
        self.tree_analysis.tag_configure("cible", foreground="#777777")
        for _, row in view.iterrows():
            is_target = bool(row.get("est_cible", False))
            risk = (str(row.get("type", "")).startswith("DROP_ID") or
                    float(row.get("manquants_pct", 0) or 0) > 50 or
                    (pd.notna(row.get("score_fuite")) and float(row.get("score_fuite")) >= .95))
            values = [truncate_reason(row.get(c, "")) if c == "raison" else format_analysis_value(c, row.get(c, ""))
                       for c in columns]
            self.tree_analysis.insert("", "end", values=values,
                                      tags=("cible",) if is_target else ("risque",) if risk else ())
        feature_item = next((item for item in self.tree_analysis.get_children("")
                             if not bool(next((r for r in payload.columns
                                               if r["colonne"] == self.tree_analysis.set(item, "colonne")), {}).get("est_cible"))), None)
        if feature_item is None and self.tree_analysis.get_children(""):
            feature_item = self.tree_analysis.get_children("")[0]
        if feature_item:
            self.tree_analysis.selection_set(feature_item)
            self.tree_analysis.focus(feature_item)
            self._analysis_selection()
        self._set_status(f"Analyse terminée : {len(view)} colonnes.")

    def _sort_analysis(self, column: str, reverse: bool) -> None:
        items = [(self.tree_analysis.set(i, column), i) for i in self.tree_analysis.get_children("")]
        present = [(v, item) for v, item in items if str(v).strip() and str(v).strip() not in {"—", "-", "nan", "NaN"}]
        blanks = [(v, item) for v, item in items if not str(v).strip() or str(v).strip() in {"—", "-", "nan", "NaN"}]
        present.sort(key=lambda x: analysis_sort_key(x[0]), reverse=reverse)
        for pos, (_, item) in enumerate(present + blanks):
            self.tree_analysis.move(item, "", pos)
        self.tree_analysis.heading(column, command=lambda: self._sort_analysis(column, not reverse))

    def _update_analysis_target_warning(self) -> None:
        if not hasattr(self, "v_analysis_warning"):
            return
        if self.v_target.get() == NO_TARGET_LABEL:
            self.v_analysis_warning.set("analyse sans cible : relations, fuites et encodage supervisé indisponibles")
        else:
            self.v_analysis_warning.set("Cible sélectionnée : relations et alertes de fuite estimées sur les données chargées.")

    def _build_tab_params(self) -> None:
        ttk = self.ttk
        box = ttk.LabelFrame(self.tab_params, text="Cible et découpage", padding=8)
        box.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        ttk.Label(box, text="Colonne cible :").grid(row=0, column=0, sticky="w")
        self.cb_target = ttk.Combobox(box, textvariable=self.v_target, state="readonly", width=32,
                                      values=[NO_TARGET_LABEL])
        self.cb_target.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(box, text="Type de tâche :").grid(row=1, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.v_task, state="readonly", width=16,
                     values=["auto", "classification", "regression"]).grid(row=1, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(box, text="Part du test (%), 0 = aucun :").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(box, textvariable=self.v_test, from_=0, to=50, increment=5, width=6).grid(
            row=2, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(box, text="Le préprocesseur est ajusté sur le train seulement (aucune fuite vers le test).",
                  foreground="#555").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        opt = ttk.LabelFrame(self.tab_params, text="Nettoyage et transformations", padding=8)
        opt.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        rows = [
            ("Mise à l'échelle :", ttk.Combobox(opt, textvariable=self.v_scaler, state="readonly", width=12,
                                                 values=["standard", "robust", "minmax", "none"])),
            ("Valeurs extrêmes :", ttk.Combobox(opt, textvariable=self.v_outlier, state="readonly", width=12,
                                                 values=["winsorize", "iqr", "none"])),
            ("Seuil haute cardinalité :", ttk.Spinbox(opt, textvariable=self.v_card, from_=2, to=200, width=8)),
            ("Seuil de manquants (0 à 1) :", ttk.Spinbox(opt, textvariable=self.v_missing, from_=0.1, to=1.0,
                                                          increment=0.05, width=8)),
            ("Seuil de colinéarité (|r|) :", ttk.Spinbox(opt, textvariable=self.v_corr, from_=0.5, to=1.0,
                                                          increment=0.01, width=8)),
            ("Seuil catégories rares :", ttk.Spinbox(opt, textvariable=self.v_rare, from_=0.0, to=0.2,
                                                      increment=0.005, width=8)),
            ("Seuil d'asymétrie :", ttk.Spinbox(opt, textvariable=self.v_skew, from_=0.1, to=5.0,
                                                 increment=0.1, width=8)),
        ]
        for i, (label, widget) in enumerate(rows):
            ttk.Label(opt, text=label).grid(row=i, column=0, sticky="w")
            widget.grid(row=i, column=1, sticky="w", padx=6, pady=2)
        checks = [
            (self.v_transform_target, "Transformer la cible si elle est très asymétrique (régression)"),
            (self.v_drop_leaky, "Supprimer automatiquement les colonnes qui fuient la cible"),
            (self.v_discrete_cat, "Entiers à faible cardinalité : encodage One-Hot (sinon numériques)"),
        ]
        for j, (var, text) in enumerate(checks, start=len(rows)):
            ttk.Checkbutton(opt, text=text, variable=var).grid(row=j, column=0, columnspan=2, sticky="w", pady=2)

        self.btn_run = ttk.Button(self.tab_params, text="▶  Lancer le prétraitement", command=self._start_run)
        self.btn_run.grid(row=1, column=0, columnspan=2, pady=14)

    def _build_tab_report(self) -> None:
        ttk, tk = self.ttk, self.tk
        frame = ttk.Frame(self.tab_report)
        frame.pack(fill="both", expand=True)
        self.txt_report = tk.Text(frame, wrap="none", font=("Consolas", 9))
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.txt_report.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.txt_report.xview)
        self.txt_report.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.txt_report.pack(fill="both", expand=True)

    def _build_tab_result(self) -> None:
        ttk = self.ttk
        bar = ttk.Frame(self.tab_result, padding=6)
        bar.pack(fill="x")
        self.btn_export = ttk.Button(bar, text="Exporter tout (CSV + rapport + préprocesseur)", command=self._export)
        self.btn_export.pack(side="left")
        ttk.Button(bar, text="Charger un préprocesseur (.joblib)", command=self._load_preprocessor).pack(
            side="left", padx=8)
        ttk.Button(bar, text="Appliquer à un nouveau fichier", command=self._apply_new_file).pack(side="left")
        self.lbl_result = ttk.Label(self.tab_result, text="", padding=6)
        self.lbl_result.pack(fill="x")
        self.tree_result = self._make_tree(self.tab_result)

    # ------------------------------------------------------------------ utilitaires
    def _fill_tree(self, tree, df: pd.DataFrame) -> None:
        view = df.iloc[: self.PREVIEW_ROWS, : self.PREVIEW_COLS]
        cols = [str(c) for c in view.columns]
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, minwidth=60, stretch=False)
        for item in tree.get_children():
            tree.delete(item)
        for row in view.itertuples(index=False):
            tree.insert("", "end", values=[f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in row])

    def _set_status(self, text: str) -> None:
        self.v_status.set(text)

    def _read_options(self) -> Options:
        def num(var, cast, default):
            try:
                return cast(var.get())
            except (ValueError, self.tk.TclError):
                return default

        return Options(
            task=self.v_task.get(),
            test_size_pct=max(0, min(50, num(self.v_test, int, 20))),
            scaler_type=self.v_scaler.get(),
            outlier_method=self.v_outlier.get(),
            cardinality_threshold=num(self.v_card, int, 10),
            missing_threshold=num(self.v_missing, float, 0.90),
            correlation_threshold=num(self.v_corr, float, 0.95),
            rare_category_threshold=num(self.v_rare, float, 0.01),
            skew_threshold=num(self.v_skew, float, 1.0),
            transform_target=bool(self.v_transform_target.get()),
            drop_leaky_columns=bool(self.v_drop_leaky.get()),
            encode_discrete_as_categorical=bool(self.v_discrete_cat.get()),
        )

    # ------------------------------------------------------------------ actions
    def _open_file(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Choisir un dataset",
            filetypes=[("Tableaux", "*.csv *.tsv *.txt *.xlsx *.xls *.json *.parquet"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            df = read_table(path)
        except Exception as exc:  # noqa: BLE001 - message affiché à l'utilisateur
            self.messagebox.showerror("Lecture impossible", str(exc))
            return
        self.df, self.path, self.result = df, path, None
        self.v_file.set(os.path.basename(path))
        target_values = [NO_TARGET_LABEL] + [str(c) for c in df.columns]
        self.cb_target["values"] = target_values
        self.cb_analysis_target["values"] = target_values
        self.v_target.set(NO_TARGET_LABEL)
        self.v_task.set("auto")  # évite de garder « regression » d'un fichier précédent
        n_missing = int(df.isna().sum().sum())
        self.lbl_info.configure(
            text=f"{len(df)} lignes x {df.shape[1]} colonnes | {n_missing} valeurs manquantes | "
                 f"aperçu : {min(len(df), self.PREVIEW_ROWS)} premières lignes")
        self._fill_tree(self.tree_data, df)
        self._set_status("Fichier chargé. Choisissez la cible dans Analyse ou Paramètres.")
        self.notebook.select(self.tab_analysis)

    def _start_run(self) -> None:
        if self.df is None:
            self.messagebox.showinfo("Aucun fichier", "Ouvrez d'abord un fichier.")
            return
        if self.analysis is None:
            self._launch_run()
            return
        summary = build_preprocess_summary(self.analysis, self.choices)
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Vérifier les choix avant lancement")
        dialog.transient(self.root)
        dialog.grab_set()
        text = self.tk.Text(dialog, width=88, height=20, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=8)
        lines = [f"Colonnes avec choix utilisateur : {len(summary['user_columns'])}"]
        lines.append("Colonnes supprimées : " + (", ".join(summary["dropped_columns"]) or "aucune"))
        lines.append("\nAvertissements :")
        lines.extend("• " + warning for warning in summary["warnings"])
        if not summary["warnings"]:
            lines.append("Aucun avertissement statistique détecté.")
        lines.append("\nLes statistiques d'analyse proviennent du fichier chargé; les transformateurs sont ajustés sur le train.")
        text.insert("1.0", "\n".join(lines)); text.configure(state="disabled")
        buttons = self.ttk.Frame(dialog, padding=8); buttons.pack(fill="x")
        self.ttk.Button(buttons, text="Revenir", command=dialog.destroy).pack(side="right", padx=4)
        self.ttk.Button(buttons, text="Lancer", command=lambda: (dialog.destroy(), self._launch_run())).pack(side="right", padx=4)
        dialog.wait_window()

    def _launch_run(self) -> None:
        if self.df is None:
            self.messagebox.showinfo("Aucun fichier", "Ouvrez d'abord un fichier.")
            return
        target = self.v_target.get()
        target = None if target == NO_TARGET_LABEL else target
        opts = self._read_options()
        if self.analysis is not None:
            opts.column_overrides = choices_to_overrides(self.analysis, self.choices)
        self.btn_run.state(["disabled"])
        self.progress.start(12)
        self._set_status("Traitement en cours...")
        df = self.df

        def work() -> None:
            try:
                self._queue.put(("ok", run_preprocessing(df, target, opts)))
            except Exception as exc:  # noqa: BLE001 - remonté à l'interface
                self._queue.put(("err", f"{exc}\n\n{traceback.format_exc()}"))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._poll)

    def _poll(self) -> None:
        try:
            kind, payload = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll)
            return
        self.progress.stop()
        self.btn_run.state(["!disabled"])
        if kind == "err":
            self._set_status("Erreur pendant le traitement.")
            self.txt_report.delete("1.0", "end")
            self.txt_report.insert("1.0", payload)
            self.notebook.select(self.tab_report)
            self.messagebox.showerror("Erreur", payload.split("\n\n")[0])
            return
        self._show_result(payload)

    def _show_result(self, result: RunResult) -> None:
        self.result, self.prep = result, result.prep
        self.txt_report.delete("1.0", "end")
        self.txt_report.insert("1.0", report_text(result))
        shown = _with_target(result.X_train, result.y_train, result.target, None)
        self._fill_tree(self.tree_result, shown)
        extra = f" | test : {result.X_test.shape[0]} lignes" if result.X_test is not None else ""
        self.lbl_result.configure(
            text=f"Train : {result.X_train.shape[0]} lignes x {result.X_train.shape[1]} features numériques{extra}"
                 f" (aperçu : {self.PREVIEW_ROWS} premières lignes)")
        leaks = result.prep.report().get("fuites_suspectees", [])
        warn = f" | ALERTE FUITE : {', '.join(leaks)}" if leaks else ""
        self._set_status(f"Terminé : {result.X_train.shape[1]} features numériques{warn}")
        self.notebook.select(self.tab_report)

    def _export(self) -> None:
        if self.result is None:
            self.messagebox.showinfo("Rien à exporter", "Lancez d'abord le prétraitement.")
            return
        folder = self.filedialog.askdirectory(title="Dossier de destination")
        if folder:
            self._do_export(folder)

    def _do_export(self, folder: str) -> None:
        try:
            written = export_result(self.result, folder)
            if self.analysis is not None:
                analysis_path = os.path.join(folder, "analyse_colonnes.csv")
                build_analysis_table(self.analysis, self.choices).to_csv(analysis_path, index=False, encoding="utf-8-sig")
                choices_path = os.path.join(folder, "choix_utilisateur.json")
                export_user_choices(self.analysis, self.choices, choices_path)
                written.extend([analysis_path, choices_path])
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Export impossible", str(exc))
            return
        self._set_status(f"Export terminé dans {folder}")
        self.messagebox.showinfo("Export terminé", "Fichiers écrits :\n" + "\n".join(os.path.basename(p) for p in written))

    def _load_preprocessor(self) -> None:
        path = self.filedialog.askopenfilename(title="Préprocesseur", filetypes=[("joblib", "*.joblib")])
        if path:
            self._do_load_preprocessor(path)

    def _do_load_preprocessor(self, path: str) -> None:
        try:
            self.prep = AutoPreprocessor.load(path)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Chargement impossible", str(exc))
            return
        self._set_status(f"Préprocesseur chargé : {os.path.basename(path)}")

    def _apply_new_file(self) -> None:
        if self.prep is None:
            self.messagebox.showinfo("Aucun préprocesseur", "Lancez un prétraitement ou chargez un fichier .joblib.")
            return
        path = self.filedialog.askopenfilename(title="Nouveau fichier à transformer")
        if not path:
            return
        out_path = self.filedialog.asksaveasfilename(title="Enregistrer le résultat", defaultextension=".csv",
                                                     filetypes=[("CSV", "*.csv")])
        if out_path:
            self._do_apply_new_file(path, out_path)

    def _do_apply_new_file(self, path: str, out_path: str) -> None:
        try:
            out = apply_to_new_data(self.prep, read_table(path))
            out.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Transformation impossible", str(exc))
            return
        self._fill_tree(self.tree_result, out)
        self.lbl_result.configure(text=f"Nouveau fichier transformé : {out.shape[0]} lignes x {out.shape[1]} colonnes")
        self._set_status(f"Résultat enregistré : {out_path}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        app = App()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Tkinter est introuvable ({exc}). Réinstallez Python depuis python.org avec l'option tcl/tk.")
    app.run()


if __name__ == "__main__":
    main()
