"""Isolated dataset inspection, transformation and export for the local web API."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from app_preprocess import (
    Options, RunResult, _looks_like_classification, _with_target, analysis_recommendation_summary, build_config,
    build_preview_table, choices_to_overrides, label_action, label_step,
    preview_for_choice, read_table, run_preprocessing,
)
from auto_preprocessor import AutoPreprocessor, ColumnOverride, _override_as_choices
from statistical_profile import association

MATRIX_LIMIT = 40
CATEGORY_LIMIT = 12
RELATION_LIMIT = 20_000
DOWNLOADS = {"train": "train.csv", "test": "test.csv", "full": "full.csv", "report_xlsx": "rapport.xlsx"}


class InputError(ValueError):
    """A dataset or selection error safe to show to the user."""


def json_safe(value):
    """Convert pandas and NumPy results to strict JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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
    counts, edges = np.histogram(values, bins=24 if values.min() != values.max() else 1)
    return [{"label": f"{low:.5g} à {high:.5g}", "count": int(count), "low": low, "high": high}
            for count, low, high in zip(counts, edges[:-1], edges[1:])]


def feature_stats(series):
    """Compute missingness, descriptive statistics and robust outlier diagnostics."""
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    values = series.dropna()
    top = [] if numeric else [{"label": str(k), "count": int(v)}
                              for k, v in values.astype(str).value_counts().head(15).items()]
    profile = {"column": series.name, "dtype": str(series.dtype), "type": "numérique" if numeric else
               "date" if pd.api.types.is_datetime64_any_dtype(series) else "catégoriel / texte",
               "missing_count": int(series.isna().sum()), "missing_pct": float(series.isna().mean() * 100),
               "unique": int(values.nunique()), "top": top}
    if not numeric:
        lengths = values.astype(str).str.len()
        profile["text_length"] = {"mean": lengths.mean(), "min": lengths.min(), "max": lengths.max()}
    elif len(values):
        q1, median, q3 = values.quantile([.25, .5, .75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        inside = values.between(lower, upper)
        mad = (values - median).abs().median()
        radius = 3.5 * mad / .67448975 if mad > 0 else None
        profile.update(mean=values.mean(), std=values.std(), min=values.min(), max=values.max(),
                       skew=values.skew(), kurtosis=values.kurt(), q1=q1, median=median, q3=q3,
                       boxplot={"low": values[inside].min(), "q1": q1, "median": median,
                                "q3": q3, "high": values[inside].max()},
                       iqr={"count": int((~inside).sum()), "lower": lower, "upper": upper},
                       robust_z={"count": int(((values - median).abs() > radius).sum()) if radius is not None else None,
                                 "lower": median - radius if radius is not None else None,
                                 "upper": median + radius if radius is not None else None})
    return profile


def feature_eda(path, limit, column):
    """Compute the chart series for a single input column."""
    frame = load_frame(path, limit)
    if column not in frame:
        raise InputError("Colonne inconnue.")
    series = frame[column]
    profile = feature_stats(series)
    profile["distribution"] = histogram(series.dropna()) if profile["type"] == "numérique" else profile["top"]
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
    numeric = [row["column"] for row in profiles if row["type"] == "numérique"]
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
            if name == target:
                continue
            evidence = association(sample[name], sample[target], name in numeric, regression, kendall=False)
            primary = next(iter(evidence["tests"]), None)
            method = {"spearman": "Spearman", "kruskal": "Kruskal · ε²", "khi_deux": "V de Cramér",
                      "fisher": "V de Cramér · Fisher"}.get(primary)
            relations.append({"column": name, "effect": evidence.get("effet_primaire"), "method": method,
                              "sample_size": evidence["effectif"], "note": evidence.get("limite")})
    duplicates = frame.duplicated()
    return {"shape": frame.shape, "features": profiles,
            "missing_pct": sum(row["missing_count"] for row in profiles) * 100 / frame.size,
            "duplicate_count": int(duplicates.sum()), "duplicate_examples": frame[duplicates].head(5).to_dict("records"),
            "pearson": {"columns": pearson.columns, "values": pearson.to_numpy(), "limit": MATRIX_LIMIT,
                        "eligible": len(numeric)},
            "cramers_v": {"columns": cat_names, "values": cat_matrix, "limit": CATEGORY_LIMIT,
                          "eligible": len(categorical), "max_categories": 40},
            "relation_sample_size": len(sample), "target_relations": relations}


def eda(path, limit, target, task):
    """Return the complete EDA payload as strict JSON."""
    return json_safe(dataset_eda(load_frame(path, limit), target, task))


def analyze_frame(frame, target, options):
    """Use the existing recommendation engine with the selected task and options."""
    return AutoPreprocessor(build_config(options), task=options.task).analyze(frame, target=target)


def recommendation(row, target):
    """Expose statistical evidence and applicable desktop actions to the browser."""
    anomalies = []
    for key, label in (("manquants_pct", "Manquants (%)"), ("outliers_pct", "Extrêmes (%)"),
                       ("faux_manquants", "Faux manquants"), ("variantes_casse_espaces", "Variantes"),
                       ("date_formats", "Formats de date")):
        if row.get(key, 0) > 0:
            anomalies.append(f"{label} · {row[key]:.3g}")
    if row.get("type", "").startswith("DROP"):
        anomalies.append(row["type"])
    if row.get("alerte_identifiant"):
        anomalies.append("Identifiant potentiel")
    if row.get("paires_correlees"):
        anomalies.append("Colinéarité")
    if abs(row.get("skew", 0)) > 1:
        anomalies.append(f"Asymétrie · {row['skew']:.3g}")
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
        for action in detail["choices"]:
            if action == "AUTO" or action in unavailable or step in {"target", "nettoyage"}:
                continue
            if detail.get("comparaison", {}).get(action, {}).get("applicable") is False:
                continue
            alternatives.append({"value": f"{step}:{action}", "label": f"{label_step(step)} · {label_action(action)}"})
    return {"column": row["colonne"], "type": row["type"], "is_target": row.get("est_cible", False),
            "anomalies": anomalies, "action": row["action"], "label": label_action(row["action"]),
            "justification": analysis_recommendation_summary(row), "alternatives": alternatives, "steps": steps}


def analyze(path, limit, target, task):
    """Run analysis and return one recommendation per input column."""
    result = analyze_frame(load_frame(path, limit), target, Options(task=task))
    return json_safe({"features": [recommendation(row, target) for row in result.columns]})


def selected_overrides(result, target, choices):
    """Validate public choice tokens and convert alternatives to engine overrides."""
    rows = {row["colonne"]: row for row in result.columns}
    selected = {}
    for column, choice in choices.items():
        if column not in rows or column == target:
            raise InputError("Le choix doit porter sur une colonne explicative existante.")
        if choice in {"recommended", "none"}:
            continue
        allowed = {item["value"] for item in recommendation(rows[column], target)["alternatives"]}
        if choice not in allowed:
            raise InputError("Transformation incompatible avec la colonne sélectionnée.")
        step, separator, action = choice.partition(":")
        selected[column] = {step: action} if separator else choice
    return choices_to_overrides(result, selected), selected


def preview(path, limit, target, task, column, choice):
    """Render chart data and the desktop comparison table for a proposed choice."""
    frame = load_frame(path, limit)
    if column not in frame or column == target:
        raise InputError("Choisissez une colonne explicative existante.")
    series = frame[column]
    profile = feature_stats(series)
    chart = histogram(series.dropna()) if profile["type"] == "numérique" else profile["top"]
    if choice == "none":
        return json_safe({"status": "ok", "before_chart": chart, "after_charts": [{"title": column, "data": chart}],
                          "notes": ["Les valeurs brutes sont conservées sur les lignes exportées."], "table": []})
    options = Options(task=task)
    result = analyze_frame(frame, target, options)
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
                      "notes": comparison["notes"], "table": build_preview_table(comparison).to_dict("records")})


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
    return {"method": type(estimator).__name__, **learned}


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
    if column in prep.categorical_imputers_:
        parameters["fill_value"] = prep.categorical_imputers_[column]
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
    workbook_sheet(workbook, "Résumé", ["Indicateur", "Valeur"],
                   ((key, value) for key, value in summary.items()))
    fields = [
        ("column", "Feature"), ("type", "Type détecté"), ("missing_pct", "Manquants (%)"),
        ("unique", "Uniques"), ("mean", "Moyenne"), ("std", "Écart-type"),
        ("min", "Minimum"), ("max", "Maximum"), ("skew", "Asymétrie"), ("top", "Modalité principale"),
        ("outliers", "Extrêmes IQR"), ("anomalies", "Anomalies"), ("recommended", "Recommandation"),
        ("justification", "Justification"), ("user_choice", "Choix utilisateur"),
        ("choice", "Transformation choisie"), ("applied", "Transformation appliquée"),
        ("parameters", "Paramètres appris sur train"), ("outputs", "Colonnes créées"),
        ("dropped", "Supprimée"), ("drop_reason", "Motif"), ("leakage", "Alerte fuite"),
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


def execute(path, limit, target, task, options, choices, output, dataset_name):
    """Fit on train, transform test, and write an atomic generation of export files."""
    raw_frame = load_frame(path, limit, keep_nonfinite=True)
    frame = raw_frame.replace([np.inf, -np.inf], np.nan)
    opts = Options(task=task, **options)
    analysis = analyze_frame(frame, target, opts)
    overrides, _ = selected_overrides(analysis, target, choices)
    passthrough = [column for column in frame if choices.get(column) == "none"]
    active = frame.drop(columns=passthrough)
    feature_names = active.drop(columns=[target]) if target is not None else active
    config = build_config(opts)
    normalized = AutoPreprocessor(config)._normalize_column_names(feature_names.iloc[:0]).columns
    name_map = dict(zip(feature_names.columns, normalized))
    opts.column_overrides = overrides
    if passthrough and len(feature_names.columns):
        # Rows equal on processed features can differ in a passthrough feature.
        # Leave those rows intact unless the user explicitly requests deduplication.
        if not any(choice.startswith("duplicates:") for choice in choices.values()):
            first = feature_names.columns[0]
            opts.column_overrides[first] = ColumnOverride({
                **_override_as_choices(overrides.get(first)), "duplicates": "KEEP",
            })
    if target is not None:
        if active[target].notna().sum() < 10:
            raise InputError("Le mode supervisé nécessite au moins dix lignes avec une cible renseignée.")
        result = run_preprocessing(active, target, opts) if name_map else fit_target_only(active, target, opts)
    else:
        if len(active) < 2:
            raise InputError("Le découpage nécessite au moins deux lignes.")
        train_data, test_data = train_test_split(active, test_size=opts.test_size_pct / 100,
                                                 random_state=opts.random_state)
        if not len(active.columns):
            result = RunResult(AutoPreprocessor(build_config(opts)), None, train_data, None, test_data)
        else:
            result = run_preprocessing(train_data, None, opts)
            result.X_test = result.prep.transform(test_data)
            result.notes = ["Mode sans cible : ajustement sur train et transformation du test."]
    train, raw_outputs, renamed = export_frame(
        raw_frame, result.X_train, passthrough, result.y_train, target, result.y_train_original)
    test, _, _ = export_frame(raw_frame, result.X_test, passthrough, result.y_test, target, result.y_test_original)
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
        leak = core["scores_fuite_par_colonne"].get(name) if name in core["fuites_suspectees"] else None
        features.append({**stats, "type": row["type"], "top": stats["top"][0]["label"] if stats["top"] else None,
                         "outliers": stats.get("iqr", {}).get("count"), "anomalies": rec["anomalies"],
                         "recommended": rec["action"], "justification": rec["justification"], "choice": choice,
                         "user_choice": "cible" if column == target else
                         {"recommended": "acceptée", "none": "aucune"}.get(choice, "alternative"),
                         "applied": applied, "parameters": parameters, "outputs": outputs, "dropped": dropped,
                         "drop_reason": (status if status.startswith("SUPPRIM") else
                                         "Aucune colonne après sélection") if dropped else None,
                         "leakage": leak})
    warnings = list(dict.fromkeys(result.notes + result.prep.warnings_))
    if passthrough:
        warnings.append("Les colonnes laissées brutes peuvent contenir du texte et des valeurs manquantes.")
    summary = {"Dataset": dataset_name, "Mode": "supervisé" if target else "non supervisé", "Cible": target,
               "Tâche": core["tache_detectee"], "Dimensions initiales": frame.shape,
               "Dimensions train": train.shape, "Dimensions test": test.shape,
               "Dimensions complètes": (len(train) + len(test), train.shape[1]),
               "Part test demandée (%)": opts.test_size_pct, "Graine": opts.random_state,
               "Colonnes exportées": train.columns, "Avertissements": warnings,
               "Nettoyage du train": core["nettoyage_dataset"],
               "Plafonds Pearson / V de Cramér": [MATRIX_LIMIT, CATEGORY_LIMIT]}
    report = json_safe({"summary": summary, "features": features, "core": {**core, "colonnes_detaillees": detail},
                        "correlations": {name: eda_result[name] for name in ("pearson", "cramers_v")}})
    for split, table in (("train", train), ("test", test)):
        table.to_csv(output / DOWNLOADS[split], index=False, encoding="utf-8-sig")
    full = output / DOWNLOADS["full"]
    train.to_csv(full, index=False, encoding="utf-8-sig")
    test.to_csv(full, mode="a", index=False, header=False, encoding="utf-8")
    write_workbook(output / DOWNLOADS["report_xlsx"], report)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return json_safe({"input_shape": frame.shape, "train_shape": train.shape, "test_shape": test.shape,
                      "full_shape": summary["Dimensions complètes"], "output_columns": train.columns,
                      "warnings": warnings, "downloads": list(DOWNLOADS)})
