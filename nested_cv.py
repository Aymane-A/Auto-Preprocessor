"""Independent outer evaluation of complete AutoPrépa pipelines.

All candidate choices are scored by fitting a new preprocessor and model on each
inner training partition. The outer validation partition is used only once,
after candidate selection and refitting on the entire outer training partition.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             get_scorer, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from auto_preprocessor import AutoPreprocessor, PreprocessConfig, to_sparse_matrix
from model_evaluation import _model


def _splitter(mode, folds, seed, time_gap_rows=0):
    if not 2 <= folds <= 5:
        raise ValueError("La CV imbriquée exige entre 2 et 5 plis par niveau.")
    if mode == "stratified":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if mode == "kfold":
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    if mode == "group":
        return GroupKFold(n_splits=folds)
    if mode == "time":
        return TimeSeriesSplit(n_splits=folds, gap=time_gap_rows)
    raise ValueError(f"Mode de découpage inconnu : {mode}.")


def _splits(mode, folds, seed, X, y, groups, time_gap_rows=0):
    splitter = _splitter(mode, folds, seed, time_gap_rows=time_gap_rows)
    if mode == "group":
        if groups is None or pd.isna(groups).any() or len(np.unique(groups)) < folds:
            raise ValueError("Les groupes non manquants doivent être au moins aussi nombreux que les plis.")
        return list(splitter.split(X, y, groups))
    if mode == "stratified":
        if pd.Series(y).value_counts().min() < folds:
            raise ValueError("Chaque classe doit compter au moins autant de lignes que de plis.")
        return list(splitter.split(X, y))
    return list(splitter.split(X, y))


def _pipeline(config, family, classification, model_parameters):
    model = _model(family, classification, config.random_state)
    if model_parameters:
        model.set_params(**model_parameters)
    stages = [AutoPreprocessor(config, task="classification" if classification else "regression")]
    if config.sparse_output:
        stages.append(FunctionTransformer(to_sparse_matrix))
    return make_pipeline(*stages, model)


def _require_safe_wrapper(pipeline):
    report = getattr(pipeline.steps[0][1], "feature_selection_report_", None) or {}
    if report.get("method") in {"rfecv", "forward", "backward"} and report.get("cv_preprocessing") != "fold_isolated":
        raise ValueError("La CV imbriquée refuse le repli unsafe_legacy du sélecteur.")


def _metrics(y_true, predicted, scores, classes, classification):
    if not classification:
        return {"mae": float(mean_absolute_error(y_true, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, predicted))),
                "r2": float(r2_score(y_true, predicted))}
    binary = len(classes) == 2
    result = {"accuracy": float(accuracy_score(y_true, predicted)),
              "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted))}
    if binary:
        result.update({"precision": float(precision_score(y_true, predicted, pos_label=classes[1], zero_division=0)),
                       "recall": float(recall_score(y_true, predicted, pos_label=classes[1], zero_division=0)),
                       "f1": float(f1_score(y_true, predicted, pos_label=classes[1], zero_division=0))})
        if scores is not None and len(np.unique(y_true)) == 2:
            truth = np.asarray(y_true) == classes[1]
            result["roc_auc"] = float(roc_auc_score(truth, scores))
            result["pr_auc"] = float(average_precision_score(truth, scores))
    else:
        result.update({"f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
                       "f1_weighted": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
                       "confusion_matrix": confusion_matrix(y_true, predicted, labels=classes).tolist()})
    return result


def nested_cross_validate(frame, target, family, *, config=None, task="auto",
                          outer_cv=3, inner_cv=3, candidates=None, model_params=None,
                          split_mode="auto", time_column=None, groups=None,
                          max_rows=5000, time_gap_rows=0):
    """Select configurations in inner CV; evaluate one refit per outer fold.

    ``candidates`` is a sequence of PreprocessConfig override dictionaries;
    ``model_params`` is a sequence of sklearn model parameter dictionaries.
    The returned fitted outer pipelines are Joblib-serializable. For time splits,
    early training observations are explicitly listed as unscored. ``time_gap_rows``
    removes that many observations immediately before each outer and inner
    validation block; the caller must derive it from the label horizon.
    """
    if target not in frame:
        raise ValueError("La colonne cible est absente.")
    if task not in {"classification", "regression"}:
        raise ValueError("La tâche doit être explicite pour éviter de l'inférer sur la validation externe.")
    if time_column is not None and time_column not in frame:
        raise ValueError("La colonne temporelle est absente.")
    if groups is not None and len(groups) != len(frame):
        raise ValueError("Les groupes ne sont pas alignés avec les lignes brutes.")
    if time_column is not None and groups is not None:
        raise ValueError("Les découpages temporel et groupé sont mutuellement exclusifs.")
    if not isinstance(time_gap_rows, int) or isinstance(time_gap_rows, bool) or time_gap_rows < 0:
        raise ValueError("time_gap_rows doit être un entier positif ou nul.")
    if time_column is None and time_gap_rows:
        raise ValueError("time_gap_rows exige une colonne temporelle.")
    base = config or PreprocessConfig(verbose=0)
    if not isinstance(base, PreprocessConfig):
        raise TypeError("config doit être un PreprocessConfig.")
    candidate_overrides = list(candidates) if candidates is not None else [{}]
    model_options = list(model_params) if model_params is not None else [{}]
    if not candidate_overrides or not model_options:
        raise ValueError("Au moins une configuration de prétraitement et de modèle est requise.")
    if len(candidate_overrides) * len(model_options) > 16:
        raise ValueError("La recherche accepte au plus 16 combinaisons de configurations.")
    allowed = set(asdict(base))
    if any(not isinstance(item, dict) or set(item) - allowed for item in candidate_overrides):
        raise ValueError("Une configuration candidate contient un paramètre inconnu.")
    if any(not isinstance(item, dict) for item in model_options):
        raise ValueError("Les paramètres du modèle doivent être des dictionnaires.")

    kept = frame[target].notna().to_numpy()
    source_positions = np.flatnonzero(kept)
    data = frame.iloc[source_positions]
    if len(data) < 20:
        raise ValueError("La CV imbriquée exige au moins 20 cibles renseignées.")
    if len(data) > max_rows:
        raise ValueError(f"La CV imbriquée accepte au plus {max_rows} lignes.")
    if time_column is not None:
        times = pd.to_datetime(data[time_column], errors="coerce", utc=True, format="mixed")
        if times.isna().any() or times.duplicated().any():
            raise ValueError("Le découpage temporel exige des dates valides et distinctes.")
        order = np.argsort(times.to_numpy(), kind="stable")
        data = data.iloc[order]
        source_positions = source_positions[order]
    y = data[target].to_numpy()
    resolved = AutoPreprocessor(base, task=task)._resolve_task(y)
    if resolved != task:
        raise ValueError("La cible est incompatible avec la tâche explicite.")
    classification = resolved == "classification"
    classes = np.unique(y) if classification else None
    mode = ("time" if time_column is not None else "group" if groups is not None else
            "stratified" if classification else "kfold") if split_mode == "auto" else split_mode
    if mode == "stratified" and not classification:
        raise ValueError("La stratification exige une cible de classification.")
    if mode == "kfold" and classification:
        raise ValueError("La classification sans groupes ni dates exige StratifiedKFold.")
    if mode == "time" and time_column is None:
        raise ValueError("Le mode temporel exige time_column.")
    if mode == "group" and groups is None:
        raise ValueError("Le mode groupé exige groups.")
    if time_column is not None and mode != "time":
        raise ValueError("La colonne temporelle doit utiliser TimeSeriesSplit.")
    if groups is not None and mode != "group":
        raise ValueError("Les groupes doivent utiliser GroupKFold.")
    excluded = [target] + ([time_column] if time_column is not None else [])
    X = data.drop(columns=excluded)
    group_values = np.asarray(groups)[source_positions] if groups is not None else None
    outer_splits = _splits(mode, outer_cv, base.random_state, X, y, group_values,
                           time_gap_rows=time_gap_rows)
    scoring = "balanced_accuracy" if classification else "neg_mean_absolute_error"
    scorer = get_scorer(scoring)
    fold_results, fitted_models = [], []
    all_positions, all_true, all_pred, all_scores = [], [], [], []
    for fold_number, (outer_train, outer_valid) in enumerate(outer_splits, 1):
        fold_started = perf_counter()
        X_train, X_valid = X.iloc[outer_train], X.iloc[outer_valid]
        y_train, y_valid = y[outer_train], y[outer_valid]
        if classification and set(np.unique(y_train)) != set(classes):
            raise ValueError(f"Pli externe {fold_number} : le train ne contient pas toutes les classes.")
        inner_groups = group_values[outer_train] if group_values is not None else None
        inner_splits = _splits(mode, inner_cv, base.random_state,
                               X_train, y_train, inner_groups, time_gap_rows=time_gap_rows)
        inner_metric_warnings = []
        if classification:
            for inner_number, (inner_train, inner_valid) in enumerate(inner_splits, 1):
                if set(np.unique(y_train[inner_train])) != set(classes):
                    raise ValueError(f"Pli externe {fold_number}, pli interne {inner_number} : "
                                     "train sans toutes les classes.")
                if set(np.unique(y_train[inner_valid])) != set(classes):
                    inner_metric_warnings.append(
                        f"Pli interne {inner_number} : validation sans toutes les classes ; "
                        "le score de recherche peut être non comparable.")
        evaluations = []
        for overrides in candidate_overrides:
            cfg = replace(base, **overrides, preserve_rows=True, transform_target=False)
            for parameters in model_options:
                scores = []
                for inner_train, inner_valid in inner_splits:
                    pipeline = _pipeline(cfg, family, classification, parameters)
                    pipeline.fit(X_train.iloc[inner_train], y_train[inner_train])
                    _require_safe_wrapper(pipeline)
                    score = float(scorer(pipeline, X_train.iloc[inner_valid], y_train[inner_valid]))
                    if not np.isfinite(score):
                        raise ValueError("Score interne non fini.")
                    scores.append(score)
                evaluations.append({"preprocessing": dict(overrides), "model": dict(parameters),
                                    "scores": scores, "mean": float(np.mean(scores))})
        chosen = max(evaluations, key=lambda item: item["mean"])
        chosen_cfg = replace(base, **chosen["preprocessing"], preserve_rows=True,
                             transform_target=False)
        final = _pipeline(chosen_cfg, family, classification, chosen["model"])
        final.fit(X_train, y_train)
        _require_safe_wrapper(final)
        predicted = final.predict(X_valid)
        score_values = None
        if classification and len(classes) == 2:
            model = final.steps[-1][1]
            if hasattr(final, "predict_proba"):
                score_values = final.predict_proba(X_valid)[:, list(model.classes_).index(classes[1])]
            elif hasattr(final, "decision_function"):
                score_values = final.decision_function(X_valid)
        metrics = _metrics(y_valid, predicted, score_values, classes, classification)
        metric_warnings = inner_metric_warnings
        if classification and set(np.unique(y_valid)) != set(classes):
            metric_warnings.append("La validation externe ne contient pas toutes les classes ; "
                                   "certaines métriques sont absentes ou non comparables entre plis.")
        positions = source_positions[outer_valid].tolist()
        fold_results.append({"fold": fold_number, "train_row_positions": source_positions[outer_train].tolist(),
                             "validation_row_positions": positions, "predictions": np.asarray(predicted).tolist(),
                             "targets": np.asarray(y_valid).tolist(),
                             "prediction_scores": None if score_values is None else np.asarray(score_values).tolist(),
                             "metrics": metrics, "selected_config": {"preprocessing": chosen["preprocessing"],
                                                                       "model": chosen["model"]},
                             "metric_warnings": metric_warnings,
                             "selected_features": list(final.steps[0][1].final_feature_names_),
                             "inner_search": evaluations,
                             "elapsed_seconds": float(perf_counter() - fold_started),
                             "pipeline_fit_count": len(inner_splits) * len(evaluations) + 1})
        fitted_models.append(final)
        all_positions.extend(positions)
        all_true.extend(y_valid)
        all_pred.extend(predicted)
        if score_values is not None:
            all_scores.extend(score_values)
    if len(all_positions) != len(set(all_positions)):
        raise RuntimeError("Une observation a été prédite plusieurs fois en validation externe.")
    global_metrics = _metrics(np.asarray(all_true), np.asarray(all_pred),
                              np.asarray(all_scores) if len(all_scores) == len(all_true) else None,
                              classes, classification)
    metric_keys = set().union(*(fold["metrics"] for fold in fold_results)) - {"confusion_matrix"}
    summary = {}
    for key in sorted(metric_keys):
        values = [fold["metrics"][key] for fold in fold_results if key in fold["metrics"]]
        summary[key] = {"mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "folds": len(values)}
    return {"protocol": "nested_cross_validation", "task": resolved, "split_mode": mode,
            "time_gap_rows": time_gap_rows,
            "outer_folds": fold_results, "outer_summary": summary,
            "global_metrics": global_metrics, "scoring_inner": scoring,
            "classes": None if classes is None else classes.tolist(),
            "predicted_row_positions": all_positions,
            "unscored_row_positions": sorted(set(source_positions.tolist()) - set(all_positions)),
            "fitted_outer_pipelines": fitted_models}
