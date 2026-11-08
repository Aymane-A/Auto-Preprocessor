"""Optional cross-validation of preprocessing candidates for a chosen model family."""

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from auto_preprocessor import AutoPreprocessor, PreprocessConfig, to_sparse_matrix


def _model(family, classification, seed):
    models = {
        "linear": (LogisticRegression(max_iter=500), Ridge()),
        "logistic": (LogisticRegression(max_iter=500), None),
        "decision_tree": (DecisionTreeClassifier(random_state=seed), DecisionTreeRegressor(random_state=seed)),
        "random_forest": (RandomForestClassifier(n_estimators=50, random_state=seed),
                          RandomForestRegressor(n_estimators=50, random_state=seed)),
        "gradient_boosting": (HistGradientBoostingClassifier(random_state=seed),
                              HistGradientBoostingRegressor(random_state=seed)),
        "knn": (KNeighborsClassifier(n_neighbors=3), KNeighborsRegressor(n_neighbors=3)),
        "svm": (SVC(), SVR()),
        "neural_network": (MLPClassifier(max_iter=300, random_state=seed),
                           MLPRegressor(max_iter=300, random_state=seed)),
    }
    if family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        return (XGBClassifier if classification else XGBRegressor)(random_state=seed, n_jobs=1)
    if family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        return (LGBMClassifier if classification else LGBMRegressor)(random_state=seed, n_jobs=1)
    estimator = models.get(family, (None, None))[0 if classification else 1]
    if estimator is None:
        raise ValueError(f"Famille {family!r} indisponible pour cette tâche.")
    return estimator


def compare_preprocessing(frame, target, family, *, config=None, task="auto", cv=3,
                          time_column=None, candidates=("none", "standard", "robust"), max_rows=5000):
    """Score up to four scaling settings using identical splits and a new fit in each fold.

    The result is exploratory CV evidence. Use an independent holdout to assess a chosen setting.
    Temporal rows are ordered stably; time-based folds require unique timestamps to avoid ties.
    """
    if not 2 <= cv <= 5 or not 1 <= len(candidates) <= 4:
        raise ValueError("Entre 2 et 5 plis et entre 1 et 4 configurations sont requis.")
    if set(candidates) - {"none", "standard", "robust", "minmax"}:
        raise ValueError("Scalers admis : none, standard, robust, minmax.")
    data = frame.dropna(subset=[target])
    if not 20 <= len(data) <= max_rows:
        raise ValueError(f"L'évaluation nécessite entre 20 et {max_rows} cibles renseignées.")
    base = config or PreprocessConfig(verbose=0)
    if base.sparse_output and family == "gradient_boosting":
        raise ValueError("Gradient boosting histogramme exige une entrée dense. Désactivez sparse_output.")
    seed = base.random_state
    resolved = AutoPreprocessor(base, task=task)._resolve_task(data[target])
    classification = resolved == "classification"
    if time_column is not None:
        times = pd.to_datetime(data[time_column], errors="coerce", format="mixed", utc=True)
        if times.isna().any() or times.duplicated().any():
            raise ValueError("Le découpage temporel exige des dates valides et distinctes.")
        data = data.iloc[np.argsort(times.to_numpy(), kind="stable")]
        splitter = TimeSeriesSplit(n_splits=cv)
    elif classification:
        if data[target].value_counts().min() < cv:
            raise ValueError("Chaque classe doit avoir au moins autant de lignes que de plis.")
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    features = data.drop(columns=[target])
    labels = data[target]
    if classification:
        labels = pd.factorize(labels, sort=True)[0]
    splits = list(splitter.split(features, labels))
    if classification and any(len(np.unique(labels[train])) < len(np.unique(labels)) for train, _ in splits):
        raise ValueError("Au moins un pli d'entraînement ne contient pas toutes les classes.")
    scoring = "balanced_accuracy" if classification else "neg_mean_absolute_error"
    results = []
    for scaler in candidates:
        cfg = replace(base, scaler_type=scaler, model_family=None, preserve_rows=True, transform_target=False,
                      supervised_encoding=base.supervised_encoding and time_column is None)
        stages = [AutoPreprocessor(cfg, task=resolved)]
        if cfg.sparse_output:
            stages.append(FunctionTransformer(to_sparse_matrix))
        pipeline = make_pipeline(*stages, _model(family, classification, seed))
        scores = cross_val_score(pipeline, features, labels, cv=splits, scoring=scoring, error_score="raise")
        results.append({"scaler": scaler, "scores": scores.tolist(), "mean": float(scores.mean()),
                        "std": float(scores.std(ddof=1))})
    return {"family": family, "scoring": scoring, "folds": cv, "results": results}
