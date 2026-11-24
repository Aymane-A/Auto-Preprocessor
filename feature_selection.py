"""Train-only feature selectors for encoded AutoPrépa output."""
from __future__ import annotations
import time
import warnings
from functools import partial
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.feature_selection import (SelectKBest, SelectPercentile, VarianceThreshold,
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SequentialFeatureSelector, SelectFromModel)
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

METHODS = ("auto", "variance", "correlation", "kbest", "chi2", "anova", "mutual_info",
           "percentile", "rfe", "rfecv", "forward", "backward", "l1", "elastic_net",
           "select_from_model", "random_forest")
SUPERVISED = set(METHODS) - {"auto", "variance", "correlation"}
WRAPPERS = {"rfe", "rfecv", "forward", "backward"}


def fit_selector(frame, target, *, method, task, count, percentile, folds, random_state,
                 max_dense_bytes, scoring=None):
    """Fit on encoded training rows only; return the fitted selector and auditable metadata."""
    if method not in METHODS:
        raise ValueError(f"Méthode de sélection inconnue : {method}.")
    if method in {"auto", "correlation"}:
        return None, None
    if method in {"rfecv", "forward", "backward"}:
        warnings.warn("fit_selector reçoit une matrice déjà prétraitée : ses plis internes ne réajustent pas le prétraitement. Utilisez AutoPreprocessor pour une validation isolée par pli.", UserWarning, stacklevel=2)
    if frame.shape[1] < 1:
        raise ValueError("Aucune feature disponible pour la sélection.")
    if method in SUPERVISED and target is None:
        raise ValueError(f"{method} exige une colonne cible supervisée.")
    if method in SUPERVISED and task not in {"classification", "regression"}:
        raise ValueError("La sélection supervisée exige une tâche de classification ou de régression.")
    count_methods = {"kbest", "chi2", "anova", "mutual_info", "rfe", "forward", "backward",
                     "l1", "elastic_net", "select_from_model", "random_forest"}
    if count is not None and method in count_methods and count > frame.shape[1]:
        raise ValueError("Le nombre de features demandées dépasse les features disponibles après encodage.")
    if method in {"forward", "backward"} and frame.shape[1] < 2:
        raise ValueError("La sélection séquentielle exige au moins deux features disponibles.")
    if method in {"forward", "backward"} and count is not None and count >= frame.shape[1]:
        raise ValueError("La sélection séquentielle exige moins de features retenues que de features disponibles.")
    if scoring is not None and method in {"rfecv", "forward", "backward"}:
        allowed = ({"accuracy", "f1", "roc_auc"} if task == "classification" else
                   {"r2", "neg_mean_squared_error"})
        if scoring not in allowed:
            raise ValueError(f"La métrique {scoring} est incompatible avec la tâche {task}.")
    if method in WRAPPERS and (len(frame) > 20000 or frame.shape[1] > 200):
        raise ValueError("Les méthodes wrapper exigent au plus 20 000 lignes et 200 features ; réduisez d'abord les dimensions.")
    if frame.shape[0] * frame.shape[1] * 8 > max_dense_bytes:
        raise ValueError("La sélection dépasserait le budget mémoire dense ; réduisez les dimensions.")
    X = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("La sélection exige des features numériques finies après imputation.")
    y = None if target is None else np.asarray(target)
    classification = task == "classification"
    resolved_scoring = scoring
    if classification and y is not None and len(np.unique(y)) > 2:
        resolved_scoring = {"f1": "f1_macro", "roc_auc": "roc_auc_ovr"}.get(scoring, scoring)
    if method in {"rfecv", "forward", "backward"} and resolved_scoring is not None:
        get_scorer(resolved_scoring)
    k = count if count is not None else min(20, X.shape[1] - 1 if method in {"forward", "backward"} else X.shape[1])
    estimator = (LogisticRegression(max_iter=500, random_state=random_state) if classification
                 else Lasso(alpha=.01, max_iter=1000, random_state=random_state))
    forest = (RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
              if classification else RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1))
    if method == "variance": selector = VarianceThreshold(threshold=0.)
    elif method == "chi2":
        if (X < 0).any(): raise ValueError("Chi-Square exige des features non négatives ; choisissez MinMax ou une autre méthode.")
        if not classification: raise ValueError("Chi-Square exige une cible de classification.")
        selector = SelectKBest(chi2, k=k)
    elif method in {"kbest", "anova"}:
        selector = SelectKBest(f_classif if classification else f_regression, k=k)
    elif method == "mutual_info":
        score = partial(mutual_info_classif if classification else mutual_info_regression,
                        random_state=random_state)
        selector = SelectKBest(score, k=k)
    elif method == "percentile": selector = SelectPercentile(f_classif if classification else f_regression, percentile=percentile)
    elif method == "rfe": selector = RFE(estimator, n_features_to_select=k)
    elif method == "rfecv": selector = RFECV(estimator, cv=folds, scoring=resolved_scoring, n_jobs=1, min_features_to_select=1)
    elif method in {"forward", "backward"}: selector = SequentialFeatureSelector(
        estimator, n_features_to_select=k, direction=method, cv=folds, scoring=resolved_scoring, n_jobs=1)
    elif method == "l1":
        model = (LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=1000, random_state=random_state)
                 if classification else Lasso(alpha=.01, max_iter=1000, random_state=random_state))
        selector = SelectFromModel(model, threshold=-np.inf, max_features=k)
    elif method == "elastic_net":
        model = (LogisticRegression(l1_ratio=.5, solver="saga", max_iter=1000, random_state=random_state)
                 if classification else ElasticNet(l1_ratio=.5, alpha=.01, random_state=random_state))
        selector = SelectFromModel(model, threshold=-np.inf, max_features=k)
    else: selector = SelectFromModel(forest, threshold=-np.inf, max_features=k)
    start = time.perf_counter()
    selector.fit(X, y) if y is not None else selector.fit(X)
    mask = selector.get_support()
    if not mask.any(): raise ValueError("La méthode n'a retenu aucune feature ; ajustez les paramètres.")
    scores = getattr(selector, "scores_", None)
    if scores is None and hasattr(selector, "estimator_"):
        fitted = selector.estimator_
        scores = getattr(fitted, "feature_importances_", None)
        if scores is None:
            weights = getattr(fitted, "coef_", None)
            if weights is not None and np.size(weights) % X.shape[1] == 0:
                scores = np.abs(weights).reshape(-1, X.shape[1]).mean(axis=0)
    ranks = getattr(selector, "ranking_", None)
    info = {"method": method, "original_count": len(frame.columns), "selected_count": int(mask.sum()),
            "selected": [str(c) for c,m in zip(frame.columns, mask) if m],
            "rejected": [str(c) for c,m in zip(frame.columns, mask) if not m],
            "scores": {str(c): (float(v) if np.isfinite(v) else None) for c,v in zip(frame.columns, scores)} if scores is not None and len(scores)==len(frame.columns) else None,
            "rankings": {str(c): int(v) for c,v in zip(frame.columns, ranks)} if ranks is not None else None,
            "elapsed_seconds": round(time.perf_counter()-start,3), "fit_rows": len(frame),
            "scoring": resolved_scoring if method in {"rfecv", "forward", "backward"} else None,
            "cv_preprocessing": "preprocessed_full_train" if method in {"rfecv", "forward", "backward"} else None,
            "reasons": {str(c): ("Retenue par le sélecteur ajusté sur le train" if m else
                                   "Écartée par le sélecteur ajusté sur le train")
                        for c, m in zip(frame.columns, mask)}}
    return selector, info
