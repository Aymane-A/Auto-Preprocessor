"""Fold-isolated wrapper selection on raw training rows.

Preprocessing is fitted once per inner training fold and never sees its validation
rows. RFECV refits RFE inside each fold; sequential search builds its candidate
space separately in each inner training fold and validates the selected subset.
A plain
sklearn Pipeline over the globally encoded matrix cannot re-learn raw categorical
vocabularies or imputation inside a fold, while a pipeline over raw columns cannot
index the generated feature candidates. Caching fold-local preprocessors is the
equivalent fit/transform boundary without repeating expensive preprocessing for
every candidate subset.
"""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold


class FoldIsolationUnavailable(ValueError):
    """An input operation cannot preserve the rows required for fold scoring."""


class FixedSupportSelector(BaseEstimator, TransformerMixin):
    """Picklable support mask for the final encoded feature order."""

    def __init__(self, support):
        self.support = tuple(bool(value) for value in support)

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.support)
        return self

    def get_support(self, indices=False):
        mask = np.asarray(self.support, dtype=bool)
        return np.flatnonzero(mask) if indices else mask

    def transform(self, X):
        return X[:, self.get_support()] if not isinstance(X, pd.DataFrame) else X.iloc[:, self.get_support()]


def _folds(raw_frame, raw_target, config, task):
    """Cache only transformations learned independently in each CV fold."""
    from auto_preprocessor import AutoPreprocessor

    if config.preserve_rows is False and any(
        value == "DROP_ROWS" or (isinstance(value, dict) and "DROP_ROWS" in str(value))
        for value in config.column_overrides.values()
    ):
        raise FoldIsolationUnavailable("Une règle DROP_ROWS empêche une validation interne à lignes constantes.")
    fold_config = replace(config, selection_method="auto", selection_include=None,
                          selection_exclude=None, preserve_rows=True)
    y = np.asarray(raw_target)
    splitter = (StratifiedKFold(n_splits=config.selection_folds, shuffle=True,
                               random_state=config.random_state) if task == "classification" else
                KFold(n_splits=config.selection_folds, shuffle=True,
                      random_state=config.random_state))
    prepared = []
    for train_idx, valid_idx in splitter.split(raw_frame, y if task == "classification" else None):
        prep = AutoPreprocessor(fold_config, task=task)
        train_raw, valid_raw = raw_frame.iloc[train_idx], raw_frame.iloc[valid_idx]
        try:
            train_encoded = prep.fit_transform(train_raw, y[train_idx])
        except ValueError as exc:
            if "preserve_rows" in str(exc):
                raise FoldIsolationUnavailable(str(exc)) from exc
            raise
        valid_encoded = prep.transform(valid_raw)
        if len(train_encoded) != len(train_idx) or len(valid_encoded) != len(valid_idx):
            raise FoldIsolationUnavailable("Une opération supprime des lignes dans un pli interne.")
        if not train_encoded.shape[1]:
            raise FoldIsolationUnavailable("Un pli interne ne contient aucune feature après prétraitement.")
        prepared.append((train_encoded, valid_encoded, y[train_idx], y[valid_idx],
                         train_raw.index.tolist(), valid_raw.index.tolist()))
    return prepared


def _score(estimator, X_train, y_train, X_valid, y_valid, scoring):
    model = clone(estimator).fit(X_train, y_train)
    value = (get_scorer(scoring)(model, X_valid, y_valid) if scoring else
             model.score(X_valid, y_valid))
    if not np.isfinite(value):
        raise ValueError("Score non fini pendant la validation interne du sélecteur.")
    return float(value)


def _greedy_on_train(estimator, train, target, method, count, scoring):
    """Choose a subset using this training partition and no validation rows."""
    names = list(train.columns)
    selected = set() if method == "forward" else set(names)
    changes = []
    while len(selected) != count:
        choices = (name for name in names if name not in selected) if method == "forward" else (
            name for name in names if name in selected)
        evaluations = []
        for candidate in choices:
            subset = selected | {candidate} if method == "forward" else selected - {candidate}
            columns = [name for name in names if name in subset]
            value = _score(estimator, train[columns], target, train[columns], target, scoring)
            evaluations.append((value, candidate))
        if not evaluations:
            raise FoldIsolationUnavailable("Aucune feature candidate dans un sous-train.")
        chosen = max(evaluations, key=lambda item: item[0])[1]
        selected = selected | {chosen} if method == "forward" else selected - {chosen}
        changes.append(chosen)
    return [name for name in names if name in selected], changes


def fit_fold_safe_selector(encoded_frame, raw_frame, raw_target, *, method,
                           config, task):
    """Choose encoded features with fold-local preprocessing and return final support."""
    if method not in {"rfecv", "forward", "backward"}:
        raise ValueError("Méthode sans validation croisée interne.")
    if len(encoded_frame) > 20_000 or encoded_frame.shape[1] > 200:
        raise ValueError("Les méthodes wrapper exigent au plus 20 000 lignes et 200 features.")
    if len(encoded_frame) * encoded_frame.shape[1] * 8 > config.max_dense_bytes:
        raise ValueError("La sélection dépasserait le budget mémoire dense.")
    if encoded_frame.shape[1] < 2:
        raise ValueError("La validation interne exige au moins deux features.")
    X_raw = raw_frame if isinstance(raw_frame, pd.DataFrame) else pd.DataFrame(raw_frame)
    if len(X_raw) != len(raw_target):
        raise ValueError("Les lignes brutes et la cible ne sont pas alignées pour la validation interne.")
    if len(X_raw) != len(encoded_frame):
        raise FoldIsolationUnavailable("Le nettoyage global a supprimé des lignes avant la sélection.")
    started = time.perf_counter()
    original = np.asarray(raw_target)
    classification = task == "classification"
    scoring = config.selection_scoring
    if scoring is not None:
        allowed = {"accuracy", "f1", "roc_auc"} if classification else {"r2", "neg_mean_squared_error"}
        if scoring not in allowed:
            raise ValueError(f"La métrique {scoring} est incompatible avec la tâche {task}.")
    if classification and len(np.unique(original)) > 2:
        scoring = {"f1": "f1_macro", "roc_auc": "roc_auc_ovr"}.get(scoring, scoring)
    estimator = (LogisticRegression(max_iter=500, random_state=config.random_state) if classification else
                 Lasso(alpha=.01, max_iter=1000, random_state=config.random_state))
    folds = _folds(X_raw, original, config, task)
    names = list(encoded_frame.columns)
    # Diagnostic union only. Sequential search below constructs its candidate
    # space independently in each fold, without consulting this union.
    candidates = list(dict.fromkeys(name for fold in folds for name in fold[0].columns))
    if len(candidates) > 200:
        raise ValueError("La sélection séquentielle exige au plus 200 candidates générées dans les plis.")
    if method in {"forward", "backward"} and min(len(fold[0].columns) for fold in folds) < 2:
        raise FoldIsolationUnavailable("Moins de deux features générées dans les sous-trains des plis.")
    score_history = []

    if method == "rfecv":
        max_count = min(len(fold[0].columns) for fold in folds)
        for count in range(1, max_count + 1):
            scores = []
            for train, valid, y_train, y_valid, _, _ in folds:
                selector = RFE(clone(estimator), n_features_to_select=count).fit(train, y_train)
                scores.append(_score(estimator, selector.transform(train), y_train,
                                     selector.transform(valid), y_valid, scoring))
            score_history.append({"count": count, "mean": float(np.mean(scores)), "scores": scores})
        best_count = max(score_history, key=lambda item: (item["mean"], -item["count"]))["count"]
        final_count = min(best_count, len(names))
        final = RFE(clone(estimator), n_features_to_select=final_count).fit(encoded_frame, original)
        support = final.get_support()
        ranks = {name: int(rank) for name, rank in zip(names, final.ranking_)}
    else:
        minimum_width = min(len(fold[0].columns) for fold in folds)
        target_count = config.selection_count if config.selection_count is not None else min(20, minimum_width - 1)
        if target_count >= minimum_width:
            raise ValueError("La sélection séquentielle exige moins de features que chaque sous-train ne génère.")
        fold_scores = []
        fold_selected = []
        for train, valid, y_train, y_valid, _, _ in folds:
            local_names, _ = _greedy_on_train(estimator, train, y_train, method, target_count, scoring)
            fold_selected.append(local_names)
            fold_scores.append(_score(estimator, train[local_names], y_train,
                                      valid[local_names], y_valid, scoring))
        score_history.append({"count": target_count, "mean": float(np.mean(fold_scores)),
                              "scores": fold_scores, "selected_by_fold": fold_selected})
        final_count = min(target_count, len(names))
        final_names, _ = _greedy_on_train(estimator, encoded_frame, original,
                                          method, final_count, scoring)
        support = np.asarray([name in final_names for name in names])
        ranks = None
    selector = FixedSupportSelector(support).fit(encoded_frame, original)
    cv_count = best_count if method == "rfecv" else target_count
    info = {"method": method, "original_count": len(names), "selected_count": int(support.sum()),
            "cv_selected_count": cv_count,
            "final_count_adjusted": int(support.sum()) != cv_count,
            "selected": [name for name, keep in zip(names, support) if keep],
            "rejected": [name for name, keep in zip(names, support) if not keep],
            "scores": None, "rankings": ranks, "elapsed_seconds": round(time.perf_counter() - started, 3),
            "fit_rows": len(encoded_frame), "scoring": scoring,
            "cv_preprocessing": "fold_isolated", "cv_scores": score_history,
            "inner_candidate_names": candidates,
            "missing_final_candidates": [name for name in candidates if name not in names],
            "selected_unavailable_at_final_fit": [],
            "candidate_names_by_fold": [list(fold[0].columns) for fold in folds],
            "cv_folds": [{"fit_rows": len(fold[4]), "validation_rows": len(fold[5]),
                          "generated_count": len(fold[0].columns),
                          "missing_final_features": [name for name in names if name not in fold[0].columns]}
                         for fold in folds],
            "reasons": {name: ("Retenue après validation croisée avec prétraitement par pli" if keep else
                               "Écartée après validation croisée avec prétraitement par pli")
                        for name, keep in zip(names, support)}}
    return selector, info
