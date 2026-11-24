"""Train-only class rebalancing on already-fitted numeric features."""
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _discrete_outputs(frame):
    """Find exact 0/1 outputs and complete one-hot groups on the original train."""
    binary = [name for name in frame if pd.api.types.is_numeric_dtype(frame[name])
              and frame[name].notna().all() and frame[name].isin([0, 1]).all()]
    candidates = {}
    for name in binary:
        for index, char in enumerate(str(name)):
            if char == "_":
                candidates.setdefault(str(name)[:index + 1], []).append(name)
    groups = []
    assigned = set()
    for _, names in sorted(candidates.items(), key=lambda item: -len(item[0])):
        if len(names) < 2 or assigned.intersection(names):
            continue
        if frame[names].sum(axis=1).eq(1).all():
            groups.append(names)
            assigned.update(names)
    return binary, groups


def _repair_synthetic(rows, binary, groups):
    """Repair only generated rows; preserve the original train byte-for-byte."""
    winners = {tuple(names): rows[names].to_numpy(dtype=float).argmax(axis=1) for names in groups}
    for name in binary:
        rows[name] = (rows[name].astype(float) >= .5).astype(int)
    for names in groups:
        rows.loc[:, names] = (np.arange(len(names))[None, :] == winners[tuple(names)][:, None]).astype(int)


def class_profile(values):
    counts = pd.Series(values).value_counts(dropna=False)
    if not len(counts):
        return {"counts": {}, "minority_ratio": None, "imbalance_ratio": None, "level": None}
    minimum, maximum = int(counts.min()), int(counts.max())
    ratio = minimum / int(counts.sum())
    return {"counts": {str(k): int(v) for k, v in counts.items()},
            "minority_ratio": ratio, "imbalance_ratio": maximum / max(minimum, 1),
            "level": "sévère" if ratio < .05 else "modéré" if ratio < .20 else None}


def rebalance(features, labels, method="none", strategy=.5, neighbors=5, seed=42):
    """Return new train data, labels, and transparent metadata; test is never passed in."""
    start = time.perf_counter()
    x = pd.DataFrame(features).reset_index(drop=True)
    y = pd.Series(labels).reset_index(drop=True)
    before = class_profile(y)
    result = {"method": method, "requested_method": method, "categorical_columns": [], "before": before, "after": before,
              "synthetic_rows": 0, "duplicated_rows": 0, "sampled_rows": 0, "warnings": [], "class_weight": None,
              "k_neighbors_used": None, "seconds": 0.}
    if method not in {"none", "class_weight", "random_over", "smote", "smotenc", "random_under"}:
        raise ValueError("Méthode de rééquilibrage inconnue.")
    if method == "none":
        return x, y, result
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("Le rééquilibrage sélectionné exige une cible binaire ; utilisez Aucun pour plusieurs classes.")
    minority, majority = counts.idxmin(), counts.idxmax()
    nmin, nmaj = int(counts[minority]), int(counts[majority])
    if nmin < 2 and method in {"smote", "smotenc"}:
        raise ValueError("Au moins deux exemples minoritaires dans le train sont requis pour SMOTE.")
    if method == "class_weight":
        result["class_weight"] = {str(k): len(y)/(2*int(v)) for k,v in counts.items()}
    else:
        if not 0 < strategy <= 1:
            raise ValueError("Le ratio minoritaire visé doit être dans ]0, 1].")
        rng = np.random.default_rng(seed)
        minority_idx = np.flatnonzero(y.eq(minority))
        majority_idx = np.flatnonzero(y.eq(majority))
        if method == "random_under":
            keep = min(nmaj, int(np.floor(nmin / strategy)))
            indices = np.concatenate([minority_idx, rng.choice(majority_idx, keep, replace=False)])
            x, y = x.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
            result["sampled_rows"] = nmaj - keep
        else:
            required = max(0, int(np.ceil(nmaj * strategy)) - nmin)
            if required:
                if method == "random_over":
                    rows = x.iloc[rng.choice(minority_idx, required, replace=True)].reset_index(drop=True)
                    result["duplicated_rows"] = required
                else:
                    binary, onehot_groups = _discrete_outputs(x)
                    categorical = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
                    continuous = [c for c in x if c not in categorical and c not in binary]
                    if method in {"smote", "smotenc"} and (categorical or binary) and (continuous or categorical):
                        categorical = list(dict.fromkeys(categorical + binary))
                        result["method"] = "SMOTENC"
                    elif binary:
                        result["method"] = "SMOTE + rounding"
                    else:
                        result["method"] = "SMOTENC" if method == "smotenc" else "SMOTE"
                    result["categorical_columns"] = list(categorical or binary)
                    if method == "smotenc" and not categorical and not binary:
                        raise ValueError("SMOTENC exige au moins une variable catégorielle non encodée.")
                    numeric_columns = [c for c in x if c not in categorical]
                    for col in numeric_columns:
                        if not np.isfinite(pd.to_numeric(x[col], errors="raise").to_numpy(dtype=float)).all():
                            raise ValueError("SMOTE exige des variables numériques finies après prétraitement.")
                    k = min(neighbors, nmin - 1)
                    if k < 1:
                        raise ValueError("SMOTE exige au moins deux exemples minoritaires.")
                    result["k_neighbors_used"] = k
                    if k < neighbors:
                        result["warnings"].append(f"Voisins SMOTE réduits de {neighbors} à {k} : effectif minoritaire insuffisant.")
                    minority_x = x.iloc[minority_idx][numeric_columns].to_numpy(dtype=float)
                    if categorical:
                        codes = pd.get_dummies(x.iloc[minority_idx][categorical].fillna("__MISSING__").astype(str),dtype=float).to_numpy()
                        distance_x = np.column_stack([minority_x,codes])
                    else:
                        distance_x = minority_x
                    nearest = NearestNeighbors(n_neighbors=k+1).fit(distance_x)
                    anchors = rng.integers(0, nmin, required)
                    neighborhood = nearest.kneighbors(distance_x, return_distance=False)[:, 1:]
                    partners = neighborhood[anchors, rng.integers(0, k, required)]
                    alpha = rng.random(required)[:, None]
                    rows = pd.DataFrame(minority_x[anchors] + alpha*(minority_x[partners]-minority_x[anchors]), columns=numeric_columns)
                    if categorical:
                        take_partner = rng.random(required) < .5
                        source = minority_idx[np.where(take_partner,partners,anchors)]
                        for col in categorical:
                            rows[col] = x.iloc[source][col].to_numpy()
                        rows = rows[x.columns]
                    elif binary:
                        _repair_synthetic(rows, binary, onehot_groups)
                    result["synthetic_rows"] = required
                x = pd.concat([x, rows], ignore_index=True)
                y = pd.concat([y, pd.Series([minority]*required)], ignore_index=True)
    result["after"] = class_profile(y)
    result["seconds"] = time.perf_counter()-start
    return x, y, result
