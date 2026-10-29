"""Verifie si un dataset contient du signal et si AutoPreprocessor aide.

Usage :  py signal_check.py fichier.csv colonne_cible [--sep ;]
Compare, en validation croisee stratifiee (preprocesseur refit a chaque pli) :
  - baseline sklearn standard (imputation + OneHot + scaler)
  - AutoPreprocessor
  - AutoPreprocessor avec cible melangee (doit donner ~0.5 : sinon fuite)
Metriques : AUC et PR-AUC (average precision), adaptees aux classes desequilibrees.
"""
from __future__ import annotations
import argparse, warnings
# pyrefly: ignore[missing-import]
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from auto_preprocessor import AutoPreprocessor

warnings.filterwarnings("ignore")


def baseline(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
                            OneHotEncoder(handle_unknown="ignore", max_categories=20)), cat),
    ])


def cv_scores(X, y, make_prep, model_fn, seed=0):
    aucs, aps = [], []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p = make_prep(X.iloc[a])
        Xa, Xb = p.fit_transform(X.iloc[a], y[a]), p.transform(X.iloc[b])
        Xa = np.nan_to_num(np.asarray(Xa.todense() if hasattr(Xa, "todense") else Xa, float))
        Xb = np.nan_to_num(np.asarray(Xb.todense() if hasattr(Xb, "todense") else Xb, float))
        m = model_fn().fit(Xa, y[a])
        s = m.predict_proba(Xb)[:, 1]
        aucs.append(roc_auc_score(y[b], s)); aps.append(average_precision_score(y[b], s))
    return np.mean(aucs), np.std(aucs), np.mean(aps)


class _Auto:
    def __init__(self): self.p = AutoPreprocessor(task="classification"); self.p.config.verbose = 0
    def fit_transform(self, X, y): return self.p.fit_transform(X, y)
    def transform(self, X): return self.p.transform(X)


class _Base:
    def __init__(self, X): self.p = baseline(X)
    def fit_transform(self, X, y): return self.p.fit_transform(X, y)
    def transform(self, X): return self.p.transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("target"); ap.add_argument("--sep", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, sep=a.sep, engine="python" if a.sep is None else "c")
    y = pd.factorize(df.pop(a.target))[0]
    print(f"{df.shape[0]} lignes, {df.shape[1]} features, classe positive = {y.mean():.2%}")
    print(f"Hasard : AUC = 0.50, PR-AUC = {y.mean():.3f}\n")
    models = {
        "RandomForest": lambda: RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1),
        "LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    }
    rng = np.random.RandomState(0); y_shuf = rng.permutation(y)
    print(f"{'modele':13s}{'preproc':22s}{'AUC':>14s}{'PR-AUC':>9s}")
    for mn, mf in models.items():
        for name, prep, yy in [("sklearn standard", lambda X: _Base(X), y),
                               ("AutoPreprocessor", lambda X: _Auto(), y),
                               ("Auto, cible melangee", lambda X: _Auto(), y_shuf)]:
            auc, sd, ap_ = cv_scores(df, yy, prep, mf)
            print(f"{mn:13s}{name:22s}{auc:8.3f} ±{sd:.3f}{ap_:9.3f}")
    print("\nLecture : si l'AUC reel est proche de 0.50, le dataset n'a pas de signal exploitable.\n"
          "Si la ligne 'cible melangee' depasse ~0.55, il y a une fuite a investiguer.")


if __name__ == "__main__":
    main()