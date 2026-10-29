"""Isole la cause de l'amélioration de credit_g : 'none' traité comme manquant, ou entiers discrets en One-Hot ?

Lancer :  py credit_g_check.py
Compare 5 variantes (même split, RF + LogisticRegression, ROC-AUC moyenne ± écart-type) :
  D            pipeline sklearn standard (référence)
  actuelle     ton auto_preprocessor.py tel quel ('none' = catégorie valide, entiers discrets numériques)
  +none_NA     'none' redevient une valeur manquante (ancien comportement)
  +discret_OHE entiers discrets en One-Hot (ancien comportement)
  ancienne     les deux anciens comportements (doit retrouver ~0.76 en RF)
"""
import sys
import warnings
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import auto_preprocessor as ap
from auto_preprocessor import AutoPreprocessor, PreprocessConfig
import benchmark_multi as bm

warnings.filterwarnings("ignore")
ORIGINAL_NA = set(ap.DEFAULT_NA_VALUES)


def make_fake(n: int = 600, seed: int = 0):
    """Jeu synthétique avec une modalité 'none' légitime (utilisé seulement avec --fake)."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    other = np.where(rng.rand(n) < 0.5 + 0.3 * (y == 1), "none", "guarantor")
    X = pd.DataFrame({"other_parties": other, "dur": rng.randint(4, 72, n),
                      "commit": rng.randint(1, 5, n), "amount": rng.exponential(3000, n)})
    return X, pd.Series(np.where(y == 1, "good", "bad"))


def run(label, X, y, make_prep, use_none_as_na, fit_transform_train=True):
    ap.DEFAULT_NA_VALUES.clear()
    ap.DEFAULT_NA_VALUES.update(ORIGINAL_NA | ({"none"} if use_none_as_na else set()))
    y_enc = pd.Series(pd.factorize(y.astype(str), sort=True)[0], index=y.index)
    res = {"RF": [], "LR": []}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_enc):
        Xt, Xv, yt, yv = X.iloc[tr], X.iloc[va], y_enc.iloc[tr], y_enc.iloc[va]
        prep = make_prep()
        if isinstance(prep, AutoPreprocessor):
            A, B = prep.fit_transform(Xt, yt), prep.transform(Xv)
        else:
            A, B = prep.fit(Xt).transform(Xt), prep.transform(Xv)
        A, B = np.asarray(A), np.asarray(B)
        for name, model in (("RF", RandomForestClassifier(100, random_state=42)),
                            ("LR", LogisticRegression(max_iter=2000))):
            model.fit(A, yt)
            res[name].append(roc_auc_score(yv, model.predict_proba(B)[:, 1]))
    print(f"{label:14s} RF {np.mean(res['RF']):.4f} ± {np.std(res['RF']):.4f}   "
          f"LR {np.mean(res['LR']):.4f} ± {np.std(res['LR']):.4f}")


def main():
    if "--fake" in sys.argv:
        X, y = make_fake()
    else:
        X, y = bm.load_dataset(bm.DATASETS_REGISTRY["credit_g"])
    print(f"credit_g : X={X.shape}, ROC-AUC (5 plis, random_state=42)\n")

    def cfg(discrete_cat):
        return lambda: AutoPreprocessor(PreprocessConfig(verbose=0, encode_discrete_as_categorical=discrete_cat),
                                        task="classification")
    run("D sklearn", X, y, bm.SklearnStandardPreprocessor, False)
    run("actuelle", X, y, cfg(False), False)
    run("+none_NA", X, y, cfg(False), True)
    run("+discret_OHE", X, y, cfg(True), False)
    run("ancienne", X, y, cfg(True), True)
    ap.DEFAULT_NA_VALUES.clear()
    ap.DEFAULT_NA_VALUES.update(ORIGINAL_NA)


if __name__ == "__main__":
    main()