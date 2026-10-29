"""
Étude d'Ablation sur le Dataset credit-g pour AutoPreprocessor.

Objectif :
Identifier précisément quelle étape de prétraitement ou quel choix de modélisation
explique l'écart de performance d'AutoPreprocessor par rapport au pipeline sklearn standard (D)
sur le dataset credit-g.

Protocole :
- Dataset : credit-g (OpenML, version 1, 1000 lignes)
- Validation croisée : StratifiedKFold (5 plis, shuffle=True, random_state=42)
- Modèles évalués :
    1. RandomForestClassifier(n_estimators=100, random_state=42)
    2. LogisticRegression(max_iter=1000, random_state=42)
- Métrique : ROC-AUC (moyenne ± écart-type sur les 5 plis)
"""

from __future__ import annotations

import copy
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from auto_preprocessor import AutoPreprocessor, PreprocessConfig
from benchmark_multi import NaiveBaselinePreprocessor, SklearnStandardPreprocessor
from typing import Dict, Tuple


def evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor_factory,
    cv: StratifiedKFold,
    desc: str = "",
) -> Dict[str, Tuple[float, float]]:
    """Évalue un préprocesseur sur les 5 plis avec RandomForest et LogisticRegression."""
    y_encoded = LabelEncoder().fit_transform(y.astype(str))
    rf_aucs = []
    lr_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded)):
        X_tr = X.iloc[train_idx].copy()
        y_tr = y_encoded[train_idx]
        X_vl = X.iloc[val_idx].copy()
        y_vl = y_encoded[val_idx]

        prep = preprocessor_factory()

        # Fit et transform équitables
        if hasattr(prep, "fit_transform"):
            X_tr_trans = prep.fit_transform(X_tr, y_tr)
        else:
            prep.fit(X_tr, y_tr)
            X_tr_trans = prep.transform(X_tr)

        X_vl_trans = prep.transform(X_vl)

        # Convertir en ndarray si DataFrame
        if isinstance(X_tr_trans, pd.DataFrame):
            X_tr_mat = X_tr_trans.to_numpy()
            X_vl_mat = X_vl_trans.to_numpy()
        else:
            X_tr_mat = X_tr_trans
            X_vl_mat = X_vl_trans

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr_mat, y_tr)
        rf_proba = rf.predict_proba(X_vl_mat)[:, 1]
        rf_aucs.append(roc_auc_score(y_vl, rf_proba))

        # 2. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_mat, y_tr)
        lr_proba = lr.predict_proba(X_vl_mat)[:, 1]
        lr_aucs.append(roc_auc_score(y_vl, lr_proba))

    return {
        "RandomForest": (float(np.mean(rf_aucs)), float(np.std(rf_aucs))),
        "LogisticRegression": (float(np.mean(lr_aucs)), float(np.std(lr_aucs))),
    }


def run_ablation_study():
    print("=" * 80)
    print(" ÉTUDE D'ABLATION DÉTAILLÉE SUR LE DATASET CREDIT-G")
    print("=" * 80)

    print("Chargement de credit-g depuis OpenML (v1)...")
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.dropna(subset=["class"]).reset_index(drop=True)
    X = df.drop(columns=["class"])
    y = df["class"]
    print(f"Dataset chargé : {X.shape[0]} lignes, {X.shape[1]} colonnes.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Références de base
    print("\n[1/3] Calcul des références (A et D)...")
    res_A = evaluate_pipeline(X, y, lambda: NaiveBaselinePreprocessor(), cv, "Baseline Naïve (A)")
    res_D = evaluate_pipeline(X, y, lambda: SklearnStandardPreprocessor(), cv, "Sklearn Standard (D)")
    res_B = evaluate_pipeline(
        X, y,
        lambda: AutoPreprocessor(PreprocessConfig(verbose=0)),
        cv, "AutoPreprocessor Défaut (B)"
    )

    ablation_experiments = [
        ("Référence A (Baseline Naïve)", lambda: NaiveBaselinePreprocessor()),
        ("Référence D (Pipeline Sklearn Standard)", lambda: SklearnStandardPreprocessor()),
        ("Référence B (AutoPrep Défaut)", lambda: AutoPreprocessor(PreprocessConfig(verbose=0))),
        (
            "1. Sans Winsorisation (outlier_method='none')",
            lambda: AutoPreprocessor(PreprocessConfig(outlier_method="none", verbose=0))
        ),
        (
            "2. Sans Correction d'Asymétrie (skew_threshold=999.0)",
            lambda: AutoPreprocessor(PreprocessConfig(skew_threshold=999.0, verbose=0))
        ),
        (
            "3. Sans Scaling (scaler_type='none')",
            lambda: AutoPreprocessor(PreprocessConfig(scaler_type="none", verbose=0))
        ),
        (
            "4. Sans Regroupement Rares (rare_category_threshold=0.0)",
            lambda: AutoPreprocessor(PreprocessConfig(rare_category_threshold=0.0, verbose=0))
        ),
        (
            "5. Sans MissingIndicator (missing_indicator_threshold=1.5)",
            lambda: AutoPreprocessor(PreprocessConfig(missing_indicator_threshold=1.5, verbose=0))
        ),
        (
            "6. Sans Filtre de Colinéarité (correlation_threshold=1.01)",
            lambda: AutoPreprocessor(PreprocessConfig(correlation_threshold=1.01, verbose=0))
        ),
    ]

    # Test d'ablation personnalisé pour les entiers discrets
    # Dans credit-g, installment_commitment, residence_since, existing_credits, num_dependents
    # sont des entiers (1, 2, 3, 4) qui ont n_unique <= 10.
    class AutoPrepNumericDiscreteContinuous(AutoPreprocessor):
        """Sous-classe de test : force les entiers à rester dans les numériques continus."""
        def _detect_types(self, X: pd.DataFrame) -> None:
            super()._detect_types(X)
            # Déplacer les NUMERIC_DISCRETE de low_card_cols_ vers num_cols_
            for col in list(self.low_card_cols_):
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.low_card_cols_.remove(col)
                    self.num_cols_.append(col)
                    self.column_metadata_[col]["type"] = "NUMERIC_CONTINUOUS"
                    self.column_metadata_[col]["reason"] = "Forcé en continu pour test d'ablation"

    ablation_experiments.append((
        "7. Entiers Discrets traités comme Continus (non One-Hot)",
        lambda: AutoPrepNumericDiscreteContinuous(PreprocessConfig(verbose=0))
    ))

    # Test d'ablation pour la gestion de 'none' comme modalité légitime
    class AutoPrepKeepNoneString(AutoPreprocessor):
        """Sous-classe de test : ne remplace pas 'none' par NaN."""
        def _clean_string_and_na(self, df: pd.DataFrame) -> pd.DataFrame:
            df_out = df.copy()
            na_set = {"?", "n/a", "na", "null", "", "-", "unknown", "nan", "nil"}
            for col in df_out.columns:
                if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]) or isinstance(df_out[col].dtype, pd.CategoricalDtype):
                    s = df_out[col].astype(str).str.strip()
                    is_missing = s.str.lower().isin(na_set) | df_out[col].isna()
                    df_out[col] = s.mask(is_missing, np.nan)
            return df_out

    ablation_experiments.append((
        "8. Préserver 'none' comme modalité légitime (non NaN)",
        lambda: AutoPrepKeepNoneString(PreprocessConfig(verbose=0))
    ))

    # Test combiné : Préserver 'none' + Entiers discrets continus
    class AutoPrepOptimizedCredit(AutoPrepKeepNoneString, AutoPrepNumericDiscreteContinuous):
        pass

    ablation_experiments.append((
        "9. Combiné : Préserver 'none' + Entiers discrets continus",
        lambda: AutoPrepOptimizedCredit(PreprocessConfig(verbose=0))
    ))

    # Test : Toutes les options désactivées (doit converger vers D)
    class AutoPrepAllDisabled(AutoPreprocessor):
        def _clean_string_and_na(self, df):
            return df
        def _detect_types(self, X):
            super()._detect_types(X)
            # Pas de colinéarité, pas de drop constant abusif
            self.columns_to_drop_ = []
            for col in list(self.low_card_cols_):
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.low_card_cols_.remove(col)
                    self.num_cols_.append(col)

    ablation_experiments.append((
        "10. Toutes options avancées désactivées (proche de D)",
        lambda: AutoPrepAllDisabled(PreprocessConfig(
            outlier_method="none",
            skew_threshold=999.0,
            scaler_type="standard",
            rare_category_threshold=0.0,
            missing_indicator_threshold=1.5,
            correlation_threshold=1.01,
            clean_string_values=False,
            verbose=0,
        ))
    ))

    print("\n[2/3] Exécution des expériences d'ablation...")
    results_table = []
    d_rf_mean, _ = res_D["RandomForest"]
    d_lr_mean, _ = res_D["LogisticRegression"]
    b_rf_mean, _ = res_B["RandomForest"]
    b_lr_mean, _ = res_B["LogisticRegression"]

    for name, factory in ablation_experiments:
        print(f"  -> Évaluation : {name}...")
        res = evaluate_pipeline(X, y, factory, cv, name)
        rf_mean, rf_std = res["RandomForest"]
        lr_mean, lr_std = res["LogisticRegression"]

        results_table.append({
            "Variante": name,
            "RF ROC-AUC": f"{rf_mean:.4f} ± {rf_std:.4f}",
            "RF vs D": f"{rf_mean - d_rf_mean:+.4f}",
            "RF vs B": f"{rf_mean - b_rf_mean:+.4f}",
            "LR ROC-AUC": f"{lr_mean:.4f} ± {lr_std:.4f}",
            "LR vs D": f"{lr_mean - d_lr_mean:+.4f}",
            "LR vs B": f"{lr_mean - b_lr_mean:+.4f}",
        })

    df_report = pd.DataFrame(results_table)
    print("\n" + "=" * 95)
    print(" TABLEAU COMPARATIF COMPLET DE L'ÉTUDE D'ABLATION (CREDIT-G)")
    print("=" * 95)
    print(df_report.to_string(index=False))

    return df_report


if __name__ == "__main__":
    run_ablation_study()
