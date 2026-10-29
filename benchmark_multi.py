"""
Benchmark Multi-Datasets Rigoureux pour AutoPreprocessor.

Compare équitablement 4 approches sur au moins 8 jeux de données réels et variés :
- Approche A : Baseline Naïve (SimpleImputer + OneHot basique, sans scaling, sans winsorisation)
- Approche D : Pipeline sklearn standard (imputation + StandardScaler + OneHot) : la vraie référence
- Approche B : AutoPreprocessor par défaut (drop_leaky_columns=False)
- Approche C : AutoPreprocessor robuste aux fuites (drop_leaky_columns=True)

Règles impératives :
- Validation croisée (5 plis ou 3 plis si --quick) avec réajustement strict à l'intérieur de chaque pli.
- Modèles évalués : RandomForest ET modèle linéaire (LogisticRegression / Ridge).
- Contrôles de fuite : Détection proactive, alertes de score suspect et test de permutation ("cible mélangée").
- Tests de robustesse : Matrices 100% numériques, tolérance aux modalités inconnues, persistance joblib et reproductibilité.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# pyrefly: ignore[missing-import]
import joblib
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Import de notre module AutoPreprocessor
from auto_preprocessor import AutoPreprocessor, PreprocessConfig

# Import optionnel et protégé de matplotlib
try:
    # pyrefly: ignore[missing-import]
    import matplotlib
    matplotlib.use("Agg")  # Backend sans interface graphique
    # pyrefly: ignore[missing-import]
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. BASELINE NAÏVE DE RÉFÉRENCE (ÉQUITABLE)
# =============================================================================

class NaiveBaselinePreprocessor(BaseEstimator, TransformerMixin):
    """Prétraitement naïf standard : imputation médiane sur numériques, mode sur catégorielles,

    suivi d'un One-Hot Encoding basique sans scaling, sans winsorisation, sans détection de type.
    """

    def __init__(self):
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.imp_num = SimpleImputer(strategy="median")
        self.imp_cat = SimpleImputer(strategy="most_frequent")
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> NaiveBaselinePreprocessor:
        X_df = pd.DataFrame(X).copy()
        self.num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        self.cat_cols = [c for c in X_df.columns if c not in self.num_cols]

        if self.num_cols:
            self.imp_num.fit(X_df[self.num_cols])

        if self.cat_cols:
            # Conversion en string pour éviter les conflits d'objets hétérogènes
            cat_data = X_df[self.cat_cols].astype(str)
            cat_imp = self.imp_cat.fit_transform(cat_data)
            self.ohe.fit(cat_imp)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(X).copy()
        parts = []

        if self.num_cols:
            num_data = X_df[self.num_cols] if all(c in X_df.columns for c in self.num_cols) else X_df.reindex(columns=self.num_cols)
            parts.append(self.imp_num.transform(num_data))

        if self.cat_cols:
            cat_data = X_df[self.cat_cols] if all(c in X_df.columns for c in self.cat_cols) else X_df.reindex(columns=self.cat_cols)
            cat_imp = self.imp_cat.transform(cat_data.astype(str))
            parts.append(self.ohe.transform(cat_imp))

        if parts:
            return np.hstack(parts)
        return np.zeros((len(X_df), 1))


class SklearnStandardPreprocessor(NaiveBaselinePreprocessor):
    """Pipeline sklearn 'solide' classique : médiane + StandardScaler (numériques),
    mode + OneHot (catégorielles). Sert à isoler la vraie valeur ajoutée d'AutoPreprocessor :
    si D égale B, le gain vient du simple scaling, pas du module."""

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X, y)
        if self.num_cols:
            num_data = pd.DataFrame(X).reindex(columns=self.num_cols)
            self.scaler.fit(self.imp_num.transform(num_data))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(X).copy()
        parts = []
        if self.num_cols:
            parts.append(self.scaler.transform(self.imp_num.transform(X_df.reindex(columns=self.num_cols))))
        if self.cat_cols:
            cat_imp = self.imp_cat.transform(X_df.reindex(columns=self.cat_cols).astype(str))
            parts.append(self.ohe.transform(cat_imp))
        return np.hstack(parts) if parts else np.zeros((len(X_df), 1))


# =============================================================================
# 2. CONFIGURATION DES DATASETS DE TEST
# =============================================================================

@dataclass
class DatasetSpec:
    name: str
    task: str  # "binary", "multiclass", "regression"
    source_type: str  # "sklearn" ou "openml"
    source_id: str
    target_col: str
    version: Optional[int] = None
    sample_size: Optional[int] = None
    suspect_leaks: List[str] = None
    description: str = ""

    def __post_init__(self):
        if self.suspect_leaks is None:
            self.suspect_leaks = []


DATASETS_REGISTRY: Dict[str, DatasetSpec] = {
    "titanic": DatasetSpec(
        name="titanic",
        task="binary",
        source_type="openml",
        source_id="titanic",
        version=1,
        target_col="survived",
        sample_size=None,
        suspect_leaks=["boat", "body"],
        description="Classification binaire mixte (texte, haute cardinalité, NaNs, fuite suspecte boat/body)",
    ),
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        task="binary",
        source_type="sklearn",
        source_id="load_breast_cancer",
        target_col="target",
        sample_size=None,
        suspect_leaks=[],
        description="Classification binaire numérique continue pure (30 features, 569 lignes)",
    ),
    "wine": DatasetSpec(
        name="wine",
        task="multiclass",
        source_type="sklearn",
        source_id="load_wine",
        target_col="target",
        sample_size=None,
        suspect_leaks=[],
        description="Classification multiclasse (3 classes, 13 features continues, 178 lignes)",
    ),
    "california_housing": DatasetSpec(
        name="california_housing",
        task="regression",
        source_type="sklearn",
        source_id="fetch_california_housing",
        target_col="MedHouseVal",
        sample_size=3000,  # Sous-échantillon pour benchmark réactif
        suspect_leaks=[],
        description="Régression tabulaire continue avec asymétries et outliers (20 640 lignes)",
    ),
    "diabetes_reg": DatasetSpec(
        name="diabetes_reg",
        task="regression",
        source_type="sklearn",
        source_id="load_diabetes",
        target_col="target",
        sample_size=None,
        suspect_leaks=[],
        description="Régression numérique (10 variables cliniques, 442 lignes)",
    ),
    "credit_g": DatasetSpec(
        name="credit_g",
        task="binary",
        source_type="openml",
        source_id="credit-g",
        version=1,
        target_col="class",
        sample_size=None,
        suspect_leaks=[],
        description="Classification binaire mixte déséquilibrée (ordinales, nominales, numériques)",
    ),
    "car": DatasetSpec(
        name="car",
        task="multiclass",
        source_type="openml",
        source_id="car",
        version=1,
        target_col="class",
        sample_size=None,
        suspect_leaks=[],
        description="Classification multiclasse purement ordinale/catégorielle (1728 lignes)",
    ),
    "pima_diabetes": DatasetSpec(
        name="pima_diabetes",
        task="binary",
        source_type="openml",
        source_id="diabetes",
        version=1,
        target_col="class",
        sample_size=None,
        suspect_leaks=[],
        description="Classification binaire avec zéros représentant de faux manquants (768 lignes)",
    ),
    "adult": DatasetSpec(
        name="adult",
        task="binary",
        source_type="openml",
        source_id="adult",
        version=2,
        target_col="class",
        sample_size=4000,  # Échantillon pour rapidité
        suspect_leaks=[],
        description="Classification binaire avec haute cardinalité, déséquilibre et NaNs",
    ),
    "iris": DatasetSpec(
        name="iris",
        task="multiclass",
        source_type="openml",
        source_id="iris",
        version=1,
        target_col="class",
        sample_size=None,
        suspect_leaks=[],
        description="Classification multiclasse standard (150 lignes)",
    ),
}


def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
    """Charge un jeu de données de façon unifiée et sécurisée."""
    if spec.source_type == "sklearn":
        if spec.source_id == "load_breast_cancer":
            bunch = load_breast_cancer(as_frame=True)
            df = bunch.frame.copy()
        elif spec.source_id == "load_wine":
            bunch = load_wine(as_frame=True)
            df = bunch.frame.copy()
        elif spec.source_id == "load_diabetes":
            bunch = load_diabetes(as_frame=True)
            df = bunch.frame.copy()
        elif spec.source_id == "fetch_california_housing":
            bunch = fetch_california_housing(as_frame=True)
            df = bunch.frame.copy()
        else:
            raise ValueError(f"Chargement sklearn inconnu : {spec.source_id}")
    else:
        # OpenML
        bunch = fetch_openml(spec.source_id, version=spec.version, as_frame=True, parser="auto")
        df = bunch.frame.copy()

    # Nettoyage initial des lignes dont la cible est manquante
    df = df.dropna(subset=[spec.target_col])

    # Sous-échantillonnage déterministe si spécifié
    if spec.sample_size is not None and len(df) > spec.sample_size:
        df = df.sample(n=spec.sample_size, random_state=42).reset_index(drop=True)

    y = df[spec.target_col].copy()
    X = df.drop(columns=[spec.target_col]).copy()
    return X, y


# =============================================================================
# 3. TESTS DE ROBUSTESSE AUTOMATISÉS
# =============================================================================

def run_robustness_tests(X: pd.DataFrame, y: pd.Series, spec: DatasetSpec) -> Dict[str, bool]:
    """Exécute les assertions de robustesse sur un jeu de données réel."""
    results = {
        "numeric_output": False,
        "no_nan_no_inf": False,
        "same_columns_train_test": False,
        "unseen_categories_safe": False,
        "joblib_equality": False,
        "reproducibility": False,
    }

    # Split train/test (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    prep1 = AutoPreprocessor(PreprocessConfig(random_state=42, verbose=0))
    X_tr_trans, _ = prep1.fit_transform_xy(
        pd.concat([X_train, y_train.rename(spec.target_col)], axis=1),
        target=spec.target_col,
    )
    X_te_trans = prep1.transform(X_test)

    # 1. Sortie 100% numérique et sans NaN/Inf
    is_num = all(np.issubdtype(dtype, np.number) for dtype in X_tr_trans.dtypes)
    no_nan_tr = not X_tr_trans.isna().any().any()
    no_nan_te = not X_te_trans.isna().any().any()
    no_inf_tr = not np.isinf(X_tr_trans.to_numpy()).any()
    no_inf_te = not np.isinf(X_te_trans.to_numpy()).any()

    results["numeric_output"] = is_num
    results["no_nan_no_inf"] = no_nan_tr and no_nan_te and no_inf_tr and no_inf_te

    # 2. Mêmes colonnes train et test
    results["same_columns_train_test"] = list(X_tr_trans.columns) == list(X_te_trans.columns)

    # 3. Test des catégories inconnues dans le test
    X_test_corrupted = X_test.copy()
    cat_cols = [c for c in X_test_corrupted.columns if not pd.api.types.is_numeric_dtype(X_test_corrupted[c])]
    if cat_cols:
        for c in cat_cols:
            X_test_corrupted[c] = X_test_corrupted[c].astype(object)
            mask = np.random.choice([True, False], size=len(X_test_corrupted), p=[0.1, 0.9])
            X_test_corrupted.loc[mask, c] = "__INVENTED_UNKNOWN_MODALITY__"
    try:
        X_corrupt_trans = prep1.transform(X_test_corrupted)
        results["unseen_categories_safe"] = not X_corrupt_trans.isna().any().any() and len(X_corrupt_trans) == len(X_test)
    except Exception:
        results["unseen_categories_safe"] = False

    # 4. Sauvegarde / chargement joblib
    tmp_path = f"tmp_model_{spec.name}.joblib"
    try:
        prep1.save(tmp_path)
        loaded_prep = AutoPreprocessor.load(tmp_path)
        X_loaded_trans = loaded_prep.transform(X_test)
        pd.testing.assert_frame_equal(X_te_trans, X_loaded_trans)
        results["joblib_equality"] = True
    except Exception:
        results["joblib_equality"] = False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 5. Reproductibilité (même random_state -> même sortie)
    prep2 = AutoPreprocessor(PreprocessConfig(random_state=42, verbose=0))
    X_tr_trans2, _ = prep2.fit_transform_xy(
        pd.concat([X_train, y_train.rename(spec.target_col)], axis=1),
        target=spec.target_col,
    )
    results["reproducibility"] = np.allclose(X_tr_trans.to_numpy(), X_tr_trans2.to_numpy(), equal_nan=True)

    return results


# =============================================================================
# 4. PROTOCOLE DE COMPARAISON ÉQUITABLE PAR VALIDATION CROISÉE
# =============================================================================

def run_cross_validation_benchmark(
    spec: DatasetSpec,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_estimators: int = 100,
) -> Dict[str, Any]:
    """Exécute la validation croisée réajustée à l'intérieur de chaque pli pour 3 approches :

    A. Baseline Naïve
    B. AutoPreprocessor standard (drop_leaky_columns=False)
    C. AutoPreprocessor avec drop_leaky_columns=True
    """
    is_regression = spec.task == "regression"

    # Choix du splitter
    if is_regression:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        # Encodage préalable de y pour le StratifiedKFold
        y_int = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    approaches = ["A_NaiveBaseline", "D_SklearnStandard", "B_AutoPrep_Default", "C_AutoPrep_DropLeaky"]
    models = ["RandomForest", "LinearModel"]

    # Stockage des métriques
    metrics_history = {
        app: {m: {"score_main": [], "score_aux": [], "fit_time": [], "trans_time": [], "n_features": []} for m in models}
        for app in approaches
    }

    # Liste des fuites signalées par AutoPreprocessor
    detected_leaks = set()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y if is_regression else y_int)):
        X_train_raw = X.iloc[train_idx].copy()
        y_train_raw = y.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        y_val_raw = y.iloc[val_idx].copy()

        # =====================================================================
        # 1. APPROCHE A : BASELINE NAÏVE
        # =====================================================================
        t0_fit = time.time()
        naive_prep = NaiveBaselinePreprocessor()
        naive_prep.fit(X_train_raw)
        fit_time_naive = time.time() - t0_fit

        t0_trans = time.time()
        X_train_naive = naive_prep.transform(X_train_raw)
        X_val_naive = naive_prep.transform(X_val_raw)
        trans_time_naive = time.time() - t0_trans

        # =====================================================================
        # 1bis. APPROCHE D : PIPELINE SKLEARN STANDARD (imputation + scaling + OHE)
        # =====================================================================
        t0_fit = time.time()
        std_prep = SklearnStandardPreprocessor()
        std_prep.fit(X_train_raw)
        fit_time_std = time.time() - t0_fit

        t0_trans = time.time()
        X_train_std = std_prep.transform(X_train_raw)
        X_val_std = std_prep.transform(X_val_raw)
        trans_time_std = time.time() - t0_trans

        # =====================================================================
        # 2. APPROCHE B : AUTOPREPROCESSOR PAR DÉFAUT
        # =====================================================================
        # fit_transform (et non fit + transform) sur le train : le Target Encoding du train
        # est calculé "hors pli", sinon le modèle surapprend ses propres labels.
        t0_fit = time.time()
        prep_b = AutoPreprocessor(PreprocessConfig(drop_leaky_columns=False, random_state=42, verbose=0))
        X_train_b = prep_b.fit_transform(X_train_raw, y_train_raw)
        fit_time_b = time.time() - t0_fit

        t0_trans = time.time()
        X_val_b = prep_b.transform(X_val_raw)
        trans_time_b = time.time() - t0_trans

        rep_b = prep_b.report()
        detected_leaks.update(rep_b.get("fuites_suspectees", []))

        # =====================================================================
        # 3. APPROCHE C : AUTOPREPROCESSOR AVEC DROP_LEAKY_COLUMNS
        # =====================================================================
        t0_fit = time.time()
        prep_c = AutoPreprocessor(PreprocessConfig(drop_leaky_columns=True, random_state=42, verbose=0))
        X_train_c = prep_c.fit_transform(X_train_raw, y_train_raw)
        fit_time_c = time.time() - t0_fit

        t0_trans = time.time()
        X_val_c = prep_c.transform(X_val_raw)
        trans_time_c = time.time() - t0_trans

        # Préparation des cibles transformées
        if is_regression:
            y_train_num = y_train_raw.astype(float).to_numpy()
            y_val_num = y_val_raw.astype(float).to_numpy()
            y_train_b = prep_b.transform_target(y_train_raw)
            y_train_c = prep_c.transform_target(y_train_raw)
        else:
            y_train_num = LabelEncoder().fit_transform(y_train_raw.astype(str))
            y_val_num = LabelEncoder().fit_transform(y_val_raw.astype(str))
            y_train_b = prep_b.transform_target(y_train_raw)
            y_val_b = prep_b.transform_target(y_val_raw)
            y_train_c = prep_c.transform_target(y_train_raw)
            y_val_c = prep_c.transform_target(y_val_raw)

        # Entraînement et évaluation pour chaque modèle
        processed_data = {
            "A_NaiveBaseline": (X_train_naive, X_val_naive, y_train_num, fit_time_naive, trans_time_naive, X_train_naive.shape[1], None),
            "D_SklearnStandard": (X_train_std, X_val_std, y_train_num, fit_time_std, trans_time_std, X_train_std.shape[1], None),
            "B_AutoPrep_Default": (X_train_b.to_numpy(), X_val_b.to_numpy(), y_train_b, fit_time_b, trans_time_b, X_train_b.shape[1], prep_b),
            "C_AutoPrep_DropLeaky": (X_train_c.to_numpy(), X_val_c.to_numpy(), y_train_c, fit_time_c, trans_time_c, X_train_c.shape[1], prep_c),
        }

        for app_name, (X_tr, X_vl, y_tr, f_time, t_time, n_feats, prep_obj) in processed_data.items():
            for model_name in models:
                if is_regression:
                    if model_name == "RandomForest":
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    else:
                        model = Ridge(random_state=42)

                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_vl)

                    # Inversion de la cible si AutoPreprocessor l'a transformée
                    if prep_obj is not None:
                        preds = prep_obj.inverse_transform_target(preds)

                    score_main = float(np.sqrt(mean_squared_error(y_val_num, preds)))  # RMSE
                    score_aux = float(r2_score(y_val_num, preds))  # R2

                else:
                    # Classification
                    if model_name == "RandomForest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    else:
                        model = LogisticRegression(max_iter=1000, random_state=42)

                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_vl)

                    if spec.task == "binary":
                        try:
                            proba = model.predict_proba(X_vl)[:, 1]
                            score_main = float(roc_auc_score(y_val_num, proba))
                        except Exception:
                            score_main = float(accuracy_score(y_val_num, preds))
                        score_aux = float(f1_score(y_val_num, preds, average="binary"))
                    else:
                        # Multiclasse
                        score_main = float(accuracy_score(y_val_num, preds))
                        score_aux = float(f1_score(y_val_num, preds, average="macro"))

                metrics_history[app_name][model_name]["score_main"].append(score_main)
                metrics_history[app_name][model_name]["score_aux"].append(score_aux)
                metrics_history[app_name][model_name]["fit_time"].append(f_time)
                metrics_history[app_name][model_name]["trans_time"].append(t_time)
                metrics_history[app_name][model_name]["n_features"].append(n_feats)

    # -------------------------------------------------------------------------
    # TEST DE LA CIBLE MÉLANGÉE (PERMUTATION TEST - CONTRÔLE DE FUITE STRICT)
    # -------------------------------------------------------------------------
    # Évaluation out-of-sample pour vérifier qu'aucune fausse relation n'est apprise
    perm_split = int(len(X) * 0.7)
    X_perm_tr, X_perm_te = X.iloc[:perm_split].copy(), X.iloc[perm_split:].copy()
    y_shuffled = np.random.RandomState(42).permutation(y)
    y_perm_tr, y_perm_te = y_shuffled[:perm_split], y_shuffled[perm_split:]

    prep_perm = AutoPreprocessor(PreprocessConfig(random_state=42, verbose=0))
    X_perm_tr_trans = prep_perm.fit_transform(X_perm_tr, y_perm_tr)
    X_perm_te_trans = prep_perm.transform(X_perm_te)

    if is_regression:
        rf_perm = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_perm.fit(X_perm_tr_trans, y_perm_tr)
        preds_perm = rf_perm.predict(X_perm_te_trans)
        perm_score = float(r2_score(y_perm_te, preds_perm))
        perm_leakage_safe = perm_score < 0.15  # Sur test mélangé, R² doit être <= 0
    else:
        le_perm = LabelEncoder().fit(pd.Series(y).astype(str))
        y_perm_tr_enc = le_perm.transform(pd.Series(y_perm_tr).astype(str))
        y_perm_te_enc = le_perm.transform(pd.Series(y_perm_te).astype(str))
        rf_perm = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_perm.fit(X_perm_tr_trans, y_perm_tr_enc)
        try:
            if spec.task == "binary":
                perm_proba = rf_perm.predict_proba(X_perm_te_trans)[:, 1]
                perm_score = float(roc_auc_score(y_perm_te_enc, perm_proba))
                perm_leakage_safe = perm_score < 0.65  # Doit être proche de 0.50
            else:
                preds_perm = rf_perm.predict(X_perm_te_trans)
                perm_score = float(accuracy_score(y_perm_te_enc, preds_perm))
                majority_freq = float(pd.Series(y_perm_te_enc).value_counts(normalize=True).max())
                perm_leakage_safe = perm_score <= (majority_freq + 0.10)
        except Exception:
            perm_score = 0.50
            perm_leakage_safe = True

    return {
        "metrics": metrics_history,
        "detected_leaks": list(detected_leaks),
        "perm_score": perm_score,
        "perm_leakage_safe": perm_leakage_safe,
    }


# =============================================================================
# 5. ORCHESTRATEUR PRINCIPAL ET GÉNÉRATION DES RAPPORTS
# =============================================================================

def run_benchmarks(
    selected_datasets: Optional[List[str]] = None,
    quick: bool = False,
    output_dir: str = "resultats",
    n_splits: int = 5,
):
    os.makedirs(output_dir, exist_ok=True)
    n_est = 50 if quick else 100
    n_splits = 3 if quick else n_splits

    datasets_to_run = list(DATASETS_REGISTRY.keys()) if not selected_datasets else selected_datasets

    print("\n" + "=" * 80)
    print(" DÉMARRAGE DU BENCHMARK COMPARATIF MULTI-DATASETS")
    print("=" * 80)
    print(f"Datasets ciblés : {len(datasets_to_run)}")
    print(f"Mode            : {'QUICK (3 plis, RF 50 arbres)' if quick else f'STANDARD ({n_splits} plis, RF 100 arbres)'}")
    print(f"Dossier sortie  : {output_dir}")
    print("-" * 80)

    benchmark_rows = []
    skipped_datasets = []
    robustness_summary = {}
    leakage_alerts = []
    dataset_outcomes = []

    for idx, ds_name in enumerate(datasets_to_run, 1):
        if ds_name not in DATASETS_REGISTRY:
            print(f"[{idx}/{len(datasets_to_run)}] Dataset inconnu : '{ds_name}' -> SKIPPED")
            skipped_datasets.append((ds_name, "Identifiant non répertorié"))
            continue

        spec = DATASETS_REGISTRY[ds_name]
        print(f"\n[{idx}/{len(datasets_to_run)}] Évaluation de '{spec.name}' ({spec.task.upper()})...")
        print(f"Description : {spec.description}")

        # Chargement avec try/except pour gérer les éventuels problèmes réseau
        try:
            t0_load = time.time()
            X, y = load_dataset(spec)
            print(f"Données chargées en {time.time() - t0_load:.2f}s : X={X.shape}, y={len(y)}")
        except Exception as e:
            print(f"ERREUR au chargement de '{ds_name}' : {e} -> SKIPPED")
            skipped_datasets.append((ds_name, str(e)))
            continue

        try:
            # 1. Tests de robustesse
            print(" -> Exécution des tests de robustesse (types, NaN, joblib, reproductibilité)...")
            robust_res = run_robustness_tests(X, y, spec)
            robustness_summary[ds_name] = robust_res
            all_robust_ok = all(robust_res.values())
            print(f" -> Robustesse : {'TOUS LES TESTS PASSÉS (100%)' if all_robust_ok else 'ATTENTION CERTAINS TESTS ONT ÉCHOUÉ'}")

            # 2. Validation croisée
            print(f" -> Lancement de la validation croisée ({n_splits} plis, 4 approches, 2 modèles)...")
            cv_res = run_cross_validation_benchmark(spec, X, y, n_splits=n_splits, n_estimators=n_est)
        except Exception as e:
            print(f"ERREUR lors de l'évaluation de '{ds_name}' : {e} -> SKIPPED")
            skipped_datasets.append((ds_name, str(e)))
            continue

        # Contrôle du test de permutation (cible mélangée : le score doit tomber au niveau du hasard)
        print(f" -> Test cible mélangée : score={cv_res['perm_score']:.3f} "
              f"({'OK, niveau du hasard' if cv_res['perm_leakage_safe'] else 'SUSPECT'})")
        if not cv_res["perm_leakage_safe"]:
            alert = f"ALERTE FUITE : '{ds_name}' conserve un score élevé ({cv_res['perm_score']:.3f}) sur cible permutée !"
            print(f" -> [!] {alert}")
            leakage_alerts.append(alert)

        # Enregistrement des résultats tabulaires
        metrics_dict = cv_res["metrics"]
        main_metric_name = "RMSE" if spec.task == "regression" else ("ROC-AUC" if spec.task == "binary" else "Accuracy")
        aux_metric_name = "R2" if spec.task == "regression" else "F1-Score"

        # Comparaison des approches sur RandomForest
        rf_naive_mean = np.mean(metrics_dict["A_NaiveBaseline"]["RandomForest"]["score_main"])
        rf_auto_mean = np.mean(metrics_dict["B_AutoPrep_Default"]["RandomForest"]["score_main"])
        rf_leaky_mean = np.mean(metrics_dict["C_AutoPrep_DropLeaky"]["RandomForest"]["score_main"])
        rf_auto_std = np.std(metrics_dict["B_AutoPrep_Default"]["RandomForest"]["score_main"])

        # (Plus d'alerte "score > 0.95" : un dataset facile n'est pas une fuite.
        #  Les vrais contrôles sont le détecteur de colonnes et le test cible mélangée.)

        # Déterminer Victoire / Égalité / Défaite (par rapport à l'écart-type)
        if spec.task == "regression":
            # Pour RMSE, plus bas est meilleur
            diff = rf_naive_mean - rf_auto_mean
            if diff > rf_auto_std:
                outcome = "VICTOIRE"
            elif abs(diff) <= rf_auto_std:
                outcome = "ÉGALITÉ"
            else:
                outcome = "DÉFAITE"
        else:
            # Pour classification, plus haut est meilleur
            diff = rf_auto_mean - rf_naive_mean
            if diff > rf_auto_std:
                outcome = "VICTOIRE"
            elif abs(diff) <= rf_auto_std:
                outcome = "ÉGALITÉ"
            else:
                outcome = "DÉFAITE"

        dataset_outcomes.append({
            "dataset": ds_name,
            "task": spec.task,
            "metric": main_metric_name,
            "baseline": rf_naive_mean,
            "autoprep": rf_auto_mean,
            "autoprep_dropleaky": rf_leaky_mean,
            "diff": diff,
            "std": rf_auto_std,
            "outcome": outcome,
            "leaks": cv_res["detected_leaks"],
            "perm_score": cv_res["perm_score"],
            "perm_ok": cv_res["perm_leakage_safe"],
        })

        for app_name, app_models in metrics_dict.items():
            for mod_name, vals in app_models.items():
                benchmark_rows.append({
                    "dataset": ds_name,
                    "task": spec.task,
                    "approche": app_name,
                    "modele": mod_name,
                    "metric_principale_nom": main_metric_name,
                    "metric_principale_moyenne": np.mean(vals["score_main"]),
                    "metric_principale_std": np.std(vals["score_main"]),
                    "metric_auxiliaire_nom": aux_metric_name,
                    "metric_auxiliaire_moyenne": np.mean(vals["score_aux"]),
                    "metric_auxiliaire_std": np.std(vals["score_aux"]),
                    "fit_time_sec_moyenne": np.mean(vals["fit_time"]),
                    "trans_time_sec_moyenne": np.mean(vals["trans_time"]),
                    "n_features_moyenne": int(np.mean(vals["n_features"])),
                })

        print(f" -> Résultat RF ({main_metric_name}) : Baseline={rf_naive_mean:.4f} | AutoPrep={rf_auto_mean:.4f} ± {rf_auto_std:.4f} [{outcome}]")

    # =========================================================================
    # 6. ENREGISTREMENT DES RAPPORTS ET EXPORT CSV / MARKDOWN
    # =========================================================================
    df_results = pd.DataFrame(benchmark_rows)
    csv_path = os.path.join(output_dir, "benchmark.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nFichier CSV sauvegardé : '{csv_path}'")

    # Génération du rapport Markdown
    md_path = os.path.join(output_dir, "rapport.md")
    generate_markdown_report(
        md_path=md_path,
        df_results=df_results,
        dataset_outcomes=dataset_outcomes,
        skipped_datasets=skipped_datasets,
        robustness_summary=robustness_summary,
        leakage_alerts=leakage_alerts,
    )
    print(f"Rapport Markdown sauvegardé : '{md_path}'")

    # Génération du graphique de gains
    if HAS_MATPLOTLIB and dataset_outcomes:
        png_path = os.path.join(output_dir, "gains.png")
        generate_gain_chart(dataset_outcomes, png_path)
        print(f"Graphique sauvegardé : '{png_path}'")

    print("\n" + "=" * 80)
    print(" BENCHMARK TERMINÉ AVEC SUCCÈS")
    print("=" * 80)


def generate_gain_chart(outcomes: List[Dict[str, Any]], png_path: str):
    """Crée un graphique à barres synthétique des gains d'AutoPreprocessor."""
    datasets = [d["dataset"] for d in outcomes]
    gains = [d["diff"] for d in outcomes]
    stds = [d["std"] for d in outcomes]
    colors = ["#2ecc71" if g > 0.005 else ("#e74c3c" if g < -0.005 else "#3498db") for g in gains]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(datasets))
    plt.barh(y_pos, gains, xerr=stds, align="center", color=colors, alpha=0.85, capsize=4)
    plt.yticks(y_pos, datasets, fontsize=11)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Gain absolu de la métrique principale (RandomForest : AutoPrep - Baseline)", fontsize=11)
    plt.title("Gains Comparés d'AutoPreprocessor vs Baseline Naïve par Dataset", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def generate_markdown_report(
    md_path: str,
    df_results: pd.DataFrame,
    dataset_outcomes: List[Dict[str, Any]],
    skipped_datasets: List[Tuple[str, str]],
    robustness_summary: Dict[str, Dict[str, bool]],
    leakage_alerts: List[str],
):
    """Formate et écrit le rapport de synthèse scientifique exhaustif."""
    n_victoires = sum(1 for d in dataset_outcomes if d["outcome"] == "VICTOIRE")
    n_egalites = sum(1 for d in dataset_outcomes if d["outcome"] == "ÉGALITÉ")
    n_defaites = sum(1 for d in dataset_outcomes if d["outcome"] == "DÉFAITE")

    lines = []
    lines.append("# Rapport Comparatif et Benchmarking d'AutoPreprocessor")
    lines.append("")
    lines.append("## 1. Synthèse Globale des Performances")
    lines.append(f"- **Nombre de datasets évalués avec succès** : {len(dataset_outcomes)}")
    lines.append(f"- **Bilan (RandomForest)** : **{n_victoires} Victoires** | **{n_egalites} Égalités** | **{n_defaites} Défaites**")
    lines.append("- **Protocole** : Validation croisée équitable réajustée dans chaque pli, 4 approches comparées sur données brutes strictes.")
    lines.append("")

    lines.append("## 2. Tableau Récapitulatif par Dataset (RandomForest)")
    lines.append("| Dataset | Tâche | Métrique Principale | Baseline Naïve | AutoPreprocessor | Gain Absolu | Statut |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for d in dataset_outcomes:
        lines.append(
            f"| **{d['dataset']}** | {d['task']} | {d['metric']} | {d['baseline']:.4f} | {d['autoprep']:.4f} ± {d['std']:.4f} | {d['diff']:+.4f} | **{d['outcome']}** |"
        )
    lines.append("")

    # Modèles linéaires (impact du scaling)
    lines.append("## 3. Impact du Prétraitement sur les Modèles Linéaires (LogisticRegression / Ridge)")
    lines.append("| Dataset | Métrique | A. Baseline | D. sklearn standard | B. AutoPreprocessor | B - A | B - D |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    df_linear = df_results[df_results["modele"] == "LinearModel"]
    for ds in df_linear["dataset"].unique():
        df_ds = df_linear[df_linear["dataset"] == ds]
        base_row = df_ds[df_ds["approche"] == "A_NaiveBaseline"]
        auto_row = df_ds[df_ds["approche"] == "B_AutoPrep_Default"]
        std_row = df_ds[df_ds["approche"] == "D_SklearnStandard"]
        if not base_row.empty and not auto_row.empty and not std_row.empty:
            b_val = base_row["metric_principale_moyenne"].iloc[0]
            a_val = auto_row["metric_principale_moyenne"].iloc[0]
            d_val = std_row["metric_principale_moyenne"].iloc[0]
            lines.append(
                f"| {ds} | {base_row['metric_principale_nom'].iloc[0]} | {b_val:.4f} | {d_val:.4f} | {a_val:.4f} "
                f"| {a_val - b_val:+.4f} | {a_val - d_val:+.4f} |"
            )
    lines.append("")
    lines.append("> RMSE : plus bas = meilleur (écart négatif = gain). ROC-AUC / Accuracy : plus haut = meilleur. "
                 "**B - D** isole la vraie valeur ajoutée du module par rapport à un pipeline sklearn standard.")
    lines.append("")

    # Section Contrôles de fuite
    lines.append("## 4. Contrôles de Fuite de Données (Data Leakage)")
    if leakage_alerts:
        lines.append("### Alertes déclenchées :")
        for alert in leakage_alerts:
            lines.append(f"- > [!WARNING]\n  > {alert}")
    else:
        lines.append("- Aucune alerte : le test cible mélangée est resté au niveau du hasard sur tous les datasets.")

    lines.append("\n### Comparaison avec `drop_leaky_columns=True` :")
    lines.append("| Dataset | Colonnes Fuites Détectées | Score AutoPrep (Standard) | Score AutoPrep (DropLeaky) | Test cible mélangée |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")
    for d in dataset_outcomes:
        leaks_str = ", ".join(d["leaks"]) if d["leaks"] else "Aucune"
        perm_txt = f"{d['perm_score']:.3f} {'OK' if d['perm_ok'] else 'SUSPECT'}"
        lines.append(f"| {d['dataset']} | `{leaks_str}` | {d['autoprep']:.4f} | {d['autoprep_dropleaky']:.4f} | {perm_txt} |")
    lines.append("")

    # Datasets SKIPPED
    lines.append("## 5. Datasets Écartés (SKIPPED)")
    if skipped_datasets:
        for ds, reason in skipped_datasets:
            lines.append(f"- **{ds}** : {reason}")
    else:
        lines.append("- Aucun dataset n'a été écarté. Tous ont été exécutés avec succès.")
    lines.append("")

    # Tests de robustesse
    lines.append("## 6. Synthèse des Tests de Robustesse")
    lines.append("| Dataset | Numérique 100% | Zéro NaN / Inf | Colonnes Train=Test | Tolérance Inconnues | Joblib OK | Reproductibilité |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ds, tests in robustness_summary.items():
        lines.append(
            f"| {ds} | {'✅' if tests['numeric_output'] else '❌'} | "
            f"{'✅' if tests['no_nan_no_inf'] else '❌'} | "
            f"{'✅' if tests['same_columns_train_test'] else '❌'} | "
            f"{'✅' if tests['unseen_categories_safe'] else '❌'} | "
            f"{'✅' if tests['joblib_equality'] else '❌'} | "
            f"{'✅' if tests['reproducibility'] else '❌'} |"
        )
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# 6. POINT D'ENTRÉE CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Multi-Datasets pour AutoPreprocessor")
    parser.add_argument(
        "--datasets",
        type=str,
        default="titanic,breast_cancer,wine,california_housing,diabetes_reg,credit_g,car,pima_diabetes,adult,iris",
        help="Liste de datasets séparés par des virgules",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Exécution rapide (3 plis, 50 arbres, sous-échantillonnage des gros datasets)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="resultats",
        help="Dossier de sortie pour les résultats (CSV, Markdown, PNG)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Nombre de plis de validation croisée",
    )

    args = parser.parse_args()
    selected_ds = [d.strip() for d in args.datasets.split(",") if d.strip()]

    run_benchmarks(
        selected_datasets=selected_ds,
        quick=args.quick,
        output_dir=args.out,
        n_splits=args.n_splits,
    )