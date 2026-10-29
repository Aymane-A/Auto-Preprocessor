"""
AutoPreprocessor - Module de prétraitement automatisé et robuste pour Machine Learning tabulaire.

Auteur: Senior ML Engineer / Data Scientist
Licence: MIT
"""

from __future__ import annotations

import copy
import datetime
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# pyrefly: ignore[missing-import]
import joblib
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pandas as pd
# pyrefly: ignore[missing-import]
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    TargetEncoder,
)

# Configuration du logger
logger = logging.getLogger("AutoPreprocessor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [AutoPreprocessor] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FeatureType(str, Enum):
    """Types de features détectés par AutoPreprocessor."""
    DROP_ID = "DROP_ID"
    DROP_CONSTANT = "DROP_CONSTANT"
    DROP_HIGH_MISSING = "DROP_HIGH_MISSING"
    DROP_DUPLICATE = "DROP_DUPLICATE"
    BINARY = "BINARY"
    ORDINAL = "ORDINAL"
    NUMERIC_CONTINUOUS = "NUMERIC_CONTINUOUS"
    NUMERIC_DISCRETE = "NUMERIC_DISCRETE"
    CATEGORICAL_LOW_CARD = "CATEGORICAL_LOW_CARD"
    CATEGORICAL_HIGH_CARD = "CATEGORICAL_HIGH_CARD"
    DATETIME = "DATETIME"
    TEXT = "TEXT"
    DROP_LEAKAGE = "DROP_LEAKAGE"


# Valeurs textuelles reconnues comme manquantes (exclut 'none' qui est fréquemment une modalité catégorielle légitime)
DEFAULT_NA_VALUES = {
    "?", "n/a", "na", "null", "", "-", "unknown",
    "nan", "nil", "undefined", "n.a.", "missing"
}

# Motifs ordinaux standards détectés automatiquement (en minuscules)
COMMON_ORDINAL_PATTERNS = [
    ["low", "medium", "high"],
    ["faible", "moyen", "fort", "eleve"],
    ["faible", "moyen", "fort"],
    ["tres faible", "faible", "moyen", "eleve", "tres eleve"],
    ["xs", "s", "m", "l", "xl", "xxl"],
    ["s", "m", "l", "xl"],
    ["bad", "poor", "average", "good", "excellent"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["mauvais", "moyen", "bon", "excellent"],
    ["primaire", "secondaire", "superieur", "master", "doctorat"],
    ["tier 1", "tier 2", "tier 3"],
    ["tier1", "tier2", "tier3"],
    ["niveau 1", "niveau 2", "niveau 3", "niveau 4"],
]


@dataclass
class PreprocessConfig:
    """Configuration complète et typée pour le prétraitement automatisé.

    Attributes:
        cardinality_threshold: Seuil séparant basse et haute cardinalité catégorielle.
        missing_threshold: Seuil au-delà duquel une colonne est éliminée pour trop de NaN.
        correlation_threshold: Seuil de corrélation linéaire (|r|) pour supprimer les colinéarités.
        rare_category_threshold: Seuil de fréquence relative en deçà duquel regrouper en '__RARE__'.
        skew_threshold: Seuil d'asymétrie absolue déclenchant PowerTransformer (Yeo-Johnson).
        imputation_strategy: Stratégie d'imputation numérique ('median', 'mean').
        scaler_type: Type de normalisation numérique ('standard', 'robust', 'minmax', 'none').
        outlier_method: Méthode de traitement des outliers ('winsorize', 'iqr', 'none').
        outlier_lower_quantile: Quantile inférieur pour la winsorisation.
        outlier_upper_quantile: Quantile supérieur pour la winsorisation.
        outlier_iqr_multiplier: Multiplicateur IQR pour le clipping.
        missing_indicator_threshold: Seuil de NaN justifiant un MissingIndicator (ex: > 5%).
        variance_threshold: Seuil de variance minimale sous lequel la feature est éliminée.
        constant_threshold: Seuil de fréquence de la valeur dominante pour considérer quasi-constante.
        max_text_features: Nombre maximal de features TF-IDF pour les colonnes de texte libre.
        text_svd_components: Nombre de composantes TruncatedSVD (0 pour désactiver SVD).
        ordinal_orders: Dictionnaire spécifiant l'ordre explicite des colonnes ordinales {col: [val1, val2, ...]}.
        target_skew_threshold: Seuil d'asymétrie de la cible en régression justifiant une transformation.
        transform_target: Activer la transformation de la cible continue si asymétrique.
        clean_column_names: Normaliser les noms de colonnes en snake_case sûr.
        clean_string_values: Stripper et homogénéiser les chaînes et remplacer les faux manquants.
        parse_numeric_strings: Convertir les chaînes formatées ('1 200', '12,5', '$300', '45%') en float.
        numeric_string_threshold: Ratio de succès requis pour convertir une colonne string en float.
        downcast_numeric: Réduire la précision numérique (float64 -> float32) pour économiser la RAM.
        drop_leaky_columns: Supprimer automatiquement les colonnes suspectées de fuite de cible.
        leakage_threshold: Score CV (colonne seule) à partir duquel une colonne est suspecte.
        leakage_min_gap: Écart minimal entre le score de la colonne suspecte et celui de la meilleure
            autre colonne (une feature légitime très prédictive est entourée d'autres features prédictives).
        min_features_for_auto_drop: Sous ce nombre de colonnes, alerte seulement, jamais de suppression.
        min_features_for_collinearity: Sous ce nombre de features, pas de filtre de colinéarité.
        random_state: Graine aléatoire pour la reproductibilité.
        verbose: Niveau de verbosité (0=Quiet, 1=Info, 2=Debug).
    """
    cardinality_threshold: int = 10
    missing_threshold: float = 0.90
    correlation_threshold: float = 0.95
    rare_category_threshold: float = 0.01
    skew_threshold: float = 1.0
    imputation_strategy: str = "median"
    scaler_type: str = "standard"
    outlier_method: str = "winsorize"
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99
    outlier_iqr_multiplier: float = 1.5
    missing_indicator_threshold: float = 0.05
    variance_threshold: float = 0.0
    constant_threshold: float = 0.99
    max_text_features: int = 15
    text_svd_components: int = 4
    ordinal_orders: Optional[Dict[str, List[Any]]] = None
    target_skew_threshold: float = 1.0
    transform_target: bool = True
    clean_column_names: bool = True
    clean_string_values: bool = True
    parse_numeric_strings: bool = True
    numeric_string_threshold: float = 0.90
    downcast_numeric: bool = True
    drop_leaky_columns: bool = False
    leakage_threshold: float = 0.95
    leakage_min_gap: float = 0.10
    min_features_for_auto_drop: int = 10
    min_features_for_collinearity: int = 10
    encode_discrete_as_categorical: bool = False
    random_state: int = 42
    verbose: int = 1


# =============================================================================
# TRANSFORMATEURS SCIKIT-LEARN SUR-MESURE (ZÉRO FUITE)
# =============================================================================

class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorisation ou clipping IQR robuste appris exclusivement sur le train."""

    def __init__(
        self,
        method: str = "winsorize",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        iqr_multiplier: float = 1.5,
    ):
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> Winsorizer:
        self.columns_ = list(X.columns)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        if self.method == "none":
            return self

        for col in self.columns_:
            series = X[col].dropna()
            if len(series) == 0:
                self.lower_bounds_[col] = -np.inf
                self.upper_bounds_[col] = np.inf
                continue

            if self.method == "winsorize":
                low = float(np.nanpercentile(series, self.lower_quantile * 100))
                high = float(np.nanpercentile(series, self.upper_quantile * 100))
            elif self.method == "iqr":
                q25 = float(np.nanpercentile(series, 25))
                q75 = float(np.nanpercentile(series, 75))
                iqr = q75 - q25
                low = q25 - self.iqr_multiplier * iqr
                high = q75 + self.iqr_multiplier * iqr
            else:
                low, high = -np.inf, np.inf

            self.lower_bounds_[col] = low
            self.upper_bounds_[col] = high

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.method == "none":
            return X_out

        for col in self.columns_:
            if col in X_out.columns:
                low = self.lower_bounds_.get(col, -np.inf)
                high = self.upper_bounds_.get(col, np.inf)
                if low <= high:
                    X_out[col] = X_out[col].clip(lower=low, upper=high)
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_, dtype=object)


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Crée un indicateur binaire pour les colonnes ayant dépassé le seuil de NaN au train."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.indicator_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> MissingIndicatorTransformer:
        self.indicator_cols_ = []
        for col in X.columns:
            missing_rate = X[col].isna().mean()
            if missing_rate > self.threshold:
                self.indicator_cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        indicators = {}
        for col in self.indicator_cols_:
            if col in X.columns:
                indicators[f"{col}_isna"] = X[col].isna().astype(np.float32)
            else:
                indicators[f"{col}_isna"] = np.zeros(len(X), dtype=np.float32)
        if indicators:
            return pd.DataFrame(indicators, index=X.index)
        return pd.DataFrame(index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"{col}_isna" for col in self.indicator_cols_], dtype=object)


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extraction cyclique et temporelle complète sans fuite pour les colonnes de type date."""

    def __init__(self):
        self.columns_: List[str] = []
        self.reference_dates_: Dict[str, pd.Timestamp] = {}
        self.has_hour_: Dict[str, bool] = {}
        self.output_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> DateTimeFeatureExtractor:
        self.columns_ = list(X.columns)
        self.reference_dates_ = {}
        self.has_hour_ = {}
        self.output_feature_names_ = []

        for col in self.columns_:
            dt_series = pd.to_datetime(X[col], errors="coerce", format="mixed")
            valid_dt = dt_series.dropna()
            if len(valid_dt) > 0:
                self.reference_dates_[col] = valid_dt.max()
                # Vérifier si l'heure est informative (>0)
                self.has_hour_[col] = bool((valid_dt.dt.hour > 0).any())
            else:
                self.reference_dates_[col] = pd.Timestamp.now()
                self.has_hour_[col] = False

            base_names = [
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
                f"{col}_quarter",
                f"{col}_is_weekend",
                f"{col}_is_month_end",
                f"{col}_days_since_ref",
                f"{col}_month_sin",
                f"{col}_month_cos",
                f"{col}_dow_sin",
                f"{col}_dow_cos",
            ]
            if self.has_hour_[col]:
                base_names.extend([f"{col}_hour_sin", f"{col}_hour_cos"])

            self.output_feature_names_.extend(base_names)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_dict = {}

        for col in self.columns_:
            if col in X.columns:
                dt_series = pd.to_datetime(X[col], errors="coerce", format="mixed")
            else:
                dt_series = pd.Series([pd.NaT] * len(X), index=X.index)

            ref_date = self.reference_dates_.get(col, pd.Timestamp.now())
            has_hr = self.has_hour_.get(col, False)

            # Remplacement des NaT par la date de référence pour éviter NaN
            filled_dt = dt_series.fillna(ref_date)

            year = filled_dt.dt.year.astype(float)
            month = filled_dt.dt.month.astype(float)
            day = filled_dt.dt.day.astype(float)
            dow = filled_dt.dt.dayofweek.astype(float)
            quarter = filled_dt.dt.quarter.astype(float)
            is_weekend = (dow >= 5).astype(float)
            is_month_end = filled_dt.dt.is_month_end.astype(float)
            days_since = (ref_date - filled_dt).dt.total_seconds() / 86400.0

            out_dict[f"{col}_year"] = year
            out_dict[f"{col}_month"] = month
            out_dict[f"{col}_day"] = day
            out_dict[f"{col}_dayofweek"] = dow
            out_dict[f"{col}_quarter"] = quarter
            out_dict[f"{col}_is_weekend"] = is_weekend
            out_dict[f"{col}_is_month_end"] = is_month_end
            out_dict[f"{col}_days_since_ref"] = days_since

            # Encodage cyclique sin / cos
            out_dict[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12.0)
            out_dict[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12.0)
            out_dict[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            out_dict[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

            if has_hr:
                hour = filled_dt.dt.hour.astype(float)
                out_dict[f"{col}_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
                out_dict[f"{col}_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        return pd.DataFrame(out_dict, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.output_feature_names_, dtype=object)


class BinaryMapper(BaseEstimator, TransformerMixin):
    """Mapping binaire 0/1 explicite, mémorisé et robuste."""

    def __init__(self):
        self.mapping_: Dict[str, Dict[Any, int]] = {}
        self.modes_: Dict[str, int] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> BinaryMapper:
        self.columns_ = list(X.columns)
        self.mapping_ = {}
        self.modes_ = {}

        for col in self.columns_:
            uniques = X[col].dropna().unique()
            # Identifier la valeur "négative" (0, False, non, no, etc.)
            val_false = None
            val_true = None

            str_map = {str(v).strip().lower(): v for v in uniques}
            false_candidates = ["0", "0.0", "false", "non", "no", "f", "faux", "n"]
            true_candidates = ["1", "1.0", "true", "oui", "yes", "t", "vrai", "y", "m", "h"]

            for c in false_candidates:
                if c in str_map:
                    val_false = str_map[c]
                    break
            for c in true_candidates:
                if c in str_map:
                    val_true = str_map[c]
                    break

            if val_false is None or val_true is None:
                # Ordre alphabétique stable
                sorted_vals = sorted(uniques, key=lambda x: str(x))
                val_false = sorted_vals[0]
                val_true = sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]

            self.mapping_[col] = {val_false: 0, val_true: 1}

            # Mode
            mapped_series = X[col].map(self.mapping_[col]).dropna()
            self.modes_[col] = int(mapped_series.mode().iloc[0]) if len(mapped_series) > 0 else 0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame(index=X.index)
        for col in self.columns_:
            if col in X.columns:
                mapped = X[col].map(self.mapping_.get(col, {})).fillna(self.modes_.get(col, 0))
                out_df[col] = mapped.astype(np.float32)
            else:
                out_df[col] = np.zeros(len(X), dtype=np.float32)
        return out_df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Regroupement des catégories rares (< seuil) en '__RARE__' et gestion de '__MISSING__'."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.frequent_categories_: Dict[str, set] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> RareCategoryGrouper:
        self.columns_ = list(X.columns)
        self.frequent_categories_ = {}

        for col in self.columns_:
            series = X[col].dropna().astype(str)
            if len(series) == 0:
                self.frequent_categories_[col] = set()
                continue
            freqs = series.value_counts(normalize=True)
            frequent = set(freqs[freqs >= self.threshold].index)
            self.frequent_categories_[col] = frequent

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            if col in X.columns:
                series = X[col].astype(str)
                # Remplace nan représenté en string par __MISSING__
                isna_mask = X[col].isna() | (series.isin(["nan", "None", "<NA>"]))
                frequent = self.frequent_categories_.get(col, set())
                
                cleaned = series.where(series.isin(frequent), "__RARE__")
                cleaned = cleaned.mask(isna_mask, "__MISSING__")
                X_out[col] = cleaned
            else:
                X_out[col] = "__MISSING__"
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_, dtype=object)


class RobustTextTransformer(BaseEstimator, TransformerMixin):
    """Extraction TF-IDF et réduction SVD pour colonnes de texte libre."""

    def __init__(self, max_features: int = 15, svd_components: int = 4, random_state: int = 42):
        self.max_features = max_features
        self.svd_components = svd_components
        self.random_state = random_state
        self.columns_: List[str] = []
        self.vectorizers_: Dict[str, TfidfVectorizer] = {}
        self.svd_models_: Dict[str, Optional[TruncatedSVD]] = {}
        self.output_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> RobustTextTransformer:
        self.columns_ = list(X.columns)
        self.vectorizers_ = {}
        self.svd_models_ = {}
        self.output_names_ = []

        for col in self.columns_:
            text_series = X[col].fillna("").astype(str)
            vec = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b",
            )
            tfidf_mat = vec.fit_transform(text_series)
            self.vectorizers_[col] = vec

            n_features = tfidf_mat.shape[1]
            if n_features == 0:
                self.svd_models_[col] = None
                self.output_names_.append(f"{col}_text_0")
                continue

            if self.svd_components > 0 and n_features > self.svd_components:
                svd = TruncatedSVD(n_components=self.svd_components, random_state=self.random_state)
                svd.fit(tfidf_mat)
                self.svd_models_[col] = svd
                col_names = [f"{col}_svd_{i}" for i in range(self.svd_components)]
            else:
                self.svd_models_[col] = None
                feature_names = vec.get_feature_names_out()
                col_names = [f"{col}_tfidf_{fn}" for fn in feature_names]

            self.output_names_.extend(col_names)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        all_dfs = []

        for col in self.columns_:
            if col in X.columns:
                text_series = X[col].fillna("").astype(str)
            else:
                text_series = pd.Series([""] * len(X), index=X.index)

            vec = self.vectorizers_.get(col)
            svd = self.svd_models_.get(col)

            if vec is None:
                zero_df = pd.DataFrame(np.zeros((len(X), 1)), index=X.index, columns=[f"{col}_text_0"])
                all_dfs.append(zero_df)
                continue

            tfidf_mat = vec.transform(text_series)
            if tfidf_mat.shape[1] == 0:
                zero_df = pd.DataFrame(np.zeros((len(X), 1)), index=X.index, columns=[f"{col}_text_0"])
                all_dfs.append(zero_df)
                continue

            if svd is not None:
                reduced = svd.transform(tfidf_mat)
                cols = [f"{col}_svd_{i}" for i in range(self.svd_components)]
                df_out = pd.DataFrame(reduced, index=X.index, columns=cols)
            else:
                dense = tfidf_mat.toarray()
                feature_names = vec.get_feature_names_out()
                cols = [f"{col}_tfidf_{fn}" for fn in feature_names]
                df_out = pd.DataFrame(dense, index=X.index, columns=cols)

            all_dfs.append(df_out)

        if all_dfs:
            return pd.concat(all_dfs, axis=1)
        return pd.DataFrame(index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.output_names_, dtype=object)


# =============================================================================
# CLASSE PRINCIPALE : AutoPreprocessor
# =============================================================================

class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """Préprocesseur automatique tabulaire haute performance compatible scikit-learn.

    Prend en charge :
    - Nettoyage préalable (faux manquants, conversion chaînes chiffrées, doublons)
    - Détection automatique et déterministe des types de colonnes
    - Winsorisation / IQR, imputation, Yeo-Johnson et normalisation
    - MissingIndicator pour valeurs manquantes informatives
    - Encodage cyclique et extraction de dates
    - Regroupement des catégories rares et One-Hot / Target Encoding
    - Élimination des colinéarités et quasi-constantes
    - Détection proactive des fuites de données
    - Zéro fuite : fit uniquement sur train, transform sur test
    """

    def __init__(
        self,
        config: Optional[PreprocessConfig] = None,
        task: str = "auto",
        **kwargs,
    ):
        self.config = config if config is not None else PreprocessConfig(**kwargs)
        self.task = task.lower()
        self.task_resolved_: str = "classification"

        # Métadonnées et traçabilité
        self.column_metadata_: Dict[str, Dict[str, Any]] = {}
        self.columns_to_drop_: List[str] = []
        self.dropped_collinear_cols_: List[str] = []
        self.dropped_variance_cols_: List[str] = []
        self.suspected_leakage_cols_: List[str] = []
        self.leakage_scores_: Dict[str, float] = {}
        self._te_oof_: Optional[np.ndarray] = None
        self.target_col_: Optional[str] = None
        self.final_feature_names_: List[str] = []
        self.target_encoder_: Optional[Any] = None
        self.target_transformer_: Optional[Any] = None
        self.target_stats_: Dict[str, Any] = {}
        self.warnings_: List[str] = []

        # Composants de transformation
        self.winsorizer_: Optional[Winsorizer] = None
        self.missing_indicator_: Optional[MissingIndicatorTransformer] = None
        self.num_imputer_: Optional[Any] = None
        self.num_power_transformer_: Optional[Dict[str, PowerTransformer]] = None
        self.num_scaler_: Optional[Any] = None
        self.binary_mapper_: Optional[BinaryMapper] = None
        self.ordinal_encoders_: Dict[str, OrdinalEncoder] = {}
        self.rare_grouper_: Optional[RareCategoryGrouper] = None
        self.ohe_encoder_: Optional[OneHotEncoder] = None
        self.target_cat_encoder_: Optional[TargetEncoder] = None
        self.freq_encoding_maps_: Dict[str, Dict[Any, float]] = {}
        self.datetime_extractor_: Optional[DateTimeFeatureExtractor] = None
        self.text_transformer_: Optional[RobustTextTransformer] = None

        # Typage interne
        self.num_cols_: List[str] = []
        self.num_skewed_cols_: List[str] = []
        self.binary_cols_: List[str] = []
        self.ordinal_cols_: List[str] = []
        self.low_card_cols_: List[str] = []
        self.high_card_cols_: List[str] = []
        self.date_cols_: List[str] = []
        self.text_cols_: List[str] = []

        self.input_columns_: List[str] = []
        self.is_fitted_: bool = False

        self._set_logging_level()

    def _set_logging_level(self):
        if self.config.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        elif self.config.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    # -------------------------------------------------------------------------
    # NETTOYAGE PRÉALABLE (ÉTAPE 0)
    # -------------------------------------------------------------------------

    def _normalize_column_names(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """Normalise les noms de colonnes en minuscules et snake_case sûr sans toucher à la cible."""
        if not self.config.clean_column_names:
            return df

        new_cols = []
        mapping = {}
        for col in df.columns:
            if target is not None and col == target:
                new_cols.append(col)
                continue
            # Remplacement des caractères spéciaux et ponctuation risqués
            clean_name = re.sub(r"[^\w\s]", "", str(col).strip())
            clean_name = re.sub(r"\s+", "_", clean_name).lower()
            if not clean_name:
                clean_name = f"col_{len(new_cols)}"
            mapping[col] = clean_name
            new_cols.append(clean_name)

        # Assurer l'unicité des noms
        seen = {}
        unique_cols = []
        for c in new_cols:
            if c not in seen:
                seen[c] = 0
                unique_cols.append(c)
            else:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")

        df_out = df.copy()
        df_out.columns = unique_cols
        return df_out

    def _clean_string_and_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les chaînes, espaces et remplace les faux manquants par NaN."""
        if not self.config.clean_string_values:
            return df

        df_out = df.copy()
        for col in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                # Strip des chaînes
                s = df_out[col].astype(str).str.strip()
                # Détection des faux manquants
                is_missing = s.str.lower().isin(DEFAULT_NA_VALUES) | df_out[col].isna()
                df_out[col] = s.mask(is_missing, np.nan)
        return df_out

    @staticmethod
    def _clean_numeric_str_value(val: Any) -> Optional[float]:
        """Convertit une chaîne représentative d'un nombre (devises, %, séparateurs) en float."""
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s or s.lower() in DEFAULT_NA_VALUES:
            return np.nan
        # Retirer symboles de devises et pourcentages
        s = re.sub(r"[\$€£¥%]", "", s).strip()
        # Retirer une unité/devise en suffixe ("67,50 MAD", "12 kg") ou un code devise en préfixe ("MAD 67")
        # (le suffixe doit suivre un nombre ; un préfixe n'est accepté que pour les codes devise,
        #  afin de ne pas transformer des identifiants comme "C1004" en nombres)
        m = re.fullmatch(r"(?:(?:MAD|DHS?|USD|EUR|GBP|CHF)\s*)?([-+]?\d[\d.,\s\xa0]*?)\s*(?:[A-Za-z]{1,4}\.?)?", s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)
        s = s.replace("\xa0", "").replace(" ", "")
        if not s:
            return np.nan
        # Gestion intelligente des virgules et points
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                # Format européen : 1.234,56 -> 1234.56
                s = s.replace(".", "").replace(",", ".")
            else:
                # Format anglo-saxon : 1,234.56 -> 1234.56
                s = s.replace(",", "")
        elif "," in s:
            # Format décimal avec virgule : 12,5 -> 12.5
            s = s.replace(",", ".")
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    def _parse_numeric_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Détecte et convertit les nombres stockés sous forme de texte (1 200, 12,5, $300, 45%)."""
        if not self.config.parse_numeric_strings:
            return df

        df_out = df.copy()
        for col in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                non_null = df_out[col].dropna()
                if len(non_null) == 0:
                    continue

                # Test de conversion
                converted_series = non_null.apply(self._clean_numeric_str_value)
                success_rate = converted_series.notna().mean()

                if success_rate >= self.config.numeric_string_threshold:
                    logger.debug(f"Colonne '{col}' convertie de texte en numérique (succès={success_rate:.1%})")
                    df_out[col] = df_out[col].apply(self._clean_numeric_str_value)

        return df_out

    def _downcast_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Réduit l'empreinte mémoire des données numériques."""
        if not self.config.downcast_numeric:
            return df
        df_out = df.copy()
        for col in df_out.columns:
            if pd.api.types.is_float_dtype(df_out[col]):
                df_out[col] = pd.to_numeric(df_out[col], downcast="float")
            elif pd.api.types.is_integer_dtype(df_out[col]):
                df_out[col] = pd.to_numeric(df_out[col], downcast="integer")
        return df_out

    def _preprocess_raw_dataframe(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """Exécute l'étape 0 de nettoyage initial sur le DataFrame."""
        df_clean = self._normalize_column_names(df, target=target)
        df_clean = self._clean_string_and_na(df_clean)
        df_clean = self._parse_numeric_strings(df_clean)
        df_clean = self._downcast_memory(df_clean)
        return df_clean

    # -------------------------------------------------------------------------
    # DÉTECTION ET TYPAGE DES COLONNES
    # -------------------------------------------------------------------------

    def _is_date_column(self, series: pd.Series) -> bool:
        """Vérifie si une colonne est convertible en date de façon fiable."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False

        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        # Exclure si purement numérique
        try:
            pd.to_numeric(non_null)
            return False
        except (ValueError, TypeError):
            pass

        # Échantillonnage pour la performance
        sample = non_null.head(100).astype(str)
        # Test de motifs de date typiques
        has_separators = sample.str.contains(r"[-/:.]", regex=True).mean() > 0.8
        if not has_separators:
            return False

        try:
            converted = pd.to_datetime(sample, errors="coerce", format="mixed")
            return bool((converted.notna().mean() >= 0.80))
        except Exception:
            return False

    def _match_ordinal_pattern(self, unique_values: List[Any]) -> Optional[List[Any]]:
        """Identifie si un ensemble de valeurs correspond à un motif ordinal standard."""
        str_uniques = [str(v).strip().lower() for v in unique_values if pd.notna(v)]
        set_uniques = set(str_uniques)

        for pattern in COMMON_ORDINAL_PATTERNS:
            pattern_set = set(pattern)
            if set_uniques.issubset(pattern_set) and len(set_uniques) >= 2:
                # Retourne les valeurs ordonnées selon le pattern
                ordered = [p for p in pattern if p in set_uniques]
                return ordered
        return None

    def _detect_types(self, X: pd.DataFrame) -> None:
        """Analyse et classifie chaque colonne avec justification explicable."""
        n_rows = len(X)
        seen_contents = {}

        self.column_metadata_ = {}
        self.num_cols_ = []
        self.binary_cols_ = []
        self.ordinal_cols_ = []
        self.low_card_cols_ = []
        self.high_card_cols_ = []
        self.date_cols_ = []
        self.text_cols_ = []
        self.columns_to_drop_ = []

        for col in X.columns:
            series = X[col]
            missing_rate = float(series.isna().mean())
            non_null = series.dropna()
            n_unique = int(non_null.nunique())

            meta = {
                "name": col,
                "missing_rate": missing_rate,
                "n_unique": n_unique,
                "type": None,
                "reason": "",
                "action": "KEEP",
            }

            # 1. Colonne avec trop de valeurs manquantes
            if missing_rate >= self.config.missing_threshold:
                meta["type"] = FeatureType.DROP_HIGH_MISSING
                meta["reason"] = f"Taux de valeurs manquantes ({missing_rate:.1%}) >= {self.config.missing_threshold:.1%}"
                meta["action"] = "DROP"
                self.columns_to_drop_.append(col)
                self.column_metadata_[col] = meta
                continue

            # 2. Colonne constante ou quasi-constante
            if n_unique <= 1:
                meta["type"] = FeatureType.DROP_CONSTANT
                meta["reason"] = "Colonne constante (<= 1 valeur unique)"
                meta["action"] = "DROP"
                self.columns_to_drop_.append(col)
                self.column_metadata_[col] = meta
                continue

            top_freq = non_null.value_counts(normalize=True).iloc[0] if len(non_null) > 0 else 0.0
            if top_freq >= self.config.constant_threshold:
                meta["type"] = FeatureType.DROP_CONSTANT
                meta["reason"] = f"Quasi-constante : la valeur dominante représente {top_freq:.1%} (seuil={self.config.constant_threshold:.1%})"
                meta["action"] = "DROP"
                self.columns_to_drop_.append(col)
                self.column_metadata_[col] = meta
                continue

            # 3. Colonne dupliquée (exactement même contenu)
            content_hash = hash(tuple(series.head(50).fillna("__NA__")))
            if content_hash in seen_contents:
                dup_col = seen_contents[content_hash]
                if series.equals(X[dup_col]):
                    meta["type"] = FeatureType.DROP_DUPLICATE
                    meta["reason"] = f"Contenu strictement identique à la colonne '{dup_col}'"
                    meta["action"] = "DROP"
                    self.columns_to_drop_.append(col)
                    self.column_metadata_[col] = meta
                    continue
            seen_contents[content_hash] = col

            # 4. Colonne d'identifiant (ID)
            is_id_candidate = (
                n_rows > 30
                and n_unique >= (n_rows * 0.98)
                and (
                    pd.api.types.is_integer_dtype(series)
                    or pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)
                )
            )
            # Vérifier si ce n'est pas un texte long
            # Une colonne date/heure quasi unique (horodatage) n'est pas un ID
            if is_id_candidate and self._is_date_column(series):
                is_id_candidate = False
            if is_id_candidate:
                mean_str_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
                if mean_str_len < 30:
                    meta["type"] = FeatureType.DROP_ID
                    meta["reason"] = f"Identifiant détecté : cardinalité ({n_unique}) proche du nombre de lignes ({n_rows})"
                    meta["action"] = "DROP"
                    self.columns_to_drop_.append(col)
                    self.column_metadata_[col] = meta
                    continue

            # 5. Détection Datetime
            if self._is_date_column(series):
                meta["type"] = FeatureType.DATETIME
                meta["reason"] = "Dates parsables avec succès ou format datetime natif"
                self.date_cols_.append(col)
                self.column_metadata_[col] = meta
                continue

            # 6. Détection Binaire
            if n_unique == 2:
                meta["type"] = FeatureType.BINARY
                meta["reason"] = "Exactement 2 valeurs distinctes (binaire)"
                self.binary_cols_.append(col)
                self.column_metadata_[col] = meta
                continue

            # 7. Détection Ordinale
            custom_orders = self.config.ordinal_orders or {}
            if col in custom_orders:
                meta["type"] = FeatureType.ORDINAL
                meta["reason"] = f"Ordre ordinal configuré manuellement : {custom_orders[col]}"
                self.ordinal_cols_.append(col)
                self.column_metadata_[col] = meta
                continue

            matched_pattern = self._match_ordinal_pattern(non_null.unique())
            if matched_pattern is not None:
                meta["type"] = FeatureType.ORDINAL
                meta["reason"] = f"Motif ordinal automatique détecté : {matched_pattern}"
                self.ordinal_cols_.append(col)
                self.column_metadata_[col] = meta
                continue

            # 8. Détection Texte Libre
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                mean_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
                avg_words = non_null.astype(str).str.split().str.len().mean() if len(non_null) > 0 else 0
                if mean_len > 25 and avg_words > 3 and n_unique > self.config.cardinality_threshold:
                    meta["type"] = FeatureType.TEXT
                    meta["reason"] = f"Texte libre (longueur moy: {mean_len:.1f} car., {avg_words:.1f} mots/cellule)"
                    self.text_cols_.append(col)
                    self.column_metadata_[col] = meta
                    continue

            # 9. Détection Numérique
            if pd.api.types.is_numeric_dtype(series):
                if (
                    pd.api.types.is_integer_dtype(series)
                    and n_unique <= self.config.cardinality_threshold
                    and self.config.encode_discrete_as_categorical
                ):
                    meta["type"] = FeatureType.NUMERIC_DISCRETE
                    meta["reason"] = f"Entier discret à faible cardinalité ({n_unique} valeurs)"
                    self.low_card_cols_.append(col)
                else:
                    meta["type"] = FeatureType.NUMERIC_CONTINUOUS
                    meta["reason"] = f"Variable numérique continue ({n_unique} valeurs distinctes)"
                    self.num_cols_.append(col)
                self.column_metadata_[col] = meta
                continue

            # 10. Catégorielle Basse ou Haute Cardinalité
            if n_unique <= self.config.cardinality_threshold:
                meta["type"] = FeatureType.CATEGORICAL_LOW_CARD
                meta["reason"] = f"Catégorielle faible cardinalité ({n_unique} <= {self.config.cardinality_threshold})"
                self.low_card_cols_.append(col)
            else:
                meta["type"] = FeatureType.CATEGORICAL_HIGH_CARD
                meta["reason"] = f"Catégorielle haute cardinalité ({n_unique} > {self.config.cardinality_threshold})"
                self.high_card_cols_.append(col)

            self.column_metadata_[col] = meta

        logger.info(
            f"Détection terminée : {len(self.num_cols_)} numériques, {len(self.binary_cols_)} binaires, "
            f"{len(self.ordinal_cols_)} ordinales, {len(self.low_card_cols_)} cat. basses, "
            f"{len(self.high_card_cols_)} cat. hautes, {len(self.date_cols_)} dates, "
            f"{len(self.text_cols_)} textes, {len(self.columns_to_drop_)} supprimées."
        )

    # -------------------------------------------------------------------------
    # DÉTECTION DE FUITE DE CIBLE (sur colonnes BRUTES, une par une)
    # -------------------------------------------------------------------------

    def _detect_leaky_columns(self, X_work: pd.DataFrame, y_arr: np.ndarray) -> None:
        """Teste chaque colonne brute SEULE : petit arbre évalué en validation croisée.

        - Colonnes numériques : valeur brute (NaN remplacé par une valeur sentinelle).
        - Colonnes catégorielles : Target Encoding appris dans chaque pli (aucune fuite).
        Une colonne est suspecte si son score est >= leakage_threshold ET dépasse d'au moins
        leakage_min_gap la meilleure AUTRE colonne, ou si son score est quasi parfait (>= 0.995).
        Une feature légitime très prédictive (ex. petalwidth sur Iris) est entourée d'autres
        features prédictives (petallength) : pas d'alerte.
        """
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        self.leakage_scores_ = {}
        y_all = np.asarray(y_arr)
        if len(y_all) < 30 or len(X_work) != len(y_all):
            return

        mask = ~pd.isna(y_all)
        if mask.sum() < 30:
            return
        X_d = X_work.iloc[np.where(mask)[0]]
        y_d = y_all[mask]

        # Sous-échantillonnage pour garder un coût raisonnable sur les gros datasets
        if len(y_d) > 20000:
            idx = np.random.RandomState(self.config.random_state).choice(len(y_d), 20000, replace=False)
            X_d, y_d = X_d.iloc[idx], y_d[idx]

        rs = self.config.random_state
        is_clf = self.task_resolved_ == "classification"
        if is_clf:
            y_d = y_d.astype(int)
            counts = np.bincount(y_d)
            counts = counts[counts > 0]
            if len(counts) < 2 or counts.min() < 2:
                return
            scoring = "roc_auc" if len(counts) == 2 else "balanced_accuracy"
            cv = StratifiedKFold(n_splits=int(min(5, counts.min())), shuffle=True, random_state=rs)
        else:
            y_d = y_d.astype(float)
            scoring = "r2"
            cv = KFold(n_splits=5, shuffle=True, random_state=rs)

        skip = set(self.date_cols_) | set(self.text_cols_)
        candidates = [c for c in X_d.columns if c not in skip and c not in self.columns_to_drop_]

        for col in candidates:
            s = X_d[col]
            tree = (DecisionTreeClassifier(max_depth=4, random_state=rs) if is_clf
                    else DecisionTreeRegressor(max_depth=4, random_state=rs))
            try:
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    v = s.astype(float)
                    v = v.fillna(float(v.min()) - 1.0 if v.notna().any() else 0.0).to_numpy().reshape(-1, 1)
                    est = tree
                else:
                    v = s.astype(str).to_numpy().reshape(-1, 1)
                    est = make_pipeline(TargetEncoder(smooth="auto", cv=5, random_state=rs), tree)
                score = float(cross_val_score(est, v, y_d, cv=cv, scoring=scoring).mean())
            except Exception as exc:  # petit effectif, classe rare, etc. : on ignore la colonne
                logger.debug(f"Test de fuite ignoré pour '{col}' : {exc}")
                continue
            if not np.isnan(score):
                self.leakage_scores_[col] = score

        small_dataset = len(candidates) < self.config.min_features_for_auto_drop
        for col, score in self.leakage_scores_.items():
            if score < self.config.leakage_threshold:
                continue
            others = [v for k, v in self.leakage_scores_.items() if k != col]
            baseline = float(max(others)) if others else 0.5
            near_perfect = score >= max(self.config.leakage_threshold, 0.995)
            if not near_perfect and score - baseline < self.config.leakage_min_gap:
                continue

            self.suspected_leakage_cols_.append(col)
            msg = (f"ALERTE FUITE DE CIBLE : la colonne '{col}' seule prédit la cible "
                   f"(score CV={score:.3f} vs meilleure autre colonne={baseline:.3f}, métrique={scoring}).")
            self.warnings_.append(msg)
            logger.warning(msg)

            if not self.config.drop_leaky_columns:
                continue
            if small_dataset:
                logger.warning(
                    f"'{col}' conservée : moins de {self.config.min_features_for_auto_drop} colonnes, "
                    f"suppression automatique désactivée (alerte seulement)."
                )
                continue
            for lst in (self.num_cols_, self.binary_cols_, self.ordinal_cols_, self.low_card_cols_,
                        self.high_card_cols_):
                if col in lst:
                    lst.remove(col)
            self.columns_to_drop_.append(col)
            meta = self.column_metadata_[col]
            meta["type"] = FeatureType.DROP_LEAKAGE
            meta["action"] = "DROP"
            meta["reason"] = f"Fuite de cible suspectée (score CV={score:.3f} vs meilleure autre={baseline:.3f})"

    # -------------------------------------------------------------------------
    # GESTION DE LA CIBLE
    # -------------------------------------------------------------------------

    def _resolve_task(self, y: Union[pd.Series, np.ndarray]) -> str:
        """Déduit automatiquement si la tâche est une classification ou régression."""
        if self.task == "regression":
            y_num = pd.to_numeric(pd.Series(y).dropna(), errors="coerce")
            if y_num.isna().any():
                msg = ("La tâche « regression » est demandée mais la cible contient du texte "
                       "(ex. 'Oui'/'Non') : passage automatique en classification.")
                self.warnings_.append(msg)
                logger.warning(msg)
                return "classification"
        if self.task != "auto":
            return self.task

        y_series = pd.Series(y).dropna()
        n_unique = y_series.nunique()
        is_num = pd.api.types.is_numeric_dtype(y_series)

        if is_num and n_unique > 15:
            return "regression"
        return "classification"

    def _fit_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Encode ou transforme la variable cible de manière robuste."""
        y_series = pd.Series(y).copy()
        self.task_resolved_ = self._resolve_task(y_series)
        self.target_stats_ = {"task": self.task_resolved_, "n_samples": len(y_series)}

        if self.task_resolved_ == "classification":
            self.target_encoder_ = LabelEncoder()
            y_encoded = self.target_encoder_.fit_transform(y_series.astype(str))
            classes = self.target_encoder_.classes_
            counts = pd.Series(y_encoded).value_counts(normalize=True)
            min_ratio = float(counts.min())

            self.target_stats_.update({
                "classes": list(classes),
                "n_classes": len(classes),
                "class_proportions": counts.to_dict(),
                "minority_class_ratio": min_ratio,
            })

            if min_ratio < 0.05:
                msg = f"Déséquilibre sévère de classe : ratio minoritaire = {min_ratio:.1%}. Recommandation : SMOTE ou class_weight='balanced'."
                self.warnings_.append(msg)
                logger.warning(msg)
            elif min_ratio < 0.20:
                msg = f"Déséquilibre modéré de classe : ratio minoritaire = {min_ratio:.1%}. Recommandation : class_weight='balanced'."
                self.warnings_.append(msg)
                logger.info(msg)

            return y_encoded

        else:
            # Régression
            y_float = y_series.astype(float).to_numpy()
            skewness = float(stats.skew(y_float, nan_policy="omit"))
            self.target_stats_["skewness"] = skewness

            if self.config.transform_target and abs(skewness) > self.config.target_skew_threshold:
                logger.info(f"Cible de régression asymétrique (skew={skewness:.2f}) -> PowerTransformer Yeo-Johnson.")
                self.target_transformer_ = PowerTransformer(method="yeo-johnson")
                y_trans = self.target_transformer_.fit_transform(y_float.reshape(-1, 1)).ravel()
                self.target_stats_["transformed"] = True
                return y_trans
            else:
                self.target_transformer_ = None
                self.target_stats_["transformed"] = False
                return y_float

    def transform_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Transforme la cible (par exemple lors de l'évaluation sur le test)."""
        y_series = pd.Series(y).copy()
        if self.task_resolved_ == "classification":
            if self.target_encoder_ is None:
                raise ValueError("Target encoder non ajusté.")
            # Gérer les classes inconnues avec -1
            known_classes = set(self.target_encoder_.classes_)
            mapped = y_series.astype(str).map(
                lambda x: self.target_encoder_.transform([x])[0] if x in known_classes else -1
            )
            return mapped.to_numpy()
        else:
            y_float = y_series.astype(float).to_numpy()
            if self.target_transformer_ is not None:
                return self.target_transformer_.transform(y_float.reshape(-1, 1)).ravel()
            return y_float

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Rétablit les valeurs d'origine de la cible pour les prédictions."""
        if self.task_resolved_ == "classification":
            if self.target_encoder_ is None:
                return y
            return self.target_encoder_.inverse_transform(np.round(y).astype(int))
        else:
            if self.target_transformer_ is not None:
                return self.target_transformer_.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
            return np.asarray(y)

    # -------------------------------------------------------------------------
    # FIT ET TRANSFORM
    # -------------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> AutoPreprocessor:
        """Ajuste l'ensemble des transformateurs sur les données d'entraînement.

        Garantit l'absence totale de fuite : toutes les statistiques (médianes, quantiles,
        encodages, références temporelles) sont estimées sur X uniquement.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._set_logging_level()
        self.warnings_ = []
        logger.info(f"Début du fit sur DataFrame de taille {X.shape}")

        # Nettoyage préalable (Étape 0)
        X_clean = self._preprocess_raw_dataframe(X)
        self.input_columns_ = list(X_clean.columns)

        # Ajustement de la cible si fournie
        y_arr = None
        if y is not None:
            y_arr = self._fit_target(y)

        # Détection des types
        self._detect_types(X_clean)

        # Filtrage des colonnes à exclure
        X_work = X_clean.drop(columns=self.columns_to_drop_, errors="ignore")

        # Détection de fuite de cible sur les colonnes brutes (avant tout encodage)
        self.suspected_leakage_cols_ = []
        self.leakage_scores_ = {}
        self._te_oof_ = None
        if y_arr is not None:
            self._detect_leaky_columns(X_work, y_arr)
            X_work = X_clean.drop(columns=self.columns_to_drop_, errors="ignore")

        # 1. Pipeline Numérique
        if self.num_cols_:
            X_num = X_work[self.num_cols_].copy()

            # Missing Indicator
            self.missing_indicator_ = MissingIndicatorTransformer(
                threshold=self.config.missing_indicator_threshold
            )
            self.missing_indicator_.fit(X_num)

            # Winsorisation / Outliers
            self.winsorizer_ = Winsorizer(
                method=self.config.outlier_method,
                lower_quantile=self.config.outlier_lower_quantile,
                upper_quantile=self.config.outlier_upper_quantile,
                iqr_multiplier=self.config.outlier_iqr_multiplier,
            )
            self.winsorizer_.fit(X_num)
            X_num_wins = self.winsorizer_.transform(X_num)

            # Imputation médiane ou moyenne
            if self.config.imputation_strategy == "mean":
                self.num_imputer_ = {col: float(X_num_wins[col].mean()) for col in self.num_cols_}
            else:
                self.num_imputer_ = {col: float(X_num_wins[col].median()) for col in self.num_cols_}

            X_num_imp = X_num_wins.fillna(self.num_imputer_)

            # PowerTransformer pour asymétrie
            self.num_power_transformer_ = {}
            self.num_skewed_cols_ = []
            for col in self.num_cols_:
                sk = float(stats.skew(X_num_imp[col].dropna()))
                if abs(sk) > self.config.skew_threshold:
                    pt = PowerTransformer(method="yeo-johnson")
                    pt.fit(X_num_imp[[col]])
                    self.num_power_transformer_[col] = pt
                    self.num_skewed_cols_.append(col)
                    X_num_imp[col] = pt.transform(X_num_imp[[col]]).ravel()

            # Scaling
            if self.config.scaler_type == "robust":
                self.num_scaler_ = RobustScaler()
            elif self.config.scaler_type == "minmax":
                self.num_scaler_ = MinMaxScaler()
            elif self.config.scaler_type == "standard":
                self.num_scaler_ = StandardScaler()
            else:
                self.num_scaler_ = None

            if self.num_scaler_ is not None:
                self.num_scaler_.fit(X_num_imp)

        # 2. Pipeline Binaire
        if self.binary_cols_:
            self.binary_mapper_ = BinaryMapper()
            self.binary_mapper_.fit(X_work[self.binary_cols_])

        # 3. Pipeline Ordinal
        if self.ordinal_cols_:
            self.ordinal_encoders_ = {}
            for col in self.ordinal_cols_:
                custom_orders = self.config.ordinal_orders or {}
                if col in custom_orders:
                    order = custom_orders[col]
                else:
                    order = self._match_ordinal_pattern(X_work[col].unique())
                    if order is None:
                        order = list(X_work[col].dropna().unique())

                enc = OrdinalEncoder(
                    categories=[order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                enc.fit(X_work[[col]].astype(str).fillna("__MISSING__"))
                self.ordinal_encoders_[col] = enc

        # 4. Pipeline Catégoriel Basse Cardinalité
        if self.low_card_cols_:
            self.rare_grouper_ = RareCategoryGrouper(threshold=self.config.rare_category_threshold)
            self.rare_grouper_.fit(X_work[self.low_card_cols_])
            X_low_grouped = self.rare_grouper_.transform(X_work[self.low_card_cols_])

            self.ohe_encoder_ = OneHotEncoder(
                handle_unknown="ignore",
                drop="if_binary",
                sparse_output=False,
            )
            self.ohe_encoder_.fit(X_low_grouped)

        # 5. Pipeline Catégoriel Haute Cardinalité
        if self.high_card_cols_:
            if y_arr is not None:
                # Target Encoding robuste Out-Of-Fold régularisé
                self.target_cat_encoder_ = TargetEncoder(
                    smooth="auto",
                    cv=5,
                    random_state=self.config.random_state,
                )
                # fit_transform applique le cross-fitting interne (cv=5) : on garde la version
                # "hors pli" du train, utilisée par fit_transform() pour éviter le surapprentissage.
                self._te_oof_ = self.target_cat_encoder_.fit_transform(
                    X_work[self.high_card_cols_].astype(str), y_arr
                )
            else:
                # Non supervisé : Frequency Encoding
                self.freq_encoding_maps_ = {}
                for col in self.high_card_cols_:
                    freqs = X_work[col].astype(str).value_counts(normalize=True).to_dict()
                    self.freq_encoding_maps_[col] = freqs

        # 6. Pipeline Dates
        if self.date_cols_:
            self.datetime_extractor_ = DateTimeFeatureExtractor()
            self.datetime_extractor_.fit(X_work[self.date_cols_])

        # 7. Pipeline Texte
        if self.text_cols_:
            self.text_transformer_ = RobustTextTransformer(
                max_features=self.config.max_text_features,
                svd_components=self.config.text_svd_components,
                random_state=self.config.random_state,
            )
            self.text_transformer_.fit(X_work[self.text_cols_])

        # Post-traitement sur le train : Sélection des colonnes, variance, corrélation et fuites
        X_trans_train = self._transform_components(X_clean)
        self._fit_post_processing(X_trans_train, y_arr)

        self.is_fitted_ = True
        logger.info(f"AutoPreprocessor ajusté avec succès. Nb features finales : {len(self.final_feature_names_)}")
        return self

    def _transform_components(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        """Applique les composants de transformation individuels et concatène le résultat."""
        parts = []

        # 1. Numériques
        if self.num_cols_ and self.winsorizer_ is not None:
            # Sécurité sur les colonnes manquantes
            X_num = pd.DataFrame(index=X_clean.index)
            for c in self.num_cols_:
                X_num[c] = pd.to_numeric(X_clean[c], errors="coerce") if c in X_clean.columns else np.nan

            # Missing Indicator
            if self.missing_indicator_ is not None:
                df_mi = self.missing_indicator_.transform(X_num)
                if not df_mi.empty:
                    parts.append(df_mi)

            # Winsorisation
            X_num_wins = self.winsorizer_.transform(X_num)

            # Imputation
            X_num_imp = X_num_wins.fillna(self.num_imputer_)

            # PowerTransformer
            if self.num_power_transformer_:
                for c, pt in self.num_power_transformer_.items():
                    if c in X_num_imp.columns:
                        X_num_imp[c] = pt.transform(X_num_imp[[c]]).ravel()

            # Scaling
            if self.num_scaler_ is not None:
                scaled_arr = self.num_scaler_.transform(X_num_imp)
                df_num_scaled = pd.DataFrame(scaled_arr, index=X_clean.index, columns=self.num_cols_)
            else:
                df_num_scaled = X_num_imp

            parts.append(df_num_scaled)

        # 2. Binaires
        if self.binary_cols_ and self.binary_mapper_ is not None:
            df_bin = self.binary_mapper_.transform(X_clean[self.binary_cols_])
            parts.append(df_bin)

        # 3. Ordinales
        if self.ordinal_cols_ and self.ordinal_encoders_:
            ord_dict = {}
            for col, enc in self.ordinal_encoders_.items():
                if col in X_clean.columns:
                    col_vals = X_clean[[col]].astype(str).fillna("__MISSING__")
                else:
                    col_vals = pd.DataFrame(["__MISSING__"] * len(X_clean), columns=[col])
                ord_dict[col] = enc.transform(col_vals).ravel()
            df_ord = pd.DataFrame(ord_dict, index=X_clean.index)
            parts.append(df_ord)

        # 4. Catégorielles Basse Cardinalité (OHE)
        if self.low_card_cols_ and self.rare_grouper_ is not None and self.ohe_encoder_ is not None:
            df_grouped = self.rare_grouper_.transform(X_clean[self.low_card_cols_])
            ohe_arr = self.ohe_encoder_.transform(df_grouped)
            ohe_cols = self.ohe_encoder_.get_feature_names_out(self.low_card_cols_)
            df_ohe = pd.DataFrame(ohe_arr, index=X_clean.index, columns=ohe_cols)
            parts.append(df_ohe)

        # 5. Catégorielles Haute Cardinalité
        if self.high_card_cols_:
            if self.target_cat_encoder_ is not None:
                X_high = X_clean[self.high_card_cols_].astype(str)
                te_arr = self.target_cat_encoder_.transform(X_high)
                te_cols = list(self.target_cat_encoder_.get_feature_names_out(self.high_card_cols_))
                df_te = pd.DataFrame(te_arr, index=X_clean.index, columns=te_cols)
                parts.append(df_te)
            elif self.freq_encoding_maps_:
                fe_dict = {}
                for col, fmap in self.freq_encoding_maps_.items():
                    if col in X_clean.columns:
                        fe_dict[f"{col}_freq"] = X_clean[col].astype(str).map(fmap).fillna(0.0)
                    else:
                        fe_dict[f"{col}_freq"] = np.zeros(len(X_clean))
                df_fe = pd.DataFrame(fe_dict, index=X_clean.index)
                parts.append(df_fe)

        # 6. Dates
        if self.date_cols_ and self.datetime_extractor_ is not None:
            df_dt = self.datetime_extractor_.transform(X_clean[self.date_cols_])
            parts.append(df_dt)

        # 7. Texte
        if self.text_cols_ and self.text_transformer_ is not None:
            df_txt = self.text_transformer_.transform(X_clean[self.text_cols_])
            parts.append(df_txt)

        if parts:
            result = pd.concat(parts, axis=1)
        else:
            result = pd.DataFrame(index=X_clean.index)

        return result

    def _fit_post_processing(self, X_trans: pd.DataFrame, y: Optional[np.ndarray]) -> None:
        """Détecte les variances quasi-nulles, colinéarités et fuites de données."""
        if X_trans.empty:
            self.final_feature_names_ = []
            return

        cols = list(X_trans.columns)
        self.dropped_variance_cols_ = []
        self.dropped_collinear_cols_ = []

        # 1. Seuil de variance
        variances = X_trans.var(numeric_only=True)
        for col in cols:
            var_val = variances.get(col, 0.0)
            is_constant = X_trans[col].nunique(dropna=True) <= 1
            if is_constant or np.isnan(var_val) or var_val <= max(self.config.variance_threshold, 1e-12):
                self.dropped_variance_cols_.append(col)

        retained_after_var = [c for c in cols if c not in self.dropped_variance_cols_]
        X_curr = X_trans[retained_after_var]

        # 2. Détection de colinéarité (|r| > correlation_threshold)
        if len(retained_after_var) >= max(2, self.config.min_features_for_collinearity) and len(X_curr) > 1:
            corr_mat = X_curr.corr().abs()
            to_drop_collinear = set()

            # Corrélation avec y si disponible
            target_corrs = {}
            if y is not None:
                for col in retained_after_var:
                    try:
                        valid_mask = ~np.isnan(X_curr[col]) & ~np.isnan(y)
                        if valid_mask.sum() > 2:
                            r, _ = stats.pearsonr(X_curr[col][valid_mask], y[valid_mask])
                            target_corrs[col] = abs(r) if not np.isnan(r) else 0.0
                        else:
                            target_corrs[col] = 0.0
                    except Exception:
                        target_corrs[col] = 0.0

            for i in range(len(retained_after_var)):
                c1 = retained_after_var[i]
                if c1 in to_drop_collinear:
                    continue
                for j in range(i + 1, len(retained_after_var)):
                    c2 = retained_after_var[j]
                    if c2 in to_drop_collinear:
                        continue
                    r_val = corr_mat.loc[c1, c2]
                    if not np.isnan(r_val) and r_val > self.config.correlation_threshold:
                        # Garder celle la plus corrélée à y ou à plus forte variance
                        score1 = target_corrs.get(c1, variances.get(c1, 0))
                        score2 = target_corrs.get(c2, variances.get(c2, 0))
                        drop_col = c2 if score1 >= score2 else c1
                        to_drop_collinear.add(drop_col)
                        logger.debug(
                            f"Colinéarité détectée entre {c1} et {c2} (|r|={r_val:.3f}) -> suppression de {drop_col}"
                        )

            self.dropped_collinear_cols_ = list(to_drop_collinear)

        # Finalisation des colonnes conservées
        all_dropped = set(self.dropped_variance_cols_) | set(self.dropped_collinear_cols_)
        self.final_feature_names_ = [c for c in cols if c not in all_dropped]

    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None, **fit_params) -> pd.DataFrame:
        """Ajuste puis transforme le train. À utiliser pour le train (et non fit() + transform()).

        Les colonnes Target Encoding du train sont remplacées par leur version "hors pli"
        (cross-fitting), sinon le modèle verrait des encodages calculés avec ses propres labels
        et surapprendrait (fuite sur le train).
        """
        self.fit(X, y)
        X_out = self.transform(X)
        if self._te_oof_ is not None and self.target_cat_encoder_ is not None and len(X_out) == len(self._te_oof_):
            te_cols = list(self.target_cat_encoder_.get_feature_names_out(self.high_card_cols_))
            for j, c in enumerate(te_cols):
                if c in X_out.columns:
                    X_out[c] = np.nan_to_num(self._te_oof_[:, j]).astype(np.float32)
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique les transformations apprises sur un nouveau DataFrame (ex: jeu de test).

        Aucune ligne n'est jamais supprimée lors de transform.
        Garantit que la sortie est 100% numérique, sans NaN, sans inf,
        avec exactement les mêmes colonnes et ordres que le train.
        """
        if not self.is_fitted_:
            raise ValueError("AutoPreprocessor n'est pas encore ajusté. Appelez 'fit' avant 'transform'.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Nettoyage préalable (Étape 0)
        X_clean = self._preprocess_raw_dataframe(X)

        # Transformation par composant
        X_trans = self._transform_components(X_clean)

        # Aligner strictement sur les colonnes retenues au fit
        out_df = pd.DataFrame(index=X_clean.index)
        for col in self.final_feature_names_:
            if col in X_trans.columns:
                out_df[col] = X_trans[col]
            else:
                out_df[col] = 0.0

        # Vérification finale de qualité des données
        # 1. Remplacement des Inf et -Inf
        for col in out_df.columns:
            vals = out_df[col].to_numpy()
            is_inf = np.isinf(vals)
            if np.any(is_inf):
                finite_vals = vals[~is_inf & ~np.isnan(vals)]
                max_val = np.max(finite_vals) if len(finite_vals) > 0 else 1.0
                min_val = np.min(finite_vals) if len(finite_vals) > 0 else 0.0
                out_df[col] = np.where(vals == np.inf, max_val, np.where(vals == -np.inf, min_val, vals))

        # 2. Imputation finale de sécurité des NaN résiduels
        out_df = out_df.fillna(0.0)

        # 3. Conversion 100% numérique
        out_df = out_df.astype(np.float32)

        return out_df

    def fit_transform_xy(
        self,
        df: pd.DataFrame,
        target: str,
    ) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
        """Méthode de commodité pour le train : sépare X et y, nettoie les lignes invalides,

        supprime les doublons du train, ajuste le pipeline et retourne (X_trans, y_trans).
        """
        if target not in df.columns:
            raise KeyError(f"La colonne cible '{target}' est absente du DataFrame fourni.")

        df_work = df.copy()

        # Étape 0 : Nettoyage des chaînes et faux manquants
        df_work = self._clean_string_and_na(df_work)

        # Suppression des doublons exacts
        n_before = len(df_work)
        df_work = df_work.drop_duplicates()
        n_dup = n_before - len(df_work)
        if n_dup > 0:
            logger.info(f"{n_dup} lignes en doublon exact supprimées du jeu d'entraînement.")

        # Suppression des lignes où la cible est manquante
        n_missing_target = df_work[target].isna().sum()
        if n_missing_target > 0:
            logger.info(f"{n_missing_target} lignes avec cible '{target}' manquante supprimées du train.")
            df_work = df_work.dropna(subset=[target])

        if len(df_work) == 0:
            raise ValueError("Le DataFrame est vide après suppression des cibles manquantes.")

        y_raw = df_work[target]
        X_raw = df_work.drop(columns=[target])

        # Ajustement et transformation
        self.target_col_ = target
        X_trans = self.fit_transform(X_raw, y_raw)
        y_trans = self.transform_target(y_raw)

        return X_trans, y_trans

    def transform_xy(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """Transforme un jeu de données (ex: test). Si la cible est présente, la retourne transformée."""
        target_name = target if target is not None else self.target_col_

        if target_name is not None and target_name in df.columns:
            X_raw = df.drop(columns=[target_name])
            y_raw = df[target_name]
            X_trans = self.transform(X_raw)
            y_trans = self.transform_target(y_raw)
            return X_trans, y_trans
        else:
            return self.transform(df)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Retourne les noms des features produites en sortie."""
        return np.array(self.final_feature_names_, dtype=object)

    # -------------------------------------------------------------------------
    # RAPPORT DÉTAILLÉ ET EXPLICABLE
    # -------------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Génère un rapport exhaustif et explicable de toutes les décisions prises."""
        columns_summary = []

        for col, meta in self.column_metadata_.items():
            status = "GARDÉE"
            if col in self.columns_to_drop_:
                status = f"SUPPRIMÉE ({meta['type'].value})"
            elif col in self.dropped_variance_cols_:
                status = "SUPPRIMÉE (Variance quasi-nulle)"
            elif col in self.dropped_collinear_cols_:
                status = "SUPPRIMÉE (Colinéarité |r| > seuil)"

            # Identifier les features créées
            created_features = [f for f in self.final_feature_names_ if f == col or f.startswith(f"{col}_")]

            columns_summary.append({
                "Colonne": col,
                "Type détecté": meta["type"].value if isinstance(meta["type"], FeatureType) else str(meta["type"]),
                "Manquants (%)": f"{meta['missing_rate']:.1%}",
                "Cardinalité": meta["n_unique"],
                "Justification": meta["reason"],
                "Statut": status,
                "Nb features créées": len(created_features),
            })

        df_report = pd.DataFrame(columns_summary)

        summary = {
            "colonnes_detaillees": df_report,
            "nb_features_initiales": len(self.input_columns_),
            "nb_features_finales": len(self.final_feature_names_),
            "tache_detectee": self.task_resolved_,
            "analyse_cible": self.target_stats_,
            "colonnes_supprimees": {
                "pre_filtrage": self.columns_to_drop_,
                "variance_nulle": self.dropped_variance_cols_,
                "colinearite": self.dropped_collinear_cols_,
                "fuites_cible": [c for c in self.suspected_leakage_cols_ if c in self.columns_to_drop_],
            },
            "fuites_suspectees": self.suspected_leakage_cols_,
            "scores_fuite_par_colonne": self.leakage_scores_,
            "avertissements": self.warnings_,
        }
        return summary

    def summary(self) -> str:
        """Affichage synthétique et formaté du rapport dans la console."""
        rep = self.report()
        lines = []
        lines.append("=" * 80)
        lines.append("               RAPPORT D'EXÉCUTION AUTOPREPROCESSOR")
        lines.append("=" * 80)
        lines.append(f"Tâche ML                : {rep['tache_detectee'].upper()}")
        lines.append(f"Dimensions initiales    : {rep['nb_features_initiales']} colonnes")
        lines.append(f"Dimensions finales      : {rep['nb_features_finales']} features numériques")
        
        if "minority_class_ratio" in rep["analyse_cible"]:
            lines.append(f"Ratio classe minoritaire: {rep['analyse_cible']['minority_class_ratio']:.1%}")
        if "skewness" in rep["analyse_cible"]:
            lines.append(f"Asymétrie de la cible   : {rep['analyse_cible']['skewness']:.2f}")

        lines.append("-" * 80)
        lines.append("Colonnes supprimées :")
        for raison, cols in rep["colonnes_supprimees"].items():
            if cols:
                lines.append(f"  - {raison:16} : {', '.join(cols)}")

        if rep["fuites_suspectees"]:
            lines.append("-" * 80)
            lines.append(f"ALERTE FUITE DE CIBLE   : {', '.join(rep['fuites_suspectees'])}")

        if rep["avertissements"]:
            lines.append("-" * 80)
            lines.append("Avertissements :")
            for w in rep["avertissements"]:
                lines.append(f"  * {w}")

        lines.append("=" * 80)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # PERSISTANCE & CHARGEMENT
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Sauvegarde l'instance avec ses métadonnées et sa configuration."""
        payload = {
            "model": self,
            "version": "1.0.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.config,
        }
        joblib.dump(payload, path)
        logger.info(f"AutoPreprocessor sauvegardé avec succès dans '{path}'")

    @classmethod
    def load(cls, path: str) -> AutoPreprocessor:
        """Charge une instance d'AutoPreprocessor sauvegardée via joblib."""
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            logger.info(f"AutoPreprocessor version {payload.get('version')} chargé depuis '{path}'")
            return payload["model"]
        elif isinstance(payload, AutoPreprocessor):
            return payload
        else:
            raise TypeError(f"Le fichier '{path}' ne contient pas d'objet AutoPreprocessor valide.")