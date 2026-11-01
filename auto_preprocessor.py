# -*- coding: utf-8 -*-
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
import unicodedata
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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
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


class Action(str, Enum):
    """Actions proposées dans l'analyse préalable."""
    DROP = "DROP"
    KEEP_NUMERIC = "KEEP_NUMERIC"
    IMPUTE_MEDIAN = "IMPUTE_MEDIAN"
    WINSORIZE = "WINSORIZE"
    YEO_JOHNSON = "YEO_JOHNSON"
    LOG1P = "LOG1P"
    SCALE_STANDARD = "SCALE_STANDARD"
    SCALE_ROBUST = "SCALE_ROBUST"
    SCALE_MINMAX = "SCALE_MINMAX"
    NO_SCALE = "NO_SCALE"
    ONEHOT = "ONEHOT"
    TARGET_ENCODE = "TARGET_ENCODE"
    FREQUENCY_ENCODE = "FREQUENCY_ENCODE"
    ORDINAL = "ORDINAL"
    BINARY_MAP = "BINARY_MAP"
    DATE_FEATURES = "DATE_FEATURES"
    TFIDF_SVD = "TFIDF_SVD"
    MISSING_INDICATOR = "MISSING_INDICATOR"


ACTION_RECOMMENDATION_STEP = {
    "DROP": "type", "KEEP_NUMERIC": "distribution", "YEO_JOHNSON": "distribution",
    "LOG1P": "distribution", "BOX_COX": "distribution", "SQRT": "distribution",
    "DATE_FEATURES": "dates", "TFIDF_SVD": "text", "ONEHOT": "encoding",
    "TARGET_ENCODE": "encoding", "FREQUENCY_ENCODE": "encoding", "ORDINAL": "encoding",
    "BINARY_MAP": "encoding",
}


def format_step_evidence(measure: str, threshold: str) -> str:
    """Formateur unique des preuves d'une recommandation d'étape."""
    return f"{str(measure).strip()} ; seuil : {str(threshold).strip()}"


def format_step_reason(action: str, measure: str, threshold: str) -> str:
    """Construit le résumé français depuis les champs de mesure et de seuil."""
    action_labels = {
        "DROP": "Suppression recommandée", "KEEP_NUMERIC": "Conserver comme numérique",
        "YEO_JOHNSON": "Transformation Yeo-Johnson", "LOG1P": "Transformation log1p",
        "BOX_COX": "Transformation Box-Cox", "SQRT": "Transformation racine carrée",
        "DATE_FEATURES": "Extraire les dates", "TFIDF_SVD": "Résumer le texte par TF-IDF/SVD",
        "ONEHOT": "Encodage One-Hot", "TARGET_ENCODE": "Encodage supervisé",
        "FREQUENCY_ENCODE": "Encodage par fréquence", "ORDINAL": "Encodage ordinal",
        "BINARY_MAP": "Mappage binaire",
    }
    label = action_labels.get(str(action), "Conserver le traitement automatique")
    evidence = str(measure).strip()
    limit = str(threshold).strip()
    if action == "DROP":
        return f"{evidence} ; seuil : {limit} ; {label}."
    return f"{evidence} (seuil : {limit}) : {label}."


def recommendation_step_name(action: Any) -> str:
    """Renvoie l'étape à l'origine de l'action recommandée."""
    value = getattr(action, "value", action)
    return ACTION_RECOMMENDATION_STEP.get(str(value), "type")


def _is_time_only_column(name: Any, series: pd.Series) -> bool:
    """Reconnaît une colonne heure sans date à partir du nom et des valeurs."""
    column = str(name).casefold()
    if not re.search(r"heure|time", column) or re.search(r"date|timestamp|datetime", column):
        return False
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False
    pattern = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    return bool(values.str.fullmatch(pattern).mean() >= 0.8)


def _parse_datetime_values(series: pd.Series, time_only: bool = False) -> pd.Series:
    """Parse des dates mixtes ou convertit les heures sur une époque stable."""
    if time_only:
        delta = pd.to_timedelta(series.astype(str).str.strip(), errors="coerce")
        return pd.Timestamp("1900-01-01") + delta
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _current_timestamp(tz=None) -> pd.Timestamp:
    """Point d'injection testable pour les règles explicites de date future."""
    return pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()


# Source unique des sélecteurs visibles et des validations du fit.
# Chaque action listée est prise en charge par le pipeline; les autres ne sont pas proposées.
SUPPORTED_ACTIONS: Dict[str, Dict[str, List[str]]] = {
    "clean_missing_tokens": {"any": ["TO_NAN", "KEEP", "SELECTED_TO_NAN"]},
    "clean_variants": {"any": ["MERGE", "KEEP"]},
    "clean_numeric_text": {"any": ["AUTO", "CONVERT", "KEEP"]},
    "clean_dates": {"any": ["AUTO", "PARSE", "KEEP", "DROP_COLUMN"]},
    "clean_impossible": {"any": ["AUTO", "NAN", "CLIP", "DROP_ROWS", "IGNORE"]},
    "duplicates": {"any": ["DROP", "KEEP"]},
    "nettoyage": {"any": ["AUTO"]},
    "type": {"any": ["AUTO", "DROP", "FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT", "KEEP_RAW"]},
    "missing": {
        "any": ["AUTO"],
        "numeric": ["AUTO", "MEDIAN", "MEAN", "MODE", "CONSTANT", "MISSING_INDICATOR"],
        "categorical_low": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "categorical_high": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "binary": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "ordinal": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "datetime": ["AUTO"], "text": ["AUTO"]},
    "outliers": {"any": ["AUTO"], "numeric": ["AUTO", "NONE", "WINSORIZE", "IQR"],
                 "binary": ["AUTO"], "ordinal": ["AUTO"], "categorical_low": ["AUTO"],
                 "categorical_high": ["AUTO"], "datetime": ["AUTO"], "text": ["AUTO"]},
    "distribution": {"any": ["AUTO"], "numeric": ["AUTO", "NONE", "LOG1P", "BOX_COX", "YEO_JOHNSON", "SQRT"],
                     "categorical": ["AUTO"], "datetime": ["AUTO"], "text": ["AUTO"]},
    "scaling": {"any": ["AUTO"], "numeric": ["AUTO", "NONE", "STANDARD", "ROBUST", "MINMAX", "MAXABS"],
                "categorical": ["AUTO"], "datetime": ["AUTO"], "text": ["AUTO"]},
    "encoding": {"any": ["AUTO"], "binary": ["AUTO", "BINARY_MAP"], "ordinal": ["AUTO", "ORDINAL"],
                 "categorical_low": ["AUTO", "ONEHOT", "ORDINAL", "FREQUENCY_ENCODE"],
                 "categorical_high": ["AUTO", "ONEHOT", "ORDINAL", "TARGET_ENCODE", "FREQUENCY_ENCODE"]},
    "dates": {"any": ["AUTO"], "datetime": ["AUTO", "DATE_FEATURES"]},
    "text": {"any": ["AUTO"], "text": ["AUTO", "TFIDF_SVD"]},
    "target": {"any": ["AUTO"]},
    "selection": {"any": ["AUTO"]},
}


def _action_group(feature_type: Any) -> str:
    """Retourne la famille utilisée par SUPPORTED_ACTIONS."""
    kind = feature_type.value if isinstance(feature_type, FeatureType) else str(feature_type)
    if kind.startswith("NUMERIC"):
        return "numeric"
    return {"BINARY": "binary", "ORDINAL": "ordinal", "CATEGORICAL_LOW_CARD": "categorical_low",
            "CATEGORICAL_HIGH_CARD": "categorical_high", "DATETIME": "datetime", "TEXT": "text"}.get(kind, "any")


def _action_supported(step: str, group: str, action: str) -> bool:
    options = SUPPORTED_ACTIONS.get(step, {})
    return action in options.get(group, options.get("any", []))


def recommendation_confidence(sample_size: int, evidence_strength: float,
                              p_value: Optional[float] = None) -> str:
    """Classe la confiance selon l'effectif et la force statistique observée.

    Règles statistiques pour les trois niveaux :
    - Échantillon insuffisant (sample_size < 30) : toujours 'faible' (loi des petits nombres).
    - Pour sample_size >= 30 :
      * Si p_value est disponible :
        - p_value < 0.001 renforce l'évidence (strength boosté à >= 0.90)
        - p_value < 0.05 renforce l'évidence (strength boosté à >= 0.75)
      * Niveau 'haute' : sample_size >= 100 ET strength >= 0.80 (effet net et grand échantillon)
      * Niveau 'moyenne' : sample_size >= 30 ET strength >= 0.45
      * Niveau 'faible' : sample_size < 30 OU strength < 0.45
    """
    if sample_size < 30:
        return "faible"
    strength = float(np.clip(evidence_strength, 0.0, 1.0))
    if p_value is not None and np.isfinite(p_value):
        if p_value < 0.001:
            strength = max(strength, 0.90)
        elif p_value < 0.05:
            strength = max(strength, 0.75)
    if sample_size >= 100 and strength >= 0.80:
        return "haute"
    if strength >= 0.45:
        return "moyenne"
    return "faible"


def _recommend_scaler(values: pd.Series) -> Tuple[str, str, str]:
    """Règle statistique unique de choix du scaler, appelée par analyze et fit."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 8:
        return "STANDARD", f"observations disponibles : {len(x)}", "échantillon < 8; scaler standard par défaut"
    q1, q3 = x.quantile(.25), x.quantile(.75)
    spread = q3 - q1
    iqr_pct = float(((x < q1 - 1.5 * spread) | (x > q3 + 1.5 * spread)).mean() * 100)
    mad = float(np.median(np.abs(x - np.median(x))))
    mad_pct = float((np.abs(x - np.median(x)) > 3.5 * 1.4826 * mad).mean() * 100) if mad else 0.0
    kurt = float(x.kurt()) if len(x) > 3 else 0.0
    try:
        p_normal = float(stats.normaltest(x.sample(min(len(x), 5000), random_state=42)).pvalue)
    except Exception:
        p_normal = float("nan")
    zero_pct = float((x == 0).mean() * 100)
    if zero_pct >= 50:
        return "MAXABS", f"zéros : {zero_pct:.1f}%", "zéros ≥ 50 %"
    if x.min() >= 0 and x.max() <= 1:
        return "MINMAX", f"bornes=[{x.min():.3g}, {x.max():.3g}]", "bornes observées dans [0, 1]"
    if iqr_pct > 5 or mad_pct > 5 or abs(kurt) > 3:
        return "ROBUST", f"outliers IQR : {iqr_pct:.2f}%, MAD : {mad_pct:.2f}%, kurtosis : {kurt:.2f}", "outliers > 5 % ou |kurtosis| > 3"
    if pd.notna(p_normal) and p_normal > .05 and iqr_pct <= 5 and mad_pct <= 5:
        return "STANDARD", f"p normalité : {p_normal:.3g}; outliers IQR : {iqr_pct:.2f}%, MAD : {mad_pct:.2f}%", "p > 0,05 et outliers ≤ 5 %"
    return "STANDARD", f"p normalité : {p_normal:.3g}; outliers IQR : {iqr_pct:.2f}%, MAD : {mad_pct:.2f}%; kurtosis : {kurt:.2f}", "Standard par défaut faute de critère spécifique"


@dataclass
class ColumnOverride:
    """Choix utilisateur pour une colonne; actions peut contenir une action ou une chaîne."""
    actions: Union[Action, str, List[Union[Action, str]], Dict[str, Any]]
    force_type: Optional[str] = None


@dataclass
class Recommendation:
    action: Action
    reason: str
    confidence: str
    alternatives: List[Action] = field(default_factory=list)


@dataclass
class StepRecommendation:
    action: str
    measure: str
    threshold: str
    explanation: str
    confidence: str
    choices: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    columns: List[Dict[str, Any]]
    sample_note: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.columns)

    def to_text(self) -> str:
        return self.to_dataframe().to_string(index=False)


# Valeurs textuelles reconnues comme manquantes (exclut 'none' qui est fréquemment une modalité catégorielle légitime)
DEFAULT_NA_VALUES = {
    "?", "n/a", "na", "null", "none", "", "-", "unknown",
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
        noise_guard: Garde-fou contre les colonnes de bruit (DATETIME, CATEGORICAL_HIGH_CARD,
            identifiants partiels). Valeurs autorisées : 'off' (désactivé),
            'warn' (avertissement sans suppression, défaut), 'drop' (suppression si mesure stable).
        noise_margin: Marge minimale d'AUC (classification) ou R² (régression) entre le score
            réel et le score sur cible permutée pour qu'une colonne soit considérée informative.
            Défaut 0.03.
        min_rows_for_noise_guard: Nombre minimal de lignes et de positifs (classification) requis
            pour activer le mode 'drop'. Sous ce seuil le mode retombe en 'warn'. Défaut 300.
        keep_columns: Liste de colonnes à ne jamais supprimer par le garde-fou, même en mode
            'drop' (ex: dates métier connues comme importantes). Défaut None.
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
    identifier_unique_threshold: float = 0.90
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
    noise_guard: str = "warn"
    noise_margin: float = 0.03
    min_rows_for_noise_guard: int = 300
    keep_columns: Optional[List[str]] = None
    random_state: int = 42
    verbose: int = 1
    column_overrides: Dict[str, Any] = field(default_factory=dict)



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
        self.time_only_: Dict[str, bool] = {}
        self.output_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> DateTimeFeatureExtractor:
        self.columns_ = list(X.columns)
        self.reference_dates_ = {}
        self.has_hour_ = {}
        self.time_only_ = {}
        self.output_feature_names_ = []

        for col in self.columns_:
            time_only = _is_time_only_column(col, X[col])
            self.time_only_[col] = time_only
            dt_series = _parse_datetime_values(X[col], time_only)
            valid_dt = dt_series.dropna()
            if len(valid_dt) > 0:
                self.reference_dates_[col] = valid_dt.max()
                # Vérifier si l'heure est informative (>0)
                self.has_hour_[col] = bool((valid_dt.dt.hour > 0).any())
            else:
                self.reference_dates_[col] = pd.Timestamp("1900-01-01")
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
                dt_series = _parse_datetime_values(X[col], self.time_only_.get(col, False))
            else:
                dt_series = pd.Series([pd.NaT] * len(X), index=X.index)

            ref_date = self.reference_dates_.get(col, pd.Timestamp("1900-01-01"))
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

    def __init__(self, threshold: float = 0.01, column_thresholds: Optional[Dict[str, float]] = None):
        self.threshold = threshold
        self.column_thresholds = column_thresholds
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
            threshold = (self.column_thresholds or {}).get(col, self.threshold)
            if not 0 <= float(threshold) <= 1:
                raise ValueError(f"Seuil de rareté invalide pour {col!r}: {threshold}; attendu entre 0 et 1.")
            frequent = set(freqs[freqs >= float(threshold)].index)
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

    def __init__(self, max_features: int = 15, svd_components: int = 4, random_state: int = 42,
                 column_components: Optional[Dict[str, int]] = None):
        self.max_features = max_features
        self.svd_components = svd_components
        self.random_state = random_state
        self.column_components = column_components
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

            components = int((self.column_components or {}).get(col, self.svd_components))
            if components < 0:
                raise ValueError(f"Nombre de composantes SVD négatif pour {col!r}.")
            if components > 0 and n_features > components:
                svd = TruncatedSVD(n_components=components, random_state=self.random_state)
                svd.fit(tfidf_mat)
                self.svd_models_[col] = svd
                col_names = [f"{col}_svd_{i}" for i in range(components)]
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
        # F2 : colonnes détectées comme numériques textuelles au fit, avec format décimal
        # {col: 'comma' | 'dot'}  - vide si parse_numeric_strings=False
        self.numeric_string_cols_: Dict[str, str] = {}
        # F1 : colonnes signalées / supprimées par le garde-fou noise_guard
        self.noise_cols_detected_: List[str] = []  # colonnes identifiées comme bruit
        self.noise_cols_dropped_: List[str] = []   # colonnes effectivement supprimées
        self.user_override_notes_: Dict[str, str] = {}
        self.column_scalers_: Dict[str, Any] = {}
        self.column_distributions_: Dict[str, str] = {}
        self.column_missing_: Dict[str, str] = {}
        self.column_outliers_: Dict[str, str] = {}
        self.column_override_params_: Dict[str, Dict[str, Any]] = {}
        self._last_keep_positions_: Optional[np.ndarray] = None
        self.cleaning_decisions_: Dict[str, Dict[str, Any]] = {}
        self.dataset_cleaning_: Dict[str, Any] = {}
        self.categorical_imputers_: Dict[str, Any] = {}
        self.frequency_override_cols_: List[str] = []

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

    def _override_value(self, column: str, step: str) -> Tuple[str, Any]:
        """Lit une action d'étape et ses paramètres depuis column_overrides."""
        all_overrides = self.config.column_overrides or {}
        raw = all_overrides.get(column)
        if raw is None:
            def normalized(name: Any) -> str:
                value = re.sub(r"[^\w\s]", "", str(name).strip())
                return re.sub(r"\s+", "_", value).lower()
            raw = next((value for name, value in all_overrides.items() if normalized(name) == column), None)
        if isinstance(raw, ColumnOverride):
            raw = raw.actions
        defaults = {"clean_missing_tokens": "TO_NAN", "clean_variants": "MERGE",
                    "clean_numeric_text": "AUTO", "clean_dates": "AUTO",
                    "clean_impossible": "AUTO", "type": "AUTO"}
        if not isinstance(raw, dict):
            return defaults.get(step, "AUTO"), None
        value = raw.get(step, defaults.get(step, "AUTO"))
        params = None
        if isinstance(value, tuple) and len(value) == 2:
            value, params = value
        if isinstance(value, Action):
            value = value.value
        return str(value).upper(), params

    def _dataset_duplicate_action(self) -> str:
        actions = set()
        for col, raw in (self.config.column_overrides or {}).items():
            if isinstance(raw, ColumnOverride):
                raw = raw.actions
            if isinstance(raw, dict) and "duplicates" in raw:
                value = raw["duplicates"]
                if isinstance(value, tuple):
                    value = value[0]
                actions.add(str(value.value if isinstance(value, Action) else value).upper())
        if len(actions) > 1:
            raise ValueError("Le choix de suppression des doublons est global : toutes les colonnes doivent avoir le même choix.")
        action = next(iter(actions), "DROP")
        if not _action_supported("duplicates", "any", action):
            raise ValueError(f"Action de doublons invalide : {action}; choix valides : {SUPPORTED_ACTIONS['duplicates']['any']}.")
        return action

    @staticmethod
    def _canonical_variant(value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value).strip().casefold())
        return "".join(ch for ch in text if not unicodedata.combining(ch))

    def _clean_string_and_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les chaînes, espaces et remplace les faux manquants par NaN."""
        if not self.config.clean_string_values:
            has_explicit = any(isinstance(v.actions if isinstance(v, ColumnOverride) else v, dict) and
                               any(k in (v.actions if isinstance(v, ColumnOverride) else v) for k in
                                   ("clean_missing_tokens", "clean_variants"))
                               for v in (self.config.column_overrides or {}).values())
            if not has_explicit:
                return df

        df_out = df.copy()
        for col in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                if self._override_value(str(col), "type")[0] == "KEEP_RAW":
                    continue
                # Strip des chaînes
                missing_action, missing_params = self._override_value(str(col), "clean_missing_tokens")
                variant_action, _ = self._override_value(str(col), "clean_variants")
                original = df_out[col]
                stripped = original.astype(str).str.strip()
                s = stripped.map(self._canonical_variant) if variant_action == "MERGE" else original.astype(str)
                # Détection des faux manquants
                if variant_action not in {"MERGE", "KEEP"}:
                    raise ValueError(f"Action clean_variants invalide pour {col!r}: {variant_action}")
                values = DEFAULT_NA_VALUES
                if missing_action == "SELECTED_TO_NAN":
                    selected = missing_params if missing_params is not None else []
                    if isinstance(selected, dict):
                        selected = selected.get("values", [])
                    if isinstance(selected, str):
                        selected = [selected]
                    values = {str(v).strip().casefold() for v in selected}
                elif missing_action not in {"TO_NAN", "KEEP"}:
                    raise ValueError(f"Action clean_missing_tokens invalide pour {col!r}: {missing_action}")
                is_missing = original.isna() if missing_action == "KEEP" else s.str.casefold().isin(values) | original.isna()
                raw_variant_count = int(max(original.dropna().astype(str).nunique() - s.dropna().nunique(), 0))
                self.cleaning_decisions_[str(col)] = {
                    "faux_manquants_detectes": int((original.notna() & s.str.strip().str.casefold().isin(DEFAULT_NA_VALUES)).sum()),
                    "faux_manquants_convertis": int((is_missing & original.notna()).sum()),
                    "variantes_fusionnees": raw_variant_count if variant_action == "MERGE" else 0,
                    "actions": {"clean_missing_tokens": missing_action, "clean_variants": variant_action},
                }
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
        """Détecte et convertit les colonnes textuelles numériques lors du FIT uniquement (F2).

        Mémorise dans ``self.numeric_string_cols_`` la liste des colonnes converties et
        leur format décimal dominant ('comma' = virgule, 'dot' = point / autre).
        NE DOIT Être appelé QUE dans fit() via _preprocess_raw_dataframe(is_fit=True).
        Pour transform(), utiliser _apply_numeric_string_cols().
        """
        df_out = df.copy()
        self.numeric_string_cols_ = {}
        for col in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                if self._override_value(str(col), "type")[0] == "KEEP_RAW":
                    continue
                numeric_action, _ = self._override_value(str(col), "clean_numeric_text")
                if numeric_action == "KEEP":
                    continue
                if numeric_action not in {"AUTO", "CONVERT"}:
                    raise ValueError(f"Action clean_numeric_text invalide pour {col!r}: {numeric_action}")
                if not self.config.parse_numeric_strings and numeric_action != "CONVERT":
                    continue
                non_null = df_out[col].dropna()
                if len(non_null) == 0:
                    continue

                # Test de conversion
                converted_series = non_null.apply(self._clean_numeric_str_value)
                success_rate = converted_series.notna().mean()
                self.cleaning_decisions_.setdefault(str(col), {}).update({
                    "nombres_texte_convertibles_pct": float(success_rate * 100),
                    "nombres_texte_convertibles": int(converted_series.notna().sum()),
                    "action_nombres_texte": numeric_action,
                })

                if numeric_action == "CONVERT" or success_rate >= self.config.numeric_string_threshold:
                    # Detecter le format decimal dominant pour traçabilite
                    sample_str = non_null.astype(str).head(50)
                    n_comma = int(sample_str.str.contains(r",\d{1,2}$", regex=True).sum())
                    fmt = "comma" if n_comma > len(sample_str) / 3 else "dot"
                    self.numeric_string_cols_[col] = fmt
                    logger.debug(
                        f"Colonne '{col}' convertie de texte en numerique au fit "
                        f"(succes={success_rate:.1%}, format={fmt})"
                    )
                    df_out[col] = df_out[col].apply(self._clean_numeric_str_value)

        return df_out

    def _apply_numeric_string_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique strictement les colonnes numériques textuelles mémorisées au fit (F2).

        Utilise uniquement ``self.numeric_string_cols_`` (défini pendant fit()).
        Ne recalcule aucun seuil sur le nouveau DataFrame.
        Valeurs non convertibles -> NaN (jamais d'exception).
        Garantit la cohérence train / test.
        """
        if not self.numeric_string_cols_:
            return df
        df_out = df.copy()
        for col in self.numeric_string_cols_:
            if col not in df_out.columns:
                continue
            if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                df_out[col] = df_out[col].apply(self._clean_numeric_str_value)
                logger.debug(f"Colonne '{col}' convertie via numeric_string_cols_ (transform, F2)")
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

    def _preprocess_raw_dataframe(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        is_fit: bool = False,
    ) -> pd.DataFrame:
        """Exécute l'étape 0 de nettoyage initial sur le DataFrame.

        Args:
            df: DataFrame à nettoyer.
            target: Nom de la colonne cible (à ne pas renommer).
            is_fit: True pendant fit() -> _parse_numeric_strings (détection + mémorisation F2).
                    False pendant transform() -> _apply_numeric_string_cols (application stricte F2).
        """
        self._last_keep_positions_ = np.ones(len(df), dtype=bool)
        df_clean = self._normalize_column_names(df, target=target)
        df_clean = self._clean_string_and_na(df_clean)
        if is_fit:
            df_clean = self._parse_numeric_strings(df_clean)
        else:
            # F2 : appliquer uniquement les colonnes memorisees au fit, sans recalculer de seuil
            df_clean = self._apply_numeric_string_cols(df_clean)
        for col in list(df_clean.columns):
            date_action, _ = self._override_value(str(col), "clean_dates")
            type_action, _ = self._override_value(str(col), "type")
            if type_action == "KEEP_RAW":
                continue
            if date_action == "DROP_COLUMN":
                self.cleaning_decisions_.setdefault(str(col), {})["action_dates"] = date_action
                df_clean = df_clean.drop(columns=[col])
                continue
            if date_action not in {"AUTO", "PARSE", "KEEP"}:
                raise ValueError(f"Action clean_dates invalide pour {col!r}: {date_action}")
            if type_action == "FORCE_NUMERIC":
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            elif type_action == "FORCE_CATEGORICAL":
                df_clean[col] = df_clean[col].astype("string")
            elif type_action == "FORCE_TEXT":
                df_clean[col] = df_clean[col].astype("string")
            elif type_action == "FORCE_DATE" or date_action == "PARSE":
                raw_date = df_clean[col]
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce", format="mixed")
                self.cleaning_decisions_.setdefault(str(col), {}).update({
                    "dates_parsables_pct": float(df_clean[col].notna().mean() * 100),
                    "dates_non_parsables": int((raw_date.notna() & df_clean[col].isna()).sum()),
                    "action_dates": date_action if date_action != "AUTO" else type_action,
                })
            elif type_action not in {"AUTO", "FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT", "KEEP_RAW", "DROP"}:
                raise ValueError(f"Type forcé invalide pour {col!r}: {type_action}")
        df_clean = self._apply_impossible_value_rules(df_clean)
        df_clean = self._downcast_memory(df_clean)
        return df_clean

    def _apply_impossible_value_rules(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Applique les règles explicites d'impossibilité et mémorise les lignes retirées."""
        out = frame.copy()
        keep = np.ones(len(out), dtype=bool)
        self.impossible_counts_ = {}
        for col in out.columns:
            type_action, _ = self._override_value(str(col), "type")
            if type_action == "KEEP_RAW":
                continue
            action, _ = self._override_value(str(col), "clean_impossible")
            if action not in {"AUTO", "NAN", "CLIP", "DROP_ROWS", "IGNORE"}:
                raise ValueError(f"Action clean_impossible invalide pour {col!r}: {action}")
            series = out[col]
            lower = upper = None
            name = str(col).casefold()
            if re.search(r"(^|_)(age|ages)(_|$)", name):
                lower, upper = 0.0, 120.0
            elif re.search(r"percent|pourcent|pct|ratio|(^|_)rate(_|$)", name):
                lower, upper = 0.0, 100.0
            elif re.search(r"prix|price|montant|amount|surface|quantity|quantite", name):
                lower = 0.0
            mask = pd.Series(False, index=out.index)
            if lower is not None or upper is not None:
                numeric = pd.to_numeric(series, errors="coerce")
                if lower is not None:
                    mask |= numeric < lower
                if upper is not None:
                    mask |= numeric > upper
            if pd.api.types.is_datetime64_any_dtype(series) and not _is_time_only_column(col, series):
                now = _current_timestamp(series.dt.tz).normalize()
                mask |= series > now
            count = int(mask.sum())
            self.impossible_counts_[str(col)] = count
            selected = "NAN" if action == "AUTO" and count else action
            self.cleaning_decisions_.setdefault(str(col), {}).update({
                "valeurs_impossibles": count, "action_valeurs_impossibles": selected,
            })
            if count == 0 or selected in {"AUTO", "IGNORE"}:
                continue
            if selected == "NAN":
                out.loc[mask, col] = np.nan
            elif selected == "CLIP":
                if pd.api.types.is_datetime64_any_dtype(series):
                    now = _current_timestamp(series.dt.tz).normalize()
                    out.loc[mask, col] = now
                else:
                    out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=lower, upper=upper)
            elif selected == "DROP_ROWS":
                keep &= ~mask.to_numpy()
        if not keep.all():
            self._last_keep_positions_ = np.asarray(self._last_keep_positions_)[keep]
            out = out.loc[keep]
        return out

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

            forced_type, _ = self._override_value(str(col), "type")
            if forced_type in {"FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT"}:
                if forced_type == "FORCE_NUMERIC":
                    meta["type"] = FeatureType.NUMERIC_CONTINUOUS
                    meta["reason"] = "Type forcé par l'utilisateur : numérique."
                    self.num_cols_.append(col)
                elif forced_type == "FORCE_DATE":
                    meta["type"] = FeatureType.DATETIME
                    meta["reason"] = "Type forcé par l'utilisateur : date."
                    self.date_cols_.append(col)
                elif forced_type == "FORCE_TEXT":
                    meta["type"] = FeatureType.TEXT
                    meta["reason"] = "Type forcé par l'utilisateur : texte."
                    self.text_cols_.append(col)
                else:
                    meta["type"] = FeatureType.CATEGORICAL_LOW_CARD if n_unique <= self.config.cardinality_threshold else FeatureType.CATEGORICAL_HIGH_CARD
                    meta["reason"] = "Type forcé par l'utilisateur : catégoriel."
                    (self.low_card_cols_ if n_unique <= self.config.cardinality_threshold else self.high_card_cols_).append(col)
                self.column_metadata_[col] = meta
                continue

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
            identifier_name = bool(re.search(r"(^|_)(id|code|ref|num)(_|$)", str(col).strip().lower()))
            unique_ratio = n_unique / max(n_rows, 1)
            is_id_candidate = (
                n_rows > 30
                and (identifier_name or (unique_ratio > self.config.identifier_unique_threshold and
                     (pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series)
                      or pd.api.types.is_string_dtype(series))))
            )
            # Vérifier si ce n'est pas un texte long
            # Une colonne date/heure quasi unique (horodatage) n'est pas un ID
            date_clean_action, _ = self._override_value(str(col), "clean_dates")
            if is_id_candidate and date_clean_action != "KEEP" and self._is_date_column(series):
                is_id_candidate = False
            if is_id_candidate:
                mean_str_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
                if mean_str_len < 30:
                    meta["type"] = FeatureType.DROP_ID
                    meta["reason"] = (f"Identifiant détecté : ratio uniques/lignes = {unique_ratio:.1%} "
                                       f"(seuil > {self.config.identifier_unique_threshold:.0%})" if unique_ratio > self.config.identifier_unique_threshold else
                                       f"Nom évocateur d'un identifiant ({col!r}); ratio uniques/lignes = {unique_ratio:.1%}")
                    meta["action"] = "DROP"
                    self.columns_to_drop_.append(col)
                    self.column_metadata_[col] = meta
                    continue

            # 5. Détection Datetime
            if date_clean_action != "KEEP" and self._is_date_column(series):
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
    # GARDE-FOU CONTRE LES COLONNES DE BRUIT (F1)
    # -------------------------------------------------------------------------

    def _detect_noise_columns(self, X_work: pd.DataFrame, y_arr: np.ndarray) -> None:
        """Teste le signal de chaque colonne candidate au bruit (DATETIME, CATEGORICAL_HIGH_CARD).

        Algorithme (sans fuite) :
        - Pour chaque colonne candidate, mesurer le score CV avec la vraie cible
          (arbre de profondeur 4, meme plis que la detection de fuite).
        - Mesurer le meme score avec la cible melangee (permutation) sur les memes plis.
        - Une colonne est 'bruit' si son score reel n'est pas superieur au score permute
          d'au moins ``noise_margin``.
        - En mode 'warn' : entree dans warnings_ et noise_cols_detected_, rien de supprime.
        - En mode 'drop' : suppression conditionnelle si les donnees sont suffisantes
          (min_rows_for_noise_guard lignes, 30 positifs en classification).
          En cas de donnees insuffisantes : retour en 'warn' avec message explicatif.
        - Les colonnes listees dans keep_columns ne sont JAMAIS supprimees.

        Les colonnes de dates et haute cardinalite sont candidates car ce sont elles
        qui produisaient ~14 features de bruit sur le dataset fraude de reference.
        """
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        mode = getattr(self.config, "noise_guard", "warn")
        if mode == "off":
            return

        margin = getattr(self.config, "noise_margin", 0.03)
        min_rows = getattr(self.config, "min_rows_for_noise_guard", 300)
        keep_cols = set(getattr(self.config, "keep_columns", None) or [])

        y_all = np.asarray(y_arr)
        n_rows = len(y_all)
        rs = self.config.random_state
        is_clf = self.task_resolved_ == "classification"

        # Verifier si les donnees sont suffisantes pour le mode 'drop'
        can_drop = mode == "drop"
        if can_drop:
            minority_count = 0
            if is_clf:
                y_int = y_all.astype(int)
                counts = np.bincount(y_int[y_int >= 0])
                minority_count = int(counts.min()) if len(counts) >= 2 else 0
            insufficient_reason = ""
            if n_rows < min_rows:
                insufficient_reason = (
                    f"seulement {n_rows} lignes < {min_rows} (min_rows_for_noise_guard)"
                )
            elif is_clf and minority_count < 30:
                insufficient_reason = (
                    f"seulement {minority_count} exemples de la classe minoritaire < 30"
                )
            if insufficient_reason:
                can_drop = False
                msg = (
                    f"noise_guard='drop' demande mais mesure instable "
                    f"({insufficient_reason}) : retour en 'warn'."
                )
                self.warnings_.append(msg)
                logger.warning(msg)

        # Colonnes candidates : dates et haute cardinalite (generatrices de bruit connu)
        candidates = list(set(self.date_cols_) | set(self.high_card_cols_))
        candidates = [c for c in candidates if c not in self.columns_to_drop_]
        if not candidates:
            return

        # CV commun aux deux mesures (reel et permute)
        mask = ~pd.isna(y_all)
        if mask.sum() < 30:
            return
        X_d = X_work.iloc[np.where(mask)[0]]
        y_d = y_all[mask]

        if is_clf:
            y_d_int = y_d.astype(int)
            counts = np.bincount(y_d_int)
            counts = counts[counts > 0]
            if len(counts) < 2 or counts.min() < 2:
                return
            scoring = "roc_auc" if len(counts) == 2 else "balanced_accuracy"
            cv = StratifiedKFold(n_splits=int(min(5, counts.min())), shuffle=True, random_state=rs)
        else:
            scoring = "r2"
            cv = KFold(n_splits=5, shuffle=True, random_state=rs)

        rng_perm = np.random.RandomState(rs)
        y_permuted = rng_perm.permutation(y_d)
        if is_clf:
            y_permuted = y_permuted.astype(int)

        for col in candidates:
            if col not in X_d.columns:
                continue

            s = X_d[col]
            tree_real = (
                DecisionTreeClassifier(max_depth=4, random_state=rs) if is_clf
                else DecisionTreeRegressor(max_depth=4, random_state=rs)
            )
            tree_perm = (
                DecisionTreeClassifier(max_depth=4, random_state=rs) if is_clf
                else DecisionTreeRegressor(max_depth=4, random_state=rs)
            )

            try:
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    v = s.astype(float)
                    v = v.fillna(float(v.min()) - 1.0 if v.notna().any() else 0.0).to_numpy().reshape(-1, 1)
                    est_real = tree_real
                    est_perm = tree_perm
                else:
                    from sklearn.preprocessing import TargetEncoder as _TE
                    v = s.astype(str).to_numpy().reshape(-1, 1)
                    est_real = make_pipeline(_TE(smooth="auto", cv=5, random_state=rs), tree_real)
                    est_perm = make_pipeline(_TE(smooth="auto", cv=5, random_state=rs), tree_perm)

                score_real = float(cross_val_score(est_real, v, y_d if is_clf else y_d, cv=cv, scoring=scoring).mean())
                score_perm = float(cross_val_score(est_perm, v, y_permuted, cv=cv, scoring=scoring).mean())
            except Exception as exc:
                logger.debug(f"Noise guard ignore pour '{col}' : {exc}")
                continue

            if np.isnan(score_real) or np.isnan(score_perm):
                continue

            is_noise = (score_real - score_perm) < margin
            logger.debug(
                f"Noise guard '{col}': score_reel={score_real:.3f} "
                f"score_permute={score_perm:.3f} "
                f"marge={score_real - score_perm:.3f} -> {'bruit' if is_noise else 'signal'}"
            )

            if not is_noise:
                continue

            # Colonne identifiee comme bruit
            self.noise_cols_detected_.append(col)
            msg = (
                f"BRUIT SUSPECT : la colonne '{col}' n'apporte pas de signal "
                f"(score_reel={score_real:.3f} vs permute={score_perm:.3f}, "
                f"marge={score_real - score_perm:.3f} < {margin}). "
                f"Mode noise_guard='{mode}'."
            )
            self.warnings_.append(msg)
            logger.warning(msg)

            if not can_drop:
                continue

            # Verifier keep_columns avant suppression
            if col in keep_cols:
                msg_keep = f"Colonne '{col}' protegee par keep_columns : conservee malgre le bruit."
                self.warnings_.append(msg_keep)
                logger.info(msg_keep)
                continue

            # Supprimer la colonne
            self.noise_cols_dropped_.append(col)
            for lst in (self.date_cols_, self.high_card_cols_):
                if col in lst:
                    lst.remove(col)
            if col not in self.columns_to_drop_:
                self.columns_to_drop_.append(col)
            meta = self.column_metadata_.get(col, {})
            meta["action"] = "DROP"
            meta["reason"] = (
                f"Bruit detecte par noise_guard (score={score_real:.3f} vs "
                f"permute={score_perm:.3f}, marge={score_real - score_perm:.3f})"
            )
            logger.info(f"Colonne '{col}' supprimee par noise_guard (bruit confirme).")

    # -------------------------------------------------------------------------
    # GESTION DE LA CIBLE
    # -------------------------------------------------------------------------


    def _resolve_task(self, y: Union[pd.Series, np.ndarray]) -> str:
        """Déduit automatiquement si la tache est une classification ou régression."""
        if self.task == "regression":
            y_num = pd.to_numeric(pd.Series(y).dropna(), errors="coerce")
            if y_num.isna().any():
                msg = ("La tache « regression » est demandée mais la cible contient du texte "
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

    def analyze(self, df: pd.DataFrame, target: Optional[str] = None) -> AnalysisResult:
        """Analyse les colonnes sans ajuster le préprocesseur ni modifier ``df``.

        Au-delà de 200 000 lignes, les statistiques sont estimées sur un échantillon
        déterministe de 200 000 lignes; le détecteur de fuite utilise le même détecteur
        que le fit, sur cet échantillon.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        raw = df.copy(deep=True)
        sample_note = ""
        if len(raw) > 200_000:
            raw = raw.sample(n=200_000, random_state=self.config.random_state)
            sample_note = "Analyse statistique sur échantillon déterministe de 200 000 lignes."
        prep = AutoPreprocessor(copy.deepcopy(self.config), task=self.task)
        if target is not None and target not in raw.columns:
            raise ValueError(f"La colonne cible '{target}' est absente du DataFrame.")
        feature_raw = raw.drop(columns=[target]) if target else raw
        clean = prep._preprocess_raw_dataframe(feature_raw, is_fit=True)
        y_arr = None
        if target is not None:
            y_arr = prep._fit_target(raw[target])
            keep_rows = np.asarray(prep._last_keep_positions_, dtype=bool)
            if len(keep_rows) == len(y_arr) and not keep_rows.all():
                y_arr = y_arr[keep_rows]
        X = clean
        prep._detect_types(X)
        leakage = {}
        if y_arr is not None:
            # Réutilise la routine existante de détection; l'instance n'est pas ajustée.
            prep._detect_leaky_columns(X.drop(columns=prep.columns_to_drop_, errors="ignore"), y_arr)
            leakage = prep.leakage_scores_
        rows = []
        for col in df.columns:
            if target is not None and col == target:
                rows.append({"colonne": str(col), "type": "TARGET", "dtype": str(df[col].dtype),
                             "manquants_pct": float(raw[col].isna().mean() * 100) if len(raw) else 0.0,
                             "uniques": int(raw[col].nunique(dropna=True)), "exemples": raw[col].dropna().astype(str).head(5).tolist(),
                             "relation_cible": np.nan, "score_fuite": np.nan, "action": "—",
                             "raison": "Colonne cible : exclue des recommandations de transformation des features.",
                             "confiance": "", "alternatives": [], "steps": {}, "est_cible": True})
                continue
            clean_name = col
            if clean_name not in prep.column_metadata_:
                clean_name = next((c for c in prep.column_metadata_ if str(c) == str(col)), col)
            raw_s = raw[col]
            s = clean[clean_name] if clean_name in clean.columns else raw_s
            meta = prep.column_metadata_.get(clean_name, {})
            kind = meta.get("type", FeatureType.CATEGORICAL_LOW_CARD)
            kind_name = kind.value if isinstance(kind, FeatureType) else str(kind)
            nonnull = s.dropna()
            nunique = int(nonnull.nunique())
            missing = float(s.isna().mean()) if len(s) else 0.0
            examples = nonnull.astype(str).head(5).tolist()
            text_values = raw_s.dropna().astype(str).str.strip()
            false_missing = int(text_values.str.lower().isin(DEFAULT_NA_VALUES).sum())
            normalized_values = text_values.str.casefold()
            if len(text_values):
                variant_frame = pd.DataFrame({"raw": text_values, "normalized": normalized_values})
                variants = variant_frame.groupby("normalized")["raw"].nunique()
                variant_count = int((variants - 1).clip(lower=0).sum())
            else:
                variant_count = 0
            action, confidence = Action.ONEHOT, "moyenne"
            alternatives: List[Action] = [Action.DROP]
            reason = ""
            stats_row: Dict[str, Any] = {}
            step_recs: Dict[str, Dict[str, Any]] = {}
            if kind_name.startswith("DROP"):
                action, confidence = Action.DROP, "haute"
                reason = meta.get("reason", "Colonne constante, identifiante ou trop manquante.")
            elif kind_name == "DATETIME":
                action = Action.DATE_FEATURES
                alternatives = [Action.DROP]
                time_only = _is_time_only_column(col, nonnull)
                dates = _parse_datetime_values(nonnull, time_only)
                parsed_dates = dates.dropna().sort_values()
                gaps = parsed_dates.diff().dropna().dt.total_seconds()
                median_gap = float(gaps.median()) if len(gaps) else np.nan
                if pd.isna(median_gap):
                    granularity = "indéterminée"
                elif median_gap < 60:
                    granularity = "seconde"
                elif median_gap < 3600:
                    granularity = "minute"
                elif median_gap < 86400:
                    granularity = "heure"
                elif median_gap < 30 * 86400:
                    granularity = "jour"
                elif median_gap < 365 * 86400:
                    granularity = "mois"
                else:
                    granularity = "année"
                stats_row.update(date_min=dates.min(), date_max=dates.max(), uniques=nunique,
                                 date_parse_rate=float(dates.notna().mean()) if len(dates) else 0.0,
                                 date_granularity=granularity, time_only=time_only)
                reason = f"{nunique} dates distinctes; extraction des composantes calendaires recommandée."
            elif kind_name == "TEXT":
                action = Action.TFIDF_SVD
                alternatives = [Action.DROP, Action.ONEHOT]
                reason = f"Texte libre détecté ({nunique} valeurs distinctes); TF-IDF/SVD convient aux textes longs."
            elif kind_name == "BINARY":
                action = Action.BINARY_MAP
                alternatives = [Action.DROP, Action.ONEHOT]
                reason = f"Deux modalités observées; un mappage binaire produit une seule variable."
            elif kind_name == "ORDINAL":
                action = Action.ORDINAL
                alternatives = [Action.ONEHOT, Action.DROP]
                reason = f"Ordre ordinal reconnu parmi {nunique} modalités; l'encodage conserve cet ordre."
            elif kind_name.startswith("CATEGORICAL") or kind_name == "NUMERIC_DISCRETE":
                action = (Action.TARGET_ENCODE if target else Action.FREQUENCY_ENCODE) if kind_name.endswith("HIGH_CARD") else Action.ONEHOT
                alternatives = [Action.ONEHOT, Action.TARGET_ENCODE, Action.FREQUENCY_ENCODE, Action.DROP]
                top = nonnull.astype(str).value_counts().head(5).to_dict()
                top_frequency = float(nonnull.astype(str).value_counts(normalize=True).iloc[0]) if len(nonnull) else np.nan
                rare = float((nonnull.astype(str).value_counts(normalize=True) < self.config.rare_category_threshold).sum() / max(nunique, 1))
                stats_row.update(top_categories=top, top_category_frequency=top_frequency,
                                 rare_category_pct=rare * 100, cardinality=nunique)
                probs = nonnull.astype(str).value_counts(normalize=True)
                stats_row["entropy"] = float(-(probs * np.log2(probs.clip(lower=1e-15))).sum())
                stats_row["natural_order"] = self._match_ordinal_pattern(nonnull.unique().tolist()) or []
                reason = f"Cardinalité de {nunique}; encodage {'supervisé régularisé' if action == Action.TARGET_ENCODE else 'One-Hot'} adapté au type détecté."
            else:
                vals = pd.to_numeric(nonnull, errors="coerce").dropna()
                skew = float(vals.skew()) if len(vals) > 2 else 0.0
                q1, q3 = vals.quantile(.25), vals.quantile(.75)
                iqr = q3 - q1
                outlier_pct = float((((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).mean()) * 100) if len(vals) else 0.0
                stats_row.update(min=vals.min() if len(vals) else np.nan, max=vals.max() if len(vals) else np.nan,
                                 mean=vals.mean() if len(vals) else np.nan, median=vals.median() if len(vals) else np.nan,
                                 std=vals.std() if len(vals) else np.nan, skew=skew,
                                 kurtosis=float(vals.kurt()) if len(vals) > 3 else np.nan, outliers_pct=outlier_pct)
                mad = float(np.median(np.abs(vals - np.median(vals)))) if len(vals) else 0.0
                mad_rate = float((np.abs(vals - np.median(vals)) > 3.5 * 1.4826 * mad).mean() * 100) if mad else 0.0
                cv = float(vals.std() / abs(vals.mean())) if len(vals) and vals.mean() else np.nan
                normality_p = np.nan
                shapiro_p = np.nan
                if len(vals) >= 8:
                    try:
                        norm_sample = vals.sample(min(5000, len(vals)), random_state=42)
                        normality_p = float(stats.normaltest(norm_sample).pvalue)
                        shapiro_p = float(stats.shapiro(norm_sample).pvalue)
                    except Exception:
                        pass
                zero_pct = float((vals == 0).mean() * 100) if len(vals) else 0.0
                stats_row.update(outliers_mad_pct=mad_rate, coefficient_variation=cv,
                                 normality_p=normality_p, shapiro_p=shapiro_p, zero_pct=zero_pct)
                action = Action.YEO_JOHNSON if abs(skew) > self.config.skew_threshold else Action.KEEP_NUMERIC
                alternatives = [Action.DROP, Action.YEO_JOHNSON, Action.LOG1P, Action.WINSORIZE,
                                Action.SCALE_STANDARD, Action.SCALE_ROBUST, Action.SCALE_MINMAX, Action.NO_SCALE]
                reason = (f"Asymétrie = {skew:.2f}; Yeo-Johnson est activé par défaut au-dessus de "
                          f"{self.config.skew_threshold:g}." if action == Action.YEO_JOHNSON else
                          f"Asymétrie = {skew:.2f}, sous le seuil {self.config.skew_threshold:g}; traitement numérique standard.")
            # Les 11 étapes sont explicites. Les étapes non configurées reprennent le comportement du pipeline.
            def rec(action_name: str, measure: str, threshold: str, explanation: str,
                    choices: List[str], conf: str = "moyenne") -> Dict[str, Any]:
                return {"action": action_name, "measure": measure, "threshold": threshold,
                        "explanation": explanation, "confidence": conf, "choices": choices}
            step_recs["nettoyage"] = rec("CLEAN_DEFAULT", f"{missing:.1%} de valeurs manquantes", "règles de nettoyage du module",
                "Nettoyage des sous-étapes sélectionnées.", ["AUTO"])
            false_tokens = text_values.str.strip().str.casefold().isin(DEFAULT_NA_VALUES)
            found_false = text_values[false_tokens].value_counts().to_dict()
            raw_nonnull = raw_s.dropna()
            numeric_converted = (raw_nonnull.map(self._clean_numeric_str_value)
                                 if (pd.api.types.is_object_dtype(raw_s) or pd.api.types.is_string_dtype(raw_s))
                                 else pd.Series(dtype=float))
            numeric_text_ratio = float(numeric_converted.notna().mean()) if len(numeric_converted) else 0.0
            convertible_values = raw_nonnull[numeric_converted.notna()] if len(numeric_converted) else pd.Series(dtype=object)
            units = sorted(set(convertible_values.astype(str).str.extract(r"([A-Za-z%]{1,8})\s*$", expand=False).dropna().str.upper())) if len(convertible_values) else []
            date_parse_rate = 0.0
            mixed_date_formats = 0
            if len(nonnull) and (kind_name == "DATETIME" or re.search(r"date|heure|time", str(col), re.I)):
                date_source = raw_s.dropna()
                time_only = _is_time_only_column(col, date_source)
                parsed_dates = _parse_datetime_values(date_source, time_only)
                date_parse_rate = float(parsed_dates.notna().mean())
                if not time_only:
                    format_keys = date_source.astype(str).map(lambda v: "ymd" if re.match(r"^\d{4}[-/]", v) else "dmy" if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", v) else "other")
                    mixed_date_formats = max(format_keys.nunique() - 1, 0)
            impossible_count = int(getattr(prep, "impossible_counts_", {}).get(str(col), 0))
            duplicate_rows = int(raw.duplicated().sum())
            step_recs["clean_missing_tokens"] = rec(
                "TO_NAN", f"{int(false_tokens.sum())} cellules; marqueurs trouvés : {found_false}", "marqueurs de valeurs manquantes reconnus",
                f"{int(false_tokens.sum())} faux manquants détectés: {found_false}.", ["TO_NAN", "KEEP", "SELECTED_TO_NAN"])
            step_recs["clean_variants"] = rec(
                "MERGE", f"{variant_count} variantes fusionnables", "même texte après trim/casse/accents",
                f"{variant_count} représentation(s) redondante(s) après normalisation des espaces, casse et accents.", ["MERGE", "KEEP"])
            step_recs["clean_numeric_text"] = rec(
                "CONVERT" if numeric_text_ratio >= self.config.numeric_string_threshold else "KEEP",
                f"{numeric_text_ratio:.1%} de nombres convertibles; unités : {units}", f"conversion si ≥ {self.config.numeric_string_threshold:.0%}",
                f"{numeric_text_ratio:.1%} des cellules textuelles sont numériques; unités repérées: {units or 'aucune'}.", ["AUTO", "CONVERT", "KEEP"])
            step_recs["clean_dates"] = rec(
                "PARSE" if date_parse_rate >= .8 and date_parse_rate > 0 else "KEEP",
                f"{date_parse_rate:.1%} parsables; {1-date_parse_rate:.1%} non parsables; formats mélangés : {mixed_date_formats}", "parser si ≥ 80 %",
                f"{date_parse_rate:.1%} des valeurs sont parsables; formats mélangés estimés à {mixed_date_formats}.", ["AUTO", "PARSE", "KEEP", "DROP_COLUMN"])
            step_recs["clean_impossible"] = rec(
                "NAN" if impossible_count else "IGNORE", f"cas impossibles : {impossible_count}",
                "age [0,120], pourcentage [0,100], montants/surfaces ≥ 0, dates ≤ aujourd'hui",
                f"{impossible_count} valeur(s) hors des bornes métier codées ont été détectées.", ["AUTO", "NAN", "CLIP", "DROP_ROWS", "IGNORE"])
            step_recs["duplicates"] = rec(
                "DROP" if duplicate_rows else "KEEP", f"lignes dupliquées : {duplicate_rows}", "égalité exacte de toutes les cellules",
                f"{duplicate_rows} ligne(s) dupliquée(s) exacte(s) dans le dataset.", ["DROP", "KEEP"])
            step_recs["type"] = rec("DROP" if kind_name.startswith("DROP") else "AUTO", f"{nunique} valeurs uniques", f"cardinalité au plus {self.config.cardinality_threshold} modalités",
                f"Type détecté : {kind_name}; {reason}", ["AUTO", "DROP", "NUMERIC", "CATEGORICAL", "DATETIME", "TEXT"], confidence)
            step_recs["missing"] = rec("MEDIAN" if kind_name.startswith("NUMERIC") else "AUTO",
                f"{missing:.1%} de valeurs manquantes", f"indicateur si > {self.config.missing_indicator_threshold:.1%}",
                f"Le taux manquant observé est {missing:.1%}; imputation ajustée sur train.",
                ["AUTO", "MEDIAN", "MEAN", "MODE", "CONSTANT", "MISSING_INDICATOR"])
            step_recs["outliers"] = rec("WINSORIZE" if kind_name.startswith("NUMERIC") else "AUTO",
                f"outliers IQR : {stats_row.get('outliers_pct', 0):.2f}%; outliers MAD : {stats_row.get('outliers_mad_pct', 0):.2f}%",
                "bornes IQR 1,5; MAD robuste 3,5",
                "Le clipping par quantiles est appliqué selon la configuration du pipeline.",
                ["AUTO", "NONE", "WINSORIZE", "IQR"])
            normality_p = stats_row.get("normality_p", np.nan)
            p_text = ("p < 0.001" if pd.notna(normality_p) and normality_p < .001 else
                      f"p = {normality_p:.3g}" if pd.notna(normality_p) else "p-value indisponible")
            normality_text = (f"normalité rejetée ({p_text})" if pd.notna(normality_p) and normality_p < .05 else
                              f"normalité non rejetée ({p_text})")
            skew = float(stats_row.get("skew", np.nan))
            skew_reaches_threshold = abs(skew) > self.config.skew_threshold
            step_recs["distribution"] = rec(action.value if action == Action.YEO_JOHNSON else "AUTO",
                f"asymétrie = {skew:.2f}; {normality_text}",
                f"seuil |skew| > {self.config.skew_threshold:g} {'atteint' if skew_reaches_threshold else 'non atteint'}", reason,
                ["AUTO", "NONE", "LOG1P", "BOX_COX", "YEO_JOHNSON", "SQRT"])
            scale_recommendation = "AUTO"
            scale_measure = "type non numérique"
            scale_threshold = "aucun"
            scale_reason = "Aucun scaler numérique recommandé pour cette famille de colonnes."
            if kind_name.startswith("NUMERIC"):
                scale_recommendation, scale_measure, scale_threshold = _recommend_scaler(s)
                scale_reason = f"{scale_measure}; seuil {scale_threshold} → {scale_recommendation}."
            step_recs["scaling"] = rec(scale_recommendation, scale_measure, scale_threshold,
                scale_reason, ["AUTO", "NONE", "STANDARD", "ROBUST", "MINMAX", "MAXABS"])
            step_recs["encoding"] = rec(action.value if action in (Action.ONEHOT, Action.TARGET_ENCODE, Action.FREQUENCY_ENCODE, Action.ORDINAL, Action.BINARY_MAP) else "NONE",
                f"{nunique} modalités", f"haute cardinalité > {self.config.cardinality_threshold} modalités", reason,
                ["AUTO", "ONEHOT", "ORDINAL", "TARGET_ENCODE", "FREQUENCY_ENCODE", "BINARY_MAP"])
            step_recs["dates"] = rec("DATE_FEATURES" if kind_name == "DATETIME" else "AUTO", f"{nunique} valeurs uniques", "date parsable",
                "Les composantes de date sont extraites si le type est détecté comme date.", ["AUTO", "DATE_FEATURES"])
            if kind_name == "DATETIME":
                min_date = stats_row.get("date_min")
                max_date = stats_row.get("date_max")
                if stats_row.get("time_only") and pd.notna(min_date) and pd.notna(max_date):
                    date_range = f"heures {min_date:%H:%M:%S} à {max_date:%H:%M:%S}, colonne horaire sans date"
                else:
                    date_range = (f"plage {min_date:%Y-%m-%d} à {max_date:%Y-%m-%d}"
                              if pd.notna(min_date) and pd.notna(max_date) else "plage indisponible")
                parse_pct = float(stats_row.get("date_parse_rate", 0.0)) * 100
                granularity = stats_row.get("date_granularity", "indéterminée")
                step_recs["dates"].update(
                    measure=f"{date_range}; valeurs parsables {parse_pct:.1f}%; granularité {granularity}",
                    threshold="date reconnue; parsing ≥ 80%",
                    explanation="La plage, le taux de parsing et la granularité justifient l'extraction des composantes calendaires.")
            step_recs["text"] = rec("TFIDF_SVD" if kind_name == "TEXT" else "AUTO", f"{nunique} valeurs uniques", "texte long détecté",
                "TF-IDF/SVD est utilisé pour les colonnes détectées comme texte libre.", ["AUTO", "TFIDF_SVD"])
            step_recs["target"] = rec("AUTO" if target else "NONE", "cible fournie" if target else "aucune cible", "classification/régression",
                "La cible est traitée séparément; les statistiques de features ne l'ajustent pas.", ["NONE", "AUTO", "TRANSFORM"])
            step_recs["selection"] = rec("AUTO", f"{nunique} valeurs uniques", f"variance > {self.config.variance_threshold}",
                "La sélection finale est ajustée exclusivement sur les données de train.", ["AUTO", "KEEP", "DROP"])
            groups = _action_group(kind)
            for step_name, groups_for_step in SUPPORTED_ACTIONS.items():
                supported = groups_for_step.get(groups, groups_for_step.get("any", ["AUTO"]))
                if step_name in step_recs:
                    step_recs[step_name]["choices"] = list(supported)
                    if step_recs[step_name]["action"] not in supported and "AUTO" in supported:
                        step_recs[step_name]["action"] = "AUTO"
            action_steps = {
                "DROP": ("type", "DROP"), "KEEP_NUMERIC": ("type", "AUTO"),
                "YEO_JOHNSON": ("distribution", "YEO_JOHNSON"), "LOG1P": ("distribution", "LOG1P"),
                "WINSORIZE": ("outliers", "WINSORIZE"), "SCALE_STANDARD": ("scaling", "STANDARD"),
                "SCALE_ROBUST": ("scaling", "ROBUST"), "SCALE_MINMAX": ("scaling", "MINMAX"),
                "NO_SCALE": ("scaling", "NONE"), "ONEHOT": ("encoding", "ONEHOT"),
                "TARGET_ENCODE": ("encoding", "TARGET_ENCODE"), "FREQUENCY_ENCODE": ("encoding", "FREQUENCY_ENCODE"),
                "ORDINAL": ("encoding", "ORDINAL"), "BINARY_MAP": ("encoding", "BINARY_MAP"),
                "DATE_FEATURES": ("dates", "DATE_FEATURES"), "TFIDF_SVD": ("text", "TFIDF_SVD"),
            }
            alternatives = [candidate for candidate in alternatives
                            if candidate.value in action_steps and
                            _action_supported(action_steps[candidate.value][0], groups,
                                              action_steps[candidate.value][1])]
            score = leakage.get(clean_name, np.nan)
            if clean_name in prep.suspected_leakage_cols_:
                action, confidence = Action.DROP, "haute"
                reason = f"Fuite de cible suspectée (score {score:.3f}); suppression recommandée."
                step_recs["type"]["action"] = "DROP"
                step_recs["type"]["measure"] = f"score fuite CV : {score:.3f}"
                step_recs["type"]["threshold"] = f"score ≥ {self.config.leakage_threshold:.3f} et écart ≥ {self.config.leakage_min_gap:.3f}"
                step_recs["type"]["explanation"] = reason
            elif confidence != "haute":
                confidence = "moyenne" if nunique > 1 else "faible"
            target_relation = np.nan
            target_spearman = np.nan
            target_cramer = np.nan
            target_mi = np.nan
            target_relation_method = ""
            relation_unreliable_identifier = False
            missing_target_p = np.nan
            if target:
                aligned = pd.DataFrame({"x": s, "target": raw[target].reindex(s.index)})
                if len(aligned) >= 8:
                    try:
                        x_num = pd.to_numeric(aligned["x"], errors="coerce")
                        y_num = pd.to_numeric(aligned["target"], errors="coerce")
                        valid = x_num.notna() & y_num.notna()
                        if valid.sum() >= 8:
                            target_spearman = float(stats.spearmanr(x_num[valid], y_num[valid]).statistic)
                    except (TypeError, ValueError):
                        pass
                    try:
                        codes = pd.factorize(aligned["x"].fillna("__MISSING__").astype(str))[0].reshape(-1, 1)
                        yy = pd.factorize(aligned["target"].astype(str))[0] if prep.task_resolved_ == "classification" else pd.to_numeric(aligned["target"], errors="coerce").fillna(0).to_numpy()
                        mi_fn = mutual_info_classif if prep.task_resolved_ == "classification" else mutual_info_regression
                        target_mi = float(mi_fn(codes, yy, discrete_features=True, random_state=self.config.random_state)[0])
                    except (ValueError, TypeError):
                        pass
                    if prep.task_resolved_ == "classification":
                        try:
                            ct = pd.crosstab(aligned["x"].fillna("__MISSING__").astype(str), aligned["target"].astype(str))
                            if ct.shape[0] > 1 and ct.shape[1] > 1:
                                chi2 = float(stats.chi2_contingency(ct)[0])
                                n_obs = ct.to_numpy().sum()
                                min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
                                if min_dim > 0 and n_obs > 0:
                                    target_cramer = float(np.sqrt(chi2 / (n_obs * min_dim)))
                        except Exception:
                            pass
                    missing_mask = aligned["x"].isna()
                    try:
                        if missing_mask.nunique() == 2 and prep.task_resolved_ == "classification":
                            table = pd.crosstab(missing_mask, aligned["target"])
                            if table.shape[0] > 1 and table.shape[1] > 1:
                                missing_target_p = float(stats.chi2_contingency(table)[1])
                        elif missing_mask.any() and (~missing_mask).any():
                            yvals = pd.to_numeric(aligned["target"], errors="coerce")
                            left, right = yvals[missing_mask].dropna(), yvals[~missing_mask].dropna()
                            if len(left) > 1 and len(right) > 1:
                                missing_target_p = float(stats.ttest_ind(left, right, equal_var=False).pvalue)
                    except (ValueError, TypeError):
                        pass
                if kind_name.startswith("NUMERIC"):
                    target_relation = (target_spearman if pd.notna(target_spearman) else
                                       target_mi if pd.notna(target_mi) else np.nan)
                elif kind_name in (FeatureType.CATEGORICAL_LOW_CARD.value, FeatureType.CATEGORICAL_HIGH_CARD.value,
                                   FeatureType.BINARY.value, FeatureType.ORDINAL.value) and prep.task_resolved_ == "classification":
                    target_relation = (target_cramer if pd.notna(target_cramer) else
                                       target_mi if pd.notna(target_mi) else np.nan)
                else:
                    target_relation = (target_mi if pd.notna(target_mi) else
                                       target_spearman if pd.notna(target_spearman) else
                                       target_cramer if pd.notna(target_cramer) else np.nan)
                unique_ratio = nunique / max(len(raw), 1)
                relation_unreliable_identifier = bool(
                    (unique_ratio > self.config.identifier_unique_threshold
                     and kind_name != FeatureType.DATETIME.value)
                    or kind_name == FeatureType.DROP_ID.value
                )
                if relation_unreliable_identifier:
                    target_relation = np.nan
                    target_spearman = target_cramer = target_mi = np.nan
                    target_relation_method = "non fiable (identifiant)"
                elif pd.notna(target_spearman) and kind_name.startswith("NUMERIC"):
                    target_relation_method = "Spearman (échelle [-1, 1])"
                elif pd.notna(target_cramer):
                    target_relation_method = "V de Cramér (échelle [0, 1])"
                elif pd.notna(target_mi):
                    target_relation_method = "information mutuelle (échelle [0, +∞[)"
                elif pd.notna(target_spearman):
                    target_relation_method = "Spearman (échelle [-1, 1])"
            if pd.notna(missing_target_p) and missing_target_p < 0.05:
                step_recs["missing"]["action"] = "MISSING_INDICATOR"
                missing_p_text = "p < 0.001" if missing_target_p < .001 else f"p = {missing_target_p:.3g}"
                step_recs["missing"]["measure"] += f"; {missing_p_text} pour le lien avec la cible"
                step_recs["missing"]["threshold"] = "p < 0,05"
                step_recs["missing"]["explanation"] = (
                    f"Le statut manquant est associé à la cible (p={missing_target_p:.3g} < 0,05); "
                    "un indicateur de manquant est recommandé."
                )
            cleanup_total = false_missing + variant_count
            if pd.api.types.is_object_dtype(raw_s) or pd.api.types.is_string_dtype(raw_s):
                parsed_numeric = raw_s.dropna().map(self._clean_numeric_str_value)
                numeric_text_count = int(parsed_numeric.notna().sum()) if len(parsed_numeric) else 0
                numeric_text_ratio = numeric_text_count / max(len(nonnull), 1)
                if numeric_text_ratio >= self.config.numeric_string_threshold:
                    cleanup_total += numeric_text_count
            else:
                numeric_text_count = 0
            date_format_count = 0
            if kind_name == "DATETIME":
                patterns = text_values.map(lambda value: (
                    "ymd" if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", value) else
                    "dmy" if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", value) else
                    "other"))
                date_format_count = int(max(patterns.nunique() - 1, 0))
            cleanup_total += date_format_count
            negative_impossible = int((pd.to_numeric(s, errors="coerce") < 0).sum()) if re.search(r"age|prix|price|montant|surface|quantite|quantity", str(col), re.I) else 0
            cleanup_total += negative_impossible
            step_recs["nettoyage"]["measure"] = (
                f"{false_missing} faux manquants; {variant_count} variantes; {numeric_text_count} nombres texte; "
                f"{date_format_count} formats de date; {negative_impossible} valeurs négatives suspectes; "
                f"{cleanup_total} cellules candidates"
            )
            step_recs["nettoyage"]["threshold"] = (
                f"nombres texte ≥ {self.config.numeric_string_threshold:.0%}; faux manquants connus; variantes normalisées"
            )
            step_recs["nettoyage"]["explanation"] = (
                f"Le nettoyage standard corrige les faux manquants ({false_missing}), variantes de casse/espaces "
                f"({variant_count}), nombres textuels détectés ({numeric_text_count}) et formats dates distincts "
                f"({date_format_count}); "
                f"{negative_impossible} valeurs négatives sont suspectes selon le nom de colonne."
            )
            sample_n = len(nonnull)
            if sample_n < 30 and confidence != "haute":
                confidence = "faible"
                for detail in step_recs.values():
                    detail["confidence"] = "faible"
                    detail["explanation"] += " Échantillon inférieur à 30 valeurs : interprétation prudente."
            for step_name, detail in step_recs.items():
                strength = 0.60
                p_evidence = None
                if step_name == "distribution":
                    if kind_name.startswith("NUMERIC"):
                        ratio = abs(skew) / max(self.config.skew_threshold, 1e-9)
                        if abs(skew) >= 2.0 or abs(skew) <= 0.20:
                            strength = 0.85
                        elif 0.85 <= ratio <= 1.15:
                            strength = 0.35
                        else:
                            strength = 0.60
                        p_evidence = normality_p if pd.notna(normality_p) else None
                    else:
                        strength = 0.60
                elif step_name == "missing":
                    if missing == 0:
                        strength = 0.90
                    elif 0.045 <= missing <= 0.055:
                        strength = 0.35
                    elif missing > 0.20:
                        strength = 0.85
                    else:
                        strength = 0.60
                    p_evidence = missing_target_p if pd.notna(missing_target_p) else None
                elif step_name == "scaling" and kind_name.startswith("NUMERIC"):
                    if zero_pct >= 50 or (stats_row.get("min", 0) >= 0 and stats_row.get("max", 1) <= 1) or outlier_pct > 10:
                        strength = 0.90
                    elif 4.5 <= outlier_pct <= 5.5:
                        strength = 0.35
                    elif outlier_pct <= 2.0 and pd.notna(normality_p) and normality_p > 0.05:
                        strength = 0.85
                    else:
                        strength = 0.60
                elif step_name == "clean_numeric_text":
                    if numeric_text_ratio >= 0.98 or numeric_text_ratio == 0:
                        strength = 0.90
                    elif 0.85 <= numeric_text_ratio <= 0.95:
                        strength = 0.35
                    else:
                        strength = 0.60
                elif step_name == "clean_missing_tokens":
                    strength = 0.90 if int(false_tokens.sum()) == 0 or int(false_tokens.sum()) > 5 else 0.60
                elif step_name == "clean_variants":
                    strength = 0.90 if variant_count == 0 or variant_count > 5 else 0.60
                elif step_name == "clean_dates":
                    strength = 0.90 if date_parse_rate >= 0.95 or date_parse_rate == 0 else (0.35 if 0.75 <= date_parse_rate <= 0.85 else 0.60)
                elif step_name == "clean_impossible":
                    strength = 0.90
                elif step_name == "duplicates":
                    strength = 0.90 if duplicate_rows == 0 or duplicate_rows > 10 else 0.60

                if step_name == "type" and detail.get("action") == "DROP":
                    detail["confidence"] = "haute"
                else:
                    detail["confidence"] = recommendation_confidence(sample_n, strength, p_evidence)
                measure_text = str(detail.get("measure", "")).strip()
                threshold_text = str(detail.get("threshold", "")).strip()
                if measure_text and measure_text not in str(detail.get("explanation", "")):
                    detail["explanation"] = (
                        f"{str(detail.get('explanation', '')).rstrip()} "
                        f"Preuve : {measure_text}; seuil : {threshold_text}."
                    )

            if action == Action.DROP and kind_name.startswith("DROP"):
                overall_strength = 1.0
            elif kind_name == FeatureType.BINARY.value:
                overall_strength = 0.90
            elif kind_name == FeatureType.CATEGORICAL_LOW_CARD.value:
                card_th = max(self.config.cardinality_threshold, 2)
                if nunique <= 4:
                    overall_strength = 0.85
                elif nunique >= card_th - 1:
                    overall_strength = 0.35
                else:
                    overall_strength = 0.60
            elif kind_name == FeatureType.CATEGORICAL_HIGH_CARD.value:
                card_th = max(self.config.cardinality_threshold, 2)
                if nunique >= 40:
                    overall_strength = 0.90
                elif nunique <= card_th + 2:
                    overall_strength = 0.35
                else:
                    overall_strength = 0.60
            elif kind_name == FeatureType.DATETIME.value:
                overall_strength = 0.90 if date_parse_rate >= 0.95 else 0.60
            elif kind_name == FeatureType.TEXT.value:
                overall_strength = 0.85
            elif kind_name.startswith("NUMERIC"):
                ratio = abs(skew) / max(self.config.skew_threshold, 1e-9)
                if abs(skew) >= 2.0 or abs(skew) <= 0.20:
                    overall_strength = 0.85
                elif 0.85 <= ratio <= 1.15:
                    overall_strength = 0.35
                else:
                    overall_strength = 0.60
            else:
                overall_strength = 0.60

            confidence = recommendation_confidence(
                sample_n, overall_strength,
                normality_p if kind_name.startswith("NUMERIC") and pd.notna(normality_p) else None)
            if action == Action.DROP and kind_name.startswith("DROP"):
                confidence = "haute"
            rows.append({"colonne": str(col), "type": kind_name, "dtype": str(df[col].dtype),
                         "manquants_pct": missing * 100, "uniques": nunique, "exemples": examples,
                         "faux_manquants": false_missing, "variantes_casse_espaces": variant_count,
                         "relation_cible": target_relation, "score_fuite": score,
                         "relation_cible_methode": target_relation_method,
                         "relation_cible_non_fiable_identifiant": relation_unreliable_identifier,
                         "action": action.value, "raison": reason, "confiance": confidence,
                         "alternatives": [a.value for a in alternatives], "steps": step_recs,
                         "spearman_cible": target_spearman, "cramer_cible": target_cramer,
                         "information_mutuelle_cible": target_mi,
                         "p_manquants_cible": missing_target_p,
                         "manquant_informatif": bool(pd.notna(missing_target_p) and missing_target_p < .05),
                         "est_cible": False,
                         **stats_row})
        # Paires de features fortement corrélées; ordre d'entrée = règle explicite du préprocesseur.
        numeric_df = raw.select_dtypes(include=[np.number]).drop(columns=[target] if target else [], errors="ignore")
        correlation_pairs = []
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr().abs()
            names = list(corr.columns)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    score_corr = corr.loc[left, right]
                    if pd.notna(score_corr) and score_corr > self.config.correlation_threshold:
                        correlation_pairs.append({"gauche": str(left), "droite": str(right), "correlation_abs": float(score_corr),
                                                  "gardee_selon_ordre": str(left)})
        row_by_name = {r["colonne"]: r for r in rows}
        for pair in correlation_pairs:
            for col, other in ((pair["gauche"], pair["droite"]), (pair["droite"], pair["gauche"])):
                if col in row_by_name:
                    rec_sel = row_by_name[col]["steps"]["selection"]
                    rec_sel["measure"] += f"; corrélation avec {other} : |r|={pair['correlation_abs']:.3f}"
                    rec_sel["threshold"] = f"|r| > {self.config.correlation_threshold:.3f}"
                    rec_sel["explanation"] = (
                        f"Redondance avec {other} : |r|={pair['correlation_abs']:.3f} > "
                        f"{self.config.correlation_threshold:.3f}; la colonne {pair['gardee_selon_ordre']} "
                        "est prioritaire selon l'ordre d'entrée. Le fit final réévalue ce choix sur le train."
                    )
        for row in rows:
            row["paires_correlees"] = [p for p in correlation_pairs if row["colonne"] in (p["gauche"], p["droite"])]
            if row.get("est_cible"):
                row["alerte_identifiant"] = False
                row["ratio_uniques_lignes"] = np.nan
                continue
            unique_ratio = row["uniques"] / max(len(raw), 1)
            id_named = bool(re.search(r"(^|_)(id|code|ref|num)(_|$)", row["colonne"].lower()))
            row["ratio_uniques_lignes"] = unique_ratio
            row["seuil_identifiant_uniques"] = self.config.identifier_unique_threshold
            if row["type"] == FeatureType.DROP_ID.value:
                row["drop_preuve"] = f"uniques = {row['uniques']} / {len(raw)} lignes ({unique_ratio:.1%})"
                row["drop_seuil"] = (f"nom évocateur ou ratio > "
                                     f"{self.config.identifier_unique_threshold:.1%} des lignes")
            elif row["type"] == FeatureType.DROP_HIGH_MISSING.value:
                row["drop_preuve"] = f"manquants = {row['manquants_pct']:.1f}%"
                row["drop_seuil"] = f">= {self.config.missing_threshold:.1%} des lignes"
            elif row["type"] == FeatureType.DROP_CONSTANT.value:
                if row["uniques"] <= 1:
                    row["drop_preuve"] = f"valeurs uniques = {row['uniques']}"
                    row["drop_seuil"] = "<= 1 valeur unique"
                else:
                    top_frequency = float(row.get("top_category_frequency", np.nan))
                    row["drop_preuve"] = f"fréquence dominante = {top_frequency:.1%}"
                    row["drop_seuil"] = f">= {self.config.constant_threshold:.1%}"
            elif row["type"] == FeatureType.DROP_DUPLICATE.value:
                row["drop_preuve"] = str(row.get("raison", "contenu identique"))
                row["drop_seuil"] = "contenu strictement identique à une colonne précédente"
            elif row["type"] == FeatureType.DROP_LEAKAGE.value:
                row["drop_preuve"] = f"score fuite CV = {row.get('score_fuite', np.nan):.3f}"
                row["drop_seuil"] = (f">= {self.config.leakage_threshold:.3f} et écart >= "
                                     f"{self.config.leakage_min_gap:.3f}")
            if str(row.get("action")) == Action.DROP.value and row.get("drop_preuve"):
                type_recommendation = row.setdefault("steps", {}).setdefault("type", {})
                type_recommendation.update(
                    action="DROP", measure=row["drop_preuve"], threshold=row["drop_seuil"],
                    rule=f"{row['type']} déclenché selon la configuration du préprocesseur",
                    explanation=(f"La suppression est recommandée : {row['drop_preuve']}; "
                                 f"seuil appliqué : {row['drop_seuil']}."))
            for recommendation in row.get("steps", {}).values():
                recommendation["evidence"] = format_step_evidence(
                    recommendation.get("measure", ""), recommendation.get("threshold", ""))
                evidence_text = f"Preuve : {recommendation.get('measure', '')}; seuil : {recommendation.get('threshold', '')}."
                if recommendation.get("measure") and recommendation["measure"] not in recommendation.get("explanation", ""):
                    recommendation["explanation"] = f"{recommendation.get('explanation', '').strip()} {evidence_text}".strip()
            primary_step = recommendation_step_name(row.get("action", "AUTO"))
            primary_recommendation = row.get("steps", {}).get(primary_step, {})
            if primary_recommendation:
                primary_recommendation["reason"] = format_step_reason(
                    row.get("action", "AUTO"), primary_recommendation.get("measure", ""),
                    primary_recommendation.get("threshold", ""))
                row["raison"] = primary_recommendation["reason"]
            row["alerte_identifiant"] = bool(row["type"] == FeatureType.DROP_ID.value or
                                              (unique_ratio > self.config.identifier_unique_threshold and row["type"] != FeatureType.DATETIME.value))
            row["relation_cible_non_fiable_identifiant"] = bool(row.get("relation_cible_non_fiable_identifiant") or row["alerte_identifiant"])
            if row["alerte_identifiant"]:
                row["relation_cible"] = np.nan
                row["spearman_cible"] = row["cramer_cible"] = row["information_mutuelle_cible"] = np.nan
                row["relation_cible_methode"] = "non fiable (identifiant)"
            if unique_ratio > self.config.identifier_unique_threshold and row["type"] not in {FeatureType.DROP_ID.value, FeatureType.DATETIME.value}:
                row["raison"] += (f" Avertissement identifiant : ratio uniques/lignes={unique_ratio:.1%} "
                                   f"(> {self.config.identifier_unique_threshold:.1%}), mais la règle de type ne l'a pas supprimée.")
                if row["confiance"] != "haute":
                    row["confiance"] = "faible"
        return AnalysisResult(rows, sample_note)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> AutoPreprocessor:
        """Ajuste l'ensemble des transformateurs sur les données d'entraînement.

        Garantit l'absence totale de fuite : toutes les statistiques (médianes, quantiles,
        encodages, références temporelles) sont estimées sur X uniquement.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._set_logging_level()
        self.warnings_ = []
        self.cleaning_decisions_ = {}
        self.numeric_string_cols_ = {}  # F2 : reinitialiser a chaque fit
        self.noise_cols_detected_ = []  # F1 : reinitialiser
        self.noise_cols_dropped_ = []   # F1 : reinitialiser
        logger.info(f"Début du fit sur DataFrame de taille {X.shape}")

        # Nettoyage préalable (Étape 0) - is_fit=True pour mémoriser numeric_string_cols_ (F2)
        X_clean = self._preprocess_raw_dataframe(X, is_fit=True)
        self.input_columns_ = list(X_clean.columns)

        # Ajustement de la cible si fournie
        y_arr = None
        if y is not None:
            keep_positions = np.asarray(self._last_keep_positions_, dtype=bool)
            if len(keep_positions) != len(y):
                raise ValueError("Impossible d'aligner les lignes supprimées par clean_impossible avec la cible.")
            if not keep_positions.all():
                y = y.iloc[np.flatnonzero(keep_positions)] if isinstance(y, pd.Series) else np.asarray(y)[keep_positions]
            y_arr = self._fit_target(y)

        # Détection des types
        self._detect_types(X_clean)

        # Application des choix explicites avant la construction des pipelines.
        self.user_override_notes_ = {}
        self.column_scalers_ = {}
        self.column_distributions_ = {}
        self.column_missing_ = {}
        self.column_outliers_ = {}
        self.column_override_params_ = {}
        self.categorical_imputers_ = {}
        self.frequency_override_cols_ = []
        for raw_col, raw_override in (self.config.column_overrides or {}).items():
            col = str(raw_col)
            if col not in X_clean.columns:
                col = re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip())).lower()
            if col not in X_clean.columns:
                continue
            values = raw_override.actions if isinstance(raw_override, ColumnOverride) else raw_override
            if isinstance(values, dict):
                choices = {}
                params = {}
                for key, value in values.items():
                    if isinstance(value, tuple) and len(value) == 2:
                        selected, params[str(key).lower()] = value
                    else:
                        selected = value
                    choices[str(key).lower()] = selected.value if isinstance(selected, Action) else str(selected)
                self.column_override_params_[col] = params
            else:
                action = values.value if isinstance(values, Action) else str(values)
                stage = {"DROP":"type", "YEO_JOHNSON":"distribution", "LOG1P":"distribution",
                         "SCALE_STANDARD":"scaling", "SCALE_ROBUST":"scaling", "SCALE_MINMAX":"scaling",
                         "NO_SCALE":"scaling", "ONEHOT":"encoding", "TARGET_ENCODE":"encoding",
                         "FREQUENCY_ENCODE":"encoding", "ORDINAL":"encoding", "BINARY_MAP":"encoding",
                         "KEEP_NUMERIC":"type", "DATE_FEATURES":"dates", "TFIDF_SVD":"text",
                         "IMPUTE_MEDIAN":"missing", "MISSING_INDICATOR":"missing", "WINSORIZE":"outliers"}.get(action, "action")
                normalized_action = {"KEEP_NUMERIC":"AUTO", "SCALE_STANDARD":"STANDARD", "SCALE_ROBUST":"ROBUST",
                                     "SCALE_MINMAX":"MINMAX", "NO_SCALE":"NONE", "IMPUTE_MEDIAN":"MEDIAN"}.get(action, action)
                choices = {stage: normalized_action}
            meta_type = self.column_metadata_.get(col, {}).get("type")
            kind = meta_type.value if isinstance(meta_type, FeatureType) else str(meta_type)
            group = _action_group(meta_type)
            for step_name, action_value in choices.items():
                if not _action_supported(step_name, group, action_value.upper()):
                    raise ValueError(
                        f"Action incompatible ou non prise en charge : {action_value!r} pour l'étape {step_name!r} "
                        f"et la colonne {col!r} (type {kind}). Choix disponibles : "
                        f"{SUPPORTED_ACTIONS.get(step_name, {}).get(group, SUPPORTED_ACTIONS.get(step_name, {}).get('any', []))}."
                    )
            action = choices.get("action", "").upper()
            if choices.get("type", choices.get("retention", "")).upper() == "DROP" or action == "DROP":
                if col not in self.columns_to_drop_:
                    self.columns_to_drop_.append(col)
                for group in (self.num_cols_, self.binary_cols_, self.ordinal_cols_, self.low_card_cols_, self.high_card_cols_, self.date_cols_, self.text_cols_):
                    if col in group:
                        group.remove(col)
                self.user_override_notes_[col] = "choix utilisateur : DROP"
                continue
            dist = choices.get("distribution", action).upper()
            if dist in {"YEO_JOHNSON", "LOG1P", "BOX_COX", "SQRT"} and not kind.startswith("NUMERIC"):
                raise ValueError(f"Override {dist} incompatible avec {col!r} (type détecté : {kind}); cette action exige une colonne numérique.")
            if choices.get("encoding", "").upper() == "ONEHOT" and kind == "NUMERIC_CONTINUOUS":
                raise ValueError(f"Override ONEHOT incompatible avec {col!r} (numérique continu).")
            encoding = choices.get("encoding", "").upper()
            if encoding == "ONEHOT" and col in self.high_card_cols_:
                self.high_card_cols_.remove(col); self.low_card_cols_.append(col)
            elif encoding == "ORDINAL" and col in self.low_card_cols_:
                self.low_card_cols_.remove(col); self.ordinal_cols_.append(col)
            elif encoding == "ORDINAL" and col in self.high_card_cols_:
                self.high_card_cols_.remove(col); self.ordinal_cols_.append(col)
            elif encoding == "FREQUENCY_ENCODE" and col in self.low_card_cols_:
                self.low_card_cols_.remove(col); self.frequency_override_cols_.append(col)
            elif encoding == "TARGET_ENCODE" and col in self.low_card_cols_:
                self.low_card_cols_.remove(col); self.high_card_cols_.append(col)
            elif encoding == "TARGET" and col in self.low_card_cols_:
                self.low_card_cols_.remove(col); self.high_card_cols_.append(col)
            if encoding == "FREQUENCY_ENCODE" and y_arr is not None:
                raise ValueError(f"FREQUENCY_ENCODE sur {col!r} n'est disponible que sans cible; avec cible, utilisez TARGET_ENCODE.")
            if encoding == "TARGET_ENCODE" and y_arr is None:
                raise ValueError(f"TARGET_ENCODE sur {col!r} exige une cible; choisissez FREQUENCY_ENCODE.")
            if encoding:
                self.user_override_notes_[col] = f"choix utilisateur : {encoding}"
            if dist == "BOX_COX" and pd.to_numeric(X_clean[col], errors="coerce").dropna().le(0).any():
                raise ValueError(f"Box-Cox impossible sur {col!r} : les valeurs doivent être strictement positives.")
            if dist in {"YEO_JOHNSON", "LOG1P", "BOX_COX", "SQRT", "NONE"}:
                self.column_distributions_[col] = dist
                self.user_override_notes_[col] = f"choix utilisateur : {dist}"
            missing_action = choices.get("missing", "").upper()
            if missing_action in {"MEDIAN", "MEAN", "MODE", "CONSTANT", "MISSING_INDICATOR", "MISSING_CATEGORY"}:
                self.column_missing_[col] = missing_action
                self.user_override_notes_[col] = f"choix utilisateur : {missing_action} (valeurs manquantes)"
            outlier_action = choices.get("outliers", "").upper()
            if outlier_action in {"NONE", "WINSORIZE", "IQR"}:
                self.column_outliers_[col] = outlier_action
                self.user_override_notes_[col] = f"choix utilisateur : {outlier_action} (valeurs extrêmes)"
            scale = choices.get("scaling", action).upper()
            if scale in {"STANDARD", "ROBUST", "MINMAX", "MAXABS", "NONE", "SCALE_STANDARD", "SCALE_ROBUST", "SCALE_MINMAX", "NO_SCALE"}:
                self.column_scalers_[col] = scale.replace("SCALE_", "").replace("NO_SCALE", "NONE")
                self.user_override_notes_[col] = f"choix utilisateur : {scale}"
            for step, val in choices.items():
                if step not in {"action", "distribution", "scaling"} and val.upper() not in {"AUTO", "NONE", "KEEP_RAW"}:
                    self.user_override_notes_[col] = f"choix utilisateur : {val.upper()} (étape {step})"

        # Un scaler statistique commun à analyze et fit: les seuils sont réestimés sur le train.
        for col in self.num_cols_:
            if col not in self.column_scalers_:
                mode = (_recommend_scaler(X_clean[col])[0] if self.config.scaler_type == "standard"
                        else self.config.scaler_type.upper())
                self.column_scalers_[col] = mode

        # Filtrage des colonnes à exclure
        X_work = X_clean.drop(columns=self.columns_to_drop_, errors="ignore")
        for col, mode in self.column_missing_.items():
            if col not in X_work.columns or col not in self.binary_cols_ + self.ordinal_cols_ + self.low_card_cols_ + self.high_card_cols_:
                continue
            if mode == "MODE":
                modes = X_work[col].dropna().mode()
                fill_value = modes.iloc[0] if len(modes) else "__MISSING__"
            elif mode == "MISSING_CATEGORY":
                fill_value = "__MISSING__"
            elif mode == "CONSTANT":
                fill_value = self.column_override_params_.get(col, {}).get("missing", "__CONSTANT__")
            else:
                continue
            self.categorical_imputers_[col] = fill_value
            X_work[col] = X_work[col].fillna(fill_value)

        # Détection de fuite de cible sur les colonnes brutes (avant tout encodage)
        self.suspected_leakage_cols_ = []
        self.leakage_scores_ = {}
        self._te_oof_ = None
        if y_arr is not None:
            self._detect_leaky_columns(X_work, y_arr)
            X_work = X_clean.drop(columns=self.columns_to_drop_, errors="ignore")
            # F1 : garde-fou contre les colonnes de bruit (date / haute cardinalité sans signal)
            self._detect_noise_columns(X_work, y_arr)
            X_work = X_clean.drop(columns=self.columns_to_drop_, errors="ignore")

        # 1. Pipeline Numérique
        if self.num_cols_:
            X_num = X_work[self.num_cols_].copy()

            # Missing Indicator
            self.missing_indicator_ = MissingIndicatorTransformer(
                threshold=self.config.missing_indicator_threshold
            )
            self.missing_indicator_.fit(X_num)
            for col, mode in self.column_missing_.items():
                if mode == "MISSING_INDICATOR" and col in X_num.columns and col not in self.missing_indicator_.indicator_cols_:
                    self.missing_indicator_.indicator_cols_.append(col)

            # Winsorisation / Outliers
            self.winsorizer_ = Winsorizer(
                method=self.config.outlier_method,
                lower_quantile=self.config.outlier_lower_quantile,
                upper_quantile=self.config.outlier_upper_quantile,
                iqr_multiplier=self.config.outlier_iqr_multiplier,
            )
            self.winsorizer_.fit(X_num)
            for col, mode in self.column_outliers_.items():
                if col not in X_num:
                    continue
                vals = X_num[col].dropna()
                if mode == "NONE" or len(vals) == 0:
                    self.winsorizer_.lower_bounds_[col] = -np.inf
                    self.winsorizer_.upper_bounds_[col] = np.inf
                elif mode == "IQR":
                    q1, q3 = vals.quantile(.25), vals.quantile(.75)
                    spread = q3 - q1
                    self.winsorizer_.lower_bounds_[col] = float(q1 - self.config.outlier_iqr_multiplier * spread)
                    self.winsorizer_.upper_bounds_[col] = float(q3 + self.config.outlier_iqr_multiplier * spread)
                elif mode == "WINSORIZE":
                    params = self.column_override_params_.get(col, {}).get("outliers", {})
                    if isinstance(params, dict):
                        lower = float(params.get("lower", self.config.outlier_lower_quantile))
                        upper = float(params.get("upper", self.config.outlier_upper_quantile))
                        self.winsorizer_.lower_bounds_[col] = float(vals.quantile(lower))
                        self.winsorizer_.upper_bounds_[col] = float(vals.quantile(upper))
            X_num_wins = self.winsorizer_.transform(X_num)

            # Imputation médiane ou moyenne
            self.num_imputer_ = {}
            for col in self.num_cols_:
                mode = self.column_missing_.get(col, self.config.imputation_strategy.upper())
                if mode == "MEAN":
                    value = X_num_wins[col].mean()
                elif mode == "MODE":
                    modes = X_num_wins[col].mode(dropna=True)
                    value = modes.iloc[0] if len(modes) else 0.0
                elif mode == "CONSTANT":
                    value = self.column_override_params_.get(col, {}).get("missing", 0.0)
                elif mode == "NONE":
                    value = X_num_wins[col].median()
                else:
                    value = X_num_wins[col].median()
                self.num_imputer_[col] = float(value) if pd.notna(value) else 0.0

            X_num_imp = X_num_wins.fillna(self.num_imputer_)

            # PowerTransformer pour asymétrie
            self.num_power_transformer_ = {}
            self.num_skewed_cols_ = []
            for col in self.num_cols_:
                sk = float(stats.skew(X_num_imp[col].dropna()))
                override_dist = self.column_distributions_.get(col)
                if override_dist == "YEO_JOHNSON" or (override_dist not in {"NONE", "LOG1P", "SQRT", "BOX_COX"} and abs(sk) > self.config.skew_threshold):
                    pt = PowerTransformer(method="yeo-johnson")
                    pt.fit(X_num_imp[[col]])
                    self.num_power_transformer_[col] = pt
                    self.num_skewed_cols_.append(col)
                    X_num_imp[col] = pt.transform(X_num_imp[[col]]).ravel()
                elif override_dist in {"LOG1P", "SQRT", "BOX_COX"}:
                    if override_dist == "LOG1P" and X_num_imp[col].le(-1).any():
                        raise ValueError(f"LOG1P impossible sur {col!r} : des valeurs sont inférieures ou égales à -1.")
                    if override_dist == "SQRT" and X_num_imp[col].lt(0).any():
                        raise ValueError(f"Racine carrée impossible sur {col!r} : valeurs négatives détectées.")
                    if override_dist == "BOX_COX":
                        pt = PowerTransformer(method="box-cox").fit(X_num_imp[[col]])
                        self.num_power_transformer_[col] = pt
                        X_num_imp[col] = pt.transform(X_num_imp[[col]]).ravel()
                    else:
                        self.num_power_transformer_[col] = override_dist
                        if override_dist == "LOG1P":
                            X_num_imp[col] = np.log1p(X_num_imp[col])
                        elif override_dist == "SQRT":
                            X_num_imp[col] = np.sqrt(X_num_imp[col].clip(lower=0))

            # Scaling
            if self.config.scaler_type == "robust":
                self.num_scaler_ = RobustScaler()
            elif self.config.scaler_type == "minmax":
                self.num_scaler_ = MinMaxScaler()
            elif self.config.scaler_type == "standard":
                self.num_scaler_ = StandardScaler()
            else:
                self.num_scaler_ = None

            overridden = set(self.column_scalers_).intersection(self.num_cols_)
            default_cols = [c for c in self.num_cols_ if c not in overridden]
            if self.num_scaler_ is not None and default_cols:
                self.num_scaler_.fit(X_num_imp[default_cols])
            for col in overridden:
                mode = self.column_scalers_[col]
                if mode == "NONE":
                    self.column_scalers_[col] = None
                else:
                    scaler_cls = {"STANDARD": StandardScaler, "ROBUST": RobustScaler,
                                  "MINMAX": MinMaxScaler, "MAXABS": MaxAbsScaler}.get(mode)
                    if scaler_cls is None:
                        raise ValueError(f"Scaler inconnu pour {col!r}: {mode}")
                    self.column_scalers_[col] = scaler_cls().fit(X_num_imp[[col]])

        # 2. Pipeline Binaire
        if self.binary_cols_:
            self.binary_mapper_ = BinaryMapper()
            self.binary_mapper_.fit(X_work[self.binary_cols_])

        # 3. Pipeline Ordinal
        if self.ordinal_cols_:
            self.ordinal_encoders_ = {}
            for col in self.ordinal_cols_:
                custom_orders = self.config.ordinal_orders or {}
                override_order = self.column_override_params_.get(col, {}).get("encoding", {})
                if isinstance(override_order, dict) and "order" in override_order:
                    order = list(override_order["order"])
                elif col in custom_orders:
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
            rare_thresholds = {}
            for col in self.low_card_cols_:
                params = self.column_override_params_.get(col, {}).get("encoding", {})
                if isinstance(params, dict) and "rare_threshold" in params:
                    rare_thresholds[col] = float(params["rare_threshold"])
            self.rare_grouper_ = RareCategoryGrouper(threshold=self.config.rare_category_threshold,
                                                     column_thresholds=rare_thresholds)
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

        for col in self.frequency_override_cols_:
            if col in X_work.columns:
                self.freq_encoding_maps_[col] = X_work[col].astype(str).value_counts(normalize=True).to_dict()

        # 6. Pipeline Dates
        if self.date_cols_:
            self.datetime_extractor_ = DateTimeFeatureExtractor()
            self.datetime_extractor_.fit(X_work[self.date_cols_])

        # 7. Pipeline Texte
        if self.text_cols_:
            text_components = {}
            for col in self.text_cols_:
                params = self.column_override_params_.get(col, {}).get("text", {})
                if isinstance(params, dict) and "components" in params:
                    text_components[col] = int(params["components"])
            self.text_transformer_ = RobustTextTransformer(
                max_features=self.config.max_text_features,
                svd_components=self.config.text_svd_components,
                random_state=self.config.random_state,
                column_components=text_components,
            )
            self.text_transformer_.fit(X_work[self.text_cols_])

        # Post-traitement sur le train : Sélection des colonnes, variance, corrélation et fuites
        X_trans_train = self._transform_components(X_clean)
        self._fit_post_processing(X_trans_train, y_arr)

        self.is_fitted_ = True
        logger.info(f"AutoPreprocessor ajusté avec succès. Nb features finales : {len(self.final_feature_names_)}")
        return self

    def _transform_components(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        X_clean = X_clean.copy()
        for col, value in self.categorical_imputers_.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(value)
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
                        if isinstance(pt, str) and pt == "LOG1P":
                            X_num_imp[c] = np.log1p(X_num_imp[c])
                        elif isinstance(pt, str) and pt == "SQRT":
                            X_num_imp[c] = np.sqrt(X_num_imp[c].clip(lower=0))
                        else:
                            X_num_imp[c] = pt.transform(X_num_imp[[c]]).ravel()

            # Scaling
            overridden = set(self.column_scalers_).intersection(self.num_cols_)
            if not overridden:
                if self.num_scaler_ is not None:
                    scaled_arr = self.num_scaler_.transform(X_num_imp)
                    df_num_scaled = pd.DataFrame(scaled_arr, index=X_clean.index, columns=self.num_cols_)
                else:
                    df_num_scaled = X_num_imp
            else:
                default_cols = [c for c in self.num_cols_ if c not in overridden]
                df_num_scaled = X_num_imp.astype(float)
                if self.num_scaler_ is not None and default_cols:
                    df_num_scaled.loc[:, default_cols] = self.num_scaler_.transform(X_num_imp[default_cols])
                for col in overridden:
                    scaler = self.column_scalers_[col]
                    if scaler is not None:
                        df_num_scaled[col] = scaler.transform(X_num_imp[[col]]).ravel()

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

        if self.frequency_override_cols_:
            freq_data = {}
            for col in self.frequency_override_cols_:
                fmap = self.freq_encoding_maps_.get(col, {})
                if col in X_clean.columns:
                    freq_data[f"{col}_freq"] = X_clean[col].astype(str).map(fmap).fillna(0.0)
                else:
                    freq_data[f"{col}_freq"] = np.zeros(len(X_clean))
            parts.append(pd.DataFrame(freq_data, index=X_clean.index))

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        duplicate_action = self._dataset_duplicate_action()
        duplicate_mask = X.duplicated(keep="first")
        duplicate_count = int(duplicate_mask.sum())
        self.dataset_cleaning_ = {"doublons_detectes": duplicate_count, "action_doublons": duplicate_action}
        if duplicate_action == "DROP" and duplicate_count:
            keep = ~duplicate_mask
            X_fit = X.loc[keep]
            if y is not None:
                y_fit = y.iloc[np.flatnonzero(keep.to_numpy())] if isinstance(y, pd.Series) and len(y) == len(X) else np.asarray(y)[keep.to_numpy()]
            else:
                y_fit = None
        else:
            X_fit, y_fit = X, y
        self.fit(X_fit, y_fit)
        X_out = self.transform(X_fit)
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
        if self._dataset_duplicate_action() == "DROP":
            X = X.loc[~X.duplicated(keep="first")]
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

        # Les doublons suivent le choix partagé avec fit_transform.
        duplicate_action = self._dataset_duplicate_action()
        n_before = len(df_work)
        if duplicate_action == "DROP":
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
        self.dataset_cleaning_["doublons_detectes"] = n_dup
        self.dataset_cleaning_["action_doublons"] = duplicate_action
        if len(X_trans) != len(y_raw):
            y_raw = y_raw.loc[X_trans.index]
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
            if len(X_trans) != len(y_raw):
                y_raw = y_raw.loc[X_trans.index]
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
                "Choix utilisateur": self.user_override_notes_.get(col, "recommandation suivie"),
                "Détails nettoyage": self.cleaning_decisions_.get(col, {}),
            })

        df_report = pd.DataFrame(columns_summary)
        cleaned_dropped = [col for col, detail in self.cleaning_decisions_.items()
                           if detail.get("action_dates") == "DROP_COLUMN"]
        for col in cleaned_dropped:
            columns_summary.append({
                "Colonne": col, "Type détecté": "DATETIME",
                "Manquants (%)": "", "Cardinalité": "", "Justification": "Colonne retirée par l'étape dates.",
                "Statut": "SUPPRIMÉE (choix utilisateur : DROP_COLUMN)", "Nb features créées": 0,
                "Choix utilisateur": "choix utilisateur : DROP_COLUMN",
                "Détails nettoyage": self.cleaning_decisions_.get(col, {}),
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
                "nettoyage": cleaned_dropped,
                "variance_nulle": self.dropped_variance_cols_,
                "colinearite": self.dropped_collinear_cols_,
                "fuites_cible": [c for c in self.suspected_leakage_cols_ if c in self.columns_to_drop_],
            },
            "fuites_suspectees": self.suspected_leakage_cols_,
            "scores_fuite_par_colonne": self.leakage_scores_,
            "nettoyage_dataset": self.dataset_cleaning_,
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
        lines.append(f"Tache ML                : {rep['tache_detectee'].upper()}")
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
        """Charge une instance d'AutoPreprocessor sauvegardée via joblib.

        Compatibilité ascendante (F2) : un fichier sauvegardé avant l'ajout de F2
        (sans attribut ``numeric_string_cols_``) se charge sans erreur avec un
        avertissement explicatif.
        """
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            logger.info(f"AutoPreprocessor version {payload.get('version')} chargé depuis '{path}'")
            obj = payload["model"]
        elif isinstance(payload, AutoPreprocessor):
            obj = payload
        else:
            raise TypeError(f"Le fichier '{path}' ne contient pas d'objet AutoPreprocessor valide.")
        # F2 : compatibilite ascendante - attributs absents dans anciens fichiers joblib
        if not hasattr(obj, "numeric_string_cols_"):
            obj.numeric_string_cols_ = {}
            msg = (
                "Fichier joblib ancien charge (avant F2) : numeric_string_cols_ absent. "
                "Comportement pre-F2 : le parsing numerique sera recalcule a chaque "
                "transform(). Refaire un fit() pour obtenir la coherence F2."
            )
            logger.warning(msg)
            if hasattr(obj, "warnings_"):
                obj.warnings_.append(msg)
        # F1 : attributs optionnels absents dans anciens fichiers
        for _attr, _default in [
            ("noise_cols_detected_", []),
            ("noise_cols_dropped_", []),
        ]:
            if not hasattr(obj, _attr):
                setattr(obj, _attr, _default)
        return obj
