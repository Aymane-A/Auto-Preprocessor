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
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (active IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
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
    SQRT = "SQRT"
    CBRT = "CBRT"
    QUANTILE = "QUANTILE"
    RANK = "RANK"
    SCALE_STANDARD = "SCALE_STANDARD"
    SCALE_ROBUST = "SCALE_ROBUST"
    SCALE_MINMAX = "SCALE_MINMAX"
    NO_SCALE = "NO_SCALE"
    ONEHOT = "ONEHOT"
    TARGET_ENCODE = "TARGET_ENCODE"
    FREQUENCY_ENCODE = "FREQUENCY_ENCODE"
    COUNT_ENCODE = "COUNT_ENCODE"
    BINARY_ENCODE = "BINARY_ENCODE"
    HASH_ENCODE = "HASH_ENCODE"
    LOO_ENCODE = "LOO_ENCODE"
    ORDINAL = "ORDINAL"
    BINARY_MAP = "BINARY_MAP"
    DATE_FEATURES = "DATE_FEATURES"
    TFIDF_SVD = "TFIDF_SVD"
    TFIDF_NGRAMS = "TFIDF_NGRAMS"
    COUNT_VECTORIZER = "COUNT_VECTORIZER"
    HASH_TEXT = "HASH_TEXT"
    MISSING_INDICATOR = "MISSING_INDICATOR"


ACTION_RECOMMENDATION_STEP = {
    "DROP": "type", "KEEP_NUMERIC": "distribution", "YEO_JOHNSON": "distribution",
    "LOG1P": "distribution", "BOX_COX": "distribution", "SQRT": "distribution",
    "CBRT": "distribution", "QUANTILE": "distribution", "RANK": "distribution",
    "COUNT_ENCODE": "encoding", "BINARY_ENCODE": "encoding", "HASH_ENCODE": "encoding", "LOO_ENCODE": "encoding",
    "DATE_FEATURES": "dates", "TFIDF_SVD": "text", "TFIDF_NGRAMS": "text", "COUNT_VECTORIZER": "text",
    "HASH_TEXT": "text", "ONEHOT": "encoding",
    "TARGET_ENCODE": "encoding", "FREQUENCY_ENCODE": "encoding", "ORDINAL": "encoding",
    "BINARY_MAP": "encoding",
}


PREPROCESS_LABELS = {
    "interface": {
        "task": "Tâche", "data": "Données", "parameters": "Paramètres",
        "result": "Résultat", "selection": "Sélection",
    },
    "steps": {
        "nettoyage": ("Nettoyage général", "Résumé des corrections détectées."),
        "clean_missing_tokens": ("Faux manquants", "Convertit certains marqueurs textuels en valeurs manquantes."),
        "clean_variants": ("Variantes de modalités", "Fusionne les différences de casse, espaces et accents."),
        "clean_numeric_text": ("Nombres écrits en texte", "Convertit les nombres importés comme chaînes."),
        "clean_dates": ("Formats de dates", "Analyse les chaînes comme dates reconnues."),
        "clean_impossible": ("Valeurs impossibles", "Traite les valeurs hors des bornes métier détectées."),
        "duplicates": ("Lignes dupliquées", "Décide du traitement des lignes identiques."),
        "type": ("Garder / Supprimer", "Conserve la colonne ou la retire des variables explicatives."),
        "missing": ("Valeurs manquantes", "Impute les absences ou ajoute un indicateur."),
        "outliers": ("Valeurs extrêmes", "Borne ou conserve les observations atypiques."),
        "distribution": ("Distribution", "Transforme la forme de la distribution numérique."),
        "scaling": ("Mise à l'échelle", "Met les variables numériques sur une échelle comparable."),
        "encoding": ("Encodage", "Convertit les catégories en variables numériques."),
        "dates": ("Dates", "Extrait des composantes calendaires."),
        "text": ("Texte libre", "Vectorise et résume le contenu textuel."),
        "target": ("Cible", "La cible est traitée séparément des features."),
        "selection": ("Sélection finale", "Retire ou conserve les variables redondantes."),
    },
    "actions": {
        "AUTO": ("Automatique", "Laisse la règle de détection choisir le traitement."),
        "CLEAN_DEFAULT": ("Nettoyage automatique", "Applique les corrections détectées."),
        "TO_NAN": ("Convertir en valeur manquante", "Reconnaît les marqueurs d'absence courants."),
        "KEEP": ("Garder tel quel", "Préserve les valeurs originales."),
        "SELECTED_TO_NAN": ("Convertir les valeurs choisies", "Ne remplace que les marqueurs sélectionnés."),
        "MERGE": ("Fusionner les variantes", "Regroupe les libellés équivalents."),
        "CONVERT": ("Convertir en numérique", "Extrait les nombres et retire les unités."),
        "PARSE": ("Analyser comme date", "Convertit les chaînes en dates."),
        "DROP_COLUMN": ("Supprimer la colonne", "Retire la colonne du dataset."),
        "NAN": ("Remplacer par manquant", "Remplace les valeurs impossibles par une absence."),
        "CLIP": ("Limiter aux bornes", "Ramène les valeurs aux limites autorisées."),
        "DROP_ROWS": ("Supprimer les lignes", "Retire les observations contenant ces valeurs."),
        "IGNORE": ("Ignorer", "Conserve les valeurs sans correction."),
        "DROP": ("Supprimer", "Retire la colonne du modèle."),
        "FORCE_NUMERIC": ("Forcer numérique", "Interprète la colonne comme une mesure numérique."),
        "FORCE_CATEGORICAL": ("Forcer catégoriel", "Traite les valeurs comme des modalités."),
        "FORCE_DATE": ("Forcer date", "Interprète les valeurs comme des dates."),
        "FORCE_TEXT": ("Forcer texte", "Traite la colonne comme du texte libre."),
        "KEEP_RAW": ("Garder brut", "Évite les corrections automatiques de contenu."),
        "MEDIAN": ("Imputer par médiane", "Remplace les absences par la médiane."),
        "MEAN": ("Imputer par moyenne", "Remplace les absences par la moyenne."),
        "KNN": ("Imputer par voisins (KNN)", "Remplace les absences par la moyenne des lignes les plus proches sur les autres colonnes numériques."),
        "ITERATIVE": ("Imputer par régression itérative", "Prédit les absences à partir des autres colonnes numériques (modèle linéaire bayésien)."),
        "MODE": ("Imputer par mode", "Remplace les absences par la modalité la plus fréquente."),
        "CONSTANT": ("Imputer par constante", "Remplace les absences par une valeur définie."),
        "MISSING_CATEGORY": ("Modalité ‘manquant’", "Conserve l'absence comme modalité catégorielle."),
        "MISSING_INDICATOR": ("Ajouter un indicateur", "Ajoute une variable signalant les absences."),
        "WINSORIZE": ("Winsoriser", "Borne les extrêmes selon les quantiles."),
        "IQR": ("Limiter par intervalle interquartile", "Borne les extrêmes selon l'IQR."),
        "ZSCORE": ("Limiter par Z-score", "Borne les valeurs au-delà de moyenne ± k écarts-types (k = 3 par défaut)."),
        "MODIFIED_ZSCORE": ("Limiter par Z-score modifié", "Borne selon médiane ± k × 1,4826 × MAD (k = 3,5 par défaut), robuste aux extrêmes."),
        "REPLACE_MEDIAN": ("Remplacer les extrêmes par la médiane", "Les valeurs au-delà des clôtures IQR prennent la médiane du train."),
        "OUTLIER_TO_NAN": ("Convertir les extrêmes en manquants", "Les valeurs au-delà des clôtures IQR deviennent manquantes puis sont imputées."),
        "NONE": ("Aucun traitement", "Conserve les valeurs sans transformation supplémentaire."),
        "LOG1P": ("Transformation logarithmique", "Compresse la queue droite de la distribution."),
        "BOX_COX": ("Transformation Box-Cox", "Rend une variable positive plus symétrique."),
        "YEO_JOHNSON": ("Transformation Yeo-Johnson", "Réduit l'asymétrie sans imposer des valeurs positives."),
        "SQRT": ("Racine carrée", "Réduit modérément l'effet des grandes valeurs."),
        "CBRT": ("Racine cubique", "Réduit l'asymétrie et accepte les valeurs négatives et nulles."),
        "QUANTILE": ("Transformation par quantiles", "Rend la distribution proche d'une loi normale en conservant l'ordre des valeurs."),
        "RANK": ("Transformation par rangs", "Remplace chaque valeur par son rang relatif (0 à 1), insensible aux extrêmes."),
        "STANDARD": ("Standardiser", "Centre et réduit la variable."),
        "SCALE_STANDARD": ("Standardiser", "Centre et réduit la variable."),
        "ROBUST": ("Échelle robuste", "Centre par médiane et réduit par IQR."),
        "SCALE_ROBUST": ("Échelle robuste", "Centre par médiane et réduit par IQR."),
        "MINMAX": ("Échelle Min-Max", "Ramène les valeurs dans une plage bornée."),
        "SCALE_MINMAX": ("Échelle Min-Max", "Ramène les valeurs dans une plage bornée."),
        "MAXABS": ("Échelle valeur absolue maximale", "Préserve les zéros et réduit par maximum absolu."),
        "NO_SCALE": ("Aucune mise à l'échelle", "Conserve les unités d'origine."),
        "ONEHOT": ("Encodage One-Hot", "Crée une variable par modalité."),
        "TARGET_ENCODE": ("Encodage supervisé", "Encode les modalités à partir de la cible hors pli."),
        "FREQUENCY_ENCODE": ("Encodage par fréquence", "Remplace chaque modalité par sa fréquence."),
        "COUNT_ENCODE": ("Encodage par comptage", "Remplace chaque modalité par son nombre d'occurrences dans le train."),
        "BINARY_ENCODE": ("Encodage binaire", "Écrit le code de chaque modalité en bits : log2(K) colonnes au lieu de K."),
        "HASH_ENCODE": ("Encodage par hachage", "Projette les modalités dans un nombre fixe de colonnes, sans dictionnaire à stocker."),
        "LOO_ENCODE": ("Encodage Leave-One-Out", "Remplace chaque modalité par la moyenne de la cible des autres lignes de la même modalité (cible binaire ou continue requise)."),
        "ORDINAL": ("Encodage ordinal", "Associe un nombre selon l'ordre fourni."),
        "BINARY_MAP": ("Mappage binaire", "Code les deux modalités en 0 et 1."),
        "DATE_FEATURES": ("Extraire les dates", "Crée des variables calendaires."),
        "TFIDF_SVD": ("Résumé TF-IDF", "Vectorise le texte puis le réduit en composantes."),
        "TFIDF_NGRAMS": ("TF-IDF avec n-grammes", "Pondère mots et suites de mots (n-grammes) par TF-IDF et garde les plus fréquents."),
        "COUNT_VECTORIZER": ("Comptage de mots (n-grammes)", "Compte les occurrences de chaque mot ou n-gramme parmi les plus fréquents du train."),
        "HASH_TEXT": ("Hachage de texte (n-grammes)", "Projette mots et n-grammes dans un nombre fixe de colonnes, sans vocabulaire à stocker."),
        "KEEP_NUMERIC": ("Garder comme numérique", "Conserve les valeurs numériques."),
        "NO_TARGET": ("Aucune cible", "Aucune relation supervisée n'est calculée."),
        "TRANSFORM": ("Transformer la cible", "Transforme la cible de régression."),
        "KEEP_SELECTION": ("Garder", "Conserve la variable dans la sélection finale."),
    },
    "types": {
        "TARGET": "Cible", "DROP_ID": "Identifiant à supprimer", "DROP_CONSTANT": "Colonne constante",
        "DROP_HIGH_MISSING": "Trop de valeurs manquantes", "DROP_DUPLICATE": "Colonne dupliquée",
        "DROP_LEAKAGE": "Fuite de cible", "BINARY": "Binaire", "ORDINAL": "Ordinale",
        "NUMERIC_CONTINUOUS": "Numérique continue", "NUMERIC_DISCRETE": "Numérique discrète",
        "CATEGORICAL_LOW_CARD": "Catégorielle faible",
        "CATEGORICAL_HIGH_CARD": "Catégorielle forte", "DATETIME": "Date / heure", "TEXT": "Texte libre",
    },
}


def format_step_evidence(measure: str, threshold: str) -> str:
    """Formateur unique des preuves d'une recommandation d'étape."""
    return f"{str(measure).strip()} ; seuil : {str(threshold).strip()}"


def format_step_reason(action: str, measure: str, threshold: str) -> str:
    """Construit le résumé français depuis les champs de mesure et de seuil."""
    label = PREPROCESS_LABELS["actions"].get(str(action), ("Automatique", ""))[0]
    evidence = str(measure).strip()
    limit = str(threshold).strip()
    if action == "DROP":
        return f"{evidence} ; seuil : {limit} ; {label}."
    return f"{evidence} (seuil : {limit}) : {label}."


def recommendation_step_name(action: Any) -> str:
    """Renvoie l'étape à l'origine de l'action recommandée."""
    value = getattr(action, "value", action)
    return ACTION_RECOMMENDATION_STEP.get(str(value), "type")


def _pipeline_distribution_values(values: pd.Series, config: PreprocessConfig) -> pd.Series:
    """Reproduit le traitement des extrêmes et l'imputation avant la règle d'asymétrie."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return numeric
    method = str(config.outlier_method).lower()
    if method == "winsorize":
        numeric = numeric.clip(numeric.quantile(config.outlier_lower_quantile),
                                numeric.quantile(config.outlier_upper_quantile))
    elif method == "iqr":
        q1, q3 = numeric.quantile([.25, .75])
        spread = q3 - q1
        numeric = numeric.clip(q1 - config.outlier_iqr_multiplier * spread,
                                q3 + config.outlier_iqr_multiplier * spread)
    return numeric


def _is_time_only_column(series: pd.Series) -> bool:
    """Reconnaît une colonne contenant uniquement des heures, d'après ses valeurs."""
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
        "numeric": ["AUTO", "MEDIAN", "MEAN", "MODE", "CONSTANT", "KNN", "ITERATIVE", "MISSING_INDICATOR"],
        "categorical_low": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "categorical_high": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "binary": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "ordinal": ["AUTO", "MODE", "MISSING_CATEGORY", "CONSTANT"],
        "datetime": ["AUTO"], "text": ["AUTO"]},
    "outliers": {"any": ["AUTO"], "numeric": ["AUTO", "NONE", "WINSORIZE", "IQR", "ZSCORE", "MODIFIED_ZSCORE", "REPLACE_MEDIAN", "OUTLIER_TO_NAN"],
                 "binary": ["AUTO"], "ordinal": ["AUTO"], "categorical_low": ["AUTO"],
                 "categorical_high": ["AUTO"], "datetime": ["AUTO"], "text": ["AUTO"]},
    "distribution": {"any": ["AUTO"], "numeric": ["AUTO", "NONE", "LOG1P", "BOX_COX", "YEO_JOHNSON", "SQRT", "CBRT", "QUANTILE", "RANK"],
                     "categorical": ["AUTO"], "datetime": ["AUTO"], "text": ["AUTO"]},
    "scaling": {"any": ["AUTO"], "numeric": ["AUTO", "NONE", "STANDARD", "ROBUST", "MINMAX", "MAXABS"],
                "categorical": ["AUTO"], "datetime": ["AUTO"], "text": ["AUTO"]},
    "encoding": {"any": ["AUTO"], "binary": ["AUTO", "BINARY_MAP"], "ordinal": ["AUTO", "ORDINAL"],
                 "categorical_low": ["AUTO", "ONEHOT", "ORDINAL", "FREQUENCY_ENCODE", "COUNT_ENCODE", "BINARY_ENCODE", "HASH_ENCODE", "LOO_ENCODE"],
                 "categorical_high": ["AUTO", "ONEHOT", "ORDINAL", "TARGET_ENCODE", "FREQUENCY_ENCODE", "COUNT_ENCODE", "BINARY_ENCODE", "HASH_ENCODE", "LOO_ENCODE"]},
    "dates": {"any": ["AUTO"], "datetime": ["AUTO", "DATE_FEATURES"]},
    "text": {"any": ["AUTO"], "text": ["AUTO", "TFIDF_SVD", "TFIDF_NGRAMS", "COUNT_VECTORIZER", "HASH_TEXT"]},
    "target": {"any": ["AUTO"]},
    "selection": {"any": ["AUTO"]},
}


# Une action donnée seule (``ColumnOverride("LOG1P")``) vise une étape précise de la colonne.
_SINGLE_ACTION_STAGE: Dict[str, str] = {"DROP":"type", "YEO_JOHNSON":"distribution", "LOG1P":"distribution", "SQRT":"distribution", "CBRT":"distribution", "QUANTILE":"distribution", "RANK":"distribution", "COUNT_ENCODE":"encoding", "BINARY_ENCODE":"encoding", "HASH_ENCODE":"encoding", "LOO_ENCODE":"encoding", "SCALE_STANDARD":"scaling", "SCALE_ROBUST":"scaling", "SCALE_MINMAX":"scaling", "NO_SCALE":"scaling", "ONEHOT":"encoding", "TARGET_ENCODE":"encoding", "FREQUENCY_ENCODE":"encoding", "ORDINAL":"encoding", "BINARY_MAP":"encoding", "KEEP_NUMERIC":"type", "DATE_FEATURES":"dates", "TFIDF_SVD":"text", "TFIDF_NGRAMS":"text", "COUNT_VECTORIZER":"text", "HASH_TEXT":"text", "IMPUTE_MEDIAN":"missing", "MISSING_INDICATOR":"missing", "WINSORIZE":"outliers"}
_SINGLE_ACTION_VALUE: Dict[str, str] = {"KEEP_NUMERIC":"AUTO", "SCALE_STANDARD":"STANDARD", "SCALE_ROBUST":"ROBUST", "SCALE_MINMAX":"MINMAX", "NO_SCALE":"NONE", "IMPUTE_MEDIAN":"MEDIAN"}


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


def _recommend_scaler(values: pd.Series, config: Optional[PreprocessConfig] = None) -> Tuple[str, str, str]:
    """Règle statistique unique de choix du scaler, appelée par analyze et fit."""
    config = config or PreprocessConfig(verbose=0)
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 8:
        return "STANDARD", f"observations disponibles : {len(x)}", "échantillon < 8; scaler standard par défaut"
    q1, q3 = x.quantile(.25), x.quantile(.75)
    spread = q3 - q1
    iqr_pct = float(((x < q1 - config.outlier_iqr_multiplier * spread) | (x > q3 + config.outlier_iqr_multiplier * spread)).mean() * 100)
    mad = float(np.median(np.abs(x - np.median(x))))
    mad_pct = float((np.abs(x - np.median(x)) > config.outlier_mad_threshold * 1.4826 * mad).mean() * 100) if mad else 0.0
    kurt = float(x.kurt()) if len(x) > 3 else 0.0
    try:
        p_normal = float(stats.normaltest(x.sample(min(len(x), 5000), random_state=42)).pvalue)
    except Exception:
        p_normal = float("nan")
    zero_pct = float((x == 0).mean() * 100)
    if zero_pct >= config.scaler_zero_threshold * 100:
        return "MAXABS", f"zéros : {zero_pct:.1f}%", f"zéros ≥ {config.scaler_zero_threshold:.1%}"
    if x.min() >= config.scaler_bounded_min and x.max() <= config.scaler_bounded_max:
        return "MINMAX", f"bornes observées : [{x.min():.3g}, {x.max():.3g}]", f"bornes incluses dans [{config.scaler_bounded_min:g}, {config.scaler_bounded_max:g}]"
    if iqr_pct > config.scaler_outlier_threshold * 100 or mad_pct > config.scaler_outlier_threshold * 100 or abs(kurt) > config.scaler_kurtosis_threshold:
        return "ROBUST", f"outliers IQR : {iqr_pct:.2f}%, MAD : {mad_pct:.2f}%, kurtosis : {kurt:.2f}", f"outliers > {config.scaler_outlier_threshold:.1%} ou |kurtosis| > {config.scaler_kurtosis_threshold:g}"
    if pd.notna(p_normal) and p_normal > config.normality_p_threshold and iqr_pct <= config.scaler_outlier_threshold * 100 and mad_pct <= config.scaler_outlier_threshold * 100:
        return "STANDARD", f"p normalité : {p_normal:.3g}; outliers IQR : {iqr_pct:.2f}%, MAD : {mad_pct:.2f}%", f"p > {config.normality_p_threshold:g} et outliers ≤ {config.scaler_outlier_threshold:.1%}"
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
    rule: str = ""


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


def _numeric_profile(values: pd.Series, config: PreprocessConfig) -> Dict[str, float]:
    """Statistiques descriptives partagées par les règles d'imputation, d'extrêmes et de distribution."""
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(x)
    profile = {"n": n, "skew": 0.0, "kurtosis": 0.0, "iqr_pct": 0.0, "mad_pct": 0.0,
               "normal_p": float("nan"), "min": float("nan"), "unique": 0, "integer": False}
    if n == 0:
        return profile
    profile.update(min=float(x.min()), unique=int(x.nunique()),
                   integer=bool(np.all(np.isclose(x, np.round(x)))))
    if n > 2:
        profile["skew"] = float(stats.skew(x))
    if n > 3:
        profile["kurtosis"] = float(x.kurt())
    q1, q3 = x.quantile(.25), x.quantile(.75)
    spread = q3 - q1
    profile["iqr_pct"] = float(((x < q1 - config.outlier_iqr_multiplier * spread) |
                                (x > q3 + config.outlier_iqr_multiplier * spread)).mean() * 100)
    mad = float(np.median(np.abs(x - np.median(x))))
    if mad:
        profile["mad_pct"] = float((np.abs(x - np.median(x)) > config.outlier_mad_threshold * 1.4826 * mad).mean() * 100)
    if n >= 8:
        try:
            profile["normal_p"] = float(stats.normaltest(x.sample(min(n, 5000), random_state=42)).pvalue)
        except Exception:
            pass
    return profile


def _outlier_bounds(values: pd.Series, mode: str, config: PreprocessConfig,
                    params: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
    """Bornes (basse, haute) apprises sur le train pour un mode de traitement des extrêmes.

    Paramètres reconnus : ``lower``/``upper`` (WINSORIZE, quantiles), ``multiplier`` (IQR, REPLACE_MEDIAN,
    OUTLIER_TO_NAN), ``threshold`` (ZSCORE, MODIFIED_ZSCORE). Le Z-score modifié sans dispersion (MAD = 0) n'a
    pas de borne définie : aucune valeur n'est alors considérée extrême.
    """
    params = params if isinstance(params, dict) else {}
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    mode = str(mode).upper()
    if x.empty or mode == "NONE":
        return -np.inf, np.inf
    if mode == "WINSORIZE":
        lower = float(params.get("lower", config.outlier_lower_quantile))
        upper = float(params.get("upper", config.outlier_upper_quantile))
        if not 0 <= lower < upper <= 1:
            raise ValueError("Winsorisation : 0 ≤ lower < upper ≤ 1 requis.")
        return float(x.quantile(lower)), float(x.quantile(upper))
    if mode in {"IQR", "REPLACE_MEDIAN", "OUTLIER_TO_NAN"}:
        k = float(params.get("multiplier", config.outlier_iqr_multiplier))
        if k <= 0:
            raise ValueError("Le multiplicateur IQR doit être > 0.")
        q1, q3 = x.quantile(.25), x.quantile(.75)
        return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))
    if mode == "ZSCORE":
        k = float(params.get("threshold", 3.0))
        if k <= 0:
            raise ValueError("Le seuil Z-score doit être > 0.")
        sd = float(x.std(ddof=0))
        return (float(x.mean() - k * sd), float(x.mean() + k * sd)) if sd > 0 else (-np.inf, np.inf)
    if mode == "MODIFIED_ZSCORE":
        k = float(params.get("threshold", config.outlier_mad_threshold))
        if k <= 0:
            raise ValueError("Le seuil du Z-score modifié doit être > 0.")
        med = float(x.median())
        mad = float(np.median(np.abs(x - med)))
        return (med - k * 1.4826 * mad, med + k * 1.4826 * mad) if mad > 0 else (-np.inf, np.inf)
    raise ValueError(f"Mode d'extrêmes inconnu : {mode}")


def _treat_outliers(values: pd.Series, mode: str, config: PreprocessConfig,
                    params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """Applique le traitement d'extrêmes ``mode`` à des valeurs numériques (valeurs manquantes exclues)."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty or str(mode).upper() == "NONE":
        return numeric
    low, high = _outlier_bounds(numeric, mode, config, params)
    mode = str(mode).upper()
    if mode == "REPLACE_MEDIAN":
        return numeric.mask((numeric < low) | (numeric > high), float(numeric.median()))
    if mode == "OUTLIER_TO_NAN":
        return numeric[(numeric >= low) & (numeric <= high)]
    return numeric.clip(low, high)


def _evaluate_outlier_methods(values: pd.Series, config: PreprocessConfig) -> Dict[str, Any]:
    """Compare les méthodes de détection sur les valeurs réelles et qualifie la nature des extrêmes.

    ``nature`` est un indice, pas un verdict : une queue continue et asymétrique est plutôt légitime, une valeur
    isolée à plus de 10 IQR de la médiane (au plus 1 % des lignes) est plutôt une erreur de saisie possible, à vérifier côté métier.
    """
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    result: Dict[str, Any] = {"methods": {}, "nature": "indéterminée"}
    if len(x) < 8:
        return result
    for name in ("IQR", "ZSCORE", "MODIFIED_ZSCORE", "WINSORIZE"):
        low, high = _outlier_bounds(x, name, config)
        outside = (x < low) | (x > high)
        result["methods"][name] = {"lower": float(low), "upper": float(high),
                                   "outlier_pct": float(outside.mean() * 100), "outlier_count": int(outside.sum())}
    iqr_pct = result["methods"]["IQR"]["outlier_pct"]
    if iqr_pct == 0:
        result["nature"] = "aucun extrême détecté"
    else:
        spread = float(x.quantile(.75) - x.quantile(.25))
        far_count = int((np.abs(x - x.median()) > 10 * spread).sum()) if spread > 0 else 0
        if 1 <= far_count <= max(1, int(0.01 * len(x))):
            result["nature"] = "valeurs isolées très éloignées : erreur de saisie possible, à vérifier"
        elif abs(float(stats.skew(x))) > config.skew_threshold:
            result["nature"] = "queue continue et asymétrique : extrêmes plutôt légitimes"
        else:
            result["nature"] = "extrêmes modérés : légitimes ou erreurs, non tranchable statistiquement"
    return result


def _recommend_outliers(values: pd.Series, config: PreprocessConfig) -> Tuple[str, str, str]:
    """Choisit NONE / IQR / WINSORIZE d'après la fréquence et la forme des valeurs extrêmes.

    - aucun extrême détecté (IQR et MAD) : NONE, borner ne ferait que déformer 2 % des valeurs;
    - queue lourde (asymétrie, kurtosis ou extrêmes > seuil) : WINSORIZE, bornage par quantiles qui
      garde l'ordre des valeurs et ne comprime pas une queue légitime comme le ferait la clôture IQR;
    - sinon extrêmes rares sur une distribution plutôt symétrique : IQR.
    """
    if not config.adaptive_recommendations or str(config.outlier_method).lower() != "winsorize":
        method = str(config.outlier_method).upper()
        return (method if method in {"NONE", "WINSORIZE", "IQR"} else "WINSORIZE",
                "méthode globale de la configuration", "adaptive_recommendations désactivé ou méthode personnalisée")
    p = _numeric_profile(values, config)
    measure = (f"extrêmes IQR : {p['iqr_pct']:.2f}%, MAD : {p['mad_pct']:.2f}%; "
               f"asymétrie : {p['skew']:.2f}; kurtosis : {p['kurtosis']:.2f}")
    heavy_limit = config.scaler_outlier_threshold * 100
    if p["n"] < 8:
        return "WINSORIZE", f"observations disponibles : {p['n']}", "échantillon < 8; bornage par quantiles par défaut"
    if p["iqr_pct"] == 0 and p["mad_pct"] == 0:
        return "NONE", measure, "aucune valeur au-delà des clôtures IQR et MAD"
    if (abs(p["skew"]) > config.skew_threshold or abs(p["kurtosis"]) > config.scaler_kurtosis_threshold
            or p["iqr_pct"] > heavy_limit):
        return ("WINSORIZE", measure,
                f"|asymétrie| > {config.skew_threshold:g} ou |kurtosis| > {config.scaler_kurtosis_threshold:g} ou extrêmes > {heavy_limit:g}%")
    return "IQR", measure, "extrêmes rares sur distribution peu asymétrique"


def _strongest_dependence(values: pd.Series, others: Optional[pd.DataFrame],
                          max_columns: int = 50, max_rows: int = 20_000) -> Tuple[float, float, str]:
    """Plus forte dépendance (|Pearson|, |Spearman|) entre ``values`` et une autre colonne numérique.

    Les corrélations utilisent les lignes où les deux colonnes sont observées (au moins 50 paires).
    Retourne (|Pearson|, |Spearman|, nom de la colonne) ou (0, 0, "") si rien n'est calculable.
    """
    best_p, best_s, best_name = 0.0, 0.0, ""
    if others is None or others.empty:
        return best_p, best_s, best_name
    x = pd.to_numeric(values, errors="coerce")
    for name in list(others.columns)[:max_columns]:
        y = pd.to_numeric(others[name], errors="coerce")
        pair = pd.concat([x, y.reindex(x.index)], axis=1).dropna()
        if len(pair) < 50 or pair.iloc[:, 0].nunique() < 3 or pair.iloc[:, 1].nunique() < 3:
            continue
        if len(pair) > max_rows:
            pair = pair.sample(max_rows, random_state=0)
        try:
            pearson = abs(float(stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic))
            spearman = abs(float(stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic))
        except Exception:
            continue
        if max(pearson, spearman) > max(best_p, best_s):
            best_p, best_s, best_name = pearson, spearman, str(name)
    return best_p, best_s, best_name


def _recommend_imputation(values: pd.Series, config: PreprocessConfig,
                          others: Optional[pd.DataFrame] = None) -> Tuple[str, str, str]:
    """Choisit MEDIAN / MEAN / MODE / KNN / ITERATIVE pour une colonne numérique.

    - entier à peu de valeurs distinctes (discret) : MODE, la valeur la plus fréquente reste une valeur possible;
    - manquants entre le seuil d'indicateur et 40 %, au moins 100 valeurs observées et une autre colonne
      numérique fortement liée (|r| ≥ 0,6) : imputation multivariée, car les autres colonnes portent de
      l'information sur la valeur absente (ITERATIVE si la relation est linéaire, |Pearson| ≥ 0,6; KNN si elle
      est seulement monotone/non linéaire, |Spearman| ≥ 0,6). Ce lien n'est qu'un indice, il ne prouve pas que
      les absences sont MAR;
    - distribution symétrique, extrêmes IQR ≤ 1 % et normalité non rejetée : MEAN;
    - sinon : MEDIAN, robuste à l'asymétrie et aux extrêmes.
    """
    if not config.adaptive_recommendations or str(config.imputation_strategy).lower() != "median":
        strategy = str(config.imputation_strategy).upper()
        return (strategy if strategy in {"MEDIAN", "MEAN"} else "MEDIAN",
                "stratégie globale de la configuration", "adaptive_recommendations désactivé ou stratégie personnalisée")
    p = _numeric_profile(values, config)
    measure = (f"asymétrie : {p['skew']:.2f}; extrêmes IQR : {p['iqr_pct']:.2f}%; "
               f"p normalité : {p['normal_p']:.3g}; valeurs distinctes : {p['unique']}")
    if p["n"] < 8:
        return "MEDIAN", f"observations disponibles : {p['n']}", "échantillon < 8; médiane par défaut"
    if p["integer"] and p["unique"] <= config.cardinality_threshold:
        return "MODE", measure, f"entiers avec ≤ {config.cardinality_threshold} valeurs distinctes"
    missing_rate = float(pd.to_numeric(values, errors="coerce").isna().mean())
    if config.missing_indicator_threshold < missing_rate <= 0.40 and p["n"] >= 100:
        pearson, spearman, partner = _strongest_dependence(values, others)
        if max(pearson, spearman) >= 0.6:
            method = "ITERATIVE" if pearson >= 0.6 else "KNN"
            return (method, f"{measure}; lien le plus fort avec {partner!r} : |Pearson| = {pearson:.2f}, |Spearman| = {spearman:.2f}; "
                    f"manquants : {missing_rate:.1%}",
                    f"manquants entre {config.missing_indicator_threshold:.0%} et 40 %, ≥ 100 observations et |r| ≥ 0,6")
    normal_ok = pd.notna(p["normal_p"]) and p["normal_p"] > config.normality_p_threshold
    if abs(p["skew"]) < 0.5 and p["iqr_pct"] <= 1.0 and normal_ok:
        return "MEAN", measure, f"|asymétrie| < 0,5, extrêmes IQR ≤ 1 % et p normalité > {config.normality_p_threshold:g}"
    return "MEDIAN", measure, "asymétrie, extrêmes ou normalité rejetée : médiane robuste"


def _evaluate_distribution_alternatives(values: pd.Series, config: PreprocessConfig) -> Dict[str, Dict[str, Any]]:
    """Évalue toutes les transformations de distribution sur les valeurs réelles de la colonne.

    Pour chacune : validité mathématique (domaine), asymétrie obtenue et perte d'information.
    Aucune n'est appliquée; c'est la preuve chiffrée présentée à l'utilisateur.
    """
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    result: Dict[str, Dict[str, Any]] = {}
    if len(x) < 3:
        return result
    lo = float(x.min())

    def add(name: str, valid: bool, reason: str, fn=None, info: str = "aucune"):
        entry: Dict[str, Any] = {"valid": bool(valid), "reason": reason, "information_loss": info, "skew_after": None}
        if valid and fn is not None:
            try:
                after = np.asarray(fn(), dtype=float).ravel()
                if np.isfinite(after).all() and len(after) > 2:
                    entry["skew_after"] = float(stats.skew(after))
                else:
                    entry["valid"], entry["reason"] = False, "résultat non fini"
            except Exception as exc:
                entry["valid"], entry["reason"] = False, f"échec : {exc}"
        result[name] = entry

    n_q = int(min(1000, len(x)))
    frame = x.to_frame()
    add("NONE", True, "aucune condition", lambda: x.to_numpy())
    add("SQRT", lo >= 0, "valeurs ≥ 0 requises" if lo < 0 else "valeurs ≥ 0", lambda: np.sqrt(x))
    add("LOG1P", lo > -1, "valeurs > -1 requises" if lo <= -1 else "valeurs > -1", lambda: np.log1p(x))
    add("CBRT", True, "défini sur tous les réels", lambda: np.cbrt(x))
    add("BOX_COX", lo > 0, "valeurs strictement positives requises" if lo <= 0 else "valeurs > 0",
        lambda: PowerTransformer(method="box-cox").fit_transform(frame))
    add("YEO_JOHNSON", True, "défini sur tous les réels", lambda: PowerTransformer(method="yeo-johnson").fit_transform(frame))
    add("QUANTILE", len(x) >= 10, "au moins 10 observations requises" if len(x) < 10 else "≥ 10 observations",
        lambda: QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=config.random_state).fit_transform(frame),
        "écrase les distances entre valeurs (seul l'ordre est conservé)")
    add("RANK", len(x) >= 10, "au moins 10 observations requises" if len(x) < 10 else "≥ 10 observations",
        lambda: QuantileTransformer(n_quantiles=n_q, output_distribution="uniform", random_state=config.random_state).fit_transform(frame),
        "écrase les distances entre valeurs (seul l'ordre est conservé)")
    return result


def _recommend_distribution(values: pd.Series, config: PreprocessConfig) -> Tuple[str, str, str]:
    """Choisit NONE / SQRT / LOG1P / YEO_JOHNSON en comparant l'asymétrie obtenue après chaque transformation.

    ``values`` doit déjà avoir subi le traitement des extrêmes. Sous le seuil d'asymétrie : NONE. Sinon, les
    candidats compatibles avec le domaine (SQRT et LOG1P exigent des valeurs ≥ 0) sont évalués et celui qui
    laisse la plus faible asymétrie absolue gagne; une transformation simple (LOG1P, SQRT, sans paramètre à
    estimer) est préférée à Yeo-Johnson si elle reste à 0,25 de la meilleure. Box-Cox reste un choix manuel
    (équivalent de Yeo-Johnson sur données strictement positives, mais refuse les valeurs ≤ 0 au transform).
    """
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(x) < 3:
        return "NONE", f"observations disponibles : {len(x)}", "échantillon < 3"
    skew0 = float(stats.skew(x))
    if not config.adaptive_recommendations:
        return ("YEO_JOHNSON" if abs(skew0) > config.skew_threshold else "NONE",
                f"asymétrie : {skew0:.2f}", f"|asymétrie| > {config.skew_threshold:g}")
    if abs(skew0) <= config.skew_threshold:
        return "NONE", f"asymétrie : {skew0:.2f}", f"|skew| > {config.skew_threshold:g} non atteint : distribution acceptable"
    scores: Dict[str, float] = {}
    if x.min() >= 0:
        scores["SQRT"] = abs(float(stats.skew(np.sqrt(x))))
        scores["LOG1P"] = abs(float(stats.skew(np.log1p(x))))
    try:
        scores["YEO_JOHNSON"] = abs(float(stats.skew(PowerTransformer(method="yeo-johnson").fit_transform(x.to_frame()).ravel())))
    except Exception:
        pass
    if not scores:
        return "NONE", f"asymétrie : {skew0:.2f}", "aucune transformation applicable"
    best = min(scores, key=scores.get)
    simple = {k: v for k, v in scores.items() if k in {"LOG1P", "SQRT"}}
    if simple:
        best_simple = min(simple, key=simple.get)
        if simple[best_simple] <= scores[best] + 0.25:
            best = best_simple
    if scores[best] >= abs(skew0):
        return "NONE", f"asymétrie : {skew0:.2f}; aucune transformation ne la réduit", "gain d'asymétrie requis"
    if scores[best] > config.skew_threshold:
        detail = ", ".join(f"{name} → {value:.2f}" for name, value in sorted(scores.items(), key=lambda kv: kv[1]))
        return ("QUANTILE", f"asymétrie : {skew0:.2f}; |asymétrie| après : {detail}",
                f"|skew| > {config.skew_threshold:g} atteint même après la meilleure transformation paramétrique; la transformation par quantiles ne dépend pas de la forme")
    detail = ", ".join(f"{name} → {value:.2f}" for name, value in sorted(scores.items(), key=lambda kv: kv[1]))
    return best, f"asymétrie : {skew0:.2f}; |asymétrie| après : {detail}", f"|skew| > {config.skew_threshold:g} atteint; plus faible asymétrie résiduelle"


MNAR_NOTE = ("MNAR (le manque dépend de la valeur absente elle-même) ne peut pas être détecté à partir des données "
             "observées : seule la connaissance du domaine peut le suggérer ou l'écarter.")


def _p_text(p_value: float) -> str:
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3g}"


def _missingness_pool(frame: pd.DataFrame, target: Optional[pd.Series] = None, max_columns: int = 30) -> Dict[str, Tuple[str, pd.Series]]:
    """Colonnes observées servant à expliquer un manque : numériques, ou catégorielles à 2-20 modalités."""
    pool: Dict[str, Tuple[str, pd.Series]] = {}
    candidates = list(frame.items())
    if target is not None:
        candidates.append((str(target.name), target.reindex(frame.index)))
    for name, series in candidates:
        if len(pool) >= max_columns:
            break
        nunique = int(series.nunique(dropna=True))
        if nunique < 2:
            continue
        if pd.api.types.is_bool_dtype(series):
            pool[str(name)] = ("cat", series.astype(object))
        elif pd.api.types.is_numeric_dtype(series) and nunique > 15:
            pool[str(name)] = ("num", pd.to_numeric(series, errors="coerce"))
        elif nunique <= 20 and nunique / max(len(series), 1) < 0.5:
            pool[str(name)] = ("cat", series)
    return pool


def diagnose_missingness(series: pd.Series, pool: Dict[str, Tuple[str, pd.Series]], config: PreprocessConfig,
                         name: str = "") -> Dict[str, Any]:
    """Diagnostic indicatif du mécanisme de manque d'une colonne (MCAR compatible / MAR plausible / indéterminé).

    Le manque est un indicateur 0/1; on teste s'il dépend des AUTRES colonnes observées :
    - colonne numérique : test de Mann-Whitney U (lignes manquantes vs observées), effet = corrélation
      bisériale par rangs |2U/(n1·n2) − 1|;
    - colonne catégorielle : test du χ² sur le tableau croisé, effet = V de Cramér.
    Les p-values sont corrigées par Bonferroni sur le nombre de tests. Une dépendance est retenue si
    p corrigé < ``target_relation_p_threshold`` ET effet ≥ ``missingness_min_effect`` (une dépendance
    statistiquement détectable mais négligeable n'est pas retenue sur un grand échantillon).

    C'est un indice, pas une preuve : MCAR ne se démontre pas, et MNAR n'est pas testable sans connaître les
    valeurs manquantes (voir ``MNAR_NOTE``).
    """
    alpha = float(config.target_relation_p_threshold)
    min_effect = float(config.missingness_min_effect)
    indicator = series.isna()
    n_missing, n_observed = int(indicator.sum()), int((~indicator).sum())
    result: Dict[str, Any] = {"verdict": "INDETERMINE", "n_missing": n_missing, "n_observed": n_observed,
                              "n_tests": 0, "alpha": alpha, "min_effect": min_effect, "top": [],
                              "mnar": MNAR_NOTE, "implication": ""}
    if n_missing == 0:
        result["verdict"] = "AUCUN"
        result["texte"] = "Aucune valeur manquante."
        return result
    if n_missing < 5 or n_observed < 5 or len(series) < 30:
        result["texte"] = (f"Indéterminé : {n_missing} valeurs manquantes et {n_observed} observées "
                           "(au moins 5 de chaque et 30 lignes sont nécessaires pour tester).")
        return result
    tests: List[Dict[str, Any]] = []
    for other, (kind, values) in pool.items():
        if other == name:
            continue
        valid = values.notna()
        try:
            if kind == "num":
                a, b = values[valid & indicator].astype(float), values[valid & ~indicator].astype(float)
                if len(a) < 5 or len(b) < 5 or values[valid].nunique() < 2:
                    continue
                u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
                effect = abs(2.0 * float(u_stat) / (len(a) * len(b)) - 1.0)
                test_name = "Mann-Whitney"
            else:
                table = pd.crosstab(indicator[valid], values[valid].astype(str))
                if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() < 30:
                    continue
                chi2, p_value, _, _ = stats.chi2_contingency(table)
                n_obs = float(table.to_numpy().sum())
                effect = float(np.sqrt(chi2 / (n_obs * (min(table.shape) - 1))))
                test_name = "χ²"
        except (ValueError, TypeError):
            continue
        if np.isfinite(p_value) and np.isfinite(effect):
            tests.append({"colonne": other, "test": test_name, "p": float(p_value), "effet": float(effect)})
    result["n_tests"] = len(tests)
    if not tests:
        result["texte"] = "Indéterminé : aucune autre colonne observée ne permet de tester la dépendance du manque."
        return result
    for t in tests:
        t["p_adj"] = float(min(1.0, t["p"] * len(tests)))
    tests.sort(key=lambda t: (t["p_adj"], -t["effet"]))
    result["top"] = tests[:3]
    explaining = [t for t in tests if t["p_adj"] < alpha and t["effet"] >= min_effect]
    if explaining:
        best = max(explaining, key=lambda t: t["effet"])
        result["verdict"] = "MAR_PLAUSIBLE"
        result["texte"] = (f"MAR plausible : le manque dépend de « {best['colonne']} » ({best['test']}, "
                           f"{_p_text(best['p_adj'])} après correction, effet = {best['effet']:.2f} ≥ {min_effect:g}); "
                           f"{len(explaining)} colonne(s) sur {len(tests)} testées.")
        result["implication"] = ("Une imputation multivariée (KNN ou itérative) ou un indicateur de manquant exploite cette "
                                 "dépendance; une imputation par médiane ou moyenne peut biaiser les relations.")
    else:
        weak = [t for t in tests if t["p_adj"] < alpha]
        result["verdict"] = "MCAR_COMPATIBLE"
        result["texte"] = (f"Compatible avec MCAR : aucune dépendance notable avec les {len(tests)} autres colonnes testées "
                           f"(p corrigé ≥ {alpha:g} ou effet < {min_effect:g})"
                           + (f"; {len(weak)} dépendance(s) détectable(s) mais négligeable(s)" if weak else "") + ". MCAR n'est pas prouvé.")
        result["implication"] = "Une imputation simple (médiane, moyenne, mode) est raisonnable si le manque est réellement aléatoire."
    return result


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
        adaptive_recommendations: Si True, l'imputation, le traitement des extrêmes et la transformation de distribution
            sont choisis colonne par colonne selon les statistiques (asymétrie, kurtosis, normalité, outliers, type),
            tant que imputation_strategy='median' et outlier_method='winsorize' (valeurs par défaut) ne sont pas modifiés.
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
    outlier_mad_threshold: float = 3.5
    scaler_zero_threshold: float = 0.50
    scaler_outlier_threshold: float = 0.05
    scaler_kurtosis_threshold: float = 3.0
    scaler_bounded_min: float = 0.0
    scaler_bounded_max: float = 1.0
    normality_p_threshold: float = 0.05
    missingness_min_effect: float = 0.20
    date_parse_threshold: float = 0.80
    target_relation_p_threshold: float = 0.05
    missing_indicator_threshold: float = 0.05
    variance_threshold: float = 0.0
    constant_threshold: float = 0.99
    identifier_unique_threshold: float = 0.90
    value_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
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
    adaptive_recommendations: bool = True
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
        self.replace_values_: Dict[str, float] = {}   # colonne -> valeur de remplacement (NaN : conversion en manquant)
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> Winsorizer:
        self.columns_ = list(X.columns)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.replace_values_ = {}

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
                    if col in self.replace_values_:
                        outside = (X_out[col] < low) | (X_out[col] > high)
                        X_out[col] = X_out[col].mask(outside, self.replace_values_[col])
                    else:
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
            time_only = _is_time_only_column(X[col])
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
                cols = [f"{col}_svd_{i}" for i in range(reduced.shape[1])]
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


class NgramTextTransformer(BaseEstimator, TransformerMixin):
    """Vectorisation de texte avec n-grammes : TF-IDF, comptage ou hachage, apprise sur le train seulement.

    - ``TFIDF_NGRAMS`` : TfidfVectorizer, ``max_features`` termes les plus fréquents du train.
    - ``COUNT_VECTORIZER`` : CountVectorizer, mêmes termes retenus, valeurs = occurrences.
    - ``HASH_TEXT`` : HashingVectorizer sans vocabulaire, ``n_features`` colonnes fixes, norme L2.
    Les mots absents du vocabulaire d'entraînement (TF-IDF, comptage) sont ignorés; le hachage les projette.
    """

    METHODS = ("TFIDF_NGRAMS", "COUNT_VECTORIZER", "HASH_TEXT")

    def __init__(self, methods: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Dict[str, Any]]] = None,
                 default_max_features: int = 15, default_n_features: int = 16):
        self.methods = methods
        self.params = params
        self.default_max_features = default_max_features
        self.default_n_features = default_n_features
        self.vectorizers_: Dict[str, Any] = {}
        self.output_names_: Dict[str, List[str]] = {}
        self.columns_: List[str] = []

    @staticmethod
    def _ngram_range(col: str, params: Dict[str, Any]) -> Tuple[int, int]:
        raw = params.get("ngram_range", (1, 2))
        try:
            low, high = int(raw[0]), int(raw[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"ngram_range invalide pour {col!r} : attendu (min, max) entiers.") from exc
        if not 1 <= low <= high <= 5:
            raise ValueError(f"ngram_range invalide pour {col!r} : 1 ≤ min ≤ max ≤ 5 requis, reçu ({low}, {high}).")
        return low, high

    def fit(self, X: pd.DataFrame, y=None) -> "NgramTextTransformer":
        from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
        self.columns_ = list(X.columns)
        self.vectorizers_, self.output_names_ = {}, {}
        for col in self.columns_:
            method = (self.methods or {}).get(col, "TFIDF_NGRAMS")
            if method not in self.METHODS:
                raise ValueError(f"Méthode texte inconnue pour {col!r} : {method}.")
            params = (self.params or {}).get(col) or {}
            ngram_range = self._ngram_range(col, params)
            text = X[col].fillna("").astype(str)
            if method == "HASH_TEXT":
                n_features = int(params.get("n_features", self.default_n_features))
                if n_features < 1:
                    raise ValueError(f"n_features doit être ≥ 1 pour {col!r}.")
                self.vectorizers_[col] = HashingVectorizer(n_features=n_features, ngram_range=ngram_range,
                                                           alternate_sign=False, norm="l2", token_pattern=r"(?u)\b\w+\b")
                self.output_names_[col] = [f"{col}_texthash_{i}" for i in range(n_features)]
                continue
            max_features = int(params.get("max_features", self.default_max_features))
            if max_features < 1:
                raise ValueError(f"max_features doit être ≥ 1 pour {col!r}.")
            cls = TfidfVectorizer if method == "TFIDF_NGRAMS" else CountVectorizer
            vec = cls(ngram_range=ngram_range, max_features=max_features, token_pattern=r"(?u)\b\w+\b")
            try:
                vec.fit(text)
            except ValueError:   # vocabulaire vide (texte vide ou uniquement des mots exclus)
                self.vectorizers_[col] = None
                self.output_names_[col] = [f"{col}_text_0"]
                continue
            prefix = "tfidf" if method == "TFIDF_NGRAMS" else "cnt"
            names, seen = [], set()
            for term in vec.get_feature_names_out():
                name = f"{col}_{prefix}_" + re.sub(r"\s+", "_", str(term))
                while name in seen:
                    name += "_"
                seen.add(name)
                names.append(name)
            self.vectorizers_[col] = vec
            self.output_names_[col] = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for col in self.columns_:
            names = self.output_names_[col]
            text = X[col].fillna("").astype(str) if col in X.columns else pd.Series([""] * len(X), index=X.index)
            vec = self.vectorizers_.get(col)
            if vec is None:
                frames.append(pd.DataFrame(np.zeros((len(X), 1)), index=X.index, columns=names))
                continue
            matrix = vec.transform(text).toarray().astype(np.float32)
            frames.append(pd.DataFrame(matrix, index=X.index, columns=names))
        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([n for col in self.columns_ for n in self.output_names_[col]], dtype=object)


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
        self.auto_imputation_: Dict[str, str] = {}
        self.multivariate_imputers_: Dict[str, Any] = {}
        self.column_override_params_: Dict[str, Dict[str, Any]] = {}
        self._last_keep_positions_: Optional[np.ndarray] = None
        self.cleaning_decisions_: Dict[str, Dict[str, Any]] = {}
        self.dataset_cleaning_: Dict[str, Any] = {}
        self.categorical_imputers_: Dict[str, Any] = {}
        self.frequency_override_cols_: List[str] = []
        self.custom_encoding_cols_: Dict[str, str] = {}
        self.custom_encoders_: Dict[str, Any] = {}
        self._loo_train_: Dict[str, np.ndarray] = {}

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
        self.ngram_text_cols_: Dict[str, str] = {}
        self.ngram_text_transformer_: Optional[NgramTextTransformer] = None

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

    def _add_override_note(self, col: str, text: str) -> None:
        """Ajoute une note de choix utilisateur sans écraser celles déjà enregistrées pour la colonne."""
        previous = self.user_override_notes_.get(col)
        self.user_override_notes_[col] = f"{previous} ; {text}" if previous else text

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
            configured_bounds = (self.config.value_bounds or {}).get(str(col))
            if configured_bounds is not None:
                lower, upper = configured_bounds
            mask = pd.Series(False, index=out.index)
            if lower is not None or upper is not None:
                numeric = pd.to_numeric(series, errors="coerce")
                if lower is not None:
                    mask |= numeric < lower
                if upper is not None:
                    mask |= numeric > upper
            if pd.api.types.is_datetime64_any_dtype(series) and not _is_time_only_column(series):
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
            unique_ratio = n_unique / max(n_rows, 1)
            textual_identifier = (pd.api.types.is_object_dtype(series) or
                                  pd.api.types.is_string_dtype(series))
            integer_sequence = False
            if pd.api.types.is_integer_dtype(series) and unique_ratio > self.config.identifier_unique_threshold:
                ordered_values = np.sort(pd.to_numeric(non_null, errors="coerce").dropna().unique())
                diffs = np.diff(ordered_values)
                integer_sequence = len(diffs) > 0 and bool(np.all(diffs == diffs[0]))
            is_id_candidate = (n_rows > 30 and unique_ratio > self.config.identifier_unique_threshold
                               and (textual_identifier or integer_sequence))
            # Vérifier si ce n'est pas un texte long
            # Une colonne date/heure quasi unique (horodatage) n'est pas un ID
            date_clean_action, _ = self._override_value(str(col), "clean_dates")
            if is_id_candidate and date_clean_action != "KEEP" and self._is_date_column(series):
                is_id_candidate = False
            if is_id_candidate:
                mean_str_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
                if mean_str_len < 30:
                    meta["type"] = FeatureType.DROP_ID
                    meta["reason"] = (f"Profil d'identifiant : uniques/lignes = {unique_ratio:.1%} "
                                       f"(seuil > {self.config.identifier_unique_threshold:.0%})")
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

        skip = set(self.date_cols_) | set(self.text_cols_) | set(self.ngram_text_cols_)
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
        missing_pool = _missingness_pool(clean, raw[target] if target is not None else None)
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
            rule_recs: Dict[str, Tuple[str, str, str]] = {}
            if kind_name.startswith("NUMERIC"):
                outlier_rule = _recommend_outliers(s, self.config)
                treated_values = _treat_outliers(s, outlier_rule[0], self.config)
                rule_recs = {"outliers": outlier_rule,
                             "distribution": _recommend_distribution(treated_values, self.config),
                             "missing": _recommend_imputation(
                                 s, self.config, clean[[c for c in prep.num_cols_ if c != clean_name and c in clean.columns]])}
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
                time_only = _is_time_only_column(nonnull)
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
                distribution_values = (_treat_outliers(s, rule_recs["outliers"][0], self.config)
                                       if rule_recs else _pipeline_distribution_values(s, self.config))
                decision_skew = float(stats.skew(distribution_values)) if len(distribution_values) > 2 else 0.0
                q1, q3 = vals.quantile(.25), vals.quantile(.75)
                iqr = q3 - q1
                outlier_pct = float((((vals < q1 - self.config.outlier_iqr_multiplier * iqr) | (vals > q3 + self.config.outlier_iqr_multiplier * iqr)).mean()) * 100) if len(vals) else 0.0
                stats_row.update(min=vals.min() if len(vals) else np.nan, max=vals.max() if len(vals) else np.nan,
                                 mean=vals.mean() if len(vals) else np.nan, median=vals.median() if len(vals) else np.nan,
                                 std=vals.std() if len(vals) else np.nan, skew=skew,
                                  kurtosis=float(vals.kurt()) if len(vals) > 3 else np.nan, outliers_pct=outlier_pct,
                                  skew_after_outlier_treatment=decision_skew)
                mad = float(np.median(np.abs(vals - np.median(vals)))) if len(vals) else 0.0
                mad_rate = float((np.abs(vals - np.median(vals)) > self.config.outlier_mad_threshold * 1.4826 * mad).mean() * 100) if mad else 0.0
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
                dist_choice = rule_recs["distribution"][0] if rule_recs else (
                    "YEO_JOHNSON" if abs(decision_skew) > self.config.skew_threshold else "NONE")
                action = {"YEO_JOHNSON": Action.YEO_JOHNSON, "LOG1P": Action.LOG1P,
                          "SQRT": Action.SQRT, "QUANTILE": Action.QUANTILE}.get(dist_choice, Action.KEEP_NUMERIC)
                alternatives = [Action.DROP, Action.YEO_JOHNSON, Action.LOG1P, Action.SQRT, Action.WINSORIZE,
                                Action.SCALE_STANDARD, Action.SCALE_ROBUST, Action.SCALE_MINMAX, Action.NO_SCALE]
                reason = (f"Asymétrie après bornage = {decision_skew:.2f}; {dist_choice} choisi car il laisse la plus faible "
                          f"asymétrie résiduelle (seuil {self.config.skew_threshold:g})." if action != Action.KEEP_NUMERIC else
                          f"Asymétrie après bornage = {decision_skew:.2f}, sous le seuil {self.config.skew_threshold:g}; traitement numérique standard.")
            # Les 11 étapes sont explicites. Les étapes non configurées reprennent le comportement du pipeline.
            def rec(action_name: str, measure: str, threshold: str, explanation: str,
                    choices: List[str], conf: str = "moyenne", rule: str = "") -> Dict[str, Any]:
                return {"action": action_name, "measure": measure, "threshold": threshold,
                        "explanation": explanation, "confidence": conf, "choices": choices,
                        "rule": rule or "règle de décision basée sur les mesures de la colonne",
                        "domain": kind_name}
            step_recs["nettoyage"] = rec("CLEAN_DEFAULT", f"{missing:.1%} de valeurs manquantes", "règles de nettoyage du module",
                "Nettoyage des sous-étapes sélectionnées.", ["AUTO"])
            false_tokens = text_values.str.strip().str.casefold().isin(DEFAULT_NA_VALUES)
            found_false_counts = text_values[false_tokens].value_counts().to_dict()
            found_false = ", ".join(
                f"{token} ({count})" for token, count in found_false_counts.items()
            ) or "aucun"
            raw_nonnull = raw_s.dropna()
            numeric_converted = (raw_nonnull.map(self._clean_numeric_str_value)
                                 if (pd.api.types.is_object_dtype(raw_s) or pd.api.types.is_string_dtype(raw_s))
                                 else pd.Series(dtype=float))
            numeric_text_ratio = float(numeric_converted.notna().mean()) if len(numeric_converted) else 0.0
            convertible_values = raw_nonnull[numeric_converted.notna()] if len(numeric_converted) else pd.Series(dtype=object)
            units = sorted(set(convertible_values.astype(str).str.extract(r"([A-Za-z%]{1,8})\s*$", expand=False).dropna().str.upper())) if len(convertible_values) else []
            units_text = ", ".join(units) if units else "aucune"
            date_parse_rate = 0.0
            mixed_date_formats = 0
            if len(nonnull) and kind_name == "DATETIME":
                date_source = raw_s.dropna()
                time_only = _is_time_only_column(date_source)
                parsed_dates = _parse_datetime_values(date_source, time_only)
                date_parse_rate = float(parsed_dates.notna().mean())
                if not time_only:
                    format_keys = date_source.astype(str).map(lambda v: "ymd" if re.match(r"^\d{4}[-/]", v) else "dmy" if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", v) else "other")
                    mixed_date_formats = max(format_keys.nunique() - 1, 0)
            impossible_count = int(getattr(prep, "impossible_counts_", {}).get(str(col), 0))
            _bounds = (self.config.value_bounds or {}).get(str(col))
            configured_bounds_text = f"[{_bounds[0]}, {_bounds[1]}]" if _bounds is not None else ""
            duplicate_rows = int(raw.duplicated().sum())
            step_recs["clean_missing_tokens"] = rec(
                "TO_NAN", f"{int(false_tokens.sum())} cellules; marqueurs trouvés : {found_false}", "marqueurs de valeurs manquantes reconnus",
                "Les marqueurs de manque reconnus deviennent des valeurs imputables; les conserver convient s'ils représentent une vraie modalité.", ["TO_NAN", "KEEP", "SELECTED_TO_NAN"])
            step_recs["clean_variants"] = rec(
                "MERGE", f"{variant_count} variantes fusionnables", "même texte après trim/casse/accents",
                "La fusion normalise les espaces, la casse et les accents pour regrouper les modalités équivalentes.", ["MERGE", "KEEP"])
            step_recs["clean_numeric_text"] = rec(
                "CONVERT" if numeric_text_ratio >= self.config.numeric_string_threshold else "KEEP",
                f"{numeric_text_ratio:.1%} de nombres convertibles; unités : {units_text}", f"conversion si ≥ {self.config.numeric_string_threshold:.0%}",
                "La conversion transforme les chaînes numériques en mesures et retire leurs unités textuelles; garder le texte convient aux codes.", ["AUTO", "CONVERT", "KEEP"])
            step_recs["clean_dates"] = rec(
                "PARSE" if date_parse_rate >= self.config.date_parse_threshold and date_parse_rate > 0 else "KEEP",
                f"{date_parse_rate:.1%} parsables; {1-date_parse_rate:.1%} non parsables; formats mélangés : {mixed_date_formats}", f"parser si ≥ {self.config.date_parse_threshold:.0%}",
                "Le parsing convertit les dates reconnues pour permettre la création de variables calendaires; garder le texte évite une interprétation incertaine.", ["AUTO", "PARSE", "KEEP", "DROP_COLUMN"])
            step_recs["clean_impossible"] = rec(
                "NAN" if impossible_count else "IGNORE", f"cas impossibles : {impossible_count}",
                (f"bornes configurées {configured_bounds_text}" if configured_bounds_text else "aucune borne configurée")
                + "; dates futures détectées automatiquement",
                "Seules les valeurs hors des bornes explicitement configurées (value_bounds) et les dates postérieures à aujourd'hui sont traitées comme impossibles; sans borne, les valeurs numériques restent inchangées.", ["AUTO", "NAN", "CLIP", "DROP_ROWS", "IGNORE"])
            step_recs["duplicates"] = rec(
                "DROP" if duplicate_rows else "KEEP", f"lignes dupliquées : {duplicate_rows}", "égalité exacte de toutes les cellules",
                "La suppression des lignes strictement identiques évite de surpondérer les répétitions accidentelles.", ["DROP", "KEEP"])
            step_recs["type"] = rec("DROP" if kind_name.startswith("DROP") else "AUTO", f"{nunique} valeurs uniques", f"identifiant si ratio > {self.config.identifier_unique_threshold:.1%}; suppression si manquants ≥ {self.config.missing_threshold:.1%}",
                "La détection conserve les variables exploitables et écarte les profils vides, constants ou identifiants selon les seuils configurés.", ["AUTO", "DROP", "NUMERIC", "CATEGORICAL", "DATETIME", "TEXT"], confidence)
            missing_explanations = {
                "MEDIAN": "Médiane : robuste à l'asymétrie et aux valeurs extrêmes.",
                "MEAN": "Moyenne : distribution symétrique, extrêmes ≤ 1 % et normalité non rejetée.",
                "MODE": "Mode : variable entière à peu de valeurs distinctes, l'imputation reste une valeur réelle.",
                "KNN": "KNN : la colonne est liée de façon monotone aux autres colonnes numériques; les lignes voisines estiment mieux la valeur absente qu'une constante.",
                "ITERATIVE": "Imputation itérative : la colonne est linéairement liée aux autres colonnes numériques; une régression exploite cette information.",
            }
            if rule_recs:
                miss_action, miss_stats, miss_limit = rule_recs["missing"]
                step_recs["missing"] = rec(miss_action,
                    f"{missing:.1%} de valeurs manquantes; {miss_stats}",
                    f"{miss_limit}; indicateur ajouté si > {self.config.missing_indicator_threshold:.1%} de manquants",
                    missing_explanations.get(miss_action, "Imputation ajustée sur le jeu d'entraînement.")
                    + " Un indicateur de manquant complète l'imputation au-delà du seuil de manquants.",
                    ["AUTO", "MEDIAN", "MEAN", "MODE", "CONSTANT", "KNN", "ITERATIVE", "MISSING_INDICATOR"])
            else:
                step_recs["missing"] = rec("AUTO",
                    f"{missing:.1%} de valeurs manquantes", f"indicateur si > {self.config.missing_indicator_threshold:.1%}",
                    "Les absences deviennent une modalité '__MISSING__' dédiée; un autre choix est possible (mode, constante).",
                    ["AUTO", "MEDIAN", "MEAN", "MODE", "CONSTANT", "MISSING_INDICATOR"])
            if rule_recs:
                out_action, out_measure, out_limit = rule_recs["outliers"]
                out_explanations = {
                    "NONE": "Aucune valeur extrême détectée : borner déformerait inutilement des valeurs légitimes.",
                    "WINSORIZE": "Queue lourde ou extrêmes fréquents : le bornage par quantiles limite leur influence sans écraser la queue comme la clôture IQR.",
                    "IQR": "Extrêmes rares sur distribution peu asymétrique : la clôture IQR les ramène aux bornes habituelles.",
                }
                step_recs["outliers"] = rec(out_action, out_measure, out_limit,
                    out_explanations.get(out_action, "Traitement des extrêmes selon la configuration."),
                    ["AUTO", "NONE", "WINSORIZE", "IQR", "ZSCORE", "MODIFIED_ZSCORE", "REPLACE_MEDIAN", "OUTLIER_TO_NAN"])
                step_recs["outliers"]["evaluations"] = _evaluate_outlier_methods(s, self.config)
            else:
                step_recs["outliers"] = rec("AUTO", "colonne non numérique", "aucun",
                    "Les valeurs extrêmes ne concernent que les colonnes numériques.", ["AUTO"])
            normality_p = stats_row.get("normality_p", np.nan)
            p_text = ("p < 0.001" if pd.notna(normality_p) and normality_p < .001 else
                      f"p = {normality_p:.3g}" if pd.notna(normality_p) else "p-value indisponible")
            normality_text = (f"normalité rejetée ({p_text})" if pd.notna(normality_p) and normality_p < self.config.normality_p_threshold else
                              f"normalité non rejetée ({p_text})")
            skew = float(stats_row.get("skew_after_outlier_treatment", stats_row.get("skew", np.nan)))
            skew_reaches_threshold = abs(skew) > self.config.skew_threshold
            distribution_explanation = (
                "Yeo-Johnson réduit l'asymétrie et accepte les valeurs négatives, ce qui aide les modèles sensibles à la distribution."
                if action == Action.YEO_JOHNSON else
                "La distribution conserve son échelle d'origine; ce choix préserve les unités et convient aux méthodes peu sensibles à l'asymétrie."
            )
            if rule_recs:
                dist_action, dist_measure, dist_limit = rule_recs["distribution"]
                dist_text = {
                    "NONE": "La distribution est acceptable : les unités d'origine sont préservées.",
                    "SQRT": "La racine carrée réduit modérément la queue droite (typique des comptages).",
                    "LOG1P": "Le log(1+x) compresse une queue droite forte sur des valeurs positives ou nulles.",
                    "YEO_JOHNSON": "Yeo-Johnson réduit l'asymétrie et accepte les valeurs négatives.",
                    "QUANTILE": "Aucune transformation paramétrique ne ramène l'asymétrie sous le seuil; la transformation par quantiles impose une forme régulière en conservant l'ordre.",
                }.get(dist_action, distribution_explanation)
                step_recs["distribution"] = rec(dist_action if dist_action != "NONE" else "NONE",
                    f"{dist_measure}; {normality_text}", dist_limit, dist_text,
                    ["AUTO", "NONE", "LOG1P", "BOX_COX", "YEO_JOHNSON", "SQRT", "CBRT", "QUANTILE", "RANK"])
                step_recs["distribution"]["evaluations"] = _evaluate_distribution_alternatives(treated_values, self.config)
            else:
                step_recs["distribution"] = rec("AUTO", "colonne non numérique", "aucun", distribution_explanation,
                    ["AUTO", "NONE", "LOG1P", "BOX_COX", "YEO_JOHNSON", "SQRT"])
            scale_recommendation = "AUTO"
            scale_measure = "type non numérique"
            scale_threshold = "aucun"
            scale_reason = "Aucune mise à l'échelle particulière n'est nécessaire pour ce type de colonne."
            if kind_name.startswith("NUMERIC"):
                scale_recommendation, scale_measure, scale_threshold = _recommend_scaler(s, self.config)
                scale_reason = {
                    "ROBUST": "L'échelle robuste centre par la médiane et réduit par l'IQR; elle limite l'effet des valeurs extrêmes.",
                    "STANDARD": "La standardisation centre et réduit la variable; elle convient aux données proches d'une distribution régulière.",
                    "MINMAX": "L'échelle Min-Max ramène la variable dans une plage bornée tout en préservant l'ordre.",
                    "MAXABS": "L'échelle par maximum absolu préserve les zéros et convient aux données creuses.",
                }.get(scale_recommendation, "Les unités d'origine sont conservées, ce qui convient aux méthodes peu sensibles à l'échelle.")
            step_recs["scaling"] = rec(scale_recommendation, scale_measure, scale_threshold,
                scale_reason, ["AUTO", "NONE", "STANDARD", "ROBUST", "MINMAX", "MAXABS"])
            encoding_explanation = {
                Action.ONEHOT: "One-Hot crée une variable par modalité et convient aux catégories sans ordre.",
                Action.TARGET_ENCODE: "L'encodage supervisé résume les modalités par une statistique de cible calculée hors pli.",
                Action.FREQUENCY_ENCODE: "L'encodage par fréquence remplace les modalités par leur fréquence d'apparition.",
                Action.ORDINAL: "L'encodage ordinal associe des nombres aux modalités selon un ordre défini.",
                Action.BINARY_MAP: "Le mappage binaire transforme deux modalités en indicateurs numériques.",
            }.get(action, "Aucun encodage particulier n'est recommandé pour le type observé.")
            step_recs["encoding"] = rec(action.value if action in (Action.ONEHOT, Action.TARGET_ENCODE, Action.FREQUENCY_ENCODE, Action.ORDINAL, Action.BINARY_MAP) else "NONE",
                f"{nunique} modalités", f"haute cardinalité > {self.config.cardinality_threshold} modalités", encoding_explanation,
                ["AUTO", "ONEHOT", "ORDINAL", "TARGET_ENCODE", "FREQUENCY_ENCODE", "BINARY_MAP"])
            step_recs["dates"] = rec("DATE_FEATURES" if kind_name == "DATETIME" else "AUTO", f"{nunique} valeurs uniques", "date parsable",
                "Les composantes calendaires rendent les dates exploitables par les modèles numériques.", ["AUTO", "DATE_FEATURES"])
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
                    threshold=f"date reconnue; parsing ≥ {self.config.date_parse_threshold:.0%}",
                    explanation="La plage, le taux de parsing et la granularité justifient l'extraction des composantes calendaires.")
            step_recs["text"] = rec("TFIDF_SVD" if kind_name == "TEXT" else "AUTO", f"{nunique} valeurs uniques", "profil de texte libre détecté selon les valeurs",
                "TF-IDF suivi de SVD représente le vocabulaire en composantes numériques compactes.", ["AUTO", "TFIDF_SVD", "TFIDF_NGRAMS", "COUNT_VECTORIZER", "HASH_TEXT"])
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
                "YEO_JOHNSON": ("distribution", "YEO_JOHNSON"), "LOG1P": ("distribution", "LOG1P"), "SQRT": ("distribution", "SQRT"),
                "QUANTILE": ("distribution", "QUANTILE"), "RANK": ("distribution", "RANK"), "CBRT": ("distribution", "CBRT"),
                "COUNT_ENCODE": ("encoding", "COUNT_ENCODE"), "BINARY_ENCODE": ("encoding", "BINARY_ENCODE"), "HASH_ENCODE": ("encoding", "HASH_ENCODE"), "LOO_ENCODE": ("encoding", "LOO_ENCODE"),
                "WINSORIZE": ("outliers", "WINSORIZE"), "SCALE_STANDARD": ("scaling", "STANDARD"),
                "SCALE_ROBUST": ("scaling", "ROBUST"), "SCALE_MINMAX": ("scaling", "MINMAX"),
                "NO_SCALE": ("scaling", "NONE"), "ONEHOT": ("encoding", "ONEHOT"),
                "TARGET_ENCODE": ("encoding", "TARGET_ENCODE"), "FREQUENCY_ENCODE": ("encoding", "FREQUENCY_ENCODE"),
                "ORDINAL": ("encoding", "ORDINAL"), "BINARY_MAP": ("encoding", "BINARY_MAP"),
                "DATE_FEATURES": ("dates", "DATE_FEATURES"), "TFIDF_SVD": ("text", "TFIDF_SVD"),
                "TFIDF_NGRAMS": ("text", "TFIDF_NGRAMS"), "COUNT_VECTORIZER": ("text", "COUNT_VECTORIZER"), "HASH_TEXT": ("text", "HASH_TEXT"),
            }
            alternatives = [candidate for candidate in alternatives
                            if candidate.value in action_steps and
                            _action_supported(action_steps[candidate.value][0], groups,
                                              action_steps[candidate.value][1])]
            score = leakage.get(clean_name, np.nan)
            if clean_name in prep.suspected_leakage_cols_ and self.config.drop_leaky_columns:
                action, confidence = Action.DROP, "haute"
                reason = f"Fuite de cible suspectée (score {score:.3f}); suppression recommandée."
                step_recs["type"]["action"] = "DROP"
                step_recs["type"]["measure"] = f"score fuite CV : {score:.3f}"
                step_recs["type"]["threshold"] = f"score ≥ {self.config.leakage_threshold:.3f} et écart ≥ {self.config.leakage_min_gap:.3f}"
                step_recs["type"]["explanation"] = reason
            elif clean_name in prep.suspected_leakage_cols_:
                # Le défaut historique est un avertissement, pas une suppression.
                # La recommandation doit refléter cette configuration pour que
                # « Tout accepter » reproduise fit_transform sans override.
                reason = (f"Fuite de cible possible (score {score:.3f}); la suppression automatique "
                          "est désactivée dans la configuration, la colonne est conservée.")
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
            if pd.notna(missing_target_p) and missing_target_p < self.config.target_relation_p_threshold:
                step_recs["missing"]["action"] = "MISSING_INDICATOR"
                missing_p_text = "p < 0.001" if missing_target_p < .001 else f"p = {missing_target_p:.3g}"
                step_recs["missing"]["measure"] += f"; {missing_p_text} pour le lien avec la cible"
                step_recs["missing"]["threshold"] = f"p < {self.config.target_relation_p_threshold:g}"
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
            configured_bounds = (self.config.value_bounds or {}).get(str(col))
            impossible_count = int(getattr(prep, "impossible_counts_", {}).get(str(col), 0))
            cleanup_total += impossible_count
            step_recs["nettoyage"]["measure"] = (
                f"{false_missing} faux manquants; {variant_count} variantes; {numeric_text_count} nombres texte; "
                f"{date_format_count} formats de date; {impossible_count} valeurs hors bornes configurées; "
                f"{cleanup_total} cellules candidates"
            )
            step_recs["nettoyage"]["threshold"] = (
                f"nombres texte ≥ {self.config.numeric_string_threshold:.0%}; faux manquants connus; variantes normalisées"
            )
            step_recs["nettoyage"]["explanation"] = (
                "Les corrections automatiques normalisent les marqueurs de manque, les modalités et les formats numériques ou calendaires reconnus. "
                "Les bornes métier ne sont vérifiées que lorsqu'elles sont explicitement configurées."
            )
            sample_n = len(nonnull)
            if sample_n < 30 and confidence != "haute":
                confidence = "faible"
                for detail in step_recs.values():
                    detail["confidence"] = "faible"
                    detail["explanation"] += " Échantillon inférieur à 30 valeurs : interprétation prudente."
            for step_name, detail in step_recs.items():
                detail["rule"] = {
                    "name": step_name,
                    "measure": detail.get("measure", ""),
                    "threshold": detail.get("threshold", ""),
                    "domain": kind_name,
                    "action": detail.get("action", "AUTO"),
                }
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
            missing_diag = diagnose_missingness(s, missing_pool, self.config, name=str(clean_name))
            rows.append({"colonne": str(col), "type": kind_name, "dtype": str(df[col].dtype),
                         "diagnostic_manquants": missing_diag,
                         "diagnostic_manquants_texte": (missing_diag["texte"] + (" " + missing_diag["implication"] if missing_diag["implication"] else "")
                                                        + " " + MNAR_NOTE) if missing_diag["verdict"] != "AUCUN" else "",
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
                         "manquant_informatif": bool(pd.notna(missing_target_p) and missing_target_p < self.config.target_relation_p_threshold),
                         "est_cible": False,
                         **stats_row})
        # Paires de features fortement corrélées; ordre d'entrée = règle explicite du préprocesseur.
        numeric_df = clean.select_dtypes(include=[np.number]).drop(
            columns=[target] + list(prep.columns_to_drop_) if target else list(prep.columns_to_drop_),
            errors="ignore")
        correlation_pairs = []
        projected_feature_count = max(numeric_df.shape[1], len(clean.columns))
        if numeric_df.shape[1] > 1 and projected_feature_count >= max(2, self.config.min_features_for_collinearity):
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
                        "Une variable redondante est écartée selon l'ordre stable des colonnes; le modèle garde la première représentation disponible."
                    )
                    if col == pair["droite"]:
                        selected = row_by_name[col]
                        selected["action"] = Action.DROP.value
                        selected["recommendation_step"] = "selection"
                        rec_sel["action"] = "DROP"
                        rec_sel["rule"] = {
                            "name": "selection_colinearite",
                            "measure": rec_sel["measure"],
                            "threshold": rec_sel["threshold"],
                            "domain": selected.get("type", ""),
                            "action": "DROP",
                        }
                        if "DROP" not in selected["alternatives"]:
                            selected["alternatives"].append("DROP")
        for row in rows:
            row["paires_correlees"] = [p for p in correlation_pairs if row["colonne"] in (p["gauche"], p["droite"])]
            if row.get("est_cible"):
                row["alerte_identifiant"] = False
                row["ratio_uniques_lignes"] = np.nan
                continue
            unique_ratio = row["uniques"] / max(len(raw), 1)
            row["ratio_uniques_lignes"] = unique_ratio
            row["seuil_identifiant_uniques"] = self.config.identifier_unique_threshold
            if row["type"] == FeatureType.DROP_ID.value:
                row["drop_preuve"] = f"uniques = {row['uniques']} / {len(raw)} lignes ({unique_ratio:.1%})"
                row["drop_seuil"] = f"ratio uniques/lignes > {self.config.identifier_unique_threshold:.1%}"
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
                    rule={"name": row["type"], "measure": row["drop_preuve"],
                          "threshold": row["drop_seuil"], "domain": row["type"],
                          "action": "DROP"},
                    explanation="La suppression est recommandée par la règle de détection correspondante.")
            for recommendation in row.get("steps", {}).values():
                recommendation["evidence"] = format_step_evidence(
                    recommendation.get("measure", ""), recommendation.get("threshold", ""))
            primary_step = row.get("recommendation_step") or recommendation_step_name(row.get("action", "AUTO"))
            primary_recommendation = row.get("steps", {}).get(primary_step, {})
            if primary_recommendation:
                primary_recommendation["reason"] = format_step_reason(
                    row.get("action", "AUTO"), primary_recommendation.get("measure", ""),
                    primary_recommendation.get("threshold", ""))
                row["raison"] = primary_recommendation["reason"]
            row["alerte_identifiant"] = row["type"] == FeatureType.DROP_ID.value
            row["relation_cible_non_fiable_identifiant"] = bool(row.get("relation_cible_non_fiable_identifiant") or row["alerte_identifiant"])
            if row["alerte_identifiant"]:
                row["relation_cible"] = np.nan
                row["spearman_cible"] = row["cramer_cible"] = row["information_mutuelle_cible"] = np.nan
                row["relation_cible_methode"] = "non fiable (identifiant)"
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
        self.auto_imputation_ = {}
        self.column_override_params_ = {}
        self.categorical_imputers_ = {}
        self.frequency_override_cols_ = []
        self.custom_encoding_cols_ = {}
        self.custom_encoders_ = {}
        self._loo_train_ = {}
        self.ngram_text_cols_ = {}
        self.ngram_text_transformer_ = None
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
                stage = _SINGLE_ACTION_STAGE.get(action, "action")
                normalized_action = _SINGLE_ACTION_VALUE.get(action, action)
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
            if dist in {"YEO_JOHNSON", "LOG1P", "BOX_COX", "SQRT", "CBRT", "QUANTILE", "RANK"} and not kind.startswith("NUMERIC"):
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
            elif encoding in {"COUNT_ENCODE", "BINARY_ENCODE", "HASH_ENCODE", "LOO_ENCODE"}:
                if col in self.low_card_cols_:
                    self.low_card_cols_.remove(col)
                elif col in self.high_card_cols_:
                    self.high_card_cols_.remove(col)
                else:
                    raise ValueError(f"{encoding} incompatible avec {col!r} : colonne catégorielle requise.")
                self.custom_encoding_cols_[col] = encoding
            if encoding == "LOO_ENCODE":
                if y_arr is None:
                    raise ValueError(f"LOO_ENCODE sur {col!r} exige une cible; choisissez COUNT_ENCODE ou FREQUENCY_ENCODE.")
                if self.task_resolved_ == "classification" and int(self.target_stats_.get("n_classes", 2)) > 2:
                    raise ValueError(f"LOO_ENCODE sur {col!r} exige une cible binaire ou continue "
                                     f"(la cible a {self.target_stats_.get('n_classes')} classes).")
            if encoding == "FREQUENCY_ENCODE" and y_arr is not None:
                raise ValueError(f"FREQUENCY_ENCODE sur {col!r} n'est disponible que sans cible; avec cible, utilisez TARGET_ENCODE.")
            if encoding == "TARGET_ENCODE" and y_arr is None:
                raise ValueError(f"TARGET_ENCODE sur {col!r} exige une cible; choisissez FREQUENCY_ENCODE.")
            if encoding:
                self._add_override_note(col, f"choix utilisateur : {encoding}")
            if dist == "BOX_COX" and pd.to_numeric(X_clean[col], errors="coerce").dropna().le(0).any():
                raise ValueError(f"Box-Cox impossible sur {col!r} : les valeurs doivent être strictement positives.")
            if dist in {"YEO_JOHNSON", "LOG1P", "BOX_COX", "SQRT", "CBRT", "QUANTILE", "RANK", "NONE"}:
                self.column_distributions_[col] = dist
                self._add_override_note(col, f"choix utilisateur : {dist}")
            text_action = choices.get("text", "").upper()
            if text_action in NgramTextTransformer.METHODS:
                if col not in self.text_cols_:
                    raise ValueError(f"{text_action} incompatible avec {col!r} (type {kind}) : colonne de texte libre requise "
                                     "(utilisez l'étape « type » = FORCE_TEXT pour traiter la colonne comme du texte).")
                self.text_cols_.remove(col)
                self.ngram_text_cols_[col] = text_action
            missing_action = choices.get("missing", "").upper()
            if missing_action in {"MEDIAN", "MEAN", "MODE", "CONSTANT", "KNN", "ITERATIVE", "MISSING_INDICATOR", "MISSING_CATEGORY"}:
                self.column_missing_[col] = missing_action
                self._add_override_note(col, f"choix utilisateur : {missing_action} (valeurs manquantes)")
            outlier_action = choices.get("outliers", "").upper()
            if outlier_action in {"NONE", "WINSORIZE", "IQR", "ZSCORE", "MODIFIED_ZSCORE", "REPLACE_MEDIAN", "OUTLIER_TO_NAN"}:
                self.column_outliers_[col] = outlier_action
                self._add_override_note(col, f"choix utilisateur : {outlier_action} (valeurs extrêmes)")
            scale = choices.get("scaling", action).upper()
            if scale in {"STANDARD", "ROBUST", "MINMAX", "MAXABS", "NONE", "SCALE_STANDARD", "SCALE_ROBUST", "SCALE_MINMAX", "NO_SCALE"}:
                self.column_scalers_[col] = scale.replace("SCALE_", "").replace("NO_SCALE", "NONE")
                self._add_override_note(col, f"choix utilisateur : {scale}")
            already_noted = {"action", "distribution", "scaling", "missing", "outliers"}
            for step, val in choices.items():
                if step in already_noted or not isinstance(val, str):
                    continue
                if val.upper() not in {"AUTO", "NONE"}:
                    self._add_override_note(col, f"choix utilisateur : {val.upper()} (étape {step})")

        # Choix statistiques par défaut (mêmes règles que analyze) pour les colonnes sans override explicite.
        self.auto_imputation_ = {}
        if self.config.adaptive_recommendations:
            for col in self.num_cols_:
                if col not in X_clean.columns:
                    continue
                if col not in self.column_outliers_:
                    self.column_outliers_[col] = _recommend_outliers(X_clean[col], self.config)[0]
                self.auto_imputation_[col] = _recommend_imputation(
                    X_clean[col], self.config, X_clean[[c for c in self.num_cols_ if c != col and c in X_clean.columns]])[0]
                if col not in self.column_missing_:
                    self.column_missing_[col] = self.auto_imputation_[col]

        # Un scaler statistique commun à analyze et fit: les seuils sont réestimés sur le train.
        for col in self.num_cols_:
            if col not in self.column_scalers_:
                mode = (_recommend_scaler(X_clean[col], self.config)[0] if self.config.scaler_type == "standard"
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
                    continue
                params = self.column_override_params_.get(col, {}).get("outliers", {})
                low, high = _outlier_bounds(vals, mode, self.config, params)
                self.winsorizer_.lower_bounds_[col] = low
                self.winsorizer_.upper_bounds_[col] = high
                self.winsorizer_.replace_values_.pop(col, None)
                if mode == "REPLACE_MEDIAN":
                    self.winsorizer_.replace_values_[col] = float(vals.median())
                elif mode == "OUTLIER_TO_NAN":
                    self.winsorizer_.replace_values_[col] = np.nan
            X_num_wins = self.winsorizer_.transform(X_num)

            # Imputation médiane ou moyenne
            self.num_imputer_ = {}
            for col in self.num_cols_:
                mode = self.column_missing_.get(col, self.config.imputation_strategy.upper())
                if mode == "MISSING_INDICATOR":
                    mode = self.auto_imputation_.get(col, self.config.imputation_strategy.upper())
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

            self._fit_multivariate_imputers(X_num_wins)
            X_num_imp = self._impute_numeric(X_num_wins)

            # PowerTransformer pour asymétrie
            self.num_power_transformer_ = {}
            self.num_skewed_cols_ = []
            if self.config.adaptive_recommendations:
                for col in self.num_cols_:
                    if col not in self.column_distributions_:
                        self.column_distributions_[col] = _recommend_distribution(X_num_wins[col], self.config)[0]
            for col in self.num_cols_:
                sk = float(stats.skew(X_num_imp[col].dropna()))
                override_dist = self.column_distributions_.get(col)
                if override_dist in {"QUANTILE", "RANK"}:
                    params = self.column_override_params_.get(col, {}).get("distribution", {})
                    n_q = int(params.get("n_quantiles", 1000)) if isinstance(params, dict) else 1000
                    if n_q < 2:
                        raise ValueError(f"n_quantiles doit être ≥ 2 pour {col!r}.")
                    qt = QuantileTransformer(n_quantiles=min(n_q, max(len(X_num_imp), 2)),
                                             output_distribution="normal" if override_dist == "QUANTILE" else "uniform",
                                             random_state=self.config.random_state).fit(X_num_imp[[col]])
                    self.num_power_transformer_[col] = qt
                    X_num_imp[col] = qt.transform(X_num_imp[[col]]).ravel()
                elif override_dist == "CBRT":
                    self.num_power_transformer_[col] = "CBRT"
                    X_num_imp[col] = np.cbrt(X_num_imp[col])
                elif override_dist == "YEO_JOHNSON" or (override_dist not in {"NONE", "LOG1P", "SQRT", "BOX_COX"} and abs(sk) > self.config.skew_threshold):
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

        # Encodages Count / Binary / Hash : état appris sur le train uniquement, aucune cible utilisée.
        for col, method in self.custom_encoding_cols_.items():
            if col not in X_work.columns:
                continue
            values = X_work[col].astype(str)
            params = self.column_override_params_.get(col, {}).get("encoding", {})
            params = params if isinstance(params, dict) else {}
            if method == "COUNT_ENCODE":
                self.custom_encoders_[col] = {"method": method, "counts": values.value_counts().to_dict()}
            elif method == "BINARY_ENCODE":
                categories = list(values.value_counts().sort_index(kind="stable").sort_values(ascending=False, kind="stable").index)
                n_bits = max(1, int(np.ceil(np.log2(len(categories) + 1))))
                self.custom_encoders_[col] = {"method": method, "codes": {c: i + 1 for i, c in enumerate(categories)}, "n_bits": n_bits}
            elif method == "LOO_ENCODE":
                sigma = float(params.get("sigma", 0.0))
                if sigma < 0:
                    raise ValueError(f"sigma doit être ≥ 0 pour {col!r}.")
                y_num = np.asarray(y_arr, dtype=float)
                if len(y_num) != len(values):
                    raise ValueError(f"LOO_ENCODE sur {col!r} : cible et lignes non alignées ({len(y_num)} vs {len(values)}).")
                frame = pd.DataFrame({"v": values.to_numpy(), "y": y_num})
                grouped = frame.groupby("v")["y"].agg(["sum", "count"])
                global_mean = float(y_num.mean())
                state = {"method": method, "sums": grouped["sum"].to_dict(), "counts": grouped["count"].to_dict(),
                         "global_mean": global_mean, "sigma": sigma}
                self.custom_encoders_[col] = state
                self._loo_train_[col] = self._loo_train_values(frame["v"], y_num, state)
            else:
                n_components = int(params.get("n_components", 8))
                if n_components < 1:
                    raise ValueError(f"n_components doit être ≥ 1 pour {col!r}.")
                self.custom_encoders_[col] = {"method": method, "n_components": n_components}

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

        if self.ngram_text_cols_:
            ngram_params = {c: (self.column_override_params_.get(c, {}).get("text") or {}) for c in self.ngram_text_cols_}
            self.ngram_text_transformer_ = NgramTextTransformer(
                methods=dict(self.ngram_text_cols_), params={c: p for c, p in ngram_params.items() if isinstance(p, dict)},
                default_max_features=self.config.max_text_features)
            self.ngram_text_transformer_.fit(X_work[list(self.ngram_text_cols_)])

        # Post-traitement sur le train : Sélection des colonnes, variance, corrélation et fuites
        X_trans_train = self._transform_components(X_clean)
        self._fit_post_processing(X_trans_train, y_arr)

        self.is_fitted_ = True
        logger.info(f"AutoPreprocessor ajusté avec succès. Nb features finales : {len(self.final_feature_names_)}")
        return self

    def _fit_multivariate_imputers(self, X_num_wins: pd.DataFrame) -> None:
        """Ajuste KNN / imputation itérative sur le train pour les colonnes qui les demandent.

        Les entrées sont les colonnes numériques après bornage, centrées-réduites sur le train (les distances KNN
        dépendent de l'échelle). Seules les cellules manquantes des colonnes demandées sont remplacées.
        """
        self.multivariate_imputers_ = {}
        for method in ("KNN", "ITERATIVE"):
            cols = [c for c in self.num_cols_ if self.column_missing_.get(c) == method and c in X_num_wins.columns]
            if not cols:
                continue
            data = X_num_wins[self.num_cols_].astype(float)
            mu = data.mean().fillna(0.0).to_numpy()
            sd = data.std(ddof=0).replace(0, 1.0).fillna(1.0).to_numpy()
            try:
                if method == "KNN":
                    k = 5
                    for col in cols:
                        params = self.column_override_params_.get(col, {}).get("missing", {})
                        if isinstance(params, dict) and "n_neighbors" in params:
                            k = int(params["n_neighbors"])
                            break
                    if k < 1:
                        raise ValueError("n_neighbors doit être ≥ 1.")
                    imputer = KNNImputer(n_neighbors=k, keep_empty_features=True)
                else:
                    imputer = IterativeImputer(max_iter=10, random_state=self.config.random_state,
                                               keep_empty_features=True)
                imputer.fit((data.to_numpy() - mu) / sd)
            except ValueError:
                raise
            except Exception as exc:  # échec numérique : retombe sur l'imputation univariée (médiane)
                self.warnings_.append(f"Imputation {method} indisponible pour {cols} ({exc}); médiane utilisée.")
                continue
            self.multivariate_imputers_[method] = (imputer, cols, mu, sd)

    def _impute_numeric(self, X_num_wins: pd.DataFrame) -> pd.DataFrame:
        """Impute les numériques : multivarié pour les colonnes KNN/ITERATIVE, valeur apprise sinon."""
        out = X_num_wins.copy()
        for method, (imputer, cols, mu, sd) in (self.multivariate_imputers_ or {}).items():
            arr = (X_num_wins[self.num_cols_].astype(float).to_numpy() - mu) / sd
            filled = pd.DataFrame(imputer.transform(arr) * sd + mu, index=X_num_wins.index, columns=self.num_cols_)
            for col in cols:
                out[col] = X_num_wins[col].where(X_num_wins[col].notna(), filled[col])
        return out.fillna(self.num_imputer_)

    def _loo_train_values(self, values: pd.Series, y: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """Valeurs Leave-One-Out du train : moyenne de la cible des AUTRES lignes de la même modalité.

        Une modalité vue une seule fois reçoit la moyenne globale. ``sigma`` > 0 ajoute un bruit gaussien
        (écart-type = sigma × écart-type de la cible) au train uniquement, pour limiter la fuite que le
        Leave-One-Out crée sur le train (la valeur encodée dépend de y_i par construction).
        """
        sums = values.map(state["sums"]).to_numpy(dtype=float)
        counts = values.map(state["counts"]).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            loo = np.where(counts > 1, (sums - y) / (counts - 1), state["global_mean"])
        if state.get("sigma", 0.0) > 0:
            rng = np.random.RandomState(self.config.random_state)
            loo = loo + rng.normal(0.0, state["sigma"] * (float(np.std(y)) or 1.0), size=len(loo))
        return loo.astype(float)

    @staticmethod
    def _apply_custom_encoder(col: str, state: Dict[str, Any], X_clean: pd.DataFrame) -> pd.DataFrame:
        """Applique un encodage Count / Binary / Hash appris sur le train (modalités inconnues : 0)."""
        values = X_clean[col].astype(str) if col in X_clean.columns else pd.Series(["__MISSING__"] * len(X_clean), index=X_clean.index)
        if state["method"] == "COUNT_ENCODE":
            return pd.DataFrame({f"{col}_count": values.map(state["counts"]).fillna(0).astype(float)}, index=X_clean.index)
        if state["method"] == "LOO_ENCODE":
            sums = values.map(state["sums"]).to_numpy(dtype=float)
            counts = values.map(state["counts"]).to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                enc = np.where(counts > 0, sums / counts, state["global_mean"])
            return pd.DataFrame({f"{col}_loo": enc}, index=X_clean.index)
        if state["method"] == "BINARY_ENCODE":
            codes = values.map(state["codes"]).fillna(0).astype(int).to_numpy()
            bits = {f"{col}_bin_{i}": ((codes >> i) & 1).astype(float) for i in range(state["n_bits"])}
            return pd.DataFrame(bits, index=X_clean.index)
        hasher = FeatureHasher(n_features=state["n_components"], input_type="string", alternate_sign=False)
        matrix = hasher.transform([[f"{col}={v}"] for v in values]).toarray()
        return pd.DataFrame(matrix, index=X_clean.index, columns=[f"{col}_hash_{i}" for i in range(state["n_components"])])

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
            X_num_imp = self._impute_numeric(X_num_wins)

            # PowerTransformer
            if self.num_power_transformer_:
                for c, pt in self.num_power_transformer_.items():
                    if c in X_num_imp.columns:
                        if isinstance(pt, str) and pt == "LOG1P":
                            X_num_imp[c] = np.log1p(X_num_imp[c])
                        elif isinstance(pt, str) and pt == "SQRT":
                            X_num_imp[c] = np.sqrt(X_num_imp[c].clip(lower=0))
                        elif isinstance(pt, str) and pt == "CBRT":
                            X_num_imp[c] = np.cbrt(X_num_imp[c])
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

        for col, state in (self.custom_encoders_ or {}).items():
            parts.append(self._apply_custom_encoder(col, state, X_clean))

        # 6. Dates
        if self.date_cols_ and self.datetime_extractor_ is not None:
            df_dt = self.datetime_extractor_.transform(X_clean[self.date_cols_])
            parts.append(df_dt)

        # 7. Texte
        if self.text_cols_ and self.text_transformer_ is not None:
            df_txt = self.text_transformer_.transform(X_clean[self.text_cols_])
            parts.append(df_txt)
        if self.ngram_text_cols_ and self.ngram_text_transformer_ is not None:
            parts.append(self.ngram_text_transformer_.transform(X_clean))

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
                        # Une règle stable indépendante de la cible: garder la première
                        # représentation selon l'ordre d'entrée et retirer la suivante.
                        drop_col = c2
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
        for col, loo_values in (self._loo_train_ or {}).items():
            name = f"{col}_loo"
            if name in X_out.columns and len(X_out) == len(loo_values):
                X_out[name] = np.nan_to_num(loo_values).astype(np.float32)
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
        analyzed = {str(row.get("colonne")): row for row in
                    getattr(getattr(self, "analysis_result_", None), "columns", [])}

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
            recommendation = analyzed.get(str(col), {})
            primary_step = (recommendation.get("recommendation_step") or
                            recommendation_step_name(recommendation.get("action", "AUTO"))) if recommendation else ""
            recommendation_detail = recommendation.get("steps", {}).get(primary_step, {})

            columns_summary.append({
                "Colonne": col,
                "Type détecté": meta["type"].value if isinstance(meta["type"], FeatureType) else str(meta["type"]),
                "Manquants (%)": f"{meta['missing_rate']:.1%}",
                "Cardinalité": meta["n_unique"],
                "Justification": meta["reason"],
                "Statut": status,
                "Nb features créées": len(created_features),
                "Choix utilisateur": self.user_override_notes_.get(col, "recommandation suivie"),
                "Action appliquée": self._reported_action(col, meta),
                "Raison recommandée": recommendation.get("raison", ""),
                "Preuve recommandée": recommendation_detail.get("evidence", ""),
                "Explication recommandée": recommendation_detail.get("explanation", ""),
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

    def _reported_action(self, col: str, meta: Dict[str, Any]) -> str:
        """Retourne l'action principale effectivement appliquée à une colonne."""
        if col in self.columns_to_drop_ or col in self.dropped_variance_cols_ or col in self.dropped_collinear_cols_:
            return "DROP"
        preferred_steps = ("type", "distribution", "outliers", "scaling", "encoding", "dates", "text")
        for step in preferred_steps:
            action, _ = self._override_value(col, step)
            if action != "AUTO":
                return action
        kind = meta["type"].value if isinstance(meta["type"], FeatureType) else str(meta["type"])
        if kind == "DATETIME":
            return "DATE_FEATURES"
        if kind == "TEXT":
            return "TFIDF_SVD"
        if kind == "BINARY":
            return "BINARY_MAP"
        if kind in {"CATEGORICAL_LOW", "CATEGORICAL_LOW_CARD"}:
            return "ONEHOT"
        if kind in {"CATEGORICAL_HIGH", "CATEGORICAL_HIGH_CARD"}:
            if self.target_stats_ and self.target_stats_.get("present"):
                return "TARGET_ENCODE"
            return "FREQUENCY_ENCODE"
        if col in self.column_distributions_:
            chosen = self.column_distributions_[col]
            return "KEEP_NUMERIC" if chosen == "NONE" else chosen
        if col in self.num_skewed_cols_:
            return "YEO_JOHNSON"
        return "KEEP_NUMERIC"

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


# =============================================================================
# APERÇU AVANT / APRÈS D'UNE COLONNE
# =============================================================================

PREVIEW_MODEL_NOTE = (
    "Rendre une distribution plus régulière n'améliore pas automatiquement les performances : les arbres et "
    "forêts sont insensibles aux transformations monotones, alors que les modèles linéaires, KNN, SVM et "
    "réseaux de neurones y sont plus sensibles. Seule une évaluation par validation croisée tranche."
)
PREVIEW_FIT_NOTE = (
    "Aperçu descriptif ajusté sur toutes les lignes fournies; le prétraitement final est ajusté sur le jeu "
    "d'entraînement uniquement."
)


def _series_profile(values: Any, iqr_multiplier: float = 1.5) -> Dict[str, float]:
    """Statistiques descriptives d'une série numérique (valeurs manquantes comptées, jamais inventées)."""
    raw = pd.Series(values)
    x = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
    obs = x.dropna().astype(float)
    profile: Dict[str, float] = {"count": int(len(x)), "missing_pct": float(x.isna().mean() * 100) if len(x) else 0.0}
    nan = float("nan")
    if obs.empty:
        profile.update(mean=nan, median=nan, std=nan, skew=nan, kurtosis=nan, min=nan, max=nan,
                       outlier_pct=nan, zero_pct=nan, normality_p=nan)
        return profile
    q1, q3 = obs.quantile(.25), obs.quantile(.75)
    spread = q3 - q1
    profile.update(
        mean=float(obs.mean()), median=float(obs.median()), std=float(obs.std(ddof=1)) if len(obs) > 1 else nan,
        skew=float(stats.skew(obs)) if len(obs) > 2 else nan, kurtosis=float(obs.kurt()) if len(obs) > 3 else nan,
        min=float(obs.min()), max=float(obs.max()),
        outlier_pct=float(((obs < q1 - iqr_multiplier * spread) | (obs > q3 + iqr_multiplier * spread)).mean() * 100),
        zero_pct=float((obs == 0).mean() * 100), normality_p=nan)
    if len(obs) >= 8 and obs.nunique() > 1:
        try:
            profile["normality_p"] = float(stats.normaltest(obs.sample(min(len(obs), 5000), random_state=0)).pvalue)
        except Exception:
            pass
    return profile


def _override_as_choices(override: Any) -> Dict[str, Any]:
    """Ramène un override (texte, Action, dict, ColumnOverride) à la forme {étape: choix} utilisée par fit()."""
    if isinstance(override, ColumnOverride):
        override = override.actions
    if isinstance(override, dict):
        return {str(key).lower(): value for key, value in override.items()}
    if override is None or override == "":
        return {}
    action = override.value if isinstance(override, Action) else str(override)
    return {_SINGLE_ACTION_STAGE.get(action, "action"): _SINGLE_ACTION_VALUE.get(action, action)}


def preview_column(df: pd.DataFrame, column: str, override: Any = None, config: Optional[PreprocessConfig] = None,
                   target: Optional[str] = None, task: str = "auto", max_rows: int = 50_000) -> Dict[str, Any]:
    """Compare une colonne avant et après le traitement choisi, sans modifier ``df``.

    ``override`` est le choix utilisateur de la colonne (même forme que ``column_overrides[colonne]``); sans
    override, la recommandation automatique est appliquée. Pour une colonne numérique, ``after`` décrit la
    colonne après nettoyage, imputation, extrêmes et distribution (avant mise à l'échelle) et ``after_scaled``
    après la mise à l'échelle choisie. Le résultat contient aussi le nombre de lignes concernées, les colonnes de
    sortie et les séries échantillonnées (``data``) pour tracer histogrammes, boîtes et fréquences.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if column not in df.columns:
        raise ValueError(f"Colonne inconnue : {column!r}.")
    if target is not None and target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' est absente du DataFrame.")
    if target == column:
        raise ValueError("L'aperçu ne s'applique pas à la colonne cible.")
    base = copy.deepcopy(config) if config is not None else PreprocessConfig(verbose=0)
    base.noise_guard, base.verbose = "off", 0
    frame = df.copy(deep=True)
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=base.random_state).sort_index()
    y = frame[target] if target else None
    choices = _override_as_choices(override)
    needs_others = any(str(v[0] if isinstance(v, (tuple, list)) else v).upper() in {"KNN", "ITERATIVE"}
                       for v in choices.values())

    def sub_frame(prep_probe: Optional[AutoPreprocessor] = None) -> pd.DataFrame:
        if not needs_others:
            return frame[[column]]
        others = [c for c in frame.columns if c != column and c != target and pd.api.types.is_numeric_dtype(frame[c])]
        return frame[[column] + others]

    def run(extra: Optional[Dict[str, Any]] = None) -> Tuple[AutoPreprocessor, pd.DataFrame]:
        cfg = copy.deepcopy(base)
        merged = dict(choices)
        merged.update(extra or {})
        merged.setdefault("duplicates", "KEEP")   # une colonne isolée a des lignes identiques : ne rien retirer
        cfg.column_overrides = {column: merged} if merged else {}
        prep = AutoPreprocessor(cfg, task=task)
        out = prep.fit_transform(sub_frame(), y)
        return prep, out

    prep, out = run()
    meta = prep.column_metadata_.get(column, {})
    meta_type = meta.get("type")
    kind = meta_type.value if isinstance(meta_type, FeatureType) else str(meta_type)
    aux_config = copy.deepcopy(base)
    aux_config.column_overrides = {column: {"duplicates": "KEEP"}}
    aux = AutoPreprocessor(aux_config, task=task)
    cleaned = aux._preprocess_raw_dataframe(frame[[column]], is_fit=True)
    before_series = cleaned[column] if column in cleaned.columns else frame[column]
    outputs = [c for c in out.columns if c == column or str(c).startswith(f"{column}_")]
    rng = np.random.default_rng(base.random_state)

    def sample(values: Any, limit: int = 5000) -> np.ndarray:
        arr = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
        return arr if len(arr) <= limit else rng.choice(arr, limit, replace=False)

    notes: List[str] = [PREVIEW_FIT_NOTE]
    result: Dict[str, Any] = {"column": column, "kind": kind, "rows": int(len(frame)), "output_columns": outputs,
                              "choices": {k: (v if not isinstance(v, (tuple, list)) else list(v)) for k, v in choices.items()},
                              "applied": {}, "data": {}, "notes": notes}
    if kind in {"DROP_ID", "DROP_CONSTANT", "DROP_HIGH_MISSING", "DROP_DUPLICATE", "DROP_LEAKAGE"} or column in prep.columns_to_drop_:
        result["status"] = "dropped"
        notes.append(f"La colonne est supprimée du jeu transformé ({meta.get('reason') or kind}).")
        return result
    if not outputs:
        result["status"] = "removed_after_selection"
        notes.append("Aucune colonne de sortie : la colonne est retirée par la sélection finale (variance, colinéarité).")
        return result
    result["status"] = "ok"

    if kind.startswith("NUMERIC") and column in out.columns:
        before_aligned = pd.to_numeric(before_series, errors="coerce").reindex(out.index)
        prep_raw, out_raw = run({"scaling": "NONE"})
        after_raw = out_raw[column] if column in out_raw.columns else out[column]
        lower = prep.winsorizer_.lower_bounds_.get(column, -np.inf) if prep.winsorizer_ is not None else -np.inf
        upper = prep.winsorizer_.upper_bounds_.get(column, np.inf) if prep.winsorizer_ is not None else np.inf
        observed = before_aligned.notna()
        outside = int((observed & ((before_aligned < lower) | (before_aligned > upper))).sum())
        changed = int((~np.isclose(before_aligned.fillna(np.nan).to_numpy(float), after_raw.to_numpy(float),
                                   rtol=1e-9, atol=1e-12, equal_nan=False)).sum())
        result.update(
            before=_series_profile(before_aligned, base.outlier_iqr_multiplier),
            after=_series_profile(after_raw, base.outlier_iqr_multiplier),
            after_scaled=_series_profile(out[column], base.outlier_iqr_multiplier),
            affected={"missing_filled": int((~observed).sum()), "outside_bounds": outside,
                      "rows_changed": changed, "rows_changed_pct": changed / max(len(out), 1) * 100},
            applied={"missing": prep.column_missing_.get(column), "outliers": prep.column_outliers_.get(column),
                     "distribution": prep.column_distributions_.get(column, "NONE"),
                     "scaling": prep.column_scalers_.get(column) if not hasattr(prep.column_scalers_.get(column), "fit")
                     else type(prep.column_scalers_[column]).__name__},
            data={"before": sample(before_aligned), "after": sample(after_raw), "after_scaled": sample(out[column])})
        dist = result["applied"]["distribution"]
        if dist in {"QUANTILE", "RANK"}:
            notes.append("QUANTILE et RANK ne conservent que l'ordre des valeurs : les distances entre valeurs sont perdues.")
        if result["applied"]["outliers"] in {"REPLACE_MEDIAN", "OUTLIER_TO_NAN"}:
            notes.append("Les extrêmes sont traités comme des erreurs : si ce sont des valeurs réelles, de l'information est perdue.")
        notes.append(PREVIEW_MODEL_NOTE)
    else:
        values = before_series.astype("string")
        counts = values.value_counts(dropna=True)
        probs = counts / max(int(counts.sum()), 1)
        result["before"] = {
            "count": int(len(values)), "missing_pct": float(values.isna().mean() * 100), "n_categories": int(len(counts)),
            "entropy": float(-(probs * np.log2(probs.clip(lower=1e-15))).sum()) if len(probs) else float("nan"),
            "top": [(str(k), float(v) * 100) for k, v in probs.head(10).items()]}
        result["after"] = {"n_columns": len(outputs),
                           "columns": {str(c): _series_profile(out[c], base.outlier_iqr_multiplier) for c in outputs[:10]}}
        result["data"] = {"before_frequencies": {k: v for k, v in result["before"]["top"]},
                          "after": {str(c): sample(out[c]) for c in outputs[:6]}}
        if len(outputs) > 10:
            notes.append(f"{len(outputs)} colonnes de sortie; les 10 premières sont détaillées.")
    return result