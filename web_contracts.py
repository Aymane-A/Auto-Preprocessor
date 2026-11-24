"""Public, bounded configuration shared by the web API and its capability catalog."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class WebOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    imbalance_method: Literal["none", "class_weight", "random_over", "smote", "smotenc", "random_under"] = "none"
    imbalance_ratio: float = Field(.5, gt=0, le=1)
    imbalance_neighbors: int = Field(5, ge=1, le=50)
    test_size_pct: int = Field(20, ge=1, le=50, title="Part du test (%)")
    scaler_type: Literal["standard", "robust", "minmax", "none"] = Field("standard", title="Mise à l’échelle globale")
    outlier_method: Literal["winsorize", "iqr", "none"] = Field("winsorize", title="Traitement global des extrêmes")
    cardinality_threshold: int = Field(10, ge=2, le=1000, title="Seuil de cardinalité")
    missing_threshold: float = Field(.9, gt=0, le=1, title="Proportion de manquants avant suppression")
    correlation_threshold: float = Field(.95, gt=0, le=1, title="Seuil de colinéarité absolue")
    rare_category_threshold: float = Field(.01, ge=0, lt=1, title="Fréquence des catégories rares")
    skew_threshold: float = Field(1., ge=0, le=100, title="Seuil d’asymétrie")
    transform_target: bool = Field(True, title="Transformer une cible continue asymétrique")
    drop_leaky_columns: bool = Field(False, title="Supprimer les fuites de cible suspectées")
    encode_discrete_as_categorical: bool = Field(False, title="Traiter les nombres discrets comme catégories")
    random_state: int = Field(42, ge=0, le=2**32-1, title="Graine reproductible")
    sparse_output: bool = Field(False, title="Représentation creuse intermédiaire")
    model_family: Literal["linear", "logistic", "decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "knn", "svm", "neural_network"] | None = Field(None, title="Famille du modèle visé")
    imputation_strategy: Literal["median", "mean"] = Field("median", title="Imputation numérique globale")
    adaptive_recommendations: bool = Field(True, title="Recommandations adaptatives par variable")
    outlier_lower_quantile: float = Field(.01, ge=0, lt=1, title="Quantile inférieur de winsorisation")
    outlier_upper_quantile: float = Field(.99, gt=0, le=1, title="Quantile supérieur de winsorisation")
    outlier_iqr_multiplier: float = Field(1.5, gt=0, le=100, title="Multiplicateur IQR")
    outlier_mad_threshold: float = Field(3.5, gt=0, le=100, title="Seuil MAD")
    missing_indicator_threshold: float = Field(.05, ge=0, le=1, title="Seuil d’indicateur de manquants")
    variance_threshold: float = Field(0., ge=0, le=1e12, title="Variance minimale des sorties")
    selection_method: Literal["auto", "variance", "correlation", "kbest", "chi2", "anova", "mutual_info", "percentile", "rfe", "rfecv", "forward", "backward", "l1", "elastic_net", "select_from_model", "random_forest"] = Field("auto", title="Méthode de sélection finale")
    selection_count: int | None = Field(None, ge=1, le=200, title="Nombre de features à sélectionner")
    selection_percentile: int = Field(50, ge=1, le=100, title="Pourcentage de features à sélectionner")
    selection_folds: int = Field(3, ge=2, le=5, title="Plis de validation du sélecteur")
    selection_scoring: Literal["accuracy", "f1", "roc_auc", "r2", "neg_mean_squared_error"] | None = Field(None, title="Métrique de validation du sélecteur")
    selection_include: list[str] | None = Field(None, max_length=200, title="Colonnes source à conserver ensemble")
    selection_exclude: list[str] | None = Field(None, max_length=200, title="Colonnes source à retirer ensemble")
    constant_threshold: float = Field(.99, gt=0, le=1, title="Proportion quasi constante")
    identifier_unique_threshold: float = Field(.9, gt=0, le=1, title="Seuil d’unicité des identifiants")
    min_features_for_collinearity: int = Field(10, ge=1, le=100000, title="Nombre minimal de sorties pour filtrer la colinéarité")
    noise_guard: Literal["off", "warn", "drop"] = Field("warn", title="Protection contre les variables de bruit")
    noise_margin: float = Field(.03, ge=0, le=1, title="Marge face à une cible permutée")
    min_rows_for_noise_guard: int = Field(300, ge=10, le=2000000, title="Effectif minimal pour supprimer le bruit")
    keep_columns: list[str] | None = Field(None, max_length=10000, title="Variables protégées du filtre de bruit")
    max_text_features: int = Field(15, ge=1, le=10000, title="Nombre maximal de termes texte")
    text_svd_components: int = Field(4, ge=0, le=1000, title="Composantes SVD du texte")
    clean_column_names: bool = Field(True, title="Normaliser les noms des colonnes")
    clean_string_values: bool = Field(True, title="Nettoyer les chaînes et faux manquants")
    parse_numeric_strings: bool = Field(True, title="Convertir les nombres stockés en texte")
    numeric_string_threshold: float = Field(.9, gt=0, le=1, title="Taux de conversion numérique requis")
    downcast_numeric: bool = Field(True, title="Réduire la précision pour économiser la mémoire")
    supervised_encoding: bool = Field(True, title="Autoriser les encodages supervisés sur le train")
    target_skew_threshold: float = Field(1., ge=0, le=100, title="Seuil d’asymétrie de la cible")
    leakage_threshold: float = Field(.95, gt=0, le=1, title="Score d’alerte de fuite")
    leakage_min_gap: float = Field(.1, ge=0, le=1, title="Écart minimal pour suspecter une fuite")
    min_features_for_auto_drop: int = Field(10, ge=1, le=100000, title="Nombre minimal de variables avant suppression des fuites")
    normality_p_threshold: float = Field(.05, gt=0, lt=1, title="Seuil de normalité")
    missingness_min_effect: float = Field(.2, ge=0, le=1, title="Effet minimal du mécanisme de manquants")
    target_relation_p_threshold: float = Field(.05, gt=0, lt=1, title="Seuil des relations à la cible")
    date_parse_threshold: float = Field(.8, gt=0, le=1, title="Taux de dates reconnues requis")
    scaler_zero_threshold: float = Field(.5, ge=0, le=1, title="Proportion de zéros pour conseiller l’échelle")
    scaler_outlier_threshold: float = Field(.05, ge=0, le=1, title="Proportion d’extrêmes pour conseiller l’échelle")
    scaler_kurtosis_threshold: float = Field(3., ge=0, le=1000, title="Kurtosis pour conseiller l’échelle")
    scaler_bounded_min: float = Field(0., title="Borne inférieure du domaine normalisé")
    scaler_bounded_max: float = Field(1., title="Borne supérieure du domaine normalisé")
    value_bounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict, title="Bornes métier par variable")
    ordinal_orders: dict[str, list[str]] | None = Field(None, title="Ordres catégoriels explicites")

    @model_validator(mode="after")
    def coherent_ranges(self):
        if self.outlier_lower_quantile >= self.outlier_upper_quantile or self.scaler_bounded_min >= self.scaler_bounded_max:
            raise ValueError("Les bornes inférieures doivent précéder les bornes supérieures.")
        if any(low is not None and high is not None and low >= high for low, high in self.value_bounds.values()):
            raise ValueError("Bornes métier incohérentes.")
        if any(len(order) < 2 or len(set(order)) != len(order) for order in (self.ordinal_orders or {}).values()):
            raise ValueError("Les ordres doivent comporter au moins deux modalités distinctes.")
        return self


class StepChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(min_length=1, max_length=64)
    parameters: str = Field("", max_length=16384)


Choice = str | dict[str, StepChoice]


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str


class AnalysisRequest(SessionRequest):
    options: WebOptions = Field(default_factory=WebOptions)
    choices: dict[str, Choice] = Field(default_factory=dict)


class Preview(SessionRequest):
    options: WebOptions = Field(default_factory=WebOptions)
    column: str
    choice: Choice = "recommended"


class Execute(AnalysisRequest):
    choices: dict[str, Choice] = Field(default_factory=dict)


class Evaluation(SessionRequest):
    options: WebOptions = Field(default_factory=WebOptions)
    family: Literal["linear", "logistic", "decision_tree", "random_forest", "gradient_boosting", "knn", "svm", "neural_network", "xgboost", "lightgbm"] = "linear"
    folds: int = Field(3, ge=2, le=5)
    time_column: str | None = None
    candidates: list[Literal["none", "standard", "robust", "minmax"]] = Field(default_factory=lambda: ["none", "standard", "robust"], min_length=1, max_length=4)


class NestedEvaluation(SessionRequest):
    options: WebOptions = Field(default_factory=WebOptions)
    family: Literal["linear", "logistic", "decision_tree", "random_forest", "gradient_boosting", "knn", "svm", "neural_network", "xgboost", "lightgbm"] = "linear"
    outer_folds: int = Field(3, ge=2, le=5)
    inner_folds: int = Field(3, ge=2, le=5)
    time_column: str | None = None
    group_column: str | None = None
    selection_counts: list[int] = Field(default_factory=list, max_length=4)
    scalers: list[Literal["none", "standard", "robust", "minmax"]] = Field(default_factory=list, max_length=4)
    model_params: list[dict[str, int | float | str | bool | None]] = Field(default_factory=lambda: [{}], min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_nested(self):
        if self.time_column is not None and self.group_column is not None:
            raise ValueError("Choisissez un découpage temporel ou groupé, pas les deux.")
        if any(value < 1 or value > 200 for value in self.selection_counts):
            raise ValueError("Les nombres de features doivent être entre 1 et 200.")
        count_methods = {"kbest", "chi2", "anova", "mutual_info", "rfe", "forward",
                         "backward", "l1", "elastic_net", "select_from_model", "random_forest"}
        if self.selection_counts and self.options.selection_method not in count_methods:
            raise ValueError("selection_counts n'agit pas sur la méthode de sélection choisie.")
        return self
