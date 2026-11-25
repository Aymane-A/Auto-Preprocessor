# Catalogue de parité du moteur

Catalogue généré depuis les schémas validés du backend le 11 novembre 2026. Chaque réglage est accessible dans Configuration → Réglages avancés. Les choix par étape proviennent du catalogue du moteur et sont filtrés selon la variable et la cible.

| Paramètre | Libellé dans les réglages | Défaut |
|---|---|---|
| `test_size_pct` | Part du test (%) | `20` |
| `scaler_type` | Mise à l’échelle globale | `"standard"` |
| `outlier_method` | Traitement global des extrêmes | `"winsorize"` |
| `cardinality_threshold` | Seuil de cardinalité | `10` |
| `missing_threshold` | Proportion de manquants avant suppression | `0.9` |
| `correlation_threshold` | Seuil de colinéarité absolue | `0.95` |
| `rare_category_threshold` | Fréquence des catégories rares | `0.01` |
| `skew_threshold` | Seuil d’asymétrie | `1.0` |
| `transform_target` | Transformer une cible continue asymétrique | `true` |
| `drop_leaky_columns` | Supprimer les fuites de cible suspectées | `false` |
| `encode_discrete_as_categorical` | Traiter les nombres discrets comme catégories | `false` |
| `random_state` | Graine reproductible | `42` |
| `sparse_output` | Représentation creuse intermédiaire | `false` |
| `model_family` | Famille du modèle visé | `null` |
| `imputation_strategy` | Imputation numérique globale | `"median"` |
| `adaptive_recommendations` | Recommandations adaptatives par variable | `true` |
| `outlier_lower_quantile` | Quantile inférieur de winsorisation | `0.01` |
| `outlier_upper_quantile` | Quantile supérieur de winsorisation | `0.99` |
| `outlier_iqr_multiplier` | Multiplicateur IQR | `1.5` |
| `outlier_mad_threshold` | Seuil MAD | `3.5` |
| `missing_indicator_threshold` | Seuil d’indicateur de manquants | `0.05` |
| `variance_threshold` | Variance minimale des sorties | `0.0` |
| `constant_threshold` | Proportion quasi constante | `0.99` |
| `identifier_unique_threshold` | Seuil d’unicité des identifiants | `0.9` |
| `min_features_for_collinearity` | Nombre minimal de sorties pour filtrer la colinéarité | `10` |
| `noise_guard` | Protection contre les variables de bruit | `"warn"` |
| `noise_margin` | Marge face à une cible permutée | `0.03` |
| `min_rows_for_noise_guard` | Effectif minimal pour supprimer le bruit | `300` |
| `keep_columns` | Variables protégées du filtre de bruit | `null` |
| `max_text_features` | Nombre maximal de termes texte | `15` |
| `text_svd_components` | Composantes SVD du texte | `4` |
| `clean_column_names` | Normaliser les noms des colonnes | `true` |
| `clean_string_values` | Nettoyer les chaînes et faux manquants | `true` |
| `parse_numeric_strings` | Convertir les nombres stockés en texte | `true` |
| `numeric_string_threshold` | Taux de conversion numérique requis | `0.9` |
| `downcast_numeric` | Réduire la précision pour économiser la mémoire | `true` |
| `supervised_encoding` | Autoriser les encodages supervisés sur le train | `true` |
| `target_skew_threshold` | Seuil d’asymétrie de la cible | `1.0` |
| `leakage_threshold` | Score d’alerte de fuite | `0.95` |
| `leakage_min_gap` | Écart minimal pour suspecter une fuite | `0.1` |
| `min_features_for_auto_drop` | Nombre minimal de variables avant suppression des fuites | `10` |
| `normality_p_threshold` | Seuil de normalité | `0.05` |
| `missingness_min_effect` | Effet minimal du mécanisme de manquants | `0.2` |
| `target_relation_p_threshold` | Seuil des relations à la cible | `0.05` |
| `date_parse_threshold` | Taux de dates reconnues requis | `0.8` |
| `scaler_zero_threshold` | Proportion de zéros pour conseiller l’échelle | `0.5` |
| `scaler_outlier_threshold` | Proportion d’extrêmes pour conseiller l’échelle | `0.05` |
| `scaler_kurtosis_threshold` | Kurtosis pour conseiller l’échelle | `3.0` |
| `scaler_bounded_min` | Borne inférieure du domaine normalisé | `0.0` |
| `scaler_bounded_max` | Borne supérieure du domaine normalisé | `1.0` |
| `value_bounds` | Bornes métier par variable | `{}` |
| `ordinal_orders` | Ordres catégoriels explicites | `null` |

## Opérations

| Étape | Actions conservées et testées |
|---|---|
| `clean_missing_tokens` | `KEEP`, `SELECTED_TO_NAN`, `TO_NAN` |
| `clean_variants` | `KEEP`, `MERGE` |
| `clean_numeric_text` | `AUTO`, `CONVERT`, `KEEP` |
| `clean_dates` | `AUTO`, `DROP_COLUMN`, `KEEP`, `PARSE` |
| `clean_impossible` | `AUTO`, `CLIP`, `DROP_ROWS`, `IGNORE`, `NAN` |
| `duplicates` | `DROP`, `KEEP` |
| `nettoyage` | `AUTO` |
| `type` | `AUTO`, `DROP`, `FORCE_CATEGORICAL`, `FORCE_DATE`, `FORCE_NUMERIC`, `FORCE_TEXT`, `KEEP_RAW` |
| `missing` | `AUTO`, `CONSTANT`, `ITERATIVE`, `KNN`, `MEAN`, `MEDIAN`, `MISSING_CATEGORY`, `MISSING_INDICATOR`, `MODE` |
| `outliers` | `AUTO`, `IQR`, `MEDIAN_REPLACE`, `MODIFIED_Z`, `MODIFIED_ZSCORE`, `NONE`, `OUTLIER_TO_NAN`, `REPLACE_MEDIAN`, `TO_MISSING`, `WINSORIZE`, `ZSCORE`, `Z_SCORE` |
| `distribution` | `AUTO`, `BOX_COX`, `CBRT`, `DISCRETIZE`, `LOG`, `LOG1P`, `NONE`, `QUANTILE`, `QUANTILE_NORMAL`, `QUANTILE_UNIFORM`, `RANK`, `SQRT`, `YEO_JOHNSON` |
| `scaling` | `AUTO`, `MAXABS`, `MINMAX`, `NONE`, `ROBUST`, `STANDARD` |
| `encoding` | `AUTO`, `BINARY_ENCODE`, `BINARY_MAP`, `COUNT_ENCODE`, `FREQUENCY_ENCODE`, `HASH_ENCODE`, `LOO_ENCODE`, `MANUAL_MAP`, `ONEHOT`, `ORDINAL`, `TARGET_ENCODE` |
| `dates` | `AUTO`, `DATE_FEATURES` |
| `text` | `AUTO`, `COUNT`, `COUNT_VECTORIZER`, `HASH`, `HASH_TEXT`, `TFIDF`, `TFIDF_NGRAM`, `TFIDF_NGRAMS`, `TFIDF_SVD` |
| `target` | `AUTO` |
| `selection` | `AUTO` |
