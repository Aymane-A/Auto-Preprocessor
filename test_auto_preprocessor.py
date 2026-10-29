"""
Suite de tests unitaires et d'intégration pour AutoPreprocessor.
Vérifie la robustesse, l'absence de fuite de données, la persistance et les cas limites.
"""

import os
import tempfile
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import pytest
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, FeatureType


@pytest.fixture
def synthetic_dataframe():
    """Génère un DataFrame synthétique hétérogène et complexe pour les tests."""
    np.random.seed(42)
    n_samples = 200

    # 1. Colonne ID (à supprimer)
    ids = [f"ID_{i:05d}" for i in range(n_samples)]

    # 2. Colonne constante (à supprimer)
    constant_col = ["FIXED_VALUE"] * n_samples

    # 3. Colonne avec 95% de valeurs manquantes (à supprimer)
    high_missing = [np.nan] * int(n_samples * 0.95) + list(np.random.randn(n_samples - int(n_samples * 0.95)))
    np.random.shuffle(high_missing)

    # 4. Colonne dupliquée
    dup_base = np.random.randn(n_samples)
    dup_copy = list(dup_base)

    # 5. Numérique continue normale avec quelques NaN
    num_norm = np.random.normal(50, 15, size=n_samples)
    num_norm[np.random.choice(n_samples, 20, replace=False)] = np.nan

    # 6. Numérique très asymétrique avec outliers extrêmes
    num_skewed = np.random.exponential(scale=10.0, size=n_samples)
    num_skewed[0] = 50000.0  # Outlier extrême
    num_skewed[1] = -10000.0  # Outlier extrême négatif

    # 7. Nombres stockés en texte avec faux manquants et devises
    price_candidates = [f"${x:,.2f}" for x in np.random.uniform(50, 5000, size=n_samples)]
    # Insérer faux manquants
    for idx in [5, 12, 45, 80]:
        price_candidates[idx] = "?"
    for idx in [20, 60, 99]:
        price_candidates[idx] = "N/A"

    # 8. Pourcentages avec virgule décimale stockés en texte
    pct_candidates = [f"{x:.1f}%".replace(".", ",") for x in np.random.uniform(0, 100, size=n_samples)]
    pct_candidates[15] = "unknown"
    pct_candidates[35] = "-"

    # 9. Colonne binaire ("oui" / "non")
    binary_choices = np.random.choice(["oui", "non"], size=n_samples, p=[0.6, 0.4])
    binary_choices[7] = "N/A"

    # 10. Colonne ordinale ("low", "medium", "high")
    ordinal_choices = np.random.choice(["low", "medium", "high"], size=n_samples, p=[0.3, 0.5, 0.2])

    # 11. Catégorielle basse cardinalité avec catégorie rare
    low_card_vals = np.random.choice(["Europe", "Asia", "Americas"], size=n_samples, p=[0.5, 0.4, 0.1])
    low_card_vals[0] = "Oceania"  # Catégorie rare (< 1%)

    # 12. Catégorielle haute cardinalité (villes)
    cities = [f"City_{np.random.randint(0, 40)}" for _ in range(n_samples)]

    # 13. Dates avec composante heure
    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D").astype(str).tolist()
    dates[10] = "2023-05-10 14:30:00"
    dates[30] = "invalid_date"

    # 14. Texte libre
    texts = [
        "Predictive modeling is transforming industry and healthcare workflows significantly",
        "Deep learning architectures are powerful for unstructured complex data",
        "A quick brown fox jumped gracefully over the sleeping dog in the forest",
        "Data science requires robust statistics and careful validation protocols",
    ] * (n_samples // 4)

    # 15. Cible classification
    target = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    # Créer quelques NaN dans la cible pour vérifier le nettoyage du train
    target = target.astype(float)
    target[3] = np.nan

    df = pd.DataFrame({
        "id_col": ids,
        "constant_col": constant_col,
        "high_missing_col": high_missing,
        "dup_base": dup_base,
        "dup_copy": dup_copy,
        "num_norm": num_norm,
        "num_skewed": num_skewed,
        "price_str": price_candidates,
        "pct_str": pct_candidates,
        "binary_col": binary_choices,
        "ordinal_col": ordinal_choices,
        "region_low_card": low_card_vals,
        "city_high_card": cities,
        "datetime_col": dates,
        "free_text": texts,
        "target": target,
    })
    return df


def test_full_pipeline_synthetic_dataset(synthetic_dataframe):
    """Vérifie le bon fonctionnement de bout en bout sur un dataset complexe."""
    df = synthetic_dataframe
    prep = AutoPreprocessor(
        PreprocessConfig(
            cardinality_threshold=10,
            missing_threshold=0.90,
            correlation_threshold=0.95,
            rare_category_threshold=0.02,
            skew_threshold=1.0,
            scaler_type="standard",
            random_state=42,
        )
    )

    # Découpage train / test
    train_df = df.iloc[:150].copy()
    test_df = df.iloc[150:].copy()

    # fit_transform_xy
    X_train_trans, y_train_trans = prep.fit_transform_xy(train_df, target="target")
    assert isinstance(X_train_trans, pd.DataFrame)
    assert len(X_train_trans) <= 150  # Lignes de cibles manquantes ou doublons retirées
    assert len(X_train_trans) == len(y_train_trans)

    # Vérification X_train_trans
    assert not X_train_trans.isna().any().any(), "X_train transformé contient des NaN !"
    assert not np.isinf(X_train_trans.to_numpy()).any(), "X_train transformé contient des Inf !"
    assert all(np.issubdtype(dtype, np.number) for dtype in X_train_trans.dtypes), "Colonnes non numériques !"

    # Transformation du test set via transform_xy
    X_test_trans, y_test_trans = prep.transform_xy(test_df, target="target")
    assert len(X_test_trans) == len(test_df), "Transform ne doit JAMAIS supprimer de lignes de test !"
    assert not X_test_trans.isna().any().any(), "X_test transformé contient des NaN !"
    assert not np.isinf(X_test_trans.to_numpy()).any(), "X_test transformé contient des Inf !"
    assert all(np.issubdtype(dtype, np.number) for dtype in X_test_trans.dtypes), "Colonnes non numériques dans test !"

    # Identité stricte des colonnes et de l'ordre entre train et test
    assert list(X_train_trans.columns) == list(X_test_trans.columns), "Les colonnes train et test diffèrent !"

    # Vérification de l'élimination des colonnes inutiles
    rep = prep.report()
    dropped_cols = prep.columns_to_drop_ + prep.dropped_variance_cols_ + prep.dropped_collinear_cols_
    assert "id_col" in prep.columns_to_drop_ or "id_col" in dropped_cols
    assert "constant_col" in prep.columns_to_drop_ or "constant_col" in dropped_cols
    assert "high_missing_col" in prep.columns_to_drop_ or "high_missing_col" in dropped_cols

    # Vérification de la conversion de chaînes chiffrées
    assert "price_str" in prep.column_metadata_
    assert prep.column_metadata_["price_str"]["type"] in [FeatureType.NUMERIC_CONTINUOUS, FeatureType.NUMERIC_DISCRETE]


def test_no_data_leakage(synthetic_dataframe):
    """Vérifie mathématiquement qu'aucune statistique du test ne fuit dans le modèle."""
    df = synthetic_dataframe.dropna(subset=["target"])
    train_df = df.iloc[:120].copy()
    test_df = df.iloc[120:].copy()

    prep = AutoPreprocessor(PreprocessConfig(outlier_method="winsorize", random_state=42))
    prep.fit(train_df.drop(columns=["target"]), y=train_df["target"])

    # 1. Vérifier que les bornes de winsorisation sont issues du train
    num_col = prep.num_cols_[0]
    train_col_vals = pd.to_numeric(
        prep._preprocess_raw_dataframe(train_df)[[num_col]][num_col], errors="coerce"
    ).dropna()
    expected_upper = float(np.nanpercentile(train_col_vals, 99))
    actual_upper = prep.winsorizer_.upper_bounds_[num_col]
    assert np.isclose(expected_upper, actual_upper, rtol=1e-3)

    # 2. Vérifier que la date de référence temporelle est le max du train
    if prep.date_cols_:
        d_col = prep.date_cols_[0]
        train_dates = pd.to_datetime(train_df[d_col], errors="coerce", format="mixed").dropna()
        expected_ref = train_dates.max()
        actual_ref = prep.datetime_extractor_.reference_dates_[d_col]
        assert expected_ref == actual_ref


def test_unseen_categories_in_test(synthetic_dataframe):
    """Vérifie la robustesse face à des modalités totalement inconnues dans le test."""
    train_df = synthetic_dataframe.iloc[:120].copy()
    test_df = synthetic_dataframe.iloc[120:].copy()

    prep = AutoPreprocessor(PreprocessConfig(random_state=42))
    prep.fit(train_df.drop(columns=["target"]), y=train_df["target"])

    # Injection de catégories inconnues et extrêmes dans le test
    test_df_corrupted = test_df.copy()
    test_df_corrupted["region_low_card"] = "Atlantis"  # Continent imaginaire
    test_df_corrupted["city_high_card"] = "Gotham_City"  # Ville inconnue
    test_df_corrupted["binary_col"] = "Peut-Etre"  # Troisième modalité inconnue
    test_df_corrupted["ordinal_col"] = "super_extreme"  # Hors motif
    test_df_corrupted["datetime_col"] = "3050-12-31"  # Date futuriste

    # Ne doit pas lever d'exception et produire une matrice valide
    X_test_trans = prep.transform(test_df_corrupted.drop(columns=["target"]))
    assert not X_test_trans.isna().any().any()
    assert not np.isinf(X_test_trans.to_numpy()).any()
    assert len(X_test_trans) == len(test_df_corrupted)


def test_joblib_save_and_load(synthetic_dataframe):
    """Vérifie la sérialisation / désérialisation exacte via joblib."""
    df = synthetic_dataframe
    train_df = df.iloc[:120].copy()
    test_df = df.iloc[120:].copy()

    prep = AutoPreprocessor(PreprocessConfig(random_state=42))
    prep.fit(train_df.drop(columns=["target"]), y=train_df["target"])
    X_test_trans_orig = prep.transform(test_df.drop(columns=["target"]))

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        prep.save(tmp_path)
        loaded_prep = AutoPreprocessor.load(tmp_path)

        assert isinstance(loaded_prep, AutoPreprocessor)
        assert loaded_prep.is_fitted_

        X_test_trans_loaded = loaded_prep.transform(test_df.drop(columns=["target"]))
        pd.testing.assert_frame_equal(X_test_trans_orig, X_test_trans_loaded)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_target_leakage_detection():
    """Vérifie qu'une colonne corrélée quasi-parfaitement à la cible déclenche une alerte."""
    np.random.seed(42)
    n = 100
    y = np.random.choice([0, 1], size=n)
    # Feature presque identique à y (fuite évidente)
    leaky_feature = y.astype(float) + np.random.normal(0, 0.001, size=n)
    normal_feature = np.random.randn(n)

    df = pd.DataFrame({
        "feat_clean": normal_feature,
        "leak_col": leaky_feature,
        "target": y,
    })

    prep = AutoPreprocessor()
    prep.fit(df.drop(columns=["target"]), y=df["target"])
    rep = prep.report()

    assert "leak_col" in rep["fuites_suspectees"]
    assert any("ALERTE FUITE DE CIBLE" in w for w in rep["avertissements"])


def test_regression_and_inverse_transform():
    """Vérifie la détection de tâche régression et la transformation inverse de la cible."""
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({"x1": np.random.randn(n), "x2": np.random.uniform(0, 10, n)})
    # Cible continue très asymétrique (exponentielle)
    y = np.random.exponential(scale=50.0, size=n)

    prep = AutoPreprocessor(PreprocessConfig(transform_target=True, skew_threshold=1.0))
    X_trans, y_trans = prep.fit_transform_xy(pd.concat([X, pd.Series(y, name="target")], axis=1), target="target")

    assert prep.task_resolved_ == "regression"
    assert prep.target_stats_.get("transformed", False) is True

    # Test inverse_transform_target
    y_recovered = prep.inverse_transform_target(y_trans)
    np.testing.assert_allclose(y, y_recovered, rtol=1e-4)


# =============================================================================
# TESTS DES CAS LIMITES
# =============================================================================

def test_edge_case_tiny_dataset():
    """Cas limite : dataset minuscule (3 lignes)."""
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0],
        "cat": ["a", "b", "a"],
        "target": [0, 1, 0],
    })
    prep = AutoPreprocessor()
    X_trans, y_trans = prep.fit_transform_xy(df, target="target")
    assert len(X_trans) == 3
    assert not X_trans.isna().any().any()


def test_edge_case_single_column():
    """Cas limite : une seule colonne feature."""
    df = pd.DataFrame({
        "single_feat": [1.5, 2.5, 3.5, 4.5, 5.5],
        "target": [0, 1, 0, 1, 0],
    })
    prep = AutoPreprocessor()
    X_trans, y_trans = prep.fit_transform_xy(df, target="target")
    assert X_trans.shape[1] >= 1
    assert not X_trans.isna().any().any()


def test_edge_case_all_missing_column():
    """Cas limite : colonne 100% NaN (doit être supprimée proprement)."""
    df = pd.DataFrame({
        "empty_col": [np.nan, np.nan, np.nan, np.nan],
        "good_col": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })
    prep = AutoPreprocessor()
    X_trans, y_trans = prep.fit_transform_xy(df, target="target")
    assert "empty_col" in prep.columns_to_drop_
    assert not X_trans.isna().any().any()


def test_edge_case_rare_target_class():
    """Cas limite : cible avec une classe ultra-rare (alerte déséquilibre)."""
    df = pd.DataFrame({
        "feat1": np.random.randn(50),
        "feat2": np.random.randn(50),
        "target": [0] * 48 + [1] * 2,  # 2 sur 50 (4% < 5%)
    })
    prep = AutoPreprocessor()
    X_trans, y_trans = prep.fit_transform_xy(df, target="target")
    assert prep.target_stats_["minority_class_ratio"] == 0.04
    assert any("Déséquilibre sévère" in w for w in prep.warnings_)


# =============================================================================
# TESTS DE FUITE (détecteur sur colonnes brutes) ET TARGET ENCODING SANS FUITE
# =============================================================================

def _titanic_like(n=600, seed=0):
    """Mini-Titanic synthétique : 'boat' n'est renseigné que pour les survivants (fuite)."""
    rng = np.random.RandomState(seed)
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.2, 0.55])
    age = np.where(rng.rand(n) < 0.2, np.nan, rng.normal(30, 12, n).clip(1, 80))
    p = 0.2 + 0.5 * (sex == "female") + 0.1 * (pclass == 1)
    y = (rng.rand(n) < p.clip(0, 0.95)).astype(int)
    boat = np.where(y == 1, rng.choice([str(i) for i in range(1, 17)], size=n), None)
    boat = np.where((rng.rand(n) < 0.03), None, boat)  # quelques survivants sans canot
    return pd.DataFrame({
        "pclass": pclass, "sex": sex, "age": age,
        "fare": rng.exponential(30, n), "sibsp": rng.randint(0, 4, n), "parch": rng.randint(0, 3, n),
        "embarked": rng.choice(["S", "C", "Q"], size=n),
        "cabin": rng.choice([f"C{i}" for i in range(40)] + [None] * 60, size=n),
        "ticket": [f"T{rng.randint(0, 400)}" for _ in range(n)],
        "home_dest": rng.choice([f"City{i}" for i in range(30)], size=n),
        "boat": boat, "target": y,
    })


def test_categorical_leak_detected_and_dropped():
    df = _titanic_like()
    prep = AutoPreprocessor(PreprocessConfig(drop_leaky_columns=True, verbose=0))
    X, y = prep.fit_transform_xy(df, target="target")
    rep = prep.report()
    assert "boat" in rep["fuites_suspectees"]
    assert "boat" in prep.columns_to_drop_
    assert prep.column_metadata_["boat"]["type"] == FeatureType.DROP_LEAKAGE
    assert not any(c.startswith("boat") for c in X.columns)
    # pas de faux positif sur les features légitimes
    assert rep["fuites_suspectees"] == ["boat"]


def test_leak_only_alerts_when_drop_disabled():
    df = _titanic_like()
    prep = AutoPreprocessor(PreprocessConfig(drop_leaky_columns=False, verbose=0))
    X, _ = prep.fit_transform_xy(df, target="target")
    assert "boat" in prep.report()["fuites_suspectees"]
    assert "boat" not in prep.columns_to_drop_
    assert any(c.startswith("boat") for c in X.columns)


def test_strong_legit_feature_not_flagged_iris():
    """Cas Iris : features très prédictives mais légitimes -> ni alerte, ni suppression."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    prep = AutoPreprocessor(PreprocessConfig(drop_leaky_columns=True, verbose=0), task="classification")
    prep.fit(iris.data.copy(), y=iris.target)
    assert prep.suspected_leakage_cols_ == []
    assert prep.columns_to_drop_ == []
    assert prep.transform(iris.data).shape[1] == 4  # pas de colinéarité sous 10 features


def test_strong_legit_features_not_flagged_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    prep = AutoPreprocessor(PreprocessConfig(drop_leaky_columns=True, verbose=0), task="classification")
    prep.fit(d.data.copy(), y=d.target)
    assert prep.suspected_leakage_cols_ == []


def test_target_encoding_train_has_no_self_leak():
    """Colonne haute cardinalité SANS lien avec la cible : l'encodage du train ne doit pas
    être corrélé à la cible (ce serait du surapprentissage sur ses propres labels)."""
    rng = np.random.RandomState(1)
    n = 600
    y = rng.randint(0, 2, n)
    df = pd.DataFrame({
        "noise_cat": [f"cat_{rng.randint(0, 200)}" for _ in range(n)],
        "a": rng.randn(n), "b": rng.randn(n), "target": y,
    })
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    X, yy = prep.fit_transform_xy(df, target="target")
    te_col = [c for c in X.columns if c.startswith("noise_cat")][0]
    r_oof = abs(np.corrcoef(X[te_col].to_numpy(), np.asarray(yy))[0, 1])
    prep2 = AutoPreprocessor(PreprocessConfig(verbose=0))
    prep2.fit(df.drop(columns=["target"]), y=df["target"])
    X_in = prep2.transform(df.drop(columns=["target"]))
    r_in = abs(np.corrcoef(X_in[te_col].to_numpy(), np.asarray(yy))[0, 1])
    assert r_oof < 0.2
    assert r_in > r_oof  # l'ancien comportement (fit puis transform) surapprend


def test_numeric_strings_with_units_and_ids():
    """Les montants avec unité ('67,50 MAD') deviennent numériques ; les IDs ('C1004') non."""
    f = AutoPreprocessor._clean_numeric_str_value
    assert f("67,50 MAD") == 67.5
    assert f("1 200") == 1200.0
    assert f("MAD 67") == 67.0
    assert f("1.234,56 €") == 1234.56
    assert np.isnan(f("C1004"))
    assert np.isnan(f("T000012"))
    assert np.isnan(f("2024-01-05"))