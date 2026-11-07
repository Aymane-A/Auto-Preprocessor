# -*- coding: utf-8 -*-
"""
Suite de tests unitaires et d'intégration pour AutoPreprocessor.
Vérifie la robustesse, l'absence de fuite de données, la persistance et les cas limites.
"""

import os
import re
import tempfile
from copy import deepcopy
from functools import lru_cache
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import pytest
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, FeatureType


DEMO_PATHS = ("data/churn_telecom_demo.csv", "data/immobilier_demo.csv", "data/fraud_signal_demo.csv")


@pytest.fixture(scope="module")
def demo_analysis():
    """Share deterministic analysis setup while isolating each test's inputs."""
    from app_preprocess import read_table

    @lru_cache(maxsize=None)
    def analyze(path, target, identifier_threshold):
        frame = read_table(path)
        config = PreprocessConfig(verbose=0, identifier_unique_threshold=identifier_threshold)
        return frame, AutoPreprocessor(config).analyze(frame, target=target)

    def get_case(path, target=None, identifier_threshold=PreprocessConfig.identifier_unique_threshold):
        frame, result = analyze(path, target, identifier_threshold)
        return frame.copy(deep=True), deepcopy(result)

    return get_case


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

    prep = AutoPreprocessor(PreprocessConfig(outlier_method="winsorize", adaptive_recommendations=False, random_state=42))
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


def _noise_df(n=1500, seed=1):
    r = np.random.RandomState(seed)
    info = r.randn(n)
    y = (info + 0.5 * r.randn(n) > 0.9).astype(int)
    d = pd.DataFrame({
        "info": info,
        "date": (pd.Timestamp("2024-01-01") + pd.to_timedelta(r.randint(0, 400, n), unit="D")).strftime("%Y-%m-%d"),
        "hc": r.choice([f"k{i}" for i in range(80)], n),
    })
    return d, y


def test_noise_guard_warn_drop_and_small_data():
    d, y = _noise_df()
    p = AutoPreprocessor(PreprocessConfig(noise_guard="warn", verbose=0), task="classification")
    p.fit(d, y)
    assert set(p.noise_cols_detected_) == {"date", "hc"} and p.noise_cols_dropped_ == []
    p = AutoPreprocessor(PreprocessConfig(noise_guard="drop", verbose=0), task="classification")
    X = p.fit_transform(d, y)
    assert set(p.noise_cols_dropped_) == {"date", "hc"} and X.shape[1] == 1
    p = AutoPreprocessor(PreprocessConfig(noise_guard="drop", keep_columns=["date"], verbose=0), task="classification")
    p.fit(d, y)
    assert p.noise_cols_dropped_ == ["hc"]
    p = AutoPreprocessor(PreprocessConfig(noise_guard="drop", verbose=0), task="classification")
    p.fit(d.iloc[:80], y[:80])
    assert p.noise_cols_dropped_ == []  # trop peu de lignes : retour en warn


def test_numeric_string_decision_is_kept_at_transform():
    r = np.random.RandomState(0)
    tr = pd.DataFrame({"prix": [f"{v:.2f} MAD".replace(".", ",") for v in r.uniform(10, 100, 400)], "x": r.randn(400)})
    y = (r.rand(400) > 0.5).astype(int)
    p = AutoPreprocessor(task="classification")
    p.config.verbose = 0
    p.fit(tr, y)
    assert "prix" in p.numeric_string_cols_
    te = pd.DataFrame({"prix": ["12,50 MAD", "n/a", "abc", "33,00 MAD", "7,10 MAD"], "x": [0.0] * 5})
    Xt = p.transform(te)
    assert Xt.shape[0] == 5 and not Xt.isna().any().any()
    assert any("prix" in c for c in Xt.columns)
def test_analyze_does_not_mutate_input():
    from auto_preprocessor import AutoPreprocessor

    frame = pd.DataFrame({"nombre": [1.0, None, 3.0], "categorie": ["a", "b", "a"]})
    before = frame.copy(deep=True)
    result = AutoPreprocessor().analyze(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert len(result.to_dataframe()) == 2


def test_analysis_demo_recommendations(demo_analysis):
    churn = demo_analysis(DEMO_PATHS[0])[1].to_dataframe().set_index("colonne")
    assert churn.loc["facture_mensuelle", "type"] == "NUMERIC_CONTINUOUS"
    fraud = demo_analysis(DEMO_PATHS[2])[1].to_dataframe().set_index("colonne")
    assert fraud.loc["heure", "action"] == "DATE_FEATURES"
    housing = demo_analysis(DEMO_PATHS[1])[1].to_dataframe().set_index("colonne")
    assert housing.loc["prix_dh", "action"] in {"LOG1P", "SQRT", "YEO_JOHNSON"}


@pytest.mark.parametrize("path", ["data/churn_telecom_demo.csv", "data/immobilier_demo.csv", "data/fraud_signal_demo.csv"])
def test_accept_analysis_recommendations_is_noop(path, demo_analysis):
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig, ColumnOverride
    frame, analysis = demo_analysis(path)
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    overrides = {row["colonne"]: ColumnOverride(row["action"]) for row in analysis.columns}
    accepted = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides)).fit_transform(frame)
    pd.testing.assert_frame_equal(baseline, accepted)


def test_accept_recommendations_matches_sklearn_dataset():
    from sklearn.datasets import load_diabetes
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig, ColumnOverride

    frame = load_diabetes(as_frame=True).data
    config = PreprocessConfig(verbose=0)
    baseline = AutoPreprocessor(config).fit_transform(frame)
    result = AutoPreprocessor(config).analyze(frame)
    config_with_recs = PreprocessConfig(verbose=0, column_overrides={
        row["colonne"]: ColumnOverride(row["action"]) for row in result.columns
    })
    accepted = AutoPreprocessor(config_with_recs).fit_transform(frame)
    pd.testing.assert_frame_equal(baseline, accepted)


def test_overrides_drop_scaler_and_validation():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig, ColumnOverride

    frame = pd.DataFrame({"x": np.arange(1, 50, dtype=float), "y": np.arange(49, dtype=float) ** 2,
                          "cat": [f"v{i % 12}" for i in range(49)], "text": ["long text"] * 49})
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0, outlier_method="none")).fit_transform(frame)
    dropped = AutoPreprocessor(PreprocessConfig(verbose=0, outlier_method="none",
        column_overrides={"x": {"type": "DROP"}})).fit_transform(frame)
    assert not any(c == "x" or c.startswith("x_") for c in dropped.columns)
    scaled = AutoPreprocessor(PreprocessConfig(verbose=0, outlier_method="none",
        column_overrides={"x": {"scaling": "ROBUST"}})).fit_transform(frame)
    assert not np.allclose(baseline["x"], scaled["x"])
    with pytest.raises(ValueError, match="incompatible"):
        AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={"text": {"distribution": "YEO_JOHNSON"}})).fit(frame)


def test_onehot_override_changes_high_cardinality_encoding():
    frame = pd.DataFrame({"category": [f"c{i % 20}" for i in range(100)],
                          "value": np.arange(100, dtype=float)})
    y = np.asarray([i % 2 for i in range(100)])
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame, y)
    onehot = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"category": {"encoding": "ONEHOT"}})).fit_transform(frame, y)
    assert baseline.shape[1] != onehot.shape[1]


def test_override_saved_and_loaded():
    from auto_preprocessor import ColumnOverride

    config = PreprocessConfig(verbose=0, column_overrides={"x": ColumnOverride({"type": "DROP"})})
    prep = AutoPreprocessor(config)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.joblib")
        prep.save(path)
        loaded = AutoPreprocessor.load(path)
    assert loaded.config.column_overrides == config.column_overrides


def test_analysis_steps_have_evidence_and_json_roundtrip():
    from auto_preprocessor import AutoPreprocessor
    from app_preprocess import export_user_choices, import_user_choices

    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(pd.DataFrame({"x": np.arange(20)}))
    assert all(all(step["measure"] and step["threshold"] and step["explanation"]
                   for step in row["steps"].values()) for row in analysis.columns)
    choices = {"x": "KEEP_NUMERIC"}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "choices.json")
        export_user_choices(analysis, choices, path)
        assert import_user_choices(analysis, path) == choices


def test_per_step_overrides_are_validated_and_applied():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    from app_preprocess import validate_step_choice

    frame = pd.DataFrame({"amount": [1., 2., 3., 4., 5., 100., np.nan, 8., 9., 10.]})
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    assert validate_step_choice(analysis, "amount", "scaling", "MINMAX") == "MINMAX"
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    robust = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"amount": {"scaling": "MINMAX"}})).fit_transform(frame)
    assert not np.allclose(baseline["amount"], robust["amount"])
    with pytest.raises(ValueError, match="choix valides"):
        validate_step_choice(analysis, "amount", "scaling", "ORDINAL")


def test_supported_categorical_missing_frequency_and_ordinal_overrides():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    frame = pd.DataFrame({"color": ["red", "blue", None, "red", "green", "blue"] * 12,
                          "level": ["low", "mid", "high"] * 24})
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    frequency = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"color": {"encoding": "FREQUENCY_ENCODE"}})).fit_transform(frame)
    assert "color_freq" in frequency.columns and "color_freq" not in baseline.columns
    missing_category = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"color": {"missing": "MODE"}})).fit_transform(frame)
    assert missing_category.shape[1] != baseline.shape[1]
    constant_fill = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"color": {"missing": ("CONSTANT", "__CUSTOM__")}})).fit_transform(frame)
    assert not constant_fill.equals(baseline)
    ordinal = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"level": {"encoding": ("ORDINAL", {"order": ["low", "mid", "high"]})}})).fit_transform(frame)
    assert "level" in ordinal.columns
    assert list(ordinal["level"].head(3)) == [0.0, 1.0, 2.0]


def test_identifier_name_is_dropped_and_explained(demo_analysis):
    result = demo_analysis(DEMO_PATHS[2])[1].to_dataframe().set_index("colonne")
    assert result.loc["client_id", "action"] == "DROP"
    assert "94.7%" in result.loc["client_id", "raison"]
    assert "90.0%" in result.loc["client_id", "raison"]


def test_all_step_recommendations_are_accepted_by_fit():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    frame = pd.DataFrame({
        "value": np.linspace(1, 100, 60),
        "category": [f"g{i % 4}" for i in range(60)],
        "date": pd.date_range("2024-01-01", periods=60).astype(str),
        "comment": [f"This is a sufficiently long free text example number {i}" for i in range(60)],
        "constant": ["same"] * 60,
    })
    config = PreprocessConfig(verbose=0)
    baseline = AutoPreprocessor(config).fit_transform(frame)
    analysis = AutoPreprocessor(config).analyze(frame)
    overrides = {row["colonne"]: {step: detail["action"] for step, detail in row["steps"].items()}
                 for row in analysis.columns}
    accepted = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides)).fit_transform(frame)
    pd.testing.assert_frame_equal(baseline, accepted)


def test_scaler_recommendations_follow_statistical_rules():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    rng = np.random.RandomState(123)
    frame = pd.DataFrame({
        "gaussian": rng.normal(size=600),
        "outlier": np.r_[rng.normal(size=599), 80.0],
        "sparse": np.r_[np.zeros(550), rng.normal(size=50)],
        "bounded": rng.uniform(0.0, 1.0, 600),
    })
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame).to_dataframe().set_index("colonne")
    steps = result["steps"]
    assert steps["outlier"]["scaling"]["action"] == "ROBUST"
    assert steps["sparse"]["scaling"]["action"] == "MAXABS"
    assert steps["bounded"]["scaling"]["action"] == "MINMAX"
    assert all(steps[c]["scaling"]["measure"] and steps[c]["scaling"]["threshold"] for c in frame.columns)


def test_analysis_exposes_only_central_supported_actions():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig, SUPPORTED_ACTIONS, _action_group, _action_supported

    frame = pd.DataFrame({"number": np.arange(40, dtype=float),
                          "category": [f"c{i % 3}" for i in range(40)],
                          "date": pd.date_range("2024-01-01", periods=40).astype(str)})
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    for row in result.columns:
        group = _action_group(row["type"])
        for step, detail in row["steps"].items():
            assert detail["choices"] == SUPPORTED_ACTIONS[step].get(
                group, SUPPORTED_ACTIONS[step].get("any", ["AUTO"]))
            assert all(_action_supported(step, group, action) for action in detail["choices"])
    with pytest.raises(ValueError, match="choix valides"):
        from app_preprocess import validate_step_choice
        validate_step_choice(result, "number", "distribution", "BINNING")


def test_cleaning_false_missing_variants_and_numeric_text_choices():
    frame = pd.DataFrame({
        "token": ["ok", "N/A", "?", "ok", "null", "ok"] * 12,
        "city": ["Café", " cafe ", "CAFE", "Rabat", "RABAT", "rabat"] * 12,
        "amount": ["67,50 MAD", "45%", "120 MAD"] * 24,
    })
    default = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    keep_missing = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"token": {"clean_missing_tokens": "KEEP"}})).fit_transform(frame)
    selected_missing = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"token": {"clean_missing_tokens": ("SELECTED_TO_NAN", {"values": ["?"]})}})).fit_transform(frame)
    keep_variants = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"city": {"clean_variants": "KEEP"}})).fit_transform(frame)
    merge_variants = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"city": {"clean_variants": "MERGE"}})).fit_transform(frame)
    convert_numeric_text = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"amount": {"clean_numeric_text": "CONVERT"}})).fit_transform(frame)
    keep_numeric_text = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"amount": {"clean_numeric_text": "KEEP"}})).fit_transform(frame)
    assert not default.equals(keep_missing)
    assert not default.equals(selected_missing)
    assert not default.equals(keep_variants)
    assert not default.equals(keep_numeric_text)
    pd.testing.assert_frame_equal(default, merge_variants)
    pd.testing.assert_frame_equal(default, convert_numeric_text)
    assert "amount" in AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame).columns


def test_cleaning_dates_and_impossible_value_actions():
    frame = pd.DataFrame({
        "date_event": pd.date_range("2024-01-01", periods=60).astype(str),
        "age": [25, -5, 130, 40, 50, 70] * 10,
    })
    parsed = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"date_event": {"clean_dates": "PARSE"}})).fit_transform(frame)
    kept_date = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"date_event": {"clean_dates": "KEEP"}})).fit_transform(frame)
    dropped_date = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"date_event": {"clean_dates": "DROP_COLUMN"}})).fit_transform(frame)
    bounds = {"age": (0., 120.)}
    clipped = AutoPreprocessor(PreprocessConfig(verbose=0, value_bounds=bounds,
        column_overrides={"age": {"clean_impossible": "CLIP"}})).fit_transform(frame)
    replaced = AutoPreprocessor(PreprocessConfig(verbose=0, value_bounds=bounds,
        column_overrides={"age": {"clean_impossible": "NAN"}})).fit_transform(frame)
    ignored = AutoPreprocessor(PreprocessConfig(verbose=0, value_bounds=bounds,
        column_overrides={"age": {"clean_impossible": "IGNORE"}})).fit_transform(frame)
    dropped_rows = AutoPreprocessor(PreprocessConfig(verbose=0, value_bounds=bounds,
        column_overrides={"age": {"clean_impossible": "DROP_ROWS"}})).fit_transform(frame)
    assert any(c.startswith("date_event_") for c in parsed.columns)
    assert not any(c.startswith("date_event_") for c in kept_date.columns)
    assert not any(c.startswith("date_event_") for c in dropped_date.columns)
    assert len(dropped_rows) < len(frame)
    assert not replaced.equals(ignored)
    assert not clipped.equals(ignored)


def test_cleaning_duplicate_choices_change_row_count():
    frame = pd.DataFrame({"feature": ["a", "b", "a", "c"] * 15})
    default = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    kept = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"feature": {"duplicates": "KEEP"}})).fit_transform(frame)
    assert len(default) == 3
    assert len(kept) == len(frame)


@pytest.mark.parametrize("forced", ["FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT", "KEEP_RAW"])
def test_forced_type_and_keep_raw_are_applied(forced):
    n = 80
    frame = pd.DataFrame({
        "feature": [str(i % 12) for i in range(60)] + ["0"] * 20,
        "date_col": pd.date_range("2024-01-01", periods=n).astype(str),
        "text_col": [f"A long sentence with enough words, sample {i % 8}" for i in range(n)],
    })
    col = "date_col" if forced == "FORCE_DATE" else "text_col" if forced == "FORCE_TEXT" else "feature"
    if forced == "KEEP_RAW":
        frame["feature"] = (["A", "a", " A "] * ((n + 2) // 3))[:n]
    result = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={col: {"type": forced}})).fit_transform(frame)
    if forced == "FORCE_DATE":
        assert any(c.startswith("date_col_") for c in result.columns)
    elif forced == "FORCE_TEXT":
        assert any(c.startswith("text_col_") for c in result.columns)
    elif forced == "FORCE_CATEGORICAL":
        assert "feature_freq" in result.columns
    elif forced == "FORCE_NUMERIC":
        assert "feature" in result.columns
    else:
        assert any("feature_" in c for c in result.columns)


def test_cleaning_parameters_validate_and_roundtrip_json_joblib():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    from app_preprocess import (export_user_choices, import_user_choices, choices_to_overrides,
                                parse_step_parameters)

    with pytest.raises(ValueError, match="percentiles"):
        parse_step_parameters("outliers", "WINSORIZE", "0.9, 0.1", "NUMERIC_CONTINUOUS")
    parsed = parse_step_parameters("outliers", "WINSORIZE", "0.05, 0.95", "NUMERIC_CONTINUOUS")
    assert parsed == {"lower": 0.05, "upper": 0.95}
    frame = pd.DataFrame({"value": np.r_[np.arange(1., 40.), 1000.]})
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    choices = {"value": {"outliers": ("WINSORIZE", parsed)}}
    with tempfile.TemporaryDirectory() as folder:
        json_path = os.path.join(folder, "choices.json")
        export_user_choices(analysis, choices, json_path)
        restored = import_user_choices(analysis, json_path)
        assert restored == choices
        overrides = choices_to_overrides(analysis, restored)
        prep = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides))
        prep.fit(frame)
        model_path = os.path.join(folder, "prep.joblib")
        prep.save(model_path)
        loaded = AutoPreprocessor.load(model_path)
    assert loaded.config.column_overrides == overrides
    assert loaded.winsorizer_.lower_bounds_["value"] == pytest.approx(frame["value"].quantile(.05))


def test_analysis_cleaning_counts_and_prelaunch_summary():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    from app_preprocess import build_preprocess_summary

    frame = pd.DataFrame({"value": ["N/A", "?", "null", "a", "b", "c", "d", "e", "f", "g"]})
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    row = result.to_dataframe().iloc[0]
    assert row["steps"]["clean_missing_tokens"]["measure"].startswith("3 cellules")
    summary = build_preprocess_summary(result, {"value": {"type": "DROP"}})
    assert summary["dropped_columns"] == ["value"]
    assert summary["user_columns"] == ["value"]


def test_churn_target_relations_are_populated_for_every_feature(demo_analysis):
    from app_preprocess import build_analysis_table

    _, result = demo_analysis(DEMO_PATHS[0], target="churn")
    table = result.to_dataframe()
    target_row = table.loc[table["colonne"] == "churn"].iloc[0]
    features = table.loc[table["colonne"] != "churn"]
    assert target_row["est_cible"]
    assert target_row["type"] == "TARGET"
    assert target_row["steps"] == {}
    assert features.loc[~features["alerte_identifiant"], "relation_cible"].notna().all()
    assert features.loc[features["alerte_identifiant"], "relation_cible"].isna().all()
    display = build_analysis_table(result).set_index("colonne")
    assert display.loc["churn", "état_choix"] == "cible"
    assert display.loc["churn", "choix_final"] == "cible"


def test_analyze_statistics_use_converted_numeric_text():
    values = ["67,50 MAD"] * 98 + ["675,00 MAD", "1200,00 MAD"]
    frame = pd.DataFrame({"facture_mensuelle": values})
    analyzed = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame).to_dataframe().iloc[0]
    independent = pd.Series([67.5] * 98 + [675.0, 1200.0])
    q1, q3 = independent.quantile([.25, .75])
    iqr = q3 - q1
    expected_outliers_pct = float(((independent < q1 - 1.5 * iqr) |
                                   (independent > q3 + 1.5 * iqr)).mean() * 100)
    assert analyzed["type"].startswith("NUMERIC")
    assert analyzed["skew"] == pytest.approx(independent.skew(), abs=1e-6)
    assert analyzed["outliers_pct"] == pytest.approx(expected_outliers_pct)
    assert analyzed["skew"] > 1
    assert analyzed["outliers_pct"] > 0


def test_analysis_display_formatting_and_confidence_thresholds():
    from app_preprocess import analysis_sort_key, format_analysis_value
    from auto_preprocessor import recommendation_confidence

    assert recommendation_confidence(10, 1.0) == "faible"
    assert recommendation_confidence(50, .60) == "moyenne"
    assert recommendation_confidence(500, .50, p_value=.0001) == "haute"
    assert format_analysis_value("manquants_pct", 12.345) == "12.3%"
    assert format_analysis_value("skew", 1.23456) == "1.23"
    assert format_analysis_value("skew", np.nan) == "—"
    assert analysis_sort_key("10.00%") < analysis_sort_key("100.00%")


def test_detail_panel_logic_is_pure_and_contains_step_evidence():
    from app_preprocess import build_analysis_detail

    frame = pd.DataFrame({"price": np.random.default_rng(7).lognormal(3, 1.2, 300)})
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    empty = build_analysis_detail(result, None)
    assert "Sélectionnez une colonne" in empty["header"]
    detail = build_analysis_detail(result, "price", {"price": {"distribution": "NONE"}})
    assert "price | type=" in detail["header"]
    assert any(row["etape"] == "distribution" and row["modifie"] and
               row["recommandation"] in {"LOG1P", "SQRT", "YEO_JOHNSON"} for row in detail["rows"])
    assert all(row["explication"] and row["preuve"] and " ; seuil : " in row["preuve"]
               and row["preuve"].split(" ; seuil : ")[0] in row["preuve"]
               and row["preuve"] not in row["explication"] for row in detail["rows"])


def test_analysis_recommendations_explanations_and_csv_export(tmp_path):
    from app_preprocess import build_analysis_detail, build_analysis_table, export_analysis_csv

    frame = pd.DataFrame({
        "amount": np.r_[np.arange(1., 100.), 1500.],
        "group": ["north", "south", "west", "east"] * 25,
    })
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    for row in result.columns:
        details = build_analysis_detail(result, row["colonne"])["rows"]
        assert details
        assert all(item["explication"].strip() and item["preuve"]
                   and " ; seuil : " in item["preuve"]
                   and item["preuve"].split(" ; seuil : ")[0] in item["preuve"]
                   and item["preuve"] not in item["explication"] for item in details)
        for step, recommendation in row["steps"].items():
            if len(recommendation.get("choices", [])) > 1:
                assert recommendation["explanation"].strip()
                assert recommendation["measure"].strip()
                assert recommendation["evidence"].startswith(recommendation["measure"])
                assert " ; seuil : " in recommendation["evidence"]
                assert recommendation["evidence"] not in recommendation["explanation"]
    table = build_analysis_table(result)
    assert table["raison"].str.len().gt(0).all()
    assert table["preuve"].str.len().gt(0).all()
    path = tmp_path / "analyse_colonnes.csv"
    export_analysis_csv(result, str(path))
    exported = pd.read_csv(path)
    assert {"Raison", "Preuve", "Explication", "Explication — Distribution"}.issubset(exported.columns)
    assert exported.loc[exported["Colonne"] == "amount", "Explication — Distribution"].notna().all()


def test_dropped_column_detail_shows_only_keep_or_drop_stage(demo_analysis):
    from app_preprocess import build_analysis_detail

    _, result = demo_analysis(DEMO_PATHS[2], target="is_fraud")
    detail = build_analysis_detail(result, "transaction_id", show_all=True)
    assert len(detail["rows"]) == 1
    assert detail["rows"][0]["etape_label"] == "Garder / Supprimer"
    assert detail["rows"][0]["choix_labels"] == ["Garder (type automatique)", "Supprimer"]


def test_empty_cleanup_steps_are_hidden_and_counted():
    from app_preprocess import build_analysis_detail

    frame = pd.DataFrame({"measure": np.arange(100, dtype=float)})
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    compact = build_analysis_detail(result, "measure")
    expanded = build_analysis_detail(result, "measure", show_all=True)
    hidden_cleaning = {"clean_missing_tokens", "clean_variants", "clean_numeric_text",
                       "clean_dates", "clean_impossible", "duplicates"}
    assert compact["masked_count"] > 0
    assert hidden_cleaning.isdisjoint({row["etape"] for row in compact["rows"]})
    assert hidden_cleaning.issubset({row["etape"] for row in expanded["rows"]})


def test_visible_analysis_labels_do_not_expose_internal_identifiers(demo_analysis):
    from app_preprocess import build_analysis_detail, format_analysis_value

    _, result = demo_analysis(DEMO_PATHS[2], target="is_fraud")
    from app_preprocess import label_action, label_step
    texts = [format_analysis_value("type", "DROP_ID"), label_action("TO_NAN"),
             label_step("clean_missing_tokens")]
    for row in result.columns:
        detail = build_analysis_detail(result, row["colonne"], show_all=True)
        texts.append(detail["header"])
        for item in detail["rows"]:
            texts.extend([item["etape_label"], item["recommandation_label"], item["preuve"],
                          item["explication_affichee"], item["choix_label"], *item["choix_labels"], item["aide_choix"]])
    visible = " ".join(texts)
    for forbidden in ("TO_NAN", "DEFAULT_NA_VALUES", "clean_missing_tokens", "{}"):
        assert forbidden not in visible
    assert "aucune" in visible.lower()


def test_display_explanation_does_not_repeat_proof_measure(demo_analysis):
    from app_preprocess import build_analysis_detail

    _, result = demo_analysis(DEMO_PATHS[2], target="is_fraud")
    for row in result.columns:
        detail = build_analysis_detail(result, row["colonne"], show_all=True)
        for item in detail["rows"]:
            assert "Preuve :" not in item["explication_affichee"]
            assert item["preuve"]
            assert item["explication_affichee"]


def test_target_suggestion_for_demo_datasets():
    from app_preprocess import read_table, suggest_target_column

    assert suggest_target_column(read_table("data/churn_telecom_demo.csv")) == "churn"
    assert suggest_target_column(read_table("data/fraud_signal_demo.csv")) == "is_fraud"
    assert suggest_target_column(read_table("data/immobilier_demo.csv")) is None


def test_tk_smoke_load_analyze_and_treeview_count():
    import time
    import tkinter
    from app_preprocess import App, read_table

    try:
        app = App()
    except tkinter.TclError as exc:
        pytest.skip(f"Tkinter indisponible dans cette session : {exc}")
    try:
        app.root.withdraw()
        path = os.path.abspath("data/churn_telecom_demo.csv")
        frame = read_table(path)
        app._load_path(path)
        assert app.v_target.get() == "churn"
        app._start_analysis()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not app.tree_analysis.get_children(""):
            app.root.update()
            time.sleep(.03)
        assert len(app.tree_analysis.get_children("")) == len(frame.columns)
        assert app.analysis is not None
    finally:
        try:
            app.root.destroy()
        except tkinter.TclError:
            pass


def test_target_provided_populates_relation_cible_for_all_features(demo_analysis):
    """Test (a) : cible fournie -> relation_cible rempli pour toutes les features."""
    from app_preprocess import build_analysis_table

    _, result = demo_analysis(DEMO_PATHS[0], target="churn")
    table = result.to_dataframe()
    target_row = table.loc[table["colonne"] == "churn"].iloc[0]
    features = table.loc[table["colonne"] != "churn"]

    assert target_row["est_cible"]
    assert target_row["type"] == "TARGET"
    assert pd.isna(target_row["relation_cible"])
    assert features.loc[~features["alerte_identifiant"], "relation_cible"].notna().all()
    assert features.loc[features["alerte_identifiant"], "relation_cible"].isna().all()
    assert (features.loc[~features["alerte_identifiant"], "relation_cible"] >= 0).all()

    # Vérification que Cramér's V, Spearman ou MI est présent pour chaque feature
    has_stat = (
        features["cramer_cible"].notna()
        | features["spearman_cible"].notna()
        | features["information_mutuelle_cible"].notna()
    )
    assert has_stat.loc[~features["alerte_identifiant"]].all()

    display = build_analysis_table(result).set_index("colonne")
    assert display.loc["churn", "état_choix"] == "cible"
    assert display.loc["churn", "choix_final"] == "cible"


def test_facture_mensuelle_stats_match_independent_pandas(demo_analysis):
    """Test (b) : facture_mensuelle : stats identiques à un calcul pandas indépendant."""
    from auto_preprocessor import AutoPreprocessor

    df, res = demo_analysis(DEMO_PATHS[0], target="churn")
    cleaned_vals = df["facture_mensuelle"].map(AutoPreprocessor._clean_numeric_str_value).dropna()
    expected_skew = float(cleaned_vals.skew())
    expected_min = float(cleaned_vals.min())
    expected_max = float(cleaned_vals.max())
    expected_mean = float(cleaned_vals.mean())
    q1, q3 = cleaned_vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    expected_outliers_pct = float(((cleaned_vals < q1 - 1.5 * iqr) | (cleaned_vals > q3 + 1.5 * iqr)).mean() * 100)

    row = next(r for r in res.columns if r["colonne"] == "facture_mensuelle")

    assert row["type"] == "NUMERIC_CONTINUOUS"
    assert row["skew"] == pytest.approx(expected_skew, abs=1e-5)
    assert row["outliers_pct"] == pytest.approx(expected_outliers_pct, abs=1e-5)
    assert row["min"] == pytest.approx(expected_min, abs=1e-5)
    assert row["max"] == pytest.approx(expected_max, abs=1e-5)
    assert row["mean"] == pytest.approx(expected_mean, abs=1e-5)


def test_analysis_formatting_no_nan_headers_and_sorting(demo_analysis):
    """Test (c) : formatage : pas de 'nan', unités dans les en-têtes, tri correct."""
    from app_preprocess import analysis_sort_key, build_analysis_table, format_analysis_value

    assert format_analysis_value("skew", np.nan) == "—"
    assert format_analysis_value("relation_cible", float("nan")) == "—"
    assert format_analysis_value("manquants_pct", np.nan) == "—"
    assert format_analysis_value("colonne", None) == "—"
    assert format_analysis_value("colonne", "") == "—"
    assert format_analysis_value("colonne", "nan") == "—"
    assert format_analysis_value("manquants_pct", 12.345) == "12.3%"
    assert format_analysis_value("outliers_pct", 0.0) == "0.0%"
    assert format_analysis_value("skew", 1.23456) == "1.23"

    _, res = demo_analysis(DEMO_PATHS[0], target="churn")
    table = build_analysis_table(res)
    for col in ["manquants_pct", "outliers_pct", "skew", "relation_cible", "confiance"]:
        if col in table.columns:
            formatted_cells = [format_analysis_value(col, val) for val in table[col]]
            assert not any("nan" in str(cell).lower() for cell in formatted_cells), f"'nan' trouvé dans {col}"

    # Clé de tri numérique et em-dash toujours à la fin
    assert analysis_sort_key("2.5%") < analysis_sort_key("10.0%")
    assert analysis_sort_key("10.0%") < analysis_sort_key("100.0%")
    assert analysis_sort_key("—")[0] == 2


def test_three_confidence_levels_statistical_rules():
    """Test (d) : les trois niveaux de confiance (faible, moyenne, haute)."""
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig, recommendation_confidence

    assert recommendation_confidence(sample_size=15, evidence_strength=0.90) == "faible"
    assert recommendation_confidence(sample_size=50, evidence_strength=0.20) == "faible"
    assert recommendation_confidence(sample_size=50, evidence_strength=0.60) == "moyenne"
    assert recommendation_confidence(sample_size=200, evidence_strength=0.55) == "moyenne"
    assert recommendation_confidence(sample_size=200, evidence_strength=0.85) == "haute"
    assert recommendation_confidence(sample_size=500, evidence_strength=0.50, p_value=0.0001) == "haute"

    df = pd.DataFrame({
        "id_clear": [f"ID_{i}" for i in range(120)],
        "borderline_cat": [f"cat_{i % 9}" for i in range(120)],
        "typical_cat": [f"cat_{i % 6}" for i in range(120)],
    })
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(df)
    confidences = {row["colonne"]: row["confiance"] for row in analysis.columns}
    assert confidences["id_clear"] == "haute"
    assert confidences["borderline_cat"] == "faible"
    assert confidences["typical_cat"] == "moyenne"

    # Échantillon insuffisant (< 30 observations) -> toujours faible
    df_small = pd.DataFrame({"valeur": np.arange(20, dtype=float)})
    analysis_small = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(df_small)
    assert analysis_small.columns[0]["confiance"] == "faible"


def test_every_recommendation_step_separates_explanation_and_evidence(demo_analysis):
    """Chaque recommandation expose séparément son explication et sa preuve."""

    for path in ["data/churn_telecom_demo.csv", "data/immobilier_demo.csv"]:
        _, res = demo_analysis(path)
        for row in res.columns:
            if row.get("est_cible"):
                continue
            for step, detail in row.get("steps", {}).items():
                if len(detail.get("choices", [])) > 1:
                    exp = detail.get("explanation", "").strip()
                    meas = detail.get("measure", "").strip()
                    assert len(exp) > 0, f"Explication vide pour {row['colonne']} -> {step}"
                    assert len(meas) > 0, f"Mesure vide pour {row['colonne']} -> {step}"
                    evidence = detail.get("evidence", "")
                    assert evidence.startswith(meas)
                    assert " ; seuil : " in evidence
                    assert evidence not in exp



def test_drop_evidence_uses_the_actual_configured_thresholds():
    from app_preprocess import analysis_recommendation_summary
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    config = PreprocessConfig(verbose=0, identifier_unique_threshold=.75,
                              missing_threshold=.75, constant_threshold=.95,
                              leakage_threshold=.80, leakage_min_gap=.20)
    frame = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(40)],
        "constant": ["x"] * 40,
        "mostly_missing": [None] * 32 + list(range(8)),
        "base": np.arange(40),
        "duplicate": np.arange(40),
    })
    result = AutoPreprocessor(config).analyze(frame)
    rows = {row["colonne"]: row for row in result.columns}
    id_proof = analysis_recommendation_summary(rows["customer_id"])["preuve"]
    missing_proof = analysis_recommendation_summary(rows["mostly_missing"])["preuve"]
    constant_proof = analysis_recommendation_summary(rows["constant"])["preuve"]
    duplicate_proof = analysis_recommendation_summary(rows["duplicate"])["preuve"]
    assert "40 / 40 lignes (100.0%)" in id_proof
    assert "75.0%" in id_proof
    assert "75.0%" in missing_proof
    assert "1 valeur unique" in constant_proof
    assert "identique" in duplicate_proof.lower()

    leakage = {"action": "DROP", "type": "DROP_LEAKAGE", "raison": "fuite",
               "drop_preuve": "score fuite CV = 0.910",
               "drop_seuil": ">= 0.800 et ecart >= 0.200",
               "steps": {"type": {"measure": "score fuite CV = 0.910",
                                    "threshold": ">= 0.800 et ecart >= 0.200"}}}
    assert "0.800" in analysis_recommendation_summary(leakage)["preuve"]
    assert "0.200" in analysis_recommendation_summary(leakage)["preuve"]


def test_identifier_target_relations_are_hidden_and_other_methods_are_named(demo_analysis):
    _, result = demo_analysis(DEMO_PATHS[2], target="is_fraud")
    rows = {row["colonne"]: row for row in result.columns}
    for name in ("transaction_id", "client_id"):
        assert pd.isna(rows[name]["relation_cible"])
        assert rows[name]["relation_cible_non_fiable_identifiant"]
        assert rows[name]["relation_cible_methode"] == "non fiable (identifiant)"
    ordinary = [row for row in result.columns if not row.get("est_cible")
                and not row.get("relation_cible_non_fiable_identifiant")]
    assert ordinary
    assert all(pd.notna(row["relation_cible"]) for row in ordinary)
    assert all(row["relation_cible_methode"] for row in ordinary)


def test_analysis_main_table_order_and_parameter_field_visibility():
    from app_preprocess import analysis_table_column_order, step_parameter_visible

    columns = analysis_table_column_order({"colonne", "type", "choix_final", "raison", "confiance",
                                           "manquants_pct", "skew", "relation_cible"})
    assert columns[:6] == ["cocher", "colonne", "type", "choix_final", "raison", "confiance"]
    assert columns.index("choix_final") < columns.index("manquants_pct")
    assert columns.index("raison") < columns.index("skew")
    assert not step_parameter_visible("type", "DROP")
    assert not step_parameter_visible("type", "AUTO")
    assert step_parameter_visible("outliers", "WINSORIZE")


def test_analysis_detail_proof_is_identical_to_primary_recommendation_for_demos(demo_analysis):
    from app_preprocess import (analysis_recommendation_summary, build_analysis_detail,
                                recommendation_step)

    for path in ("data/churn_telecom_demo.csv", "data/immobilier_demo.csv",
                 "data/fraud_signal_demo.csv"):
        _, result = demo_analysis(path, identifier_threshold=.90)
        for row in result.columns:
            if row.get("est_cible"):
                continue
            summary = analysis_recommendation_summary(row)
            step = recommendation_step(row)
            detail = build_analysis_detail(result, row["colonne"], show_all=True)
            panel_row = next(item for item in detail["rows"] if item["etape"] == step)
            assert panel_row["preuve"] == summary["preuve"], (path, row["colonne"])
            assert row["steps"][step]["reason"] == summary["raison"]
            assert not re.search(r"\w+=\S", summary["raison"])
            assert "Garder / Supprimer=" not in summary["raison"]
            assert not any(token in summary["raison"] for token in
                           ("encoding=", "distribution=", "type=", "dates="))


def test_identifier_panel_proof_uses_unique_ratio_and_config_threshold(demo_analysis):
    from app_preprocess import analysis_recommendation_summary, build_analysis_detail

    _, result = demo_analysis(DEMO_PATHS[2], identifier_threshold=.90)
    row = next(item for item in result.columns if item["colonne"] == "client_id")
    summary = analysis_recommendation_summary(row)
    detail = build_analysis_detail(result, "client_id")
    assert summary["preuve"] == detail["rows"][0]["preuve"]
    assert "94.7%" in summary["preuve"]
    assert "90.0%" in summary["preuve"]
    assert "cardinalite" not in summary["preuve"].lower()


def test_analysis_reasons_use_selected_stage_measure_and_date_details(demo_analysis):
    from app_preprocess import analysis_recommendation_summary

    for path in ("data/churn_telecom_demo.csv", "data/immobilier_demo.csv",
                 "data/fraud_signal_demo.csv"):
        _, result = demo_analysis(path)
        for row in result.columns:
            if row.get("est_cible"):
                continue
            reason = analysis_recommendation_summary(row)["raison"]
            assert not re.search(r"\w+=\S", reason)
            assert "Garder / Supprimer" not in reason
            assert "DROP_ID" not in reason and "NUMERIC_CONTINUOUS" not in reason
            if row.get("action") == "DATE_FEATURES":
                assert "heures" in reason.lower() or "plage" in reason.lower()
                assert "parsables" in reason.lower()
                assert "granularite" in reason.lower() or "granularité" in reason.lower()


def test_phase1_source_encoding_and_demo_recommendation_text(demo_analysis):
    from pathlib import Path
    from app_preprocess import analysis_recommendation_summary

    mojibake = re.compile(r"(?:Ã[©-ÿ]|â€|Â\s)")
    for filename in ("auto_preprocessor.py", "app_preprocess.py"):
        source = Path(filename).read_text(encoding="utf-8-sig")
        assert not mojibake.search(source)
        assert "# -*- coding: utf-8 -*-" in source.splitlines()[:2]

    technical = re.compile(r"\b(?:skew|cardinality|normality_p|p_cible|IQR|MAD)=\S", re.I)
    for filename in ("data/churn_telecom_demo.csv", "data/immobilier_demo.csv",
                     "data/fraud_signal_demo.csv"):
        _, result = demo_analysis(filename)
        for row in result.columns:
            summary = analysis_recommendation_summary(row)
            for text in (summary["raison"], summary["preuve"], summary["explication"]):
                assert not mojibake.search(text)
                assert "p 0" not in text.lower()
                assert "plus de" not in text.lower()
                assert not technical.search(text), (filename, row["colonne"], text)


def test_phase1_montant_proof_and_csv_export_keep_unicode_and_p_value(tmp_path, demo_analysis):
    from app_preprocess import analysis_recommendation_summary, export_analysis_csv

    _, result = demo_analysis(DEMO_PATHS[2], target="is_fraud")
    row = next(item for item in result.columns if item["colonne"] == "montant")
    proof = analysis_recommendation_summary(row)["preuve"]
    assert "|skew| > 1" in proof
    assert "p < 0.001" in proof
    output = tmp_path / "analyse_colonnes.csv"
    export_analysis_csv(result, str(output))
    data = output.read_text(encoding="utf-8-sig")
    assert "asym" + chr(0xE9) + "trie" in data
    assert "p < 0.001" in data


def test_phase1_read_table_decodes_utf8_and_cp1252_accents(tmp_path):
    from app_preprocess import read_table

    expected = "Caf" + chr(0xE9)
    for encoding in ("utf-8", "cp1252"):
        path = tmp_path / (encoding + ".csv")
        path.write_bytes(("ville\n" + expected + "\n").encode(encoding))
        assert read_table(str(path)).iloc[0, 0] == expected


def test_phase1_time_only_features_do_not_depend_on_current_date(monkeypatch):
    import auto_preprocessor as module
    from app_preprocess import analysis_recommendation_summary
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    frame = pd.DataFrame({"heure": ["00:00:12", "08:15:00", "16:30:05", "23:59:41"] * 20})
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    row = analysis.columns[0]
    reason = analysis_recommendation_summary(row)["raison"]
    assert row["action"] == "DATE_FEATURES"
    assert "heures 00:00:12" in reason and "23:59:41" in reason
    assert "colonne horaire sans date" in reason

    monkeypatch.setattr(module, "_current_timestamp", lambda tz=None: pd.Timestamp("2001-01-01"))
    first = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    monkeypatch.setattr(module, "_current_timestamp", lambda tz=None: pd.Timestamp("2099-12-31"))
    second = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    pd.testing.assert_frame_equal(first, second)


def test_phase2_recommendation_matches_report_action_on_demo_datasets(demo_analysis):
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    for path in ("data/churn_telecom_demo.csv", "data/immobilier_demo.csv",
                 "data/fraud_signal_demo.csv"):
        frame, analysis = demo_analysis(path)
        prep = AutoPreprocessor(PreprocessConfig(verbose=0))
        prep.fit_transform(frame)
        report = prep.report()["colonnes_detaillees"].set_index("Colonne")
        for row in analysis.columns:
            if row.get("est_cible"):
                continue
            assert report.loc[row["colonne"], "Action appliquée"] == row["action"]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("include_target", [False, True])
def test_phase2_accept_all_recommendations_matches_baseline(task, include_target):
    from auto_preprocessor import AutoPreprocessor, ColumnOverride, PreprocessConfig

    n = 120
    frame = pd.DataFrame({
        "value": np.linspace(1., 25., n) ** 1.4,
        "category": [f"group_{i % 15}" for i in range(n)],
        "small_category": [f"s{i % 4}" for i in range(n)],
        "event_date": pd.date_range("2023-01-01", periods=n).astype(str),
    })
    rng = np.random.default_rng(43)
    y = rng.integers(0, 2, size=n) if task == "classification" else rng.normal(40., 8., size=n)
    if include_target:
        frame["target"] = y
    features = frame.drop(columns=["target"], errors="ignore")
    config = PreprocessConfig(verbose=0)
    analysis = AutoPreprocessor(config, task=task).analyze(
        frame, target="target" if include_target else None)
    overrides = {row["colonne"]: ColumnOverride(row["action"])
                 for row in analysis.columns if not row.get("est_cible")}
    fit_target = y if include_target else None
    baseline = AutoPreprocessor(config, task=task).fit_transform(features, fit_target)
    accepted = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides), task=task).fit_transform(features, fit_target)
    pd.testing.assert_frame_equal(baseline, accepted)


def test_phase2_test_data_cannot_change_train_fitted_statistics():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    train = pd.DataFrame({"amount": [1., 2., np.nan, 4., 5., 7.],
                          "category": ["a", "b", "a", "b", "c", "a"]})
    test_a = pd.DataFrame({"amount": [8., np.nan], "category": ["new", "a"]})
    test_b = pd.DataFrame({"amount": [1e9, -1e9], "category": ["other", "other"]})
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    train_result = prep.fit_transform(train)
    learned_imputer = dict(prep.num_imputer_)
    learned_frequency = {col: values.copy() for col, values in prep.freq_encoding_maps_.items()}
    prep.transform(test_a)
    prep.transform(test_b)
    assert prep.num_imputer_ == learned_imputer
    assert prep.freq_encoding_maps_ == learned_frequency
    pd.testing.assert_frame_equal(train_result, AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(train))


def test_phase2_save_load_transform_keeps_overrides_and_unknown_category_behavior(tmp_path):
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    train = pd.DataFrame({"category": ["a", "b", "a", "c"] * 10,
                          "amount": np.arange(40, dtype=float)})
    test = pd.DataFrame({"category": ["unseen", "a", "other"], "amount": [50., 51., 52.]})
    prep = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"category": {"encoding": "ONEHOT"}, "amount": {"scaling": "ROBUST"}}))
    prep.fit(train)
    expected = prep.transform(test)
    path = tmp_path / "phase2.joblib"
    prep.save(str(path))
    loaded = AutoPreprocessor.load(str(path))
    actual = loaded.transform(test)
    pd.testing.assert_frame_equal(expected, actual)
    assert loaded.config.column_overrides == prep.config.column_overrides
    report = loaded.report()["colonnes_detaillees"].set_index("Colonne")
    assert "choix utilisateur" in report.loc["category", "Choix utilisateur"]
    assert "choix utilisateur" in report.loc["amount", "Choix utilisateur"]


def additional_invariant_frames():
    """Jeux variés indépendants des noms de colonnes et règles métier particulières."""
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
    frames = [loader(as_frame=True).data for loader in (load_wine, load_breast_cancer, load_diabetes)]
    for seed in (310, 311, 312):
        rng = np.random.default_rng(seed)
        size = 160
        continuous = rng.lognormal(mean=1.0, sigma=1.4, size=size)
        continuous[0] = 1e4
        frame = pd.DataFrame({
            "continuous_measure": continuous,
            "discrete_measure": rng.integers(0, 8, size=size),
            "binary_category": rng.choice(["a", "b"], size=size),
            "five_category": rng.choice(list("abcde"), size=size),
            "many_category": rng.choice([f"class_{i}" for i in range(50)], size=size),
            "calendar_values": pd.date_range("2024-01-01", periods=size).astype(str),
            "clock_values": [f"{i % 24:02d}:{i % 60:02d}:00" for i in range(size)],
            "free_text": [f"sample phrase topic {i % 17}" for i in range(size)],
            "sequential_integer": np.arange(size),
            "opaque_text_identifier": [f"C{i:04d}" for i in range(size)],
            "constant_feature": ["same"] * size,
            "empty_feature": [np.nan] * size,
            "partial_missing": [np.nan if i % 5 < 2 else float(i) for i in range(size)],
            "false_missing_markers": ["?" if i % 19 == 0 else "valid" for i in range(size)],
            "numeric_text_measure": [f"{value:.2f} EUR" for value in continuous],
        })
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def shared_invariant_cases(demo_analysis):
    cases = [demo_analysis(path) for path in DEMO_PATHS]
    for frame in additional_invariant_frames():
        result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
        cases.append((frame, result))
    return cases


@pytest.fixture
def invariant_cases(shared_invariant_cases):
    return deepcopy(shared_invariant_cases)


def _display_texts_for_analysis(frame):
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    from app_preprocess import analysis_recommendation_summary

    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    texts = []
    for row in result.columns:
        summary = analysis_recommendation_summary(row)
        texts.extend(summary[key] for key in ("raison", "preuve", "explication"))
        for step in row.get("steps", {}).values():
            texts.extend(step.get(key, "") for key in ("reason", "evidence", "explanation"))
    return result, texts


def test_invariant_c1_recommendation_text_is_single_source_across_outputs(tmp_path, invariant_cases):
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    from app_preprocess import (analysis_recommendation_summary, build_analysis_detail,
                                build_analysis_table, export_analysis_csv, report_text, RunResult)

    forbidden = re.compile(r"Ã[©-ÿ]|â€|Â\s|\w+=\S|[{}\[\]]|\bNone\b|\bnan\b|\bTO_NAN\b|\bDEFAULT_NA_VALUES\b|\bclean_\w+", re.I)
    for frame, result in invariant_cases:
        prep = AutoPreprocessor(PreprocessConfig(verbose=0))
        prep.analysis_result_ = result
        fitted = prep.fit_transform(frame)
        report = prep.report()["colonnes_detaillees"].set_index("Colonne")
        rendered_report = report_text(RunResult(prep, None, fitted, None,
                                                n_rows_in=len(frame), n_cols_in=frame.shape[1]))
        assert not forbidden.search(rendered_report)
        table = build_analysis_table(result)
        for i, row in enumerate(result.columns):
            summary = analysis_recommendation_summary(row)
            assert table.loc[i, "raison"] == summary["raison"]
            assert table.loc[i, "preuve"] == summary["preuve"]
            assert table.loc[i, "explication"] == summary["explication"]
            assert summary["raison"] in rendered_report
            assert summary["preuve"] in rendered_report
            assert summary["explication"] in rendered_report
            if row["colonne"] in report.index:
                assert report.loc[row["colonne"], "Raison recommandée"] == summary["raison"]
                assert report.loc[row["colonne"], "Action appliquée"] == row["action"]
                step_name = next((name for name, detail in row["steps"].items()
                                  if detail.get("reason") == summary["raison"]), None)
                if step_name:
                    assert report.loc[row["colonne"], "Preuve recommandée"] == row["steps"][step_name]["evidence"]
            for text in summary.values():
                assert not forbidden.search(str(text)), (row["colonne"], text)
            detail = build_analysis_detail(result, row["colonne"], show_all=True)
            for step_row in detail["rows"]:
                source_step = row["steps"][step_row["etape"]]
                assert step_row["preuve"] == source_step["evidence"]
                assert step_row["explication"] == source_step["explanation"]
        output = tmp_path / "analysis.csv"
        export_analysis_csv(result, str(output))
        exported = pd.read_csv(output, encoding="utf-8-sig")
        assert exported["Raison"].tolist() == table["raison"].tolist()


def test_invariant_c2_each_decision_exposes_its_evaluated_rule(invariant_cases):
    for _, result in invariant_cases:
        for row in result.columns:
            for step, recommendation in row.get("steps", {}).items():
                if len(recommendation.get("choices", [])) <= 1:
                    continue
                assert recommendation.get("rule"), (row["colonne"], step)
                assert recommendation.get("measure")
                assert recommendation.get("threshold")
                assert recommendation.get("evidence")
                assert isinstance(recommendation["rule"], dict)
                assert recommendation["rule"]["measure"] == recommendation["measure"]
                assert recommendation["rule"]["threshold"] == recommendation["threshold"]
                assert recommendation["rule"]["domain"] == row["type"]


def test_invariant_c3_explanations_are_measure_consistent_and_step_free(invariant_cases):
    from app_preprocess import LABELS

    step_labels = [label for label, _ in LABELS["steps"].values()]
    for _, result in invariant_cases:
        for row in result.columns:
            for step, recommendation in row.get("steps", {}).items():
                explanation = recommendation.get("explanation", "")
                if len(recommendation.get("choices", [])) <= 1:
                    continue
                assert explanation.strip()
                assert recommendation.get("measure", "") not in explanation
                assert recommendation.get("evidence", "") not in explanation
                assert not any(label in explanation for label in step_labels)


def test_invariant_c4_source_and_label_accents_are_preserved():
    from pathlib import Path
    from app_preprocess import LABELS

    for filename in ("auto_preprocessor.py", "app_preprocess.py"):
        source = Path(filename).read_text(encoding="utf-8")
        assert not re.search(r"(?:Ã[©-ÿ]|â€|Â\s)", source)
    all_labels = " ".join(str(item) for group in LABELS.values()
                           for item in group.values())
    for word in ("Tâche", "Données", "Paramètres", "Résultat", "Sélection"):
        assert word in all_labels


def test_invariant_c6_user_column_names_are_not_reformatted():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    from app_preprocess import build_analysis_table

    column = "Étage = {Nord}"
    frame = pd.DataFrame({column: np.arange(40, dtype=float)})
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    table = build_analysis_table(result)
    assert table.loc[0, "colonne"] == column


def test_invariant_c4_no_unaccented_french_words_in_sources():
    """Les mots français courants gardent leur accent circonflexe dans les sources."""
    import re
    from pathlib import Path

    here = Path(__file__).parent
    # « \b » n'accroche pas les identifiants techniques comme tache_detectee (le « _ » est un caractère de mot).
    forbidden = re.compile(r"\b[Tt]aches?\b")
    for name in ("app_preprocess.py", "auto_preprocessor.py"):
        text = (here / name).read_text(encoding="utf-8")
        hits = [m.group(0) for m in forbidden.finditer(text)]
        assert not hits, f"{name} contient un mot sans accent : {hits}"


def test_invariant_c4_task_label_keeps_circumflex():
    from pathlib import Path

    source = (Path(__file__).parent / "app_preprocess.py").read_text(encoding="utf-8")
    assert "Tâche" in source
    

def test_invariant_every_applied_override_is_noted_without_overwrite():
    # pyrefly: ignore[missing-import]
    import pandas as pd, numpy as np
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"num": rng.normal(size=80)})
    df.loc[::9, "num"] = np.nan
    y = pd.Series(rng.integers(0, 2, 80))
    cfg = PreprocessConfig(column_overrides={"num": {"missing": "MEDIAN", "outliers": "WINSORIZE", "scaling": "ROBUST"}})
    pre = AutoPreprocessor(cfg)
    pre.fit_transform(df, y)
    note = pre.user_override_notes_["num"]
    for expected in ("MEDIAN", "WINSORIZE", "ROBUST"):
        assert expected in note, f"override {expected} non consigné : {note}"
    assert note.count("MEDIAN") == 1, f"doublon dans la note : {note}"

# ---------------------------------------------------------------------------
# Règles statistiques de recommandation (une seule règle pour analyze et fit)
# ---------------------------------------------------------------------------

def _cfg():
    return PreprocessConfig(verbose=0)


def test_rule_outliers_none_iqr_winsorize():
    from auto_preprocessor import _recommend_outliers
    rng = np.random.default_rng(1)
    assert _recommend_outliers(pd.Series(np.linspace(0, 1, 200)), _cfg())[0] == "NONE"
    mild = pd.Series(np.append(rng.normal(size=300), 4.2))          # 1 extrême isolé, distribution symétrique
    assert _recommend_outliers(mild, _cfg())[0] == "IQR"
    heavy = pd.Series(rng.lognormal(0, 1.5, 400))                    # queue lourde
    assert _recommend_outliers(heavy, _cfg())[0] == "WINSORIZE"


def test_rule_imputation_mean_median_mode():
    from auto_preprocessor import _recommend_imputation
    rng = np.random.default_rng(2)
    assert _recommend_imputation(pd.Series(rng.normal(50, 5, 500)), _cfg())[0] == "MEAN"
    assert _recommend_imputation(pd.Series(rng.lognormal(0, 1, 500)), _cfg())[0] == "MEDIAN"
    assert _recommend_imputation(pd.Series(rng.integers(0, 4, 500)), _cfg())[0] == "MODE"


def test_rule_distribution_respects_domain_and_gain():
    from auto_preprocessor import _recommend_distribution
    rng = np.random.default_rng(3)
    assert _recommend_distribution(pd.Series(rng.normal(size=500)), _cfg())[0] == "NONE"
    positive = pd.Series(rng.lognormal(0, 1, 500))
    assert _recommend_distribution(positive, _cfg())[0] in {"LOG1P", "SQRT", "YEO_JOHNSON"}
    shifted = pd.Series(rng.lognormal(0, 1, 500) - 2)                  # valeurs négatives
    assert _recommend_distribution(shifted, _cfg())[0] in {"YEO_JOHNSON", "NONE"}
    assert _recommend_distribution(shifted, _cfg())[0] not in {"LOG1P", "SQRT", "BOX_COX"}


def test_recommended_equals_applied_for_numeric_rules():
    from auto_preprocessor import AutoPreprocessor
    rng = np.random.default_rng(4)
    n = 300
    frame = pd.DataFrame({"symetrique": rng.normal(50, 5, n), "revenu": rng.lognormal(3, 1.2, n),
                          "compte": rng.poisson(0.6, n).astype(float)})
    frame.loc[::13, "symetrique"] = np.nan
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    analysis = prep.analyze(frame)
    fitted = AutoPreprocessor(PreprocessConfig(verbose=0)).fit(frame)
    for row in analysis.columns:
        col, steps = row["colonne"], row["steps"]
        assert fitted.column_outliers_[col] == steps["outliers"]["action"], col
        expected_missing = steps["missing"]["action"]
        assert fitted.auto_imputation_[col] == expected_missing, col
        assert fitted.column_distributions_[col] == steps["distribution"]["action"], col


def test_text_transformer_custom_components_output_names():
    from auto_preprocessor import RobustTextTransformer
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"a": [" ".join(rng.choice(words, 6)) for _ in range(80)],
                          "b": [" ".join(rng.choice(words, 6)) for _ in range(80)]})
    tr = RobustTextTransformer(max_features=10, svd_components=4, column_components={"a": 2, "b": 3}).fit(frame)
    out = tr.transform(frame)
    assert list(out.columns) == list(tr.get_feature_names_out())
    assert sum(c.startswith("a_svd_") for c in out.columns) == 2
    assert sum(c.startswith("b_svd_") for c in out.columns) == 3


# ---------------------------------------------------------------------------
# Imputation multivariée (KNN / itérative)
# ---------------------------------------------------------------------------

def _correlated_frame(n=400, missing=0.15, seed=11, monotone=False):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    other = np.exp(base) if monotone else base
    frame = pd.DataFrame({"a": base * 10 + 50, "b": other * 5 + rng.normal(scale=0.3, size=n),
                          "c": rng.normal(size=n)})
    frame.loc[rng.random(n) < missing, "b"] = np.nan
    return frame


def test_rule_recommends_multivariate_imputation_only_with_strong_link():
    from auto_preprocessor import _recommend_imputation
    frame = _correlated_frame()
    assert _recommend_imputation(frame["b"], _cfg(), frame[["a", "c"]])[0] == "ITERATIVE"
    mono = _correlated_frame(monotone=True)
    assert _recommend_imputation(mono["b"], _cfg(), mono[["a", "c"]])[0] in {"KNN", "ITERATIVE"}
    assert _recommend_imputation(frame["b"], _cfg(), frame[["c"]])[0] != "ITERATIVE"     # aucun lien
    low = _correlated_frame(missing=0.02)
    assert _recommend_imputation(low["b"], _cfg(), low[["a", "c"]])[0] not in {"KNN", "ITERATIVE"}  # peu de manquants


@pytest.mark.parametrize("method", ["KNN", "ITERATIVE"])
def test_multivariate_imputation_fills_and_beats_median(method):
    frame = _correlated_frame(missing=0.2)
    truth = _correlated_frame(missing=0.0)
    mask = frame["b"].isna()
    out = AutoPreprocessor(PreprocessConfig(verbose=0, scaler_type="none", outlier_method="none",
                                            column_overrides={"b": {"missing": method, "distribution": "NONE", "scaling": "NONE"}}
                                            )).fit_transform(frame)
    assert not out.isna().any().any()
    median_err = (truth.loc[mask, "b"] - truth["b"].median()).abs().mean()
    imputed_err = (out.loc[mask, "b"] - truth.loc[mask, "b"]).abs().mean()
    assert imputed_err < median_err


def test_multivariate_imputer_is_fitted_on_train_only_and_n_neighbors_param():
    frame = _correlated_frame(missing=0.2)
    train, test = frame.iloc[:300], frame.iloc[300:]
    cfg = PreprocessConfig(verbose=0, column_overrides={"b": {"missing": ("KNN", {"n_neighbors": 3})}})
    prep = AutoPreprocessor(cfg).fit(train)
    imputer, cols, _, _ = prep.multivariate_imputers_["KNN"]
    assert imputer.n_neighbors == 3 and cols == ["b"]
    assert not prep.transform(test).isna().any().any()
    # le train ajusté n'a pas vu le test : changer le test ne change pas l'imputeur
    before = imputer.get_params()
    AutoPreprocessor(cfg).fit(train).transform(test.assign(a=test["a"] * 100))
    assert imputer.get_params() == before


def test_accept_recommendations_with_multivariate_imputation_is_noop():
    from auto_preprocessor import ColumnOverride
    frame = _correlated_frame()
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    assert next(r for r in analysis.columns if r["colonne"] == "b")["steps"]["missing"]["action"] == "ITERATIVE"
    overrides = {r["colonne"]: ColumnOverride({step: d["action"] for step, d in r["steps"].items()
                                              if step in {"missing", "outliers", "distribution"}})
                 for r in analysis.columns}
    accepted = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides)).fit_transform(frame)
    pd.testing.assert_frame_equal(baseline, accepted)


# ---------------------------------------------------------------------------
# Transformations Quantile / Rank / Cube-root et encodages Count / Binary / Hash
# ---------------------------------------------------------------------------

def _skewed_frame(n=300, seed=21):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.lognormal(1, 1.3, n) - 3,                       # négatifs présents
                         "y": rng.normal(size=n)})


@pytest.mark.parametrize("choice", ["CBRT", "QUANTILE", "RANK"])
def test_new_distribution_choices_apply_and_reduce_skew(choice):
    # pyrefly: ignore[missing-import]
    from scipy import stats as sps
    frame = _skewed_frame()
    out = AutoPreprocessor(PreprocessConfig(verbose=0, adaptive_recommendations=False, scaler_type="none",
                                            column_overrides={"x": {"distribution": choice, "scaling": "NONE",
                                                                   "outliers": "NONE"}})).fit_transform(frame)
    assert not out.isna().any().any()
    assert abs(sps.skew(out["x"])) < abs(sps.skew(frame["x"]))
    if choice == "RANK":
        assert out["x"].min() >= 0 and out["x"].max() <= 1


def test_quantile_n_quantiles_param_and_unseen_values():
    frame = _skewed_frame()
    train, test = frame.iloc[:200], frame.iloc[200:].copy()
    test.loc[test.index[0], "x"] = 1e6                                               # valeur hors plage d'entraînement
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={"x": {"distribution": ("QUANTILE", {"n_quantiles": 50})}})).fit(train)
    assert prep.num_power_transformer_["x"].n_quantiles == 50
    assert np.isfinite(prep.transform(test).to_numpy()).all()
    with pytest.raises(ValueError):
        AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={"x": {"distribution": ("QUANTILE", {"n_quantiles": 1})}})).fit(train)


def test_distribution_evaluations_report_validity():
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(_skewed_frame())
    ev = next(r for r in result.columns if r["colonne"] == "x")["steps"]["distribution"]["evaluations"]
    assert ev["BOX_COX"]["valid"] is False and "strictement positives" in ev["BOX_COX"]["reason"]
    assert ev["SQRT"]["valid"] is False and ev["CBRT"]["valid"] and ev["YEO_JOHNSON"]["skew_after"] is not None
    assert ev["QUANTILE"]["information_loss"] != "aucune"


def _cat_frame(n=400, k=20, seed=31):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(k)]
    return pd.DataFrame({"cat": rng.choice(cats, n, p=np.r_[0.3, np.full(k - 1, 0.7 / (k - 1))]),
                         "num": rng.normal(size=n)}), rng.integers(0, 2, n)


@pytest.mark.parametrize("method,prefix", [("COUNT_ENCODE", "cat_count"), ("BINARY_ENCODE", "cat_bin_"), ("HASH_ENCODE", "cat_hash_")])
def test_new_encodings_with_and_without_target_and_unseen_categories(method, prefix):
    frame, y = _cat_frame()
    train, test = frame.iloc[:300].copy(), frame.iloc[300:].copy()
    test.loc[test.index[0], "cat"] = "jamais_vu"
    cfg = PreprocessConfig(verbose=0, noise_guard="off", column_overrides={"cat": {"encoding": method}})
    for target in (None, pd.Series(y[:300], index=train.index)):
        prep = AutoPreprocessor(cfg).fit(train, target)
        out = prep.transform(test)
        assert any(c.startswith(prefix) for c in out.columns), list(out.columns)
        assert not out.isna().any().any()
        assert not any(c.startswith("cat_") and "onehot" in c.lower() for c in out.columns)


def test_encoding_shapes_and_params():
    frame, _ = _cat_frame(k=20)
    base = dict(verbose=0, noise_guard="off", variance_threshold=-1.0, correlation_threshold=2.0)
    binary = AutoPreprocessor(PreprocessConfig(**base, column_overrides={"cat": {"encoding": "BINARY_ENCODE"}})).fit(frame)
    assert binary.custom_encoders_["cat"]["n_bits"] == 5                              # ceil(log2(20+1))
    hashed = AutoPreprocessor(PreprocessConfig(**base, column_overrides={"cat": {"encoding": ("HASH_ENCODE", {"n_components": 4})}})).fit(frame)
    assert hashed.custom_encoders_["cat"]["n_components"] == 4
    with pytest.raises(ValueError):
        AutoPreprocessor(PreprocessConfig(**base, column_overrides={"cat": {"encoding": ("HASH_ENCODE", {"n_components": 0})}})).fit(frame)
    with pytest.raises(ValueError):
        AutoPreprocessor(PreprocessConfig(**base, column_overrides={"num": {"encoding": "COUNT_ENCODE"}})).fit(frame)


def test_encoding_choices_exposed_by_analysis():
    frame, _ = _cat_frame()
    row = next(r for r in AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame).columns if r["colonne"] == "cat")
    assert {"COUNT_ENCODE", "BINARY_ENCODE", "HASH_ENCODE"} <= set(row["steps"]["encoding"]["choices"])


# ---------------------------------------------------------------------------
# Méthodes de traitement des extrêmes : Z-score, Z-score modifié, médiane, conversion en manquant
# ---------------------------------------------------------------------------

def _outlier_frame(n=300, seed=41):
    rng = np.random.default_rng(seed)
    x = rng.normal(100, 10, n)
    x[:3] = [900.0, -700.0, 1500.0]
    return pd.DataFrame({"x": x, "z": rng.normal(size=n)})


def _fit_outliers(frame, spec, **kw):
    cfg = PreprocessConfig(verbose=0, adaptive_recommendations=False, scaler_type="none", noise_guard="off",
                           column_overrides={"x": {"outliers": spec, "distribution": "NONE", "scaling": "NONE"}}, **kw)
    prep = AutoPreprocessor(cfg)
    return prep, prep.fit_transform(frame)


@pytest.mark.parametrize("method", ["ZSCORE", "MODIFIED_ZSCORE", "IQR", "WINSORIZE"])
def test_clipping_methods_bound_extremes(method):
    frame = _outlier_frame()
    prep, out = _fit_outliers(frame, method)
    assert out["x"].max() < 900 and out["x"].min() > -700
    assert out["x"].max() <= prep.winsorizer_.upper_bounds_["x"] + 1e-9
    assert not out.isna().any().any()


def test_replace_median_and_to_nan_behaviour():
    frame = _outlier_frame()
    median = float(frame["x"].median())
    prep, out = _fit_outliers(frame, "REPLACE_MEDIAN")
    assert np.isclose(out["x"].iloc[0], median) and np.isclose(out["x"].iloc[1], median)
    prep2, out2 = _fit_outliers(frame, "OUTLIER_TO_NAN")
    assert not out2.isna().any().any()                       # converti en manquant puis imputé
    assert np.isnan(prep2.winsorizer_.replace_values_["x"])
    clean_median = float(frame["x"].iloc[3:].median())
    assert abs(out2["x"].iloc[0] - clean_median) < 5         # imputé par la médiane du train


def test_outlier_method_parameters_change_bounds():
    frame = _outlier_frame()
    tight, _ = _fit_outliers(frame, ("ZSCORE", {"threshold": 1.5}))
    wide, _ = _fit_outliers(frame, ("ZSCORE", {"threshold": 5.0}))
    assert tight.winsorizer_.upper_bounds_["x"] < wide.winsorizer_.upper_bounds_["x"]
    narrow_iqr, _ = _fit_outliers(frame, ("IQR", {"multiplier": 0.5}))
    wide_iqr, _ = _fit_outliers(frame, ("IQR", {"multiplier": 3.0}))
    assert narrow_iqr.winsorizer_.upper_bounds_["x"] < wide_iqr.winsorizer_.upper_bounds_["x"]
    with pytest.raises(ValueError):
        _fit_outliers(frame, ("ZSCORE", {"threshold": 0}))
    with pytest.raises(ValueError):
        _fit_outliers(frame, ("IQR", {"multiplier": -1}))


def test_outlier_bounds_learned_on_train_only():
    frame = _outlier_frame()
    train, test = frame.iloc[:200], frame.iloc[200:].copy()
    test.loc[test.index[0], "x"] = 5e5
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, adaptive_recommendations=False, scaler_type="none", noise_guard="off",
                                             column_overrides={"x": {"outliers": "MODIFIED_ZSCORE", "distribution": "NONE", "scaling": "NONE"}})).fit(train)
    upper = prep.winsorizer_.upper_bounds_["x"]
    assert prep.transform(test)["x"].max() <= upper + 1e-9 < 5e5


def test_modified_zscore_without_dispersion_flags_nothing():
    from auto_preprocessor import _outlier_bounds
    low, high = _outlier_bounds(pd.Series([5.0] * 50 + [6.0]), "MODIFIED_ZSCORE", PreprocessConfig(verbose=0))
    assert (low, high) == (-np.inf, np.inf)


def test_outlier_methods_evidence_and_choices_in_analysis():
    row = next(r for r in AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(_outlier_frame()).columns if r["colonne"] == "x")
    step = row["steps"]["outliers"]
    assert {"ZSCORE", "MODIFIED_ZSCORE", "REPLACE_MEDIAN", "OUTLIER_TO_NAN"} <= set(step["choices"])
    methods = step["evaluations"]["methods"]
    assert methods["IQR"]["outlier_count"] >= 3 and methods["ZSCORE"]["upper"] > methods["MODIFIED_ZSCORE"]["upper"]
    assert "erreur de saisie" in step["evaluations"]["nature"]
    clean = next(r for r in AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(
        pd.DataFrame({"x": np.linspace(0, 1, 100)})).columns if r["colonne"] == "x")
    assert clean["steps"]["outliers"]["evaluations"]["nature"] == "aucun extrême détecté"


# ---------------------------------------------------------------------------
# Aperçu avant / après
# ---------------------------------------------------------------------------

def _preview_frame(n=300, seed=51):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.lognormal(3, 1, n), "cat": rng.choice(list("abcde"), n), "y": rng.integers(0, 2, n)})
    frame.loc[::10, "x"] = np.nan
    return frame


def test_preview_numeric_before_after_stats_and_no_mutation():
    from auto_preprocessor import preview_column
    frame = _preview_frame()
    snapshot = frame.copy(deep=True)
    p = preview_column(frame, "x", {"distribution": "LOG1P", "outliers": "IQR", "scaling": "STANDARD"})
    pd.testing.assert_frame_equal(frame, snapshot)                       # données d'origine intactes
    assert p["status"] == "ok" and p["rows"] == 300
    assert p["before"]["count"] == 300 and np.isclose(p["before"]["missing_pct"], 10.0)   # aucune ligne retirée
    assert abs(p["after"]["skew"]) < abs(p["before"]["skew"])
    assert p["after"]["missing_pct"] == 0 and np.isclose(p["after_scaled"]["std"], 1.0, atol=0.02)
    assert p["affected"]["missing_filled"] == 30 and p["affected"]["outside_bounds"] > 0
    assert p["applied"]["distribution"] == "LOG1P" and p["applied"]["outliers"] == "IQR"
    assert set(p["data"]) == {"before", "after", "after_scaled"}


def test_preview_matches_real_pipeline_output():
    from auto_preprocessor import preview_column
    frame = _preview_frame()
    override = {"distribution": "SQRT", "scaling": "ROBUST", "outliers": "WINSORIZE"}
    p = preview_column(frame, "x", override)
    real = AutoPreprocessor(PreprocessConfig(verbose=0, noise_guard="off",
                                             column_overrides={"x": {**override, "duplicates": "KEEP"}})).fit_transform(frame[["x"]])
    assert np.isclose(p["after_scaled"]["mean"], real["x"].mean()) and np.isclose(p["after_scaled"]["std"], real["x"].std(), rtol=1e-2)


def test_preview_default_uses_recommendation_and_categorical_summary():
    from auto_preprocessor import preview_column
    frame = _preview_frame()
    auto = preview_column(frame, "x")
    assert auto["applied"]["distribution"] in {"NONE", "LOG1P", "SQRT", "YEO_JOHNSON", "QUANTILE"}
    cat = preview_column(frame, "cat", "ONEHOT")
    assert cat["before"]["n_categories"] == 5 and cat["after"]["n_columns"] >= 4
    count = preview_column(frame, "cat", {"encoding": "COUNT_ENCODE"}, target="y")
    assert count["output_columns"] == ["cat_count"]


def test_preview_multivariate_imputation_and_errors():
    from auto_preprocessor import preview_column
    p = preview_column(_correlated_frame(missing=0.2), "b", {"missing": "ITERATIVE"})
    assert p["status"] == "ok" and p["after"]["missing_pct"] == 0
    with pytest.raises(ValueError):
        preview_column(_preview_frame(), "inconnue")
    with pytest.raises(ValueError):
        preview_column(_preview_frame(), "y", target="y")


def test_preview_dropped_column_reports_status():
    from auto_preprocessor import preview_column
    frame = _preview_frame().assign(constante=1)
    p = preview_column(frame, "constante")
    assert p["status"] == "dropped" and p["output_columns"] == []


def test_preview_table_figure_and_png_export(tmp_path):
    from app_preprocess import build_preview_table, build_preview_figure, export_preview_png, preview_for_choice
    from auto_preprocessor import preview_column
    frame = _preview_frame()
    numeric = preview_column(frame, "x", {"distribution": "LOG1P"})
    table = build_preview_table(numeric)
    assert list(table.columns) == ["statistique", "avant", "après traitement", "après échelle"]
    assert "Lignes modifiées" in set(table["statistique"])
    categorical = build_preview_table(preview_column(frame, "cat", "ONEHOT"))
    assert "Colonnes de sortie" in set(categorical["statistique"])
    for preview, name in ((numeric, "num.png"), (preview_column(frame, "cat", "ONEHOT"), "cat.png")):
        export_preview_png(preview, str(tmp_path / name))
        assert (tmp_path / name).stat().st_size > 1000
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target="y")
    viaChoice = preview_for_choice(frame, result, "x", {"x": {"distribution": "CBRT"}}, PreprocessConfig(verbose=0), target="y")
    assert viaChoice["applied"]["distribution"] == "CBRT"

# ---------------------------------------------------------------------------
# Leave-One-Out encoding
# ---------------------------------------------------------------------------

def _loo_frame(n=300, seed=61):
    rng = np.random.default_rng(seed)
    cat = rng.choice([f"c{i}" for i in range(12)], n)
    y = (rng.random(n) < np.where(np.isin(cat, ["c1", "c2"]), 0.8, 0.2)).astype(int)
    return pd.DataFrame({"cat": cat, "num": rng.normal(size=n)}), y


def _loo_config(spec="LOO_ENCODE"):
    return PreprocessConfig(verbose=0, noise_guard="off", variance_threshold=-1.0, correlation_threshold=2.0,
                            column_overrides={"cat": {"encoding": spec}})


def test_loo_train_values_exclude_own_label_and_test_uses_full_mean():
    frame, y = _loo_frame()
    train, test = frame.iloc[:200].copy(), frame.iloc[200:].copy()
    prep = AutoPreprocessor(_loo_config())
    out = prep.fit_transform(train, pd.Series(y[:200]))
    assert "cat_loo" in out.columns
    data = pd.DataFrame({"c": train["cat"].to_numpy(), "y": y[:200]})
    for i in (0, 5, 17, 120):
        others = data[data["c"] == data["c"].iloc[i]].drop(data.index[i])["y"]
        assert np.isclose(out["cat_loo"].iloc[i], others.mean(), atol=1e-6)
    test.iloc[0, 0] = "jamais_vu"
    encoded = prep.transform(test)["cat_loo"]
    state = prep.custom_encoders_["cat"]
    assert np.isclose(encoded.iloc[0], state["global_mean"])
    known = test["cat"].iloc[1]
    sums, counts = state["sums"][known], state["counts"][known]
    assert np.isclose(encoded.iloc[1], sums / counts, atol=1e-6)


def test_loo_singleton_category_gets_global_mean_and_state_is_train_only():
    frame, y = _loo_frame()
    train = frame.iloc[:200].copy()
    train.iloc[0, 0] = "rare_singleton"
    prep = AutoPreprocessor(_loo_config())
    out = prep.fit_transform(train, pd.Series(y[:200]))
    assert np.isclose(out["cat_loo"].iloc[0], prep.custom_encoders_["cat"]["global_mean"])
    before = dict(prep.custom_encoders_["cat"]["sums"])
    prep.transform(frame.iloc[200:])
    assert prep.custom_encoders_["cat"]["sums"] == before


def test_loo_validations_and_sigma_parameter():
    frame, y = _loo_frame()
    with pytest.raises(ValueError, match="exige une cible"):
        AutoPreprocessor(_loo_config()).fit(frame)
    with pytest.raises(ValueError, match="binaire ou continue"):
        AutoPreprocessor(_loo_config()).fit(frame, pd.Series(np.random.default_rng(0).integers(0, 3, len(frame))))
    with pytest.raises(ValueError, match="sigma"):
        AutoPreprocessor(_loo_config(("LOO_ENCODE", {"sigma": -1}))).fit(frame, pd.Series(y))
    with pytest.raises(ValueError, match="incompatible"):
        AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={"num": {"encoding": "LOO_ENCODE"}})).fit(frame, pd.Series(y))
    plain = AutoPreprocessor(_loo_config()).fit_transform(frame, pd.Series(y))["cat_loo"].to_numpy()
    noisy = AutoPreprocessor(_loo_config(("LOO_ENCODE", {"sigma": 0.5}))).fit_transform(frame, pd.Series(y))["cat_loo"].to_numpy()
    assert not np.allclose(plain, noisy)
    noisy_again = AutoPreprocessor(_loo_config(("LOO_ENCODE", {"sigma": 0.5}))).fit_transform(frame, pd.Series(y))["cat_loo"].to_numpy()
    np.testing.assert_allclose(noisy, noisy_again)           # reproductible (random_state)


def test_loo_regression_target_and_save_load(tmp_path):
    frame, _ = _loo_frame()
    target = pd.Series(np.random.default_rng(5).normal(10, 3, len(frame)) + frame["cat"].str[1:].astype(int))
    prep = AutoPreprocessor(_loo_config()).fit(frame.iloc[:200], target.iloc[:200])
    expected = prep.transform(frame.iloc[200:])
    path = tmp_path / "loo.joblib"
    prep.save(str(path))
    pd.testing.assert_frame_equal(expected, AutoPreprocessor.load(str(path)).transform(frame.iloc[200:]))


def test_loo_exposed_by_analysis_and_gui_parameters():
    from app_preprocess import parse_step_parameters, step_parameter_visible, pedagogical_explanation
    frame, _ = _loo_frame()
    row = next(r for r in AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame).columns if r["colonne"] == "cat")
    assert "LOO_ENCODE" in row["steps"]["encoding"]["choices"]
    assert parse_step_parameters("encoding", "LOO_ENCODE", "0.2", "CATEGORICAL_LOW_CARD") == {"sigma": 0.2}
    assert parse_step_parameters("encoding", "HASH_ENCODE", "4", "CATEGORICAL_LOW_CARD") == {"n_components": 4}
    with pytest.raises(ValueError):
        parse_step_parameters("encoding", "LOO_ENCODE", "-1", "CATEGORICAL_LOW_CARD")
    with pytest.raises(ValueError):
        parse_step_parameters("encoding", "HASH_ENCODE", "0", "CATEGORICAL_LOW_CARD")
    assert step_parameter_visible("encoding", "LOO_ENCODE") and step_parameter_visible("encoding", "HASH_ENCODE")
    assert "cible" in pedagogical_explanation("encoding", "LOO_ENCODE", "m", "t")


def test_loo_preview_uses_target():
    from auto_preprocessor import preview_column
    frame, y = _loo_frame()
    frame["y"] = y
    preview = preview_column(frame, "cat", {"encoding": "LOO_ENCODE"}, target="y")
    assert preview["status"] == "ok" and preview["output_columns"] == ["cat_loo"]


# ---------------------------------------------------------------------------
# Texte : TF-IDF / comptage / hachage avec n-grammes
# ---------------------------------------------------------------------------

def _text_frame(n=200, seed=71):
    rng = np.random.default_rng(seed)
    words = "red blue green fast slow quick model data science deep learning forest tree market price risk".split()
    return pd.DataFrame({"txt": [" ".join(rng.choice(words, 6)) for _ in range(n)], "num": rng.normal(size=n)}), \
        rng.integers(0, 2, n)


def _text_config(spec):
    return PreprocessConfig(verbose=0, noise_guard="off", variance_threshold=-1.0, correlation_threshold=2.0,
                            column_overrides={"txt": {"text": spec}})


@pytest.mark.parametrize("spec,prefix,width", [
    (("TFIDF_NGRAMS", {"ngram_range": (1, 2), "max_features": 12}), "txt_tfidf_", 12),
    (("COUNT_VECTORIZER", {"ngram_range": [1, 1], "max_features": 8}), "txt_cnt_", 8),
    (("HASH_TEXT", {"ngram_range": (1, 2), "n_features": 6}), "txt_texthash_", 6),
])
def test_text_vectorizers_shapes_unseen_text_and_train_only_vocabulary(spec, prefix, width):
    frame, y = _text_frame()
    train, test = frame.iloc[:150].copy(), frame.iloc[150:].copy()
    prep = AutoPreprocessor(_text_config(spec))
    out = prep.fit_transform(train, pd.Series(y[:150]))
    assert sum(c.startswith(prefix) for c in out.columns) == width
    test["txt"] = "totally unseen words appear here"
    out_test = prep.transform(test)
    assert list(out_test.columns) == list(out.columns)
    assert np.isfinite(out_test.to_numpy()).all()
    assert not any(c.startswith("txt_") and c not in out.columns for c in out_test.columns)
    if spec[0] == "HASH_TEXT":
        assert (out_test[[c for c in out.columns if c.startswith(prefix)]].to_numpy().sum(axis=1) > 0).all()


def test_count_vectorizer_counts_occurrences_and_ngrams_include_bigrams():
    frame = pd.DataFrame({"txt": [f"alpha beta alpha gamma delta epsilon w{i % 12}" for i in range(40)]
                          + [f"zeta eta theta iota kappa lambda w{i % 12}" for i in range(40)]})
    out = AutoPreprocessor(_text_config(("COUNT_VECTORIZER", {"ngram_range": (1, 1), "max_features": 20}))).fit_transform(frame)
    assert set(out["txt_cnt_alpha"].unique()) == {0.0, 2.0}
    bigrams = AutoPreprocessor(_text_config(("TFIDF_NGRAMS", {"ngram_range": (2, 2), "max_features": 30}))).fit_transform(frame)
    assert "txt_tfidf_alpha_beta" in bigrams.columns


def test_text_vectorizer_validations_and_gui_parsing():
    from app_preprocess import parse_step_parameters, step_parameter_visible
    frame, y = _text_frame()
    with pytest.raises(ValueError, match="ngram_range"):
        AutoPreprocessor(_text_config(("TFIDF_NGRAMS", {"ngram_range": (3, 1)}))).fit(frame)
    with pytest.raises(ValueError, match="n_features"):
        AutoPreprocessor(_text_config(("HASH_TEXT", {"n_features": 0}))).fit(frame)
    with pytest.raises(ValueError, match="incompatible"):
        AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={"num": {"text": "TFIDF_NGRAMS"}})).fit(frame)
    assert parse_step_parameters("text", "TFIDF_NGRAMS", "1, 2, 20", "TEXT") == {"ngram_range": [1, 2], "max_features": 20}
    assert parse_step_parameters("text", "HASH_TEXT", "1, 3, 32", "TEXT") == {"ngram_range": [1, 3], "n_features": 32}
    assert parse_step_parameters("text", "COUNT_VECTORIZER", "", "TEXT") is None
    for bad in ("3, 1", "0, 2", "1, 9", "a, b", "1"):
        with pytest.raises(ValueError):
            parse_step_parameters("text", "TFIDF_NGRAMS", bad, "TEXT")
    assert all(step_parameter_visible("text", a) for a in ("TFIDF_NGRAMS", "COUNT_VECTORIZER", "HASH_TEXT"))


def test_text_vectorizers_listed_by_analysis_and_default_unchanged(tmp_path):
    frame, y = _text_frame()
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    row = next(r for r in analysis.columns if r["colonne"] == "txt")
    assert {"TFIDF_NGRAMS", "COUNT_VECTORIZER", "HASH_TEXT"} <= set(row["steps"]["text"]["choices"])
    assert row["steps"]["text"]["action"] == "TFIDF_SVD"
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    assert any(c.startswith("txt_") for c in baseline.columns)
    prep = AutoPreprocessor(_text_config(("TFIDF_NGRAMS", {"ngram_range": (1, 2), "max_features": 10}))).fit(frame)
    path = tmp_path / "text.joblib"
    prep.save(str(path))
    pd.testing.assert_frame_equal(prep.transform(frame), AutoPreprocessor.load(str(path)).transform(frame))


# ---------------------------------------------------------------------------
# Diagnostic du mécanisme de manque (MCAR / MAR / MNAR)
# ---------------------------------------------------------------------------

def _missingness_frame(n=600, seed=81):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    income = rng.normal(50, 12, n)
    mcar = income.copy()
    mcar[rng.random(n) < 0.2] = np.nan
    mar = income.copy()
    mar[rng.random(n) < 0.7 / (1 + np.exp(-(age - 48) / 2))] = np.nan
    return pd.DataFrame({"age": age, "group": rng.choice(list("abc"), n), "income_mcar": mcar, "income_mar": mar,
                         "rare_missing": [np.nan if i < 3 else float(i) for i in range(n)]})


def test_missingness_diagnostic_distinguishes_mcar_compatible_from_mar():
    rows = {r["colonne"]: r for r in AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(_missingness_frame()).columns}
    mcar, mar = rows["income_mcar"]["diagnostic_manquants"], rows["income_mar"]["diagnostic_manquants"]
    assert mcar["verdict"] == "MCAR_COMPATIBLE" and "MCAR n'est pas prouvé" in mcar["texte"]
    assert mar["verdict"] == "MAR_PLAUSIBLE" and mar["top"][0]["colonne"] == "age"
    assert mar["top"][0]["p_adj"] < 0.05 and mar["top"][0]["effet"] >= 0.2
    assert "multivariée" in mar["implication"]
    assert rows["age"]["diagnostic_manquants"]["verdict"] == "AUCUN" and rows["age"]["diagnostic_manquants_texte"] == ""
    assert rows["rare_missing"]["diagnostic_manquants"]["verdict"] == "INDETERMINE"


def test_missingness_diagnostic_never_claims_mnar_and_does_not_change_recommendations():
    frame = _missingness_frame()
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    from auto_preprocessor import MNAR_NOTE
    for row in result.columns:
        text = row["diagnostic_manquants_texte"]
        if text:
            assert MNAR_NOTE in text and "MNAR confirmé" not in text
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    overrides = {r["colonne"]: {step: d["action"] for step, d in r["steps"].items()
                                if step in {"missing", "outliers", "distribution"}} for r in result.columns}
    accepted = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides)).fit_transform(frame)
    pd.testing.assert_frame_equal(baseline, accepted)


def test_missingness_diagnostic_uses_target_and_categorical_columns():
    rng = np.random.default_rng(9)
    n = 500
    y = rng.integers(0, 2, n)
    value = rng.normal(size=n)
    value[(y == 1) & (rng.random(n) < 0.5)] = np.nan
    cat = rng.choice(list("xyz"), n)
    value2 = rng.normal(size=n)
    value2[(cat == "x") & (rng.random(n) < 0.6)] = np.nan
    frame = pd.DataFrame({"value": value, "cat": cat, "value2": value2, "y": y})
    rows = {r["colonne"]: r["diagnostic_manquants"] for r in
            AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target="y").columns if not r.get("est_cible")}
    assert rows["value"]["verdict"] == "MAR_PLAUSIBLE" and rows["value"]["top"][0]["colonne"] == "y"
    assert rows["value2"]["verdict"] == "MAR_PLAUSIBLE" and rows["value2"]["top"][0]["colonne"] == "cat"
    assert rows["value2"]["top"][0]["test"] == "χ²"


def test_missingness_diagnostic_shown_in_detail_and_csv_export(tmp_path):
    from app_preprocess import build_analysis_detail, export_analysis_csv
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(_missingness_frame())
    detail = build_analysis_detail(result, "income_mar")
    assert "MAR plausible" in detail["diagnostic_manquants"]
    assert build_analysis_detail(result, "age")["diagnostic_manquants"] == ""
    assert build_analysis_detail(result, None)["diagnostic_manquants"] == ""
    path = tmp_path / "analysis.csv"
    export_analysis_csv(result, str(path))
    exported = pd.read_csv(path, encoding="utf-8-sig").set_index("Colonne")
    assert "MAR plausible" in exported.loc["income_mar", "Diagnostic des manquants"]


def test_diagnose_missingness_threshold_parameter_controls_verdict():
    from auto_preprocessor import diagnose_missingness, _missingness_pool
    frame = _missingness_frame()
    pool = _missingness_pool(frame[["age", "group"]])
    strict = diagnose_missingness(frame["income_mar"], pool, PreprocessConfig(verbose=0, missingness_min_effect=0.99))
    default = diagnose_missingness(frame["income_mar"], pool, PreprocessConfig(verbose=0))
    assert default["verdict"] == "MAR_PLAUSIBLE" and strict["verdict"] == "MCAR_COMPATIBLE"
