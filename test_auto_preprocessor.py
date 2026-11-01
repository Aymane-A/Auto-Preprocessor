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


def test_analysis_demo_recommendations():
    from auto_preprocessor import AutoPreprocessor
    from app_preprocess import read_table

    prep = AutoPreprocessor()
    churn = prep.analyze(read_table("data/churn_telecom_demo.csv")).to_dataframe().set_index("colonne")
    assert churn.loc["facture_mensuelle", "type"] == "NUMERIC_CONTINUOUS"
    fraud = prep.analyze(read_table("data/fraud_signal_demo.csv")).to_dataframe().set_index("colonne")
    assert fraud.loc["heure", "action"] == "DATE_FEATURES"
    housing = prep.analyze(read_table("data/immobilier_demo.csv")).to_dataframe().set_index("colonne")
    assert housing.loc["prix_dh", "action"] == "YEO_JOHNSON"


@pytest.mark.parametrize("path", ["data/churn_telecom_demo.csv", "data/immobilier_demo.csv", "data/fraud_signal_demo.csv"])
def test_accept_analysis_recommendations_is_noop(path):
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig, ColumnOverride
    from app_preprocess import read_table

    frame = read_table(path)
    baseline = AutoPreprocessor(PreprocessConfig(verbose=0)).fit_transform(frame)
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
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


def test_identifier_name_is_dropped_and_explained():
    from auto_preprocessor import AutoPreprocessor
    from app_preprocess import read_table

    frame = read_table("data/fraud_signal_demo.csv")
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame).to_dataframe().set_index("colonne")
    assert result.loc["client_id", "action"] == "DROP"
    assert "ratio uniques/lignes" in result.loc["client_id", "raison"]


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
    clipped = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"age": {"clean_impossible": "CLIP"}})).fit_transform(frame)
    replaced = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"age": {"clean_impossible": "NAN"}})).fit_transform(frame)
    ignored = AutoPreprocessor(PreprocessConfig(verbose=0,
        column_overrides={"age": {"clean_impossible": "IGNORE"}})).fit_transform(frame)
    dropped_rows = AutoPreprocessor(PreprocessConfig(verbose=0,
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
    assert row["steps"]["clean_missing_tokens"]["measure"].startswith("cellules=3")
    summary = build_preprocess_summary(result, {"value": {"type": "DROP"}})
    assert summary["dropped_columns"] == ["value"]
    assert summary["user_columns"] == ["value"]


def test_churn_target_relations_are_populated_for_every_feature():
    from app_preprocess import build_analysis_table, read_table

    frame = read_table("data/churn_telecom_demo.csv")
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target="churn")
    table = result.to_dataframe()
    target_row = table.loc[table["colonne"] == "churn"].iloc[0]
    features = table.loc[table["colonne"] != "churn"]
    assert target_row["est_cible"]
    assert target_row["type"] == "TARGET"
    assert target_row["steps"] == {}
    assert features["relation_cible"].notna().all()
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

    frame = pd.DataFrame({"price": np.r_[np.arange(1., 60.), 500., 900.]})
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    empty = build_analysis_detail(result, None)
    assert "Sélectionnez une colonne" in empty["header"]
    detail = build_analysis_detail(result, "price", {"price": {"distribution": "NONE"}})
    assert "price | type=" in detail["header"]
    assert any(row["etape"] == "distribution" and row["modifie"] and
               row["recommandation"] == "YEO_JOHNSON" for row in detail["rows"])
    assert all(row["explication"] and row["preuve"].split(" ; seuil : ")[0] in row["explication"]
               for row in detail["rows"])


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
        assert all(item["explication"].strip() and item["preuve"].split(" ; seuil : ")[0] in item["explication"]
                   for item in details)
        for step, recommendation in row["steps"].items():
            if len(recommendation.get("choices", [])) > 1:
                assert recommendation["explanation"].strip()
                assert recommendation["measure"].strip()
                assert recommendation["measure"] in recommendation["explanation"]
    table = build_analysis_table(result)
    assert table["raison"].str.len().gt(0).all()
    assert table["preuve"].str.len().gt(0).all()
    path = tmp_path / "analyse_colonnes.csv"
    export_analysis_csv(result, str(path))
    exported = pd.read_csv(path)
    assert {"raison", "preuve", "explication", "explication_distribution"}.issubset(exported.columns)
    assert exported.loc[exported["colonne"] == "amount", "explication_distribution"].notna().all()


def test_target_provided_populates_relation_cible_for_all_features():
    """Test (a) : cible fournie -> relation_cible rempli pour toutes les features."""
    from app_preprocess import build_analysis_table, read_table
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    frame = read_table("data/churn_telecom_demo.csv")
    result = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target="churn")
    table = result.to_dataframe()
    target_row = table.loc[table["colonne"] == "churn"].iloc[0]
    features = table.loc[table["colonne"] != "churn"]

    assert target_row["est_cible"]
    assert target_row["type"] == "TARGET"
    assert pd.isna(target_row["relation_cible"])
    assert features["relation_cible"].notna().all()
    assert (features["relation_cible"] >= 0).all()

    # Vérification que Cramér's V, Spearman ou MI est présent pour chaque feature
    has_stat = (
        features["cramer_cible"].notna()
        | features["spearman_cible"].notna()
        | features["information_mutuelle_cible"].notna()
    )
    assert has_stat.all()

    display = build_analysis_table(result).set_index("colonne")
    assert display.loc["churn", "état_choix"] == "cible"
    assert display.loc["churn", "choix_final"] == "cible"


def test_facture_mensuelle_stats_match_independent_pandas():
    """Test (b) : facture_mensuelle : stats identiques à un calcul pandas indépendant."""
    from app_preprocess import read_table
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    df = read_table("data/churn_telecom_demo.csv")
    cleaned_vals = df["facture_mensuelle"].map(AutoPreprocessor._clean_numeric_str_value).dropna()
    expected_skew = float(cleaned_vals.skew())
    expected_min = float(cleaned_vals.min())
    expected_max = float(cleaned_vals.max())
    expected_mean = float(cleaned_vals.mean())
    q1, q3 = cleaned_vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    expected_outliers_pct = float(((cleaned_vals < q1 - 1.5 * iqr) | (cleaned_vals > q3 + 1.5 * iqr)).mean() * 100)

    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    res = prep.analyze(df, target="churn")
    row = next(r for r in res.columns if r["colonne"] == "facture_mensuelle")

    assert row["type"] == "NUMERIC_CONTINUOUS"
    assert row["skew"] == pytest.approx(expected_skew, abs=1e-5)
    assert row["outliers_pct"] == pytest.approx(expected_outliers_pct, abs=1e-5)
    assert row["min"] == pytest.approx(expected_min, abs=1e-5)
    assert row["max"] == pytest.approx(expected_max, abs=1e-5)
    assert row["mean"] == pytest.approx(expected_mean, abs=1e-5)


def test_analysis_formatting_no_nan_headers_and_sorting():
    """Test (c) : formatage : pas de 'nan', unités dans les en-têtes, tri correct."""
    from app_preprocess import analysis_sort_key, build_analysis_table, format_analysis_value, read_table
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    assert format_analysis_value("skew", np.nan) == "—"
    assert format_analysis_value("relation_cible", float("nan")) == "—"
    assert format_analysis_value("manquants_pct", np.nan) == "—"
    assert format_analysis_value("colonne", None) == "—"
    assert format_analysis_value("colonne", "") == "—"
    assert format_analysis_value("colonne", "nan") == "—"
    assert format_analysis_value("manquants_pct", 12.345) == "12.3%"
    assert format_analysis_value("outliers_pct", 0.0) == "0.0%"
    assert format_analysis_value("skew", 1.23456) == "1.23"

    df = read_table("data/churn_telecom_demo.csv")
    res = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(df, target="churn")
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


def test_every_recommendation_step_has_nonempty_explanation_with_measure():
    """Test (e) : chaque recommandation (colonne x étape) a une explication non vide avec la mesure."""
    from app_preprocess import read_table
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig

    for path in ["data/churn_telecom_demo.csv", "data/immobilier_demo.csv"]:
        df = read_table(path)
        res = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(df)
        for row in res.columns:
            if row.get("est_cible"):
                continue
            for step, detail in row.get("steps", {}).items():
                if len(detail.get("choices", [])) > 1:
                    exp = detail.get("explanation", "").strip()
                    meas = detail.get("measure", "").strip()
                    assert len(exp) > 0, f"Explication vide pour {row['colonne']} -> {step}"
                    assert len(meas) > 0, f"Mesure vide pour {row['colonne']} -> {step}"
                    assert meas in exp or meas.split(";")[0] in exp, (
                        f"Mesure '{meas}' non trouvée dans explication '{exp}' ({row['colonne']} -> {step})"
                    )

