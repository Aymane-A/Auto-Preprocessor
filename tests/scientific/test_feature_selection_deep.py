"""Comprehensive deep audit and validation of all feature selection methods and edge cases."""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, VarianceThreshold,
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SequentialFeatureSelector, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from feature_selection import fit_selector, METHODS, SUPERVISED, WRAPPERS
from auto_preprocessor import AutoPreprocessor, PreprocessConfig


# =============================================================================
# 1. FIXTURES
# =============================================================================

@pytest.fixture
def binary_classification_data():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    const = np.ones(n)
    y = ((x1 + 0.5 * x2) > 0).astype(int)
    df = pd.DataFrame({
        "signal1": x1,
        "signal2": x2,
        "noise1": noise1,
        "noise2": noise2,
        "const": const,
    })
    return df, y


@pytest.fixture
def multiclass_classification_data():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    # 3 balanced classes
    y = np.where(x1 > 0.5, 0, np.where(x2 > 0.0, 1, 2))
    df = pd.DataFrame({
        "signal1": x1,
        "signal2": x2,
        "noise": noise,
    })
    return df, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    const = np.zeros(n)
    y = 3.0 * x1 - 2.0 * x2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({
        "signal1": x1,
        "signal2": x2,
        "noise": noise,
        "const": const,
    })
    return df, y


@pytest.fixture
def nonnegative_classification_data():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.uniform(0.1, 10.0, size=n)
    x2 = rng.uniform(0.1, 5.0, size=n)
    noise = rng.uniform(0.1, 2.0, size=n)
    y = (x1 > 5.0).astype(int)
    df = pd.DataFrame({
        "feat1": x1,
        "feat2": x2,
        "noise": noise,
    })
    return df, y


# =============================================================================
# 2. CLASSIFICATION TESTS ACROSS ALL METHODS
# =============================================================================

@pytest.mark.parametrize("method", [
    "variance", "kbest", "anova", "mutual_info", "percentile",
    "rfe", "rfecv", "forward", "backward", "l1", "elastic_net",
    "select_from_model", "random_forest"
])
def test_all_methods_binary_classification(binary_classification_data, method):
    df, y = binary_classification_data
    # Filter out const for methods that might fail with constant or test filtering
    selector, report = fit_selector(
        df, y, method=method, task="classification", count=2,
        percentile=40, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert report is not None
    assert report["method"] == method
    assert report["fit_rows"] == len(df)
    assert report["selected_count"] > 0
    assert report["selected_count"] <= df.shape[1]
    assert len(report["selected"]) == report["selected_count"]
    assert len(report["rejected"]) == df.shape[1] - report["selected_count"]
    # Check that returned selected features match get_support()
    mask = selector.get_support()
    assert report["selected"] == [col for col, keep in zip(df.columns, mask) if keep]


def test_chi2_classification(nonnegative_classification_data):
    df, y = nonnegative_classification_data
    selector, report = fit_selector(
        df, y, method="chi2", task="classification", count=2,
        percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert report["method"] == "chi2"
    assert report["selected_count"] == 2
    assert "feat1" in report["selected"]


@pytest.mark.parametrize("method", [
    "kbest", "anova", "mutual_info", "percentile", "l1", "elastic_net", "random_forest"
])
def test_multiclass_classification(multiclass_classification_data, method):
    df, y = multiclass_classification_data
    selector, report = fit_selector(
        df, y, method=method, task="classification", count=2,
        percentile=60, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert report["selected_count"] == 2
    assert "signal1" in report["selected"]


# =============================================================================
# 3. REGRESSION TESTS ACROSS ALL METHODS
# =============================================================================

@pytest.mark.parametrize("method", [
    "variance", "kbest", "anova", "mutual_info", "percentile",
    "rfe", "rfecv", "forward", "backward", "l1", "elastic_net",
    "select_from_model", "random_forest"
])
def test_all_methods_regression(regression_data, method):
    df, y = regression_data
    selector, report = fit_selector(
        df, y, method=method, task="regression", count=2,
        percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert report is not None
    assert report["method"] == method
    assert report["selected_count"] > 0
    # In regression, signal1 and signal2 have strong linear coefficients
    if method in {"kbest", "anova", "rfe", "forward", "l1", "elastic_net"}:
        assert "signal1" in report["selected"] or "signal2" in report["selected"]


def test_chi2_rejects_regression(nonnegative_classification_data):
    df, y = nonnegative_classification_data
    # Chi2 on continuous regression target must raise ValueError
    with pytest.raises(ValueError, match="classification"):
        fit_selector(
            df, np.linspace(0, 1, len(df)), method="chi2", task="regression",
            count=2, percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
        )


# =============================================================================
# 4. UNSUPERVISED MODE VALIDATION
# =============================================================================

def test_unsupervised_variance(binary_classification_data):
    df, _ = binary_classification_data
    selector, report = fit_selector(
        df, None, method="variance", task="unsupervised", count=None,
        percentile=None, folds=None, random_state=42, max_dense_bytes=10_000_000
    )
    assert report["method"] == "variance"
    # Constant column must be dropped
    assert "const" in report["rejected"]
    assert "const" not in report["selected"]


@pytest.mark.parametrize("method", sorted(SUPERVISED))
def test_unsupervised_rejects_supervised_methods(binary_classification_data, method):
    df, _ = binary_classification_data
    with pytest.raises(ValueError, match="exige une colonne cible"):
        fit_selector(
            df, None, method=method, task="unsupervised", count=2,
            percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
        )


def test_auto_and_correlation_are_noop_in_fit_selector(binary_classification_data):
    df, y = binary_classification_data
    sel_auto, rep_auto = fit_selector(
        df, y, method="auto", task="classification", count=2,
        percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert sel_auto is None and rep_auto is None
    sel_corr, rep_corr = fit_selector(
        df, y, method="correlation", task="classification", count=2,
        percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert sel_corr is None and rep_corr is None


# =============================================================================
# 5. SCIENTIFIC CORRECTNESS & EDGE CASES
# =============================================================================

def test_chi2_negative_values_raise(binary_classification_data):
    df, y = binary_classification_data  # contains normal distribution (negatives)
    with pytest.raises(ValueError, match="non négatives"):
        fit_selector(
            df, y, method="chi2", task="classification", count=2,
            percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
        )


def test_empty_frame_raises(binary_classification_data):
    _, y = binary_classification_data
    empty = pd.DataFrame(index=range(10))
    with pytest.raises(ValueError, match="Aucune feature"):
        fit_selector(
            empty, y[:10], method="kbest", task="classification", count=2,
            percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
        )


def test_more_count_than_available_raises(binary_classification_data):
    df, y = binary_classification_data
    with pytest.raises(ValueError, match="dépasse les features disponibles"):
        fit_selector(
            df, y, method="kbest", task="classification", count=df.shape[1] + 10,
            percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
        )


def test_nan_or_inf_raises(binary_classification_data):
    df, y = binary_classification_data
    df_nan = df.copy()
    df_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="numériques finies"):
        fit_selector(
            df_nan, y, method="kbest", task="classification", count=2,
            percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
        )


def test_single_feature_remains():
    df = pd.DataFrame({"single": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    selector, report = fit_selector(
        df, y, method="kbest", task="classification", count=1,
        percentile=100, folds=2, random_state=42, max_dense_bytes=1_000_000
    )
    assert report["selected"] == ["single"]
    assert report["selected_count"] == 1


# =============================================================================
# 6. DATA LEAKAGE & PIPELINE INTEGRITY (PHASE 4)
# =============================================================================

def test_test_set_values_cannot_alter_feature_selection(binary_classification_data):
    df, y = binary_classification_data
    df_train = df.iloc[:70].copy()
    y_train = y[:70]
    df_test1 = df.iloc[70:].copy()
    df_test2 = df.iloc[70:].copy()
    # Inject massive values / noise in test2
    df_test2["noise1"] += 10000.0
    df_test2["noise2"] -= 50000.0

    prep1 = AutoPreprocessor(PreprocessConfig(
        selection_method="kbest", selection_count=2, verbose=0
    ), task="classification")
    prep1.fit(df_train, y_train)

    prep2 = AutoPreprocessor(PreprocessConfig(
        selection_method="kbest", selection_count=2, verbose=0
    ), task="classification")
    prep2.fit(df_train, y_train)

    # Both fitted preprocessors must have the exact same selected features
    assert prep1.final_feature_names_ == prep2.final_feature_names_
    assert prep1.feature_selection_report_["scores"] == prep2.feature_selection_report_["scores"]

    # Transform both test sets
    out_test1 = prep1.transform(df_test1)
    out_test2 = prep1.transform(df_test2)
    assert list(out_test1.columns) == list(out_test2.columns) == prep1.final_feature_names_


def test_save_load_preserves_exact_selected_features(tmp_path, binary_classification_data):
    df, y = binary_classification_data
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method="random_forest", selection_count=2, random_state=42, verbose=0
    ), task="classification")
    prep.fit(df, y)
    orig_features = list(prep.final_feature_names_)

    model_path = tmp_path / "model_rf.joblib"
    prep.save(str(model_path))
    loaded = AutoPreprocessor.load(str(model_path))

    assert loaded.final_feature_names_ == orig_features
    test_out = loaded.transform(df.iloc[:5])
    assert list(test_out.columns) == orig_features


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_mutual_info_preprocessor_roundtrip(tmp_path, binary_classification_data, regression_data, task):
    df, y = binary_classification_data if task == "classification" else regression_data
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method="mutual_info", selection_count=2, random_state=42, verbose=0
    ), task=task)
    prep.fit(df, y)
    expected = prep.transform(df.iloc[:6])
    destination = tmp_path / f"mutual_{task}.joblib"
    prep.save(str(destination))
    restored = AutoPreprocessor.load(str(destination))
    assert restored.config.selection_method == "mutual_info"
    assert restored.final_feature_names_ == prep.final_feature_names_
    pd.testing.assert_frame_equal(restored.transform(df.iloc[:6]), expected)


@pytest.mark.parametrize("scoring,expected", [("f1", "f1_macro"), ("roc_auc", "roc_auc_ovr")])
def test_multiclass_cv_scoring_is_resolved(multiclass_classification_data, scoring, expected):
    df, y = multiclass_classification_data
    selector, report = fit_selector(df, y, method="forward", task="classification", count=1,
        percentile=50, folds=2, scoring=scoring, random_state=42, max_dense_bytes=10_000_000)
    assert selector.get_support().sum() == 1
    assert report["scoring"] == expected


def test_cv_scoring_rejects_incompatible_task(regression_data):
    df, y = regression_data
    with pytest.raises(ValueError, match="incompatible"):
        fit_selector(df, y, method="rfecv", task="regression", count=None,
            percentile=50, folds=2, scoring="accuracy", random_state=42, max_dense_bytes=10_000_000)


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_sequential_default_count_is_valid(binary_classification_data, method):
    df, y = binary_classification_data
    selector, report = fit_selector(df.drop(columns="const"), y, method=method,
        task="classification", count=None, percentile=50, folds=2,
        random_state=42, max_dense_bytes=10_000_000)
    assert 0 < report["selected_count"] < df.shape[1] - 1


@pytest.mark.parametrize("method", [
    "variance", "correlation", "kbest", "chi2", "anova", "mutual_info",
    "percentile", "rfe", "rfecv", "forward", "backward", "l1",
    "elastic_net", "select_from_model", "random_forest",
])
def test_every_method_integrates_with_train_test_transform(binary_classification_data, method):
    frame, y = binary_classification_data
    frame = frame.drop(columns="const").copy()
    frame.loc[::9, "noise1"] = np.nan
    train, test = frame.iloc[:75], frame.iloc[75:]
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method=method, selection_count=2, selection_percentile=50,
        selection_folds=2, scaler_type="minmax", min_features_for_collinearity=100,
        random_state=42, verbose=0,
    ), task="classification")
    train_out = prep.fit_transform(train, y[:75])
    test_out = prep.transform(test)
    assert list(train_out.columns) == list(test_out.columns) == prep.final_feature_names_
    assert prep.feature_selection_report_["method"] == method
    assert prep.feature_selection_report_["selected"] == prep.final_feature_names_
    assert set(prep.feature_selection_report_["reasons"]) == set(prep.feature_selection_report_["selected"] + prep.feature_selection_report_["rejected"])


# =============================================================================
# 7. INDEPENDENT SCIKIT-LEARN COMPARISON (PHASE 6)
# =============================================================================

def test_independent_variance_threshold_matches(binary_classification_data):
    df, _ = binary_classification_data
    X = df.to_numpy(dtype=float)
    # Reference Scikit-Learn
    sk_sel = VarianceThreshold(threshold=0.0)
    sk_sel.fit(X)
    sk_mask = sk_sel.get_support()

    # AutoPrépa
    ap_sel, ap_rep = fit_selector(
        df, None, method="variance", task="unsupervised", count=None,
        percentile=None, folds=None, random_state=42, max_dense_bytes=10_000_000
    )
    assert list(ap_sel.get_support()) == list(sk_mask)
    assert ap_rep["selected"] == [col for col, m in zip(df.columns, sk_mask) if m]


def test_independent_kbest_anova_matches(binary_classification_data):
    df, y = binary_classification_data
    X = df.drop(columns="const").to_numpy(dtype=float)
    cols = [c for c in df.columns if c != "const"]
    # Reference Scikit-Learn
    sk_sel = SelectKBest(f_classif, k=2)
    sk_sel.fit(X, y)
    sk_mask = sk_sel.get_support()

    # AutoPrépa
    ap_sel, ap_rep = fit_selector(
        df.drop(columns="const"), y, method="kbest", task="classification", count=2,
        percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert list(ap_sel.get_support()) == list(sk_mask)
    np.testing.assert_allclose(ap_sel.scores_, sk_sel.scores_)
    assert ap_rep["selected"] == [col for col, m in zip(cols, sk_mask) if m]


def test_independent_mutual_info_matches(binary_classification_data):
    df, y = binary_classification_data
    X = df.drop(columns="const").to_numpy(dtype=float)
    cols = [c for c in df.columns if c != "const"]
    # Reference Scikit-Learn
    score_fn = lambda a, b: mutual_info_classif(a, b, random_state=42)
    sk_sel = SelectKBest(score_fn, k=2)
    sk_sel.fit(X, y)
    sk_mask = sk_sel.get_support()

    # AutoPrépa
    ap_sel, ap_rep = fit_selector(
        df.drop(columns="const"), y, method="mutual_info", task="classification", count=2,
        percentile=50, folds=3, random_state=42, max_dense_bytes=10_000_000
    )
    assert list(ap_sel.get_support()) == list(sk_mask)
    np.testing.assert_allclose(ap_sel.scores_, sk_sel.scores_)
