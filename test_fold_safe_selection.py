"""Inner-CV isolation, generated names and persistence for wrapper selectors."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from auto_preprocessor import AutoPreprocessor, PreprocessConfig


def mixed_data(multiclass=False):
    rng = np.random.default_rng(91)
    amount = rng.normal(size=72)
    frame = pd.DataFrame({
        "amount": amount,
        "group": np.tile(["north", "south", "east"], 24),
        "noise": rng.normal(size=72),
    })
    frame.loc[::8, "amount"] = np.nan
    target = np.digitize(amount, [-.4, .4]) if multiclass else (amount > 0).astype(int)
    return frame, target


@pytest.mark.parametrize("method", ["rfecv", "forward", "backward"])
@pytest.mark.parametrize("multiclass", [False, True])
def test_fold_isolation_with_numeric_categorical_missing_and_reload(tmp_path, method, multiclass):
    frame, y = mixed_data(multiclass)
    config = PreprocessConfig(selection_method=method, selection_count=2,
                              selection_folds=2, min_features_for_collinearity=100,
                              random_state=91, outlier_method="none", verbose=0)
    prep = AutoPreprocessor(config, task="classification")
    train = prep.fit_transform(frame.iloc[:54], y[:54])
    holdout = prep.transform(frame.iloc[54:])
    report = prep.feature_selection_report_
    assert report["cv_preprocessing"] == "fold_isolated"
    assert len(report["cv_folds"]) == 2
    assert list(train.columns) == list(holdout.columns) == report["selected"]
    assert any(name.startswith("group_") for name in report["selected"] + report["rejected"])
    assert all(fold["fit_rows"] + fold["validation_rows"] == 54 for fold in report["cv_folds"])
    path = tmp_path / f"{method}-{multiclass}.joblib"
    prep.save(str(path))
    loaded = AutoPreprocessor.load(str(path))
    pd.testing.assert_frame_equal(loaded.transform(frame.iloc[54:]), holdout)
    assert loaded.final_feature_names_ == prep.final_feature_names_


def test_validation_fold_cannot_contribute_to_fitted_imputation(monkeypatch):
    frame, y = mixed_data()
    observed = []
    original_fit = AutoPreprocessor.fit

    def inspect_fit(self, X, target=None):
        result = original_fit(self, X, target)
        if self.config.selection_method == "auto" and "amount" in self.num_imputer_:
            observed.append((list(X.index), self.num_imputer_["amount"]))
        return result

    monkeypatch.setattr(AutoPreprocessor, "fit", inspect_fit)
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method="rfecv", selection_folds=2, outlier_method="none",
        min_features_for_collinearity=100, random_state=91, verbose=0,
    ), task="classification")
    prep.fit(frame, y)
    expected = [list(frame.iloc[indices].index) for indices, _ in
                StratifiedKFold(n_splits=2, shuffle=True, random_state=91).split(frame, y)]
    assert [indices for indices, _ in observed] == expected
    changed = frame.copy()
    changed.loc[~changed.index.isin(expected[0]), "amount"] = 1e6
    first_fold_stat = observed[0][1]
    observed.clear()
    AutoPreprocessor(prep.config, task="classification").fit(changed, y)
    assert observed and observed[0][0] == expected[0]
    assert observed[0][1] == pytest.approx(first_fold_stat)
    assert all(len(indices) < len(frame) for indices, _ in observed)


def test_unsupported_fold_operation_retains_explicit_warning(monkeypatch):
    from fold_safe_selection import FoldIsolationUnavailable
    frame, y = mixed_data()

    def unavailable(*args, **kwargs):
        raise FoldIsolationUnavailable("Opération qui supprime des lignes")

    monkeypatch.setattr("fold_safe_selection._folds", unavailable)
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method="rfecv", selection_folds=2,
        min_features_for_collinearity=100, verbose=0,
    ), task="classification")
    prep.fit(frame.iloc[:54], y[:54])
    assert prep.feature_selection_report_["cv_preprocessing"] == "unsafe_legacy"
    assert any("non isolée" in warning and "optimiste" in warning for warning in prep.warnings_)


@pytest.mark.parametrize("method", ["rfecv", "forward", "backward"])
def test_fold_safe_regression(method):
    frame, _ = mixed_data()
    rng = np.random.default_rng(5)
    target = 2 * frame["amount"].fillna(0).to_numpy() + rng.normal(scale=.1, size=len(frame))
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method=method, selection_count=2, selection_folds=2,
        min_features_for_collinearity=100, random_state=91, verbose=0,
    ), task="regression")
    output = prep.fit_transform(frame.iloc[:54], target[:54])
    assert prep.feature_selection_report_["cv_preprocessing"] == "fold_isolated"
    assert list(output.columns) == list(prep.transform(frame.iloc[54:]).columns)


@pytest.mark.parametrize("requested,resolved", [("f1", "f1_macro"), ("roc_auc", "roc_auc_ovr")])
def test_fold_safe_multiclass_scoring(requested, resolved):
    frame, target = mixed_data(multiclass=True)
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method="forward", selection_count=2, selection_folds=2,
        selection_scoring=requested, min_features_for_collinearity=100,
        random_state=91, verbose=0,
    ), task="classification")
    prep.fit(frame, target)
    report = prep.feature_selection_report_
    assert report["cv_preprocessing"] == "fold_isolated"
    assert report["scoring"] == resolved
    assert all(np.isfinite(value) for step in report["cv_scores"] for value in step["scores"])
