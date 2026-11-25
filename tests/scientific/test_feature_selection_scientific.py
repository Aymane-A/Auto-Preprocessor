"""Scientific boundaries for feature selection, including an external holdout."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from auto_preprocessor import AutoPreprocessor, PreprocessConfig
from feature_selection import fit_selector


def _mixed_binary():
    rng = np.random.default_rng(126)
    signal = rng.normal(size=72)
    frame = pd.DataFrame({
        "signal": signal,
        "noise": rng.normal(size=72),
        "group": np.tile(["red", "blue", "green"], 24),
    })
    frame.loc[::9, "signal"] = np.nan
    return frame, (signal > 0).astype(int)


@pytest.mark.parametrize("method", ["rfecv", "forward", "backward"])
def test_external_fold_is_independent_of_inner_selection(method, tmp_path, monkeypatch):
    """Outer validation labels/values cannot affect any inner fitted object."""
    import fold_safe_selection

    frame, target = _mixed_binary()
    outer_train, outer_valid = next(StratifiedKFold(
        n_splits=2, shuffle=True, random_state=17).split(frame, target))
    train = frame.iloc[outer_train]
    valid = frame.iloc[outer_valid]
    seen = []
    original = fold_safe_selection._folds

    def inspect_folds(raw_frame, raw_target, config, task):
        assert set(raw_frame.index) == set(train.index)
        folds = original(raw_frame, raw_target, config, task)
        for fold in folds:
            fit_ids, valid_ids = set(fold[4]), set(fold[5])
            assert not fit_ids & valid_ids
            assert fit_ids | valid_ids == set(train.index)
            assert not (fit_ids | valid_ids) & set(valid.index)
            seen.append((fit_ids, valid_ids))
        return folds

    monkeypatch.setattr(fold_safe_selection, "_folds", inspect_folds)
    config = PreprocessConfig(selection_method=method, selection_count=2,
                              selection_folds=2, random_state=17,
                              min_features_for_collinearity=100,
                              outlier_method="none", verbose=0)
    first = AutoPreprocessor(config, task="classification")
    first_train = first.fit_transform(train, target[outer_train])
    first_valid = first.transform(valid)
    model = LogisticRegression(max_iter=500).fit(first_train, target[outer_train])
    accuracy = accuracy_score(target[outer_valid], model.predict(first_valid))
    assert 0 <= accuracy <= 1
    assert first.feature_selection_report_["cv_preprocessing"] == "fold_isolated"
    assert len(seen) == config.selection_folds

    altered = frame.copy()
    altered.loc[valid.index, "signal"] = 1e6
    altered.loc[valid.index, "group"] = "outer_only"
    altered_target = target.copy()
    altered_target[outer_valid] = 1 - altered_target[outer_valid]
    second = AutoPreprocessor(config, task="classification")
    second_train = second.fit_transform(altered.iloc[outer_train], altered_target[outer_train])
    pd.testing.assert_frame_equal(first_train, second_train)
    assert first.feature_selection_report_["cv_scores"] == second.feature_selection_report_["cv_scores"]
    assert first.final_feature_names_ == second.final_feature_names_
    assert list(first_valid.columns) == list(first_train.columns)
    assert list(second.transform(altered.iloc[outer_valid]).columns) == list(second_train.columns)
    assert "outer_only" not in " ".join(second.final_feature_names_)
    second_model = LogisticRegression(max_iter=500).fit(second_train, altered_target[outer_train])
    inverted_accuracy = accuracy_score(altered_target[outer_valid], second_model.predict(first_valid))
    assert inverted_accuracy == pytest.approx(1 - accuracy)

    artifact = tmp_path / "outer-preprocessor.joblib"
    first.save(str(artifact))
    loaded = AutoPreprocessor.load(str(artifact))
    pd.testing.assert_frame_equal(first_valid, loaded.transform(valid))


def test_real_housing_independent_external_split_and_unknown_category():
    """Use a tracked real dataset, including missing values and a new category."""
    source = Path(__file__).resolve().parents[2] / "data" / "immobilier_demo.csv"
    data = pd.read_csv(source)[["surface_m2", "nb_pieces", "ville", "etat", "etage", "prix_dh"]]
    data = data.iloc[:160].copy()
    target = data.pop("prix_dh").to_numpy()
    train_idx, valid_idx = next(StratifiedKFold(
        n_splits=2, shuffle=True, random_state=23).split(data, pd.qcut(target, 2, labels=False)))
    train, valid = data.iloc[train_idx], data.iloc[valid_idx]
    config = PreprocessConfig(selection_method="kbest", selection_count=3,
                              min_features_for_collinearity=100,
                              outlier_method="none", random_state=23, verbose=0)
    prep = AutoPreprocessor(config, task="regression")
    train_encoded = prep.fit_transform(train, target[train_idx])
    valid_encoded = prep.transform(valid.assign(ville="UNSEEN_CITY"))
    assert train_encoded.columns.tolist() == valid_encoded.columns.tolist() == prep.final_feature_names_
    assert np.isfinite(valid_encoded.to_numpy(dtype=float)).all()
    changed = data.copy()
    changed.loc[valid.index, "Age"] = 1e6
    other = AutoPreprocessor(config, task="regression").fit(
        changed.iloc[train_idx], target[train_idx])
    assert other.final_feature_names_ == prep.final_feature_names_


def test_real_titanic_binary_holdout_and_reload(tmp_path):
    """Validate on the locally available original Titanic observations."""
    source = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
    if not source.exists():
        pytest.skip("Le dataset Titanic local n'est pas livré avec le dépôt.")
    data = pd.read_csv(source)[["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]].iloc[:160]
    frame, target = data.drop(columns="Survived"), data["Survived"].to_numpy()
    train_idx, valid_idx = next(StratifiedKFold(
        n_splits=2, shuffle=True, random_state=11).split(frame, target))
    prep = AutoPreprocessor(PreprocessConfig(
        selection_method="kbest", selection_count=3,
        min_features_for_collinearity=100, outlier_method="none",
        random_state=11, verbose=0,
    ), task="classification")
    encoded_train = prep.fit_transform(frame.iloc[train_idx], target[train_idx])
    encoded_valid = prep.transform(frame.iloc[valid_idx].assign(Embarked="UNKNOWN_PORT"))
    assert encoded_train.columns.tolist() == encoded_valid.columns.tolist()
    assert np.isfinite(encoded_valid.to_numpy(dtype=float)).all()
    artifact = tmp_path / "titanic.joblib"
    prep.save(str(artifact))
    pd.testing.assert_frame_equal(AutoPreprocessor.load(str(artifact)).transform(
        frame.iloc[valid_idx].assign(Embarked="UNKNOWN_PORT")), encoded_valid)


def test_chi_square_supports_multiclass_nonnegative_data():
    rng = np.random.default_rng(45)
    frame = pd.DataFrame(rng.integers(0, 8, size=(90, 4)), columns=list("abcd"))
    target = np.repeat([0, 1, 2], 30)
    selector, report = fit_selector(frame, target, method="chi2", task="classification",
                                    count=2, percentile=50, folds=2, random_state=45,
                                    max_dense_bytes=1_000_000)
    assert report["selected"] == list(frame.columns[selector.get_support()])
    assert selector.transform(frame.to_numpy()).shape == (90, 2)


@pytest.mark.parametrize("method", ["variance", "correlation"])
def test_unsupervised_filter_ignores_external_validation_values(method):
    rng = np.random.default_rng(52)
    base = rng.normal(size=72)
    frame = pd.DataFrame({"base": base, "copy": base * 2,
                          "noise": rng.normal(size=72), "constant": np.ones(72)})
    train, valid = frame.iloc[:54], frame.iloc[54:]
    config = PreprocessConfig(selection_method=method,
                              min_features_for_collinearity=2,
                              correlation_threshold=.9,
                              outlier_method="none", verbose=0)
    first = AutoPreprocessor(config, task="unsupervised").fit(train)
    altered = frame.copy()
    altered.loc[valid.index, "base"] = 1e9
    altered.loc[valid.index, "constant"] = np.arange(len(valid))
    second = AutoPreprocessor(config, task="unsupervised").fit(altered.iloc[:54])
    assert first.final_feature_names_ == second.final_feature_names_
    assert "constant" not in first.final_feature_names_
    assert list(first.transform(valid).columns) == first.final_feature_names_
