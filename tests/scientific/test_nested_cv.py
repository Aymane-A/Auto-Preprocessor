"""Nested-CV isolation and prediction accounting tests."""
import joblib
import numpy as np
import pandas as pd
import pytest

from auto_preprocessor import AutoPreprocessor, PreprocessConfig
from fold_safe_selection import _folds
from nested_cv import nested_cross_validate


def _frame(kind="binary", n=48):
    rng = np.random.default_rng(187)
    signal = rng.normal(size=n)
    frame = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n),
                          "category": np.tile(["a", "b", "c"], n // 3)})
    frame.loc[::9, "signal"] = np.nan
    if kind == "binary":
        target = (signal > 0).astype(int)
    elif kind == "multiclass":
        target = np.digitize(signal, [-.4, .4])
    else:
        target = signal * 2 + rng.normal(scale=.1, size=n)
    frame["target"] = target
    return frame


@pytest.mark.parametrize("kind,family,metric", [
    ("binary", "logistic", "roc_auc"),
    ("multiclass", "logistic", "f1_macro"),
    ("regression", "linear", "rmse"),
])
def test_nested_cv_metrics_unique_predictions_and_joblib(kind, family, metric, tmp_path):
    frame = _frame(kind)
    cfg = PreprocessConfig(selection_method="kbest", selection_count=2,
                           outlier_method="none", min_features_for_collinearity=100,
                           random_state=187, verbose=0)
    result = nested_cross_validate(frame, "target", family, config=cfg,
                                   task="regression" if kind == "regression" else "classification",
                                   outer_cv=2, inner_cv=2,
                                   candidates=[{"selection_count": 1}, {"selection_count": 2}])
    assert metric in result["global_metrics"]
    assert len(result["outer_folds"]) == 2
    assert sorted(result["predicted_row_positions"]) == list(range(len(frame)))
    assert not result["unscored_row_positions"]
    for fold in result["outer_folds"]:
        assert not set(fold["train_row_positions"]) & set(fold["validation_row_positions"])
        assert len(fold["predictions"]) == len(fold["validation_row_positions"])
        assert fold["selected_config"]["preprocessing"]["selection_count"] in (1, 2)
        assert len(fold["inner_search"]) == 2
        assert metric in fold["metrics"]
    path = tmp_path / "nested.joblib"
    joblib.dump(result, path)
    loaded = joblib.load(path)
    assert loaded["global_metrics"] == result["global_metrics"]
    valid_rows = result["outer_folds"][0]["validation_row_positions"]
    np.testing.assert_array_equal(
        loaded["fitted_outer_pipelines"][0].predict(frame.drop(columns="target").iloc[valid_rows]),
        result["outer_folds"][0]["predictions"])


def test_external_labels_and_values_cannot_choose_configuration():
    frame = _frame()
    cfg = PreprocessConfig(selection_method="variance",
                           outlier_method="none", min_features_for_collinearity=100,
                           random_state=187, verbose=0)
    options = [{"scaler_type": "standard"}, {"scaler_type": "robust"}]
    groups = np.repeat(np.arange(12), 4)
    first = nested_cross_validate(frame, "target", "logistic", config=cfg, task="classification",
                                  outer_cv=2, inner_cv=2, candidates=options, groups=groups)
    first_fold = first["outer_folds"][0]
    changed = frame.copy()
    validation = first_fold["validation_row_positions"]
    changed.loc[validation, "signal"] = 1e6
    changed.loc[validation, "category"] = "external_only"
    changed.loc[validation, "target"] = 1 - changed.loc[validation, "target"]
    # The second outer fold has a different train; inspect the first fold only.
    second = nested_cross_validate(changed, "target", "logistic", config=cfg,
                                   task="classification", outer_cv=2, inner_cv=2,
                                   candidates=options, groups=groups)
    other_fold = second["outer_folds"][0]
    assert other_fold["train_row_positions"] == first_fold["train_row_positions"]
    assert other_fold["inner_search"] == first_fold["inner_search"]
    assert other_fold["selected_config"] == first_fold["selected_config"]
    assert other_fold["selected_features"] == first_fold["selected_features"]
    assert other_fold["metrics"] != first_fold["metrics"]


def test_fold_filters_and_unknown_category_use_only_inner_train():
    frame = _frame()
    raw = frame.drop(columns="target")
    target = frame["target"].to_numpy()
    raw["duplicate"] = raw["noise"] * 2
    cfg = PreprocessConfig(selection_method="forward", selection_count=2,
                           selection_folds=2, min_features_for_collinearity=2,
                           correlation_threshold=.9, outlier_method="none",
                           random_state=187, verbose=0)
    folds = _folds(raw, target, cfg, "classification")
    from sklearn.model_selection import StratifiedKFold
    first_train, first_valid = next(StratifiedKFold(
        n_splits=2, shuffle=True, random_state=187).split(raw, target))
    altered = raw.copy()
    altered.loc[first_valid, "noise"] = 1e6
    altered.loc[first_valid, "category"] = "validation_only"
    changed = _folds(altered, target, cfg, "classification")
    assert folds[0][0].columns.tolist() == changed[0][0].columns.tolist()
    pd.testing.assert_frame_equal(folds[0][0], changed[0][0])
    assert not any("validation_only" in name for name in changed[0][0].columns)
    assert changed[0][0].columns.tolist() == changed[0][1].columns.tolist()
    assert "duplicate" not in folds[0][0].columns or "noise" not in folds[0][0].columns


def test_sequential_candidates_are_local_to_each_inner_train():
    from sklearn.model_selection import StratifiedKFold
    from fold_safe_selection import fit_fold_safe_selector

    frame = _frame(n=72)
    raw, target = frame.drop(columns="target"), frame["target"].to_numpy()
    _, first_valid = next(StratifiedKFold(
        n_splits=2, shuffle=True, random_state=187).split(raw, target))
    raw.loc[first_valid[0], "category"] = "validation_only"
    config = PreprocessConfig(selection_method="forward", selection_count=2,
                              selection_folds=2, rare_category_threshold=0,
                              min_features_for_collinearity=100,
                              outlier_method="none", random_state=187, verbose=0)
    final_frame = AutoPreprocessor(
        PreprocessConfig(selection_method="auto", rare_category_threshold=0,
                         min_features_for_collinearity=100,
                         outlier_method="none", random_state=187, verbose=0),
        task="classification").fit_transform(raw, target)
    _, report = fit_fold_safe_selector(final_frame, raw, target, method="forward",
                                       config=config, task="classification")
    assert not any("validation_only" in name for name in report["candidate_names_by_fold"][0])
    assert not any("validation_only" in name for name in report["cv_scores"][0]["selected_by_fold"][0])


def test_group_and_time_protocols_keep_boundaries():
    frame = _frame(n=60)
    cfg = PreprocessConfig(selection_method="variance", outlier_method="none",
                           min_features_for_collinearity=100, random_state=187, verbose=0)
    groups = np.repeat(np.arange(12), 5)
    grouped = nested_cross_validate(frame, "target", "logistic", config=cfg,
                                    task="classification", outer_cv=2, inner_cv=2,
                                    groups=groups)
    for fold in grouped["outer_folds"]:
        assert not set(groups[fold["train_row_positions"]]) & set(groups[fold["validation_row_positions"]])
    temporal = frame.copy()
    temporal["when"] = pd.date_range("2024-01-01", periods=len(frame), freq="D")
    timed = nested_cross_validate(temporal, "target", "logistic", config=cfg,
                                  task="classification", outer_cv=2, inner_cv=2,
                                  time_column="when")
    assert timed["unscored_row_positions"]
    for fold in timed["outer_folds"]:
        assert max(fold["train_row_positions"]) < min(fold["validation_row_positions"])


def test_web_nested_evaluation_keeps_exploratory_scores_separate(tmp_path):
    from web_processing import evaluate_nested

    frame = _frame()
    source = tmp_path / "nested.csv"
    frame.to_csv(source, index=False)
    result = evaluate_nested(source, 1_000_000, "target", "classification",
                             {"selection_method": "variance", "outlier_method": "none",
                              "min_features_for_collinearity": 100, "random_state": 187},
                             "logistic", 2, 2, None, None, [], [], [{}])
    assert result["protocol"] == "nested_cross_validation"
    assert "fitted_outer_pipelines" not in result
    assert len(result["outer_folds"]) == 2
    assert "global_metrics" in result
    assert "inner_search" in result["outer_folds"][0]


def test_real_titanic_nested_binary_validation():
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
    if not source.exists():
        pytest.skip("Le dataset Titanic local n'est pas livré avec le dépôt.")
    frame = pd.read_csv(source)[["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]].iloc[:90]
    result = nested_cross_validate(
        frame.rename(columns={"Survived": "target"}), "target", "logistic",
        config=PreprocessConfig(selection_method="kbest", selection_count=2,
                                outlier_method="none", min_features_for_collinearity=100,
                                random_state=31, verbose=0),
        task="classification", outer_cv=2, inner_cv=2)
    assert len(result["predicted_row_positions"]) == len(frame)
    assert "roc_auc" in result["global_metrics"]


def test_packaged_iris_dataset_reproducible_multiclass_validation():
    from sklearn.datasets import load_iris

    frame = load_iris(as_frame=True).frame.rename(columns={"target": "species"})
    result = nested_cross_validate(
        frame, "species", "logistic",
        config=PreprocessConfig(selection_method="kbest", selection_count=2,
                                outlier_method="none", min_features_for_collinearity=100,
                                random_state=31, verbose=0),
        task="classification", outer_cv=2, inner_cv=2)
    assert len(result["predicted_row_positions"]) == len(frame)
    assert len(result["global_metrics"]["confusion_matrix"]) == 3


def test_nested_rejects_global_task_inference():
    with pytest.raises(ValueError, match="explicite"):
        nested_cross_validate(_frame(), "target", "logistic", outer_cv=2,
                              inner_cv=2)


def test_nested_seed_is_reproducible_and_missing_targets_keep_source_positions():
    frame = _frame(n=60)
    frame.loc[[3, 17], "target"] = np.nan
    config = PreprocessConfig(selection_method="variance", outlier_method="none",
                              min_features_for_collinearity=100, random_state=71, verbose=0)
    args = dict(config=config, task="classification", outer_cv=2, inner_cv=2)
    first = nested_cross_validate(frame, "target", "logistic", **args)
    second = nested_cross_validate(frame, "target", "logistic", **args)
    expected = sorted(set(range(60)) - {3, 17})
    assert sorted(first["predicted_row_positions"]) == expected
    assert sorted(second["predicted_row_positions"]) == expected
    assert first["global_metrics"] == second["global_metrics"]
    assert [(f["validation_row_positions"], f["selected_config"], f["predictions"])
            for f in first["outer_folds"]] == [
            (f["validation_row_positions"], f["selected_config"], f["predictions"])
            for f in second["outer_folds"]]
    assert all(f["elapsed_seconds"] >= 0 and f["pipeline_fit_count"] == 3
               for f in first["outer_folds"])


def test_nested_rejects_class_too_rare_for_inner_stratification():
    frame = _frame(n=60)
    frame["target"] = 0
    frame.loc[:3, "target"] = 1
    config = PreprocessConfig(selection_method="variance", outlier_method="none",
                              random_state=71, verbose=0)
    with pytest.raises(ValueError, match="Chaque classe"):
        nested_cross_validate(frame, "target", "logistic", config=config,
                              task="classification", outer_cv=2, inner_cv=3)


def test_nested_temporal_split_rejects_training_without_a_class():
    frame = _frame(n=60)
    frame["target"] = 0
    frame.loc[45:, "target"] = 1
    frame["when"] = pd.date_range("2024-01-01", periods=60, freq="D")
    config = PreprocessConfig(selection_method="variance", outlier_method="none",
                              random_state=71, verbose=0)
    with pytest.raises(ValueError, match="train ne contient pas toutes les classes"):
        nested_cross_validate(frame, "target", "logistic", config=config,
                              task="classification", outer_cv=2, inner_cv=2,
                              time_column="when")


@pytest.mark.parametrize("method", ["rfecv", "forward", "backward"])
def test_nested_wrapper_retrains_inside_every_outer_fold(method):
    frame = _frame(n=72)
    result = nested_cross_validate(
        frame, "target", "logistic",
        config=PreprocessConfig(selection_method=method, selection_count=2,
                                selection_folds=2, outlier_method="none",
                                min_features_for_collinearity=100,
                                random_state=187, verbose=0),
        task="classification", outer_cv=2, inner_cv=2)
    assert len(result["predicted_row_positions"]) == len(frame)
    assert all(pipeline.steps[0][1].feature_selection_report_["cv_preprocessing"] == "fold_isolated"
               for pipeline in result["fitted_outer_pipelines"])


def test_nested_cv_refuses_unsafe_legacy_score(monkeypatch):
    from fold_safe_selection import FoldIsolationUnavailable

    def unavailable(*args, **kwargs):
        raise FoldIsolationUnavailable("Opération à lignes variables")

    monkeypatch.setattr("fold_safe_selection._folds", unavailable)
    with pytest.warns(UserWarning, match="matrice déjà prétraitée"):
        with pytest.raises(ValueError, match="unsafe_legacy"):
            nested_cross_validate(
                _frame(n=72), "target", "logistic",
                config=PreprocessConfig(selection_method="forward", selection_count=2,
                                        selection_folds=2, outlier_method="none",
                                        min_features_for_collinearity=100,
                                        random_state=187, verbose=0),
                task="classification", outer_cv=2, inner_cv=2)


def test_nested_http_route_registered_and_contract_validated():
    from api_server import create_app
    from web_contracts import NestedEvaluation

    app = create_app()
    assert any(route.path == "/api/evaluate-nested" and "POST" in route.methods
               for route in app.routes)
    request = NestedEvaluation.model_validate({"session_id": "test", "family": "logistic",
                                                "outer_folds": 2, "inner_folds": 2})
    assert request.model_dump()["outer_folds"] == 2
    with pytest.raises(ValueError, match="selection_counts"):
        NestedEvaluation.model_validate({"session_id": "test", "selection_counts": [2],
                                         "options": {"selection_method": "rfecv"}})
