"""End-to-end checks for selection preview, export and reload."""
import numpy as np
import pandas as pd
import pytest

from auto_preprocessor import AutoPreprocessor
from web_processing import execute, selection_preview, recommendation_train_frame, analyze_choices, processing_options


def test_mutual_information_preview_matches_export_and_reload(tmp_path):
    rng = np.random.default_rng(47)
    signal = rng.normal(size=100)
    frame = pd.DataFrame({
        "signal": signal,
        "group": np.tile(["north", "south", "west", "east"], 25),
        "noise": rng.normal(size=100),
        "target": (signal + rng.normal(scale=.35, size=100) > 0).astype(int),
    })
    frame.loc[::11, "signal"] = np.nan
    source = tmp_path / "selection.csv"
    frame.to_csv(source, index=False)
    options = {"selection_method": "mutual_info", "selection_count": 2,
               "test_size_pct": 20, "random_state": 47,
               "min_features_for_collinearity": 100}
    preview = selection_preview(source, 10_000_000, "target", "classification", options, {})
    output = tmp_path / "export"
    output.mkdir()
    result = execute(source, 10_000_000, "target", "classification",
                     options, {}, output, "selection.csv")
    expected = [name for name in result["output_columns"] if name != "target"]
    assert preview["selected"] == expected
    assert preview["feature_sources"]
    assert preview["details"]["fit_rows"] == result["train_shape"][0]
    loaded = AutoPreprocessor.load(str(output / "preprocesseur.joblib"))
    new_data = frame.drop(columns="target").iloc[:3]
    transformed = loaded.transform(new_data)
    assert list(transformed.columns) == loaded.final_feature_names_
    assert loaded.feature_selection_report_["selected"] == loaded.final_feature_names_


def test_holdout_values_do_not_change_recommendations():
    rng = np.random.default_rng(13)
    frame = pd.DataFrame({"signal": rng.normal(size=100), "noise": rng.normal(size=100),
                          "target": np.tile([0, 1], 50)})
    options = {"test_size_pct": 20, "random_state": 13}
    train = recommendation_train_frame(frame, "target", "classification", options)
    changed = frame.copy()
    changed.loc[~changed.index.isin(train.index), "noise"] = 1e9
    first = analyze_choices(recommendation_train_frame(frame, "target", "classification", options),
                            "target", processing_options("classification", options), {})
    second = analyze_choices(recommendation_train_frame(changed, "target", "classification", options),
                             "target", processing_options("classification", options), {})
    assert [(row["colonne"], row["action"], row["raison"]) for row in first.columns] == [
        (row["colonne"], row["action"], row["raison"]) for row in second.columns]


@pytest.mark.parametrize("method", ["rfecv", "forward", "backward"])
def test_fold_safe_wrapper_preview_export_and_reload(tmp_path, method):
    rng = np.random.default_rng(31)
    signal = rng.normal(size=72)
    frame = pd.DataFrame({"signal": signal, "group": np.tile(["a", "b", "c"], 24),
                          "noise": rng.normal(size=72), "target": (signal > 0).astype(int)})
    frame.loc[::10, "noise"] = np.nan
    source = tmp_path / "wrapper.csv"
    frame.to_csv(source, index=False)
    options = {"selection_method": method, "selection_count": 2, "selection_folds": 2,
               "test_size_pct": 20, "min_features_for_collinearity": 100, "random_state": 31}
    preview = selection_preview(source, 10_000_000, "target", "classification", options, {})
    output = tmp_path / "export"
    output.mkdir()
    result = execute(source, 10_000_000, "target", "classification",
                     options, {}, output, "wrapper.csv")
    assert preview["details"]["cv_preprocessing"] == "fold_isolated"
    assert preview["selected"] == [name for name in result["output_columns"] if name != "target"]
    loaded = AutoPreprocessor.load(str(output / "preprocesseur.joblib"))
    assert loaded.feature_selection_report_["cv_preprocessing"] == "fold_isolated"
    assert list(loaded.transform(frame.drop(columns="target").iloc[:5]).columns) == loaded.final_feature_names_
