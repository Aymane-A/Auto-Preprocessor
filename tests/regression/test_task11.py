"""Targeted regressions for audit findings in identifier and preview behavior."""
import numpy as np
import pandas as pd

from auto_preprocessor import AutoPreprocessor, PreprocessConfig
from web_processing import dataset_eda, execute, selection_preview


def test_unique_integer_amount_is_retained_and_survives_save_load(tmp_path):
    train = pd.DataFrame({"amount": np.arange(40, dtype=float),
                          "category": ["a", "b", "c", "a"] * 10})
    test = pd.DataFrame({"amount": [45., 46.], "category": ["unseen", "a"]})
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={
        "amount": {"scaling": "ROBUST"}, "category": {"encoding": "ONEHOT"}}))
    prep.fit(train)
    assert prep.column_metadata_["amount"]["type"].value != "DROP_ID"
    expected = prep.transform(test)
    assert "amount" in expected.columns
    path = tmp_path / "model.joblib"
    prep.save(str(path))
    loaded = AutoPreprocessor.load(str(path))
    pd.testing.assert_frame_equal(expected, loaded.transform(test))


def test_gapped_unique_identifier_and_explicit_drop(tmp_path):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"record_id": rng.choice(np.arange(1000, 100000), 80, replace=False),
                          "amount": np.arange(80) + 20})
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    prep._detect_types(frame)
    assert prep.column_metadata_["record_id"]["type"].value == "DROP_ID"
    assert prep.column_metadata_["amount"]["type"].value != "DROP_ID"
    overview = dataset_eda(frame, None, "auto")
    assert next(row for row in overview["features"] if row["column"] == "record_id")["detected_type"] == "ID_LIKE"
    text_overview = dataset_eda(pd.DataFrame({"record_key": [f"R{value:04d}" for value in range(80)]}), None, "auto")
    assert text_overview["features"][0]["detected_type"] == "ID_LIKE"
    forced = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides={
        "record_id": {"type": "DROP"}}))
    output = forced.fit_transform(frame)
    assert forced.column_metadata_["record_id"]["type"].value == "DROP_ID"
    assert "record_id" not in output.columns


def test_selection_preview_counts_raw_transformed_and_target(tmp_path):
    frame = pd.DataFrame({"measure": np.arange(80, dtype=float) + .25,
                          "raw": ["a", "b"] * 40, "target": [0, 1] * 40})
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    preview = selection_preview(path, 1024 * 1024, "target", "classification",
                                {"selection_method": "variance"}, {"raw": "none"})
    assert "measure" in preview["selected"] and "raw" in preview["selected"]
    assert preview["selected_count"] == 2
    assert preview["output_count"] == 3
    assert preview["target_included"] is True
    (tmp_path / "exports").mkdir()
    actual = execute(path, 1024 * 1024, "target", "classification",
                     {"selection_method": "variance"}, {"raw": "none"},
                     tmp_path / "exports", "input.csv")
    assert actual["train_shape"][1] == actual["test_shape"][1] == preview["output_count"]
    assert set(preview["selected"] + ["target"]) == set(actual["output_columns"])
