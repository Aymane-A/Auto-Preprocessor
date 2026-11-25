"""Unique integer keys with gaps are identifiers; measurements and repeated counts are not."""
import numpy as np
import pandas as pd

from auto_preprocessor import AutoPreprocessor, PreprocessConfig
from web_processing import dataset_eda, feature_eda, select_visualizations, selected_overrides


def test_gapped_integer_key_is_drop_id_without_eda_charts(tmp_path):
    rng = np.random.default_rng(17)
    key = pd.Series(rng.choice(np.arange(67, 72940), 120, replace=False), name="arbitrary")
    target = pd.Series([0, 1] * 60, name="label")
    frame = pd.DataFrame({"arbitrary": key, "label": target})
    visual = select_visualizations(key, target, "classification")
    assert visual["detected_type"] == "ID_LIKE"
    assert [item["kind"] for item in visual["recommended_visualization"]] == ["summary"]
    assert visual["recommended_visualization"][0]["cardinality"] == 120
    assert len(visual["recommended_visualization"][0]["examples"]) == 5
    source = tmp_path / "sample.csv"
    frame.to_csv(source, index=False)
    detail = feature_eda(source, 1024**2, "arbitrary", "label", "classification")
    assert "distribution" not in detail and "boxplot" not in detail
    overview = dataset_eda(frame, "label", "classification")
    assert "arbitrary" not in overview["pearson"]["columns"]
    assert "arbitrary" not in [relation["column"] for relation in overview["target_relations"]]
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target="label")
    row = next(item for item in analysis.columns if item["colonne"] == "arbitrary")
    assert row["type"] == "DROP_ID" and row["action"] == "DROP"
    _, chosen = selected_overrides(analysis, "label", {})
    assert chosen["arbitrary"] == "DROP"


def test_repeated_integers_and_unique_float_measurements_are_not_ids():
    repeated = pd.Series(np.arange(120), name="count")
    repeated.iloc[-1] = repeated.iloc[-2]
    measurement = pd.Series(np.arange(120, dtype=float) + .25, name="measurement")
    integer_valued_float = pd.Series(np.arange(120, dtype=float), name="numeric_float")
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    prep._detect_types(pd.concat([repeated, measurement, integer_valued_float], axis=1))
    for name in ("count", "measurement"):
        assert prep.column_metadata_[name]["type"].value != "DROP_ID"
    assert prep.column_metadata_["numeric_float"]["type"].value != "DROP_ID"
    assert select_visualizations(repeated)["detected_type"] != "ID_LIKE"
    assert select_visualizations(measurement)["detected_type"] != "ID_LIKE"


def test_nullable_integer_identifier_uses_observed_denominator():
    values = pd.Series(pd.array([None] * 12 + list(range(67, 167)), dtype="Int64"), name="key")
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    prep._detect_types(values.to_frame())
    assert prep.column_metadata_["key"]["type"].value == "DROP_ID"
    assert select_visualizations(values)["detected_type"] == "ID_LIKE"
