"""Boundary regressions behind the eight previously failing tests."""
import numpy as np
import pandas as pd

from auto_preprocessor import (AutoPreprocessor, PreprocessConfig,
                               _unique_integer_identifier)


def test_identifier_boundary_keeps_large_real_measure_and_recognizes_named_keys():
    keys = pd.Series(np.arange(80) + 6000, name="device_code")
    sequence = pd.Series(np.arange(80), name="sequence")
    measure = pd.Series(np.linspace(1e40, 2e40, 80), name="magnitude")
    assert _unique_integer_identifier(keys)
    assert _unique_integer_identifier(sequence)
    assert not _unique_integer_identifier(measure)
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, downcast_numeric=False,
                                             outlier_method="none", scaler_type="none",
                                             noise_guard="off"))
    prep._detect_types(pd.DataFrame({"magnitude": measure}))
    assert prep.column_metadata_["magnitude"]["type"].value == "NUMERIC_CONTINUOUS"


def test_sparse_missing_mode_rule_preserves_explicit_choice_and_common_missing_level():
    short = pd.DataFrame({"color": ["a", "b", None, "a", "c"],
                          "other": [1, 2, 3, 4, 5]})
    automatic = AutoPreprocessor(PreprocessConfig(verbose=0, outlier_method="none",
                                                   min_features_for_collinearity=100))
    automatic.fit(short)
    assert automatic.categorical_imputers_.get("color") is None
    explicit = AutoPreprocessor(PreprocessConfig(verbose=0, outlier_method="none",
                                                  min_features_for_collinearity=100,
                                                  column_overrides={"color": {"missing": "MODE"}}))
    explicit.fit(short)
    assert explicit.categorical_imputers_["color"] == "a"
    long = pd.DataFrame({"color": ["a"] * 50 + ["b"] * 48 + [None] * 2,
                         "other": np.arange(100)})
    sparse = AutoPreprocessor(PreprocessConfig(verbose=0, outlier_method="none"))
    sparse.fit(long)
    assert sparse.column_missing_["color"] == "MODE"


def test_missing_pairwise_correlation_does_not_trigger_auto_drop_before_imputation():
    frame = pd.DataFrame({"sequence": np.arange(80),
                          "partial": [float(i) if i % 5 else np.nan for i in range(80)],
                          "other": np.sin(np.arange(80))})
    config = PreprocessConfig(verbose=0, min_features_for_collinearity=2,
                              correlation_threshold=.95)
    analysis = AutoPreprocessor(config).analyze(frame)
    row = next(item for item in analysis.columns if item["colonne"] == "partial")
    assert row["action"] != "DROP"
    prep = AutoPreprocessor(config)
    prep.fit(frame)
    assert "partial" not in prep.dropped_collinear_cols_


def test_identifier_evidence_reports_line_and_observed_denominators():
    frame = pd.DataFrame({"customer_id": [f"C{i}" for i in range(40)]})
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame)
    row = analysis.columns[0]
    assert row["type"] == "DROP_ID"
    assert "40 / 40 lignes (100.0%)" in row["drop_preuve"]
    assert "40 / 40 valeurs observées (100.0%)" in row["drop_preuve"]
