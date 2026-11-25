"""Focused regression checks for Titanic and Stroke polish."""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from auto_preprocessor import AutoPreprocessor, PreprocessConfig, _series_profile
from web_processing import select_visualizations, write_workbook


def test_sparse_missing_category_uses_train_mode_without_level():
    frame = pd.DataFrame({
        "embarked": ["S"] * 45 + ["C"] * 32 + ["Q"] * 20 + [None] * 2,
        "age": np.linspace(18, 70, 99),
    })
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, clean_column_names=False))
    output = prep.fit_transform(frame)
    assert "embarked___MISSING__" not in prep.get_feature_names_out()
    assert prep.categorical_imputers_["embarked"] == "s"
    assert prep.column_missing_["embarked"] == "MODE"
    assert not any(name.endswith("_isna") for name in prep.get_feature_names_out())
    assert len(output) == len(frame)


def test_stroke_float_id_and_adjusted_skew():
    ids = pd.Series(np.arange(1001, 1121, dtype=float) * 3, name="id")
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    prep._detect_types(pd.DataFrame({"id": ids, "age": np.repeat(np.arange(30), 4) + .5}))
    assert prep.column_metadata_["id"]["type"].value == "DROP_ID"
    assert select_visualizations(ids)["detected_type"] == "ID_LIKE"
    values = pd.Series([1, 2, 2, 3, 9, 12, 55], dtype=float)
    assert _series_profile(values)["skew"] == values.skew()


def test_high_cardinality_uses_observed_denominator():
    frame = pd.DataFrame({
        "cabin": [f"C{i}" for i in range(23)] + [None] * 77,
        "target": [0, 1] * 50,
    })
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target="target")
    cabin = next(row for row in analysis.columns if row["colonne"] == "cabin")
    assert cabin["action"] == "DROP"
    assert "23/23" in cabin["steps"]["type"]["measure"]


def test_excel_summary_and_dropped_operation_are_readable(tmp_path: Path):
    path = tmp_path / "report.xlsx"
    feature = {"column": "PassengerId", "type": "DROP_ID", "dropped": True,
               "drop_reason": "Identifiant", "steps": {"clean_missing_tokens": {"action": "KEEP"}}}
    write_workbook(path, {"summary": {"Dimensions initiales": [891, 12],
                                      "Colonnes exportées": ["Age", "Survived"],
                                      "Gestion du déséquilibre": {"method": "none", "synthetic_rows": 0}},
                          "features": [feature], "correlations": {}})
    book = load_workbook(path, read_only=True)
    summary = dict(book["Résumé"].values)
    assert summary["Dimensions initiales"] == "891 × 12"
    assert "Colonnes de sortie" in summary["Colonnes exportées"]
    assert "{" not in summary["Gestion du déséquilibre"]
    operations = list(book["Opérations"].values)
    assert len(operations) == 2
    assert operations[1][2] == "Supprimée (DROP_ID)"
