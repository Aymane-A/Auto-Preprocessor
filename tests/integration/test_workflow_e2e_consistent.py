"""End-to-end consistency test verifying Phase 5 workflow requirements."""
import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from openpyxl import load_workbook
import web_processing as wp
from auto_preprocessor import AutoPreprocessor


def test_complete_workflow_titanic_consistency():
    t_path = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
    if not t_path.is_file():
        pytest.skip("Le dataset Titanic local n'est pas livré avec le dépôt.")
    limit = 200 * 1024 * 1024
    target = "Survived"
    task = "classification"

    # 1. Upload summary
    upload = wp.upload_summary(t_path, limit)
    assert upload["shape"] == [891, 12]

    # 2. Select target and task
    wp.validate_config(t_path, limit, target, task)

    # 3. Execute EDA
    eda = wp.eda(t_path, limit, target, task)
    assert eda["shape"] == [891, 12]

    # 4. Configure preprocessing & Analyze recommendations
    with tempfile.TemporaryDirectory() as tmp_dir:
        job_path = Path(tmp_dir) / "analysis.joblib"
        analysis = wp.analyze(t_path, limit, target, task, {}, {}, job_path)
        recs = {r["column"]: r for r in analysis["features"]}
        assert recs["PassengerId"]["action"] == "DROP"

    # 5. Request /api/selection-preview
    preview = wp.selection_preview(t_path, limit, target, task, {"selection_method": "variance"}, {})
    assert preview["status"] == "estimated_train_fit"
    assert preview["target_included"] is True
    assert "note" in preview and "estimé" in preview["note"].lower()

    # 6. Execute pipeline
    with tempfile.TemporaryDirectory(prefix="e2e-run-") as run_dir:
        folder = Path(run_dir)
        actual = wp.execute(t_path, limit, target, task, {"selection_method": "variance", "test_size_pct": 20}, {}, folder, "titanic.csv")

        # 7. Compare preview dimensions with actual execution dimensions
        # Expected train rows: 712, output columns: 15
        assert actual["train_shape"][0] == 712
        assert actual["test_shape"][0] == 179
        assert actual["train_shape"][1] == preview["output_count"] == 15
        assert set(actual["output_columns"]) == set(preview["selected"] + [target])

        # 8. Inspect the selection report
        report_data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        assert "features" in report_data

        # 9. Export train.csv and test.csv
        train_df = pd.read_csv(folder / "train.csv")
        test_df = pd.read_csv(folder / "test.csv")
        assert len(train_df) == 712
        assert len(test_df) == 179
        assert list(train_df.columns) == list(test_df.columns) == actual["output_columns"]

        # 10. Reload preprocessor
        prep = AutoPreprocessor.load(str(folder / "preprocesseur.joblib"))

        # 11. Transform new data (slice of raw data)
        raw_slice = pd.read_csv(t_path).iloc[:10].drop(columns=[target])
        new_trans = prep.transform(raw_slice)

        # 12. Compare resulting feature names and order
        expected_feature_cols = [c for c in actual["output_columns"] if c != target]
        assert list(new_trans.columns) == expected_feature_cols
