"""Regression coverage for the shared web options and composed action editor."""
from io import BytesIO
from pathlib import Path
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from api_server import create_app
from auto_preprocessor import SUPPORTED_ACTIONS
from web_contracts import WebOptions
import web_processing as work


@pytest.fixture(scope="module")
def action_source(tmp_path_factory):
    rng = np.random.default_rng(51)
    count = 96
    frame = pd.DataFrame({
        "x": rng.lognormal(2, .5, count), "aux": rng.normal(size=count),
        "category": ["a", "b", "c"] * 32,
        "date": pd.date_range("2020-01-01", periods=count).strftime("%Y-%m-%d"),
        "text": [f"Le client décrit {['paiement carte bancaire commerce', 'virement urgent international agence', 'retrait liquide guichet proximité', 'prêt investissement remboursement crédit'][i % 4]} séquence {i % 13}" for i in range(count)],
        "numeric_text": [f"{i % 30 + 1},50 €" for i in range(count)],
        "label": np.arange(count) % 2,
    })
    frame.loc[[3, 8, 11], "x"] = np.nan
    frame.loc[[2, 7], "category"] = "?"
    path = tmp_path_factory.mktemp("actions") / "actions.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def action_context(action_source):
    """Reuse the unchanged input profile; each action receives private copies."""
    frame = work.load_frame(action_source, 1_000_000)
    analysis = work.analyze_frame(frame, "label", work.processing_options("classification"))
    return frame, analysis


ACTION_CASES = sorted({(step, action) for step, groups in SUPPORTED_ACTIONS.items()
                       for actions in groups.values() for action in actions})


@pytest.mark.parametrize("step,action", ACTION_CASES, ids=[f"{step}-{action}" for step, action in ACTION_CASES])
def test_each_public_step_action(action_context, step, action):
    column = {"encoding": "category", "dates": "date", "text": "text", "clean_dates": "date",
              "clean_numeric_text": "numeric_text", "clean_missing_tokens": "category"}.get(step, "x")
    if action == "FORCE_NUMERIC":
        column = "numeric_text"
    if action == "FORCE_DATE":
        column = "date"
    if action == "FORCE_TEXT":
        column = "text"
    if action == "MISSING_CATEGORY":
        column = "category"
    parameters = {"CONSTANT": "7", "SELECTED_TO_NAN": "?", "MANUAL_MAP": '{"a":0,"b":1,"c":2}',
                  "ORDINAL": "a,b,c", "WINSORIZE": "0.01,0.99", "TFIDF_SVD": "2",
                  "HASH_ENCODE": "8", "HASH_TEXT": "1,2,16", "TFIDF_NGRAMS": "1,2,12",
                  "COUNT_VECTORIZER": "1,2,12", "LOO_ENCODE": "0"}.get(action, "")
    # The original engine catalog restricts BINARY_MAP to genuinely binary inputs.
    frame, analysis = action_context
    frame = frame.copy(deep=True)
    if action == "BINARY_MAP":
        frame["category"] = ["a", "b"] * 48
    if action == "TARGET_ENCODE":
        frame["category"] = [f"group{i % 24}" for i in range(96)]
    if action in {"BINARY_MAP", "TARGET_ENCODE"}:
        analysis = work.analyze_frame(frame, "label", work.processing_options("classification"))
    else:
        analysis = deepcopy(analysis)
    row = next(row for row in analysis.columns if row["colonne"] == column)
    assert action in row["steps"][step]["choices"], (column, step, action, row["type"])
    overrides, selected = work.selected_overrides(analysis, "label", {column: {step: {"action": action, "parameters": parameters}}})
    assert column in overrides
    from app_preprocess import preview_for_choice, build_config
    config = build_config(work.processing_options("classification"))
    config.clean_column_names = False
    result = preview_for_choice(frame, analysis, column, selected, config, "label", "classification")
    assert result["status"] in {"ok", "dropped", "removed_after_selection"}
    assert len(result["samples"]) <= 12


def test_shared_options_composed_choices_and_new_exports():
    frame = pd.DataFrame({"raw": [f"=row{i}" for i in range(80)], "Measure Name": np.random.default_rng(9).uniform(1, 100, 80),
                          "category": ["a", "b"] * 40, "label": np.arange(80) % 2})
    frame.loc[4, "Measure Name"] = np.nan
    choices = {"raw": "none", "Measure Name": {
        "missing": {"action": "CONSTANT", "parameters": "7"},
        "distribution": {"action": "NONE", "parameters": ""},
        "scaling": {"action": "MINMAX", "parameters": ""},
    }}
    options = {"variance_threshold": 0., "min_features_for_collinearity": 2,
               "value_bounds": {"Measure Name": [0, 150]}, "noise_guard": "off"}
    with TestClient(create_app(), base_url="http://localhost") as client:
        catalog = client.get("/api/capabilities")
        assert catalog.status_code == 200
        assert set(catalog.json()["options"]["properties"]) == set(WebOptions.model_fields)
        loaded = client.post("/api/upload", files={"file": ("audit.csv", frame.to_csv(index=False).encode())}).json()
        sid = loaded["session_id"]
        assert client.post("/api/config", json={"session_id": sid, "mode": "supervised", "target": "label"}).status_code == 200
        body = {"session_id": sid, "options": options, "choices": choices}
        analyzed = client.post("/api/analyze", json=body)
        assert analyzed.status_code == 200, analyzed.text
        preview = client.post("/api/preview", json={"session_id": sid, "column": "Measure Name", "choice": choices["Measure Name"], "options": options})
        assert preview.status_code == 200, preview.text
        assert preview.json()["applied"]["scaling"] == "MinMaxScaler"
        done = client.post("/api/preprocess", json=body)
        assert done.status_code == 200, done.text
        report = client.get(f"/api/report/{sid}").json()
        assert report["summary"]["Options appliquées"]["noise_guard"] == "off"
        feature = next(row for row in report["features"] if row["column"] == "Measure Name")
        assert feature["parameters"]["fill_value"] == 7
        assert feature["choice"] == choices["Measure Name"]
        assert "steps" in feature and "diagnostics" in feature
        assert client.get(f"/api/results-eda/{sid}").json()["shape"] == done.json()["full_shape"]
        assert client.post("/api/export-excel", json={"session_id": sid}).status_code == 200
        exported = client.get(f"/api/download/{sid}/full_xlsx")
        book = load_workbook(BytesIO(exported.content), read_only=True)
        rows = list(book.active.values)
        assert len(rows) == done.json()["full_shape"][0] + 1
        raw_index = rows[0].index("raw")
        assert all(str(row[raw_index]).startswith("'=row") for row in rows[1:])
        book.close()
        assert client.get(f"/api/download/{sid}/report_json").json()["features"] == report["features"]
        import joblib
        fitted = joblib.load(BytesIO(client.get(f"/api/download/{sid}/preprocessor").content))["model"]
        transformed = fitted.transform(frame.drop(columns=["raw", "label"]))
        assert set(transformed.columns) == set(done.json()["output_columns"]) - {"raw", "label"}
        assert client.post("/api/analyze", json={**body, "choices": {"Measure Name": {"missing": {"action": "CONSTANT", "parameters": "nan"}}}}).status_code == 422
        assert client.post("/api/analyze", json={**body, "options": {"outlier_lower_quantile": .9, "outlier_upper_quantile": .1}}).status_code == 422


@pytest.mark.parametrize("step,action,parameters", [
    ("missing", "KNN", "3"), ("distribution", "QUANTILE", "12"),
    ("outliers", "IQR", "2"), ("outliers", "Z_SCORE", "2.5"),
    ("outliers", "MODIFIED_Z", "3"), ("outliers", "REPLACE_MEDIAN", "2"),
    ("text", "TFIDF_NGRAM", "1,2,12"), ("text", "COUNT", "1,2,12"), ("text", "HASH", "1,2,16"),
])
def test_step_parameters_reach_preview(action_source, step, action, parameters):
    column = "text" if step == "text" else "x"
    result = work.preview(action_source, 1_000_000, "label", "classification", column,
                          {step: {"action": action, "parameters": parameters}})
    assert result["status"] == "ok"
    assert result["samples"]


def test_forced_type_updates_applicable_actions(tmp_path):
    path = tmp_path / "types.csv"
    frame = pd.DataFrame({"x": [10] * 40 + [20] * 20 + [30] * 12, "label": np.arange(72) % 2})
    frame.to_csv(path, index=False)
    choice = {"type": {"action": "FORCE_CATEGORICAL", "parameters": ""}, "encoding": {"action": "COUNT_ENCODE", "parameters": ""}}
    result = work.analyze(path, 1_000_000, "label", "classification", choices={"x": choice})
    row = next(row for row in result["features"] if row["column"] == "x")
    assert row["type"] == "CATEGORICAL_LOW_CARD"
    assert work.preview(path, 1_000_000, "label", "classification", "x", choice)["status"] == "ok"


def test_web_config_covers_engine_options():
    from dataclasses import fields
    from auto_preprocessor import PreprocessConfig
    from app_preprocess import build_config
    internal = {"column_overrides", "verbose", "preserve_rows", "max_dense_bytes"}
    assert {field.name for field in fields(PreprocessConfig)} - internal <= set(WebOptions.model_fields)
    options = WebOptions(max_text_features=23, variance_threshold=.05, clean_column_names=False, supervised_encoding=False)
    config = build_config(work.processing_options("auto", options.model_dump()))
    assert config.max_text_features == 23 and config.variance_threshold == .05
    assert not config.clean_column_names and not config.supervised_encoding


@pytest.mark.parametrize("format", ["csv", "tsv", "txt", "xlsx", "json", "parquet"])
def test_import_formats_and_complete_report(format):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"x": rng.normal(size=40), "category": ["a", "b"] * 20})
    stream = BytesIO()
    if format in {"csv", "tsv", "txt"}:
        content = frame.to_csv(index=False, sep="\t" if format == "tsv" else ",").encode()
    elif format == "xlsx":
        frame.to_excel(stream, index=False)
        content = stream.getvalue()
    elif format == "json":
        content = frame.to_json(orient="records").encode()
    else:
        frame.to_parquet(stream, index=False)
        content = stream.getvalue()
    with TestClient(create_app(), base_url="http://localhost") as client:
        response = client.post("/api/upload", files={"file": (f"table.{format}", content)})
        assert response.status_code == 200, response.text
        sid = response.json()["session_id"]
        assert client.post("/api/config", json={"session_id": sid, "mode": "unsupervised"}).status_code == 200
        done = client.post("/api/preprocess", json={"session_id": sid})
        assert done.status_code == 200, done.text
        report = client.get(f"/api/report/{sid}").json()
        book = load_workbook(BytesIO(client.get(f"/api/download/{sid}/report_xlsx").content))
        assert book.sheetnames == ["Résumé", "Features", "Corrélations", "Colonnes de sortie", "Opérations", "Diagnostics"]
        assert book["Features"].max_row == 3
        expected = {(feature["column"], step) for feature in report["features"] for step in feature["steps"]}
        assert {(row[0], row[1]) for row in list(book["Opérations"].values)[1:]} == expected
        decoded = {}
        for column, key, part, value in list(book["Diagnostics"].values)[1:]:
            decoded.setdefault((column, key), []).append((part, value))
        for feature in report["features"]:
            for key, value in feature["diagnostics"].items():
                parts = [text[1:] if text.startswith(("'=", "'+", "'-", "'@")) else text
                         for _, text in sorted(decoded[feature["column"], key])]
                assert json.loads("".join(parts)) == value
        book.close()
        assert client.get(f"/api/download/{sid}/preprocessor").status_code == 200
        session = client.app.state.store.sessions[sid]
        session.lock.acquire()
        assert client.delete(f"/api/session/{sid}").status_code == 409
        session.lock.release()
        assert client.delete(f"/api/session/{sid}").status_code == 200
        assert client.get(f"/api/report/{sid}").status_code == 404
        assert not session.folder.exists()


@pytest.mark.parametrize("family", ["linear", "logistic", "decision_tree", "random_forest", "gradient_boosting", "knn", "svm", "neural_network", "xgboost", "lightgbm"])
def test_predictive_comparison_endpoint(family):
    rng = np.random.default_rng(41)
    frame = pd.DataFrame({"x": rng.normal(size=64), "z": rng.normal(size=64)})
    frame["label"] = np.arange(64) % 2
    with TestClient(create_app(), base_url="http://localhost") as client:
        sid = client.post("/api/upload", files={"file": ("cv.csv", frame.to_csv(index=False).encode())}).json()["session_id"]
        assert client.post("/api/config", json={"session_id": sid, "mode": "supervised", "target": "label"}).status_code == 200
        response = client.post("/api/evaluate", json={"session_id": sid, "family": family, "folds": 2, "candidates": ["standard"]})
        if response.status_code == 422 and family in {"xgboost", "lightgbm"}:
            assert "bibliothèque" in response.json()["detail"]
        else:
            assert response.status_code == 200, response.text
            assert response.json()["adaptive_recommendations"] is False
            assert len(response.json()["results"][0]["scores"]) == 2
            assert response.json()["results"][0]["scaler"] == "standard"


def test_regression_target_export_and_temporal_validation():
    rng = np.random.default_rng(15)
    frame = pd.DataFrame({"row_key": [f"r{i}" for i in range(80)], "x": rng.normal(size=80),
                          "date": pd.date_range("2020-01-01", periods=80).strftime("%Y-%m-%d"),
                          "target": rng.lognormal(2, 1.4, 80)})
    with TestClient(create_app(), base_url="http://localhost") as client:
        sid = client.post("/api/upload", files={"file": ("regression.csv", frame.to_csv(index=False).encode())}).json()["session_id"]
        assert client.post("/api/config", json={"session_id": sid, "mode": "supervised", "target": "target", "task": "regression"}).status_code == 200
        options = {"target_skew_threshold": .1, "test_size_pct": 25, "noise_guard": "off"}
        done = client.post("/api/preprocess", json={"session_id": sid, "options": options, "choices": {"row_key": "none"}})
        assert done.status_code == 200, done.text
        assert done.json()["train_shape"][0] == 60 and done.json()["test_shape"][0] == 20
        table = pd.read_csv(BytesIO(client.get(f"/api/download/{sid}/full").content))
        assert "target_original" in table
        np.testing.assert_allclose(table["target_original"], table["row_key"].map(frame.set_index("row_key")["target"]))
        comparison = client.post("/api/evaluate", json={"session_id": sid, "family": "linear", "folds": 2,
                                                       "time_column": "date", "candidates": ["none"], "options": options})
        assert comparison.status_code == 200, comparison.text
        assert comparison.json()["scoring"] == "neg_mean_absolute_error"
