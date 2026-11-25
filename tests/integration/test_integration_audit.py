"""Dataset-independent HTTP regression tests for the integration audit."""
from io import BytesIO
from importlib.util import find_spec
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api_server import create_app


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed", "wide", "regression", "tiny", "medium", "large"])
def test_dataset_independent_contracts(kind):
    rng = np.random.default_rng(83)
    count = {"tiny": 3, "medium": 1000, "large": 20000}.get(kind, 64)
    frame = pd.DataFrame({"âge / 年齢 ?#%": rng.normal(size=count)})
    if kind in {"medium", "large"}:
        frame = pd.DataFrame(rng.normal(size=(count, 6)), columns=[f"mesure {i}" for i in range(6)])
    elif kind == "categorical":
        frame = pd.DataFrame({"بلد": ["Maroc", "France"] * 32, "constructor": ["oui", "non", "?"] * 21 + [None]})
    elif kind == "mixed":
        frame["__proto__"] = [f"id-{i}" for i in range(count)]
        frame["booléen"] = np.arange(count) % 2 == 0
        frame["date heure"] = pd.date_range("2024-01-01", periods=count, freq="h")
        frame["texte"] = [f"Le client décrit une opération numéro {i} avec des détails" for i in range(count)]
        frame["constante"] = 7
        frame["vide"] = np.nan
        frame.loc[:20, "âge / 年齢 ?#%"] = np.nan
        frame = pd.concat([frame, frame.iloc[[22]]], ignore_index=True)
    elif kind == "wide":
        frame = pd.DataFrame(rng.normal(size=(count, 45)), columns=[f"mesure {i}" for i in range(45)])
    elif kind == "regression":
        frame["résultat continu"] = rng.lognormal(size=count)
    target = "résultat continu" if kind == "regression" else None
    with TestClient(create_app(), base_url="http://localhost") as client:
        uploaded = client.post("/api/upload", files={"file": ("fixture.csv", frame.to_csv(index=False).encode())})
        assert uploaded.status_code == 200, uploaded.text
        summary = uploaded.json()
        assert summary["shape"] == list(frame.shape)
        assert summary["columns"] == list(frame.columns)
        sid = summary["session_id"]
        config = {"session_id": sid, "mode": "supervised" if target else "unsupervised", "target": target, "task": "regression" if target else "auto"}
        assert client.post("/api/config", json=config).status_code == 200
        eda = client.get(f"/api/eda/{sid}")
        assert eda.status_code == 200, eda.text
        assert {f["column"] for f in eda.json()["features"]} == set(frame.columns)
        for column in frame.columns if kind != "wide" else frame.columns[:1]:
            detail = client.get(f"/api/eda/{sid}/feature/{quote(column, safe='')}")
            assert detail.status_code == 200, detail.text
            assert detail.json()["column"] == column
        choices = {column: "none" for column in frame if column != target}
        body = {"session_id": sid, "choices": choices}
        analysis = client.post("/api/analyze", json=body)
        assert analysis.status_code == 200, analysis.text
        assert [f["column"] for f in analysis.json()["features"]] == list(frame.columns)
        column = next(c for c in frame if c != target)
        preview = client.post("/api/preview", json={"session_id": sid, "column": column, "choice": "none"})
        assert preview.status_code == 200, preview.text
        done = client.post("/api/preprocess", json=body)
        assert done.status_code == 200, done.text
        data = client.get(f"/api/download/{sid}/full")
        assert data.status_code == 200
        exported = pd.read_csv(BytesIO(data.content))
        assert list(exported.shape) == done.json()["full_shape"]
        # Every original row remains present when all predictors are explicitly raw.
        assert len(exported) == len(frame)
        assert set(frame.columns) <= set(exported.columns)
        assert client.get(f"/api/report/{sid}").status_code == 200
        assert client.post("/api/config", json={**config, "target": "absent", "mode": "supervised"}).status_code == 422
        assert client.get(f"/api/download/{sid}/full").content == data.content
        assert client.delete(f"/api/session/{sid}").status_code == 200
        assert client.get(f"/api/eda/{sid}").status_code == 404


def test_evaluation_catalog_matches_installed_backends():
    with TestClient(create_app(), base_url="http://localhost") as client:
        catalog = client.get("/api/capabilities").json()["evaluation"]["properties"]
        assert catalog["folds"]["minimum"] == 2
        assert catalog["folds"]["maximum"] == 5
        assert catalog["folds"]["default"] == 3
        assert "linear" in catalog["family"]["enum"]
        for family in ("xgboost", "lightgbm"):
            assert (family in catalog["family"]["enum"]) == (find_spec(family) is not None)


def test_same_origin_frontend_on_alternate_loopback_port():
    with TestClient(create_app(), base_url="http://localhost:8011") as client:
        assert client.get("/api/capabilities", headers={"Origin": "http://localhost:8011"}).status_code == 200
        assert client.get("/api/capabilities", headers={"Origin": "https://example.org"}).status_code == 403
