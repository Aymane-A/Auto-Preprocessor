"""HTTP contracts, data isolation, and exported workbook validation."""

from io import BytesIO
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split

from api_server import create_app

DATA = Path(__file__).parent / "data"


def payload(response, status=200):
    """Check status and reject nonstandard JSON constants."""
    assert response.status_code == status, response.text

    def bad_constant(value):
        """Fail on NaN and infinity literals."""
        pytest.fail(f"Non-finite JSON literal: {value}")

    return json.loads(response.text, parse_constant=bad_constant)


def upload(client, name, contents):
    """Upload one tabular file to the local API."""
    return payload(client.post("/api/upload", files={"file": (name, contents)}))


def configure(client, session_id, target=None):
    """Configure a dataset with or without a target."""
    return payload(client.post("/api/config", json={"session_id": session_id, "target": target,
                                                    "mode": "supervised" if target else "unsupervised"}))


@pytest.mark.parametrize(("dataset", "target", "raw_column", "feature", "alternative"), [
    ("fraud_signal_demo.csv", "is_fraud", "transaction_id", "montant", "scaling:MINMAX"),
    ("churn_telecom_demo.csv", "churn", "id_client", "contrat", "encoding:COUNT_ENCODE"),
])
def test_demo_flow(dataset, target, raw_column, feature, alternative):
    """Run both supervised demos through analysis, preview, fitting and download."""
    with TestClient(create_app(), base_url="http://localhost") as client:
        loaded = upload(client, dataset, (DATA / dataset).read_bytes())
        session_id = loaded["session_id"]
        body = {"session_id": session_id}
        assert len(loaded["preview"]) == 20
        configure(client, session_id, target)
        eda = payload(client.get(f"/api/eda/{session_id}"))
        assert len(eda["features"]) == loaded["shape"][1]
        assert eda["target_relations"]
        assert payload(client.get(f"/api/eda/{session_id}/feature/{feature}"))["distribution"]
        analysis = payload(client.post("/api/analyze", json=body))
        row = next(item for item in analysis["features"] if item["column"] == feature)
        assert row["justification"]["preuve"]
        assert alternative in {item["value"] for item in row["alternatives"]}
        preview = payload(client.post("/api/preview", json={**body, "column": feature, "choice": alternative}))
        assert preview["status"] == "ok"
        assert preview["before_chart"] and preview["after_charts"]
        result = payload(client.post("/api/preprocess", json={**body, "choices": {
            raw_column: "none", feature: alternative,
        }}))
        assert result["train_shape"][0] > result["test_shape"][0] > 0
        report = payload(client.get(f"/api/report/{session_id}"))
        tables = {}
        for split in ("train", "test", "full"):
            response = client.get(f"/api/download/{session_id}/{split}")
            assert response.status_code == 200
            tables[split] = pd.read_csv(BytesIO(response.content))
            assert list(tables[split]) == result["output_columns"]
        pd.testing.assert_frame_equal(tables["full"], pd.concat([tables["train"], tables["test"]], ignore_index=True))
        assert tables["full"][target].notna().all()
        response = client.get(f"/api/download/{session_id}/report_xlsx")
        assert response.status_code == 200
        workbook = load_workbook(BytesIO(response.content))
        assert workbook.sheetnames == ["Résumé", "Features", "Corrélations", "Colonnes de sortie"]
        header, *rows = list(workbook["Features"].values)
        assert len(rows) == loaded["shape"][1]
        xlsx = {row[0]: dict(zip(header, row)) for row in rows}
        core = {row["Colonne"]: row.get("Action appliquée") for row in report["core"]["colonnes_detaillees"]}
        for row in report["features"]:
            assert xlsx[row["column"]]["Transformation appliquée"] == row["applied"]
            if row["column"] in core and core[row["column"]] is not None:
                assert row["applied"] == core[row["column"]]
        assert xlsx[raw_column]["Transformation appliquée"] == "NONE"
        assert xlsx[feature]["Choix utilisateur"] == "alternative"


def test_train_only_and_target_alignment():
    """Verify a learned scaler and both exported targets against the exact split."""
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({"=raw": [f"row{i}" for i in range(120)], "measurement": rng.normal(size=120),
                          "label": np.arange(120) % 2})
    train, test = train_test_split(frame, test_size=.2, random_state=42, stratify=frame["label"])
    frame.loc[test.index, "measurement"] += 1000
    with TestClient(create_app(), base_url="http://localhost") as client:
        loaded = upload(client, "audit.csv", frame.to_csv(index=False).encode())
        session_id = loaded["session_id"]
        configure(client, session_id, "label")
        result = payload(client.post("/api/preprocess", json={"session_id": session_id,
            "options": {"outlier_method": "none", "skew_threshold": 100},
            "choices": {"=raw": "none", "measurement": "scaling:STANDARD"}}))
        report = payload(client.get(f"/api/report/{session_id}"))
        row = next(row for row in report["features"] if row["column"] == "measurement")
        assert row["parameters"]["scaling"]["mean_"] == pytest.approx(train["measurement"].mean())
        assert row["parameters"]["scaling"]["mean_"] != pytest.approx(frame["measurement"].mean())
        for split, expected in (("train", train), ("test", test)):
            table = pd.read_csv(BytesIO(client.get(f"/api/download/{session_id}/{split}").content))
            assert list(table) == result["output_columns"]
            assert set(table["=raw"]) == set(expected["=raw"])
            assert table["label"].tolist() == table["=raw"].map(frame.set_index("=raw")["label"]).tolist()
        book = load_workbook(BytesIO(client.get(f"/api/download/{session_id}/report_xlsx").content))
        assert book["Features"]["A2"].value == "'=raw"
        assert book["Features"]["A2"].data_type == "s"


def test_unsupervised_and_all_raw():
    """Exercise both unsupervised fitting and an all-passthrough split."""
    frame = pd.DataFrame({"count": np.sin(np.arange(40)), "group": ["a", "b"] * 20})
    with TestClient(create_app(), base_url="http://localhost") as client:
        loaded = upload(client, "plain.csv", frame.to_csv(index=False).encode())
        session_id = loaded["session_id"]
        configure(client, session_id)
        body = {"session_id": session_id}
        assert payload(client.get(f"/api/eda/{session_id}"))["target_relations"] == []
        assert len(payload(client.post("/api/analyze", json=body))["features"]) == 2
        assert payload(client.post("/api/preprocess", json=body))["test_shape"][0] == 8
        raw = payload(client.post("/api/preprocess", json={**body, "choices": {"count": "none", "group": "none"}}))
        assert raw["full_shape"] == [40, 2]
        table = pd.read_csv(BytesIO(client.get(f"/api/download/{session_id}/full").content))
        pd.testing.assert_frame_equal(table.sort_values("count").reset_index(drop=True),
                                      frame.sort_values("count").reset_index(drop=True))


def test_invalid_requests_and_cleanup():
    """Reject oversize, malformed, path-like and unauthorized requests and expire data."""
    app = create_app(max_table_bytes=64)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.post("/api/upload", files={"file": ("large.csv", b"a,b\n" * 20)}).status_code == 413
        assert client.post("/api/upload", files={"file": ("invalid.xlsx", b"not Excel")}).status_code == 422
        assert client.post("/api/upload", files={"file": ("payload.joblib", b"pickle")}).status_code == 415
        assert client.post("/api/upload", json={"path": "data/a.csv"}).status_code == 415
        assert client.post("/api/upload", headers={"Origin": "https://example.org"}).status_code == 403
        assert not list(app.state.store.root.iterdir())
        loaded = upload(client, "../source.csv", b"a,b\n1,1\n2,1\n")
        assert loaded["filename"] == "source.csv"
        session_id = loaded["session_id"]
        body = {"session_id": session_id, "mode": "supervised", "target": "b"}
        assert client.post("/api/config", json={**body, "path": "C:/data.csv"}).status_code == 422
        assert client.post("/api/config", json=body).status_code == 422
        session = app.state.store.sessions[session_id]
        session.expires = time.monotonic() - 1
        assert client.get(f"/api/eda/{session_id}").status_code == 404
        assert not session.folder.exists()


def test_excel_all_raw_supervised():
    """Read a real Excel upload and fit the target with every feature kept raw."""
    frame = pd.DataFrame({"clé source": [f"clé-{i}" for i in range(24)], "label": np.arange(24) % 2})
    buffer = BytesIO()
    frame.to_excel(buffer, index=False)
    with TestClient(create_app(), base_url="http://localhost") as client:
        loaded = upload(client, "source.xlsx", buffer.getvalue())
        session_id = loaded["session_id"]
        configure(client, session_id, "label")
        result = payload(client.post("/api/preprocess", json={"session_id": session_id,
                         "choices": {"clé source": "none"}}))
        assert result["full_shape"] == [24, 2]
        csv = client.get(f"/api/download/{session_id}/full")
        table = pd.read_csv(BytesIO(csv.content))
        pd.testing.assert_frame_equal(table.sort_values("clé source").reset_index(drop=True),
                                      frame.sort_values("clé source").reset_index(drop=True))
