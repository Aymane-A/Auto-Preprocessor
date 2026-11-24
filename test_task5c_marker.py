"""The provenance marker must not become an exported model feature."""
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
from openpyxl import load_workbook

from api_server import create_app


def test_smote_marker_is_only_in_separate_train_provenance_csv():
    source = Path(__file__).parent / "data" / "fraud_signal_demo.csv"
    with TestClient(create_app(), base_url="http://localhost") as client:
        uploaded = client.post("/api/upload", files={"file": (source.name, source.read_bytes())})
        assert uploaded.status_code == 200, uploaded.text
        sid = uploaded.json()["session_id"]
        configured = client.post("/api/config", json={"session_id": sid, "mode": "supervised",
                                                   "target": "is_fraud", "task": "classification"})
        assert configured.status_code == 200, configured.text
        response = client.post("/api/preprocess", json={"session_id": sid,
                                                         "options": {"imbalance_method": "smote"}})
        assert response.status_code == 200, response.text
        result = response.json()
        tables = {}
        for kind in ("train", "test", "full", "train_provenance"):
            download = client.get(f"/api/download/{sid}/{kind}")
            assert download.status_code == 200, download.text
            tables[kind] = pd.read_csv(BytesIO(download.content))
        assert "__synthetic__" not in result["output_columns"]
        assert all("__synthetic__" not in tables[kind] for kind in ("train", "test", "full"))
        provenance = tables["train_provenance"]
        assert list(provenance) == ["row_index", "__synthetic__"]
        assert len(provenance) == result["imbalance"]["synthetic_rows"] > 0
        assert provenance["__synthetic__"].all()
        assert provenance["row_index"].is_unique
        assert provenance["row_index"].between(0, len(tables["train"]) - 1).all()
        assert tables["train"].iloc[provenance["row_index"]]["is_fraud"].nunique() == 1
        report = client.get(f"/api/report/{sid}").json()
        documented = report["summary"]["Gestion du déséquilibre"]["provenance"]
        assert documented["file"] == "train_provenance.csv"
        assert documented["row_index_base"] == 0
        assert documented["marked_rows"] == len(provenance)
        workbook = load_workbook(BytesIO(client.get(f"/api/download/{sid}/report_xlsx").content), read_only=True)
        assert any("train_provenance.csv" in str(row[1]) for row in workbook["Résumé"].values
                   if row[0] == "Gestion du déséquilibre")
