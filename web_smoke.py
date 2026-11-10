"""Exercise the running local service on demo or synthetic fraud-like data."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import httpx
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def create_large_csv(path):
    """Write 285,000 rows, 30 numeric predictors and a roughly 1% binary target."""
    rng = np.random.default_rng(42)
    positives = 0
    for start in range(0, 285_000, 15_000):
        values = rng.normal(size=(15_000, 30))
        label = (rng.random(len(values)) < (.006 + .015 * (values[:, 0] > .7))).astype(int)
        positives += int(label.sum())
        frame = pd.DataFrame(values, columns=[f"v{i:02}" for i in range(1, 31)])
        frame["is_fraud"] = label
        frame.to_csv(path, mode="a" if start else "w", header=not start, index=False, float_format="%.15f")
    return {"bytes": path.stat().st_size, "rows": 285_000, "columns": 31, "positives": positives}


def checked(client, method, path, **kwargs):
    """Check a live HTTP request and print its duration."""
    start = perf_counter()
    response = client.request(method, path, **kwargs)
    print(f"{method} {path.split('/')[2] if path.startswith('/api/') else 'frontend'}: {response.status_code}, {perf_counter() - start:.2f} s", flush=True)
    response.raise_for_status()
    return response


def smoke(client, path, target, large):
    """Upload, configure, process and verify downloadable results via HTTP."""
    response = checked(client, "GET", "/")
    assert "AutoPrépa" in response.text
    with path.open("rb") as handle:
        upload = checked(client, "POST", "/api/upload", files={"file": (path.name, handle)}).json()
    print(json.dumps({"uploaded": upload["shape"], "upload_worker": upload["worker"]}), flush=True)
    session = upload["session_id"]
    body = {"session_id": session}
    checked(client, "POST", "/api/config", json={**body, "mode": "supervised", "target": target})
    choices = {}
    if not large:
        feature, raw, option = (("montant", "transaction_id", "scaling:MINMAX") if target == "is_fraud"
                                else ("contrat", "id_client", "encoding:COUNT_ENCODE"))
        checked(client, "GET", f"/api/eda/{session}")
        checked(client, "GET", f"/api/eda/{session}/feature/{feature}")
        checked(client, "POST", "/api/analyze", json=body)
        checked(client, "POST", "/api/preview", json={**body, "column": feature, "choice": option})
        choices = {feature: option, raw: "none"}
    result = checked(client, "POST", "/api/preprocess", json={**body, "choices": choices}).json()
    print(json.dumps({key: result[key] for key in ("input_shape", "train_shape", "test_shape", "full_shape", "worker")}),
          flush=True)
    report = checked(client, "GET", f"/api/report/{session}").json()
    workbook = checked(client, "GET", f"/api/download/{session}/report_xlsx")
    with BytesIO(workbook.content) as data:
        book = load_workbook(data, read_only=True)
        assert book["Features"].max_row == len(report["features"]) + 1
        book.close()
    if not large:
        for split in ("train", "test", "full"):
            data = checked(client, "GET", f"/api/download/{session}/{split}")
            frame = pd.read_csv(BytesIO(data.content))
            assert list(frame) == result["output_columns"]
            assert list(frame.shape) == result[f"{split}_shape"]


def main():
    """Run the selected demo against localhost, generating temporary data if needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fraud", "churn", "large"), default="fraud")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Loopback server used for this validation run.")
    args = parser.parse_args()
    data = Path(__file__).parent / "data"
    with httpx.Client(base_url=args.base_url, timeout=650) as client:
        if args.dataset == "large":
            with TemporaryDirectory(prefix="autoprep-stress-") as folder:
                path = Path(folder) / "fraud_large.csv"
                print(json.dumps(create_large_csv(path)), flush=True)
                smoke(client, path, "is_fraud", True)
        else:
            name, target = (("fraud_signal_demo.csv", "is_fraud") if args.dataset == "fraud"
                            else ("churn_telecom_demo.csv", "churn"))
            smoke(client, data / name, target, False)


if __name__ == "__main__":
    main()
