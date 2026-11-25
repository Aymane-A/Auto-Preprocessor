import os
import time
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from app_preprocess import read_table
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, to_sparse_matrix
from isolated_processing import ProcessLimits, run_isolated
from model_evaluation import compare_preprocessing


def test_sparse_pipeline_preserves_dense_schema_and_values():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "value": rng.normal(size=80),
        "category": rng.choice(["a", "b", "c"], size=80),
        "text": [f"water bird tree term{i % 17}" for i in range(80)],
        "hashed": [f"topic_{i % 9}" for i in range(80)],
    })
    config = dict(verbose=0, preserve_rows=True, column_overrides={
        "text": {"type": "FORCE_TEXT", "text": "COUNT_VECTORIZER"},
        "hashed": {"encoding": ("HASH_ENCODE", {"n_components": 16})},
    })
    dense = AutoPreprocessor(PreprocessConfig(**config)).fit_transform(frame)
    prep = AutoPreprocessor(PreprocessConfig(**config, sparse_output=True))
    result = prep.fit_transform(frame)
    matrix = to_sparse_matrix(result)
    assert result.columns.equals(dense.columns)
    assert isinstance(result["category_a"].dtype, pd.SparseDtype)
    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_allclose(matrix.toarray(), dense.to_numpy(), atol=1e-6)


def test_sparse_pipeline_never_densifies_wide_text(monkeypatch):
    n = 150
    frame = pd.DataFrame({"text": [f"document unique token{i}" for i in range(n)]})
    config = dict(verbose=0, preserve_rows=True, max_dense_bytes=400,
                  correlation_threshold=1, column_overrides={
        "text": {"type": "FORCE_TEXT", "text": ("HASH_TEXT", {"n_features": 512})},
    })
    with pytest.raises(ValueError, match="sortie dense"):
        AutoPreprocessor(PreprocessConfig(**config)).fit_transform(frame)
    densify = Mock(side_effect=AssertionError("Sparse feature matrix was densified"))
    monkeypatch.setattr(sparse.csr_matrix, "toarray", densify)
    prep = AutoPreprocessor(PreprocessConfig(**config, sparse_output=True))
    features = prep.fit_transform(frame)
    unseen = prep.transform(pd.DataFrame({"text": ["unknown term", ""]}, index=[3, 3]))
    assert features.shape[0] == n and features.shape[1] > 100
    assert features.columns.equals(unseen.columns)
    assert unseen.index.tolist() == [3, 3]
    assert to_sparse_matrix(unseen).shape == unseen.shape
    densify.assert_not_called()


def test_sparse_cross_validation():
    frame = pd.DataFrame({"label": ["a", "b", "c"] * 24, "target": [0, 1] * 36})
    scores = compare_preprocessing(frame, "target", "logistic",
                                   config=PreprocessConfig(sparse_output=True, verbose=0),
                                   cv=2, candidates=("none",))
    assert np.isfinite(scores["results"][0]["scores"]).all()


def test_process_isolation_and_peak_ram(tmp_path):
    file = tmp_path / "isolated.csv"
    file.write_text("value\n1\n2\n", encoding="utf-8")
    pid = run_isolated(os.getpid)
    assert pid != os.getpid()
    frame, usage = run_isolated(read_table, file, with_usage=True)
    assert frame["value"].tolist() == [1, 2]
    assert usage.peak_rss_bytes > 0 and usage.elapsed_seconds > 0


def test_process_wall_timeout():
    with pytest.raises(TimeoutError, match="Délai"):
        run_isolated(time.sleep, 10, limits=ProcessLimits(timeout_seconds=.2))


@pytest.mark.skipif(os.name != "nt", reason="Checks the Windows Job Object commit limit")
def test_windows_process_memory_limit():
    with pytest.raises((ValueError, RuntimeError), match="mémoire|limite"):
        run_isolated(bytearray, 256 * 1024**2, limits=ProcessLimits(memory_bytes=96 * 1024**2))
