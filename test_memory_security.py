import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import auto_preprocessor as ap
from app_preprocess import App
from signal_check import _finite_model_matrix


@pytest.mark.parametrize("kind", ["robust", "ngram"])
def test_large_text_rejected_before_vectorizing_or_densifying(kind, monkeypatch):
    rows, width = 10_001, 2_000
    train = pd.DataFrame({"text": ["training word"]})
    if kind == "robust":
        transformer = ap.RobustTextTransformer(
            max_features=width, column_modes={"text": "HASH"}, max_dense_bytes=1_000_000_000)
    else:
        transformer = ap.NgramTextTransformer(
            methods={"text": "HASH_TEXT"}, params={"text": {"n_features": width}},
            max_dense_bytes=1_000_000_000)
    transformer.fit(train)
    monkeypatch.setattr(transformer.vectorizers_["text"], "transform",
                        Mock(side_effect=AssertionError("Vectorized before preflight")))
    monkeypatch.setattr(sparse.csr_matrix, "toarray", Mock(side_effect=AssertionError("Densified")))
    batch = pd.DataFrame({"text": ["unseen word"] * rows})
    with pytest.raises(ValueError, match="20,002,000 cellules"):
        transformer.transform(batch)


def test_wide_vocabulary_sparse_output_keeps_dimensions_and_index(monkeypatch):
    width = 2_048
    train = pd.DataFrame({"text": [f"term{i}" for i in range(width)]})
    batch = pd.DataFrame({"text": ["term37", "unseen", ""]}, index=[5, 5, 2])
    options = dict(methods={"text": "COUNT_VECTORIZER"}, default_max_features=width, max_dense_bytes=256)
    with pytest.raises(ValueError, match="24,576 octets"):
        ap.NgramTextTransformer(**options).fit(train).transform(batch)
    transformer = ap.NgramTextTransformer(**options, sparse_output=True).fit(train)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", Mock(side_effect=AssertionError("Densified")))
    result = transformer.transform(batch)
    assert result.shape == (3, width)
    assert result.index.equals(batch.index)
    assert result.columns.tolist() == transformer.get_feature_names_out().tolist()
    matrix = result.sparse.to_coo().tocsr()
    assert matrix.shape == (3, width) and matrix.nnz == 1
    assert matrix[0, result.columns.get_loc("text_cnt_term37")] == 1
    assert transformer.transform(batch.iloc[:0]).shape == (0, width)


@pytest.mark.parametrize("sparse_output", [False, True])
def test_robust_text_empty_unseen_and_zero_rows(sparse_output):
    train = pd.DataFrame({"text": ["known token", "known other"], "blank": [None, ""]})
    transformer = ap.RobustTextTransformer(column_modes={"text": "TFIDF"},
                                           sparse_output=sparse_output).fit(train)
    batch = pd.DataFrame({"text": ["newword", None]}, index=[3, 3])
    result = transformer.transform(batch)
    assert result.columns.tolist() == transformer.get_feature_names_out().tolist()
    assert result.index.equals(batch.index)
    assert (result == 0).all().all()
    assert transformer.transform(batch.iloc[:0]).shape == (0, result.shape[1])


def test_svd_workspace_checked_before_fit(monkeypatch):
    fail_if_called = Mock(side_effect=AssertionError("SVD fit allocated before preflight"))
    monkeypatch.setattr(ap.TruncatedSVD, "fit", fail_if_called)
    frame = pd.DataFrame({"text": [f"quartz copper crystal term{i}" for i in range(40)]})
    with pytest.raises(ValueError, match="Espace de travail SVD"):
        ap.RobustTextTransformer(svd_components=2, max_dense_bytes=1_000).fit(frame)
    fail_if_called.assert_not_called()


def test_schema_limit_rejects_oversized_hash_configuration():
    transformer = ap.NgramTextTransformer(methods={"text": "HASH_TEXT"},
                                           default_n_features=ap.MAX_FEATURE_COLUMNS + 1,
                                           sparse_output=True)
    with pytest.raises(ValueError, match="100,000 colonnes"):
        transformer.fit(pd.DataFrame({"text": ["small input"]}))


def test_hash_and_onehot_rejected_before_encoder_runs(monkeypatch):
    frame = pd.DataFrame({"category": ["first", "second", "third"] * 6})
    state = {"method": "HASH_ENCODE", "n_components": 16}
    fail_if_called = Mock(side_effect=AssertionError("Encoder ran before preflight"))
    monkeypatch.setattr(ap.FeatureHasher, "transform", fail_if_called)
    with pytest.raises(ValueError, match="Encodage catégoriel.*2,304 octets"):
        ap.AutoPreprocessor._apply_custom_encoder("category", state, frame, max_dense_bytes=1_000)
    prep = ap.AutoPreprocessor(ap.PreprocessConfig(verbose=0, preserve_rows=True)).fit(frame)
    monkeypatch.setattr(prep.ohe_encoder_, "transform", fail_if_called)
    prep.config.max_dense_bytes = 400
    with pytest.raises(ValueError, match="Matrice transformée.*432 octets"):
        prep.transform(frame)
    fail_if_called.assert_not_called()


def test_combined_pipeline_budget_catches_individually_small_blocks(monkeypatch):
    frame = pd.DataFrame({"left": ["red", "blue", "green", "white"],
                          "right": ["river", "sea", "lake", "pond"]})
    overrides = {
        "left": {"type": "FORCE_TEXT", "text": "TFIDF"},
        "right": {"type": "FORCE_TEXT", "text": "COUNT_VECTORIZER"},
    }
    prep = ap.AutoPreprocessor(ap.PreprocessConfig(
        column_overrides=overrides, preserve_rows=True, verbose=0, max_dense_bytes=300)).fit(frame)
    fail_if_called = Mock(side_effect=AssertionError("Text conversion ran before combined preflight"))
    monkeypatch.setattr(prep.text_transformer_, "transform", fail_if_called)
    monkeypatch.setattr(prep.ngram_text_transformer_, "transform", fail_if_called)
    with pytest.raises(ValueError, match="Matrice transformée.*512 octets"):
        prep.transform(pd.concat([frame, frame]))
    fail_if_called.assert_not_called()


def test_sparse_signal_matrix_is_sanitized_without_densifying():
    matrix = sparse.csr_matrix(([np.nan, 2.0], ([0, 1], [3, 1])), shape=(2, 250_000))
    clean = _finite_model_matrix(matrix)
    assert sparse.issparse(clean) and clean.shape == matrix.shape
    assert clean[0, 3] == 0 and clean[1, 1] == 2
    assert np.isnan(matrix[0, 3])


def test_duplicate_detection_has_no_pandas_downcast_warning():
    frame = pd.DataFrame({"a": pd.Series([1, 3, 7, 8], dtype=object),
                          "b": pd.Series([1, 3, 7, 8], dtype=object)})
    prep = ap.AutoPreprocessor(ap.PreprocessConfig(verbose=0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        prep._detect_types(frame)
    assert prep.column_metadata_["b"]["type"] == ap.FeatureType.DROP_DUPLICATE


def test_mixed_pipeline_preserves_duplicate_indices_and_target_positions():
    frame = pd.DataFrame({"value": [-1., 21., 23., 37., 46., 51.],
                          "category": ["a", "b", "c", "a", "b", "c"],
                          "target": [5., 9., 11., 13., 17., 19.]},
                         index=[4, 4, 9, 4, 9, 9])
    prep = ap.AutoPreprocessor(ap.PreprocessConfig(
        value_bounds={"value": (0, None)}, transform_target=False, verbose=0,
        column_overrides={"value": {"clean_impossible": "DROP_ROWS", "scaling": "STANDARD"},
                          "category": {"encoding": ("HASH_ENCODE", {"n_components": 8})}}), task="regression")
    X, y = prep.fit_transform_xy(frame, "target")
    assert X.index.tolist() == [4, 9, 4, 9, 9]
    assert y.tolist() == [9., 11., 13., 17., 19.]
    assert "value" in X and any(name.startswith("category_hash_") for name in X)
    assert np.isfinite(X.to_numpy()).all()


@pytest.mark.parametrize("answer", [False, None, "yes"])
def test_model_load_requires_explicit_boolean_confirmation(tmp_path, monkeypatch, answer):
    path = tmp_path / "untrusted.joblib"
    path.write_bytes(b"untrusted payload")
    app = App.__new__(App)
    app.prep = previous = object()
    app.messagebox = Mock()
    app.messagebox.askyesno.return_value = answer
    load = Mock(side_effect=AssertionError("Untrusted model deserialized"))
    monkeypatch.setattr(ap.joblib, "load", load)
    app._do_load_preprocessor(str(path))
    app.messagebox.askyesno.assert_called_once()
    app.messagebox.showerror.assert_not_called()
    load.assert_not_called()
    assert app.prep is previous


def test_trusted_model_loads_only_after_prompt(tmp_path, monkeypatch):
    path = tmp_path / "trusted.joblib"
    frame = pd.DataFrame({"value": [2., 3., 5., 7., 11.]})
    prep = ap.AutoPreprocessor(ap.PreprocessConfig(verbose=0)).fit(frame)
    prep.save(str(path))
    app = App.__new__(App)
    app.messagebox = Mock()
    app.messagebox.askyesno.return_value = True
    app._set_status = Mock()
    original_load = ap.joblib.load

    def checked_load(filename):
        app.messagebox.askyesno.assert_called_once()
        return original_load(filename)

    monkeypatch.setattr(ap.joblib, "load", checked_load)
    app._do_load_preprocessor(str(path))
    app.messagebox.showerror.assert_not_called()
    app._set_status.assert_called_once()
    pd.testing.assert_frame_equal(app.prep.transform(frame), prep.transform(frame))
