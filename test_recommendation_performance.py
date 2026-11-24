"""Exact parser equivalence and recommendation request lifecycle regressions."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

import api_server
from auto_preprocessor import AutoPreprocessor
import web_processing as work


@pytest.mark.parametrize('dtype', [object, 'string', 'category'])
def test_distinct_parser_matches_scalar_for_every_row_and_duplicate_index(dtype):
    values = pd.Series(['67,50 MAD', 'MAD 67', '12 kg', 'C1004', '2024-01-05', '?', None,
                        '1.234,56', '1,234.56', '-1.5', '50%', 'nan', 'NaT', '', '67,50 MAD'], dtype=dtype)
    values.index = [3] * len(values)
    expected = values.astype(object).map(AutoPreprocessor._clean_numeric_str_value).astype(float)
    actual = AutoPreprocessor._parse_numeric_series(values)
    pd.testing.assert_series_equal(actual, expected)
    assert AutoPreprocessor._parse_numeric_series(values.iloc[:0]).empty


def test_distinct_parser_keeps_mixed_scalar_types_separate():
    values = pd.Series([True, 1, False, 0, 1., np.nan, pd.NA, pd.NaT, None,
                        pd.Timestamp('2020-01-01'), np.datetime64('2024'), '1', 'True'], dtype=object)
    pd.testing.assert_series_equal(AutoPreprocessor._parse_numeric_series(values),
                                  values.map(AutoPreprocessor._clean_numeric_str_value).astype(float))


def test_repeated_numeric_strings_are_parsed_once(monkeypatch):
    original = AutoPreprocessor._clean_numeric_str_value
    calls = []
    def observed(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(AutoPreprocessor, '_clean_numeric_str_value', staticmethod(observed))
    values = pd.Series(['67,50 MAD', 'C1004', None] * 1000)
    actual = AutoPreprocessor._parse_numeric_series(values)
    assert len(calls) == 3
    assert actual.iloc[0] == 67.5 and actual.iloc[1:].isna().sum() == 2000


def test_recommendations_match_legacy_scalar_conversion(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({'prix': [f'{i % 19},50 MAD' for i in range(96)],
                          'segment': ['Or', 'Argent', None] * 32,
                          'mesure': rng.lognormal(size=96), 'label': np.arange(96) % 2})
    path = tmp_path / 'mixed.csv'
    frame.to_csv(path, index=False)
    optimized = work.analyze(path, 1_000_000, 'label', 'classification')
    monkeypatch.setattr(AutoPreprocessor, '_parse_numeric_series', staticmethod(
        lambda values: values.map(AutoPreprocessor._clean_numeric_str_value).astype(float)))
    reference = work.analyze(path, 1_000_000, 'label', 'classification')
    assert optimized == reference


@pytest.fixture
def client_and_calls(monkeypatch):
    calls = []
    def in_process(function, *args, limits):
        if function is work.analyze:
            calls.append(args)
        start = time.perf_counter()
        return function(*args), {'elapsed_seconds': time.perf_counter()-start, 'peak_rss_bytes': 0}
    monkeypatch.setattr(api_server, 'isolated', in_process)
    with TestClient(api_server.create_app(), base_url='http://localhost') as client:
        frame = pd.DataFrame({'measure': np.arange(48) % 17 + .3, 'band': ['a', 'b', None] * 16,
                              'label': np.arange(48) % 2})
        uploaded = client.post('/api/upload', files={'file': ('input.csv', frame.to_csv(index=False).encode())})
        sid = uploaded.json()['session_id']
        assert client.post('/api/config', json={'session_id': sid, 'mode': 'supervised', 'target': 'label'}).status_code == 200
        yield client, calls, sid, frame


def test_successful_analysis_retry_is_cached_and_request_changes_are_not(client_and_calls):
    client, calls, sid, frame = client_and_calls
    body = {'session_id': sid}
    first = client.post('/api/analyze', json=body)
    assert first.status_code == 200 and first.headers['X-Analysis-Cache'] == 'MISS'
    retry = client.post('/api/analyze', json=body)
    assert retry.headers['X-Analysis-Cache'] == 'HIT'
    assert retry.json() == first.json() and len(calls) == 1
    for change in ({'options': {'scaler_type': 'robust'}}, {'choices': {'measure': 'none'}}):
        response = client.post('/api/analyze', json={**body, **change})
        assert response.status_code == 200 and response.headers['X-Analysis-Cache'] == 'MISS'
    assert len(calls) == 3
    assert client.post('/api/config', json={'session_id': sid, 'mode': 'unsupervised'}).status_code == 200
    response = client.post('/api/analyze', json=body)
    assert response.headers['X-Analysis-Cache'] == 'MISS'
    assert all(not feature['is_target'] for feature in response.json()['features'])
    assert len(calls) == 4
    new = client.post('/api/upload', files={'file': ('second.csv', b'different\n1\n2\n3\n')}).json()['session_id']
    assert client.post('/api/config', json={'session_id': new, 'mode': 'unsupervised'}).status_code == 200
    response = client.post('/api/analyze', json={'session_id': new})
    assert response.headers['X-Analysis-Cache'] == 'MISS'
    assert [f['column'] for f in response.json()['features']] == ['different']
    assert len(calls) == 5


def test_failed_analysis_is_retryable_and_never_cached(client_and_calls, monkeypatch):
    client, calls, sid, _ = client_and_calls
    original = api_server.isolated
    failed = False
    def once(function, *args, limits):
        nonlocal failed
        if function is work.analyze and not failed:
            failed = True
            raise HTTPException(504, 'Délai de traitement dépassé.')
        return original(function, *args, limits=limits)
    monkeypatch.setattr(api_server, 'isolated', once)
    body = {'session_id': sid}
    error = client.post('/api/analyze', json=body)
    assert error.status_code == 504 and 'Réessayez' in error.json()['detail']
    assert client.post('/api/analyze', json=body).headers['X-Analysis-Cache'] == 'MISS'
    assert client.post('/api/analyze', json=body).headers['X-Analysis-Cache'] == 'HIT'
    assert len(calls) == 1
    assert client.delete(f'/api/session/{sid}').status_code == 200
    assert client.post('/api/analyze', json=body).status_code == 404


def test_concurrent_retries_do_not_start_duplicate_analysis(client_and_calls, monkeypatch):
    client, calls, sid, _ = client_and_calls
    original = api_server.isolated
    started, release = threading.Event(), threading.Event()
    def blocked(function, *args, limits):
        if function is work.analyze:
            started.set()
            assert release.wait(10)
        return original(function, *args, limits=limits)
    monkeypatch.setattr(api_server, 'isolated', blocked)
    body = {'session_id': sid}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, '/api/analyze', json=body)
        try:
            assert started.wait(10)
            assert client.post('/api/analyze', json=body).status_code == 409
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 200
    assert client.post('/api/analyze', json=body).headers['X-Analysis-Cache'] == 'HIT'
    assert len(calls) == 1
