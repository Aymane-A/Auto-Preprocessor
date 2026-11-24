import numpy as np
import pandas as pd
import pytest
from feature_selection import fit_selector

@pytest.fixture
def data():
    rng = np.random.default_rng(8)
    x = rng.normal(size=80)
    return pd.DataFrame({'signal': x, 'noise': rng.normal(size=80), 'constant': np.ones(80)}), (x > 0).astype(int)

@pytest.mark.parametrize('method', ['variance','kbest','anova','mutual_info','percentile','rfe','rfecv','forward','backward','l1','elastic_net','select_from_model','random_forest'])
def test_selector_fits_train_and_preserves_order(data, method):
    frame, y = data
    selector, report = fit_selector(frame, y, method=method, task='classification', count=2,
        percentile=50, folds=2, random_state=8, max_dense_bytes=1_000_000)
    assert report['fit_rows'] == 80
    assert report['selected'] == [name for name, keep in zip(frame, selector.get_support()) if keep]
    assert report['selected_count'] + len(report['rejected']) == frame.shape[1]


def test_supervised_requires_target_and_chi_square_requires_nonnegative(data):
    frame, y = data
    with pytest.raises(ValueError, match='cible'):
        fit_selector(frame, None, method='kbest', task='classification', count=1,
            percentile=50, folds=2, random_state=8, max_dense_bytes=1_000_000)
    with pytest.raises(ValueError, match='non négatives'):
        fit_selector(frame, y, method='chi2', task='classification', count=1,
            percentile=50, folds=2, random_state=8, max_dense_bytes=1_000_000)

def test_manual_grouping_and_invalid_target():
    from auto_preprocessor import AutoPreprocessor, PreprocessConfig
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"signal": rng.normal(size=60), "group": ["a", "b", "c"] * 20,
                          "noise": rng.normal(size=60)})
    y = (frame.signal > 0).astype(int)
    prep = AutoPreprocessor(PreprocessConfig(selection_method="kbest", selection_count=1,
        selection_include=["group"], selection_exclude=["noise"], verbose=0), task="classification")
    transformed = prep.fit_transform(frame, y)
    assert "noise" not in transformed.columns
    assert any(name.startswith("group") for name in transformed.columns)
    assert prep.report()["selection_features"]["manual_include"] == ["group"]
    with pytest.raises(ValueError, match="cible"):
        AutoPreprocessor(PreprocessConfig(selection_include=["target"], verbose=0), task="classification").fit_transform_xy(
            frame.assign(target=y), target="target")
