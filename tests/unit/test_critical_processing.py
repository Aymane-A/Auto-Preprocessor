"""Critical regression coverage: cached analysis, stable powers, leakage and raw overrides."""
from pathlib import Path
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
import pytest
from scipy import stats
from sklearn.preprocessing import PowerTransformer

from auto_preprocessor import AutoPreprocessor, PreprocessConfig
from isolated_processing import ProcessLimits, run_isolated
from statistical_profile import fit_distribution, apply_distribution
import web_processing as work


@pytest.mark.parametrize('action', ['BOX_COX', 'YEO_JOHNSON'])
def test_well_conditioned_power_matches_public_reference(action):
    values = pd.Series(np.random.default_rng(19).lognormal(1, .6, 200), name='measurement')
    transformer = fit_distribution(values, action, 42)
    reference = PowerTransformer(method='box-cox' if action=='BOX_COX' else 'yeo-johnson').fit_transform(values.to_frame()).ravel()
    np.testing.assert_allclose(apply_distribution(values, transformer, action), reference, atol=2e-6, rtol=2e-6)
    assert transformer.fallback_ is None


def test_unstable_calendar_power_has_finite_fitted_fallback_and_named_warning():
    values = pd.Series([2015.] * 110 + [2016.] * 280 + [2017.] * 210, name='année événement')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        transformer = fit_distribution(values, 'BOX_COX', 42)
        transformed = apply_distribution(values, transformer, 'BOX_COX')
    assert transformer.fallback_ in {'YEO_JOHNSON', 'QUANTILE_NORMAL'}
    assert 'année événement' in transformer.warning_ and 'lambda' in transformer.warning_
    assert np.isfinite(transformed).all() and np.var(transformed) > 0
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert transformer.lambdas_ is None or np.abs(transformer.lambdas_).max() < 5


def test_large_finite_unscaled_output_is_not_overflowed_by_float32_cast():
    frame = pd.DataFrame({'magnitude': np.linspace(1e40, 2e40, 40)})
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, downcast_numeric=False, outlier_method='none', scaler_type='none',
                            column_overrides={'magnitude': {'distribution':'NONE','scaling':'NONE','outliers':'NONE'}}, noise_guard='off'))
    result = prep.fit_transform(frame)
    assert np.isfinite(result.to_numpy()).all()
    assert result['magnitude'].dtype == np.float64
    np.testing.assert_array_equal(result['magnitude'], frame['magnitude'])
    assert any('float64' in warning and 'magnitude' in warning for warning in prep.warnings_)


def source(tmp_path):
    rng = np.random.default_rng(83)
    label = rng.permutation(np.arange(80) % 2)
    frame = pd.DataFrame({'measure':rng.normal(size=80), 'after_event':np.where(label, 'cancelled', 'accepted'), 'label':label})
    path = tmp_path/'source.csv'
    frame.to_csv(path,index=False)
    return frame,path


def test_descriptive_analysis_is_reused_but_train_fit_is_not_skipped(tmp_path, monkeypatch):
    frame,path=source(tmp_path)
    artifact=tmp_path/'analysis.joblib'
    response=work.analyze(path,1000000,'label','classification',artifact=artifact)
    row=next(feature for feature in response['features'] if feature['column']=='after_event')
    assert row['action']=='DROP' and any('ALERTE FUITE DE CIBLE' in value for value in row['anomalies'])
    stored=joblib.load(artifact)['analysis']
    assert work.selected_overrides(stored,'label',{'after_event':'DROP'})[0]['after_event'] is not None
    def unexpected_analysis(*args,**kwargs):
        pytest.fail('Descriptive analysis was recomputed instead of reusing its verified artifact')
    monkeypatch.setattr(AutoPreprocessor,'analyze',unexpected_analysis)
    output=tmp_path/'export';output.mkdir()
    result=work.execute(path,1000000,'label','classification',{}, {'measure':'none'},output,'source.csv',artifact)
    assert result['timings']['analysis_reused'] is True
    assert result['timings']['pipeline_stages']['type_detection'] > 0
    assert result['timings']['pipeline_stages']['leakage_detection'] > 0
    exported=pd.read_csv(output/'full.csv')
    assert 'after_event' not in exported.columns
    np.testing.assert_array_equal(np.sort(exported['measure']),np.sort(pd.read_csv(path)['measure']))
    report=__import__('json').loads((output/'report.json').read_text())
    dropped=next(feature for feature in report['features'] if feature['column']=='after_event')
    assert dropped['dropped'] and dropped['leakage']==1
    raw=next(feature for feature in report['features'] if feature['column']=='measure')
    assert raw['applied']=='NONE' and not raw['dropped']


def test_cached_analysis_never_crosses_dataset_or_type_configuration(tmp_path,monkeypatch):
    frame,path=source(tmp_path)
    artifact=tmp_path/'analysis.joblib'
    work.analyze(path,1000000,'label','classification',artifact=artifact)
    calls=[];original=AutoPreprocessor.analyze
    def observe(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(AutoPreprocessor,'analyze',observe)
    work.analyze(path,1000000,'label','classification',choices={'measure':'none'},artifact=artifact)
    assert not calls
    work.analyze(path,1000000,'label','classification',choices={'measure':'type:FORCE_CATEGORICAL'},artifact=artifact)
    assert len(calls)==1
    second=tmp_path/'other.csv';frame.rename(columns={'measure':'different'}).to_csv(second,index=False)
    result=work.analyze(second,1000000,'label','classification',artifact=artifact)
    assert len(calls)==2 and result['features'][0]['column']=='different'


def slow_worker(marker):
    Path(marker).write_text(str(os.getpid()))
    time.sleep(30)
    Path(str(marker)+'.late').write_text('orphan')


def test_timeout_worker_is_terminated_before_retry(tmp_path):
    marker=tmp_path/'worker.pid'
    before={child.pid for child in psutil.Process().children(recursive=True)}
    with pytest.raises(TimeoutError):
        run_isolated(slow_worker, marker, limits=ProcessLimits(timeout_seconds=4))
    if marker.exists():
        assert not psutil.pid_exists(int(marker.read_text()))
    assert not Path(str(marker)+'.late').exists()
    assert {child.pid for child in psutil.Process().children(recursive=True)} <= before
    assert run_isolated(os.getpid,limits=ProcessLimits(timeout_seconds=15)) != os.getpid()


def test_report_names_the_effective_fallback_instead_of_the_requested_power():
    frame=pd.DataFrame({'event_year':[2015.]*110+[2016.]*280+[2017.]*210})
    prep=AutoPreprocessor(PreprocessConfig(verbose=0,preserve_rows=True,
        column_overrides={'event_year':{'distribution':'BOX_COX','duplicates':'KEEP'}}))
    output=prep.fit_transform(frame)
    assert np.isfinite(output.to_numpy()).all()
    assert prep.report()['colonnes_detaillees'].iloc[0]['Action appliquée'] in {'QUANTILE_NORMAL','YEO_JOHNSON'}
    assert any('event_year' in warning for warning in prep.warnings_)


def test_combined_csv_reuses_correct_logical_header_with_quotes_and_newlines(tmp_path):
    frame=pd.DataFrame({'mesure\n"année,؟':[1,2,3,4,5,6], 'groupe':['a','b','a','b','a','b']})
    path=tmp_path/'quoted.csv';frame.to_csv(path,index=False)
    output=tmp_path/'csv';output.mkdir()
    result=work.execute(path,1000000,None,'auto',{}, {name:'none' for name in frame},output,'quoted.csv')
    combined=pd.read_csv(output/'full.csv')
    expected=pd.concat([pd.read_csv(output/'train.csv'),pd.read_csv(output/'test.csv')],ignore_index=True)
    pd.testing.assert_frame_equal(combined,expected)
    assert combined.shape==tuple(result['full_shape'])


def test_repeated_string_cleaning_keeps_original_values_and_missingness(monkeypatch):
    from auto_preprocessor import DEFAULT_NA_VALUES
    frame=pd.DataFrame({'label':['  ÉTÉ  ','Ete',' ? ',None,'  été  ','NOËL','',pd.NA]*100})
    original=frame['label']
    normalized=original.astype(str).str.strip().map(AutoPreprocessor._canonical_variant)
    expected=normalized.mask(normalized.str.casefold().isin(DEFAULT_NA_VALUES)|original.isna(),np.nan)
    calls=[];scalar=AutoPreprocessor._canonical_variant
    def observed(value):
        calls.append(value);return scalar(value)
    monkeypatch.setattr(AutoPreprocessor,'_canonical_variant',staticmethod(observed))
    prep=AutoPreprocessor(PreprocessConfig(verbose=0))
    result=prep._clean_string_and_na(frame)
    pd.testing.assert_series_equal(result['label'],expected)
    assert len(calls)==len(original.astype(str).str.strip().unique())
