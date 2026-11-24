"""Regression tests for confirmed Task 3b chart semantics and existing overrides."""
import calendar
import numpy as np
import pandas as pd
import pytest
import web_processing as work


def charts(values,target=None,forced='AUTO'):
    return work.select_visualizations(pd.Series(values,name='arbitrary'),target,'classification',forced)


@pytest.mark.parametrize('count',[11,12,15,16])
def test_aggregation_threshold_applies_to_frequency_and_target_rate(count):
    values=[f'category {i%count}' for i in range(count*20)]
    target=pd.Series(np.arange(len(values))%2,name='label')
    result=charts(values,target)
    primary,rate=result['recommended_visualization']
    assert primary['kind']==('frequency_bar' if count<=15 else 'top_categories')
    for chart in [primary,rate]:
        assert len(chart['data'])==(count if count<=15 else 11)
        assert chart['hidden_categories']==(0 if count<=15 else count-10)
        assert sum(bool(r.get('is_other')) for r in chart['data'])==(0 if count<=15 else 1)


@pytest.mark.parametrize('values,expected',[
    (['December','January','February']*20,['January','February','December']),
    (['DÉC.','Févr.','Janv.']*20,['Janv.','Févr.','DÉC.']),
    (['Sunday','Wednesday','Monday']*20,['Monday','Wednesday','Sunday']),
    (['high','low','medium']*20,['low','medium','high']),
])
def test_natural_order_comes_from_values_and_is_shared_by_rates(values,expected):
    result=charts(values,pd.Series(np.arange(len(values))%2,name='label'))
    assert result['detected_type']=='ORDINAL'
    assert [c['kind'] for c in result['recommended_visualization']]==['ordered_bar','target_rate']
    for chart in result['recommended_visualization']:
        assert [d['label'] for d in chart['data']]==expected
        assert not any(d.get('is_other') for d in chart['data'])
    forced=charts(values,forced='FORCE_CATEGORICAL')
    assert [d['label'] for d in forced['recommended_visualization'][0]['data']]==expected


def test_all_twelve_months_are_present_in_calendar_order_not_frequency_order():
    months=list(calendar.month_name)[1:]
    values=[month for i,month in enumerate(reversed(months)) for _ in range(i+2)]
    result=charts(values)
    assert [d['label'] for d in result['recommended_visualization'][0]['data']]==months
    assert result['recommended_visualization'][0]['horizontal'] is True


def test_declared_ordinal_categories_keep_every_observed_level():
    s=pd.Series(pd.Categorical(['last','middle','first']*20,categories=['first','middle','last','unused'],ordered=True),name='random')
    result=work.select_visualizations(s)
    assert [d['label'] for d in result['recommended_visualization'][0]['data']]==['first','middle','last']


@pytest.mark.parametrize('size',[31,53])
def test_discrete_bars_remain_numeric_order_despite_reverse_frequencies(size):
    values=[i for i in range(1,size+1) for _ in range(i+1)]
    result=charts(values)
    assert result['recommended_visualization'][0]['kind']=='value_counts'
    assert [float(d['label']) for d in result['recommended_visualization'][0]['data']]==list(range(1,size+1))


def test_possible_codes_avoid_metric_plots_and_allow_existing_force_numeric():
    codes=np.tile(np.arange(1,402,2),10)
    target=pd.Series(np.tile(np.arange(201)%2,10),name='label')
    result=charts(codes,target)
    assert result['detected_type']=='NUMERIC_CODES'
    assert [c['kind'] for c in result['recommended_visualization']]==['top_categories','target_rate']
    assert 'codes numériques: distribution non interprétable' in result['recommended_visualization'][0]['reason']
    assert result['code_evidence']['repeated_ratio']>.8
    forced=charts(codes,target,'FORCE_NUMERIC')
    assert forced['detected_type']=='NUMERIC_CONTINUOUS' and forced['display_type_override']=='FORCE_NUMERIC'
    assert [c['kind'] for c in forced['recommended_visualization']]==['histogram','boxplot','target_boxplot']
    categorical=charts(codes,target,'FORCE_CATEGORICAL')
    assert categorical['detected_type']=='CATEGORICAL_HIGH_CARD'
    assert [c['kind'] for c in categorical['recommended_visualization']]==['top_categories','target_rate']


def test_duration_with_zero_and_strong_ordered_target_signal_is_not_relabelled_code():
    values=np.tile(np.arange(201),10)
    y=pd.Series((values>100).astype(int),name='label')
    assert charts(values,y)['detected_type']=='NUMERIC_CONTINUOUS'
    assert charts(values+1,y)['detected_type']=='NUMERIC_CONTINUOUS'


def test_small_near_constant_nominal_set_is_not_grouped():
    result=charts(['category a']*998+['category b','category c'])
    chart=result['recommended_visualization'][0]
    assert chart['kind']=='frequency_bar' and len(chart['data'])==3
    assert not any(d.get('is_other') for d in chart['data'])


def test_invalid_display_override_is_not_silently_ignored():
    with pytest.raises(work.InputError):charts([1,2,3],forced='MADE_UP')
    with pytest.raises(work.InputError):charts(['a','b','c'],forced='FORCE_NUMERIC')
