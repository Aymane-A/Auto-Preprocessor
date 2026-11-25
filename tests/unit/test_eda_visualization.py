"""Data-based EDA policy, denominators and honest chart contracts."""
import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
import web_processing as work


def test_titanic_id_fixture_and_gapped_id_backend_contract(tmp_path):
    fixture = json.loads((Path(__file__).resolve().parents[2] / "frontend/src/__fixtures__/passengerId.json").read_text(encoding="utf-8-sig"))
    assert fixture["recommended_visualization"][0]["kind"] == "summary"
    assert fixture["recommended_visualization"][0]["cardinality"] == 891
    assert fixture["recommended_visualization"][0]["examples"] == ["1", "2", "3", "4", "5"]
    assert fixture["top"] == [] and "distribution" not in fixture
    frame = pd.DataFrame({"PassengerId": np.arange(1, 892) * 2 - 1,
                          "Survived": np.arange(891) % 2})
    path = tmp_path / "gapped_ids.csv"
    frame.to_csv(path, index=False)
    result = work.feature_eda(path, 1_000_000, "PassengerId", "Survived", "classification")
    assert result["detected_type"] == "ID_LIKE"
    assert result["recommended_visualization"][0]["kind"] == "summary"
    assert result["top"] == [] and "distribution" not in result
    assert result["display_type_override"] == "AUTO"
    assert result["code_evidence"]["possible_codes"] is False
    assert result["relationship_note"] == "Identifiant : aucune relation graphique pertinente avec la cible."


def select(values, target=None, task='auto', name='feature'):
    return work.select_visualizations(pd.Series(values,name=name),target,task)


def kinds(result):return [c['kind'] for c in result['recommended_visualization']]


@pytest.mark.parametrize('values',[[0,1]*30,[True,False]*30,['a','b']*30])
def test_binary_uses_two_count_and_percentage_bars(values):
    r=select(values)
    assert kinds(r)==['binary_bar']
    data=r['recommended_visualization'][0]['data']
    assert len(data)==2 and sum(v['count'] for v in data)==60
    assert sum(v['percent'] for v in data)==100


def test_low_and_high_cardinality_counts_are_sorted_and_conserve_mass():
    low=select(['a']*50+['b']*30+['c']*20+[None]*10)
    assert kinds(low)==['frequency_bar']
    assert [v['count'] for v in low['recommended_visualization'][0]['data']]==[50,30,20]
    high=select(['une modalité très longue '+str(i) for i in range(20)]*10)
    chart=high['recommended_visualization'][0]
    assert chart['kind']=='top_categories' and chart['horizontal']
    assert chart['hidden_categories']==10 and len(chart['data'])==11
    assert chart['data'][-1]['is_other'] and chart['data'][-1]['count']==100
    assert sum(d['count'] for d in chart['data'])==200
    assert sum(d['percent'] for d in chart['data'])==100


def test_integer_float_counts_are_discrete_and_ordered():
    r=select([0.,2.,1.,2.,None]*20)
    assert r['detected_type']=='NUMERIC_DISCRETE' and kinds(r)==['value_counts']
    assert [v['label'] for v in r['recommended_visualization'][0]['data']]==['0.0','1.0','2.0']


def test_dominant_share_and_missingness_precede_any_distribution():
    r=select([0]*997+[1,2,3])
    assert kinds(r)==['dominant_share']
    assert r['recommended_visualization'][0]['data'][0]['percent']==99.7
    r=select([None]*950+[i/3 for i in range(50)])
    assert kinds(r)[0]=='missingness'
    assert [v['count'] for v in r['recommended_visualization'][0]['data']]==[950,50]
    assert r['recommended_visualization'][0]['denominator']==1000
    assert kinds(select([None]*10))==['missingness','summary']


@pytest.mark.parametrize('skewed',[False,True])
def test_continuous_histogram_bins_count_all_observations_and_disclose_scale(skewed):
    rng=np.random.default_rng(54)
    values=np.r_[rng.normal(size=1000),[-5,1e7]] if skewed else rng.normal(size=1000)
    r=select(values);hist,box=r['recommended_visualization']
    assert [hist['kind'],box['kind']]==['histogram','boxplot']
    assert hist['scale']==('symlog' if skewed else 'linear')
    edges=[v['low'] for v in hist['data']]+[hist['data'][-1]['high']]
    assert np.array_equal(np.histogram(values,bins=edges)[0],[v['count'] for v in hist['data']])
    assert sum(v['count'] for v in hist['data'])==len(values)
    assert 'Aucune observation exclue' in hist['reason'] if skewed else 'aucune observation exclue' in hist['reason']


@pytest.mark.parametrize('native',[True,False])
def test_dates_are_chronological_monthly_counts_with_real_gaps(native):
    values=['2024-01-02','2024-01-05','2024-03-01']*20
    r=select(pd.to_datetime(values) if native else values)
    assert r['detected_type']=='DATETIME' and kinds(r)==['time_series']
    data=r['recommended_visualization'][0]['data']
    assert [v['label'] for v in data]==['2024-01','2024-02','2024-03']
    assert [v['count'] for v in data]==[40,0,20]
    assert ('Type datetime natif' if native else '60 valeurs observées') in r['recommended_visualization'][0]['reason']
    assert kinds(select(['2015','2016','2017']*20))==['frequency_bar']


def test_identifiers_and_free_text_get_cardinality_and_examples():
    for values,expected in [([f'R{i:04d}' for i in range(80)],'ID_LIKE'),
            ([f'Le client décrit une opération bancaire avec des détails précis numéro {i}' for i in range(80)],'TEXT')]:
        r=select(values)
        assert r['detected_type']==expected and kinds(r)==['summary']
        assert r['recommended_visualization'][0]['cardinality']==80
        assert len(r['recommended_visualization'][0]['examples'])==5
        if expected=='TEXT':assert '80 valeurs observées, sans échantillonnage' in r['recommended_visualization'][0]['reason']


def test_target_relationships_update_with_type_and_exclude_self():
    target=pd.Series([0,1]*100,name='outcome')
    continuous=select(np.random.default_rng(5).normal(size=200),target)
    assert kinds(continuous)[-1]=='target_boxplot'
    chart=continuous['recommended_visualization'][-1]
    assert chart['domain']=={'low':min(g['box']['low'] for g in chart['groups']),'high':max(g['box']['high'] for g in chart['groups'])}
    assert kinds(select(np.arange(200)%5,target))[-1]=='target_distribution'
    assert kinds(select(['a','b','c','d']*50,target))[-1]=='target_rate'
    assert kinds(work.select_visualizations(target,target))==['binary_bar']
    assert 'target_rate' not in kinds(select(['a','b']*100,target,'regression'))
    other=pd.Series(np.arange(200)%3,name='outcome')
    r=select(['a','b']*100,other,'classification')
    assert r['target_type']=='MULTICLASS' and kinds(r)==['binary_bar']


def test_rate_denominators_use_complete_pairs_and_weight_the_other_bucket():
    x=pd.Series([f'c{i%20}' for i in range(200)]+[None,'c0'],name='feature')
    y=pd.Series([i%2 for i in range(200)]+[1,None],name='outcome')
    chart=work.select_visualizations(x,y)['recommended_visualization'][-1]
    assert chart['kind']=='target_rate'
    assert sum(d['total'] for d in chart['data'])==200
    assert sum(d['count'] for d in chart['data'])==100
    for d in chart['data']:assert d['percent']==d['count']*100/d['total']
    r=work.select_visualizations(pd.Series(['a','b',None,None],name='feature'),pd.Series([None,None,0,1],name='label'))
    assert 'Aucune paire complète' in r['relationship_note']


def test_names_do_not_choose_chart_policy_or_mutate_input():
    x=pd.Series([0,1,2,3]*20,name='company');before=x.copy(deep=True)
    a=work.select_visualizations(x)
    b=work.select_visualizations(x.rename('arbitrary'))
    assert a==b
    pd.testing.assert_series_equal(x,before)


def test_actual_autres_category_is_not_confused_with_aggregate_bucket():
    r=select(['Autres']*997+['a','b','c'])
    rows=r['recommended_visualization'][0]['data']
    # Task 3b: small nominal sets retain every category, even near constant.
    assert [d['label'] for d in rows]==['Autres','a','b','c']
    r=select(['Autres']*2000+[f'rare{i}' for i in range(20)])
    rows=r['recommended_visualization'][0]['data']
    assert [d['label'] for d in rows]==['Autres','Autres (regroupement 1)']
    assert rows[1]['is_other'] and rows[1]['count']==20
    r=select(['Autres']*100+[str(i) for i in range(20)]*10)
    rows=r['recommended_visualization'][0]['data']
    assert len({d['label'] for d in rows})==len(rows)


def test_unused_categorical_levels_do_not_create_fictitious_bars():
    values=pd.Series(pd.Categorical(['a','b']*30,categories=['a','b','unused']),name='feature')
    r=work.select_visualizations(values)
    assert kinds(r)==['binary_bar']
    assert [d['label'] for d in r['recommended_visualization'][0]['data']]==['a','b']
