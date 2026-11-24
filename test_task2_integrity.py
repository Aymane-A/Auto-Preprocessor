"""Regression checks for confirmed audit failures; no fitted statistics are mocked."""
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from auto_preprocessor import AutoPreprocessor, PreprocessConfig
import web_processing as work


def test_probability_underflow_is_a_bound_and_zero_effects_stay_zero():
    payload = work.json_safe({'p': 0., 'q_primaire': np.float64(0), 'p_value': 1e-90,
                             'effect': 0., 'missing': np.nan, 'score_fuite': np.nan})
    assert payload == {'p': '< 0.001', 'q_primaire': '< 0.001', 'p_value': 1e-90,
                       'effect': 0., 'missing': None, 'score_fuite': 'non applicable'}
    json.dumps(payload, allow_nan=False)


def test_binary_chart_counts_observations_without_boxplot(tmp_path):
    path = tmp_path/'binary.csv'
    pd.DataFrame({'flag': [0,1,1,0,1,np.nan]}).to_csv(path,index=False)
    result = work.feature_eda(path, 100000, 'flag')
    assert result['binary'] and 'boxplot' not in result
    assert [v['count'] for v in result['distribution']] == [2,3]
    assert result['missing_count'] == 1
    assert len(work.histogram([0,1,1])) == 2


def test_quasi_constant_numeric_evidence_uses_observed_frequency():
    frame = pd.DataFrame({'mostly_zero': [0]*997+[1,2,3], 'measure': np.arange(1000)})
    result = AutoPreprocessor(PreprocessConfig(verbose=0,preserve_rows=True)).analyze(frame)
    row = next(r for r in result.columns if r['colonne']=='mostly_zero')
    assert row['type'] == 'DROP_CONSTANT'
    assert row['top_category_frequency'] == .997
    assert '99.7%' in row['drop_preuve'] and 'nan%' not in row['drop_preuve']


def test_leakage_review_is_name_independent_and_does_not_recommend_drop():
    base = {'score_fuite': .649, 'cramer_cible': .488, 'information_mutuelle_cible': .132}
    assert work.leakage_review({**base,'colonne':'arbitrary_name'})
    assert not work.leakage_review({**base,'colonne':'reservation_status_date','score_fuite': .51})
    assert not work.leakage_review({**base,'fuite_suspectee': True})
    assert not work.leakage_review({**base,'score_fuite': np.nan})


def test_leakage_drop_proof_matches_its_actual_decision():
    row = {'colonne':'proxy','type':'BINARY','action':'BINARY_MAP','fuite_suspectee':True,
           'score_fuite':1.,'alternatives':[], 'steps': {'type': {'action':'AUTO',
           'evidence':'stale unrelated evidence','choices':['AUTO','DROP']}}}
    result=work.recommendation(row,'target')
    assert result['action']=='DROP'
    assert '1.000' in result['justification']['preuve']
    assert 'stale' not in result['justification']['preuve']


def test_excel_retains_probability_bound_and_sampling_text(tmp_path):
    path=tmp_path/'report.xlsx'
    report={'summary': {'Échantillonnage de l’analyse':'Aucun échantillonnage', 'Colonnes exportées':[]},
            'features':[{'column':'x','diagnostics':work.json_safe({'p':0.,'score_fuite':np.nan})}],
            'correlations':{}}
    work.write_workbook(path,report)
    book=load_workbook(path,data_only=True)
    assert book['Résumé']['B2'].value == 'Aucun échantillonnage'
    cells=[c.value for row in book['Diagnostics'] for c in row]
    assert '"< 0.001"' in cells and '"non applicable"' in cells
    book.close()


def test_results_eda_retains_source_warning_on_derived_columns(tmp_path):
    pd.DataFrame({'event_month':[1,2,3,4], 'unrelated':[7,8,9,10]}).to_csv(tmp_path/'full.csv',index=False)
    warning={'source':'event','warning':'Fuite possible, revue manuelle'}
    (tmp_path/'report.json').write_text(json.dumps({'summary':{'Cible':None,'Tâche':'auto',
        'Avertissements par sortie':{'event_month':warning}}}),encoding='utf-8')
    result=work.result_eda(tmp_path)
    derived=next(f for f in result['features'] if f['column']=='event_month')
    assert derived['leakage_review']==warning
    assert 'leakage_review' not in next(f for f in result['features'] if f['column']=='unrelated')
