import numpy as np
import pandas as pd
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, _recommend_distribution, _target_encoder
from web_processing import selected_overrides, select_visualizations, leakage_review, feature_stats

def test_accepted_drop_survives_split_and_explicit_force_wins():
    frame = pd.DataFrame({'sequence': np.arange(100), 'constant': 1, 'empty': np.nan, 'label': [0,1]*50})
    analysis = AutoPreprocessor(PreprocessConfig(verbose=0)).analyze(frame, target='label')
    overrides, chosen = selected_overrides(analysis, 'label', {})
    assert all(chosen[c] == 'DROP' for c in ['sequence','constant','empty'])
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides, noise_guard='off'))
    out = prep.fit_transform(frame.drop(columns='label').sample(80, random_state=3))
    assert out.shape[1] == 0
    _, chosen = selected_overrides(analysis, 'label', {'sequence': 'type:FORCE_NUMERIC'})
    assert chosen['sequence'] == {'type': 'FORCE_NUMERIC'}

def test_discrete_counts_keep_numeric_distance_and_skip_zero_iqr():
    x = pd.Series([0]*76 + [1,2,3,4,5,6]*4, name='events')
    prep = AutoPreprocessor(PreprocessConfig(verbose=0))
    prep._detect_types(x.to_frame())
    assert prep.column_metadata_['events']['type'].value == 'NUMERIC_DISCRETE'
    assert prep.num_cols_ == ['events']
    assert _recommend_distribution(x, prep.config)[0] == 'NONE'
    assert feature_stats(x)['iqr']['count'] == 0

def test_unique_sequence_has_summary_without_target_rates():
    result = select_visualizations(pd.Series(range(100), name='sequence'), pd.Series([0,1]*50, name='label'))
    assert result['detected_type'] == 'ID_LIKE'
    assert [c['kind'] for c in result['recommended_visualization']] == ['summary']

def test_small_categories_are_aggregated_in_target_rates():
    x = pd.Series(['common']*20 + ['other']*15 + ['tiny']*2, name='category')
    result = select_visualizations(x, pd.Series([0,1]*18+[0], name='label'))
    chart = next(c for c in result['recommended_visualization'] if c['kind']=='target_rate')
    assert 'tiny' not in [d['label'] for d in chart['data']]
    assert sum(d['total'] for d in chart['data']) == len(x)

def test_review_is_neutral_and_dropped_columns_are_silent():
    row = dict(score_fuite=.7, cramer_cible=.4, information_mutuelle_cible=.1, action='ONEHOT')
    assert leakage_review(row).startswith('Forte association')
    assert leakage_review({**row, 'action':'DROP'}) is None
    assert leakage_review({**row, 'cramer_cible':0}) is None

def test_target_encoding_shrinks_pure_singletons():
    encoder = _target_encoder(2, 42)
    x = pd.DataFrame({'category':['a','b','c','d','e','f']})
    encoder.fit(x, np.array([0,1,0,1,0,1]))
    out = encoder.transform(x)
    assert np.all((out > 0) & (out < 1))

def test_typing_does_not_depend_on_identifier_or_coordinate_names():
    from statistical_profile import semantic_type
    values = pd.Series(np.linspace(28.1,34.2,120),name='latitude')
    assert semantic_type(values,'NUMERIC_CONTINUOUS',{}) == semantic_type(values.rename('measurement'),'NUMERIC_CONTINUOUS',{})
    x=pd.DataFrame({'device_code':np.tile([1,2,3],40)})
    prep=AutoPreprocessor(PreprocessConfig(verbose=0))
    prep._detect_types(x)
    first=prep.column_metadata_['device_code']['type']
    prep._detect_types(x.rename(columns={'device_code':'measurement'}))
    assert prep.column_metadata_['measurement']['type'] == first

def test_high_cardinality_grouping_and_missing_indicator_are_train_fitted():
    x = pd.DataFrame({'category': ['common']*20 + [f'v{i}' for i in range(30)] + [None]*10})
    y = np.array([0,1]*30)
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, noise_guard='off', min_features_for_collinearity=100,
        column_overrides={'category': {'duplicates':'KEEP'}}))
    train = prep.fit_transform(x,y)
    test = prep.transform(pd.DataFrame({'category':['novel-level',None,'v0','common']}))
    assert 'category_isna' in train
    assert list(train.columns) == list(test.columns)
    assert test.category_isna.tolist() == [0,1,0,0]
    assert test.category.iloc[0] == test.category.iloc[2]
    assert np.isfinite(test.to_numpy()).all()
    assert prep.high_rare_grouper_.frequent_categories_['category'] == {'common'}
    assert prep._reported_action('category',prep.column_metadata_['category']) == 'TARGET_ENCODE'
