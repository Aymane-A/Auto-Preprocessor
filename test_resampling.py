import numpy as np
import pandas as pd
import pytest
from resampling import class_profile, rebalance
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, _duplicate_mask

def test_smote_segments_and_reproducibility():
    x=pd.DataFrame({'a':[0,1,2,3,10,12,14], 'b':[0,0,0,0,10,12,14]})
    y=pd.Series([0]*4+[1]*3)
    first, labels, info=rebalance(x,y,'smote',1,5,42)
    second, _, _=rebalance(x,y,'smote',1,5,42)
    pd.testing.assert_frame_equal(first,second)
    assert info['k_neighbors_used']==2 and info['synthetic_rows']==1
    assert info['warnings']
    assert labels.value_counts().to_dict()=={0:4,1:4}
    pd.testing.assert_frame_equal(first.iloc[:7],x.astype(float),check_dtype=False)
    assert first.iloc[-1].a==first.iloc[-1].b and 10<=first.iloc[-1].a<=14

def test_random_methods_keep_majority_and_ratio():
    x=pd.DataFrame({'a':range(100)})
    y=pd.Series([0]*90+[1]*10)
    over, labels, meta=rebalance(x,y,'random_over',.5,5,4)
    assert labels.value_counts().to_dict()=={0:90,1:45}
    assert meta['synthetic_rows']==0 and len(over)==135
    under, labels, meta=rebalance(x,y,'random_under',.5,5,4)
    assert labels.value_counts().to_dict()=={0:20,1:10}
    assert meta['sampled_rows']==70
    weighted,labels,meta=rebalance(x,y,'class_weight')
    pd.testing.assert_frame_equal(weighted,x)
    assert meta['class_weight']=={'0':100/180,'1':5}

def test_guards_and_profiles():
    assert class_profile([0]*99+[1])['level']=='sévère'
    with pytest.raises(ValueError,match='binaire'):
        rebalance(pd.DataFrame({'a':[1,2,3]}),[0,1,2],'smote')
    with pytest.raises(ValueError,match='finies'):
        rebalance(pd.DataFrame({'a':[1.,2.,3.,np.nan,4.]}),[0,0,0,1,1],'smote',1)
    with pytest.raises(ValueError,match='SMOTENC'):
        rebalance(pd.DataFrame({'a':[1,2,3,4,5]}),[0,0,0,1,1],'smotenc',1)
    x=pd.DataFrame({'a':[1,2,3,4,5], 'cat':['a','b','c','d','e']})
    out,_,meta=rebalance(x,[0,0,0,1,1],'smotenc',1)
    assert meta['synthetic_rows']==1 and out.cat.iloc[-1] in {'d','e'}

def test_single_minority_supports_weights_and_random_sampling_but_rejects_smote():
    x=pd.DataFrame({'a':range(11)})
    y=[0]*10+[1]
    weighted, labels, info=rebalance(x,y,'class_weight')
    assert len(weighted)==len(labels)==11 and info['class_weight']['1']==5.5
    over, labels, info=rebalance(x,y,'random_over',.5)
    assert labels.value_counts().to_dict()=={0:10,1:5}
    assert info['duplicated_rows']==4 and over.iloc[-4:].a.eq(10).all()
    with pytest.raises(ValueError,match='deux exemples minoritaires'):
        rebalance(x,y,'smote',.5)

def test_smotenc_keeps_categories_and_original_rows():
    x=pd.DataFrame({'numeric':[0,1,2,3,5,8,10], 'category':['a','b','c','d','e','hot','cold']})
    y=[0]*5+[1]*2
    out,labels,info=rebalance(x,y,'smotenc',1,5,42)
    pd.testing.assert_frame_equal(out.iloc[:len(x)].reset_index(drop=True),x,check_dtype=False)
    assert labels.value_counts().to_dict()=={0:5,1:5}
    assert set(out.iloc[len(x):].category).issubset({'hot','cold'})
    assert info['k_neighbors_used']==1 and info['synthetic_rows']==3


def test_smote_preserves_binary_onehot_originals_and_untouched_test():
    train = pd.DataFrame({
        'age': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 50, 52, 54],
        'hypertension': [0, 1] * 6 + [0],
        'heart_disease': [1, 0] * 6 + [1],
        'smoking_status_never': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        'smoking_status_former': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        'smoking_status_current': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    })
    labels = pd.Series([0] * 10 + [1] * 3)
    test = train.iloc[:2].copy(deep=True)
    original = train.copy(deep=True)
    out, new_labels, report = rebalance(train, labels, 'smote', .5, 2, 4)
    pd.testing.assert_frame_equal(out.iloc[:len(train)].reset_index(drop=True), original, check_dtype=False)
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(test, original.iloc[:2])
    synthetic = out.iloc[len(train):]
    binary = [c for c in train if c != 'age']
    assert synthetic[binary].isin([0, 1]).all().all()
    assert synthetic[[c for c in binary if c.startswith('smoking_status_')]].sum(axis=1).eq(1).all()
    assert report['method'] == 'SMOTENC'
    assert set(report['categorical_columns']) == set(binary)
    assert report['synthetic_rows'] == 2 and len(new_labels) == 15


def test_all_binary_fallback_repairs_only_synthetic_rows():
    train = pd.DataFrame({'flag': [0, 1] * 5 + [0, 1, 0],
                          'group_a': [1, 0] * 5 + [1, 0, 1],
                          'group_b': [0, 1] * 5 + [0, 1, 0]})
    out, _, report = rebalance(train, [0] * 10 + [1] * 3, 'smote', .5, 2, 4)
    assert report['method'] == 'SMOTE + rounding'
    pd.testing.assert_frame_equal(out.iloc[:len(train)].reset_index(drop=True), train, check_dtype=False)
    assert out.iloc[len(train):].isin([0, 1]).all().all()
    assert out.iloc[len(train):][['group_a', 'group_b']].sum(axis=1).eq(1).all()

def test_large_duplicate_mask_matches_exact_pandas_result():
    frame=pd.DataFrame({'numeric':np.arange(200_100), 'text':['x']*200_100})
    frame.loc[[10,20,30], 'numeric']=7
    frame.loc[[20,30], 'text']='other'
    expected=frame.duplicated(keep='first')
    pd.testing.assert_series_equal(_duplicate_mask(frame),expected)

def test_vectorized_target_encoding_keeps_unknown_class_marker():
    prep=AutoPreprocessor(PreprocessConfig(),task='classification')
    prep._fit_target(pd.Series(['major','minor','major']))
    assert prep.transform_target(pd.Series(['minor','unknown','major'])).tolist()==[1,-1,0]
