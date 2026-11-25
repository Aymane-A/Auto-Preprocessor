"""Synthetic regression checks for tracked nested-CV split behavior."""
import numpy as np
import pandas as pd
import pytest

from nested_cv import _splits, nested_cross_validate


def test_stratified_splits_reject_classes_smaller_than_fold_count():
    target = np.array([0] * 8 + [1] * 2)
    frame = pd.DataFrame({"value": np.arange(len(target))})

    with pytest.raises(ValueError, match="Chaque classe"):
        _splits("stratified", 3, 187, frame, target, None)


def test_stratified_splits_are_reproducible():
    target = np.tile([0, 1], 30)
    frame = pd.DataFrame({"value": np.arange(len(target))})

    first = _splits("stratified", 3, 187, frame, target, None)
    second = _splits("stratified", 3, 187, frame, target, None)

    for (first_train, first_valid), (second_train, second_valid) in zip(first, second):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_valid, second_valid)


def test_grouped_outer_and_inner_splits_keep_groups_disjoint():
    groups = np.repeat(np.arange(12), 4)
    frame = pd.DataFrame({"value": np.arange(len(groups))})
    outer = _splits("group", 2, 187, frame, groups, groups)

    for train, valid in outer:
        assert not set(groups[train]) & set(groups[valid])
        inner = _splits("group", 2, 187, frame.iloc[train],
                        groups[train], groups[train])
        for inner_train, inner_valid in inner:
            assert not set(groups[train][inner_train]) & set(groups[train][inner_valid])


def test_time_split_gap_prevents_next_day_label_overlap():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    label_dates = dates + pd.Timedelta(days=1)
    frame = pd.DataFrame({"value": np.arange(len(dates))})
    splits = _splits("time", 3, 187, frame, np.arange(len(dates)), None,
                     time_gap_rows=1)

    for train, valid in splits:
        assert valid[0] - train[-1] - 1 == 1
        assert label_dates[train].max() < dates[valid].min()


def test_nested_cv_requires_time_column_when_configuring_a_gap():
    frame = pd.DataFrame({"feature": np.arange(20), "target": np.arange(20)})

    with pytest.raises(ValueError, match="time_gap_rows exige une colonne temporelle"):
        nested_cross_validate(frame, "target", "linear", task="regression",
                              outer_cv=2, inner_cv=2, time_gap_rows=1)
