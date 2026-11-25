"""The distribution explanation must match the unchanged selection rule."""
import numpy as np
import pandas as pd
import pytest

import auto_preprocessor as module
from auto_preprocessor import PreprocessConfig, _recommend_distribution


@pytest.mark.parametrize("skews,yeo,choice,phrase", [
    ((2.0, 0.40, 0.15), 0.01, "LOG1P", "écart 0.14 ≤ tolérance 0,25"),
    ((2.0, 0.40, 0.35), 0.01, "YEO_JOHNSON", "plus faible asymétrie résiduelle"),
    ((2.0, 1.40, 1.30), 1.20, "QUANTILE", "après le candidat retenu"),
    ((1.1, 1.25, 1.20), 1.00, "NONE", "le candidat retenu ne la réduit pas"),
])
def test_distribution_reason_matches_chosen_rule(monkeypatch, skews, yeo, choice, phrase):
    readings = iter(skews)
    monkeypatch.setattr(module.stats, "skew", lambda values: next(readings))
    monkeypatch.setattr(module, "_preview_yeo_johnson", lambda frame, preview: np.zeros((len(frame), 1)))
    monkeypatch.setattr(module, "_distribution_skew", lambda values: yeo)
    action, measure, reason = _recommend_distribution(
        pd.Series(np.arange(1, 20)), PreprocessConfig(skew_threshold=1.0))
    assert action == choice
    assert phrase in measure + "; " + reason
