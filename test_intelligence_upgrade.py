import json
import pickle

import numpy as np
import pandas as pd
import pytest

from app_preprocess import App, build_analysis_detail, import_user_choices, parse_step_parameters, validate_step_choice
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, RobustTextTransformer
from model_evaluation import compare_preprocessing
from statistical_profile import DISTRIBUTIONS, OUTLIER_ACTIONS, adjusted_p_values, parse_dates


def make_prep(overrides=None, **config):
    return AutoPreprocessor(PreprocessConfig(verbose=0, column_overrides=overrides or {}, **config))


@pytest.mark.parametrize("action", DISTRIBUTIONS[1:])
def test_distribution_preview_and_execution(action):
    frame = pd.DataFrame({"value": np.linspace(.5, 100, 60) ** 1.5}, index=np.arange(60) + 30)
    original = frame.copy()
    prep = make_prep({"value": {"distribution": action, "scaling": "NONE"}}, outlier_method="none")
    analysis = prep.analyze(frame)
    detail = analysis.columns[0]["steps"]["distribution"]["comparaison"][action]
    assert detail["applicable"] and detail["effectif"] == len(frame)
    result = prep.fit_transform(frame)
    assert result.index.equals(frame.index)
    assert result["value"].var() == pytest.approx(detail["variance_apres"], rel=1e-5)
    pd.testing.assert_frame_equal(prep.transform(frame), result)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("action,minimum", [("BOX_COX", 0), ("LOG", 0), ("LOG1P", -1), ("SQRT", -.01)])
def test_distribution_domain_checked_in_analysis_fit_and_inference(action, minimum):
    invalid = pd.DataFrame({"value": np.linspace(minimum, 5, 40)})
    analysis = make_prep().analyze(invalid)
    with pytest.raises(ValueError, match="hors domaine"):
        validate_step_choice(analysis, "value", "distribution", action)
    prep = make_prep({"value": {"distribution": action}})
    with pytest.raises(ValueError, match="hors domaine"):
        prep.fit(invalid)
    prep.fit(invalid + 2)
    with pytest.raises(ValueError, match="hors domaine"):
        prep.transform(invalid)


@pytest.mark.parametrize("action", OUTLIER_ACTIONS)
def test_outlier_choices_execute_and_report_impact(action):
    frame = pd.DataFrame({"value": np.r_[np.linspace(1., 10., 80), 1000.]})
    prep = make_prep({"value": {"outliers": action, "distribution": "NONE", "scaling": "NONE"}},
                     outlier_method="none")
    detail = prep.analyze(frame).columns[0]["steps"]["outliers"]["comparaison"][action]
    result = prep.fit_transform(frame)
    assert (detail["pct_modifie"] == 0) == (action == "NONE")
    assert np.isfinite(result["value"]).all()
    if action == "TO_MISSING":
        assert result["value"].iloc[-1] == pytest.approx(frame["value"].iloc[:-1].median())


@pytest.mark.parametrize("mode", ["KNN", "ITERATIVE"])
def test_multivariate_imputation_reuses_training_fit(mode):
    x = np.linspace(0, 10, 40)
    values = x * 2 + .1
    values[[4, 10]] = np.nan
    frame = pd.DataFrame({"value": values, "predictor": x})
    prep = make_prep({"value": {"missing": mode, "scaling": "NONE"}}, outlier_method="none",
                     correlation_threshold=1)
    prep.fit(frame)
    first = prep.transform(pd.DataFrame({"value": [np.nan, 4.1], "predictor": [2., 6.]}))
    second = prep.transform(pd.DataFrame({"value": [np.nan, 999.], "predictor": [2., 100.]}))
    assert first["value"].iloc[0] == pytest.approx(second["value"].iloc[0])
    assert first["value"].iloc[0] == pytest.approx(4.1, abs=1.1)


@pytest.mark.parametrize("mode", ["TFIDF", "TFIDF_NGRAM", "COUNT", "HASH", "TFIDF_SVD"])
def test_text_modes_handle_empty_vocabulary_and_unseen_terms(mode):
    text = RobustTextTransformer(column_modes={"text": mode}, column_components={"text": 2})
    text.fit(pd.DataFrame({"text": ["Some sample words", "Another longer sentence", "Useful written text"]}))
    unseen = text.transform(pd.DataFrame({"text": ["qzxqzx", ""]}, index=[9, 3]))
    assert unseen.index.tolist() == [9, 3]
    assert unseen.columns.tolist() == text.get_feature_names_out().tolist()
    assert (unseen.iloc[1] == 0).all()
    text.fit(pd.DataFrame({"text": ["!!!", "", None]}))
    assert text.transform(pd.DataFrame({"text": ["unseen"]})).shape[0] == 1


def test_semantic_profiles_missingness_and_target_association():
    n = 80
    frame = pd.DataFrame({
        "Feature Name": [np.nan] * 40 + np.linspace(5, 20, 40).tolist(),
        "group": ["a"] * 40 + ["b"] * 40,
        "latitude": np.linspace(28, 34, n),
        "flag": pd.Series([True, False] * 40, dtype="boolean"),
        "device_code": np.arange(n) + 6000,
        "almost": [1] * 79 + [2],
        "text": ["Calm green trees grow here.", "Bright red roses bloom everywhere.",
                 "Cold air feels crisp today.", "Big cats rest under shade."] * 20,
        "empty": np.nan,
        "target": [None] + ["no"] * 39 + ["yes"] * 40,
    }, index=pd.RangeIndex(n))
    frame.index = [i // 2 for i in range(n)]
    result = make_prep().analyze(frame, target="target")
    rows = {row["colonne"]: row for row in result.columns}
    assert rows["Feature Name"]["profil"]["manquants"] == 40
    assert rows["Feature Name"]["diagnostic_manquants"]["statut"] == "MAR plausible"
    assert rows["Feature Name"]["manquant_informatif"]
    assert rows["group"]["analyse_cible"]["q_primaire"] < .05
    assert rows["latitude"]["type_semantique"] == "GEOGRAPHIC_COORDINATE"
    assert rows["flag"]["type_semantique"] == "BOOLEAN"
    assert rows["device_code"]["type"] == "DROP_ID"
    assert rows["almost"]["profil"]["quasi_constante"]
    assert rows["text"]["type"] == "TEXT"
    assert rows["empty"]["profil"]["effectif"] == 0
    assert build_analysis_detail(result, "Feature Name")["profil"]["q1"] > 5


def test_duplicate_indices_and_dropped_rows_keep_positional_targets():
    frame = pd.DataFrame({"age": [25., -1., 35., 35., 40., 50.],
                          "target": [10., 12., 23., 24., 31., 42.]}, index=[5, 5, 5, 5, 3, 3])
    prep = AutoPreprocessor(PreprocessConfig(verbose=0, transform_target=False, value_bounds={"age": (0, 120)}, column_overrides={
        "age": {"clean_impossible": "DROP_ROWS"}}), task="regression")
    X, y = prep.fit_transform_xy(frame, "target")
    assert X.index.tolist() == [5, 5, 3, 3]
    assert y.tolist() == [10., 23., 31., 42.]
    X, y = prep.transform_xy(frame, "target")
    assert len(X) == len(y) == 4


@pytest.mark.parametrize("encoding,suffix", [("COUNT_ENCODE", "count"), ("FREQUENCY_ENCODE", "freq")])
def test_supervised_run_can_choose_unsupervised_encoding(encoding, suffix):
    frame = pd.DataFrame({"category": [f"group_{i % 13}" for i in range(80)]})
    labels = np.arange(len(frame)) % 2
    prep = make_prep({"category": {"encoding": encoding}}, preserve_rows=True)
    prep.fit_transform(frame, labels)
    assert prep.transform(pd.DataFrame({"category": ["unseen"]}))[f"category_{suffix}"].iloc[0] == 0


@pytest.mark.parametrize("encoding,prefix", [("HASH_ENCODE", "hash"), ("BINARY_ENCODE", "bin")])
def test_compact_categorical_encoding_keeps_the_training_schema(encoding, prefix):
    frame = pd.DataFrame({"category": [f"group_{i % 13}" for i in range(80)]})
    prep = make_prep({"category": {"encoding": encoding}}, preserve_rows=True)
    output = prep.fit_transform(frame)
    assert all(column.startswith(f"category_{prefix}_") for column in output)
    test = prep.transform(pd.DataFrame({"category": ["unknown", None]}))
    assert output.columns.equals(test.columns)
    assert np.isfinite(test.to_numpy()).all()
    if encoding == "BINARY_ENCODE":
        assert (test == 0).all().all()


def test_discrete_numeric_encoded_as_category_has_valid_recommendations():
    frame = pd.DataFrame({"measurement": np.arange(50) * .31, "count": np.arange(50) % 3})
    prep = make_prep(encode_discrete_as_categorical=True)
    row = prep.analyze(frame).columns[1]
    assert row["type"] == "NUMERIC_DISCRETE"
    assert row["steps"]["distribution"]["choices"] == ["AUTO"]
    assert row["steps"]["encoding"]["action"] == "ONEHOT"


def test_model_candidates_are_fold_safe_and_deterministic():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"value": rng.normal(size=60), "noise": rng.normal(size=60)})
    frame["target"] = (frame["value"] > 0).astype(int)
    kwargs = dict(family="logistic", candidates=("none", "standard"), cv=2)
    assert compare_preprocessing(frame, "target", **kwargs) == compare_preprocessing(frame, "target", **kwargs)
    assert make_prep(model_family="random_forest").analyze(frame[["value"]]).columns[0]["steps"]["scaling"]["action"] == "NONE"


def test_input_validation_and_trusted_joblib_gate(tmp_path, monkeypatch):
    analysis = make_prep().analyze(pd.DataFrame({"value": np.linspace(1, 3, 40)}))
    choices = tmp_path / "choices.json"
    for payload in ([], {"version": 2, "choices": {}},
                    {"choices": {"value": {"missing": ["CONSTANT", float("inf")]}}},
                    {"choices": {"value": {"outliers": ["WINSORIZE", {"lower": .9, "upper": .1}]}}}):
        choices.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            import_user_choices(analysis, str(choices))
    model = tmp_path / "untrusted.joblib"
    model.write_bytes(pickle.dumps({}))
    app = App.__new__(App)
    app.messagebox = type("Dialogs", (), {"askyesno": lambda *args: False})()
    monkeypatch.setattr(AutoPreprocessor, "load", lambda *args: pytest.fail("Unsafe file was loaded"))
    app._do_load_preprocessor(str(model))


def test_date_parsing_and_p_value_adjustment():
    dates = parse_dates(pd.Series(["01/02/2022", "22:38:02", "bad"]))
    assert (dates.iloc[0].month, dates.iloc[0].day) == (2, 1)
    assert (dates.iloc[1].year, dates.iloc[1].hour, dates.iloc[1].minute) == (1970, 22, 38)
    assert pd.isna(dates.iloc[2])
    assert adjusted_p_values({"a": .01, "b": .04, "c": np.nan}) == {"b": .04, "a": .02}


def test_manual_mapping_and_explicit_ordinal_order():
    frame = pd.DataFrame({"category": ["red", "blue", "green"] * 15})
    params = parse_step_parameters("encoding", "MANUAL_MAP", '{"red": 5, "blue": -2}', "CATEGORICAL_LOW_CARD")
    prep = make_prep({"category": {"encoding": ("MANUAL_MAP", params)}})
    prep.fit(frame)
    assert prep.transform(pd.DataFrame({"category": ["red", "unknown", "blue"]}))["category_mapped"].tolist() == [5, 0, -2]
    with pytest.raises(ValueError, match="ordre explicite"):
        make_prep({"category": {"encoding": "ORDINAL"}}).fit(frame)


def test_temporal_regression_candidates_use_chronological_folds():
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=42),
                          "value": rng.normal(size=42), "target": rng.normal(size=42)})
    result = compare_preprocessing(frame.iloc[::-1], "target", "linear", time_column="date",
                                   candidates=("none",), cv=2, task="regression")
    assert result["scoring"] == "neg_mean_absolute_error"
    assert len(result["results"][0]["scores"]) == 2
