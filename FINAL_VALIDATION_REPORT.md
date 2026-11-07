# Auto-Preprocessor final validation

**Verdict — READY FOR CONTROLLED PILOT**

The Python application passed the complete suite on the tested Windows environment, including the Tkinter smoke test. The verdict covers controlled local workloads with trusted models and bounded inputs.

## Full pytest results

| Run | Collected | Passed | Failed | Skipped | Warnings | Time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline at `1d26e98` | 194 | 193 | 0 | 1 | 15 | 151.31 s |
| Final working tree | 200 | 200 | 0 | 0 | 14 | 67.47 s |

The observed elapsed time fell by **83.84 s (55.4%)**. The baseline ran in the workspace sandbox, where Tcl could not initialize. The final run used the same Python installation outside that sandbox, ran the original GUI smoke test successfully, and included six additional cases. This is a comparison of observed runs on the same machine under those conditions. No test remained uncompleted in the final run.

Final command

```powershell
python -m pytest -W default -rs --durations=20 --basetemp=.pytest_cache/final-validated-source
```

The final run reported 14 warnings from rare-class stratification, unknown One-Hot categories, precision loss on nearly identical values, and the joblib/NumPy array-loading deprecation. Python compilation of every tracked Python file, `git diff --check`, and `python -m pip check` passed. An AST comparison against HEAD verified that all original test functions and their assertions remained present and unchanged.

## Performance diagnosis and changes

The three baseline invariant tests took 22.69 s, 15.11 s, and 14.64 s. They independently analyzed the same nine datasets. Other tests repeatedly analyzed identical demo frames to assert different properties of the UI and reports. These deterministic analyses now share module-scoped setup. Each consuming test receives copies of the data and nested results. The tests that verify fits, leakage, serialization, and the GUI still execute those paths. Dataset sizes, statistical assertions, CV folds, and seeds were preserved.

A representative cProfile run over breast-cancer analysis and supervised churn analysis attributed 3.38 s to the numerical comparison, 2.84 s to legacy distribution previews, 1.54 s to numerical cleaning comparisons, 1.36 s to missingness evidence, and 0.50 s to leakage diagnostics. Text evidence took 0.08 s. Repeated analysis and statistical estimation were the measured bottlenecks. The changes reuse a successful Yeo–Johnson preview for the same treated column within an analysis, reuse invariant descriptive statistics and numeric-string parsing, and stringify only the five displayed examples. No fit is shared across datasets, train/test splits, or CV folds.

A sequential cProfile comparison against HEAD on those same two inputs measured **14.079 s before and 13.624 s after**. Yeo–Johnson optimizer calls went from **88 to 68** and numeric-string parser calls from **74,652 to 50,752**. The complete analysis outputs matched on both inputs. The larger pytest improvement primarily reflects reuse of test setup.

### Ten slowest final tests

| Time | Phase | Test |
|---:|---|---|
| 16.87 s | 9.23 s setup + 7.64 s call | `test_invariant_c1_recommendation_text_is_single_source_across_outputs` |
| 4.52 s | call | `test_analysis_demo_recommendations` |
| 3.10 s | call | `test_tk_smoke_load_analyze_and_treeview_count` |
| 2.60 s | call | `test_churn_target_relations_are_populated_for_every_feature` |
| 1.76 s | call | `test_dropped_column_detail_shows_only_keep_or_drop_stage` |
| 1.76 s | call | `test_strong_legit_feature_not_flagged_breast_cancer` |
| 1.48 s | call | `test_preview_table_figure_and_png_export` |
| 1.41 s | call | `test_accept_analysis_recommendations_is_noop[data/churn_telecom_demo.csv]` |
| 1.35 s | call | `test_phase2_recommendation_matches_report_action_on_demo_datasets` |
| 1.20 s | call | `test_accept_recommendations_matches_sklearn_dataset` |

Pytest attributes shared fixture setup to its first consumer. Other entries use their reported call times.

## Demo and edge-case validation

The three CSVs ran through the application's analysis and supervised preprocessing path with an 80/20 split and seed 42. For every run, each feature had a profile, an action and recommendation steps. Train and holdout indexes were disjoint, feature schemas matched, all outputs were finite, and the input frame was unchanged. Classification labels were checked against the source values at the retained indexes. Regression targets were inverse-transformed and compared against the source values.

| Demo | Input rows × columns | Profiled features | Train output | Holdout output | Largest checked allocation | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Churn telecom | 3,015 × 12 | 11/11 | 2,405 × 35 | 603 × 35 | 91,390 cells / 731,120 bytes | 3.494 s |
| Fraud signal | 1,000 × 14 | 13/13 | 800 × 48 | 200 × 48 | 51,200 cells / 409,600 bytes | 2.739 s |
| Immobilier | 2,000 × 10 | 9/9 | 1,600 × 20 | 400 × 20 | 32,000 cells / 256,000 bytes | 2.543 s |

The churn validation found that all 603 holdout labels previously became unknown because the training path normalized `Non` and `Oui` before fitting the target encoder. The paired transform now uses the same normalization. The corrected holdout contained 436 `non`, 167 `oui`, and zero unknowns. The new regression test also verifies that genuinely unseen labels retain the `-1` code and duplicate indexes retain row alignment. The fraud holdout contained 191 negatives and nine positives. Joblib used its logical-CPU fallback in the immobilier run when WMIC was unavailable.

The edge-case exercise used duplicate indexes, both signs of infinity, invalid dates, mixed values, a fully missing column, and a constant column. All six rows and their index labels survived, and the 12 output features were finite. Infinity input still emitted NumPy warnings in exploratory intermediate calculations. Separate tests covered the empty fitted schema, a 2,048-term sparse text vocabulary, and 1,024 categorical levels encoded into a bounded 32-column schema.

## Security and packaging

- Joblib loading requires explicit GUI trust confirmation. Tests exercised refusal, non-affirmative responses, and an affirmative trusted model round trip. Joblib can execute arbitrary code. A public service must keep uploaded models away from this loader.
- CSV and choices JSON were rejected before parsing at configured size limits. Existing tests exercised invalid JSON structure, version and transformation parameters. The production limits are 100 MiB for table inputs and 1 MiB for choices. Path checks accepted a local path with spaces and rejected a directory, a missing file and an HTTP URL. They are regular-file checks rather than a workspace allowlist.
- An actual Excel workbook opened successfully. A compressed input exceeding the configured expanded-byte limit and a separate archive with 10,001 entries were both rejected before calling the workbook parser. Archive metadata and file size checks are input guards. Controlled deployment still needs parser isolation for hostile uploads.
- Resource tests verified pre-allocation rejection, sparse output shape and zero fill, and behavior with wide text and categorical vocabularies. The defaults are 20,000,000 dense cells, 160,000,000 bytes per checked allocation, and 100,000 generated columns. These estimates do not bound total process memory or all sparse and parser buffers.
- Tested environment was Windows with Python 3.13.15, pytest 8.0.0, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.8.0, joblib 1.3.2, openpyxl 3.1.5, and Tcl/Tk 8.6.15. The Windows executable build and a cross-version dependency matrix were outside this validation. No executable compatibility claim is made.

## Files changed

- `auto_preprocessor.py` — reuse of same-column statistical work and consistent target normalization in the paired fit/transform API.
- `statistical_profile.py` — reuse of baseline moments, cardinality, median, and the scaler input frame.
- `test_auto_preprocessor.py` — shared deterministic demo and invariant analysis setup.
- `test_memory_security.py` — six additional target-alignment, resource, path, Excel, JSON, and empty-input checks.
- `FINAL_VALIDATION_REPORT.md` — measurements and validation record.
