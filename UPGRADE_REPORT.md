# Auto-Preprocessor intelligence upgrade

The measurements and initial test counts below describe the original upgrade commit. The subsequent merge with `main` retained its adaptive preprocessing rules, configurable value bounds, row-level preview, and Leave-One-Out encoding alongside the upgrade's statistical profiles and optional model evaluation.

## Repository review

| Files inspected | What they cover |
|---|---|
| `auto_preprocessor.py` | Raw cleaning, type routing, analysis, leakage diagnostics, fit and transform, selection, target alignment, joblib persistence |
| `app_preprocess.py` | File loading, analysis and processing workers, Tkinter controls, choices import and export, train/test workflow |
| `test_auto_preprocessor.py` | Existing core, statistics, integration, and serialization tests |
| `benchmark_multi.py`, `signal_check.py`, `credit_g_check.py`, `ablation_credit_g.py` | Baselines, cross-validation entry points, signal checks, OpenML benchmarks, and ablation |
| `README.md`, `requirements.txt`, `build_exe.bat`, `data/*.csv` | User workflow, dependencies, packaging, and all three local demos |

Changed `auto_preprocessor.py` and `app_preprocess.py` in place. Added `statistical_profile.py` for reusable statistics and transform semantics, `model_evaluation.py` for optional fold-safe evaluation, and `test_intelligence_upgrade.py` for regression coverage. The original test file was preserved.

## Delivered behavior

- Analysis returns `type_semantique`, `confiance_type`, `raison_type`, and a structured `profil` for features. It retains the legacy type field used by the execution pipeline. Semantic evidence distinguishes numerical and categorical binaries, booleans, discrete measurements, probable coordinates, mixed columns, and high concentration. Free-text recognition uses token structure, punctuation, vocabulary, and code-like patterns alongside length and cardinality.
- Numerical profiles record quantiles, variance, coefficient of variation, sign proportions, infinities, and IQR severity. Distribution labels use moment and shape heuristics. Shapiro–Wilk is computed on at most 5,000 observations. D’Agostino–Pearson is reported from 20 observations onward. Category profiles include entropy, concentration, top counts, imbalance, and the share of observations in rare categories. Text profiles include term and bigram counts from up to 2,000 documents. Date profiles include date validity, calendar counts, span, and median spacing.
- Missingness diagnostics examine each partially missing feature against every other feature on a shared deterministic sample of at most 5,000 rows. The result reports how many comparisons met test requirements and applies Benjamini–Hochberg within each set of tests. An observed association supports the label MAR plausible. The mechanism remains indeterminate when evidence is insufficient. A missingness indicator recommendation tied to the target requires adjusted q ≤ 0.05 and effect size ≥ 0.10.
- Optional target diagnostics exclude missing target rows and report effect size and significance from applicable Pearson, Spearman, Kendall, Kruskal–Wallis, point-biserial, chi-square, or Fisher tests. Pearson includes a 95% interval. A primary test per feature receives a multiple-testing adjustment across features. The legacy exploratory mutual-information and suspected-leakage fields remain available. Association is descriptive evidence.
- The numerical comparison shows observed versus transformed variance, skewness, distinct-value loss, and constant results. New executable choices include log, cube root, normal and uniform quantiles, empirical rank, and ordinal binning. Logarithm, square-root, and Box–Cox domains are checked before fitting and at inference. Nonfinite numerical input is treated as missing in the numerical pipeline. Preview fitting uses up to 5,000 observations, while domain checks examine the full analysis sample. Results describe one operation at a time on observed data and can differ from the final training pipeline after cleaning and imputation.
- Outlier previews report bounds, affected share, variance, skewness, and distinct-value loss. New options include Z-score and modified-Z clipping, median replacement, and conversion to missing before imputation. Statistical flags cannot establish that an extreme value is an error. KNN and iterative imputation fit on numeric training predictors and require 5–10,000 training rows and 2–50 numeric predictors.
- Categorical additions include count, frequency with a target present, compact binary, hash, and explicit manual mapping. The UI estimates width and describes handling of unknown categories. Ordinal encoding requires a recognized order or explicit mapping of order. Target encodings used for training feature selection and model input are out of fold. Text supports unigram TF-IDF, TF-IDF with bigrams, token counts, hashing, and TF-IDF with SVD. Blank vocabulary and unseen terms retain a stable schema. Date handling uses a fixed reference for clock-only fields, preserves year-first ISO parsing, and adds day-of-year and observed hour/minute detail.
- The French **Profil et comparaison** view shows profiles, evidence, confidence, applicable alternatives, and preview metrics for selected features. The existing per-column editor, multi-selection, reset, and export remain the main controls. A model-family control changes the default scaling guidance for tree families. The existing analysis and execution workers handle statistical computation outside the Tkinter event loop.
- `compare_preprocessing(frame, target, family, ...)` compares up to four scaling settings across 2–5 folds. It uses stratified folds for classification, ordinary K-fold for regression, and chronological splits with distinct timestamps when a time column is supplied. The full preprocessing fit runs inside each model pipeline. Chronological evaluation uses unsupervised category encoding. Model families include linear/logistic, trees, random forests, boosting, KNN, SVM, and neural networks. Optional XGBoost and LightGBM imports occur only when requested.
- Positional masks keep targets aligned when rows are removed even if index labels repeat. Table inputs are limited to 100 MiB and Excel archive expansion is checked before reading. Configuration JSON is limited to 1 MiB and checked for shape, version, action, and parameters. The GUI asks the operator to confirm trust before loading a joblib file. Direct Python joblib loading remains a trusted-file API. Large dense One-Hot and output matrices raise explicit errors.

## Validation results

The comparison loaded the original engine from commit `07d8a25` and ran the upgraded engine on the same frames. The application execution path used an 80/20 holdout with seed 42. Coverage counts features with both a profile and step recommendations. Each completed run verified finite output, equal feature and target row counts, and an unchanged input frame.

| Dataset | Input rows × columns | Coverage | Analysis before → after | Execution | Train / test output | Input / output bytes |
|---|---:|---:|---:|---:|---|---:|
| Churn telecom | 3,015 × 12 | 11/11 | 1.471 → 2.668 s | 1.246 s | 2,405 × 35 / 603 × 35 | 1,581,809 / 445,184 |
| Fraud signal | 1,000 × 14 | 13/13 | 0.943 → 1.705 s | 1.048 s | 800 × 47 / 200 × 47 | 652,795 / 196,000 |
| Immobilier | 2,000 × 10 | 9/9 | 1.057 → 1.768 s | 0.700 s | 1,600 × 20 / 400 × 20 | 585,421 / 258,640 |
| Synthetic edge cases | 120 × 10 | 9/9 | 0.323 → 0.481 s | 0.384 s | 64 × 8 / 16 × 8 | 43,792 / 3,200 |

| Dataset | Semantic types detected | Recommendation differences and blocked operations |
|---|---|---|
| Churn telecom | 1 identifier, 1 datetime, 3 discrete, 1 continuous, 4 nominal, 1 categorical binary | Original primary actions retained. Log and Box–Cox blocked for one feature each. |
| Fraud signal | 2 identifiers, 2 datetime, 1 continuous, 1 discrete, 7 nominal | Original primary actions retained. The tested numerical domains permitted all distribution previews. |
| Immobilier | 5 discrete, 2 nominal, 1 categorical binary, 1 text | `description` moved from One-Hot to TF-IDF/SVD. Log and Box–Cox blocked for two features each. |
| Synthetic edge cases | 1 identifier, 1 all-missing, 1 constant, 2 continuous, 1 high-cardinality nominal, 1 datetime, 1 text, 1 mixed | The finite part of `value` supported a Yeo–Johnson recommendation. Text moved from a categorical leakage alert to TF-IDF/SVD. Two features blocked log and Box–Cox. One feature also blocked log1p and square root. |

All four executions completed with zero processing errors. The original engine emitted failed leakage-scoring folds on the synthetic infinities. The upgraded leakage diagnostic treats them as missing. Joblib emitted a Windows CPU discovery diagnostic during the immobilier run and used its fallback. The synthetic fixture includes a fully missing feature, a constant, infinities, a large positive outlier, a zero-heavy feature, invalid dates, mixed strings, and 40 missing targets.

Final command `python -m pytest -q --basetemp=.pytest_cache/upgrade-final-validation` produced **95 passed, 0 failed, 26 warnings** under Python 3.13.15. Warnings came from rare-class stratification and joblib’s NumPy array loading. The first baseline run encountered a Windows access error in pytest’s default temporary directory. The workspace-local temporary directory allowed all fixtures to run. Final Python compilation and `git diff --check` passed. Tkinter data adapters and the joblib trust gate were tested without launching the desktop interface.

## Known limits and readiness


- The output remains a dense numeric DataFrame. Sparse vectorizers are used internally. Large sparse-first workflows, streaming file ingestion, and process-isolated parsers remain outside this implementation.
- Temporal interpolation, forward/backward filling, and automatic unit harmonization need explicit time order, entity boundaries, and domain policy. Use the provided per-feature imputers and clean units upstream where that context is available. The statistical profiles do not claim verified multimodality, seasonality, or language detection.
- A general callable custom-transform registry is absent. Target encoding uses the existing cross-fitting path. The merged project also exposes Leave-One-Out encoding for binary and continuous targets. Fixed-width hash, frequency, count, binary, or explicit manual mappings are available as compact alternatives. Cross-validation currently compares scaling configurations. XGBoost and LightGBM integrations were not exercised in this validation.
- The UI has reset controls, statistical comparisons, and a before/after preview for individual features. It does not maintain a choice-history stack. Model scores are exposed through Python. For supervised model selection, run exploratory analysis on training data only. The GUI can still profile the whole loaded file.
- The four-dataset exercise measures Python DataFrame memory including indexes, rather than peak process memory. The added statistical work can increase analysis latency. Pairwise redundancy is capped at 500 input numeric columns during analysis, and missingness diagnostics still scale with the number of partially missing features and total features. Whole-dataset analysis is deterministic on its documented sample beyond 200,000 rows.
- Suitable for local exploratory preprocessing and controlled pilots with supported tabular inputs. A production deployment that accepts untrusted uploads or very wide datasets would need process isolation, resource and throughput tests, explicit model-validation policy, and interactive GUI verification. The local OpenML-dependent benchmark and Windows executable build were outside this validation.
