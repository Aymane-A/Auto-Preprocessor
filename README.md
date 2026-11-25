# AutoPrepa

**A local Windows application for inspecting and preparing tabular data for machine learning.**

AutoPrepa is a university engineering project that provides a scikit-learn-compatible preprocessing module and a desktop application for preparing tabular datasets.

It supports supervised classification and regression, as well as unsupervised preprocessing. The application is designed to work with different tabular datasets without requiring dataset-specific preprocessing code.

## Features

- Import CSV, TSV, TXT, XLSX, XLS, JSON, and Parquet files.
- Inspect missing values, distributions, outliers, duplicates, correlations, and relationships to a selected target.
- Review and modify per-column preprocessing recommendations.
- Configure imputation, scaling, outlier handling, categorical and text encoding, feature selection, and class-imbalance handling.
- Compare preprocessing configurations using cross-validation, including nested evaluation options.
- Export transformed training, testing, and full datasets, along with reports and the fitted preprocessor.
- Run the web interface or the Tkinter desktop application locally.

Recommendations are statistical guidance, not a guarantee of predictive performance or scientific validity. Evaluate the resulting workflow using a split appropriate to the data and an independent validation protocol.

## Requirements

- Windows 10 or Windows 11
- Python 3.13 — currently validated version
- Node.js 20.19+ or 22.12+
- npm

Python runtime dependencies are listed in `requirements.txt`.

Development and testing dependencies are listed in `requirements-dev.txt`.

Frontend dependencies and build tools are locked in `frontend/package-lock.json`.

## Installation and Launch

### 1. Clone the repository

```powershell
git clone https://github.com/Aymane-A/Auto-Preprocessor.git
cd Auto-Preprocessor
```

### 2. Set up the Python environment

Run the following commands from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Install and build the frontend

```powershell
Push-Location frontend

npm ci
npm run build

Pop-Location
```

### 4. Start the application

```powershell
python serve.py
```

Open the application in your browser:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

To stop the server, press `Ctrl+C` in the terminal.

### Desktop application

After installing the Python requirements, launch the Tkinter desktop interface:

```powershell
python app_preprocess.py
```

### PowerShell activation alternative

If PowerShell blocks virtual-environment activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe serve.py
```

## Import and Prepare a Dataset

1. Select a local dataset in the Import step.
2. Choose supervised mode and select a target column for classification or regression, or choose unsupervised mode when no target is available.
3. Review the exploratory analysis and preprocessing recommendations.
4. Configure the pipeline and inspect the preview.
5. Run preprocessing.
6. Download the transformed data and reports from the export step.

Dataset files are processed by the local service and are not uploaded to a remote application service.

The web server binds to `127.0.0.1` by default.

## Supported Formats and Limits

### File formats

| Format | Details |
|---|---|
| CSV, TSV, TXT | Comma, semicolon, tab, and pipe separators are detected. UTF-8 and Latin-1 encodings are tried. |
| XLSX | Read using `openpyxl`. |
| XLS | Read using `xlrd`. |
| JSON | Read as a pandas table. |
| Parquet | Read using `pyarrow`. |

Macro-enabled XLSM files are not accepted by the web uploader. Save them as XLSX before importing.

### Resource limits

| Resource | Default limit |
|---|---:|
| Web upload size | 200 MiB |
| Expanded Excel archive size | 300 MiB |
| Excel archive entries | 10,000 |
| Per-entry Excel compression ratio | 1,000 |
| Parquet rows | 2,000,000 |
| Parquet uncompressed size | 300 MiB |
| Worker memory | 2 GiB |
| Worker timeout | 120 seconds |
| Desktop input size | 100 MiB |

Web upload limits can be adjusted using:

- `AUTOPREP_MAX_TABLE_MB`
- `AUTOPREP_WORKER_MB`
- `AUTOPREP_TIMEOUT_SECONDS`

These limits are intended for trusted local workloads. They do not guarantee that a large dataset will fit in available system memory.

Wrapper-based feature-selection methods are intentionally bounded and may be significantly slower than filter methods.

## Testing

Run the Python test suite from the repository root:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The latest local test run completed successfully:

**594 passed, 72 warnings**

The suite includes unit tests, integration workflows, scientific validation, regression checks, and feature-selection tests.

The reported result corresponds to the tested local project state. Re-run the suite against the current commit before making release or publication claims.

### Frontend tests and build

```powershell
Push-Location frontend

npm ci
npm test
npm run build

Pop-Location
```

Run the Python and frontend checks before considering a release or publication.

## Demo and Dataset Provenance

The `data` directory contains deterministic, project-generated synthetic demo and test fixtures.

These fixtures contain no sourced customer or property records and are not required by the application to import or process user datasets.

Regenerate the fixtures from the repository root:

```powershell
python data/generate_demo_fixtures.py
```

See [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md) for details about their generation.

Common local dataset files are ignored by Git. Keep personal inputs under `local_datasets/` or `datasets/` rather than adding them to the demo fixture directory.

The project's license status is documented separately. Check the repository's licensing information before redistributing or reusing the software.

## Troubleshooting

### The page does not load

Confirm that `npm run build` completed successfully and that the server is still running at:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

### Upload rejected

Check the file extension, size, and readability. Verify that the file does not exceed the configured limits.

### Processing times out or reaches memory limits

Reduce the dataset size or adjust the local worker limits for a trusted workload.

Do not expose the service to a network to bypass these limits.

### Excel or Parquet import fails

Check that the corresponding runtime dependency is installed:

- `openpyxl` for XLSX
- `xlrd` for XLS
- `pyarrow` for Parquet

### PowerShell activation is denied

Use the `.venv\Scripts\python.exe` commands described in the installation section without activating the environment.

## Security and Local Processing

AutoPrepa is designed for local processing.

The API listens on loopback by default and processes uploads in isolated worker processes with memory and time limits.

However, these limits do not restrict worker filesystem or network access, and they do not make hostile files safe.

- Use trusted input files.
- Keep the service bound to `127.0.0.1`.
- Do not expose the application on a public interface without a separate security review.
- Use a separately restricted environment when processing untrusted datasets.

**Local processing is not a security sandbox.**

## Project Status

AutoPrepa is an evolving university engineering project.

Its goal is to make tabular preprocessing more transparent, configurable, and reproducible while keeping data preparation separate from model evaluation.

Scientific validation remains dependent on the dataset, preprocessing configuration, model, and evaluation protocol.

Feedback, bug reports, and suggestions are welcome.

---

**Developed by [Aymane A.](https://github.com/Aymane-A)**

[View the repository on GitHub](https://github.com/Aymane-A/Auto-Preprocessor)
