"""Reproducible large-CSV stress run using the same isolated worker as the GUI."""

import argparse
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory

from app_preprocess import read_table
from auto_preprocessor import AutoPreprocessor, PreprocessConfig, to_sparse_matrix
from isolated_processing import ProcessLimits, run_isolated


def process_stress_csv(path, hash_width):
    frame = read_table(path)
    config = PreprocessConfig(
        verbose=0, sparse_output=True, preserve_rows=True, correlation_threshold=1,
        clean_string_values=False, parse_numeric_strings=False, adaptive_recommendations=False,
        column_overrides={
            "category": {"type": "FORCE_CATEGORICAL", "encoding": "ONEHOT"},
            "text": {"type": "FORCE_TEXT", "text": ("HASH_TEXT", {"n_features": hash_width})},
        },
    )
    features = AutoPreprocessor(config).fit_transform(frame)
    matrix = to_sparse_matrix(features)
    return matrix.shape, matrix.nnz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=900_000)
    parser.add_argument("--hash-width", type=int, default=2_048)
    parser.add_argument("--memory-mib", type=int, default=1_536)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--directory", type=Path)
    options = parser.parse_args()
    with TemporaryDirectory(dir=options.directory, prefix="autoprep-stress-") as directory:
        path = Path(directory, "large.csv")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("measurement,category,text\n")
            for row in range(options.rows):
                handle.write(f"{row % 10000}.25,cat_{row % 64},weather water river tree wind "
                             f"cloud daylight rainfall stone word{row % 4096:04d}\n")
        size = path.stat().st_size
        print(f"Input bytes: {size:,}", flush=True)
        limits = ProcessLimits(options.memory_mib * 1024**2, options.timeout)
        (shape, nnz), usage = run_isolated(process_stress_csv, path, options.hash_width,
                                          limits=limits, with_usage=True)
        if shape[0] != options.rows:
            raise RuntimeError(f"Expected {options.rows:,} rows, received {shape[0]:,}.")
        print(f"Output shape: {shape}, nnz: {nnz:,}")
        print(f"Worker peak RSS: {usage.peak_rss_bytes:,} bytes; elapsed: {usage.elapsed_seconds:.2f} s")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
