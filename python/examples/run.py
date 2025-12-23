#!/usr/bin/env python3
"""
run.py

Tiny runnable demo that loads a phase-level trace CSV and runs the baseline
time-integrated NNLS estimator to recover step-level coefficients
(beta0, beta1, beta2).

Why this script uses dynamic import
-----------------------------------
Your estimator file is named with dots:
  baseline/baseline.timeintegral.py

Dotted filenames are awkward to import as standard Python modules. To avoid
forcing any renames, this script loads that file directly via importlib and
then calls estimate_betas_baseline().

Usage
-----
From the python/ directory (your current layout):

  python examples/run.py --csv data/traces/sample.csv --chunk-size 64

You can also pass an absolute or repo-relative CSV path.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_baseline_module(baseline_file: Path):
    """
    Dynamically load the baseline estimator module from a file path.

    NOTE (Python 3.12 / dataclasses):
    -------------------------------
    dataclasses may consult sys.modules[cls.__module__] during class decoration.
    When loading via importlib, we must register the module in sys.modules
    *before* executing it, otherwise sys.modules.get(...) returns None and
    dataclasses can fail with:
      AttributeError: 'NoneType' object has no attribute '__dict__'

    Inputs
    ------
    baseline_file : Path
        Path to baseline/baseline.timeintegral.py

    Output
    ------
    module : Python module object
        Exposes estimate_betas_baseline(phases: pd.DataFrame, chunk_size: int, ...).
    """
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline estimator file not found: {baseline_file}")

    spec = importlib.util.spec_from_file_location("baseline_timeintegral", str(baseline_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {baseline_file}")

    module = importlib.util.module_from_spec(spec)

    # Critical for Python 3.12 dataclasses: register before exec_module().
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _default_paths():
    """
    Compute defaults robustly, regardless of current working directory.

    Assumes this script lives at: python/examples/run.py
    and that "python/" is the working project root for the code artifact.

    Returns
    -------
    python_root : Path
        The python/ directory in your repo.
    default_csv : Path
        python/data/traces/sample.csv
    baseline_file : Path
        python/baseline/baseline.timeintegral.py
    """
    script_dir = Path(__file__).resolve().parent  # .../python/examples
    python_root = script_dir.parent               # .../python
    default_csv = python_root / "data" / "traces" / "sample.csv"
    baseline_file = python_root / "baseline" / "baseline.timeintegral.py"
    return python_root, default_csv, baseline_file


def main() -> int:
    python_root, default_csv, baseline_file = _default_paths()

    parser = argparse.ArgumentParser(description="Run baseline trace-only beta estimation on a CSV trace.")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="Path to phase trace CSV.")
    parser.add_argument("--chunk-size", type=int, default=64, help="Prefill chunk size C (tokens per step).")
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Optional clamp for tiny phase durations when forming lambda_hat (default 0.0 = paper-faithful).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    baseline = _load_baseline_module(baseline_file)

    phases = pd.read_csv(csv_path)
    res = baseline.estimate_betas_baseline(
        phases,
        chunk_size=args.chunk_size,
        min_duration_sec=args.min_duration_sec,
    )

    print("=== Baseline beta estimation ===")
    print(f"Python root: {python_root}")
    print(f"CSV:         {csv_path}")
    print(f"Chunk size:  {args.chunk_size}")
    print(f"T_min:       {args.min_duration_sec}")
    print()
    print("Estimated betas:")
    print(f"  beta0 (sec/step):  {res.beta0:.6f}")
    print(f"  beta1 (sec/token): {res.beta1:.6f}")
    print(f"  beta2 (sec/token): {res.beta2:.6f}")
    print()
    print("Diagnostics:")
    for k, v in res.diagnostics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
