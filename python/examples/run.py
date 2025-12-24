#!/usr/bin/env python3
"""
examples/run.py

Tiny runnable demo: load a phase-level trace CSV and run the baseline
time-integrated NNLS estimator to recover step-level coefficients:
  (beta0, beta1, beta2).

Key assumption
--------------
You are running this from the repo's `python/` directory after installing the
package (recommended):

  python -m pip install -e .

Usage
-----
From the `python/` directory:

  python examples/run.py
  python examples/run.py --csv data/traces/sample.csv --chunk-size 64
  python examples/run.py --correction-mode end
  python examples/run.py --correction-mode uniform
  python examples/run.py --correction-mode none
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from estimators.baseline import estimate_betas_baseline


def _default_csv_path() -> Path:
    """
    Default trace location relative to this file, not the working directory.

    Layout assumption:
      python/examples/run.py
      python/data/traces/sample.csv
    """
    script_dir = Path(__file__).resolve().parent          # .../python/examples
    python_root = script_dir.parent                       # .../python
    return python_root / "data" / "traces" / "sample.csv"


def _resolve_path(p: str) -> Path:
    """
    Resolve a user-provided path:
      - expand ~
      - if relative, interpret relative to the current working directory
      - return an absolute Path
    """
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def main() -> int:
    default_csv = _default_csv_path()

    parser = argparse.ArgumentParser(
        description="Run baseline trace-only beta estimation on a phase-level CSV trace."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help="Path to phase trace CSV (absolute or relative). Default: data/traces/sample.csv",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Prefill chunk size C (tokens per prefill step).",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help=(
            "Optional clamp for tiny phase durations when forming lambda_hat. "
            "Default 0.0 = paper-faithful (no clamp)."
        ),
    )
    parser.add_argument(
        "--correction-mode",
        type=str,
        default="end",
        choices=["end", "uniform", "none"],
        help=(
            "Partial-chunk correction mode for prefill chunking: "
            "'end' (default), 'uniform', or 'none'."
        ),
    )

    args = parser.parse_args()

    # Resolve and validate the trace path.
    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load the trace.
    phases = pd.read_csv(csv_path)

    # Run estimation.
    res = estimate_betas_baseline(
        phases,
        chunk_size=args.chunk_size,
        min_duration_sec=args.min_duration_sec,
        correction_mode=args.correction_mode,
    )

    # Print a compact, human-readable report.
    print("=== Baseline beta estimation ===")
    print(f"CSV:             {csv_path}")
    print(f"Chunk size:      {args.chunk_size}")
    print(f"T_min clamp:     {args.min_duration_sec}")
    print(f"Correction mode: {args.correction_mode}")
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
