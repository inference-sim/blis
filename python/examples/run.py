#!/usr/bin/env python3
"""
examples/run.py

Runnable demo: load a phase-level trace CSV and run either:
  - baseline time-integrated NNLS estimator, or
  - iterative step-density reweighted estimator (MM-style NNLS).

Usage
-----
From the `python/` directory (recommended after `pip install -e .`):

  # Baseline
  python examples/run.py
  python examples/run.py --algo baseline --correction-mode end
  python examples/run.py --algo baseline --correction-mode uniform
  python examples/run.py --algo baseline --correction-mode none

  # Iterative
  python examples/run.py --algo iterative --correction-mode beta_informed
  python examples/run.py --algo iterative --correction-mode end
  python examples/run.py --algo iterative --max-outer-iters 30 --tol 1e-7
  python examples/run.py --algo iterative --damping-eta 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from estimators import estimate_betas_baseline
from estimators import estimate_betas_iterative


# -------------------------
# Path helpers
# -------------------------
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


# -------------------------
# CLI
# -------------------------
def _build_parser(default_csv: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trace-only beta estimation (baseline or iterative) on a phase-level CSV trace."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help="Path to phase trace CSV (absolute or relative). Default: data/traces/sample.csv",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="baseline",
        choices=["baseline", "iterative"],
        help="Which estimator to run: baseline (default) or iterative.",
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
        choices=["end", "uniform", "none", "beta_informed"],
        help=(
            "Prefill partial-chunk correction mode. "
            "For baseline: end|uniform|none. "
            "For iterative: end|uniform|none|beta_informed."
        ),
    )

    # Iterative-only controls (ignored for baseline, but we still parse them for convenience)
    parser.add_argument(
        "--max-outer-iters",
        type=int,
        default=25,
        help="(Iterative only) Maximum number of outer MM iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="(Iterative only) Convergence threshold on ||beta^{m+1} - beta^m||_2.",
    )
    parser.add_argument(
        "--damping-eta",
        type=float,
        default=1.0,
        help="(Iterative only) Damping eta in (0,1]. 1.0 means no damping.",
    )
    parser.add_argument(
        "--init-via-baseline-correction",
        type=str,
        default="end",
        choices=["end", "uniform", "none"],
        help="(Iterative only) Baseline correction mode used for beta^(0) initialization (if init_beta not provided).",
    )

    return parser


def _print_common_header(*, algo: str, csv_path: Path, chunk_size: int, min_duration_sec: float, correction_mode: str) -> None:
    print(f"=== {algo} beta estimation ===")
    print(f"CSV:             {csv_path}")
    print(f"Chunk size:      {chunk_size}")
    print(f"T_min clamp:     {min_duration_sec}")
    print(f"Correction mode: {correction_mode}")
    print()


def main() -> int:
    default_csv = _default_csv_path()
    parser = _build_parser(default_csv)
    args = parser.parse_args()

    # Resolve and validate the trace path.
    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    # Load the trace.
    phases = pd.read_csv(csv_path)

    # Dispatch by algorithm.
    if args.algo == "baseline":
        if args.correction_mode == "beta_informed":
            raise ValueError(
                "correction-mode=beta_informed is only valid for --algo iterative. "
                "Use end|uniform|none for baseline."
            )

        _print_common_header(
            algo="Baseline",
            csv_path=csv_path,
            chunk_size=args.chunk_size,
            min_duration_sec=args.min_duration_sec,
            correction_mode=args.correction_mode,
        )

        res = estimate_betas_baseline(
            phases,
            chunk_size=args.chunk_size,
            min_duration_sec=args.min_duration_sec,
            correction_mode=args.correction_mode,  # type: ignore[arg-type]
        )

        print("Estimated betas:")
        print(f"  beta0 (sec/step):  {res.beta0:.6f}")
        print(f"  beta1 (sec/token): {res.beta1:.6f}")
        print(f"  beta2 (sec/token): {res.beta2:.6f}")
        print()
        print("Diagnostics:")
        for k, v in res.diagnostics.items():
            print(f"  {k}: {v}")

        return 0

    # iterative
    _print_common_header(
        algo="Iterative",
        csv_path=csv_path,
        chunk_size=args.chunk_size,
        min_duration_sec=args.min_duration_sec,
        correction_mode=args.correction_mode,
    )
    print("Iterative controls:")
    print(f"  max_outer_iters:             {args.max_outer_iters}")
    print(f"  tol:                         {args.tol}")
    print(f"  damping_eta:                 {args.damping_eta}")
    print(f"  init_via_baseline_correction:{args.init_via_baseline_correction}")
    print()

    res = estimate_betas_iterative(
        phases,
        chunk_size=args.chunk_size,
        min_duration_sec=args.min_duration_sec,
        prefill_correction_mode=args.correction_mode,  # type: ignore[arg-type]
        max_outer_iters=args.max_outer_iters,
        tol=args.tol,
        damping_eta=args.damping_eta,
        init_via_baseline_correction=args.init_via_baseline_correction,
    )

    print("Estimated betas:")
    print(f"  beta0 (sec/step):  {res.beta0:.6f}")
    print(f"  beta1 (sec/token): {res.beta1:.6f}")
    print(f"  beta2 (sec/token): {res.beta2:.6f}")
    print()
    print(f"Converged: {res.converged}  (outer iters: {res.n_outer_iters})")
    print()

    print("Diagnostics:")
    for k, v in res.diagnostics.items():
        print(f"  {k}: {v}")

    # Print a small tail of beta history (useful but not noisy).
    beta_hist = res.history.get("beta", None)
    if beta_hist is not None and len(beta_hist) > 0:
        tail = beta_hist[-min(5, len(beta_hist)) :]
        print("\nBeta history (last few iters):")
        start_idx = len(beta_hist) - len(tail)
        for j, b in enumerate(tail, start=start_idx):
            print(f"  iter {j:>3d}: beta0={b[0]:.6f}, beta1={b[1]:.6f}, beta2={b[2]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
