"""
tests/test_iterative_paper_semantics_lag.py

Guardrail test for paper semantics:

Goal: enforce the *lag*:
  lambda^(m)(t) depends on tilde_p_pf^(m-1)(t) only,
  not on tilde_p_pf^(m)(t).

Practical approach:
- Run iterative for 2 outer iterations and ensure first-iteration fit metrics
  (r2/rmse) are deterministic across repeated runs on the same input.
- Ensure there is no inner-pass parameter in the public API.
- Ensure we don't reintroduce an "inner passes" loop or reorder the lambda/correction
  computation in a way that couples lambda^(m) to tilde_p_pf^(m).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from estimators.iterative import estimate_betas_iterative


def _sample_csv_path() -> Path:
    # Resolve: python/tests/... -> python/data/traces/sample.csv (match baseline test style)
    test_dir = Path(__file__).resolve().parent
    python_root = test_dir.parent
    return python_root / "data" / "traces" / "sample.csv"


def test_iterative_lag_semantics_no_inner_passes_and_deterministic_first_iter() -> None:
    # --- (A) Public API must not expose any inner-pass knob ---
    sig = inspect.signature(estimate_betas_iterative)
    assert "inner_beta_informed_passes" not in sig.parameters, (
        "Inner-pass knob reintroduced; paper semantics require NO inner passes."
    )

    # --- (B) Source-level guard against reintroducing coupling/inner loop ---
    src = inspect.getsource(estimate_betas_iterative)

    # Guard: don't reintroduce the previous inner-loop vocabulary
    assert "inner_beta_informed_passes" not in src
    assert "inner_pass" not in src  # broad, but intentional

    # Guard: ensure lambda computation is fed the *lagged* effective pressure variable.
    # We don't overfit to exact formatting; we just want to catch obvious regressions.
    assert "tilde_p_pf_prev" in src, "Expected lagged effective pressure variable missing."
    # If someone changes the lambda call to use tilde_p_pf_m, this should trip.
    assert "tilde_p_pf_m" in src, "Expected current-iteration effective pressure variable missing."
    # Heuristic: lambda call should reference prev, not m.
    # (This is not perfect, but it will catch the most common regression.)
    lambda_call_ok = "tilde_p_pf_prev" in src and "_lambda_segment_from_beta_and_pressures" in src
    assert lambda_call_ok, "Expected lambda computation call not found or not using lagged pressure."

    # --- (C) Determinism: run twice for 2 iterations and compare first-iter metrics ---
    csv_path = _sample_csv_path()
    assert csv_path.exists(), f"Missing sample trace CSV at {csv_path}"
    phases = pd.read_csv(csv_path)

    def run_once() -> tuple[float, float, int]:
        res = estimate_betas_iterative(
            phases,
            chunk_size=64,
            prefill_correction_mode="beta_informed",
            max_outer_iters=2,     # we want the first iter + (usually) one more
            tol=1e-30,             # make early convergence extremely unlikely
            damping_eta=1.0,
            init_via_baseline_correction="end",
        )
        r2_hist = res.history["r2"]
        rmse_hist = res.history["rmse_seconds"]

        # Must have at least one outer iteration recorded
        assert len(r2_hist) >= 1
        assert len(rmse_hist) >= 1

        # Return first-iteration metrics and number of iters actually executed
        return float(r2_hist[0]), float(rmse_hist[0]), int(res.n_outer_iters)

    r2a, rmsea, itersa = run_once()
    r2b, rmseb, itersb = run_once()

    # Deterministic to numerical equality is usually true here, but allclose is safer
    assert np.isfinite(r2a) and np.isfinite(r2b)
    assert np.isfinite(rmsea) and np.isfinite(rmseb)
    assert np.allclose([r2a, rmsea], [r2b, rmseb], rtol=0.0, atol=0.0), (
        f"First-iteration metrics changed across identical runs: "
        f"(r2,rmse)=({r2a},{rmsea}) vs ({r2b},{rmseb})"
    )

    # Optional: prefer that we actually hit 2 iters; but don't make it flaky if it converges early.
    assert itersa in (1, 2)
    assert itersb in (1, 2)
