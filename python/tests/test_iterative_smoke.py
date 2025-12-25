# tests/test_iterative_smoke.py
"""
Smoke + invariants test for the iterative estimator.

This mirrors the spirit of the baseline end-to-end tests:
- Ensure the iterative estimator runs on a tiny synthetic trace.
- Ensure outputs are finite, non-negative (NNLS), and structurally consistent.
- Ensure diagnostics + history contain the expected keys and lengths.

This test does NOT validate statistical correctness of recovered betas.
"""

from __future__ import annotations

import math

import pandas as pd

from estimators.iterative import estimate_betas_iterative


def _tiny_phases_df() -> pd.DataFrame:
    """
    Minimal phase-level trace with overlap.

    Two requests (A, B), each with:
      - one prefill phase
      - one decode phase

    Overlap ensures pressures are non-trivial.
    """
    return pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 96, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 6.0, "prefill_tokens": 0,  "decode_tokens": 4},
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 5.0, "prefill_tokens": 0,  "decode_tokens": 2},
        ]
    )


def _isfinite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_iterative_smoke_and_invariants() -> None:
    phases = _tiny_phases_df()

    res = estimate_betas_iterative(
        phases,
        chunk_size=64,
        prefill_correction_mode="beta_informed",
        max_outer_iters=25,
        tol=1e-9,
        damping_eta=1.0,
        init_via_baseline_correction="end",
    )

    # ------------------------------------------------------------------
    # Basic result sanity
    # ------------------------------------------------------------------
    assert _isfinite(res.beta0)
    assert _isfinite(res.beta1)
    assert _isfinite(res.beta2)

    # NNLS => non-negative coefficients (allow tiny numerical noise)
    assert res.beta0 >= -1e-12
    assert res.beta1 >= -1e-12
    assert res.beta2 >= -1e-12

    assert isinstance(res.converged, bool)
    assert isinstance(res.n_outer_iters, int)
    assert res.n_outer_iters >= 1

    # ------------------------------------------------------------------
    # Diagnostics invariants
    # ------------------------------------------------------------------
    diag = res.diagnostics
    assert isinstance(diag, dict)

    # Always expected scalar controls/summary
    for k in [
        "converged",
        "n_outer_iters",
        "tol",
        "damping_eta",
        "init_beta0",
        "init_beta1",
        "init_beta2",
        "final_beta0",
        "final_beta1",
        "final_beta2",
        "mean_steps",
        "mean_duration_seconds",
    ]:
        assert k in diag, f"Missing diagnostics key: {k}"

    # Final-iteration metrics should exist once we ran >= 1 outer iteration
    for k in ["final_r2", "final_rmse_seconds", "final_delta_beta_l2"]:
        assert k in diag, f"Missing diagnostics key: {k}"
        assert _isfinite(float(diag[k]))

    # r2 might be nan in degenerate cases, but should be finite for this trace
    assert _isfinite(float(diag["final_r2"]))

    # ------------------------------------------------------------------
    # History invariants
    # ------------------------------------------------------------------
    hist = res.history
    assert isinstance(hist, dict)

    for k in ["beta", "delta_beta_l2", "r2", "rmse_seconds", "notes"]:
        assert k in hist, f"Missing history key: {k}"

    beta_hist = hist["beta"]
    delta_hist = hist["delta_beta_l2"]
    r2_hist = hist["r2"]
    rmse_hist = hist["rmse_seconds"]
    notes = hist["notes"]

    # Strong typing expectations (we keep these lists in iterative.py)
    assert isinstance(beta_hist, list)
    assert isinstance(delta_hist, list)
    assert isinstance(r2_hist, list)
    assert isinstance(rmse_hist, list)
    assert isinstance(notes, list)

    # beta history includes the initialization beta^(0), then one per outer iter
    assert len(beta_hist) == res.n_outer_iters + 1
    assert len(delta_hist) == res.n_outer_iters
    assert len(r2_hist) == res.n_outer_iters
    assert len(rmse_hist) == res.n_outer_iters
    assert len(notes) == res.n_outer_iters

    # Check beta tuples have the right shape and are finite
    for b in beta_hist:
        assert isinstance(b, tuple)
        assert len(b) == 3
        assert _isfinite(b[0]) and _isfinite(b[1]) and _isfinite(b[2])

    # Check per-iter metrics are finite
    for d in delta_hist:
        assert _isfinite(float(d))
        assert float(d) >= 0.0

    for r2 in r2_hist:
        assert _isfinite(float(r2))

    for rmse in rmse_hist:
        assert _isfinite(float(rmse))
        assert float(rmse) >= 0.0
