"""
tests/test_timeintegral_e2e.py

Minimal end-to-end tests for the public API:

  estimate_betas_baseline(phases: pd.DataFrame, chunk_size: int, ...)

Goals
-----
- Ensure the estimator runs end-to-end on a tiny trace without raising.
- Validate basic invariants:
    * returns BaselineBetaResult
    * betas are finite and non-negative (NNLS constraint)
    * diagnostics dictionary contains key fields and finite values
- Keep the test stable: do NOT assert exact beta values (those can vary with
  small numerical changes, sklearn versions, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline.timeintegral import BaselineBetaResult, estimate_betas_baseline


def test_estimate_betas_baseline_smoke_two_requests() -> None:
    """
    Two requests (A,B), each with:
      - one prefill phase
      - one decode phase

    The timing is arranged so there is overlap in prefill and decode,
    exercising:
      - time grid construction
      - sweep-line overlap counts
      - pressure integration
      - uniform partial-chunk correction (mu may be > 0 depending on prefill_tokens)
      - NNLS fit
    """
    phases = pd.DataFrame(
        [
            # Request A
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 96, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 6.0, "prefill_tokens": 0,  "decode_tokens": 4},
            # Request B
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 5.0, "prefill_tokens": 0,  "decode_tokens": 2},
        ]
    )

    res = estimate_betas_baseline(phases, chunk_size=64)

    # Type / shape checks
    assert isinstance(res, BaselineBetaResult)

    # NNLS constraint should ensure non-negative coefficients
    assert res.beta0 >= 0.0
    assert res.beta1 >= 0.0
    assert res.beta2 >= 0.0

    # Finiteness checks (no NaNs / infs)
    assert np.isfinite(res.beta0)
    assert np.isfinite(res.beta1)
    assert np.isfinite(res.beta2)

    # Diagnostics should exist and include core keys
    diag = res.diagnostics
    for k in [
        "n_phases",
        "r2",
        "rmse_seconds",
        "mean_duration_seconds",
        "mean_steps",
        "mean_A_pf_tokens",
        "mean_A_dec_tokens",
    ]:
        assert k in diag, f"Missing diagnostics key: {k}"

    # Sanity: n_phases should match input row count
    assert diag["n_phases"] == float(len(phases))

    # These should all be finite (r2 can be nan in degenerate cases, but not here)
    assert np.isfinite(diag["rmse_seconds"])
    assert np.isfinite(diag["mean_duration_seconds"])
    assert np.isfinite(diag["mean_steps"])
    assert np.isfinite(diag["mean_A_pf_tokens"])
    assert np.isfinite(diag["mean_A_dec_tokens"])


def test_estimate_betas_baseline_respects_min_duration_clamp() -> None:
    """
    A tiny-duration phase can produce huge lambda_hat (N/T). The optional clamp
    is a robustness knob; this test ensures the code path executes.

    We don't check exact values—just that it runs and returns finite betas.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0,     "t_end": 0.001, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 0.001,   "t_end": 1.001, "prefill_tokens": 0,  "decode_tokens": 1},
        ]
    )

    # Clamp durations to avoid extreme lambda_hat.
    res = estimate_betas_baseline(phases, chunk_size=64, min_duration_sec=0.1)

    assert res.beta0 >= 0.0 and np.isfinite(res.beta0)
    assert res.beta1 >= 0.0 and np.isfinite(res.beta1)
    assert res.beta2 >= 0.0 and np.isfinite(res.beta2)
