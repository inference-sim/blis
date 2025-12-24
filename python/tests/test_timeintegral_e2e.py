"""
tests/test_timeintegral_e2e.py

End-to-end + invariants + contract tests for the *public* baseline API:

    estimate_betas_baseline(phases: pd.DataFrame, chunk_size: int, ...)

Philosophy
----------
These tests aim to "lock in" the estimator's *contract* without being brittle.

What we assert (stable contracts)
---------------------------------
1) The estimator runs end-to-end on representative tiny traces (smoke tests).
2) Output type and core invariants:
   - returns BaselineBetaResult
   - betas are finite and non-negative (NNLS constraint)
   - diagnostics dict has expected keys and finite values
     (except r2 may be NaN in degenerate cases; we avoid hard requirements on r2)
3) Behavior on supported modes / knobs:
   - correction_mode ∈ {"end","uniform","none"} executes and returns sane outputs
   - min_duration_sec clamp code path executes
4) Validation / error handling for invalid inputs:
   - missing required columns
   - invalid phase_type values
   - non-positive durations (t_end <= t_start)
   - empty input
   - invalid chunk_size
   - multiple prefill rows per request (baseline schema assumption)
   - invalid provided N_steps (negative or non-integer-ish)
5) Schema nuance (important, easy to misunderstand):
   - If *prefill rows* are present and correction_mode != "none",
     the estimator requires prefill_tokens to compute partial-chunk missing mass μ_r.
   - If traces provide only N_steps but omit prefill_tokens, callers must either:
        (i) set correction_mode="none", or
       (ii) include prefill_tokens for prefill rows.

What we intentionally *do not* assert (brittle / version-dependent)
------------------------------------------------------------------
- Exact beta values. NNLS results can vary slightly across sklearn / BLAS.
- Exact R^2 (same reason; also can be sensitive on small traces).
- Exact intermediate exposures A_pf_i/A_dec_i since the public API does not
  expose them. (If you later add a debug hook, we can add stronger math checks.)

Assumptions
-----------
- The estimator is implemented in baseline.timeintegral.
- The public return type is BaselineBetaResult.

Run
---
pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baseline.timeintegral import BaselineBetaResult, estimate_betas_baseline


# ---------------------------------------------------------------------------
# Small shared helpers (keep tests readable and consistent)
# ---------------------------------------------------------------------------

_DIAG_KEYS: tuple[str, ...] = (
    "n_phases",
    "r2",
    "rmse_seconds",
    "mean_duration_seconds",
    "mean_steps",
    "mean_A_pf_tokens",
    "mean_A_dec_tokens",
)


def _assert_result_sane(res: BaselineBetaResult, *, expect_n_phases: int | None = None) -> None:
    """Assert stable, contract-level invariants for BaselineBetaResult."""
    assert isinstance(res, BaselineBetaResult)

    # NNLS constraint: coefficients must be non-negative.
    assert res.beta0 >= 0.0
    assert res.beta1 >= 0.0
    assert res.beta2 >= 0.0

    # Finiteness: no NaNs/infs in coefficients.
    assert np.isfinite(res.beta0)
    assert np.isfinite(res.beta1)
    assert np.isfinite(res.beta2)

    # Diagnostics keys exist.
    diag = res.diagnostics
    for k in _DIAG_KEYS:
        assert k in diag, f"Missing diagnostics key: {k}"

    # Diagnostics: these should be finite in any reasonable non-degenerate run.
    # r2 can be NaN in degenerate cases, so we do not require it to be finite here.
    assert np.isfinite(diag["rmse_seconds"])
    assert np.isfinite(diag["mean_duration_seconds"])
    assert np.isfinite(diag["mean_steps"])
    assert np.isfinite(diag["mean_A_pf_tokens"])
    assert np.isfinite(diag["mean_A_dec_tokens"])

    # n_phases matches the input row count when provided.
    if expect_n_phases is not None:
        assert diag["n_phases"] == float(expect_n_phases)


def _two_request_overlap_trace(*, prefill_tokens_a: int = 96, prefill_tokens_b: int = 96) -> pd.DataFrame:
    """
    Construct a tiny trace with overlap in both prefill and decode.

    Designed to exercise:
      - time grid construction
      - sweep-line overlap counts
      - pressure integration
      - partial-chunk correction (mu > 0 if prefill_tokens not multiple of chunk)
      - NNLS fit

    Note: By default prefill_tokens=96 with chunk=64 gives:
      N=ceil(96/64)=2, rho=96-64*(2-1)=32, mu=32 (>0), so correction is non-trivial.
    """
    return pd.DataFrame(
        [
            # Request A
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": prefill_tokens_a, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 6.0, "prefill_tokens": 0,               "decode_tokens": 4},
            # Request B (prefill overlaps A; decode overlaps A)
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": prefill_tokens_b, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 5.0, "prefill_tokens": 0,               "decode_tokens": 2},
        ]
    )


# ---------------------------------------------------------------------------
# Core smoke tests
# ---------------------------------------------------------------------------

def test_estimate_betas_baseline_smoke_two_requests_default_mode() -> None:
    """Baseline smoke: default correction_mode ('end') runs and returns sane outputs."""
    phases = _two_request_overlap_trace()
    res = estimate_betas_baseline(phases, chunk_size=64)
    _assert_result_sane(res, expect_n_phases=len(phases))


@pytest.mark.parametrize("mode", ["end", "uniform", "none"])
def test_estimate_betas_baseline_smoke_all_correction_modes(mode: str) -> None:
    """
    The public API supports three correction modes.

    We assert:
      - each mode runs end-to-end
      - returns finite, non-negative coefficients
      - includes diagnostics
    """
    phases = _two_request_overlap_trace()
    res = estimate_betas_baseline(phases, chunk_size=64, correction_mode=mode)  # type: ignore[arg-type]
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_estimate_betas_baseline_respects_min_duration_clamp() -> None:
    """
    A tiny-duration phase can produce huge lambda_hat (N/T). The clamp is a robustness knob.
    This test ensures the code path executes and returns sane outputs.

    We do not assert exact values.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0,   "t_end": 0.001, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 0.001, "t_end": 1.001, "prefill_tokens": 0,  "decode_tokens": 1},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64, min_duration_sec=0.1)
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_end_correction_handles_prefill_ending_at_global_max_time() -> None:
    """
    Regression guard: end-localized correction explicitly clamps when t_end == grid[-1].

    Construct a trace where a prefill ends at the global maximum timestamp among all phases.
    We assert that the estimator does not crash and returns sane outputs.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0,  "t_end": 10.0, "prefill_tokens": 96, "decode_tokens": 0},  # ends at max
            {"request_id": "A", "phase_type": "decode",  "t_start": 10.0, "t_end": 12.0, "prefill_tokens": 0,  "decode_tokens": 2},
            {"request_id": "B", "phase_type": "decode",  "t_start": 1.0,  "t_end": 9.0,  "prefill_tokens": 0,  "decode_tokens": 8},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64, correction_mode="end")
    _assert_result_sane(res, expect_n_phases=len(phases))


# ---------------------------------------------------------------------------
# Provided N_steps path (schema variations)
# ---------------------------------------------------------------------------

def test_uses_provided_n_steps_without_token_columns_decode_only() -> None:
    """
    Contract: If n_steps_col is present and the trace contains ONLY decode rows,
    the estimator should not require token columns.

    Why decode-only works without token columns:
      - N_i is read from N_steps for all phases
      - no prefill rows => no partial-chunk correction is computed
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "decode", "t_start": 0.0, "t_end": 2.0, "N_steps": 2},
            {"request_id": "B", "phase_type": "decode", "t_start": 0.5, "t_end": 3.5, "N_steps": 3},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64, n_steps_col="N_steps")
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_prefill_present_requires_prefill_tokens_unless_correction_disabled() -> None:
    """
    Contract nuance (intentional and easy to miss):

    If prefill rows are present and correction_mode != "none", the estimator needs
    prefill_tokens to compute partial-chunk missing mass μ_r for the correction.
    Therefore, providing only N_steps is insufficient unless correction is disabled.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "N_steps": 2},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 4.0, "N_steps": 2},
        ]
    )

    # Default correction_mode="end" should raise due to missing prefill_tokens.
    with pytest.raises(ValueError, match=r"Missing required columns: \['prefill_tokens'\]"):
        estimate_betas_baseline(phases, chunk_size=64, n_steps_col="N_steps")

    # Disabling correction should allow operation with N_steps-only schema.
    res = estimate_betas_baseline(phases, chunk_size=64, n_steps_col="N_steps", correction_mode="none")
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_uses_provided_n_steps_with_prefill_tokens_present() -> None:
    """
    If N_steps is provided AND prefill_tokens is present for prefill rows,
    the estimator should run under the default correction mode.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "N_steps": 2, "prefill_tokens": 96},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 4.0, "N_steps": 2, "prefill_tokens": 0, "decode_tokens": 2},
            {"request_id": "B", "phase_type": "decode",  "t_start": 1.0, "t_end": 3.0, "N_steps": 2, "prefill_tokens": 0, "decode_tokens": 2},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64, n_steps_col="N_steps")
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_non_integerish_n_steps_raises() -> None:
    """N_steps must be integer-ish (paper assumption N_i ∈ ℕ)."""
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "decode", "t_start": 0.0, "t_end": 1.0, "N_steps": 1.2},
        ]
    )
    with pytest.raises(ValueError, match=r"integer-valued"):
        estimate_betas_baseline(phases, chunk_size=64, n_steps_col="N_steps")


def test_negative_n_steps_raises() -> None:
    """Negative step counts are invalid and must raise."""
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "decode", "t_start": 0.0, "t_end": 1.0, "N_steps": -1},
        ]
    )
    with pytest.raises(ValueError):
        estimate_betas_baseline(phases, chunk_size=64, n_steps_col="N_steps")


# ---------------------------------------------------------------------------
# Baseline schema assumption: at most one prefill row per request
# ---------------------------------------------------------------------------

def test_multiple_prefill_rows_per_request_raises() -> None:
    """
    The baseline implementation assumes a single prefill row per request.
    If prefill is segmented into multiple rows for a request, the code should raise.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "prefill", "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 3.0, "prefill_tokens": 0,  "decode_tokens": 1},
        ]
    )
    with pytest.raises(ValueError, match=r"Multiple prefill rows per request"):
        estimate_betas_baseline(phases, chunk_size=64)


# ---------------------------------------------------------------------------
# Valid-but-varied traces (guard against hidden coupling assumptions)
# ---------------------------------------------------------------------------

def test_only_decode_phases_runs() -> None:
    """No prefill rows should still work (prefill pressures all zero; no correction)."""
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "decode", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 0, "decode_tokens": 2},
            {"request_id": "B", "phase_type": "decode", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 0, "decode_tokens": 2},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64)
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_only_prefill_phases_runs() -> None:
    """No decode rows should still work (decode pressures all zero)."""
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 96, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 96, "decode_tokens": 0},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64)
    _assert_result_sane(res, expect_n_phases=len(phases))


def test_no_overlap_runs() -> None:
    """A trace with no overlap between phases should still construct a valid grid and run."""
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 0,  "decode_tokens": 1},
            {"request_id": "B", "phase_type": "prefill", "t_start": 3.0, "t_end": 4.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 4.0, "t_end": 5.0, "prefill_tokens": 0,  "decode_tokens": 1},
        ]
    )
    res = estimate_betas_baseline(phases, chunk_size=64)
    _assert_result_sane(res, expect_n_phases=len(phases))


# ---------------------------------------------------------------------------
# Validation / error handling (must raise)
# ---------------------------------------------------------------------------

def test_empty_dataframe_raises() -> None:
    """Empty input must raise (public API contract)."""
    phases = pd.DataFrame(columns=["request_id", "phase_type", "t_start", "t_end", "prefill_tokens", "decode_tokens"])
    with pytest.raises(ValueError, match=r"No phase rows"):
        estimate_betas_baseline(phases, chunk_size=64)


def test_invalid_chunk_size_raises() -> None:
    """chunk_size must be positive."""
    phases = _two_request_overlap_trace()
    with pytest.raises(ValueError, match=r"chunk_size must be positive"):
        estimate_betas_baseline(phases, chunk_size=0)


def test_missing_required_columns_raises() -> None:
    """Missing core schema columns must raise."""
    phases = pd.DataFrame([{"request_id": "A", "phase_type": "decode", "t_start": 0.0}])  # missing t_end
    with pytest.raises(ValueError, match=r"Missing required columns"):
        estimate_betas_baseline(phases, chunk_size=64)


def test_invalid_phase_type_raises() -> None:
    """phase_type must be 'prefill' or 'decode' (case-insensitive)."""
    phases = pd.DataFrame(
        [{"request_id": "A", "phase_type": "oops", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 0, "decode_tokens": 1}]
    )
    with pytest.raises(ValueError, match=r"phase_type must be one of"):
        estimate_betas_baseline(phases, chunk_size=64)


def test_non_positive_duration_raises() -> None:
    """Any row with t_end <= t_start must raise."""
    phases = pd.DataFrame(
        [{"request_id": "A", "phase_type": "decode", "t_start": 1.0, "t_end": 1.0, "prefill_tokens": 0, "decode_tokens": 1}]
    )
    with pytest.raises(ValueError, match=r"non-positive durations"):
        estimate_betas_baseline(phases, chunk_size=64)
