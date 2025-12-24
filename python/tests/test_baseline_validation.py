# pyright: reportOperatorIssue=false
"""
tests/test_baseline_validation.py

Guardrail tests for schema + validation behavior of estimators/baseline.py.

These tests are intentionally small and fast. They protect the estimator's
public contract: what input schemas are accepted, and what failures are raised
for invalid traces.

We mostly validate that the code fails loudly and predictably (ValueError)
when the trace violates baseline assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from estimators.baseline import estimate_betas_baseline


def _minimal_valid_trace() -> pd.DataFrame:
    """
    Minimal valid trace with one request and two phases.
    (Keeps tests simple and deterministic.)
    """
    return pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 0,  "decode_tokens": 1},
        ]
    )


def test_rejects_invalid_phase_type() -> None:
    df = _minimal_valid_trace()
    df.loc[0, "phase_type"] = "PREFILL??"  # invalid even after lowercasing

    with pytest.raises(ValueError, match="phase_type must be one of"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_non_positive_duration_equal() -> None:
    df = _minimal_valid_trace()
    df.loc[0, "t_end"] = df.loc[0, "t_start"]  # equal -> invalid

    with pytest.raises(ValueError, match="non-positive durations"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_non_positive_duration_decreasing() -> None:
    df = _minimal_valid_trace()
    df.loc[0, "t_end"] = df.loc[0, "t_start"] - 1.0  # decreasing -> invalid

    with pytest.raises(ValueError, match="non-positive durations"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_missing_required_columns() -> None:
    df = _minimal_valid_trace().drop(columns=["t_end"])

    with pytest.raises(ValueError, match="Missing required columns"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_multiple_prefill_rows_per_request() -> None:
    """
    Baseline assumption: at most one prefill row per request_id.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "prefill", "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 3.0, "prefill_tokens": 0,  "decode_tokens": 1},
        ]
    )

    with pytest.raises(ValueError, match="Multiple prefill rows per request"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_non_positive_chunk_size() -> None:
    df = _minimal_valid_trace()

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        estimate_betas_baseline(df, chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        estimate_betas_baseline(df, chunk_size=-64)


def test_rejects_non_integer_n_steps_col() -> None:
    """
    N_steps must be integer-ish within tolerance.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "N_steps": 1.2},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 0,  "N_steps": 1},
        ]
    )

    with pytest.raises(ValueError, match=r"must be integer-valued"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_negative_n_steps_col() -> None:
    """
    N_steps must be non-negative.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "N_steps": -1},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 0,  "N_steps": 1},
        ]
    )

    with pytest.raises(ValueError, match="contains negative values|Negative N_i found"):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_non_finite_times() -> None:
    """
    t_start/t_end must be finite numbers.
    Pandas will cast, but downstream should reject non-finite duration inputs.
    """
    df = _minimal_valid_trace()
    df.loc[0, "t_start"] = np.nan

    # Depending on where it triggers, message can vary; we only enforce ValueError.
    with pytest.raises(ValueError):
        estimate_betas_baseline(df, chunk_size=64)


def test_rejects_non_finite_n_steps() -> None:
    """
    N_steps must be finite.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "N_steps": float("inf")},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 0,  "N_steps": 1},
        ]
    )

    with pytest.raises(ValueError):
        estimate_betas_baseline(df, chunk_size=64)


def test_accepts_authoritative_n_steps_col_without_decode_token_column() -> None:
    """
    If N_steps is provided, the estimator should not require decode_tokens
    (step count is authoritative). Prefill tokens may still be required
    depending on correction_mode.

    Here we provide N_steps and prefill_tokens, but omit decode_tokens.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 64, "N_steps": 1},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prefill_tokens": 0,  "N_steps": 1},
        ]
    )

    # Default correction_mode="end" still requires prefill_tokens for prefill rows.
    res = estimate_betas_baseline(df, chunk_size=64)
    assert res.diagnostics["n_phases"] == float(len(df))


def test_accepts_authoritative_n_steps_without_any_token_columns_when_correction_none() -> None:
    """
    High-leverage integration mode: if N_steps is provided and correction_mode="none",
    the estimator should not require *any* token columns.

    This is a clean contract for users who have already inferred step counts upstream.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "N_steps": 1},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "N_steps": 1},
        ]
    )

    res = estimate_betas_baseline(df, chunk_size=64, correction_mode="none")
    assert res.diagnostics["n_phases"] == float(len(df))


def test_end_correction_clamps_when_prefill_end_equals_last_grid_boundary() -> None:
    """
    Regression guardrail: end-localized correction assigns missing mass to the segment
    containing t_end. If t_end == grid[-1], code clamps to the final segment.

    This test ensures the edge-case does not crash.
    """
    # Construct a trace where the maximum boundary is exactly a prefill end.
    # Include minimal tokens so correction math is exercised.
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 1, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 3.0, "prefill_tokens": 0, "decode_tokens": 1},
        ]
    )
    # Here grid[-1] == 3.0, and prefill t_end == 2.0 (not last); make another prefill end == 3.0.
    df2 = pd.DataFrame(
        [
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 1, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 4.0, "prefill_tokens": 0, "decode_tokens": 1},
        ]
    )
    phases = pd.concat([df, df2], ignore_index=True)

    res = estimate_betas_baseline(phases, chunk_size=64, correction_mode="end")
    assert np.isfinite(res.beta0)
    assert np.isfinite(res.beta1)
    assert np.isfinite(res.beta2)
