# pyright: reportOperatorIssue=false

"""
tests/test_timeintegral_validation.py

Guardrail tests for schema + validation behavior of baseline/timeintegral.py.

These tests are intentionally small and fast. They protect the estimator's
public contract: what input schemas are accepted, and what failures are raised
for invalid traces.

We mostly validate that the code fails loudly and predictably (ValueError)
when the trace violates baseline assumptions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from baseline.timeintegral import estimate_betas_baseline


def _minimal_valid_trace() -> pd.DataFrame:
    """
    Minimal valid trace with one request and two phases.
    (Keeps tests simple and deterministic.)
    """
    return pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prompt_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prompt_tokens": 0,  "decode_tokens": 1},
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
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prompt_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "prefill", "t_start": 1.0, "t_end": 2.0, "prompt_tokens": 64, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 3.0, "prompt_tokens": 0,  "decode_tokens": 1},
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


def test_accepts_authoritative_n_steps_col_without_token_columns() -> None:
    """
    If N_steps is provided, the estimator should not require prompt_tokens/decode_tokens.
    (It still needs prompt_tokens for uniform correction, but ONLY for prefill rows.)

    Here we provide N_steps and prompt_tokens, but omit decode_tokens.
    """
    df = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 1.0, "prompt_tokens": 64, "N_steps": 1},
            {"request_id": "A", "phase_type": "decode",  "t_start": 1.0, "t_end": 2.0, "prompt_tokens": 0,  "N_steps": 1},
        ]
    )

    # Should run end-to-end.
    res = estimate_betas_baseline(df, chunk_size=64)
    assert res.diagnostics["n_phases"] == float(len(df))
