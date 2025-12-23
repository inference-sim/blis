# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false

"""
tests/test_timeintegral_correction_exposures.py

Core semantics tests for the baseline estimator's exposure construction:

  A_pf_i  = lambda_hat_i * ∫ p_pf_full(t) dt  -  ∫ corr_rate(t) dt
  A_dec_i = lambda_hat_i * ∫ p_dec(t) dt

These tests intentionally avoid running the full regression; they target the
unit-sensitive "uniform partial-chunk correction" logic and its interaction
with the integrated-pressure exposure formula.

Why these tests matter
----------------------
The most error-prone part of the baseline is mixing signals with different units:

- p_pf_full(t) is in tokens/step
- corr_rate(t) is in tokens/second
- lambda_hat_i is in steps/second

So:
- lambda_hat_i * ∫ p_pf_full(t) dt  has units tokens
- ∫ corr_rate(t) dt already has units tokens and MUST be subtracted directly
  (it must NOT be multiplied by lambda_hat_i)

These tests lock that down with hand-checkable constructions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baseline.timeintegral import (
    _build_global_time_grid_from_df,
    _build_uniform_partial_chunk_correction_rate,
    _check_baseline_prefill_uniqueness,
    _compute_durations_and_lambda_hat,
    _compute_exposures,
    _infer_or_read_step_counts,
    _reconstruct_pressures,
    _validate_and_normalize_schema,
)


def _prepare_single_prefill_df(*, prefill_tokens: int, t_start: float, t_end: float) -> pd.DataFrame:
    """Minimal 1-row trace with a single prefill phase."""
    return pd.DataFrame(
        [
            {
                "request_id": "R1",
                "phase_type": "prefill",
                "t_start": float(t_start),
                "t_end": float(t_end),
                "prefill_tokens": int(prefill_tokens),
                # decode_tokens is unused for prefill, but required by step inference
                "decode_tokens": 0,
            }
        ]
    )


def _pipeline_until_exposures(phases: pd.DataFrame, *, chunk_size: int) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Run the internal pipeline pieces up through exposure computation.

    Returns
    -------
    df : pd.DataFrame
        Contains N_i, T_i, lambda_hat, A_pf_i, A_dec_i.
    grid : np.ndarray
    p_pf_full : np.ndarray
    I_pf_full : np.ndarray
    corr_rate : np.ndarray
    I_corr : np.ndarray
    """
    df = _validate_and_normalize_schema(
        phases,
        request_id_col="request_id",
        phase_type_col="phase_type",
        t_start_col="t_start",
        t_end_col="t_end",
    )

    df = _infer_or_read_step_counts(
        df,
        chunk_size=chunk_size,
        phase_type_col="phase_type",
        prefill_tokens_col="prefill_tokens",
        decode_tokens_col="decode_tokens",
        n_steps_col="N_steps",
    )

    df = _compute_durations_and_lambda_hat(
        df,
        t_start_col="t_start",
        t_end_col="t_end",
        min_duration_sec=0.0,
    )

    prefill_df = _check_baseline_prefill_uniqueness(
        df, request_id_col="request_id", phase_type_col="phase_type"
    )
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col="t_start", t_end_col="t_end")

    p_pf_full, p_dec, I_pf_full, I_dec = _reconstruct_pressures(
        df,
        grid=grid,
        starts=starts,
        ends=ends,
        chunk_size=chunk_size,
        phase_type_col="phase_type",
    )

    corr_rate, I_corr = _build_uniform_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=chunk_size,
        prefill_df=prefill_df,
        t_start_col="t_start",
        t_end_col="t_end",
        prefill_tokens_col="prefill_tokens",
        n_steps_col="N_steps",
    )

    lam = df["lambda_hat"].to_numpy(dtype=np.float64)
    df = _compute_exposures(
        df,
        starts=starts,
        ends=ends,
        grid=grid,
        lam=lam,
        p_pf_full=p_pf_full,
        p_dec=p_dec,
        I_pf_full=I_pf_full,
        I_dec=I_dec,
        corr_rate=corr_rate,
        I_corr=I_corr,
    )

    return df, grid, p_pf_full, I_pf_full, corr_rate, I_corr


def _integral_piecewise_constant(grid: np.ndarray, y_segment: np.ndarray, a: float, b: float) -> float:
    """
    Simple reference integrator for piecewise-constant signals over a grid.
    (Slow but transparent; used only for testing.)

    y_segment[j] is constant on [grid[j], grid[j+1]).
    """
    a_f = float(a)
    b_f = float(b)
    if b_f <= a_f:
        return 0.0

    total = 0.0
    for j in range(len(grid) - 1):
        seg_a = float(grid[j])
        seg_b = float(grid[j + 1])
        left = max(a_f, seg_a)
        right = min(b_f, seg_b)
        if right > left:
            total += float(y_segment[j]) * (right - left)
    return float(total)


def test_uniform_partial_chunk_correction_integrates_to_mu_tokens() -> None:
    """
    Choose P such that mu = C - rho > 0. With one prefill request over [0,2):

      - p_pf_full(t) = C * (#active prefill) = C (tokens/step)
      - lambda_hat = N / T where N = ceil(P/C)
      - corr_rate is uniform on [0,2) with total integrated mass mu tokens

    This test asserts:
      ∫_0^2 corr_rate(t) dt == mu
    """
    C = 64
    P = 96  # ceil(96/64)=2; rho=32; mu=32
    mu_expected = 32.0

    phases = _prepare_single_prefill_df(prefill_tokens=P, t_start=0.0, t_end=2.0)
    df, grid, _, _, corr_rate, _ = _pipeline_until_exposures(phases, chunk_size=C)

    corr_tokens = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=2.0)
    assert corr_tokens == pytest.approx(mu_expected)

    # Pylance-friendly cast: pandas scalar -> Python float
    assert float(df.loc[0, "N_i"]) == pytest.approx(2.0)


def test_exposure_formula_subtracts_correction_in_tokens_not_scaled_by_lambda() -> None:
    """
    Assert the baseline exposure formula:

      A_pf = lambda_hat * ∫ p_pf_full(t) dt - mu

    For a single request over [0,2):
      p_pf_full(t) = C tokens/step
      ∫ p_pf_full dt = C * 2
      lambda_hat = N/T = 2/2 = 1 step/sec
      => lambda_hat * ∫ p_pf_full dt = 128 tokens
      mu = 32 tokens
      => A_pf = 96 tokens
    """
    C = 64
    P = 96
    phases = _prepare_single_prefill_df(prefill_tokens=P, t_start=0.0, t_end=2.0)

    df, grid, p_pf_full, _, corr_rate, _ = _pipeline_until_exposures(phases, chunk_size=C)

    # Cast pandas scalars explicitly to satisfy strict type checkers
    N = float(df.loc[0, "N_i"])
    T = float(df.loc[0, "T_i"])
    lam = float(df.loc[0, "lambda_hat"])

    assert lam == pytest.approx(N / T)

    int_pf_full = _integral_piecewise_constant(grid, p_pf_full, a=0.0, b=2.0)  # (tokens/step)*sec
    corr_tokens = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=2.0)  # tokens

    A_pf_expected = lam * int_pf_full - corr_tokens
    assert A_pf_expected == pytest.approx(96.0)

    assert float(df.loc[0, "A_pf_i"]) == pytest.approx(A_pf_expected)
    assert float(df.loc[0, "A_dec_i"]) == pytest.approx(0.0)


def test_uniform_partial_chunk_correction_zero_when_prefill_multiple_of_chunk() -> None:
    """
    If P is exactly a multiple of C, then mu = 0 and correction integrates to 0.
    """
    C = 64
    P = 128  # mu=0

    phases = _prepare_single_prefill_df(prefill_tokens=P, t_start=10.0, t_end=15.0)
    df, grid, _, _, corr_rate, _ = _pipeline_until_exposures(phases, chunk_size=C)

    corr_tokens = _integral_piecewise_constant(grid, corr_rate, a=10.0, b=15.0)
    assert corr_tokens == pytest.approx(0.0)

    # Non-negativity sanity
    assert float(df.loc[0, "A_pf_i"]) >= 0.0
