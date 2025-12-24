# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false

"""
tests/test_timeintegral_correction_exposures.py

Core semantics tests for the baseline estimator's exposure construction:

  A_pf_i  = lambda_hat_i * ∫ p_pf_full(t) dt  -  ∫ corr_rate(t) dt
  A_dec_i = lambda_hat_i * ∫ p_dec(t) dt

These tests intentionally avoid running the full regression; they target the
unit-sensitive correction logic (uniform/end/none) and its interaction
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
    _build_end_partial_chunk_correction_rate,
    _build_partial_chunk_correction_rate
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

def _prepare_two_prefills_df(
    *,
    p1: int,
    s1: float,
    e1: float,
    p2: int,
    s2: float,
    e2: float,
) -> pd.DataFrame:
    """Two prefill rows (distinct request_ids), no decode rows."""
    return pd.DataFrame(
        [
            {
                "request_id": "R1",
                "phase_type": "prefill",
                "t_start": float(s1),
                "t_end": float(e1),
                "prefill_tokens": int(p1),
                "decode_tokens": 0,
            },
            {
                "request_id": "R2",
                "phase_type": "prefill",
                "t_start": float(s2),
                "t_end": float(e2),
                "prefill_tokens": int(p2),
                "decode_tokens": 0,
            },
        ]
    )


def test_uniform_partial_chunk_correction_partial_interval_integrates_fractionally() -> None:
    """
    Uniform mode spreads mu evenly over the prefill interval.
    Therefore integrating over a sub-interval should yield the corresponding fraction.

    Example: one prefill over [0,2), mu=32 => rate=16 tokens/sec.
      ∫_0^1 corr_rate dt = 16 tokens
      ∫_1^2 corr_rate dt = 16 tokens
    """
    C = 64
    P = 96  # N=2, rho=32, mu=32
    phases = _prepare_single_prefill_df(prefill_tokens=P, t_start=0.0, t_end=2.0)

    df, grid, _, _, corr_rate, _ = _pipeline_until_exposures(phases, chunk_size=C)

    corr_0_1 = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=1.0)
    corr_1_2 = _integral_piecewise_constant(grid, corr_rate, a=1.0, b=2.0)

    assert corr_0_1 == pytest.approx(16.0)
    assert corr_1_2 == pytest.approx(16.0)

    # Total still equals mu
    corr_total = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=2.0)
    assert corr_total == pytest.approx(32.0)

    # Exposure still equals P
    assert float(df.loc[0, "A_pf_i"]) == pytest.approx(float(P))


def test_uniform_partial_chunk_correction_adds_overlapping_prefills() -> None:
    """
    With two overlapping prefills, uniform correction rates add linearly.

    Construct:
      R1: [0,2), P=96 => mu1=32 => rate1=16 tokens/sec
      R2: [1,3), P=96 => mu2=32 => rate2=16 tokens/sec

    Then:
      corr_rate = 16 on [0,1)
                 32 on [1,2)   (both active)
                 16 on [2,3)
    And total integrated mass over [0,3) is 64 tokens.
    """
    C = 64
    phases = _prepare_two_prefills_df(
        p1=96, s1=0.0, e1=2.0,
        p2=96, s2=1.0, e2=3.0,
    )

    df, grid, _, _, corr_rate, _ = _pipeline_until_exposures(phases, chunk_size=C)

    # Reference integrals by sub-interval
    c01 = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=1.0)
    c12 = _integral_piecewise_constant(grid, corr_rate, a=1.0, b=2.0)
    c23 = _integral_piecewise_constant(grid, corr_rate, a=2.0, b=3.0)

    assert c01 == pytest.approx(16.0)
    assert c12 == pytest.approx(32.0)
    assert c23 == pytest.approx(16.0)

    c_total = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=3.0)
    assert c_total == pytest.approx(64.0)


    # In an overlap scenario, A_pf_i is NOT equal to "own tokens" P_i.
    # It includes global prefill pressure (C * #active_prefills) integrated over the phase,
    # and subtracts the *global* correction mass over that same interval.
    #
    # For R1 over [0,2):
    #   p_pf_full = 64 on [0,1), 128 on [1,2)  => ∫ p dt = 64*1 + 128*1 = 192
    #   lambda_hat = N/T = 2/2 = 1
    #   corr_rate = 16 on [0,1), 32 on [1,2) => ∫ corr dt = 16*1 + 32*1 = 48
    #   A_pf = 1*192 - 48 = 144
    #
    # Symmetrically, R2 over [1,3) also yields 144.
    assert set(df["phase_type"].unique()) == {"prefill"}
    assert float(df.loc[0, "A_pf_i"]) == pytest.approx(144.0)
    assert float(df.loc[1, "A_pf_i"]) == pytest.approx(144.0)


def test_build_end_partial_chunk_correction_integrates_to_mu_tokens() -> None:
    """
    End-localized mode assigns the entire mu to the single segment containing t_end.
    So the integrated correction over the full prefill interval must still equal mu.

    Use a grid with at least two segments by adding a decode row (doesn't affect correction).
    """
    C = 64
    P = 96  # mu=32

    phases = pd.DataFrame(
        [
            {"request_id": "R1", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": P, "decode_tokens": 0},
            # Add a decode phase to create an extra grid boundary at 3.0 (optional but makes grid less trivial)
            {"request_id": "R1", "phase_type": "decode",  "t_start": 2.0, "t_end": 3.0, "prefill_tokens": 0, "decode_tokens": 1},
        ]
    )

    df = _validate_and_normalize_schema(
        phases,
        request_id_col="request_id",
        phase_type_col="phase_type",
        t_start_col="t_start",
        t_end_col="t_end",
    )
    df = _infer_or_read_step_counts(
        df,
        chunk_size=C,
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
    prefill_df = _check_baseline_prefill_uniqueness(df, request_id_col="request_id", phase_type_col="phase_type")
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col="t_start", t_end_col="t_end")

    corr_rate, _ = _build_end_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=C,
        prefill_df=prefill_df,
        t_start_col="t_start",
        t_end_col="t_end",
        prefill_tokens_col="prefill_tokens",
        n_steps_col="N_steps",
    )

    # Under half-open segments [grid[j], grid[j+1}), if t_end lands exactly on a grid boundary
    # (here t_end = 2.0 and grid contains 2.0), the "containing segment" is the NEXT segment [2,3).
    # So integrating over [0,2) returns 0, while integrating over [2,3) returns mu.
    corr_0_2 = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=2.0)
    corr_2_3 = _integral_piecewise_constant(grid, corr_rate, a=2.0, b=3.0)
    corr_0_3 = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=3.0)

    assert corr_0_2 == pytest.approx(0.0)
    assert corr_2_3 == pytest.approx(32.0)
    assert corr_0_3 == pytest.approx(32.0)


def test_end_partial_chunk_correction_is_zero_before_end_segment() -> None:
    """
    For end-localized correction, corr_rate should be nonzero only on the segment containing t_end.
    Therefore, integrating corr_rate over an interval that ends strictly before t_end should be 0.
    """
    C = 64
    P = 96  # mu=32

    phases = pd.DataFrame(
        [
            {"request_id": "R1", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": P, "decode_tokens": 0},
            # Add a second phase to ensure multiple segments exist even if boundaries coincide oddly elsewhere
            {"request_id": "X",  "phase_type": "prefill", "t_start": 10.0, "t_end": 11.0, "prefill_tokens": 64, "decode_tokens": 0},
        ]
    )

    df = _validate_and_normalize_schema(
        phases,
        request_id_col="request_id",
        phase_type_col="phase_type",
        t_start_col="t_start",
        t_end_col="t_end",
    )
    df = _infer_or_read_step_counts(
        df,
        chunk_size=C,
        phase_type_col="phase_type",
        prefill_tokens_col="prefill_tokens",
        decode_tokens_col="decode_tokens",
        n_steps_col="N_steps",
    )
    df = _compute_durations_and_lambda_hat(df, t_start_col="t_start", t_end_col="t_end", min_duration_sec=0.0)
    prefill_df = _check_baseline_prefill_uniqueness(df, request_id_col="request_id", phase_type_col="phase_type")
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col="t_start", t_end_col="t_end")

    corr_rate, _ = _build_end_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=C,
        prefill_df=prefill_df[prefill_df["request_id"].eq("R1")],  # isolate R1 for clarity
        t_start_col="t_start",
        t_end_col="t_end",
        prefill_tokens_col="prefill_tokens",
        n_steps_col="N_steps",
    )

    # Interval strictly before the end time 2.0
    corr_before_end = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=1.999)
    assert corr_before_end == pytest.approx(0.0)


def test_none_mode_correction_is_identically_zero() -> None:
    """
    In mode='none', corr_rate should be all zeros and integrate to 0 everywhere.
    """
    C = 64
    phases = _prepare_single_prefill_df(prefill_tokens=96, t_start=0.0, t_end=2.0)

    df = _validate_and_normalize_schema(
        phases,
        request_id_col="request_id",
        phase_type_col="phase_type",
        t_start_col="t_start",
        t_end_col="t_end",
    )
    df = _infer_or_read_step_counts(
        df,
        chunk_size=C,
        phase_type_col="phase_type",
        prefill_tokens_col="prefill_tokens",
        decode_tokens_col="decode_tokens",
        n_steps_col="N_steps",
    )
    df = _compute_durations_and_lambda_hat(df, t_start_col="t_start", t_end_col="t_end", min_duration_sec=0.0)
    prefill_df = _check_baseline_prefill_uniqueness(df, request_id_col="request_id", phase_type_col="phase_type")
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col="t_start", t_end_col="t_end")

    corr_rate, _ = _build_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=C,
        prefill_df=prefill_df,
        t_start_col="t_start",
        t_end_col="t_end",
        prefill_tokens_col="prefill_tokens",
        n_steps_col="N_steps",
        mode="none",
    )

    assert np.allclose(corr_rate, 0.0)
    corr_tokens = _integral_piecewise_constant(grid, corr_rate, a=0.0, b=2.0)
    assert corr_tokens == pytest.approx(0.0)


def test_exposure_matches_prefill_tokens_for_single_prefill_phase_under_uniform() -> None:
    """
    For a single prefill-only trace, the baseline construction is designed so that:
      A_pf_i == P_i
    (because lam * ∫ p_pf_full dt = C*N and corr integrates to (C*N - P).)
    This test checks it for a couple of values, including a partial chunk and exact multiple.
    """
    C = 64
    for P in [1, 63, 64, 65, 96, 127, 128]:
        phases = _prepare_single_prefill_df(prefill_tokens=P, t_start=0.0, t_end=2.0)
        df, _, _, _, _, _ = _pipeline_until_exposures(phases, chunk_size=C)
        assert float(df.loc[0, "A_pf_i"]) == pytest.approx(float(P))
        assert float(df.loc[0, "A_dec_i"]) == pytest.approx(0.0)

def test_end_partial_chunk_correction_last_boundary_clamps_to_final_segment() -> None:
    """
    If t_end equals grid[-1], implementation clamps to the final segment S-1.
    This test forces the final segment to be [2,3) (dt=1), so the full mu appears on [2,3).
    """
    C = 64
    P = 96  # mu=32

    phases = pd.DataFrame(
        [
            {"request_id": "R1", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": P, "decode_tokens": 0},
            {"request_id": "Z",  "phase_type": "decode",  "t_start": 0.0, "t_end": 1.0, "prefill_tokens": 0, "decode_tokens": 1},
            # Force a grid boundary at 2.0 so the final segment is [2,3)
            {"request_id": "Y",  "phase_type": "decode",  "t_start": 2.0, "t_end": 2.0001, "prefill_tokens": 0, "decode_tokens": 1},
        ]
    )

    df = _validate_and_normalize_schema(
        phases,
        request_id_col="request_id",
        phase_type_col="phase_type",
        t_start_col="t_start",
        t_end_col="t_end",
    )
    df = _infer_or_read_step_counts(
        df,
        chunk_size=C,
        phase_type_col="phase_type",
        prefill_tokens_col="prefill_tokens",
        decode_tokens_col="decode_tokens",
        n_steps_col="N_steps",
    )
    df = _compute_durations_and_lambda_hat(df, t_start_col="t_start", t_end_col="t_end", min_duration_sec=0.0)
    prefill_df = _check_baseline_prefill_uniqueness(df, request_id_col="request_id", phase_type_col="phase_type")
    _, _, grid = _build_global_time_grid_from_df(df, t_start_col="t_start", t_end_col="t_end")

    assert np.any(np.isclose(grid, 2.0))
    assert np.any(np.isclose(grid, 3.0))
    assert grid[-1] == pytest.approx(3.0)

    corr_rate, _ = _build_end_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=C,
        prefill_df=prefill_df[prefill_df["request_id"].eq("R1")],
        t_start_col="t_start",
        t_end_col="t_end",
        prefill_tokens_col="prefill_tokens",
        n_steps_col="N_steps",
    )

    assert _integral_piecewise_constant(grid, corr_rate, a=2.0, b=3.0) == pytest.approx(32.0)
    assert _integral_piecewise_constant(grid, corr_rate, a=1.0, b=3.0) == pytest.approx(32.0)
