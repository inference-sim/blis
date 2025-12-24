#!/usr/bin/env python3
"""
baseline_step_betas.py

Baseline trace-only estimator for step-level execution coefficients (betas)
for a vLLM-style inference engine.

This file implements the *baseline* estimator described in the paper section
"Baseline: Time-Integrated NNLS Estimation" (Eq. (baseline_nnls)):

High-level pipeline
-------------------
Given a trace of *phase instances* (one row per prefill/decode phase):
  1) Validate/normalize schema and timestamps.
  2) Infer trace-level step counts N_i (if not already provided).
  3) Compute phase-local step density proxy λ̂_i = N_i / T_i (Eq. (baseline_lambda_hat)).
  4) Reconstruct time-varying token pressures from overlap (Eq. (pressures)):
       - p_pf_full(t) = C * (# active prefill phases at t)   [tokens/step]
       - p_dec(t)     =      (# active decode phases at t)   [tokens/step]
  5) Apply partial-chunk correction for prefill chunking (mode: end/uniform/none):
       - subtract missing prefill mass μ_r in integrated form (tokens), not per-step.
  6) Form baseline integrated exposures (Eq. (baseline_integrated_exposures)):
       - A_pf_i  [tokens]
       - A_dec_i [tokens]
  7) Fit NNLS (via sklearn LinearRegression(positive=True), no intercept) (Eq. (baseline_nnls)):
       T_i ≈ β0*N_i + β1*A_pf_i + β2*A_dec_i

IMPORTANT invariants
--------------------
- This is a trace-only estimator: it never uses per-step timing or step boundaries.
- The simulator later consumes only β = (β0, β1, β2); it generates its own step-level
  token counts and KV dynamics. No trace timing is embedded into the simulator.

Notation bridge to paper
------------------------
- Phase i:
    t_{i,s}, t_{i,e}  -> df[t_start_col], df[t_end_col]
    T_i               -> df["T_i"]
    N_i               -> df["N_i"]
    λ̂_i              -> df["lambda_hat"]
    A_pf_i, A_dec_i   -> df["A_pf_i"], df["A_dec_i"]
- Chunk size:
    C                 -> chunk_size
- Pressures:
    p_pf_full(t)      -> p_pf_full (piecewise-constant over time-grid segments)
    p_dec(t)          -> p_dec
- Partial-chunk correction (mode-dependent):
    corr_rate(t)      -> corr_rate  (tokens/second over segments)
    ∫ corr_rate dt    -> corr_tokens (tokens)

Shapes convention
-----------------
Let:
  n = number of phase instances (rows in df)
  M = number of unique time boundaries in the global time grid
  S = M-1 = number of time segments (intervals [grid[j], grid[j+1]))

Then:
  starts, ends          : (n,) float64
  grid                  : (M,) float64
  n_pf, n_dec            : (S,) int64
  p_pf_full, p_dec       : (S,) float64   (tokens/step)
  corr_rate              : (S,) float64   (tokens/second)
  I_pf_full, I_dec       : (M,) float64   (integrals over time; see docs below)
  I_corr                 : (M,) float64
  A_pf, A_dec            : (n,) float64   (tokens)

Dependencies
------------
pip install numpy pandas scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# =============================================================================
# Result structure
# =============================================================================
@dataclass(frozen=True)
class BaselineBetaResult:
    """
    Output of baseline beta estimation.

    Attributes
    ----------
    beta0 : float
        Per-step fixed overhead β0 in seconds/step.
        Interpreted as the additive cost of executing one busy-loop step
        independent of token processing. See Eq. (step_model).

    beta1 : float
        Prefill token cost β1 in seconds/token.
        Multiplies the prefill token exposure A_pf_i (tokens) in Eq. (baseline_predictor).

    beta2 : float
        Decode token cost β2 in seconds/token.
        Multiplies the decode token exposure A_dec_i (tokens) in Eq. (baseline_predictor).

    diagnostics : Dict[str, float]
        Convenience metrics for sanity checks only (R^2, RMSE, feature means, etc.).
        These are not used by the estimator itself.

    Notes on units
    --------------
    The regression model is:
        T_i ≈ β0*N_i + β1*A_pf_i + β2*A_dec_i
    where:
        T_i    : seconds
        N_i    : steps
        A_*_i  : tokens
    so each term has units of seconds.
    """
    beta0: float
    beta1: float
    beta2: float
    diagnostics: Dict[str, float]


# =============================================================================
# Small helpers (kept tiny and well-commented)
# =============================================================================
def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Validate that all required columns exist.

    Inputs
    ------
    df : pd.DataFrame
        Any DataFrame.
    cols : list[str]
        Column names that must exist.

    Outputs
    -------
    None
        Raises ValueError if any required columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


ArrayLike1D = Union[np.ndarray, pd.Series]


def _assert_integerish(x: ArrayLike1D, *, name: str, tol: float = 1e-6) -> None:
    """
    Assert that an array is non-negative and integer-valued up to tolerance.

    This is used to enforce the paper assumption that N_i is an integer
    step count (N_i ∈ ℕ).

    Inputs
    ------
    x : np.ndarray or pd.Series, shape (n,)
        Candidate step counts.
    name : str
        Name used in error messages.
    tol : float
        Allowed deviation from nearest integer.

    Outputs
    -------
    None
        Raises ValueError if negatives exist or values are non-integer within tol.

    Notes
    -----
    Accepts float arrays because upstream inference uses ceil() and casts to float.
    """
    arr = x.to_numpy(dtype=np.float64) if isinstance(x, pd.Series) else x.astype(np.float64, copy=False)

    if np.any(arr < -tol):
        raise ValueError(f"{name} contains negative values.")
    if np.any(np.abs(arr - np.round(arr)) > tol):
        raise ValueError(f"{name} must be integer-valued within tol={tol}.")


def _build_time_grid(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Construct the global time grid used for piecewise-constant overlap signals.

    The estimator assumes that activity counts (how many phases are active)
    only change at phase boundaries. Therefore, pressures are piecewise-constant
    on intervals between unique boundaries.

    Inputs
    ------
    starts : np.ndarray, shape (n,)
        Phase start times (seconds).
    ends : np.ndarray, shape (n,)
        Phase end times (seconds).

    Outputs
    -------
    grid : np.ndarray, shape (M,)
        Sorted unique timestamps from starts and ends.
        Defines segments [grid[j], grid[j+1]) for j=0..M-2.

    Raises
    ------
    ValueError
        If there are fewer than 2 distinct timestamps.
    """
    grid = np.unique(np.concatenate([starts, ends]))
    grid.sort()
    if grid.size < 2:
        raise ValueError("Need at least two distinct timestamps for a time grid.")
    return grid


def _sweepline_counts(grid: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Sweep-line active phase counts per segment of the time grid.

    For a set of intervals [starts[m], ends[m]), this computes how many intervals
    are active on each segment [grid[j], grid[j+1]).

    Inputs
    ------
    grid : np.ndarray, shape (M,)
        Global time grid (sorted unique boundaries).
    starts : np.ndarray, shape (k,)
        Start times for a subset of phases (e.g., prefill-only).
    ends : np.ndarray, shape (k,)
        End times for that same subset.

    Outputs
    -------
    counts : np.ndarray, shape (S=M-1,)
        counts[j] = number of intervals active on [grid[j], grid[j+1]).

    Notes
    -----
    Implementation uses a standard difference-array / prefix-sum trick:
      delta[start_idx] += 1
      delta[end_idx]   -= 1
      counts = cumsum(delta)[:-1]
    """
    M = grid.size
    delta = np.zeros(M, dtype=np.int64)

    s_idx = np.searchsorted(grid, starts, side="left")
    e_idx = np.searchsorted(grid, ends, side="left")

    np.add.at(delta, s_idx, 1)
    np.add.at(delta, e_idx, -1)

    return np.cumsum(delta)[:-1]


def _prefix_integral(grid: np.ndarray, y_segment: np.ndarray) -> np.ndarray:
    """
    Prefix integral for a piecewise-constant signal defined on grid segments.

    If y_segment[j] is the constant value of y(t) on [grid[j], grid[j+1]),
    then this returns an array I such that:
        I[k] = ∫_{grid[0]}^{grid[k]} y(t) dt

    Inputs
    ------
    grid : np.ndarray, shape (M,)
        Time grid.
    y_segment : np.ndarray, shape (S=M-1,)
        Segment values (constant per segment).

    Outputs
    -------
    I : np.ndarray, shape (M,)
        Prefix integral evaluated at grid points.

    Units
    -----
    If y_segment is:
      - tokens/step, then I is (tokens/step)*seconds
      - tokens/second, then I is tokens
    """
    dt = grid[1:] - grid[:-1]        # shape (S,)
    area = y_segment * dt            # shape (S,)
    I = np.zeros(grid.size, dtype=np.float64)
    I[1:] = np.cumsum(area)
    return I


def _integral_on_interval(
    grid: np.ndarray,
    I: np.ndarray,
    a: float,
    b: float,
    y_segment: np.ndarray,
) -> float:
    """
    Compute ∫_{a}^{b} y(t) dt where y is piecewise-constant on the time grid.

    Inputs
    ------
    grid : np.ndarray, shape (M,)
        Time grid boundaries.
    I : np.ndarray, shape (M,)
        Prefix integral from _prefix_integral(grid, y_segment).
    a : float
        Interval start time (seconds).
    b : float
        Interval end time (seconds).
    y_segment : np.ndarray, shape (S=M-1,)
        Piecewise-constant segment values of y(t).

    Output
    ------
    integral : float
        Value of the definite integral over [a, b).

    Notes
    -----
    Uses:
      - prefix integrals for fully-covered interior segments
      - explicit left/right partial segment corrections

    This function is called in a per-phase loop, so it must be correct and stable.
    """
    if b <= a:
        return 0.0

    ia = np.searchsorted(grid, a, side="right") - 1
    ib = np.searchsorted(grid, b, side="right") - 1

    ia = int(np.clip(ia, 0, grid.size - 2))
    ib = int(np.clip(ib, 0, grid.size - 2))

    if ia == ib:
        return float(y_segment[ia] * (b - a))

    full = float(I[ib] - I[ia + 1])  # ∫ grid[ia+1]..grid[ib]
    left = float(y_segment[ia] * (grid[ia + 1] - a))
    right = float(y_segment[ib] * (b - grid[ib]))
    return left + full + right


# =============================================================================
# Refactoring: small, reusable pipeline functions
# =============================================================================
def _validate_and_normalize_schema(
    phases: pd.DataFrame,
    *,
    request_id_col: str,
    phase_type_col: str,
    t_start_col: str,
    t_end_col: str,
) -> pd.DataFrame:
    """
    Validate the input schema and normalize basic types.

    Conceptually, this corresponds to the paper's assumption that the trace provides
    well-defined phase instances with:
      - request identifier
      - phase type (prefill or decode)
      - start/end times defining a positive duration

    Inputs
    ------
    phases : pd.DataFrame, shape (n, ...)
        One row per phase instance i.
        Required columns (names configurable via *_col parameters):
          - request_id_col : request identifier (any hashable)
          - phase_type_col : string "prefill" or "decode" (case-insensitive)
          - t_start_col    : phase start time t_{i,s} (seconds)
          - t_end_col      : phase end time t_{i,e} (seconds)

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Copy of the input with:
          - t_start_col, t_end_col cast to float
          - phase_type_col normalized to lowercase
        All original columns are preserved (no data is dropped).

    Raises
    ------
    ValueError
        If required columns are missing, phase_type has invalid values,
        or any row has non-positive duration (t_end <= t_start).
    """
    df = phases.copy()

    _require_columns(df, [request_id_col, phase_type_col, t_start_col, t_end_col])

    df[t_start_col] = df[t_start_col].astype(float)
    df[t_end_col] = df[t_end_col].astype(float)

    if (df[t_end_col] <= df[t_start_col]).any():
        bad = df[df[t_end_col] <= df[t_start_col]][[request_id_col, phase_type_col, t_start_col, t_end_col]]
        raise ValueError(f"Found non-positive durations (t_end <= t_start):\n{bad}")

    df[phase_type_col] = df[phase_type_col].astype(str).str.lower()
    valid_types = {"prefill", "decode"}
    bad_types = sorted(set(df[phase_type_col]) - valid_types)
    if bad_types:
        raise ValueError(f"phase_type must be one of {sorted(valid_types)}. Bad values: {bad_types}")

    return df


def _infer_or_read_step_counts(
    df: pd.DataFrame,
    *,
    chunk_size: int,
    phase_type_col: str,
    prefill_tokens_col: str,
    decode_tokens_col: str,
    n_steps_col: str,
) -> pd.DataFrame:
    """
    Ensure trace-inferred step counts N_i exist and are integer-valued.

    Paper mapping (Problem Setup):
      - For decode phases: N_i = number of decode tokens (1 token/step).
      - For prefill with chunk size C: N_i = ceil(P_i / C).

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain phase_type_col.
        If n_steps_col is missing, must also contain:
          - prefill_tokens_col (prefill token count P_i, used only for prefill)
          - decode_tokens_col (decode token count, used only for decode)
    chunk_size : int
        Prefill chunk size C (tokens per prefill step), C > 0.
    phase_type_col : str
        Column name for phase type ("prefill" or "decode").
    prefill_tokens_col : str
        Column name for P_i.
    decode_tokens_col : str
        Column name for decode token count.
    n_steps_col : str
        Column name for step count N_i (optional input; inferred if missing).

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Same DataFrame with:
          - df[n_steps_col] present (float values but integer-ish)
          - df["N_i"] present as float (shape (n,))
        Values satisfy the integer-ish check (within tolerance).

    Notes
    -----
    The estimator uses N_i as a *known* regressor and as the numerator of λ̂_i = N_i/T_i.
    """
    if n_steps_col not in df.columns:
        _require_columns(df, [prefill_tokens_col, decode_tokens_col])

        pre = df[phase_type_col].eq("prefill")
        dec = df[phase_type_col].eq("decode")

        df[n_steps_col] = np.nan

        P = df.loc[pre, prefill_tokens_col].astype(float).to_numpy()
        D = df.loc[dec, decode_tokens_col].astype(float).to_numpy()

        # Prefill: ceil(P/C)
        df.loc[pre, n_steps_col] = np.ceil(P / float(chunk_size))

        # Decode: 1 token per step
        df.loc[dec, n_steps_col] = D

    df["N_i"] = df[n_steps_col].astype(float).to_numpy()
    _assert_integerish(df["N_i"], name="N_steps")
    return df


def _compute_durations_and_lambda_hat(
    df: pd.DataFrame,
    *,
    t_start_col: str,
    t_end_col: str,
    min_duration_sec: float,
) -> pd.DataFrame:
    """
    Compute per-phase observed duration T_i and phase-local step density proxy λ̂_i.

    Paper mapping:
      - T_i = t_{i,e} - t_{i,s}
      - λ̂_i = N_i / T_i  (Eq. (baseline_lambda_hat))  [steps/second]

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain:
          - df["N_i"] (steps, shape (n,))
          - t_start_col, t_end_col (seconds)
    min_duration_sec : float
        Optional clamp applied to each duration when forming λ̂_i.
        If > 0: T_i := max(T_i, min_duration_sec).
        If == 0: paper-faithful behavior (just require T_i > 0).

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Adds:
          - df["T_i"]        : float64, shape (n,), seconds
          - df["lambda_hat"] : float64, shape (n,), steps/second

    Raises
    ------
    ValueError
        If any resulting T_i <= 0.
    """
    T = (df[t_end_col] - df[t_start_col]).to_numpy(dtype=np.float64)
    if min_duration_sec > 0:
        T = np.maximum(T, float(min_duration_sec))
    df["T_i"] = T

    if (df["T_i"] <= 0).any():
        raise ValueError("All phase durations must be positive.")

    df["lambda_hat"] = (df["N_i"] / df["T_i"]).to_numpy(dtype=np.float64)
    return df


def _check_baseline_prefill_uniqueness(
    df: pd.DataFrame,
    *,
    request_id_col: str,
    phase_type_col: str,
) -> pd.DataFrame:
    """
    Enforce a baseline-only trace schema assumption: one prefill row per request.

    Why this exists
    ---------------
    The partial-chunk correction is implemented per request using the single
    prefill interval [t_start, t_end). If a request's prefill is split into multiple
    rows/segments, this baseline implementation would need the segments to be made unique first.

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain request_id_col and phase_type_col.

    Outputs
    -------
    prefill_df : pd.DataFrame, shape (n_pf, ...)
        Subset of df with only prefill phases (phase_type == "prefill").

    Raises
    ------
    ValueError
        If any request_id appears in more than one prefill row.
    """
    prefill_df = df[df[phase_type_col].eq("prefill")]
    if not prefill_df.empty:
        counts = prefill_df[request_id_col].value_counts()
        if (counts > 1).any():
            bad_ids = counts[counts > 1].index.tolist()[:10]
            raise ValueError(
                "Multiple prefill rows per request found. Baseline assumes at most one. "
                f"Make the segments unique first. Example request_ids: {bad_ids}"
            )
    return prefill_df


def _build_global_time_grid_from_df(
    df: pd.DataFrame,
    *,
    t_start_col: str,
    t_end_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract phase start/end arrays and construct the global time grid.

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain t_start_col and t_end_col.

    Outputs
    -------
    starts : np.ndarray, shape (n,), float64
        Phase start times.
    ends : np.ndarray, shape (n,), float64
        Phase end times.
    grid : np.ndarray, shape (M,), float64
        Sorted unique union of starts and ends.

    Semantics
    ---------
    The grid defines S = M-1 segments.
    All reconstructed time-varying signals are represented as arrays of length S
    (one value per segment).
    """
    starts = df[t_start_col].to_numpy(dtype=np.float64)
    ends = df[t_end_col].to_numpy(dtype=np.float64)
    grid = _build_time_grid(starts, ends)
    return starts, ends, grid


def _reconstruct_pressures(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_size: int,
    phase_type_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct piecewise-constant token pressures from phase overlap.

    Paper mapping (Trace-derived token pressures):
      - p_pf_full(t) = C * (# active prefill phases at t)
      - p_dec(t)     =      (# active decode phases at t)

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain phase_type_col with values "prefill"/"decode".
    grid : np.ndarray, shape (M,)
        Global time grid.
    starts : np.ndarray, shape (n,)
        Phase start times aligned with df rows.
    ends : np.ndarray, shape (n,)
        Phase end times aligned with df rows.
    chunk_size : int
        C, tokens per prefill step.
    phase_type_col : str
        Column name for phase type.

    Outputs
    -------
    p_pf_full : np.ndarray, shape (S=M-1,), float64
        Prefill pressure per segment, units: tokens/step.
    p_dec : np.ndarray, shape (S=M-1,), float64
        Decode pressure per segment, units: tokens/step.
        (Decode is 1 token/step per active decode request.)
    I_pf_full : np.ndarray, shape (M,), float64
        Prefix integral of p_pf_full over time:
          I_pf_full[k] = ∫_{grid[0]}^{grid[k]} p_pf_full(t) dt
        units: (tokens/step)*seconds.
    I_dec : np.ndarray, shape (M,), float64
        Prefix integral of p_dec over time, same units.

    Notes
    -----
    These pressures are expressed per step (tokens/step), not per unit time.
    This is why later we multiply time-integrals by λ̂_i (steps/second) to get tokens.
    """
    pre_mask = df[phase_type_col].eq("prefill").to_numpy()
    dec_mask = df[phase_type_col].eq("decode").to_numpy()

    n_pf = _sweepline_counts(grid, starts[pre_mask], ends[pre_mask]) if pre_mask.any() else np.zeros(grid.size - 1, dtype=np.int64)
    n_dec = _sweepline_counts(grid, starts[dec_mask], ends[dec_mask]) if dec_mask.any() else np.zeros(grid.size - 1, dtype=np.int64)

    p_pf_full = float(chunk_size) * n_pf.astype(np.float64)
    p_dec = n_dec.astype(np.float64)

    I_pf_full = _prefix_integral(grid, p_pf_full)
    I_dec = _prefix_integral(grid, p_dec)

    return p_pf_full, p_dec, I_pf_full, I_dec


def _build_uniform_partial_chunk_correction_rate(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the baseline's *uniform redistribution* partial-chunk correction signal.

    Paper mapping (Prefill chunking and partial-chunk correction):
      - For each prefill request r:
          N_r = ceil(P_r / C)
          ρ_r = P_r - C*(N_r-1)  in (0, C]
          μ_r = C - ρ_r          in [0, C)
      - Under uniform redistribution over the prefill interval [rs, re):
          corr_rate_r(t) = μ_r / (re - rs)  [tokens/second] on [rs, re)

    The estimator uses the correction only through integrated form:
        corr_tokens(a,b) = ∫_a^b corr_rate(t) dt  [tokens]

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Full phase table; used for column presence checks.
    grid : np.ndarray, shape (M,)
        Global time grid.
    chunk_size : int
        C, tokens per prefill step.
    prefill_df : pd.DataFrame, shape (n_pf, ...)
        Prefill-only subset (one row per request, enforced by caller).
        Must contain:
          - t_start_col, t_end_col
          - prefill_tokens_col (P_r)
          - n_steps_col (N_r)
    t_start_col, t_end_col : str
        Column names for prefill interval boundaries.
    prefill_tokens_col : str
        Column name for number of prefill tokens P_r.
    n_steps_col : str
        Column name for inferred prefill steps N_r.

    Outputs
    -------
    corr_rate : np.ndarray, shape (S=M-1,), float64
        Aggregate correction rate over all active prefill requests:
          corr_rate[j] = sum_r μ_r/(re-rs) for requests active on segment j
        units: tokens/second.
    I_corr : np.ndarray, shape (M,), float64
        Prefix integral of corr_rate over time:
          I_corr[k] = ∫_{grid[0]}^{grid[k]} corr_rate(t) dt
        units: tokens.

    Notes on why units differ
    -------------------------
    - p_pf_full is tokens/step, so ∫ p_pf_full dt is (tokens/step)*seconds.
    - corr_rate is tokens/second, so ∫ corr_rate dt is tokens.

    In the exposure computation, corr_tokens must be subtracted directly from A_pf_i
    (which is also tokens). It must NOT be multiplied by λ̂_i.
    """
    corr_rate = np.zeros(grid.size - 1, dtype=np.float64)

    if prefill_df.empty:
        I_corr = _prefix_integral(grid, corr_rate)
        return corr_rate, I_corr

    _require_columns(df, [prefill_tokens_col])

    rs = prefill_df[t_start_col].to_numpy(dtype=np.float64)
    re = prefill_df[t_end_col].to_numpy(dtype=np.float64)
    P = prefill_df[prefill_tokens_col].astype(float).to_numpy()
    N = prefill_df[n_steps_col].astype(float).to_numpy()

    rho = P - float(chunk_size) * (N - 1.0)
    rho = np.clip(rho, 0.0, float(chunk_size))
    mu = float(chunk_size) - rho
    mu = np.clip(mu, 0.0, float(chunk_size))

    dur = re - rs
    if np.any(dur <= 0):
        raise ValueError("Prefill interval with non-positive duration found; cannot compute uniform correction.")

    rate = mu / dur  # tokens/second

    delta = np.zeros(grid.size, dtype=np.float64)
    s_idx = np.searchsorted(grid, rs, side="left")
    e_idx = np.searchsorted(grid, re, side="left")

    np.add.at(delta, s_idx, rate)
    np.add.at(delta, e_idx, -rate)

    corr_rate = np.cumsum(delta)[:-1]
    I_corr = _prefix_integral(grid, corr_rate)
    return corr_rate, I_corr

def _build_end_partial_chunk_correction_rate(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the baseline's *end-localized* partial-chunk correction signal.

    Paper mapping (End-localized partial-chunk correction):
      - For each prefill request r:
          N_r = ceil(P_r / C)
          ρ_r = P_r - C*(N_r-1)  in (0, C]
          μ_r = C - ρ_r          in [0, C)
      - End-localized assumption:
          The final (possibly partial) prefill step occurs near the trace-visible
          end of prefill at time t_{r,e}. We therefore assign the entire missing
          mass μ_r to the *single* time-grid segment that contains t_{r,e}.

    Concretely, letting j(r) be the segment index such that:
        t_{r,e} ∈ [grid[j(r)], grid[j(r)+1]),   (with boundary clamping)
    we define a piecewise-constant correction rate:
        corr_rate(t) = Σ_r  μ_r / Δt_{j(r)}    on segment j(r),
    where Δt_{j} = grid[j+1] - grid[j].

    The estimator uses the correction only through integrated form:
        corr_tokens(a,b) = ∫_a^b corr_rate(t) dt  [tokens]

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Full phase table; used for column presence checks.
    grid : np.ndarray, shape (M,)
        Global time grid.
    chunk_size : int
        C, tokens per prefill step.
    prefill_df : pd.DataFrame, shape (n_pf, ...)
        Prefill-only subset (one row per request, enforced by caller).
        Must contain:
          - t_end_col (t_{r,e})
          - prefill_tokens_col (P_r)
          - n_steps_col (N_r)
        Note: t_start_col is accepted for signature compatibility but is not
        required for end-localized placement (only t_end matters).
    t_start_col, t_end_col : str
        Column names for prefill interval boundaries.
        (Only t_end_col is used by this implementation.)
    prefill_tokens_col : str
        Column name for number of prefill tokens P_r.
    n_steps_col : str
        Column name for inferred prefill steps N_r.

    Outputs
    -------
    corr_rate : np.ndarray, shape (S=M-1,), float64
        Aggregate correction rate over all requests, units: tokens/second.
        For end-localized correction, each request contributes to *exactly one*
        segment: the one containing its t_end.
    I_corr : np.ndarray, shape (M,), float64
        Prefix integral of corr_rate over time:
          I_corr[k] = ∫_{grid[0]}^{grid[k]} corr_rate(t) dt
        units: tokens.

    Notes on why units differ
    -------------------------
    - p_pf_full is tokens/step, so ∫ p_pf_full dt is (tokens/step)*seconds.
    - corr_rate is tokens/second, so ∫ corr_rate dt is tokens.

    In the exposure computation, corr_tokens must be subtracted directly from A_pf_i
    (which is also tokens). It must NOT be multiplied by λ̂_i.

    Edge cases / clamping
    ---------------------
    - If t_end equals the last grid boundary (grid[-1]), we clamp to the final
      segment index S-1. This preserves the "assign to the smallest grid interval
      containing t_end" intent under half-open segment conventions.
    - If a segment has zero duration (should not happen with unique sorted grid),
      we raise to avoid division by zero.
    """
    corr_rate = np.zeros(grid.size - 1, dtype=np.float64)

    if prefill_df.empty:
        I_corr = _prefix_integral(grid, corr_rate)
        return corr_rate, I_corr

    _require_columns(df, [prefill_tokens_col])

    # Pull per-request quantities
    te = prefill_df[t_end_col].to_numpy(dtype=np.float64)
    P = prefill_df[prefill_tokens_col].astype(float).to_numpy()
    N = prefill_df[n_steps_col].astype(float).to_numpy()

    # Compute missing mass μ_r = C - ρ_r, where ρ_r = P_r - C*(N_r-1)
    rho = P - float(chunk_size) * (N - 1.0)
    rho = np.clip(rho, 0.0, float(chunk_size))
    mu = float(chunk_size) - rho
    mu = np.clip(mu, 0.0, float(chunk_size))

    # Identify the grid segment containing t_end.
    # Using half-open segments [grid[j], grid[j+1]), we compute:
    #   j = rightmost grid index <= t_end, then clamp to [0, S-1].
    S = grid.size - 1
    j_end = np.searchsorted(grid, te, side="right") - 1
    j_end = np.clip(j_end, 0, S - 1).astype(np.int64)

    # Segment durations for rate conversion (tokens -> tokens/sec)
    dt = grid[1:] - grid[:-1]  # shape (S,)
    if np.any(dt <= 0):
        raise ValueError("Non-positive grid segment duration found; cannot compute end-localized correction.")

    # Each request contributes μ_r tokens to exactly one segment; convert to rate.
    rate = mu / dt[j_end]  # tokens/second (per request, on its end segment)

    # Accumulate per-segment rates. (No need for a delta sweep: each request hits one segment.)
    np.add.at(corr_rate, j_end, rate)

    I_corr = _prefix_integral(grid, corr_rate)
    return corr_rate, I_corr


PartialChunkCorrectionMode = Literal["uniform", "end", "none"]


def _build_partial_chunk_correction_rate(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
    mode: PartialChunkCorrectionMode = "end",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch wrapper for partial-chunk correction modes.

    This function exists to:
      (i) support a baseline ablation ("none"),
      (ii) support the paper-described alternatives ("uniform", "end").

    Modes
    -----
    - mode="uniform":
        Uniformly redistributes missing mass μ_r over the entire prefill interval.
        Calls _build_uniform_partial_chunk_correction_rate.

    - mode="end" (default):
        Assigns the entire missing mass μ_r to the grid segment containing the
        prefill end timestamp t_{r,e}. Calls _build_end_partial_chunk_correction_rate.

    - mode="none":
        No correction is applied (baseline ablation). Returns an all-zero correction
        rate and corresponding prefix integral.

    Inputs / Outputs
    ----------------
    Same as _build_uniform_partial_chunk_correction_rate for compatibility.
    Returns (corr_rate, I_corr) with:
      - corr_rate shape (S=M-1,), units tokens/second
      - I_corr shape (M,), units tokens
    """
    if mode == "uniform":
        return _build_uniform_partial_chunk_correction_rate(
            df,
            grid=grid,
            chunk_size=chunk_size,
            prefill_df=prefill_df,
            t_start_col=t_start_col,
            t_end_col=t_end_col,
            prefill_tokens_col=prefill_tokens_col,
            n_steps_col=n_steps_col,
        )

    if mode == "end":
        return _build_end_partial_chunk_correction_rate(
            df,
            grid=grid,
            chunk_size=chunk_size,
            prefill_df=prefill_df,
            t_start_col=t_start_col,
            t_end_col=t_end_col,
            prefill_tokens_col=prefill_tokens_col,
            n_steps_col=n_steps_col,
        )

    if mode == "none":
        corr_rate = np.zeros(grid.size - 1, dtype=np.float64)
        I_corr = _prefix_integral(grid, corr_rate)
        return corr_rate, I_corr

    raise ValueError(f"Unknown correction mode: {mode}. Expected one of: 'uniform', 'end', 'none'.")


def _compute_exposures(
    df: pd.DataFrame,
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    grid: np.ndarray,
    lam: np.ndarray,
    p_pf_full: np.ndarray,
    p_dec: np.ndarray,
    I_pf_full: np.ndarray,
    I_dec: np.ndarray,
    corr_rate: np.ndarray,
    I_corr: np.ndarray,
) -> pd.DataFrame:
    """
    Compute baseline integrated token exposures A_pf_i and A_dec_i for each phase instance.

    Paper mapping (Baseline integrated exposures, Eq. (baseline_integrated_exposures)):
      A_pf_i  = ∫_{t_s}^{t_e} \tilde p_pf(t) * λ̂_i dt
      A_dec_i = ∫_{t_s}^{t_e} p_dec(t)       * λ̂_i dt

    In this baseline implementation:
      - \tilde p_pf(t) is represented implicitly as:
            p_pf_full(t) - (correction expressed in integrated tokens)
      - So we compute:
            A_pf_i = λ̂_i * ∫ p_pf_full(t) dt  -  ∫ corr_rate(t) dt
            A_dec_i = λ̂_i * ∫ p_dec(t) dt

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain:
          - df["lambda_hat"] (steps/second)
        This function adds:
          - df["A_pf_i"], df["A_dec_i"] (tokens)
    starts, ends : np.ndarray, shape (n,)
        Phase boundaries aligned with df rows.
    grid : np.ndarray, shape (M,)
        Global time grid.
    lam : np.ndarray, shape (n,)
        λ̂_i per phase, steps/second.
    p_pf_full, p_dec : np.ndarray, shape (S=M-1,)
        Pressures per segment, tokens/step.
    I_pf_full, I_dec : np.ndarray, shape (M,)
        Prefix integrals of pressures, units: (tokens/step)*seconds.
    corr_rate : np.ndarray, shape (S=M-1,)
        Partial-chunk correction rate per segment, tokens/second
    I_corr : np.ndarray, shape (M,)
        Prefix integral of corr_rate, units: tokens.

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Adds:
          - A_pf_i  : float64, shape (n,), tokens
          - A_dec_i : float64, shape (n,), tokens

    Critical unit reminder (do not change)
    --------------------------------------
    corr_tokens = ∫ corr_rate dt is already in TOKENS, so we subtract it directly.
    We do NOT multiply corr_tokens by lam[i].
    """
    A_pf = np.zeros(len(df), dtype=np.float64)
    A_dec = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        a = float(starts[i])
        b = float(ends[i])

        int_pf_full = _integral_on_interval(grid, I_pf_full, a, b, p_pf_full)  # (tokens/step)*sec
        int_dec = _integral_on_interval(grid, I_dec, a, b, p_dec)              # (tokens/step)*sec
        corr_tokens = _integral_on_interval(grid, I_corr, a, b, corr_rate)     # tokens

        A_pf[i] = lam[i] * int_pf_full - corr_tokens
        A_dec[i] = lam[i] * int_dec

    df["A_pf_i"] = A_pf
    df["A_dec_i"] = A_dec
    return df


def _fit_nnls_betas(
    df: pd.DataFrame,
) -> tuple[LinearRegression, float, float, float, np.ndarray, np.ndarray]:
    """
    Fit non-negative least squares regression for β.

    Paper mapping:
      min_{β >= 0} Σ_i ( (β0*N_i + β1*A_pf_i + β2*A_dec_i) - T_i )^2

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain numeric columns:
          - "N_i"     : steps
          - "A_pf_i"  : tokens
          - "A_dec_i" : tokens
          - "T_i"     : seconds

    Outputs
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted model with fit_intercept=False and positive=True.
        Coefficients correspond to (β0, β1, β2).
    beta0, beta1, beta2 : float
        Estimated coefficients.
    X : np.ndarray, shape (n, 3)
        Design matrix:
          X[:,0] = N_i
          X[:,1] = A_pf_i
          X[:,2] = A_dec_i
    y : np.ndarray, shape (n,)
        Response vector y = T_i.

    Notes
    -----
    sklearn's LinearRegression(positive=True) performs NNLS-like constrained fitting
    for this simple setting.
    """
    X = np.column_stack(
        [
            df["N_i"].to_numpy(dtype=np.float64),
            df["A_pf_i"].to_numpy(dtype=np.float64),
            df["A_dec_i"].to_numpy(dtype=np.float64),
        ]
    )
    y = df["T_i"].to_numpy(dtype=np.float64)

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    beta0, beta1, beta2 = map(float, model.coef_)
    return model, beta0, beta1, beta2, X, y


def _compute_diagnostics(
    df: pd.DataFrame,
    *,
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute optional sanity-check diagnostics.

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Used to report feature means and row count.
    model : LinearRegression
        Fitted model used to compute predictions.
    X : np.ndarray, shape (n, 3)
        Design matrix (see _fit_nnls_betas).
    y : np.ndarray, shape (n,)
        Observed durations T_i.

    Outputs
    -------
    diagnostics : Dict[str, float]
        Keys:
          - n_phases
          - r2
          - rmse_seconds
          - mean_duration_seconds
          - mean_steps
          - mean_A_pf_tokens
          - mean_A_dec_tokens

    Notes
    -----
    These are not used in estimation; they are guardrails for engineers:
    - r2/rmse: is the linear model plausibly fitting?
    - means: helps detect unit mistakes (e.g., exposures wildly off scale).
    """
    y_hat = model.predict(X)
    resid = y_hat - y
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(df))) if len(df) else float("nan")

    return {
        "n_phases": float(len(df)),
        "r2": r2,
        "rmse_seconds": rmse,
        "mean_duration_seconds": float(np.mean(y)) if len(y) else float("nan"),
        "mean_steps": float(np.mean(df["N_i"])) if len(df) else float("nan"),
        "mean_A_pf_tokens": float(np.mean(df["A_pf_i"])) if len(df) else float("nan"),
        "mean_A_dec_tokens": float(np.mean(df["A_dec_i"])) if len(df) else float("nan"),
    }


# =============================================================================
# Main estimator
# =============================================================================
def estimate_betas_baseline(
    phases: pd.DataFrame,
    *,
    chunk_size: int,
    # Column names (override if your schema differs)
    request_id_col: str = "request_id",
    phase_type_col: str = "phase_type",       # "prefill" or "decode"
    t_start_col: str = "t_start",
    t_end_col: str = "t_end",
    prefill_tokens_col: str = "prefill_tokens", # used for prefill step count + correction
    decode_tokens_col: str = "decode_tokens", # used for decode step count
    n_steps_col: str = "N_steps",             # optional; if missing, inferred
    min_duration_sec: float = 0.0,            # set >0 if you want to clamp ultra-small durations
    correction_mode: PartialChunkCorrectionMode = "end"
) -> BaselineBetaResult:
    """
    Baseline estimator (time-integrated NNLS) with configurable partial-chunk correction.

    This function is the public entrypoint. It orchestrates the full pipeline described
    at the top of this file, without exposing internal representations.

    Inputs
    ------
    phases : pd.DataFrame, shape (n, ...)
        One row per phase instance i.

        Required core columns (names are configurable):
          - request_id_col : request identifier
          - phase_type_col : "prefill" or "decode"
          - t_start_col    : phase start time t_{i,s} (seconds)
          - t_end_col      : phase end time t_{i,e} (seconds)

        If n_steps_col is NOT provided, also required:
          - prefill_tokens_col : number of prefill tokens P_i (prefill only)
          - decode_tokens_col : decode tokens (decode only)

        Optional:
          - n_steps_col : if present, treated as authoritative trace-inferred N_i.

    chunk_size : int
        Prefill chunk size C > 0 (tokens per prefill step).
        Used for:
          - inferring prefill steps N_i = ceil(P_i/C) when needed
          - forming full-chunk prefill pressure p_pf_full(t) = C * (#active prefill)
          - computing partial-chunk missing mass μ_r for the correction

    correction_mode : {"end", "uniform", "none"}, default="end"
        Partial-chunk correction mode for prefill chunking:
          - "end": end-localized correction (paper default). Assigns missing
            prefill mass μ_r to the grid segment containing the prefill end time.
          - "uniform": uniformly redistributes μ_r over the entire prefill interval.
          - "none": disables partial-chunk correction (baseline ablation).
          
    Column-name parameters
    ----------------------
    All *_col parameters let you map your trace schema into the expected semantics.

    min_duration_sec : float
        Optional safety clamp for T_i when forming λ̂_i. See _compute_durations_and_lambda_hat.

    Outputs
    -------
    BaselineBetaResult
        (beta0, beta1, beta2, diagnostics), where:
          - beta0 : seconds/step
          - beta1 : seconds/token (prefill)
          - beta2 : seconds/token (decode)

    Estimation meaning (paper)
    --------------------------
    This corresponds to solving Eq. (baseline_nnls) using baseline features
    from Eq. (baseline_integrated_exposures) with the specified partial-chunk
    correction mode (default: end-localized).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    # 1) Validate schema and normalize
    df = _validate_and_normalize_schema(
        phases,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
    )

    if df.empty:
        raise ValueError("No phase rows provided after validation.")

    # 2) Get / infer step counts N_i
    df = _infer_or_read_step_counts(
        df,
        chunk_size=chunk_size,
        phase_type_col=phase_type_col,
        prefill_tokens_col=prefill_tokens_col,
        decode_tokens_col=decode_tokens_col,
        n_steps_col=n_steps_col,
    )

    # 2b) Phase durations and phase-local step density proxy λ̂_i
    df = _compute_durations_and_lambda_hat(
        df,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        min_duration_sec=min_duration_sec,
    )

    for col in ["N_i", "T_i", "lambda_hat"]:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values in {col}.")
    if (df["N_i"] < 0).any():
        raise ValueError("Negative N_i found.")
    if (df["T_i"] <= 0).any():
        raise ValueError("Non-positive T_i found.")

    # Baseline schema assumption: at most one prefill row per request
    prefill_df = _check_baseline_prefill_uniqueness(
        df,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
    )

    # 3) Build global time grid
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col=t_start_col, t_end_col=t_end_col)

    # 4) Reconstruct pressures via sweep-line overlap
    p_pf_full, p_dec, I_pf_full, I_dec = _reconstruct_pressures(
        df,
        grid=grid,
        starts=starts,
        ends=ends,
        chunk_size=chunk_size,
        phase_type_col=phase_type_col,
    )

    # 5) partial-chunk correction
    corr_rate, I_corr = _build_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=chunk_size,
        prefill_df=prefill_df,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        prefill_tokens_col=prefill_tokens_col,
        n_steps_col=n_steps_col,
        mode=correction_mode,
    )

    # 6) Compute exposures A_pf_i and A_dec_i
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

    mask_bad = (~np.isfinite(df["A_pf_i"])) | (~np.isfinite(df["A_dec_i"]))
    if mask_bad.any():
        bad = df.loc[mask_bad]
        raise ValueError(
            "Non-finite exposures found (A_pf_i/A_dec_i). "
            f"Bad rows (first 5):\n{bad.head(5)}"
        )
    
    # 7) NNLS regression
    model, beta0, beta1, beta2, X, y = _fit_nnls_betas(df)

    # 8) Diagnostics
    diagnostics = _compute_diagnostics(df, model=model, X=X, y=y)

    return BaselineBetaResult(beta0=beta0, beta1=beta1, beta2=beta2, diagnostics=diagnostics)


# =============================================================================
# Minimal example
# =============================================================================
def _example() -> None:
    """
    Tiny end-to-end example.

    Creates two requests (A,B), each with:
      - one prefill phase
      - one decode phase

    This is only a smoke test demonstrating expected input schema.
    """
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 96, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 6.0, "prefill_tokens": 0,  "decode_tokens": 4},
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 64, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 5.0, "prefill_tokens": 0,  "decode_tokens": 2},
        ]
    )

    res = estimate_betas_baseline(phases, chunk_size=64)
    print("Estimated betas (baseline):")
    print(f"  beta0 (sec/step):  {res.beta0:.6f}")
    print(f"  beta1 (sec/token): {res.beta1:.6f}")
    print(f"  beta2 (sec/token): {res.beta2:.6f}")
    print("\nDiagnostics:")
    for k, v in res.diagnostics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__": # pragma: no cover
    _example()
