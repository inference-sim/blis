#!/usr/bin/env python3
"""
baseline_step_betas.py

Baseline trace-only estimator for step-level execution coefficients (betas)
for a vLLM-style inference engine.

This is the *baseline* described in the paper:

  - reconstruct overlap-based token pressures from phase overlap
  - use a phase-local constant step density proxy λ̂_i = N_i / T_i
    to convert time integrals (seconds-weighted) into step-aggregated token totals
  - solve NNLS for β = (β0, β1, β2)

Critically, this implementation includes the *uniform redistribution* special
case of the partial-chunk correction in its baseline usage:
  - the correction is only needed through integrated form
  - under uniform redistribution, it subtracts exactly μ_r tokens per request
    from the prefill exposure (in tokens), independent of step localization

-------------------------------------------------------------------------------
Model (paper notation)
-------------------------------------------------------------------------------
Step model:
  Δt_k = β0 + β1 * T_pf_k + β2 * T_dec_k

Phase-level regression (baseline):
  T_i ≈ β0 * N_i + β1 * A_pf_i + β2 * A_dec_i

where:
  - N_i = trace-inferred step count
  - λ̂_i = N_i / T_i (steps / second)

Baseline exposures (tokens):
  A_pf_i  = ∫_{t_s}^{t_e} \tilde p_pf(t) * λ̂_i dt
  A_dec_i = ∫_{t_s}^{t_e} p_dec(t)       * λ̂_i dt

Pressures (tokens / step):
  p_pf_full(t) = C * (# active prefill phases at t)
  p_dec(t)     =     (# active decode phases at t)

Partial-chunk correction:
  Each prefill request r has missing mass μ_r = C - ρ_r ∈ [0, C)
  Under uniform redistribution (baseline usage), we subtract μ_r tokens per request
  from the prefill exposure *in integrated form* over any window [a,b).

-------------------------------------------------------------------------------
Dependencies
-------------------------------------------------------------------------------
pip install numpy pandas scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

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
    """
    beta0: float  # seconds / step
    beta1: float  # seconds / token (prefill)
    beta2: float  # seconds / token (decode)
    diagnostics: Dict[str, float]


# =============================================================================
# Small helpers (kept tiny and well-commented)
# =============================================================================
def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

ArrayLike1D = Union[np.ndarray, pd.Series]

def _assert_integerish(x: ArrayLike1D, *, name: str, tol: float = 1e-6) -> None:
    """
    Paper assumes N_i is an integer step count.
    Accept either a numpy array or a pandas Series for convenience.
    """
    arr = x.to_numpy(dtype=np.float64) if isinstance(x, pd.Series) else x.astype(np.float64, copy=False)

    if np.any(arr < -tol):
        raise ValueError(f"{name} contains negative values.")
    if np.any(np.abs(arr - np.round(arr)) > tol):
        raise ValueError(f"{name} must be integer-valued within tol={tol}.")


def _build_time_grid(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Global time grid built from all phase start/end boundaries.
    Between grid points, activity is constant, so pressures are piecewise constant.
    """
    grid = np.unique(np.concatenate([starts, ends]))
    grid.sort()
    if grid.size < 2:
        raise ValueError("Need at least two distinct timestamps for a time grid.")
    return grid


def _sweepline_counts(grid: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Sweep-line active-counts per segment [grid[j], grid[j+1]).

    Pattern:
      delta[start_idx] += 1
      delta[end_idx]   -= 1
      count[j] = cumsum(delta)[j] for each segment j
    """
    M = grid.size
    delta = np.zeros(M, dtype=np.int64)

    s_idx = np.searchsorted(grid, starts, side="left")
    e_idx = np.searchsorted(grid, ends, side="left")

    np.add.at(delta, s_idx, 1)
    np.add.at(delta, e_idx, -1)

    # counts defined per segment => length M-1
    return np.cumsum(delta)[:-1]


def _prefix_integral(grid: np.ndarray, y_segment: np.ndarray) -> np.ndarray:
    """
    Prefix integral for piecewise-constant signal y(t) over segments.

    If y_segment[j] is y on [grid[j], grid[j+1]),
    then I[k] = ∫_{grid[0]}^{grid[k]} y(t) dt.
    """
    dt = grid[1:] - grid[:-1]
    area = y_segment * dt
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
    Compute ∫_{a}^{b} y(t) dt for y piecewise-constant on grid segments.
    Uses prefix integrals for full segments, plus end corrections.
    """
    if b <= a:
        return 0.0

    # segment indices covering a and b
    ia = np.searchsorted(grid, a, side="right") - 1
    ib = np.searchsorted(grid, b, side="right") - 1

    ia = int(np.clip(ia, 0, grid.size - 2))
    ib = int(np.clip(ib, 0, grid.size - 2))

    if ia == ib:
        return float(y_segment[ia] * (b - a))

    # full segments strictly inside (ia+1 .. ib-1)
    full = float(I[ib] - I[ia + 1])  # ∫ grid[ia+1]..grid[ib]
    left = float(y_segment[ia] * (grid[ia + 1] - a))
    right = float(y_segment[ib] * (b - grid[ib]))
    return left + full + right


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
    prompt_tokens_col: str = "prompt_tokens", # used for prefill step count + correction
    decode_tokens_col: str = "decode_tokens", # used for decode step count
    n_steps_col: str = "N_steps",             # optional; if missing, inferred
    min_duration_sec: float = 0.0,            # set >0 if you want to clamp ultra-small durations
) -> BaselineBetaResult:
    """
    Baseline estimator (time-integrated NNLS) with uniform partial-chunk correction.

    Parameters
    ----------
    phases:
      DataFrame with one row per phase instance (prefill or decode).
    chunk_size:
      Prefill chunk size C (tokens per prefill step), must be positive.
    min_duration_sec:
      Optional safety clamp: if > 0, clamp T_i to at least this value when forming λ̂_i.
      (Default 0.0 keeps paper-faithful behavior: simply require T_i > 0.)

    Returns
    -------
    BaselineBetaResult(beta0, beta1, beta2, diagnostics)
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    df = phases.copy()

    # -------------------------------------------------------------------------
    # 1) Validate schema and normalize
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2) Get / infer step counts N_i
    # -------------------------------------------------------------------------
    if n_steps_col not in df.columns:
        _require_columns(df, [prompt_tokens_col, decode_tokens_col])

        pre = df[phase_type_col].eq("prefill")
        dec = df[phase_type_col].eq("decode")

        df[n_steps_col] = np.nan
        P = df.loc[pre, prompt_tokens_col].astype(float).to_numpy()
        D = df.loc[dec, decode_tokens_col].astype(float).to_numpy()

        # Prefill: ceil(P/C)
        df.loc[pre, n_steps_col] = np.ceil(P / float(chunk_size))

        # Decode: 1 token per step
        df.loc[dec, n_steps_col] = D

    df["N_i"] = df[n_steps_col].astype(float).to_numpy()
    _assert_integerish(df["N_i"], name="N_steps")

    # Phase durations and phase-local step density proxy
    T = (df[t_end_col] - df[t_start_col]).to_numpy(dtype=np.float64)
    if min_duration_sec > 0:
        T = np.maximum(T, float(min_duration_sec))
    df["T_i"] = T

    if (df["T_i"] <= 0).any():
        raise ValueError("All phase durations must be positive.")

    df["lambda_hat"] = (df["N_i"] / df["T_i"]).to_numpy(dtype=np.float64)  # steps/second

    # Baseline assumption: at most one prefill row per request
    prefill_df = df[df[phase_type_col].eq("prefill")]
    if not prefill_df.empty:
        counts = prefill_df[request_id_col].value_counts()
        if (counts > 1).any():
            bad_ids = counts[counts > 1].index.tolist()[:10]
            raise ValueError(
                "Multiple prefill rows per request found. Baseline assumes at most one. "
                f"Merge segments first. Example request_ids: {bad_ids}"
            )

    # -------------------------------------------------------------------------
    # 3) Build global time grid
    # -------------------------------------------------------------------------
    starts = df[t_start_col].to_numpy(dtype=np.float64)
    ends = df[t_end_col].to_numpy(dtype=np.float64)
    grid = _build_time_grid(starts, ends)

    # -------------------------------------------------------------------------
    # 4) Reconstruct pressures via sweep-line overlap
    # -------------------------------------------------------------------------
    pre_mask = df[phase_type_col].eq("prefill").to_numpy()
    dec_mask = df[phase_type_col].eq("decode").to_numpy()

    n_pf = _sweepline_counts(grid, starts[pre_mask], ends[pre_mask]) if pre_mask.any() else np.zeros(grid.size - 1)
    n_dec = _sweepline_counts(grid, starts[dec_mask], ends[dec_mask]) if dec_mask.any() else np.zeros(grid.size - 1)

    # Pressures (tokens/step) on each grid segment
    p_pf_full = float(chunk_size) * n_pf.astype(np.float64)
    p_dec = n_dec.astype(np.float64)

    # Prefix integrals for fast integration on arbitrary phase windows
    I_pf_full = _prefix_integral(grid, p_pf_full)  # units: (tokens/step)*sec
    I_dec = _prefix_integral(grid, p_dec)          # units: (tokens/step)*sec

    # -------------------------------------------------------------------------
    # 5) Uniform partial-chunk correction (baseline usage)
    # -------------------------------------------------------------------------
    # For each prefill request r:
    #   rho_r = P_r - C*(N_r-1) in (0, C]
    #   mu_r  = C - rho_r       in [0, C)
    #
    # Under UNIFORM redistribution over the prefill interval [rs, re):
    # we subtract missing mass at a constant *token rate*:
    #   corr_rate_r(t) = mu_r / (re - rs)   [tokens / second] on [rs, re)
    #
    # Then for any phase window [a, b):
    #   corr_tokens(a, b) = ∫_a^b corr_rate(t) dt   [tokens]
    #
    # IMPORTANT:
    # corr_tokens is already in TOKENS (not tokens/step * seconds),
    # so it must be subtracted DIRECTLY from A_pf (which is also tokens).
    corr_rate = np.zeros(grid.size - 1, dtype=np.float64)  # tokens/second on segments

    if not prefill_df.empty:
        _require_columns(df, [prompt_tokens_col])

        rs = prefill_df[t_start_col].to_numpy(dtype=np.float64)
        re = prefill_df[t_end_col].to_numpy(dtype=np.float64)
        P = prefill_df[prompt_tokens_col].astype(float).to_numpy()
        N = prefill_df[n_steps_col].astype(float).to_numpy()

        # Compute rho and mu robustly
        rho = P - float(chunk_size) * (N - 1.0)
        rho = np.clip(rho, 0.0, float(chunk_size))
        mu = float(chunk_size) - rho
        mu = np.clip(mu, 0.0, float(chunk_size))

        dur = re - rs
        if np.any(dur <= 0):
            raise ValueError("Prefill interval with non-positive duration found; cannot compute uniform correction.")

        rate = mu / dur  # tokens/second

        # Sweep-line to build total corr_rate(t) across all prefill requests
        delta = np.zeros(grid.size, dtype=np.float64)
        s_idx = np.searchsorted(grid, rs, side="left")
        e_idx = np.searchsorted(grid, re, side="left")

        np.add.at(delta, s_idx, rate)
        np.add.at(delta, e_idx, -rate)

        corr_rate = np.cumsum(delta)[:-1]  # tokens/second per segment

    I_corr = _prefix_integral(grid, corr_rate)  # units: tokens

    # -------------------------------------------------------------------------
    # 6) Compute exposures A_pf_i and A_dec_i for each phase instance
    # -------------------------------------------------------------------------
    A_pf = np.zeros(len(df), dtype=np.float64)
    A_dec = np.zeros(len(df), dtype=np.float64)

    lam = df["lambda_hat"].to_numpy(dtype=np.float64)

    for i in range(len(df)):
        a = float(starts[i])
        b = float(ends[i])

        # Time integrals of pressures (piecewise constant on grid)
        int_pf_full = _integral_on_interval(grid, I_pf_full, a, b, p_pf_full)  # (tokens/step)*sec
        int_dec = _integral_on_interval(grid, I_dec, a, b, p_dec)              # (tokens/step)*sec

        # Integrated uniform correction (TOKENS)
        corr_tokens = _integral_on_interval(grid, I_corr, a, b, corr_rate)     # tokens

        # Convert pressure-time integrals to token exposures using λ̂_i (steps/sec).
        #
        # Units:
        #   lam[i] * int_pf_full  => (steps/sec) * ((tokens/step)*sec) = tokens
        #   lam[i] * int_dec      => tokens
        #   corr_tokens           => tokens   (already!)
        #
        # Therefore, DO NOT multiply corr_tokens by lam[i].
        A_pf[i] = lam[i] * int_pf_full - corr_tokens
        A_dec[i] = lam[i] * int_dec

    df["A_pf_i"] = A_pf
    df["A_dec_i"] = A_dec

    # -------------------------------------------------------------------------
    # 7) NNLS regression: T_i ≈ β0*N_i + β1*A_pf_i + β2*A_dec_i
    # -------------------------------------------------------------------------
    X = np.column_stack([
        df["N_i"].to_numpy(dtype=np.float64),
        df["A_pf_i"].to_numpy(dtype=np.float64),
        df["A_dec_i"].to_numpy(dtype=np.float64),
    ])
    y = df["T_i"].to_numpy(dtype=np.float64)

    # sklearn NNLS via positive=True
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    beta0, beta1, beta2 = map(float, model.coef_)

    # -------------------------------------------------------------------------
    # 8) Diagnostics (sanity checks only)
    # -------------------------------------------------------------------------
    y_hat = model.predict(X)
    resid = y_hat - y
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(df))) if len(df) else float("nan")

    diagnostics = {
        "n_phases": float(len(df)),
        "r2": r2,
        "rmse_seconds": rmse,
        "mean_duration_seconds": float(np.mean(y)) if len(y) else float("nan"),
        "mean_steps": float(np.mean(df["N_i"])) if len(df) else float("nan"),
        "mean_A_pf_tokens": float(np.mean(df["A_pf_i"])) if len(df) else float("nan"),
        "mean_A_dec_tokens": float(np.mean(df["A_dec_i"])) if len(df) else float("nan"),
    }

    return BaselineBetaResult(beta0=beta0, beta1=beta1, beta2=beta2, diagnostics=diagnostics)


# =============================================================================
# Minimal example
# =============================================================================
def _example() -> None:
    phases = pd.DataFrame(
        [
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prompt_tokens": 96, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 6.0, "prompt_tokens": 0,  "decode_tokens": 4},
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prompt_tokens": 64, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 5.0, "prompt_tokens": 0,  "decode_tokens": 2},
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


if __name__ == "__main__":
    _example()
