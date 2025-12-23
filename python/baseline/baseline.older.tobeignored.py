#!/usr/bin/env python3
"""
baseline_step_betas.py

A self-contained implementation of the **baseline**
(trace-only) estimator for step-level execution coefficients (betas) for a
vLLM-style inference engine, matching the baseline method described in the 
LaTeX writeup.

-------------------------------------------------------------------------------
What problem does this solve?
-------------------------------------------------------------------------------
We want to estimate step-level execution coefficients:

    step_duration = beta0 + beta1 * (prefill_tokens_in_step) + beta2 * (decode_tokens_in_step)

where:
  - beta0 >= 0 is a fixed per-step overhead             [seconds / step]
  - beta1 >= 0 is a per-prefill-token cost              [seconds / token]
  - beta2 >= 0 is a per-decode-token cost               [seconds / token]

We assume the engine executes steps sequentially (a single busy-loop).

However, production traces typically do NOT expose:
  - step boundaries,
  - per-step timings,
  - per-step token counts.

Instead, traces provide PHASE instances (per request):
  - a PREFILL phase (prompt processing), and
  - a DECODE phase (token generation),
with start/end timestamps and token counts.

This baseline estimator is "trace-only": it uses only phase-level fields from
traces plus known engine scheduling semantics.

-------------------------------------------------------------------------------
Baseline estimator in one equation (per phase i)
-------------------------------------------------------------------------------
For each phase instance i, we model its wall-clock duration T_i as:

    T_i  ≈  beta0 * N_i  +  beta1 * A_pf_i  +  beta2 * A_dec_i

where:
  - T_i     = observed phase duration (seconds)
  - N_i     = trace-inferred step count for the phase (steps)
  - A_pf_i  = integrated prefill exposure for that phase (tokens)
  - A_dec_i = integrated decode exposure for that phase (tokens)

This is a linear regression with nonnegative coefficients (NNLS).

-------------------------------------------------------------------------------
How do we compute exposures A_pf_i and A_dec_i from phases only?
-------------------------------------------------------------------------------
We reconstruct overlap-based "token pressures" from phase overlap in time.
Pressures are expressed in **tokens per step**:

  - Prefill full-chunk pressure:
        p_pf_full(t) = C * (# active prefill phases at time t)    [tokens / step]
        where C is the prefill chunk size.

  - Decode pressure:
        p_dec(t) = 1 * (# active decode phases at time t)         [tokens / step]
        since decode produces ~1 token per active request per step.

The baseline uses a phase-local step-density proxy:
        lambda_hat_i = N_i / T_i                                  [steps / second]
to convert time integrals into token exposures:

  - A_pf_i  = lambda_hat_i * ∫_{t_s}^{t_e} p_pf_tilde(t) dt
  - A_dec_i = lambda_hat_i * ∫_{t_s}^{t_e} p_dec(t)      dt

where p_pf_tilde(t) includes the partial-chunk correction below.

-------------------------------------------------------------------------------
Partial-chunk correction (uniform redistribution special case)
-------------------------------------------------------------------------------
Prefill processes chunks of size C tokens per step, except the final chunk may
be smaller. Treating all prefill steps as full chunks biases the pressure.

For each request r in prefill:
  - Prompt length: P_r (tokens)
  - Prefill steps: N_r (steps) = ceil(P_r / C)
  - Tokens in last chunk:
        rho_r = P_r - C*(N_r - 1)   in (0, C]
  - Missing mass relative to a full chunk:
        mu_r  = C - rho_r           in [0, C)

The writeup defines:
    p_pf_tilde(t) = p_pf_full(t) - Σ_r mu_r * w_r(t)

In the baseline we implement the **uniform redistribution** choice:
  - w_r(t) is uniform over request r's prefill interval.

Importantly, the baseline only needs ∫ p_pf_tilde(t) dt, so we apply the
correction at the INTEGRATED level over any window [a, b):

    ∫_{a}^{b} Σ_r mu_r w_r(t) dt
      = Σ_r mu_r * overlap_fraction(r, [a,b))

where overlap_fraction is the fraction of r's prefill interval overlapping [a,b).

This is dimensionally clean and matches the intended baseline behavior.

-------------------------------------------------------------------------------
Assumptions (explicit)
-------------------------------------------------------------------------------
1) Each request has at most ONE prefill phase row and at most ONE decode phase row.
   (If not, you must merge segments first or extend the logic.)
2) Decode semantics are "1 decode token per step per active decode request".
3) Prefill semantics are "C prefill tokens per step per active prefill request",
   with the uniform partial-chunk correction applied via mu_r.

-------------------------------------------------------------------------------
Dependencies
-------------------------------------------------------------------------------
- numpy
- pandas
- scikit-learn

Install:
    pip install numpy pandas scikit-learn

-------------------------------------------------------------------------------
How to run
-------------------------------------------------------------------------------
You can import and call estimate_betas_baseline(...) from your code, 
or run this file directly to see a tiny example:

    python baseline_step_betas.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# -----------------------------
# Output structure
# -----------------------------
@dataclass(frozen=True)
class BaselineBetaResult:
    """
    Result of baseline beta estimation.

    Attributes
    ----------
    beta0 : float
        Per-step fixed overhead (seconds / step).
    beta1 : float
        Prefill token cost (seconds / token).
    beta2 : float
        Decode token cost (seconds / token).
    diagnostics : dict
        Quick sanity-check stats for the fit.
    """
    beta0: float
    beta1: float
    beta2: float
    diagnostics: Dict[str, float]


# =============================================================================
# Main estimator
# =============================================================================
def estimate_betas_baseline(
    phases: pd.DataFrame,
    *,
    chunk_size: int,
    # Column names (edit these if your schema differs)
    request_id_col: str = "request_id",
    phase_type_col: str = "phase_type",       # "prefill" or "decode"
    t_start_col: str = "t_start",
    t_end_col: str = "t_end",
    prompt_tokens_col: str = "prompt_tokens", # used for prefill + correction
    decode_tokens_col: str = "decode_tokens", # used for decode step count
    n_steps_col: str = "N_steps",             # optional; if missing, inferred
) -> BaselineBetaResult:
    """
    Estimate (beta0, beta1, beta2) using the baseline time-integrated NNLS
    estimator, INCLUDING the uniform partial-chunk correction.

    Parameters
    ----------
    phases : pd.DataFrame
        One row per phase instance. Minimal required columns:
          - request_id_col  : request identifier (str/int)
          - phase_type_col  : "prefill" or "decode"
          - t_start_col     : phase start time (float)
          - t_end_col       : phase end time (float)

        Additionally required if n_steps_col is not provided:
          - prompt_tokens_col (for prefill)
          - decode_tokens_col (for decode)

        Additionally required for partial-chunk correction:
          - prompt_tokens_col (for prefill phases)

        IMPORTANT: We assume at most one prefill row per request.
    chunk_size : int
        Prefill chunk size C (tokens per prefill step).
    request_id_col, phase_type_col, t_start_col, t_end_col : str
        Column names for identifying and timing phases.
    prompt_tokens_col, decode_tokens_col : str
        Token-count columns used to infer step counts and compute partial-chunk correction.
    n_steps_col : str
        Optional column containing trace-inferred step count for each phase row.
        If present, we use it. If missing, we infer:
          - prefill: N_steps = ceil(prompt_tokens / chunk_size)
          - decode : N_steps = decode_tokens

    Returns
    -------
    BaselineBetaResult
        Estimated betas (nonnegative) and simple diagnostics.

    Raises
    ------
    ValueError
        If required columns are missing, durations are invalid, or assumptions are violated.
    """
    EPS = 1e-12  # small epsilon to avoid floating-point corner cases

    # Work on a copy so we never mutate the caller's DataFrame.
    df = phases.copy()

    # -------------------------------------------------------------------------
    # 0) Validate basic schema and normalize types
    # -------------------------------------------------------------------------
    required_cols = {request_id_col, phase_type_col, t_start_col, t_end_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"phases is missing required columns: {sorted(missing)}")

    # Convert times to floats
    df[t_start_col] = df[t_start_col].astype(float)
    df[t_end_col] = df[t_end_col].astype(float)

    # Validate positive durations
    if (df[t_end_col] <= df[t_start_col]).any():
        bad = df[df[t_end_col] <= df[t_start_col]][
            [request_id_col, phase_type_col, t_start_col, t_end_col]
        ]
        raise ValueError(f"Found phases with non-positive duration (t_end <= t_start):\n{bad}")

    # Normalize phase type strings
    df[phase_type_col] = df[phase_type_col].astype(str).str.lower()
    valid_types = {"prefill", "decode"}
    bad_types = sorted(set(df[phase_type_col]) - valid_types)
    if bad_types:
        raise ValueError(f"phase_type must be one of {sorted(valid_types)}. Bad values: {bad_types}")

    # -------------------------------------------------------------------------
    # 1) Ensure we have step counts N_i per phase (either provided or inferred)
    # -------------------------------------------------------------------------
    if n_steps_col not in df.columns:
        # To infer N_steps we need token columns.
        if prompt_tokens_col not in df.columns or decode_tokens_col not in df.columns:
            raise ValueError(
                f"If {n_steps_col!r} is missing, phases must include both "
                f"{prompt_tokens_col!r} and {decode_tokens_col!r}."
            )

        df[n_steps_col] = np.nan
        pre_mask = df[phase_type_col].eq("prefill")
        dec_mask = df[phase_type_col].eq("decode")

        # Prefill: N_steps = ceil(P / C)
        P = df.loc[pre_mask, prompt_tokens_col].astype(float).to_numpy()
        df.loc[pre_mask, n_steps_col] = np.ceil(P / float(chunk_size))

        # Decode: N_steps = decode_tokens (1 token per step)
        D = df.loc[dec_mask, decode_tokens_col].astype(float).to_numpy()
        df.loc[dec_mask, n_steps_col] = D

    # Validate N_steps
    df["N_i"] = df[n_steps_col].astype(float)
    if (df["N_i"] < 0).any():
        bad = df[df["N_i"] < 0][[request_id_col, phase_type_col, "N_i"]]
        raise ValueError(f"Found negative N_steps values:\n{bad}")

    # Phase durations and phase-local step density proxy
    df["T_i"] = df[t_end_col] - df[t_start_col]     # seconds
    if (df["T_i"] <= 0).any():  # redundant given earlier check, but keeps things explicit
        raise ValueError("All phase durations must be positive.")
    df["lambda_hat"] = (df["N_i"] / df["T_i"]).astype(np.float64)

    # -------------------------------------------------------------------------
    # 2) Enforce modeling assumption: at most one prefill row per request
    # -------------------------------------------------------------------------
    prefill_rows = df[df[phase_type_col].eq("prefill")]
    if len(prefill_rows) > 0:
        prefill_counts = prefill_rows[request_id_col].value_counts()
        if (prefill_counts > 1).any():
            bad_ids = prefill_counts[prefill_counts > 1].index.tolist()[:10]
            raise ValueError(
                "Found requests with multiple prefill phases. "
                "This baseline implementation assumes at most one prefill phase per request. "
                "Please merge them into a single prefill interval per request. "
                f"Example request_ids: {bad_ids}"
            )

    # -------------------------------------------------------------------------
    # 3) Build a global time grid from all phase boundary timestamps
    # -------------------------------------------------------------------------
    # This grid allows us to treat pressures as piecewise-constant between boundaries.
    time_points = np.unique(np.concatenate([df[t_start_col].to_numpy(), df[t_end_col].to_numpy()]))
    time_points.sort()
    if len(time_points) < 2:
        raise ValueError("Need at least two distinct timestamps to build the time grid.")

    t_left = time_points[:-1]          # interval starts
    t_right = time_points[1:]          # interval ends
    mid = (t_left + t_right) / 2.0     # midpoints used to detect activity on each interval

    # -------------------------------------------------------------------------
    # 4) Construct overlap-based pressures on each interval [t_left[j], t_right[j])
    # -------------------------------------------------------------------------
    # p_pf_full[j] = C * (# active prefill phases)    tokens/step
    # p_dec[j]     = 1 * (# active decode phases)    tokens/step
    p_pf_full = np.zeros_like(mid, dtype=float)
    p_dec = np.zeros_like(mid, dtype=float)

    for j, tm in enumerate(mid):
        active = (df[t_start_col] <= tm) & (tm < df[t_end_col])
        active_prefill = active & df[phase_type_col].eq("prefill")
        active_decode = active & df[phase_type_col].eq("decode")

        p_pf_full[j] = float(chunk_size) * float(active_prefill.sum())
        p_dec[j] = 1.0 * float(active_decode.sum())

    def integrate_piecewise_constant(a: float, b: float, y: np.ndarray) -> float:
        """
        Compute ∫_{a}^{b} y(t) dt when y is piecewise-constant on the global grid.

        - y[j] is the value on interval [t_left[j], t_right[j]).
        - We compute overlap with [a,b) and sum y[j] * overlap_duration.
        """
        if b <= a:
            return 0.0
        overlap = (t_left < b) & (t_right > a)
        if not np.any(overlap):
            return 0.0
        overlap_dt = np.minimum(t_right[overlap], b) - np.maximum(t_left[overlap], a)
        return float(np.sum(y[overlap] * overlap_dt))

    # -------------------------------------------------------------------------
    # 5) Compute partial-chunk correction values mu_r per prefill request
    # -------------------------------------------------------------------------
    # mu_r = C - rho_r, where rho_r = P_r - C*(N_r-1) in (0, C]
    # We store: request_id -> (prefill_start, prefill_end, mu_r)
    mu_by_request: Dict[object, Tuple[float, float, float]] = {}

    if len(prefill_rows) > 0:
        if prompt_tokens_col not in df.columns:
            raise ValueError(
                f"Need {prompt_tokens_col!r} to compute partial-chunk correction for prefill phases."
            )

        for row in prefill_rows.itertuples(index=False):
            rid = getattr(row, request_id_col)
            rs = float(getattr(row, t_start_col))
            re = float(getattr(row, t_end_col))

            P = float(getattr(row, prompt_tokens_col))
            N = float(getattr(row, n_steps_col))

            # Defensive check: prefill must have at least 1 step in this model
            if N < 1:
                raise ValueError(
                    f"Prefill request {rid} has N_steps={N} < 1; cannot compute partial-chunk correction."
                )

            rho = P - float(chunk_size) * (N - 1.0)

            # Sanity check: rho should be in (0, C] up to tiny tolerance
            if not (-1e-9 < rho <= float(chunk_size) + 1e-9):
                raise ValueError(
                    f"Bad rho={rho} for request {rid}. Check prompt_tokens={P}, "
                    f"N_steps={N}, chunk_size={chunk_size}."
                )

            rho = float(np.clip(rho, 0.0, float(chunk_size)))
            mu = float(chunk_size) - rho
            mu = float(np.clip(mu, 0.0, float(chunk_size)))  # mu in [0, C)

            mu_by_request[rid] = (rs, re, mu)

    # -------------------------------------------------------------------------
    # 6) Compute exposures A_pf_i and A_dec_i for each phase row i
    # -------------------------------------------------------------------------
    # For phase i with window [a,b):
    #
    #   int_pf_full = ∫ p_pf_full(t) dt          units: (tokens/step) * seconds
    #   int_dec     = ∫ p_dec(t) dt              units: (tokens/step) * seconds
    #
    # Uniform partial-chunk correction (integrated):
    #   correction_tokens = Σ_r mu_r * overlap_fraction(r, [a,b))
    #
    # Then exposures in TOKENS:
    #   A_pf_i  = lambda_hat_i * int_pf_full  -  lambda_hat_i * correction_tokens
    #   A_dec_i = lambda_hat_i * int_dec
    #
    # Both A_pf_i and A_dec_i end up in tokens, so beta1/beta2 are seconds/token.
    A_pf = np.zeros(len(df), dtype=float)
    A_dec = np.zeros(len(df), dtype=float)

    starts = df[t_start_col].to_numpy(dtype=np.float64)
    ends = df[t_end_col].to_numpy(dtype=np.float64)
    lambda_hat = df["lambda_hat"].to_numpy(dtype=np.float64)

    for i in range(len(df)):
        a = float(starts[i])
        b = float(ends[i])
        lam_hat = float(lambda_hat[i])
        int_pf_full = integrate_piecewise_constant(a, b, p_pf_full)
        int_dec = integrate_piecewise_constant(a, b, p_dec)

        # Integrated correction for the window [a,b):
        correction_tokens = 0.0
        for (rs, re, mu) in mu_by_request.values():
            overlap = max(0.0, min(b, re) - max(a, rs))
            if overlap <= EPS:
                continue
            frac = overlap / (re - rs)
            correction_tokens += mu * frac  # tokens

        A_pf[i] = lam_hat * int_pf_full - lam_hat * correction_tokens
        A_dec[i] = lam_hat * int_dec

    df["A_pf_i"] = A_pf
    df["A_dec_i"] = A_dec

    # -------------------------------------------------------------------------
    # 7) Solve NNLS regression: T_i ≈ beta0*N_i + beta1*A_pf_i + beta2*A_dec_i
    # -------------------------------------------------------------------------
    # Design matrix X has columns [N_i, A_pf_i, A_dec_i].
    X = np.column_stack(
        [
            df["N_i"].to_numpy(dtype=float),
            df["A_pf_i"].to_numpy(dtype=float),
            df["A_dec_i"].to_numpy(dtype=float),
        ]
    )
    y = df["T_i"].to_numpy(dtype=float)

    # Nonnegative least squares using sklearn: positive=True enforces coef_ >= 0.
    # fit_intercept=False because the model already includes beta0*N_i as the step overhead term.
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    beta0, beta1, beta2 = map(float, model.coef_)

    # -------------------------------------------------------------------------
    # 8) Diagnostics (simple)
    # -------------------------------------------------------------------------
    y_hat = model.predict(X)
    resid = y_hat - y
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(df))) if len(df) > 0 else float("nan")

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
# Example usage (runs when you execute this file directly)
# =============================================================================
def _example() -> None:
    """
    Tiny example that demonstrates the required input schema.

    Two requests:
      - Request A:
          prefill: t=[0,2), prompt_tokens=96
          decode : t=[2,6), decode_tokens=4
      - Request B:
          prefill: t=[1,3), prompt_tokens=64
          decode : t=[3,5), decode_tokens=2

    With chunk_size C=64:
      - A prefill N_steps = ceil(96/64) = 2; last chunk has rho=32 => mu=32
      - B prefill N_steps = ceil(64/64) = 1; last chunk has rho=64 => mu=0

    NOTE: This example is just for demonstrating inputs/outputs; it is not meant
    to be a physically realistic trace.
    """
    phases = pd.DataFrame(
        [
            # Request A
            {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0,
             "prompt_tokens": 96, "decode_tokens": 0},
            {"request_id": "A", "phase_type": "decode", "t_start": 2.0, "t_end": 6.0,
             "prompt_tokens": 0, "decode_tokens": 4},

            # Request B
            {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0,
             "prompt_tokens": 64, "decode_tokens": 0},
            {"request_id": "B", "phase_type": "decode", "t_start": 3.0, "t_end": 5.0,
             "prompt_tokens": 0, "decode_tokens": 2},
        ]
    )

    result = estimate_betas_baseline(phases, chunk_size=64)

    print("Estimated betas (baseline):")
    print(f"  beta0 (sec/step):  {result.beta0:.6f}")
    print(f"  beta1 (sec/token): {result.beta1:.6f}")
    print(f"  beta2 (sec/token): {result.beta2:.6f}")
    print("\nDiagnostics:")
    for k, v in result.diagnostics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _example()
