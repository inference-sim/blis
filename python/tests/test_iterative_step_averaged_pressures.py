"""
tests/test_iterative_step_averaged_pressures.py

High-ROI unit tests for the iterative primitive:

    _step_averaged_pressures_per_phase()

This is the backbone of the iterative algorithm: it converts grid-level signals
(lambda, tilde_p_pf, p_dec) into per-phase step-averaged pressures used in NNLS.

These tests are fully deterministic and do not depend on sklearn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estimators.iterative import _step_averaged_pressures_per_phase


def test_step_averaged_pressures_constant_lambda_reduces_to_time_average() -> None:
    # Grid: 3 segments of length 1
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    S = grid.size - 1

    # Constant lambda => weighting proportional to time
    lambda_seg = np.ones(S, dtype=np.float64) * 2.0  # steps/sec

    # Piecewise-constant signals on segments
    tilde_p_pf = np.array([10.0, 20.0, 30.0], dtype=np.float64)  # tokens/step
    p_dec = np.array([1.0, 3.0, 5.0], dtype=np.float64)          # tokens/step

    # Two phases:
    #  - phase0: [0,2) covers seg0 + seg1 equally
    #  - phase1: [1,3) covers seg1 + seg2 equally
    df = pd.DataFrame({"dummy": [0, 1]})
    starts = np.array([0.0, 1.0], dtype=np.float64)
    ends = np.array([2.0, 3.0], dtype=np.float64)

    bar_pf, bar_dec = _step_averaged_pressures_per_phase(
        df=df,
        grid=grid,
        starts=starts,
        ends=ends,
        lambda_seg=lambda_seg,
        tilde_p_pf_seg=tilde_p_pf,
        p_dec=p_dec,
    )

    # With constant lambda, bar_* is just the average across time in the interval.
    # phase0: avg of seg0 and seg1 = (10 + 20)/2 = 15
    # phase1: avg of seg1 and seg2 = (20 + 30)/2 = 25
    assert np.allclose(bar_pf, np.array([15.0, 25.0], dtype=np.float64))

    # phase0: (1 + 3)/2 = 2
    # phase1: (3 + 5)/2 = 4
    assert np.allclose(bar_dec, np.array([2.0, 4.0], dtype=np.float64))


def test_step_averaged_pressures_nonconstant_lambda_weights_segments_by_lambda() -> None:
    # Grid: 2 segments of length 1
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    S = grid.size - 1

    # Non-constant lambda: seg0 has much larger weight than seg1
    lambda_seg = np.array([10.0, 1.0], dtype=np.float64)  # steps/sec

    tilde_p_pf = np.array([100.0, 0.0], dtype=np.float64)  # tokens/step
    p_dec = np.array([0.0, 10.0], dtype=np.float64)        # tokens/step

    # Single phase spanning both segments: [0,2)
    df = pd.DataFrame({"dummy": [0]})
    starts = np.array([0.0], dtype=np.float64)
    ends = np.array([2.0], dtype=np.float64)

    bar_pf, bar_dec = _step_averaged_pressures_per_phase(
        df=df,
        grid=grid,
        starts=starts,
        ends=ends,
        lambda_seg=lambda_seg,
        tilde_p_pf_seg=tilde_p_pf,
        p_dec=p_dec,
    )

    # Compute expected weighted average:
    # Λ = ∫ lambda dt = 10*1 + 1*1 = 11
    # bar_pf = (∫ tilde_p_pf * lambda dt) / Λ
    #        = (100*10*1 + 0*1*1) / 11 = 1000/11
    # bar_dec = (0*10*1 + 10*1*1) / 11 = 10/11
    assert np.allclose(bar_pf[0], 1000.0 / 11.0)
    assert np.allclose(bar_dec[0], 10.0 / 11.0)


def test_step_averaged_pressures_handles_tiny_lambda_by_returning_zeroes() -> None:
    # If lambda is ~0 over the interval, Λ_i <= 0 triggers the safe fallback.
    grid = np.array([0.0, 1.0], dtype=np.float64)
    lambda_seg = np.array([0.0], dtype=np.float64)
    tilde_p_pf = np.array([123.0], dtype=np.float64)
    p_dec = np.array([456.0], dtype=np.float64)

    df = pd.DataFrame({"dummy": [0]})
    starts = np.array([0.0], dtype=np.float64)
    ends = np.array([1.0], dtype=np.float64)

    bar_pf, bar_dec = _step_averaged_pressures_per_phase(
        df=df,
        grid=grid,
        starts=starts,
        ends=ends,
        lambda_seg=lambda_seg,
        tilde_p_pf_seg=tilde_p_pf,
        p_dec=p_dec,
    )

    assert np.allclose(bar_pf, 0.0)
    assert np.allclose(bar_dec, 0.0)
