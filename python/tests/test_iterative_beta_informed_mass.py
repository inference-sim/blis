"""
tests/test_iterative_beta_informed_mass.py

High-ROI unit test for iterative-only beta_informed correction:

- Verifies mass conservation:
    ∫ c^(m)(t) dt == sum_r mu_r   (up to floating tolerance)

This guards the most error-prone part of the iterative estimator (windowing +
segment overlap logic) and is independent of sklearn/numpy version differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from estimators.iterative import _beta_informed_correction_rate_on_grid


def test_beta_informed_correction_mass_conservation_single_request() -> None:
    # Simple 4-second prefill with 4 steps (N_r = 4).
    # Choose P_r so that rho_r = 1 (last chunk has 1 token) => mu_r = C - 1.
    C = 64.0
    mu_r = C - 1.0

    prefill_req = pd.DataFrame(
        [
            {
                "request_id": "R",
                "t_rs": 0.0,
                "t_re": 4.0,
                "N_r": 4.0,
                "P_r": 1.0 + C * (4.0 - 1.0),  # rho = 1
                "mu_r": mu_r,
            }
        ]
    )

    # Grid with 1-second segments: [0,1), [1,2), [2,3), [3,4)
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    dt = grid[1:] - grid[:-1]
    assert np.all(dt > 0)

    # Constant lambda over all segments (steps/sec). Any positive constant should conserve mass.
    lambda_seg = np.ones(grid.size - 1, dtype=np.float64) * 2.0  # 2 steps/sec

    corr_rate = _beta_informed_correction_rate_on_grid(
        grid=grid,
        lambda_seg=lambda_seg,
        prefill_req=prefill_req,
    )

    # Basic shape + nonnegativity
    assert corr_rate.shape == (grid.size - 1,)
    assert np.all(corr_rate >= -1e-12)

    # Mass conservation: integral over time equals mu_r (tokens)
    mass = float(np.sum(corr_rate * dt))
    assert np.isfinite(mass)
    assert abs(mass - mu_r) <= 1e-9


def test_beta_informed_correction_mass_conservation_two_requests_additivity() -> None:
    C = 64.0

    # Two requests with different mu
    mu_a = 10.0
    mu_b = 20.0

    prefill_req = pd.DataFrame(
        [
            {"request_id": "A", "t_rs": 0.0, "t_re": 4.0, "N_r": 4.0, "P_r": 0.0, "mu_r": mu_a},
            {"request_id": "B", "t_rs": 1.0, "t_re": 5.0, "N_r": 4.0, "P_r": 0.0, "mu_r": mu_b},
        ]
    )

    # Grid covering both: [0,1), [1,2), [2,3), [3,4), [4,5)
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    dt = grid[1:] - grid[:-1]
    assert np.all(dt > 0)

    # Non-uniform lambda to exercise weighting (still should conserve mass).
    lambda_seg = np.array([1.0, 2.0, 4.0, 2.0, 1.0], dtype=np.float64)

    corr_rate = _beta_informed_correction_rate_on_grid(
        grid=grid,
        lambda_seg=lambda_seg,
        prefill_req=prefill_req,
    )

    assert corr_rate.shape == (grid.size - 1,)
    assert np.all(corr_rate >= -1e-12)

    mass = float(np.sum(corr_rate * dt))
    assert np.isfinite(mass)
    assert abs(mass - (mu_a + mu_b)) <= 1e-9


def test_beta_informed_correction_empty_prefill_table_is_zero() -> None:
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    lambda_seg = np.ones(grid.size - 1, dtype=np.float64)

    prefill_req = pd.DataFrame(columns=["request_id", "t_rs", "t_re", "N_r", "P_r", "mu_r"])
    corr_rate = _beta_informed_correction_rate_on_grid(grid=grid, lambda_seg=lambda_seg, prefill_req=prefill_req)

    assert corr_rate.shape == (grid.size - 1,)
    assert np.allclose(corr_rate, 0.0)
