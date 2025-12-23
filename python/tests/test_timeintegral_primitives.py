"""
tests/test_timeintegral_primitives.py

High-leverage unit tests for the lowest-level math primitives in
baseline/timeintegral.py.

These functions are the foundation for pressures, correction signals,
and token exposure computations.
"""

from __future__ import annotations

import numpy as np
import pytest

from baseline.timeintegral import (
    _build_time_grid,
    _integral_on_interval,
    _prefix_integral,
    _sweepline_counts,
)


# ---------------------------------------------------------------------
# _build_time_grid
# ---------------------------------------------------------------------
def test_build_time_grid_sorts_and_uniques() -> None:
    # Intentionally unsorted and with duplicates
    starts = np.array([5.0, 1.0, 1.0, 3.0], dtype=np.float64)
    ends = np.array([6.0, 2.0, 4.0, 4.0], dtype=np.float64)

    grid = _build_time_grid(starts, ends)

    # Sorted unique union of starts and ends
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    assert np.array_equal(grid, expected)


def test_build_time_grid_requires_two_distinct_timestamps() -> None:
    starts = np.array([1.0, 1.0], dtype=np.float64)
    ends = np.array([1.0, 1.0], dtype=np.float64)  # all same -> only one distinct timestamp

    with pytest.raises(ValueError, match="at least two distinct timestamps"):
        _build_time_grid(starts, ends)


# ---------------------------------------------------------------------
# _sweepline_counts
# ---------------------------------------------------------------------
def test_sweepline_counts_simple_overlap() -> None:
    """
    Two intervals:
      A: [0, 2)
      B: [1, 3)

    Grid: [0,1,2,3)
    Counts per segment:
      [0,1): 1
      [1,2): 2
      [2,3): 1
    """
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    starts = np.array([0.0, 1.0], dtype=np.float64)
    ends = np.array([2.0, 3.0], dtype=np.float64)

    counts = _sweepline_counts(grid, starts, ends)
    expected = np.array([1, 2, 1], dtype=np.int64)
    assert np.array_equal(counts, expected)


def test_sweepline_counts_half_open_semantics() -> None:
    """
    Intervals are treated as [start, end). So:
      A: [0, 1)
      B: [1, 2)
    do NOT overlap.

    Grid: [0,1,2)
    Counts per segment:
      [0,1): 1
      [1,2): 1
    """
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    starts = np.array([0.0, 1.0], dtype=np.float64)
    ends = np.array([1.0, 2.0], dtype=np.float64)

    counts = _sweepline_counts(grid, starts, ends)
    expected = np.array([1, 1], dtype=np.int64)
    assert np.array_equal(counts, expected)


# ---------------------------------------------------------------------
# _integral_on_interval (+ _prefix_integral)
# ---------------------------------------------------------------------
def test_integral_on_interval_single_segment_partial() -> None:
    """
    y(t) is piecewise-constant on segments:
      grid = [0,1,2]
      y = [10 on [0,1), 20 on [1,2)]

    Integrate on a sub-interval fully inside a single segment:
      integral over [0.2, 0.8) = 10 * (0.6) = 6
    """
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y_segment = np.array([10.0, 20.0], dtype=np.float64)
    I = _prefix_integral(grid, y_segment)

    val = _integral_on_interval(grid, I, a=0.2, b=0.8, y_segment=y_segment)
    assert val == pytest.approx(10.0 * 0.6)


def test_integral_on_interval_multi_segment_with_partials() -> None:
    """
    Same y(t) as above:
      y = [10 on [0,1), 20 on [1,2)]

    Integrate over [0.5, 1.5):
      left partial in seg0: 10 * 0.5 = 5
      right partial in seg1: 20 * 0.5 = 10
      total = 15
    """
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y_segment = np.array([10.0, 20.0], dtype=np.float64)
    I = _prefix_integral(grid, y_segment)

    val = _integral_on_interval(grid, I, a=0.5, b=1.5, y_segment=y_segment)
    assert val == pytest.approx(15.0)


def test_integral_on_interval_exact_boundaries() -> None:
    """
    Integrate exactly on segment boundaries:
      integral over [0,2) = 10*1 + 20*1 = 30
      integral over [1,2) = 20*1 = 20
    """
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y_segment = np.array([10.0, 20.0], dtype=np.float64)
    I = _prefix_integral(grid, y_segment)

    v_full = _integral_on_interval(grid, I, a=0.0, b=2.0, y_segment=y_segment)
    v_tail = _integral_on_interval(grid, I, a=1.0, b=2.0, y_segment=y_segment)

    assert v_full == pytest.approx(30.0)
    assert v_tail == pytest.approx(20.0)


def test_integral_on_interval_degenerate_returns_zero() -> None:
    """
    By contract: if b <= a, integral is 0.0.
    """
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y_segment = np.array([10.0, 20.0], dtype=np.float64)
    I = _prefix_integral(grid, y_segment)

    assert _integral_on_interval(grid, I, a=1.0, b=1.0, y_segment=y_segment) == 0.0
    assert _integral_on_interval(grid, I, a=1.5, b=1.2, y_segment=y_segment) == 0.0
