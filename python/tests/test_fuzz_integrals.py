"""
tests/test_fuzz_integrals.py

Hypothesis property tests for the piecewise-constant integration primitive:

    _integral_on_interval(grid, I, a, b, y_segment)

We compare it against a slow but transparent reference implementation.

This catches boundary bugs around:
- a/b landing exactly on grid points
- a/b inside the same segment vs multiple segments
- tiny dt segments
- degenerate intervals (a == b)
- ordering / clipping behavior

Note: We intentionally constrain inputs so a,b lie within the grid range,
since the production code typically integrates over phase intervals that come
from the same set of grid boundaries.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from estimators.baseline import _integral_on_interval, _prefix_integral


def _ref_integral_piecewise_constant(grid: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    """Slow reference integral: sum overlap of [a,b) with each grid segment."""
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
            total += float(y[j]) * (right - left)
    return float(total)


@st.composite
def piecewise_constant_case(draw):
    """
    Generate a valid (grid, y_segment, a, b) test case.

    - grid: strictly increasing float64 array, length M in [2, 25]
    - y_segment: float64 array length S=M-1
    - a, b: chosen within [grid[0], grid[-1]] with a <= b
    """
    M = draw(st.integers(min_value=2, max_value=25))

    # Build an increasing grid by accumulating positive increments.
    # Keep increments not-too-small to reduce floating issues, but still varied.
    incs: List[float] = draw(
        st.lists(
            st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=M - 1,
            max_size=M - 1,
        )
    )
    start = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))

    grid = np.empty(M, dtype=np.float64)
    grid[0] = float(start)
    for i in range(1, M):
        grid[i] = grid[i - 1] + float(incs[i - 1])

    # Segment values
    y_segment = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=M - 1,
            max_size=M - 1,
        )
    )
    y = np.array(y_segment, dtype=np.float64)

    # Choose a,b within [grid[0], grid[-1]]
    a = draw(st.floats(min_value=float(grid[0]), max_value=float(grid[-1]), allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=float(grid[0]), max_value=float(grid[-1]), allow_nan=False, allow_infinity=False))
    if b < a:
        a, b = b, a

    # Occasionally force a or b to land exactly on a grid boundary (edge coverage)
    if draw(st.booleans()):
        a = float(draw(st.sampled_from(list(grid))))
    if draw(st.booleans()):
        b = float(draw(st.sampled_from(list(grid))))
    if b < a:
        a, b = b, a

    return grid, y, float(a), float(b)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(piecewise_constant_case())
def test_integral_on_interval_matches_reference(case) -> None:
    grid, y_segment, a, b = case

    I = _prefix_integral(grid, y_segment)
    fast = _integral_on_interval(grid, I, a, b, y_segment)
    ref = _ref_integral_piecewise_constant(grid, y_segment, a, b)

    # Use a tolerance that is tight but robust to floating arithmetic
    assert fast == pytest.approx(ref, rel=1e-10, abs=1e-10)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(piecewise_constant_case())
def test_integral_zero_when_a_equals_b(case) -> None:
    grid, y_segment, a, _b = case
    I = _prefix_integral(grid, y_segment)
    val = _integral_on_interval(grid, I, a, a, y_segment)
    assert val == pytest.approx(0.0, abs=0.0)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(piecewise_constant_case())
def test_integral_additivity_over_split_point(case) -> None:
    """
    Property: ∫_a^b y = ∫_a^m y + ∫_m^b y for any m in [a,b].

    This catches subtle off-by-one and boundary handling errors.
    """
    grid, y_segment, a, b = case
    if b <= a:
        return

    I = _prefix_integral(grid, y_segment)

    m = (a + b) / 2.0
    whole = _integral_on_interval(grid, I, a, b, y_segment)
    left = _integral_on_interval(grid, I, a, m, y_segment)
    right = _integral_on_interval(grid, I, m, b, y_segment)

    assert whole == pytest.approx(left + right, rel=1e-10, abs=1e-10)
