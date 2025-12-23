"""
tests/test_timeintegral_sample_trace.py

Integration/regression test: run the estimator on the checked-in sample trace.

We assert only invariants (finite/nonnegative outputs), not exact beta values,
to keep this stable across sklearn/numpy versions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline.timeintegral import BaselineBetaResult, estimate_betas_baseline


def test_estimate_betas_on_sample_csv() -> None:
    # Resolve: python/tests/... -> python/data/traces/sample.csv
    test_dir = Path(__file__).resolve().parent
    python_root = test_dir.parent
    csv_path = python_root / "data" / "traces" / "sample.csv"

    assert csv_path.exists(), f"Missing sample trace CSV at {csv_path}"

    phases = pd.read_csv(csv_path)
    res = estimate_betas_baseline(phases, chunk_size=64)

    assert isinstance(res, BaselineBetaResult)

    # NNLS => nonnegative
    assert res.beta0 >= 0.0
    assert res.beta1 >= 0.0
    assert res.beta2 >= 0.0

    # Finite
    assert np.isfinite(res.beta0)
    assert np.isfinite(res.beta1)
    assert np.isfinite(res.beta2)

    # Diagnostics invariants
    diag = res.diagnostics
    assert diag["n_phases"] == float(len(phases))
    assert np.isfinite(diag["rmse_seconds"])
    assert "r2" in diag
