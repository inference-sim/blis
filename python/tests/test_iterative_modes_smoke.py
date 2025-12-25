"""
tests/test_iterative_modes_smoke.py

Smoke tests: confirm the supported correction modes don't crash.

We assert only invariants (finite/nonnegative outputs, basic history consistency),
not exact beta values, to keep this stable across sklearn/numpy versions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from estimators.iterative import IterativeBetaResult, estimate_betas_iterative


def _sample_csv_path() -> Path:
    # Resolve: python/tests/... -> python/data/traces/sample.csv (match baseline test style)
    test_dir = Path(__file__).resolve().parent
    python_root = test_dir.parent
    return python_root / "data" / "traces" / "sample.csv"


def _assert_common_invariants(res: IterativeBetaResult) -> None:
    assert isinstance(res, IterativeBetaResult)

    # NNLS => nonnegative (allow tiny numerical noise)
    assert res.beta0 >= -1e-12
    assert res.beta1 >= -1e-12
    assert res.beta2 >= -1e-12

    # Finite betas
    assert np.isfinite(res.beta0)
    assert np.isfinite(res.beta1)
    assert np.isfinite(res.beta2)

    # Diagnostics present + finite
    diag = res.diagnostics
    assert "final_r2" in diag
    assert np.isfinite(float(diag["final_r2"]))
    assert "final_rmse_seconds" in diag
    assert np.isfinite(float(diag["final_rmse_seconds"]))

    # History consistency
    beta_hist = res.history["beta"]
    r2_hist = res.history["r2"]
    rmse_hist = res.history["rmse_seconds"]
    delta_hist = res.history["delta_beta_l2"]

    assert len(beta_hist) == res.n_outer_iters + 1
    assert len(r2_hist) == res.n_outer_iters
    assert len(rmse_hist) == res.n_outer_iters
    assert len(delta_hist) == res.n_outer_iters


def test_iterative_supported_modes_do_not_crash_on_sample_trace() -> None:
    csv_path = _sample_csv_path()
    assert csv_path.exists(), f"Missing sample trace CSV at {csv_path}"

    phases = pd.read_csv(csv_path)

    for mode in ["end", "uniform", "none", "beta_informed"]:
        res = estimate_betas_iterative(
            phases,
            chunk_size=64,
            prefill_correction_mode=mode,  # type: ignore[arg-type]
            max_outer_iters=10,
            tol=1e-12,
            damping_eta=1.0,
            init_via_baseline_correction="end",
        )
        _assert_common_invariants(res)
