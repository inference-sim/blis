# tests/test_iterative_sample_trace.py
"""
Integration/regression test: run the iterative estimator on the checked-in sample trace.

We assert only invariants (finite/nonnegative outputs), not exact beta values,
to keep this stable across sklearn/numpy versions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from estimators.iterative import IterativeBetaResult, estimate_betas_iterative


def test_iterative_runs_on_sample_csv_all_modes() -> None:
    # Resolve: python/tests/... -> python/data/traces/sample.csv  (match baseline test)
    test_dir = Path(__file__).resolve().parent
    python_root = test_dir.parent
    csv_path = python_root / "data" / "traces" / "sample.csv"

    assert csv_path.exists(), f"Missing sample trace CSV at {csv_path}"

    phases = pd.read_csv(csv_path)

    for mode in ["end", "uniform", "none", "beta_informed"]:
        res = estimate_betas_iterative(
            phases,
            chunk_size=64,
            prefill_correction_mode=mode,  # type: ignore[arg-type]
            max_outer_iters=25,
            tol=1e-6,
            damping_eta=1.0,
            init_via_baseline_correction="end",
        )

        assert isinstance(res, IterativeBetaResult)

        # NNLS => nonnegative (allow tiny numerical noise)
        assert res.beta0 >= -1e-12
        assert res.beta1 >= -1e-12
        assert res.beta2 >= -1e-12

        # Finite
        assert np.isfinite(res.beta0)
        assert np.isfinite(res.beta1)
        assert np.isfinite(res.beta2)

        # Diagnostics invariants (we expect these to exist in your iterative impl)
        diag = res.diagnostics
        assert diag["n_outer_iters"] == float(res.n_outer_iters)
        assert np.isfinite(diag.get("final_rmse_seconds", float("nan")))
        assert "final_r2" in diag  # mirrors baseline's "r2" expectation
        assert np.isfinite(diag["final_r2"])

        # History lengths are consistent
        beta_hist = res.history["beta"]
        r2_hist = res.history["r2"]
        rmse_hist = res.history["rmse_seconds"]
        delta_hist = res.history["delta_beta_l2"]

        assert len(beta_hist) == res.n_outer_iters + 1
        assert len(r2_hist) == res.n_outer_iters
        assert len(rmse_hist) == res.n_outer_iters
        assert len(delta_hist) == res.n_outer_iters
