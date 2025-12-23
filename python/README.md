# Trace-Only Beta Estimation (Python)

This directory contains the reference Python implementation of the
**trace-only baseline estimator** for step-level execution coefficients
\((\beta_0,\beta_1,\beta_2)\) described in the paper.

The estimator reconstructs token pressures from phase overlap,
forms time-integrated exposures, and fits a non-negative least-squares model.
No step boundaries or engine instrumentation are required.

---

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run the baseline example

```bash
python baseline/baseline.timeintegral.py
```

This runs a small synthetic example and prints the estimated coefficients
and basic diagnostics. You should see output similar to the following.

```text
Estimated betas (baseline):
  beta0 (sec/step):  0.000000
  beta1 (sec/token): 0.014338
  beta2 (sec/token): 0.515229

Diagnostics:
  n_phases: 4.0
  r2: 0.8517629407351839
  rmse_seconds: 0.33343334333658375
  mean_duration_seconds: 2.5
  mean_steps: 2.25
  mean_A_pf_tokens: 76.0
  mean_A_dec_tokens: 2.625
```

## Run the baseline with sample traces

```bash
python examples/run.py --csv data/traces/sample.csv --chunk-size 64
```

---

## Code layout

```text
baseline/
  baseline.timeintegral.py   # baseline time-integrated NNLS estimator
```

The main entry point is:

```python
estimate_betas_baseline(phases: pd.DataFrame, chunk_size: int, ...)
```

See the function docstring for expected input schema and semantics.

---

## Notes

- Default settings reproduce the paper’s baseline estimator exactly.
- Optional robustness knobs (e.g., duration clamping) are documented in code.
- Iterative step-density reweighting will be implemented separately (future work).
