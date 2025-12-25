# Trace-Only Beta Estimation (Python)

This directory contains the reference Python implementation of the
**trace-only baseline and iterative estimators** for step-level execution coefficients \((\beta_0,\beta_1,\beta_2)\) described in the paper.

The estimators reconstructs token pressures from phase overlap,
forms time-integrated and (latent) step-averaged exposures, and fits a non-negative least-squares model. No step boundaries or engine instrumentation are required.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the baseline estimator (self-contained smoke test)

```bash
python estimators/baseline.py
```

This runs a small self-contained synthetic example and prints the estimated coefficients and basic diagnostics. You should see output similar to the following.

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

## Run the iterative estimator (self-contained smoke test)

```bash
python estimators/iterative.py
```

You should see output similar to the following.

```text
Estimated betas (iterative):
  beta0 (sec/step):  0.000000
  beta1 (sec/token): 0.016305
  beta2 (sec/token): 0.566230

Converged: True  (outer iters: 7)

Diagnostics:
  converged: True
  n_outer_iters: 7.0
  tol: 1e-06
  damping_eta: 1.0
  init_beta0: 1.007132667617689
  init_beta1: 0.001426533523537803
  init_beta2: 0.0
  final_beta0: 0.0
  final_beta1: 0.01630452988345182
  final_beta2: 0.5662300313808896
  final_r2: 0.872478801097554
  final_rmse_seconds: 0.3092586282981195
  final_delta_beta_l2: 1.923534171473337e-07
  mean_steps: 2.25
  mean_duration_seconds: 2.5
  n_phases: 4.0

Beta history (last few iters):
  iter   3: beta0=0.000000, beta1=0.016249, beta2=0.565666
  iter   4: beta0=0.000000, beta1=0.016301, beta2=0.566179
  iter   5: beta0=0.000000, beta1=0.016305, beta2=0.566224
  iter   6: beta0=0.000000, beta1=0.016305, beta2=0.566230
  iter   7: beta0=0.000000, beta1=0.016305, beta2=0.566230
```

## Run with sample traces

For the baseline algorithm, try:

```bash
python examples/run.py --csv data/traces/sample.csv --chunk-size 64
```

For the iterative algorithm, try:

```bash
python examples/run.py --csv data/traces/sample.csv --chunk-size 64 --algo iterative --correction-mode beta_informed --damping-eta 1.0
```

The CSV is expected to contain one row per phase instance (prefill or decode),
with start/end timestamps and token counts.

## Code layout

```text
estimators/
  baseline.py   # baseline time-integrated NNLS estimator
  iterative.py  # Majorization-Minimization (MM)-style iterative NNLS estimator

```

The main entry points are:

```python
estimate_betas_baseline(phases: pd.DataFrame, chunk_size: int, ...)
estimate_betas_iterative(phases: pd.DataFrame, chunk_size: int, ...)
```

See their docstrings for the precise schema and the mapping to paper notation.


## Testing and Coverage

The project uses `pytest` for unit tests and `pytest-cov` for coverage reporting.

### Install test dependencies

From this directory, with the virtual environment activated:

```bash
pip install -e '.[dev]'
```

### Run tests

Always invoke pytest via the Python interpreter to ensure it runs inside
the active virtual environment:

`python -m pytest`

#### Run tests with coverage

To run the full test suite with branch coverage and see missing lines:

```bash
python -m pytest \
  --cov=estimators \
  --cov-branch \
  --cov-report=term-missing
```

This reports coverage only for the estimators/ package (not examples).

## Notes

- Default settings reproduce the paper’s baseline and iterative estimators exactly.
- Optional robustness knobs (e.g., duration clamping) are documented in code.
- Use `python -m`. For instance, use `python -m pytest` and not `pytest`, to avoid accidentally running a system-installed `pytest` outside the virtual environment (common on macOS).
