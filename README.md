# BLIS: Trace-Only Step-Level Calibration for LLM Inference

This repository accompanies the paper describing **BLIS**, a collection of trace-only methods for estimating step-level execution coefficients for vLLM-style inference engines.

The core idea is to infer per-step execution costs using only production traces (phase start/end times and token counts), without engine instrumentation or per-step timing.

---

## Repository structure

```text
paper/    # LaTeX source for the paper
python/   # Reference Python implementation of the baseline estimator
```

---

## Getting started

- See `paper/README.md` for building the paper.
- See `python/README.md` to run the baseline estimator.

---

## Notes

- The Python code implements the **baseline time-integrated NNLS estimator**
  described in the paper.
- Estimated coefficients parameterize a simulator; no trace timing is embedded
  in the simulator itself.

This repo is intended for research, reproducibility, and artifact evaluation.
