# BLIS: Blackbox Inference Simulation

This directory contains the LaTeX source for the BLIS paper.

## Requirements (macOS)

A standard LaTeX distribution (e.g., TeX Live or MacTeX). On macOS, the easiest option is to install MacTeX:
https://www.tug.org/mactex/

## Build

To generate `blis.pdf`:

```bash
pdflatex blis.tex
bibtex blis
pdflatex blis.tex
pdflatex blis.tex
```