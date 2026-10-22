# FEP — Physics-Based Reference Calculations

## Overview

Free Energy Perturbation (FEP) calculations from Hahn et al., reformatted to match the ML model evaluation structure for fair comparison. This means FEP predictions are structured in the same format as the 100 80/20 train test splits. Note that calculating statistics for FEP performance such as RMSE, over these 100 pseudo train/test splits can result in slighly different values than when taking the avg. RMSE over all FEP predictions.

## Structure

- **`DDG/`** — Relative binding free energy (ddG) values from FEP
- **`DG/`** — Absolute binding free energy (dG) values from FEP

