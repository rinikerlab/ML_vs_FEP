# FEP — Absolute Binding Free Energy (ΔG)

## Overview

Absolute dG predictions from FEP calculations (OFF and OPLS3e methods). Same formatting process as FEP/DDG but with absolute binding free energies instead of relative values.

## Workflow by Method (OFF & OPLS)

### Extract and format DG Values for Train/Test Splits
- **Scripts**: 
  - `OFF_RANDOM_GET_DG.ipynb` — Format predictions for RANDOM splits
  - `OFF_STRAT_GET_DG.ipynb` — Format predictions for STRAT splits
- **Input**: FEP dG data from Hahn et al.
- **Process**: For each of 100 splits, format dG values matching ML train/test structure
- **Output**: 
  - `OFF_RANDOM_dG_pred&error.json` — Predictions formatted like ML outputs
  - `OFF_STRAT_dG_pred&error.json` — Same structure, stratified splits

Identical for OPLS
