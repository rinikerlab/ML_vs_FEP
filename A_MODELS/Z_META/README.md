# Z_META 

## Overview
Ridge regression meta-model that combines predictions from all 4 base ML models (2D, 3D, Pycomsia, MIF). Leave-One-Out (LOO) cross-validation is implemented to validate the meta model.

## Structure

- **`DDG/`** — Relative binding free energy (ΔΔG) meta-model
- **`DG/`** — Absolute binding free energy (ΔG) meta-model

Each contains:
- **`RANDOM/`** — Random split cross-validation
- **`STRAT/`** — Stratified split cross-validation

## Model Details

Ridge regression ensemble that:
1. Combines predictions from 2D, 3D, Pycomsia, and MIF base models
2. Removes highly correlated features (correlation threshold: 0.9)
3. Removes low-variance features (variance threshold: 0.001)
4. Trains Ridge models (alpha=0.1) per target (LOOCV)

## Output
- Formatted JSON file with the meta-model's LOOCV predictions


