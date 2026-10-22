# Z_META DG — Ridge Ensemble Meta-Model for Absolute Binding Free Energy

## Overview

Ridge regression meta-model that combines ΔG (absolute binding free energy) predictions from 4 base ML models using Leave-One-Out (LOO) cross-validation.

## Structure

- **`RANDOM/`** — Random 100-split cross-validation
- **`STRAT/`** — Stratified 100-split cross-validation

Each contains:
- **`RIDGE.ipynb`** — Ridge regression LOO training and evaluation
- **`dG_ridge_LOO_pred&error.json`** — LOO predictions and RMSE metrics

## Base Models

The meta-model combines predictions from:
1. **2D** — 2D molecular descriptor model
2. **3D** — 3D molecular descriptor model
3. **Pycomsia** — CoMSIA-based model
4. **MIF** — Molecular Interaction Field model

## Model Details

Ridge regression with:
- Correlation-based feature removal (threshold: 0.9)
- Variance-based feature removal (threshold: 0.001)
- LOO cross-validation for unbiased evaluation
- Ridge parameter alpha=0.1

## Purpose

Assess whether combining base model predictions improves absolute binding energy predictions:
- Does the ensemble outperform individual models?
- Which base model predictions are most valuable for absolute binding affinity?
- How does absolute (ΔG) ensemble performance compare to relative (ΔΔG)?

## Comparison with DDG

Absolute binding energy (ΔG) is more challenging to predict than relative binding energy (ΔΔG). This meta-model tests whether ensemble learning helps bridge the gap.

## Output

- Predictions for each compound and split
- RMSE per target protein
- Model weights indicating base model importance

## Notes

Absolute binding energy (ΔG) predictions; compare with DDG meta-model for relative binding energies.
