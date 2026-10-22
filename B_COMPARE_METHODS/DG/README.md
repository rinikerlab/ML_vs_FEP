# DG Analysis — Absolute Binding Free Energy Comparison

## Overview

Comparative analysis of all methods predicting absolute binding free energy (ΔG). This is more challenging than relative binding energy (ddG) but provides absolute binding strength predictions useful for compound ranking.

## Subdirectories

### `ABSOLUTE_SLOPE_ANALYSIS/`
CDF plots of prediction slopes for absolute dG values across models and targets.
- Slope distribution analysis showing prediction linearity
- Comparison of systematic biases between methods

### `CONSENSUS_FEP/`
Consensus models combining FEP with ML predictions for absolute dG.
- OPLS and OFF force field variants
- Assessment of ensemble benefit for absolute vs relative values

### `EFFECT_SIZE_PLOTS/`
Statistical effect size quantifying which methods significantly outperform others for dG prediction.
- Pairwise method comparisons
- Effect size magnitude visualization

### `WINS/`
Binary comparison results showing win/loss records for method pairs.
- Which methods most frequently produce better dG predictions
- Per-target performance variation

## Key Observations

Absolute dG prediction is generally more difficult than relative (ddG) because:
- Requires accurate prediction of binding constant itself
- Systematic errors in reference ligand prediction accumulate
- ML models optimized for relative affinity may not transfer to absolute scales

## Comparison vs. DDG Analysis

Both analyses use identical statistical methods but on different targets (absolute vs. relative). Differences in method ranking between dG and ddG analysis reveal which methods specialize in affinity changes vs. absolute binding strength.
