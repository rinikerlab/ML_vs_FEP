# 2D Random Forest — Stratified Splits

## Overview

2D Morgan fingerprint Random Forest models trained with stratified train/test splits. Stratification ensures each split has balanced representation of weak, medium, and strong binders. This is necessary for the MCC analysis.


## Stratification Strategy

Compounds are categorized into three classes based on experimental dG relative to median:
- **Class 0:** dG < median - δ (weak binders)
- **Class 1:** median - δ ≤ dG ≤ median + δ (moderate binders)
- **Class 2:** dG > median + δ (strong binders)


## Workflow

Identical to RANDOM (see README.md) variant but with stratified sampling:
1. Categorize compounds by dG range
2. Stratified 80/20 split (100 times)


## Output

### CSV Files
- **experiments_all.csv** — Experiment metadata with RMSE, MCC, and balanced accuracy per split
- **predictions_all.csv** — Per-sample predictions with compound name, target, SMILES, and exp. label

### JSON File: `dG_pred_&_error_all_splits.json`

**Structure overview:**
```
[
  {
    "target_name_1": {
      "ligand_name": {
        "prediction": <float>,
        "experimental": <float>
      }
    },
    "target_name_2": { ... }
  },
  { ... },  # split 2
  ...
]
```

**Details:**
- Root: Array of splits (100 total, indexed by position)
- Each split: Dictionary with target names as keys
- Each target: Dictionary with ligand names (or ligand pair names in B_DDG_FROM_DG) as keys
- Each ligand: Contains predicted dG and experimental dG values as floats

## Notes

- Some targets with low number of datapoints or small dynamic ranges are omitted as they cant fulfill a stratified 80/20 split.
- STRAT results were only used for MCC analysis.
