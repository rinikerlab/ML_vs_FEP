# 2D Random Forest — Stratified Splits

## Overview

2D Morgan fingerprint Random Forest models trained with random train/test splits. 



## Workflow

1. **Load data** — Load molecule SMILES and dG values from CSV
2. **Define descriptors** — Morgan fingerprints (radius=3, fpSize=1024, count-based)
3. **Precompute fingerprints** — Generate and cache fingerprints in parallel per target
4. **Run grid search** — For each target, evaluate parameter combinations:
   - Descriptor: Morgan count fingerprints
   - Feature filtering: Drop correlated features (threshold=0.9)
   - Model: Random Forest (n_estimators=50, max_depth=5, min_samples_split=4)
   - Cross-validation: 100 random 80/20 splits
5. **Collect results** — Store experiments metadata and predictions for all splits & targets. 
6. **DDG** — Convert dG values to ddG values for every pair of predictions in the test set.


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


