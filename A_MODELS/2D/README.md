# 2D Models — Morgan Fingerprint Random Forest

## Overview

"ML model using on 2D based Fingerprint. The results for both random and stratified sampling are produced in this folder". The code as is produces only results from the models reported in the paper. However the original code is given that can be adapted to search through multiple Pipelines/models if one wishes so. At the momement the grid search only covers a single pipeline/model, nameley the one shown in the paper.

## Workflow

Each split variant (RANDOM, STRAT) contains:

1. **Model Search** (`A_RF_MORGAN_COUNT.py/ipynb`) — Grid search over Pipelines: Preprocessing methods, Models, Hyperparameters and feature selection strategies. The code as is only runs through the model presented in the paper but the full grid search code is provided incase someone wants to make alterations and try further models.
2. **Obtain ddG Values** (`A_DDG_FROM_DG.ipynb`) — Format predictions and convert dG predictions to all possible ddG's within the test set.


## Output

- JSON file: For a single model (no gridsearch) this file holds formatted predictions for every split. If a grid search is done on multiple pipelines/models use the CSV files to get results from all models. The JSON file will only show results from the final model used in the gridsearch.
- CSV files: aggregated results for all Pipelines searched. Can be used to compare pipelines performance.

## Notes

- Morgan fingerprints are 2D hashed descriptors (radius-based circular fingerprints)
- Captures local molecular patterns but not 3D shape or conformation
- Faster to generate and requires no 3D structure preprocessing (usually obtained from docking)
