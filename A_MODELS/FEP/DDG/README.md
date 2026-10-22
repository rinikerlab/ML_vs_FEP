# FEP — Relative Binding Free Energy (ΔΔG)

## Overview

ΔΔG predictions directly from the perturpation graph of OFF and OPLS3e methods found in Hahn et al. 
The folder Z_COMPARE_METHODS_OF_DDG_CALCULATION contains results from comparing the two methods mentioned in the paper to obtain ddG values from absolute dG values

## Workflow by Method (OFF & OPLS)

### Step 1: Build FEP Perturbation Graph
- **Script**: `1_OFF_GRAPH.ipynb` / `1_OPLS_GRAPH.ipynb`
- **Input**: Raw FEP data from Hahn et al.
- **Output**: `OPENFF_FEP_DDG_FROM_GRAPH.csv` / `OPLS_FEP_DDG_FROM_GRAPH.csv`
  - List of all possible ΔΔG values obtained from the FEP Perturbation graph. Method described in paper, section "Obtaining ∆∆Gbind from FEP experiments". 

### Step 2: Format DDG Values for Train/Test Splits
- **Scripts**: 
  - `2_OFF_RANDOM_GET_DDG_FROM_GRAPH_FEP.ipynb` — Format predictions for RANDOM splits
  - `2_OFF_STRAT_GET_DDG_FROM_GRAPH_FEP_STRAT copy.ipynb` — Format predictions for STRAT splits
- **Output**: 
  - `OFF_RANDOM_ddG_FROM_FEP_GRAPH_pred&error.json` — Predictions formatted like ML outputs
  - `OFF_STRAT_ddG_FROM_FEP_GRAPH_pred&error.json` — Same structure, stratified splits


