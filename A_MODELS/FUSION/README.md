# FUSION — Consensus Ensemble (Meta-Model + FEP)

## Overview
Creates consensus predictions by combining the meta-model FEP predictions using three fusion strategies: average, max, and min.

## Structure

- **`DDG/`** — Relative binding free energy (ΔΔG) consensus predictions
- **`DG/`** — Absolute binding free energy (ΔG) consensus predictions

Each contains:
- **`A_CONSENSUS_FEP.ipynb`** — Script to generate consensus predictions
- **`A_OUTPUT/`** — Final fusion predictions (6 files: AVG/MAX/MIN × OFF/OPLS)

## Approach

Combines two prediction sources:
1. **Meta-model**: Ridge ensemble of 4 base models (2D, 3D, Pycomsia, MIF)
2. **FEP predictions**: Done once for OFF and once OPLS force fields

For each combination, generates three consensus strategies:
- **Average**: Mean of meta-model and FEP predictions
- **Max**: Takes the higher prediction
- **Min**: Takes the lower prediction

## Output

Six JSON files per binding type:
- `ddG_ridge_LOO_pred&error_AVG_MET_OFF.json` — Average fusion (meta + OFF)
- `ddG_ridge_LOO_pred&error_MAX_MET_OFF.json` — Max fusion (meta + OFF)
- `ddG_ridge_LOO_pred&error_MIN_MET_OFF.json` — Min fusion (meta + OFF)
- Same three for OPLS

## Notes
- Uses only RANDOM splits 

