# PYC — PyComsia Original Fields (3D QSAR)

## Overview

3D QSAR method using PyComsia with original (unmodified) field descriptors as described in the Pycomsia Paper. 

## Workflow

Each sampling varaint (RANDOM, STRAT) contains:

1. **QSAR Model** (`A_QSAR_ORIGINAL_PYCOMSIA_FIELDS.ipynb`) — Trains standard PyComsia QSAR model
2. **Prediction Formatting** (`B_FORMAT_PREDICTIONS.ipynb`) — Extracts predictions and formats them.
3. **ddG Conversion** (`C_DDG_FROM_DG.ipynb`) — Converts dG to ddG format


## Note on Pycomsia code
- The field computation remains the original, but individual script elements have been modified to support 80/20 random or stratified splits.