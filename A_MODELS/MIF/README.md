# MIF — Molecular Interaction Fields (3D QSAR)

## Overview

Modified 3D QSAR method using rdkits Molecular Interaction Fields (MIF) descriptors. Adapted from PyComsia.

## Workflow

Each split (RANDOM, STRAT) contains:

1. **QSAR Model** (`A_QSAR_MIF.ipynb`) — Trains MIF-based QSAR models using PyComsia backend
2. **Prediction Formatting** (`B_FORMAT_PREDICTIONS.ipynb`) — Extracts and standardizes results
3. **ddG Conversion** (`C_DDG_FROM_DG.ipynb`) — Converts absolute to relative binding free energy


