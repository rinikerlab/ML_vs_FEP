# Model Training and Predictions

## Overview

This directory contains implementations of different methods to predict binding free energy. Each method folder follows a consistent workflow:
1. Generate/train predictions
2. Format results into standardized JSON structure
3. Convert dG values to ddG (relative binding free energy) if needed

## Model Methods

### `2D/` — 2D Molecular Features
Random Forest models trained on Morgan Count fingerprints (2D molecular features). Split into:
- **`RANDOM/`** — Random train/test splits
- **`STRAT/`** — Stratified splits (balanced dG ranges)

### `3D/` — 3D Molecular Features  
Random Forest models trained on 3D molecular descriptors (geometry-dependent features). Same split structure as 2D.

### `FEP/` — Physics-Based Free Energy Perturbation
Reference calculations from Hahn et al., reformatted into the same 100-split structure for benchmarking.
- **`DDG/`** — Relative free energy changes
- **`DG/`** — Absolute free energy values

### `MIF/` — Molecular Interaction Fields (3D QSAR)
Comparative molecular field analysis (CoMFA-like) using modified PyComsia with MIF features.
- **`RANDOM/`** and **`STRAT/`** splits
- Each includes custom pycomsia implementation

### `PYC/` — PyComsia Original Fields
Standard 3D QSAR using original PyComsia field descriptors.
- **`RANDOM/`** and **`STRAT/`** splits

### `FUSION/` — Meta-Learner Ensemble
Combines predictions from 2D, 3D, MIF, and PyComsia using meta-learning approach.
- **`DDG/`** and **`DG/`** variants

### `Z_META/` — FEP Meta Models
Alternative FEP-based predictions (OFF and OPLS force fields).

## Output Format

Each model/split combination produces:
- JSON predictions for all test compounds and splits
- CSV files with experimental vs. predicted values
- Standardized directory structure for downstream analysis in `B_COMPARE_METHODS/`

## Sampling Strategies

- **RANDOM:** Fully random train/test splits
- **STRAT:** Stratified by experimental dG range (balanced classes)


## Structure
Methods and the respecitve sampling method (random or stratified) follow this organiation:

A_MODELS
├── MethodA (eg. 2D)
│   ├── Sampling1
│   │   ├── Step1
│   │   └── Step2
│   └── Sampling2 (eg. RANDOM)
│       └── Step1
└── MethodB
    └── Sampling1
        └── Step1