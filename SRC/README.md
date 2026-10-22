# Source Code — Shared Utilities

## Overview

Reusable Python functions and utilities shared across model training and analysis scripts.

## Contents

### `A_MODELS/`
Utility module for model-related operations:
- **`DDG_FROM_DG.py`** — Convert absolute dG predictions to relative ddG values using all pairwise ligand combinations

### `DDG_ANALYSIS/`
Utilities for analyzing and comparing ddG predictions:
- Helper functions for effect size computation
- Statistical comparison functions
- Visualization utilities


## Key Functions

**`ddG_from_dG(input_json, output_json)`**
- Input: JSON with dG predictions per compound
- Output: JSON with ddG (pairwise differences) for all ligand pairs per target
- Used by all models to convert predictions to relative binding energy format

## Dependencies

Requires:
- Python 3.12+
- Standard scientific stack (numpy, pandas, scipy)
- JSON library (built-in)
- scikit-learn, RDKit (from requirements.txt)

## Usage

Functions in this directory are imported by notebooks and scripts in `A_MODELS/` and `B_COMPARE_METHODS/`.
