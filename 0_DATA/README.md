# Input Data

## Overview

This directory contains all molecular data and reference information needed by the model training and analysis scripts.

## Contents

### `COMPOUNDS/`
Molecular compound data in multiple formats:
- **`SDF/`** — Individual SDF files for each compound
- **`PER_TARGET_SDF/`** — Single SDF file per target containing all compounds for that target
- **`_renders/`** — Pre-rendered molecule images (PNG)
- CSV dataframes with compound properties and experimental binding free energies

### `HAHN/`
Reference FEP calculations from Hahn et al., used as a baseline method for comparison.

### `N_COMPOUNDS_PER_TARGET.json`
JSON dictionary mapping target names to the number of compounds available for that target. Used by scripts for validation and reporting.

## Data Format

Compounds are associated with protein targets and include experimental dG and ddG (relative binding free energy) values used as ground truth for model training and evaluation.
