# FUSION Meta-Learner — Relative Binding Free Energy (ΔΔG)

## Overview

Meta-learning ensemble combining 2D, 3D, MIF, and PyComsia predictions to estimate relative binding free energy (ddG). A secondary model learns to weight predictions from each base method.

## Structure

- **`A_CONSENSUS_FEP.ipynb`** — Meta-learner model training and ensemble creation
- **`A_OUTPUT/`** — Final ensemble predictions

## Workflow

1. Collect all base model predictions (2D-RANDOM, 3D-RANDOM, MIF-RANDOM, PYC-RANDOM)
2. Train meta-model (typically a linear model or neural network) on training compounds
3. Learn optimal weights for each base model's contributions
4. Apply ensemble to test splits
5. Evaluate ensemble vs. individual base models

## Base Models

- **2D:** Morgan fingerprints with RF
- **3D:** 3D descriptors with RF  
- **MIF:** Molecular Interaction Fields QSAR
- **PYC:** Standard PyComsia Fields QSAR

Combines diversity of:
- Feature types (2D, 3D, field-based)
- Learning algorithms (tree-based, QSAR)
- Prediction approaches

## Meta-Learner Design

Learns weights reflecting each base model's predictive power:
- May discover that some methods dominate
- May reveal complementary information across methods
- Typically outperforms best single model

## Output

- Ensemble ddG predictions
- Per-sample predictions and confidence
- Used in final comparative analysis (`B_COMPARE_METHODS/`)

## Key Question

Can ensemble approach combine diverse methods to beat individual performers? Does diversity provide complementary information?

## Notes

Only uses RANDOM splits (not STRAT) for consistency across base models.
