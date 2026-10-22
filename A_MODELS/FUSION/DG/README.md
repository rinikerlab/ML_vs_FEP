# FUSION Meta-Learner — Absolute Binding Free Energy (ΔG)

## Overview

Meta-learning ensemble combining base model predictions for absolute binding free energy (dG). Secondary model learns optimal weighting of 2D, 3D, MIF, and PyComsia contributions.

## Structure

- **`A_CONSENSUS_FEP.ipynb`** — Meta-learner training on absolute dG
- **`A_OUTPUT/`** — Ensemble predictions

## Workflow

Identical to DDG variant but:
1. Uses absolute dG predictions from all base models
2. Meta-model learns weights optimized for absolute binding strength
3. Ensemble optimized for predicting absolute affinity (harder task)

## Base Models

Same as DDG variant:
- 2D Morgan fingerprints
- 3D descriptors
- MIF fields
- PyComsia standard fields

## Expected Differences vs. DDG Ensemble

Absolute dG meta-learner may:
- Show different optimal weights (absolute dG requires different base model balance)
- Achieve lower ensemble gain (absolute dG is harder to predict)
- Reveal which methods best capture absolute binding strength

## Meta-Learning Benefits for Absolute dG

Ensemble approach particularly valuable for dG because:
- No single method may capture absolute scale
- Combination of approaches may compensate for systematic errors
- Weighting lets method focus on where it excels

## Output

Absolute dG ensemble predictions in standard format.

## Key Comparison

- **DDG ensemble vs. DG ensemble:** Assess whether ensemble benefits transfer across prediction tasks
- **DG ensemble vs. best single model:** Does combining methods help for absolute binding?

## Notes

Absolute binding energy is fundamentally harder; ensemble provides opportunity to mitigate individual method biases through weighted combination.
