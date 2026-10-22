# DDG Analysis — Relative Binding Free Energy Comparison

## Overview

Comparative analysis of all methods predicting relative binding free energy (ΔΔG).

## Subdirectories

### `ABSOLUTE_SLOPE_ANALYSIS/`
Plots showing the distribution of prediction slopes vs. experimental values across all models and targets.
- `SLOPES_DDG_RAND.ipynb` — Cumulative distribution function of slope distributions (random split)
- `SLOPES_P_VS_TRUE.ipynb` — Predicted vs. true slope analysis

### `CONSENSUS_FEP/`
Plot of consensus predictions combining FEP with ML models, and comparison against pure FEP or ML baseline.
- **OPLS variants** — Consensus using OPLS force field FEP
- **OFF variants** — Consensus using Open Force Field FEP

### `EFFECT_SIZE_PLOTS/`
Statistical effect size comparisons showing how significantly one method outperforms another.


### `META_WITH_FEP/`
Effect-size comparison including the meta-model that takes the FEP prediction as an additional input (see `A_MODELS/Z_META_FEP/`).

### `WINS/`
Comprison of meta-learner and FEP counting how often one method beats another across test compounds.

### `Comments`
~/B_COMPARE_METHODS/DDG/EFFECT_SIZE_PLOTS/COMPARE_STAT_RANDOM_SPED_UP_DONT_USE_FOR_REPRODUCING.ipynb is a sped up version of the analysis script that skips some none essential analyses. Do not use this script for reprocuding, rather use the original ~/B_COMPARE_METHODS/DDG/EFFECT_SIZE_PLOTS/COMPARE_STAT_RANDOM.ipynb
