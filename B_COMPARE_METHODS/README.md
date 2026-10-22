# Comparative Analysis of Methods

## Overview

Statistical comparison and analysis of all binding free energy prediction methods. Evaluates model performance, identifies winners/losers, computes effect sizes, and visualizes prediction accuracy distributions.

## Structure

### By Free Energy Type

- **`DDG/`** — Analysis using relative binding free energy (ΔΔG) values
- **`DG/`** — Analysis using absolute binding free energy (ΔG) values

### Analysis Types (within each DDG/DG folder)

- **`CONSENSUS_FEP/`** — Consensus predictions and comparison with FEP
- **`EFFECT_SIZE_PLOTS/`** — Statistical effect size visualizations
- **`ABSOLUTE_SLOPE_ANALYSIS/`** — CDF plots of prediction slope distributions
- **`WINS/`** — Head-to-head model comparison (win/loss counts)

## Workflow

1. **Collect predictions** from all models in `A_MODELS/`
2. **Statistical analysis** — correlations, slopes, effect sizes
3. **Comparative visualization** — which methods outperform others
4. **Detailed examination** — per-target and per-compound analysis

## Key Metrics

- Kendall's τ (rank correlation)
- Pearson correlation coefficient
- Prediction slopes vs experimental
- Win/loss rates (binary comparisons)
- Effect sizes (separation between method pairs)

## Output

- Jupyter notebooks with analysis and figures
- CSV/JSON files with numerical results
- PNG plots for thesis publication

## Notes

- Both DDG and DG analyses provide complementary perspectives
- CONSENSUS notebooks show results from combining FEP and ML
- WINS shows which methods beat which most frequently

