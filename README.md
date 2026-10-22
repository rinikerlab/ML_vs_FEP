# Basic Machine Learning vs. FEP: A Comparative Analysis for Protein-Ligand Binding Affinity Prediction

This repository accompanies the publication *"Basic Machine Learning vs. FEP: A Comparative Analysis for Protein-Ligand Binding Affinity Prediction"* (DOI: XXX )

**Authors:** Stefan Chrin, Dr. Gregory Landrum, Prof. Dr. Sereina Riniker


## Overview

This repository contains the complete workflow and results for the comparison of machine learning and FEP for predicting protein-ligand binding free energies. The work evaluates multiple ML approaches (2D features, 3D features, 3D QSAR) against FEP calculations and proposes a meta-learner ensemble for optimal predictions.

> **Data sources:**
> - **Experimental binding data and input structures** — obtained from [Zenodo record 10782775](https://zenodo.org/records/10782775), following the same source used in Hahn et al.
> - **FEP reference results** — taken from the GitHub repository [protein-ligand-benchmark-analysis](https://github.com/dfhahn/protein-ligand-benchmark-analysis) accompanying Hahn et al. ([10.1021/acs.jcim.4c00417](https://doi.org/10.1021/acs.jcim.4c00417)).

## Installation

### Main Environment (Step 1)

```bash
conda env create -f ML_v_FEP.yaml
conda activate ML_v_FEP
```

All scripts (including ones utilising 3DQSAR/PyComsia) should be run with this ML_v_FEP environment.

### PyComsia Environment (Step 2)

```bash
conda env create -f PYCOMSIA.yml
conda activate pycomsia
```

**Why a separate environment?**

Two ML methods run PyComsia which has specific dependency requirements. All notebooks auto-activate this environment when needed. For any additional manual work with 3D QSAR models, explicitly activate `pycomsia`.


### Execution Order (if re-running)

1. Generate predictions in `A_MODELS/` 
2. Run statistical analysis in `B_COMPARE_METHODS/`

(see sub-folder READMEs)


### Exploring Results

1. **Statistical comparisons & visualizations**
   ```
   B_COMPARE_METHODS/
   ├── DDG/EFFECT_SIZE_PLOTS/         # Statistical comparisons (relative / ΔΔG)
   ├── DG/EFFECT_SIZE_PLOTS/          # Statistical comparisons (absolute / ΔG)
   └── WINS/                          # Number of "wins" when methods are compared
   ```

2. **Individual model predictions**
   ```
   A_MODELS/{2D,3D,FEP,MIF,PYC,FUSION}/*/{RANDOM,STRAT}/
   ```



## Directory Structure

```
├── 0_DATA/                      # Input datasets and compound information
│   └── COMPOUNDS/               # Molecular structures and reference data
├── A_MODELS/                    # Model training and predictions
│   ├── 2D/                      # Random Forest with Morgan fingerprints
│   ├── 3D/                      # Random Forest with 3D descriptors
│   ├── FEP/                     # Physics-based FEP reference calculations
│   ├── MIF/                     # Molecular Interaction Fields (3D QSAR)
│   ├── PYC/                     # PyComsia original descriptors
│   ├── Z_META/                  # Ridge meta-learner over the four ML models
│   ├── Z_META_FEP/              # Meta-learner with the FEP prediction as extra input
│   └── FUSION/                  # Meta-learner ensemble
├── B_COMPARE_METHODS/           # Statistical analysis and comparisons
│   ├── DDG/                     # Relative binding free energy analysis
│   └── DG/                      # Absolute binding free energy analysis
├── SRC/                         # Utility functions and shared code
├── Z_AUX/                       # Auxiliary analysis notebooks
├── ML_v_FEP.yaml                # Main Conda environment to run all scripts with
└── PYCOMSIA.yml                 # PyComsia Conda environment for 3D QSAR
```

See individual README files for more information.


## Notes on PyComsia

The 3D QSAR implementation (MIF and PYC models) uses code from the PyComsia package with modifications 

(Haga, C. L., Le, C. N., Yang, X. D., & Phinney, D. G. (2025). Py-CoMSIA: An Open-Source Implementation of Comparative Molecular Similarity Indices Analysis in Python. Pharmaceuticals, 18(3), 440. https://doi.org/10.3390/ph18030440)

- **License:** See `PYCOMSIA_LICENSE.txt`
- **Why separate environment:** To ensure no conflicts arise and the original Pycomsia environment is used as intended
- **When needed:** Only for 3D QSAR models in `A_MODELS/MIF/` and `A_MODELS/PYC/`
- **Auto-activation:** All notebooks using PyComsia auto-activate the `pycomsia` environment.
- **Default environment:** Run ALL notebooks with `ML_v_FEP`



## License

- **License:** See `ETH_LICENSE.txt`
- **PyComsia code:** See `PYCOMSIA_LICENSE.txt`


