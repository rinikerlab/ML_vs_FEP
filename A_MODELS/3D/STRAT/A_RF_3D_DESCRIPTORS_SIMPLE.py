""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

# --- Paths and parameters --------------
df_path = "../../../0_DATA/COMPOUNDS/0_MOLECULES.csv"
sdf_folder = "../../../0_DATA/COMPOUNDS/SDF"
output_path = "./A_RF_3D_DESCRIPTORS"
desc_cache = "./precomputed_descriptors.npy"
n_splits = 100

TARGETS_TO_SKIP = ("galectin","p2","thrombin","cdk8","hif2a","hunt","eg5","jnk1")

# --- Imports ---
import time, os, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, RobustScaler
 

##########################################
# --- Timer helper ---
##########################################
start_time = time.time()
def print_with_time(msg):
    print(f"{msg} | Elapsed: {time.time() - start_time:.2f}s", flush=True)

##########################################
# --- Descriptor computation helper ---
##########################################
def compute_rdkit3d(sdf_list):
    feats = []
    for n, sdf in enumerate(sdf_list):
        if n % 10 == 0:
            print_with_time(f"Computing molecule {n}/{len(sdf_list)}")
        mol = Chem.MolFromMolFile(sdf, removeHs=False)
        if mol is None:
            raise ValueError(f"Bad sdf {sdf}")
        vals = [
            rdMolDescriptors.CalcPBF(mol),
            rdMolDescriptors.CalcPMI1(mol),
            rdMolDescriptors.CalcPMI2(mol),
            rdMolDescriptors.CalcPMI3(mol),
            rdMolDescriptors.CalcNPR1(mol),
            rdMolDescriptors.CalcNPR2(mol),
            rdMolDescriptors.CalcRadiusOfGyration(mol),
            rdMolDescriptors.CalcInertialShapeFactor(mol),
            rdMolDescriptors.CalcEccentricity(mol),
            rdMolDescriptors.CalcAsphericity(mol),
            rdMolDescriptors.CalcSpherocityIndex(mol)
        ]
        vals += list(rdMolDescriptors.CalcAUTOCORR3D(mol))
        vals += list(rdMolDescriptors.CalcRDF(mol))
        vals += list(rdMolDescriptors.CalcMORSE(mol))
        vals += list(rdMolDescriptors.CalcWHIM(mol))
        vals += list(rdMolDescriptors.CalcGETAWAY(mol))
        pt = Chem.GetPeriodicTable()
        radii = [pt.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        dclv = rdMolDescriptors.DoubleCubicLatticeVolume(mol, radii, False, True, 1.4, -1)
        vals += [dclv.GetCompactness(), dclv.GetPackingDensity(),
                 dclv.GetSurfaceArea(), dclv.GetVDWVolume(), dclv.GetVolume()]
        feats.append(vals)
    print_with_time("Finished descriptor computation")
    return np.array(feats, dtype=float)

##########################################
# --- Data ---
##########################################
print_with_time("Loading dataframe...")
df = pd.read_csv(df_path)
df["sdf_path"] = df["filename"].apply(lambda x: os.path.join(sdf_folder, x))
print_with_time(f"Loaded {len(df)} molecules")

# --- Precompute descriptors ---
if os.path.exists(desc_cache):
    print_with_time("Loading cached descriptors")
    X_all = np.load(desc_cache)
else:
    print_with_time("Computing descriptors...")
    X_all = compute_rdkit3d(df["sdf_path"].tolist())
    np.save(desc_cache, X_all)
    print_with_time(f"Saved cache: {desc_cache}")

##########################################
# --- Other pipeline parts ---
##########################################
class ScalerWithFilter(BaseEstimator, TransformerMixin):
    def __init__(self, method='robust', threshold=0.001):
        self.method = method; self.threshold = threshold
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.method == 'robust':
            med = X.median(); mad = (X - med).abs().median()
            self.kept = mad[mad >= self.threshold].index
            self.scaler = RobustScaler().fit(X[self.kept])
        else:
            self.var = VarianceThreshold(self.threshold).fit(X)
            self.scaler = StandardScaler().fit(self.var.transform(X))
        return self
    def transform(self, X):
        X = pd.DataFrame(X)
        if self.method == 'robust':
            return self.scaler.transform(X[self.kept])
        else:
            return self.scaler.transform(self.var.transform(X))

class DropCorrelatedFeaturesWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9): self.threshold = threshold
    def fit(self, X, y=None):
        self.filt = DropCorrelatedFeatures(threshold=self.threshold).fit(pd.DataFrame(X))
        return self
    def transform(self, X):
        return self.filt.transform(pd.DataFrame(X)).values

##########################################
# --- Fixed pipeline ---
##########################################
pipe = Pipeline([
    ("corr_filter", DropCorrelatedFeaturesWrapper(threshold=0.9)),
    ("scaler", ScalerWithFilter(method='robust', threshold=0.001)),
    ("model", RandomForestRegressor(
        n_jobs=1, n_estimators=50, max_depth=5,
        min_samples_split=4, max_features='sqrt', random_state=1))
])

cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

# --- keep the same categorize_by_median ---
def categorize_by_median(y, delta=1):
    med = np.median(y)
    return np.where(y < med - delta, 0, np.where(y > med + delta, 2, 1)), med

##########################################
# --- Run ---
##########################################
all_res = []
split_predictions = []  # store all per-split nested dicts

for target in sorted(df["target"].unique()):  # sorted order
    if target in TARGETS_TO_SKIP:
        print("Skipping", target)
        continue

    sub = df[df["target"] == target].reset_index(drop=True)
    idx = sub.index.values
    X, y = X_all[idx], sub["DG_conv"].values
    y_strat, med = categorize_by_median(y, delta=1)  # strat labels
    print_with_time(f"Target {target}, n={len(sub)}, median={med:.3f}")

    # --- use stratified shuffle split ---
    for split_index, (train_idx, test_idx) in enumerate(cv.split(X, y_strat)):
        print_with_time(f"Split {split_index} start (train={len(train_idx)}, test={len(test_idx)})")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print_with_time("Starting fit() ...")
        pipe.fit(X_train, y_train)
        print_with_time("Model fitted, starting predict() ...")

        y_pred = pipe.predict(X_test)
        print_with_time("Prediction done.")

        # --- per-split predictions ---
        if len(split_predictions) <= split_index:
            split_predictions.append({})
        if target not in split_predictions[split_index]:
            split_predictions[split_index][target] = {}

        for j, idx_test in enumerate(test_idx):
            lig_name = sub.loc[idx_test, "ligand"] if "ligand" in sub.columns else sub.loc[idx_test, "filename"]
            lig_name = f"lig_{lig_name}"
            split_predictions[split_index][target][lig_name] = {
                "prediction": float(y_pred[j]),
                "experimental": float(y_test[j])
            }

        print_with_time(f"Split {split_index} completed.")


# --- save outputs ---
os.makedirs(output_path, exist_ok=True)
with open(os.path.join(output_path, "dG_pred_&_error_all_splits.json"), "w") as f:
    json.dump(split_predictions, f, indent=2)

print_with_time("All done.")
