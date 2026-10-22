""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

# --- Paths and parameters --------------
DATA_PATH    = "../../../0_DATA/COMPOUNDS/0_MOLECULES.csv"
N_SPLITS     = 100
TEST_SIZE    = 0.2
RANDOM_STATE = 42
SET_N_JOBS   = 1
test_mode    = False  # set to True and only a single target will run through pipeline
output_path = "./A_RF_MORGAN_COUNT/"

# --- Import packages ---

import os, json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from feature_engine.selection import DropCorrelatedFeatures
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed
from itertools import chain
import time
from datetime import datetime
    

#######################################
#                Setup                #
#######################################

# Load Data 
df = pd.read_csv(DATA_PATH, sep=",")

# Get times
start_time = time.time()

def print_with_time(msg):
    elapsed = time.time() - start_time
    print(f"{msg} | Elapsed: {elapsed:.2f}s", flush=True)
    
start_datetime = datetime.now()
print(f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


# stratification helper function to get classes
def categorize_by_median(y_vals, delta=1):
    median_val = np.median(y_vals)
    y_strat = np.where(
        y_vals <  median_val - delta, 0,
        np.where(y_vals >  median_val + delta, 2, 1)
    )
    return y_strat, median_val

# Define CV
custom_cv = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# ensure output directory exists
os.makedirs(output_path, exist_ok=True)



#######################################
#  Define descriptors and processes   #
#######################################

# --- Define Descriptors ---
class RDKitMorganFingerprintGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, radius=2, fpSize=2048, useCount=False):
        self.radius = radius; self.fpSize = fpSize; self.useCount = useCount
        self.cache = {}  # cache SMILES -> fingerprint so recomputing is unecessary
    def fit(self, X, y=None): return self
    def transform(self, X):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.fpSize,
            countSimulation=self.useCount
        )
        fps = []
        for smi in X:
            if smi in self.cache:
                arr = self.cache[smi]
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None: raise ValueError(f"Invalid SMILES: {smi}")
                arr = np.zeros(self.fpSize, dtype=int)
                # convert fingerprint to numpy
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
                self.cache[smi] = arr
            fps.append(arr)
        return np.array(fps)

class RDKitMorganCountFingerprintGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, radius=2, fpSize=2048):
        self.radius = radius
        self.fpSize = fpSize
        self.cache = {}  # cache SMILES -> count-fingerprint

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.fpSize
        )
        fps = []
        for smi in X:
            if smi in self.cache:
                arr = self.cache[smi]
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"Invalid SMILES encountered: {smi}")
                fp = fp_gen.GetCountFingerprint(mol)
                arr = np.zeros(self.fpSize, dtype=int)
                DataStructs.ConvertToNumpyArray(fp, arr)
                self.cache[smi] = arr
            fps.append(arr)
        fps_array = np.array(fps)
        #check_constant_descriptor(fps_array)
        return fps_array
    
    def __str__(self):
        return f"{self.__class__.__name__}(radius={self.radius}, fpSize={self.fpSize})"



class RDKitAtomPairFingerprintGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, bitSize=2048):
        self.bitSize = bitSize
        self.cache = {}  # cache SMILES -> atom-pair fingerprint
        
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=self.bitSize)
        fps = []
        for smi in X:
            if smi in self.cache:
                arr = self.cache[smi]
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None: raise ValueError(f"Invalid SMILES: {smi}")
                arr = np.zeros(self.bitSize, dtype=int)
                # convert atom-pair fingerprint to numpy
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
                self.cache[smi] = arr
            fps.append(arr)
        return np.array(fps)
    
    def __str__(self):
        return f"{self.__class__.__name__}(bitSize={self.bitSize})"


class RDKitDescriptorsGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        from rdkit.Chem import Descriptors
        feats = []
        for smi in X:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: raise ValueError(f"Invalid SMILES: {smi}")
            feats.append([func(mol) for _, func in Descriptors.descList])
        return np.array(feats)
    
    def __str__(self):
        return self.__class__.__name__
    

# --- Define potential preprocessing steps --
class DropCorrelatedFeaturesWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8): self.threshold = threshold
    
    def fit(self, X, y=None):
        self.drop = DropCorrelatedFeatures(threshold=self.threshold)\
            .fit(pd.DataFrame(X))
        return self
    
    def transform(self, X):
        return self.drop.transform(pd.DataFrame(X)).values
    
    def __str__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"


class ScalerWithFilter(BaseEstimator, TransformerMixin):
    def __init__(self, method='zscore', threshold=1e-4):
        self.method = method; self.threshold = threshold
        
    def fit(self, X, y=None):
        import pandas as _pd
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import StandardScaler, RobustScaler
        X_df = _pd.DataFrame(X)
        if self.method == 'zscore':
            self.filter_ = VarianceThreshold(self.threshold).fit(X_df)
            self.scaler_ = StandardScaler().fit(self.filter_.transform(X_df))
        else:
            med = X_df.median(); mad = (X_df - med).abs().median()
            self.keep_ = mad[mad >= self.threshold].index
            self.scaler_ = RobustScaler().fit(X_df[self.keep_].values)
        return self
    
    def transform(self, X, y=None):
        import pandas as _pd
        X_df = _pd.DataFrame(X)
        if hasattr(self, 'filter_'):
            return self.scaler_.transform(self.filter_.transform(X_df))
        return self.scaler_.transform(X_df[self.keep_].values)
    
    def __str__(self):
        return f"{self.__class__.__name__}(method='{self.method}', threshold={self.threshold})"



#######################################
#           Search options            #
#######################################

# descriptor & scaler options
descriptor_options = [
    RDKitMorganCountFingerprintGenerator(radius=3, fpSize=1024)
]

scaler_options = [
    ScalerWithFilter(method='zscore', threshold=0.001)
]

# define grids for different models
param_grid_rf = {
    'descriptor': descriptor_options,
    'corr_filter': ['passthrough', DropCorrelatedFeaturesWrapper(threshold=0.9)],
    'model': [RandomForestRegressor(n_jobs=1, max_features='sqrt')],
    'model__n_estimators': [100],
    'model__max_depth': [5],
    'model__min_samples_split': [2],
    'model__random_state': [1]
}

param_grid_lasso = {
    'descriptor': descriptor_options,
    'corr_filter': [DropCorrelatedFeaturesWrapper(threshold=0.9),
                    DropCorrelatedFeaturesWrapper(threshold=0.8)],
    'model': [Lasso(max_iter=15000, random_state=42)],
    'model__alpha': [0.01, 0.1, 1.0]
}

param_grid_ridge = {
    'descriptor': descriptor_options,
    'corr_filter': [DropCorrelatedFeaturesWrapper(threshold=0.9),
                    DropCorrelatedFeaturesWrapper(threshold=0.8)],
    'model': [Ridge(solver='svd')],
    'model__alpha': [0.01, 0.1, 1.0]
}

param_grid_xgb = {
    'descriptor': descriptor_options,
    'corr_filter': ['passthrough'],
    'model': [XGBRegressor(objective='reg:squarederror', n_jobs=1,
                            subsample=1.0, colsample_bytree=1.0, random_state=42)],
    'model__n_estimators': [50, 100],
    'model__max_depth': [5,10]
}

param_grid_lgbm = {
    'descriptor': descriptor_options,
    'corr_filter': ['passthrough'],
    'model': [LGBMRegressor(n_jobs=1,
                            subsample=1.0, colsample_bytree=1.0, random_state=42, verbose=0, min_data_in_leaf=1)],
    'model__n_estimators': [50, 100],
    'model__max_depth': [5,10],
    'model__min_data_in_bin': [1, 2]
}


param_grid_bayesian = {
    'descriptor': descriptor_options,
    'corr_filter': [DropCorrelatedFeaturesWrapper(threshold=0.8)],
    'model': [BayesianRidge()]
}

# Combine all parameter grids
param_grid = [
    param_grid_rf
]

# build param_list
# Scaler should only be used on 2D numerical descriptors but not on fps
param_list = []
for g in param_grid:
    for params in ParameterGrid(g):
        if isinstance(params['descriptor'], RDKitDescriptorsGenerator):
            for scaler in scaler_options:
                p = params.copy(); p['scaler'] = scaler
                param_list.append(p)
        else:
            params['scaler'] = 'passthrough'
            param_list.append(params)
            
print_with_time("Setup complete. Now running Grid Search...")

base_pipeline = Pipeline([
    ('descriptor', RDKitMorganFingerprintGenerator()),
    ('scaler', 'passthrough'),
    ('corr_filter', 'passthrough'),
    ('model', RandomForestRegressor(n_jobs=1, max_features='sqrt'))
])



#######################################
#                Run                  #
#######################################

# updated: add smiles list
# Function to use in parallel for a single target
def run_param_combo_for_target(params, descriptor_name, Xf, smiles_list, y, y_strat, target, cv, job_id):
    # Note how fingerprint generation isn't part of the pipeline as this is precomputed
    params_for_pipe = {k: v for k, v in params.items() if k != 'descriptor'}
    params_for_pipe['descriptor'] = 'passthrough'
    pipe = base_pipeline.set_params(**params_for_pipe)
    exps=[]; preds=[]
    for i,(ti,te) in enumerate(cv.split(Xf,y_strat)):
        exp_id = (job_id - 1)*N_SPLITS + i + 1
        Xtr,ytr = Xf[ti], y[ti]
        Xte,yte = Xf[te], y[te]
        pipe.fit(Xtr,ytr)
        ypred = pipe.predict(Xte)
        pred_classes = categorize_by_median(ypred)[0] # Look at what classes the predictions corespond to (default delta is 1)
        # Calculate classification accuracy statistics based on stratification classes
        mcc    = matthews_corrcoef(y_strat[te], pred_classes)
        balacc = balanced_accuracy_score(y_strat[te], pred_classes)
        rmse  = float(np.sqrt(((yte-ypred)**2).mean()))
        exps.append({
            'experiment_id':    exp_id,
            'target':           target,
            'split_idx':        i,
            'params':           json.dumps(params, default=str),
            'test_score':       rmse,            # regression RMSE
            'mcc':              mcc,             # new
            'balac': balacc          # new
        })
        for idx,true,pr in zip(te,yte,ypred):
            preds.append({
                'experiment_id':    exp_id,
                'target':          target,
                'split_idx':       i,
                'descriptor_type': descriptor_name,
                'prediction':      float(pr),
                'y_true':          float(true),
                'smiles':          smiles_list[idx]
            })
    return exps, preds


# main loop
from joblib import Parallel, delayed
from itertools import chain

all_exp  = []
all_pred = []
experiment_counter = 1  # global experiment counter (each target, model type/HP and split is its own experiment)


for target in df["target"].unique():
    # Remove certain targets
    if target in ["galectin","p2","thrombin","cdk8","hif2a","hunt", "jnk1", "eg5"]:
        continue
    sub = df[df["target"] == target].reset_index(drop=True)
    X = sub["smiles"].tolist()
    y_vals  = sub["DG_conv"].values
    y_strat, med = categorize_by_median(y_vals)
    
    # precompute fps per descriptor in parallel
    print_with_time(f"Precomputing fps for {target}")
    
    def compute_fp(desc, X):
        fps = desc.fit(X).transform(X)
        return str(desc), fps

    fps_cache = dict(
        Parallel(n_jobs=SET_N_JOBS)(
            delayed(compute_fp)(desc, X)
            for desc in sorted(descriptor_options, key=str)
    )
)

    # loop combos (parallel)
    jobs = [
        (
            {**p, 'descriptor': str(p['descriptor'])},   # keep all keys, overwrite descriptor
            str(p['descriptor']),
            fps_cache[str(p['descriptor'])],
            X
        )
        for p in param_list
    ]

    # Pre-enumerate jobs for parallel processing
    jobs_enum = list(enumerate(jobs, start=experiment_counter))


    results = Parallel(n_jobs=SET_N_JOBS)(
        delayed(run_param_combo_for_target)(
            params_mod, descriptor_name, Xf, X, y_vals, y_strat, target, custom_cv, job_id
        )
        for job_id, (params_mod, descriptor_name, Xf, X) in jobs_enum
    )

    experiment_counter += len(jobs)


    # flatten per-target results
    exps, preds = zip(*results)
    exps = list(chain.from_iterable(exps))
    preds = list(chain.from_iterable(preds))

    all_exp.extend(exps)
    all_pred.extend(preds)

# save all 
exp_df = pd.DataFrame(all_exp)
pred_df = pd.DataFrame(all_pred)

pred_df = pred_df.merge(
    df[["smiles", "name"]],
    on="smiles",
    how="left"
)

exp_df.to_csv(f"{output_path}experiments_all.csv", index=False)
pred_df.to_csv(f"{output_path}predictions_all.csv", index=False)


# Format the predictions for the remaining pipeline (ddG calculation, performance analysis etc.)
def df_to_nested_list(df):
    """Takes df containing predictions and formats into a list of dictionaries
       Each dictionary in the list represents a split
       Each of these dictionaries in turn containes a dictionary per target with all the information on corresponding predictions"""
    out_list = []
    for split, split_df in sorted(df.groupby("split_idx")):
        split_dict = {}  # one split
        for target, target_df in sorted(split_df.groupby("target")):
            target_dict = {}
            for _, row in target_df.iterrows():
                ligand = row["name"]
                target_dict[ligand] = {
                    "prediction": row["prediction"],
                    "experimental": row["y_true"]
                }
            split_dict[target] = target_dict
        out_list.append(split_dict)
    return out_list

# make formatted list
nested_list = df_to_nested_list(pred_df)

# save json
with open(f"{output_path}dG_pred_&_error_all_splits.json", "w") as f:
    json.dump(nested_list, f, indent=2)



# Finish
end_datetime = datetime.now()
print(f"End: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("\n")
print_with_time("Finished")
