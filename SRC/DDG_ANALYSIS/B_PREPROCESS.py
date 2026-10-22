""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

import json
import os
import numpy as np
from scipy.stats import kendalltau, linregress, spearmanr  
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from A_W_TAU import window_tau

import pandas as pd

def lookup_value(csv_path, key_column, key, value_column, target=None):
    """
    Lookup a value in a CSV file given a key and column names.

    Ligand names are only unique *within* a target (597 rows share 479 distinct
    names), so `target` must be given to avoid silently returning another
    target's row via .iloc[0].
    """
    try:
        df = pd.read_csv(csv_path)
        row = df[df[key_column] == key]
        if target is not None:
            row = row[row["target"] == target]
        if not row.empty:
            return row.iloc[0][value_column]
        else:
            print(f"Key '{key}' not found in column '{key_column}'.")
            return None
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


#-------------------------------------------------------------------------------------------------------------------

def preprocess(files, TARGETS_TO_SKIP, MCC=False):
    
    #---------------------------------------------------
    def categorize_by_median(y_vals, median_val ,delta=1):
        y_strat = np.where(
            y_vals <  median_val - delta, 0,
            np.where(y_vals >  median_val + delta, 2, 1)
        )
        return y_strat, median_val
    #---------------------------------------------------

    results_all = {}
    bad_splits = {}

    for method, file_path in files.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
        method_results = {}

        for split_idx, split in enumerate(data):
            for target in split:
                if target in TARGETS_TO_SKIP:
                    continue
                pairs = split[target]
                pred_list, exp_list = [], []

                for pair in sorted(pairs):
                    values = pairs[pair]
                    pred_list.append(values["prediction"])
                    exp_list.append(values["experimental"])
                if MCC==True:
                    # median is constant within a target, so any of its ligands gives it;
                    # `target` is required because ligand names repeat across targets
                    class_median = lookup_value("../../../0_DATA/COMPOUNDS/0_MOLECULES.csv", "name", str(pair), "median", target=target)
                    if class_median is None or not isinstance(class_median, float):
                        raise ValueError(f"Missing or invalid median for ligand '{pair}' in target '{target}': {class_median}")

                # RMSE
                mse = mean_squared_error(exp_list, pred_list)
                rmse = np.sqrt(mse)
                rmse_DIY = np.sqrt(np.mean((np.array(pred_list) - np.array(exp_list))**2))
                if rmse - rmse_DIY != 0:
                    raise ValueError(f"RMSE mismatch: {rmse} vs {rmse_DIY}")

                if np.isnan(pred_list).any() or np.isnan(exp_list).any():
                    raise ValueError(f"NaN in input: method={method}, target={target}, split={split_idx}")
                if len(pred_list) < 2 or len(exp_list) < 2:
                    raise ValueError(f"Too few points: method={method}, target={target}, split={split_idx}")

                if np.all(np.array(pred_list) == pred_list[0]) or np.all(np.array(exp_list) == exp_list[0]):
                    print(f"Constant preds/labels: method={method}, target={target}, split={split_idx}")
                    bad_splits.setdefault(target, set()).add(split_idx)
                    continue

                # ranking metrics
                tau, _ = kendalltau(pred_list, exp_list, nan_policy='raise', variant="b")
                if np.isnan(tau):
                    print(f"Kendall's tau NaN: method={method}, target={target}, split={split_idx}")

                # Spearman correlation
                rho, _ = spearmanr(pred_list, exp_list)
                if np.isnan(rho):
                    print(f"Spearman NaN: method={method}, target={target}, split={split_idx}")

                w_tau = window_tau(exp_list, pred_list, tol=0.1, silent=True)
                if np.isnan(w_tau):  # warn on NaN in wtau -> is due to similar inputs in X and Y
                    print(f"Window tau NaN: method={method}, target={target}, split={split_idx}  (NaN is generally due to constant (or in the case of window tau, verry simmilar) values in one of the two input arrays)")
                    bad_splits.setdefault(target, set()).add(split_idx)
                    continue

                raw_slope, _, _, _, _ = linregress(exp_list, pred_list)
                slope_diff = abs(raw_slope - 1)

                if MCC==True:
                    ypred = np.array(pred_list)
                    ytrue = np.array(exp_list)
                    pred_classes, _ = categorize_by_median(ypred, class_median)
                    true_classes,  _ = categorize_by_median(ytrue, class_median)
                    mcc = matthews_corrcoef(true_classes, pred_classes)

                # build entry
                entry = {
                    "rmse": rmse,
                    "kendalls_tau": tau,
                    "window_tau": w_tau,
                    "slope_diff": slope_diff,
                    "spearman": rho,           
                }

                if MCC:
                    entry["mcc"] = mcc

                method_results.setdefault(target, {})[split_idx] = entry

        results_all[method] = method_results

    # remove bad splits
    for method in results_all:
        for target in list(results_all[method].keys()):
            if target in bad_splits:
                for bad_split in bad_splits[target]:
                    results_all[method][target].pop(bad_split, None)

    # print a summary of removed splits
    if bad_splits:
        print("\nRemoved split summary:")
        for target, splits in bad_splits.items():
            print(f"  {target}: {len(splits)} splits removed → {sorted(splits)}")
    else:
        print("\nNo splits removed.")

    # NaN check
    for method, target_data in results_all.items():
        for target, split_data in target_data.items():
            for split, metrics in split_data.items():
                for metric, value in metrics.items():
                    if pd.isna(value) and metric != "window_tau":
                        print(f"NaN: method={method}, target={target}, split={split}, metric={metric}")

    return results_all, bad_splits
