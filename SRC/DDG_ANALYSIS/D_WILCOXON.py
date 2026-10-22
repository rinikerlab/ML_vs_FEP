""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

import json
import math
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import wilcoxon
import numpy as np

def compute_wilcox_posthoc(results_all, metrics=None, targets_to_skip=None):

    import numpy as np

    wilcox_posthoc_results = {metric: {} for metric in metrics}
    methods = sorted(results_all.keys())
    targets = sorted(t for t in next(iter(results_all.values())).keys() if t not in targets_to_skip)

    for metric in metrics:
        for target in targets:
            data = {}
            for method in methods:
                splits = sorted(results_all[method][target].keys())
                vals = []
                for s in splits:
                    if metric not in results_all[method][target][s]:
                        print(f"Missing metric {metric} → target {target} → method {method} → split {s}")
                        raise KeyError(metric)
                    vals.append(results_all[method][target][s][metric])
                data[method] = vals
            vs_df = pd.DataFrame(data)

            if metric != "window_tau":
                if vs_df.isnull().values.any():
                    print(f"NaN found in vs_df for metric={metric}, target={target}")
                    print("NaN locations (row, method):")
                    nan_locs = vs_df.isnull()
                    for row_idx, row in nan_locs.iterrows():
                        for method, is_nan in row.items():
                            if is_nan:
                                print(f"  Row {row_idx}, Method {method}")

            skip_target = False
            if metric == "mcc":
                methods_list = list(vs_df.columns)
                for i, m1 in enumerate(methods_list):
                    for m2 in methods_list[i+1:]:
                        x, y = vs_df[m1].values, vs_df[m2].values
                        if np.allclose(x, y, equal_nan=True):
                            print(f"Identical MCC values: {target} → {m1} vs {m2}")
                            skip_target = True
                            break
                    if skip_target:
                        break
            if skip_target:
                continue  # skip Wilcoxon for this target because mcc is same over all splits


            vs_long = vs_df.melt(var_name='method', value_name='value')
            posthoc = sp.posthoc_wilcoxon(vs_long, group_col='method', val_col='value', p_adjust='bonferroni')
            wilcox_posthoc_results[metric][target] = posthoc

    for target, df in wilcox_posthoc_results["kendalls_tau"].items():
        if df.isnull().values.any():
            print(f"NaN found in wilcox_posthoc_results['kendalls_tau'] for target: {target}")

    return wilcox_posthoc_results

#--------------------------------------------------------------------------------------------------------------------------------------

def sign_plot(
    results_all,
    wilcox_posthoc_results,
    DISPLAY_ORDER,
    target_counts_path="/../../0_DATA/N_COMPOUNDS_PER_TARGET.json",
    metrics=None,
    targets_to_skip=None,
    n_cols=4,
    arrow_threshold=0.05,
    alpha=0.9,
):
    
    # Names to be plotted of methods
    NAME_MAP = {
        "Pycomsia": "PYC."

    }
    
    # Use n_cols subplots per row
    n_cols = n_cols
    arrow_threshold = arrow_threshold  # threshold for arrow annotation
    methods = DISPLAY_ORDER
    #methods.reverse()

    # Define Targets (in order of datapoint counts)
    with open(target_counts_path, "r") as f:
        target_counts = json.load(f)

    targets = sorted(
        (t for t in next(iter(results_all.values())).keys() if t not in targets_to_skip),
        key=lambda x: (target_counts.get(x, float("inf")), 0 if x == "pde10" else 1 if x == "p38" else 2),
    )

    # ensure 'tyk' then 'cdk 2' appear first if present
    priority = ["tyk2", "cdk 2"]
    targets = [t for t in priority if t in targets] + [t for t in targets if t not in priority]

    n_targets = len(targets)
    n_rows = math.ceil(n_targets / n_cols)

    for metric in metrics:

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_targets > 1 else [axes]

        for i, target in enumerate(targets):
            
            if target not in wilcox_posthoc_results[metric]:
                print(f"Skipping {target} for {metric} (no Wilcoxon results)")
                axes[i].axis("off")
                continue

            # Build wide-format DataFrame from the split-level data for this metric and target.
            data = {}
            for method in methods:
                splits = sorted(results_all[method][target].keys())
                vals = []
                for s in splits:
                    if metric not in results_all[method][target][s]:
                        print(f"Missing metric {metric} → target {target} → method {method} → split {s}")
                        raise KeyError(metric)
                    vals.append(results_all[method][target][s][metric])
                data[method] = vals
            vs_df = pd.DataFrame(data)[methods]

            # Get the posthoc two-sided p-value matrix for this target and metric.
            posthoc_df = wilcox_posthoc_results[metric][target]

            # Plot the significance matrix using sign_plot.
            posthoc_df = posthoc_df.loc[methods, methods]
            renamed_df = posthoc_df.rename(index=NAME_MAP, columns=NAME_MAP)
            sp.sign_plot(renamed_df, alpha=alpha, ax=axes[i])

            posthoc_df = posthoc_df.loc[methods, methods]
            
            # enlarge method labels
            axes[i].tick_params(axis="x", labelsize=13)  # adjust as needed
            axes[i].tick_params(axis="y", labelsize=13)

            # bold target name with extra top padding
            axes[i].set_title(
                f"{target} (n={target_counts.get(target, '?')})",
                fontsize=15,
                pad=10,  # increase as needed
    )

            # For directional annotation using one-sided tests.
            for idx_row in range(len(methods)):
                for idx_col in range(idx_row + 1, len(methods)):
                    two_sided_p = posthoc_df.loc[methods[idx_row], methods[idx_col]]
                    if two_sided_p < arrow_threshold:
                        # Test if method at row is greater than method at column.
                        stat1, p1 = wilcoxon(
                            vs_df[methods[idx_row]],
                            vs_df[methods[idx_col]],
                            alternative="greater",
                        )
                        if p1 < arrow_threshold:
                            if metric in {"rmse", "slope_diff"}:
                                arrow_symbol = "↓"
                            else:
                                arrow_symbol = "←" # row method significantly greater than column method

                        else:
                            # Otherwise test the opposite alternative.
                            stat2, p2 = wilcoxon(
                                vs_df[methods[idx_col]],
                                vs_df[methods[idx_row]],
                                alternative="greater",
                            )
                            if p2 < arrow_threshold:
                                if metric in {"rmse", "slope_diff"}:
                                    arrow_symbol = "←"
                                else:
                                    arrow_symbol = "↓"  # column method significantly greater than row method
                            else:
                                arrow_symbol = ""
                        axes[i].annotate(
                            arrow_symbol,
                            (idx_col + 0.5, idx_row + 0.5),
                            color="black",
                            fontsize=16,
                            ha="center",
                            va="center",
                        )

        # Turn off unused subplots.
        for j in range(i + 1, n_rows * n_cols):
            axes[j].axis("off")

        # Print title before showing the plot instead of using fig.suptitle
        print(f"{metric} significance by target")
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(right=0.92)
        plt.show()
