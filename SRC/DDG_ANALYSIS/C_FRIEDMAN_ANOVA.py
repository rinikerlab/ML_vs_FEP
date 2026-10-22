""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

from scipy.stats import friedmanchisquare

def run_friedman(results_all, metrics, TARGETS_TO_SKIP=None):
    if TARGETS_TO_SKIP is None:
        TARGETS_TO_SKIP = []

    # check if split keys in '2D'-'bace' are sorted
    if 0:
        a = sorted(results_all["2D"]["bace"].keys()) 
        b = list(results_all["2D"]["bace"].keys())
        if not a == b:
            raise ValueError("Fail in sanity check: Keys in results_all['2D']['bace'] are not sorted correctly.")

    # init result container: one dict per metric
    friedman_results = {metric: {} for metric in metrics}

    # get list of targets, excluding skip list
    targets = sorted(
        t for t in next(iter(results_all.values())).keys()
        if t not in TARGETS_TO_SKIP
    )

    for metric in metrics:
        # skip mcc (there is a signifcant difference)
        if metric == "mcc":
            continue
        for target in targets:
            # collect metric values for each method over sorted splits
            method_values = [
                [results_all[method][target][s][metric] for s in sorted(results_all[method][target].keys())]
                for method in results_all
            ]
            # run Friedman test across methods
            stat, p_val = friedmanchisquare(*method_values)

            # store test result
            friedman_results[metric][target] = {
                "statistic": float(stat),
                "p_value": float(p_val)
            }

    # return all test results
    return friedman_results
