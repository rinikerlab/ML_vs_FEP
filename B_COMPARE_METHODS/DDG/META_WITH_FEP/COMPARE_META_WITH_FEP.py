"""
Effect-size dot plots including the meta-learner that takes the FEP prediction
as an additional input feature (see A_MODELS/Z_META_FEP/).

Produces the same figure style as the main-text comparisons: per-target mean of
the metric over the 100 splits, best method in blue, methods significantly worse
than the best in orange (Wilcoxon signed-rank, p < 0.01), the rest in grey.

"""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))      # repo root
sys.path.insert(0, os.path.join(ROOT, "SRC"))

from DDG_ANALYSIS.B_PREPROCESS import preprocess                   # noqa: E402
from DDG_ANALYSIS.E_EFFECT_SIZE_PLOT import plot_significance_ranks  # noqa: E402

A = os.path.join(ROOT, "A_MODELS")


FF = (sys.argv[1] if len(sys.argv) > 1 else "OFF").upper()
if FF not in ("OFF", "OPLS"):
    sys.exit("usage: python COMPARE_META_WITH_FEP.py [OFF|OPLS]")
FF_LABEL = {"OFF": "OFF", "OPLS": "OPLS3e"}[FF]
META_LABEL = f"Meta-learner+ ({FF_LABEL})"

files = {
    "RFR 2D":                   f"{A}/2D/RANDOM/B_DDG_FROM_DG/ddG_pred_&_error_all_splits.json",
    "RFR 3D":                   f"{A}/3D/RANDOM/B_DDG_FROM_DG/ddG_pred_&_error_all_splits.json",
    "Pycomsia":                 f"{A}/PYC/RANDOM/C_DDG_FROM_DG/ddG_pred_&_error_all_splits.json",
    "Pycomsia (MIF)":           f"{A}/MIF/RANDOM/C_DDG_FROM_DG/ddG_pred_&_error_all_splits.json",
    "FEP (OFF)":                f"{A}/FEP/DDG/OFF/OFF_RANDOM_ddG_FROM_FEP_GRAPH_pred&error.json",
    "FEP (OPLS3e)":             f"{A}/FEP/DDG/OPLS/OPLS_RANDOM_ddG_FROM_FEP_GRAPH_pred&error.json",
    "Meta-learner":             f"{A}/Z_META/DDG/RANDOM/ddG_ridge_LOO_pred&error.json",
    META_LABEL:                 f"{A}/Z_META_FEP/DDG/RANDOM/{FF}/ddG_ridge_with_{FF}_LOO_pred&error.json",
}


DISPLAY_ORDER = ["FEP (OPLS3e)", "FEP (OFF)",
                 META_LABEL, "Meta-learner",
                 "Pycomsia (MIF)", "Pycomsia", "RFR 3D", "RFR 2D"]

TARGETS_TO_SKIP = ("galectin", "thrombin", "p2") # targets with too few compounds (as mentioned in paper), these are not included in the main-text comparisons
ALPHA = 0.01
METRICS = ["rmse", "window_tau", "slope_diff"]

with contextlib.redirect_stdout(io.StringIO()):     # preprocess() is verbose
    results_all, _ = preprocess(files, TARGETS_TO_SKIP)

plot_significance_ranks(
    results_all, DISPLAY_ORDER, ALPHA=ALPHA, METRICS=METRICS,
    target_counts_path=os.path.join(ROOT, "0_DATA", "N_COMPOUNDS_PER_TARGET.json"),
    COUNT_EFFECTS=True, prefix=f"META_{FF}_",
)
