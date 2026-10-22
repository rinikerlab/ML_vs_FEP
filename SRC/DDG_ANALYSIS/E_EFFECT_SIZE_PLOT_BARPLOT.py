""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

def plot_effectsize_barplot(
    results_all, DISPLAY_ORDER, ALPHA=None, METRICS=None,
    target_counts_path="../0_DATA/N_COMPOUNDS_PER_TARGET.json",
    show_titles=False, 
    COUNT_EFFECTS=False,
    prefix="",
    fusion=None    
):
    """
    Inputs:
    - results_all: dict of methods -> targets -> splits -> stats dict
    - DISPLAY_ORDER: list of methods in desired display order
    - ALPHA: float significance threshold
    - METRICS: list of metrics
    - target_counts_path: str path to json
    - show_titles: bool draw title
    - COUNT_EFFECTS: bool compute effect size counts
    Output:
    - None
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    import numpy as np
    import json

    # variable placeholders
    alpha_scatter = 0.8
    scatter_size  = 60
    jitter        = 0.1
    METRICS       = METRICS
    palette       = sns.color_palette("colorblind")
    BEST_COLOR    = palette[0]
    EQUIV_COLOR   = "grey"
    SIG_COLOR     = palette[1]
    METHODS       = sorted(results_all.keys())

    NAME_MAP = {"p2": "bace_p2", "hunt": "bace hunt"}

    DISPLAY_ORDER.reverse()

    # effect-size thresholds
    EFFECT_THRESHOLDS = {
        "rmse": 0.4,
        "kendalls_tau": 0.2,
        "window_tau": 0.2,
        "slope_diff": 0.25,
        "mcc": 0.2,
        "spearman": 0.2,
    }

    ticks = {
        "rmse": [0.0,1.0,2.0,3.0,4.0],
        "kendalls_tau": [-0.25,0,0.25,0.5,0.75,1],
        "window_tau": [-0.25,0,0.25,0.5,0.75,1],
        "slope_diff": [0.0,0.25,0.5,0.75,1,1.25],
        "mcc": [-0.25,0.0,0.25,0.5,0.75,1],
        "spearman": [-0.25,0,0.25,0.5,0.75,1],
    }
    lims = {
        "rmse": (0.0,4),
        "kendalls_tau": (-0.25,1.0),
        "window_tau": (-0.25,1.0),
        "slope_diff": (0.0,1.25),
        "mcc": (-0.25,1),
        "spearman": (-0.25,1.0),
    }

    titles = {
        "rmse": "RMSE [kcal/mol]",
        "kendalls_tau": "Kendalls tau",
        "window_tau": "Window tau",
        "slope_diff": "slope minus 1",
        "mcc": "MCC",
        "spearman": "Spearman rho",
    }
    xlabels = titles

    SKIP_TARGETS = {"galectin","p2","thrombin"}

    with open(target_counts_path) as f:
        target_counts = json.load(f)

    # flatten results
    records = []
    for m in METHODS:
        for t in sorted(results_all[m]):
            for s, stats in results_all[m][t].items():
                for met in METRICS:
                    records.append({
                        "method": m,
                        "target": t,
                        "split":  s,
                        "metric": met,
                        "value":  stats[met]
                    })
    df = pd.DataFrame(records)

    sorted_targets = [
        t for t in sorted(target_counts, key=lambda x: target_counts[x])
        if t not in SKIP_TARGETS
    ]

    # ---------------------------------------------------------------
    # grouped barplot
    # ---------------------------------------------------------------

    for met in METRICS:

        if COUNT_EFFECTS:
            effect_counts = {m: [] for m in METHODS}

        all_means = []
        all_colors = []

        for tgt in sorted_targets:

            sub = df[(df.metric == met) & (df.target == tgt)]
            if sub.empty:
                continue
            

            common = sorted(set.intersection(*[
                set(sub.query("method == @m").split) for m in METHODS
            ]))
            if len(common) <= 95:
                raise ValueError("too many splits removed: less than 95")

            d = sub[sub.split.isin(common)].copy()

            mean_vals = d.groupby("method").value.mean()

            best_all = mean_vals.idxmin() if met in ["rmse","slope_diff"] else mean_vals.idxmax()

            if fusion is not None:
                fusion_means = mean_vals[mean_vals.index.isin(fusion)]
                best = fusion_means.idxmin() if met in ["rmse","slope_diff"] else fusion_means.idxmax()
            else:
                best = best_all

            best_mean = mean_vals[best]

            if COUNT_EFFECTS:
                thresh = EFFECT_THRESHOLDS.get(met,0)
                for m in METHODS:
                    if m == best:
                        continue
                    if met in ["rmse","slope_diff"]:
                        eff = mean_vals[m] - best_mean
                    else:
                        eff = best_mean - mean_vals[m]
                    if eff > thresh:
                        effect_counts[best].append(tgt)
                        break

            colors = {}
            base = d.query("method == @best_all").set_index("split").value
            for m in DISPLAY_ORDER:
                if m == best_all:
                    colors[m] = BEST_COLOR
                    continue
                other = d.query("method == @m").set_index("split").value
                _, p = wilcoxon(base, other, zero_method="wilcox")
                colors[m] = SIG_COLOR if p < ALPHA else EQUIV_COLOR

            all_means.append(mean_vals)
            all_colors.append(colors)

        # -----------------------------------------------------------
        # single grouped barplot
        # -----------------------------------------------------------

        n_targets = len(all_means)
        n_methods = len(DISPLAY_ORDER)

        x_pos = np.arange(n_targets) * 1.4
        bar_width = 0.9 / n_methods

        plt.figure(figsize=(max(10, n_targets * 0.6), 4))

        for i in range(n_targets):
            mv = all_means[i]
            cols = all_colors[i]
            for j, m in enumerate(DISPLAY_ORDER):
                if m in mv:
                    plt.bar(
                        x_pos[i] + j*bar_width,
                        mv[m],
                        width=bar_width,
                        color=cols[m],
                        alpha=alpha_scatter,
                        edgecolor="black",       # <<< black outlines
                        linewidth=0.6            # <<< black outlines
                    )



        xticks = [NAME_MAP.get(t, t) for t in sorted_targets]

        plt.xticks(x_pos + 0.45, xticks, rotation=45, ha="right", fontsize=12)
        plt.ylabel(xlabels[met], fontsize=14, fontweight='bold')
        plt.ylim(*lims[met])
        plt.yticks(ticks[met], fontsize=12)
        plt.grid(True, axis="y", ls="--", lw=0.5)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        if met == "rmse":
            base = ticks["rmse"]                    # major
            mids = [(a+b)/2 for a,b in zip(base[:-1], base[1:])]  # minor

            ax = plt.gca()
            ax.set_yticks(base)                     # major with labels
            ax.set_yticks(mids, minor=True)         # minor without labels


        if show_titles:
            plt.title(titles[met])

        plt.tight_layout()
        plt.savefig(prefix + met + ".png", dpi=600, bbox_inches="tight")
        plt.show()

        if COUNT_EFFECTS:
            print("\nEffect-size counts for metric", met)
            for m in METHODS:
                lst = [t for t in sorted_targets if t in effect_counts[m]]
                print("-", m, ":", len(lst), "targets ->", lst)
