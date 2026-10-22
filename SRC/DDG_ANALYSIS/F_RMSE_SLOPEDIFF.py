""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

def plot_RMSE_slope(results_all, DISPLAY_ORDER, ALPHA=None, METRICS=None, target_counts_path="0_DATA/N_COMPOUNDS_PER_TARGET.json", show_titles=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import wilcoxon
    import numpy as np
    import json
    print("DDD")

    # placeholders
    METRICS       = ["rmse", "slope_diff"]
    plot_alpha    = 0.7
    palette       = sns.color_palette('colorblind')
    BEST_COLOR    = palette[0]  # blue
    EQUIV_COLOR   = "grey"
    SIG_COLOR     = palette[1]  # green
    METHODS       = sorted(results_all.keys())
    
    
    ticks = {
        "rmse": [1, 2 ,3 , 4],
        "slope_diff": [0.25, 0.5, 0.75, 1, 1.25],
    }
    
    lims = {
        "rmse": (0.0, 2.0),
        "slope_diff": (0.0, 1.25),
    }
    
    titles = {
        "rmse": "RMSE [kcal/mol]",
        "slope_diff": "|slope − 1|",
    }
    
    xlabels = {
        "rmse": "RMSE [kcal/mol]",
        "slope_diff": "|slope − 1|",
    }

    # Load FEP metrics & target counts
    with open(target_counts_path) as f:
        target_counts = json.load(f)

    # Build tidy dataframe
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

    # Significance plots (both metrics per subplot)
    t_data = []
    for tgt, sub in df.groupby("target"):
        common = sorted(set.intersection(*[
            set(sub.query("method == @m").split) for m in METHODS]))
        if len(common) > 95:
            t_data.append((tgt, sub[sub.split.isin(common)].copy(), len(common)))
        else:
            raise ValueError(f"too many splits removed: less than 95 splits for target {tgt}")
    if not t_data:
        return

    t_data.sort(key=lambda x: target_counts.get(x[0], float("inf")))
    n_tgt = len(t_data)
    n_col, n_row = 4, int(np.ceil(n_tgt / 4))
    plt.style.use('tableau-colorblind10')
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(3 * n_col, max(3, 0.3 * len(DISPLAY_ORDER)) * n_row)
    )
    if show_titles:
        fig.suptitle("RMSE & |slope − 1|", y=0.98, fontsize=12)

    axes = axes.flatten()
    for ax in axes[n_tgt:]:
        ax.axis("off")

    for i, (tgt, d, _) in enumerate(t_data):
        ax = axes[i]
        means = d.groupby(["metric","method"]).value.mean().unstack("metric")
        best_r  = means["rmse"].idxmin()
        best_s  = means["slope_diff"].idxmin()

        base_r = d.query("metric=='rmse' and method==@best_r").set_index("split").value
        base_s = d.query("metric=='slope_diff' and method==@best_s").set_index("split").value
        colors_r, colors_s = {}, {}
        for m in DISPLAY_ORDER:
            if m == best_r:
                colors_r[m] = BEST_COLOR
            else:
                _, p = wilcoxon(base_r, d.query("metric=='rmse' and method==@m").set_index("split").value, zero_method="wilcox")
                colors_r[m] = SIG_COLOR if p < ALPHA else EQUIV_COLOR
            if m == best_s:
                colors_s[m] = BEST_COLOR
            else:
                _, p = wilcoxon(base_s, d.query("metric=='slope_diff' and method==@m").set_index("split").value, zero_method="wilcox")
                colors_s[m] = SIG_COLOR if p < ALPHA else EQUIV_COLOR

        y_pos = np.arange(1, len(DISPLAY_ORDER)+1)
        # bottom axis: RMSE (circle-marker)
        for y, m in zip(y_pos, DISPLAY_ORDER):
            if m in means.index:
                ax.scatter(
                    means.at[m,"rmse"], y,
                    color=colors_r[m],
                    s=60, marker='o',
                    alpha=plot_alpha
                )
        ax.set_xlim(*lims["rmse"])
        ax.set_xticks(ticks["rmse"])
        ax.set_xlabel(xlabels["rmse"])
        ax.axvline(means.at[best_r,"rmse"], ls="--", color="black", lw=1)

        # top axis: slope_diff (cross-marker)
        ax2 = ax.twiny()
        for y, m in zip(y_pos, DISPLAY_ORDER):
            if m in means.index:
                ax2.scatter(
                    means.at[m,"slope_diff"], y,
                    color=colors_s[m],
                    s=60, marker='x',
                    alpha=plot_alpha
                )
        ax2.set_xlim(*lims["slope_diff"])
        ax2.set_xticks(ticks["slope_diff"])
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.set_ticks_position('top')
        ax2.set_xlabel(xlabels["slope_diff"], labelpad=10)
        ax2.axvline(means.at[best_s,"slope_diff"], ls="--", color="black", lw=1)

        # y-axis
        labels = [m if m!="Pycomsia" else "PYC." for m in DISPLAY_ORDER]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title(
            f"{tgt} (n={target_counts.get(tgt,'?')})",
            pad=12
        )

    # global legend outside on right
    handles = [
        Line2D([0],[0], marker='o', color='black', linestyle='None', label=titles["rmse"]),
        Line2D([0],[0], marker='x', color='black', linestyle='None', label=titles["slope_diff"])
    ]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout(rect=[0,0,0.9,0.98])
    plt.show()
