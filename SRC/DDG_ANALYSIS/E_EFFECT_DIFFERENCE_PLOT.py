""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

def plot_differences(
    results_all, DISPLAY_ORDER, ALPHA=None, METRICS=None,
    target_counts_path="../0_DATA/N_COMPOUNDS_PER_TARGET.json",
    show_titles=True, REF_METHOD=None
):
    """
    Inputs:
    - REF_METHOD: str, reference method name
    Output:
    - Barplots of Δ mean(metric) vs REF_METHOD for each target
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    import numpy as np
    import json

    # vars
    alpha_bar = 0.8
    bar_width = 0.7
    palette = sns.color_palette("colorblind")
    EQUIV_COLOR = "grey"
    SIG_COLOR = palette[1]
    METHODS = sorted(results_all.keys())
    DISPLAY_ORDER.reverse()

    ticks = {
        "rmse": [-2, -1, 0, 1, 2],
        "kendalls_tau": [-1, -0.5, 0, 0.5, 1],
        "window_tau": [-1, -0.5, 0, 0.5, 1],
        "slope_diff": [-1, -0.5, 0, 0.5, 1],
        "mcc": [-1, -0.5, 0, 0.5, 1],
        "spearman": [-1, -0.5, 0, 0.5, 1],
    }
    lims = {k: (-1, 1) for k in ticks}
    titles = {
        "rmse": "ΔRMSE vs ref",
        "kendalls_tau": "ΔKendall’s τ vs ref",
        "window_tau": "ΔWindow τ vs ref",
        "slope_diff": "Δ|slope−1| vs ref",
        "mcc": "ΔMCC vs ref",
        "spearman": "ΔSpearman’s ρ vs ref",
    }

    # read counts
    with open(target_counts_path) as f:
        target_counts = json.load(f)

    # dataframe
    records = []
    for m in METHODS:
        for t in sorted(results_all[m]):
            for s, stats in results_all[m][t].items():
                for met in METRICS:
                    records.append({
                        "method": m,
                        "target": t,
                        "split": s,
                        "metric": met,
                        "value": stats[met]
                    })
    df = pd.DataFrame(records)

    SKIP_TARGETS = {"galectin", "p2", "thrombin"}
    sorted_targets = [
        t for t in sorted(target_counts, key=lambda x: target_counts[x])
        if t not in SKIP_TARGETS
    ]

    for met in METRICS:
        n_tgt = len(sorted_targets)
        n_col, n_row = 5, int(np.ceil(n_tgt / 4))
        fig, axes = plt.subplots(
            n_row, n_col,
            figsize=(3.5 * n_col, 3 * n_row)
        )
        if show_titles:
            fig.suptitle(titles[met], y=0.98, fontsize=12)
        axes = axes.flatten()
        for ax in axes[n_tgt:]:
            ax.axis("off")

        for i, tgt in enumerate(sorted_targets):
            ax = axes[i]
            sub = df[(df.metric == met) & (df.target == tgt)]
            if sub.empty or REF_METHOD not in sub.method.values:
                ax.axis("off")
                continue

            common = sorted(set.intersection(*[
                set(sub.query("method == @m").split) for m in METHODS
            ]))
            if len(common) <= 95:
                raise ValueError(f"too many splits removed: <95 for target {tgt}")
            d = sub[sub.split.isin(common)].copy()

            mean_vals = d.groupby("method").value.mean()
            if REF_METHOD not in mean_vals:
                ax.axis("off")
                continue
            ref_mean = mean_vals[REF_METHOD]
            mean_diffs = mean_vals - ref_mean

            base = d.query("method == @REF_METHOD").set_index("split").value
            diffs = []
            colors = []

            for m in DISPLAY_ORDER:
                if m == REF_METHOD or m not in mean_diffs:
                    continue
                other = d.query("method == @m").set_index("split").value
                if met in ["rmse", "slope_diff"]:
                    _, p = wilcoxon(base, other, alternative="less", zero_method="wilcox")
                else:
                    _, p = wilcoxon(base, other, alternative="greater", zero_method="wilcox")
                colors.append(SIG_COLOR if p < ALPHA else EQUIV_COLOR)
                diffs.append(mean_diffs[m])

            x_pos = np.arange(len(diffs))
            ax.bar(
                x_pos, diffs, color=colors,
                alpha=alpha_bar, width=bar_width
            )

            ax.axhline(0, color="black", lw=1)
            ax.set_ylim(*lims[met])
            ax.set_yticks(ticks[met])
            ax.set_xticks(x_pos)
            labels = [m if m != "Pycomsia" else "PYC." for m in DISPLAY_ORDER if m != REF_METHOD]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            if i % n_col != 0:
                ax.set_ylabel("")
            ax.set_title(f"{tgt} (n={target_counts.get(tgt, '?')})", fontsize=10)
            ax.grid(True, axis="y", ls="--", lw=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
