""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

def plot_significance_ranks(
    results_all, DISPLAY_ORDER, ALPHA=None, METRICS=None,
    target_counts_path="../0_DATA/N_COMPOUNDS_PER_TARGET.json",
    show_titles=False, 
    COUNT_EFFECTS=False,
    prefix="",
    fusion=None    
):
    """
    Inputs:
    - results_all: dict of methods → targets → splits → stats dict
    - DISPLAY_ORDER: list of methods in desired display order
    - ALPHA: float, significance threshold for Wilcoxon test
    - METRICS: list of metric names to plot
    - target_counts_path: str, path to JSON with target counts
    - show_titles: bool, whether to show super-title per metric
    - COUNT_EFFECTS: bool, whether to count targets where best beats others by threshold
    Output:
    - None (displays matplotlib figures; prints counts if COUNT_EFFECTS is on)
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    import numpy as np
    import json

    # ===== STYLING PARAMETERS =====
    from matplotlib.ticker import FuncFormatter

    # Formatter to remove annoying trailing zeros from tick labels
    def format_ticks(x, pos):
        if x == int(x):
            return f'{int(x)}'
        else:
            return f'{x:.2g}'.rstrip('0').rstrip('.')

    tick_formatter = FuncFormatter(format_ticks)

    # Scatter plot
    alpha_scatter      = 0.8
    scatter_size       = 180

    # Subplot layout
    n_col_subplots     = 5
    figsize_width_mult = 5.4     # multiplier for width: figsize_width = figsize_width_mult * n_col
    figsize_height_mult= 1     # multiplier for height: figsize_height = figsize_height_mult * len(methods) * n_row
    hspace_between     = 0.31   
    wspace_between     = 0.25      
    bottom_margin      = 0.09     # bottom margin for x-axis label
    xlabel_y_pos       = 0.03     # y position of x-axis label
    suptitle_y_pos     = 0.98     # y position of suptitle

    # Font sizes
    fontsize_title     = 26      # subplot titles
    fontsize_xlabel    = 32      # x-axis label
    fontsize_ylabel    = 26      # y-axis: method names
    fontsize_tickmark  = 19      # x-axis: tick numbers
    fontsize_suptitle  = 18       # overall figure title
    title_pad          = 15      # space between subplot and title
    xticklabels_bold   = True    

    # Tick mark sizes
    ticklength         = 7      
    tickwidth          = 1.3     

    # Line styles and widths
    linewidth_box      = 1.5      # box/axis line width
    linewidth_grid     = 1.5      # grid line width
    linestyle_grid     = "--"     # grid line style
    linewidth_axvline  = 1.8     # vertical line (best model) width
    linestyle_axvline  = "--"     # vertical line (best model) style

    # Figure saving
    dpi_save           = 300
    pad_inches_save    = 0

    # ===== END STYLING PARAMETERS =====

    jitter             = 0.1
    METRICS            = METRICS
    palette            = sns.color_palette('colorblind')
    BEST_COLOR         = palette[0]
    EQUIV_COLOR        = "grey"
    SIG_COLOR          = palette[1]
    METHODS            = sorted(results_all.keys())

    NAME_MAP = {"p2": "bace_p2", "hunt": "bace hunt"}

    DISPLAY_ORDER.reverse()

    # effect-size thresholds per metric
    EFFECT_THRESHOLDS = {
        "rmse": 0.4,
        "kendalls_tau": 0.2,
        "window_tau": 0.2,
        "slope_diff": 0.25,
        "mcc": 0.2,
        "spearman": 0.2,          
    }

    # axis settings
    ticks = {
        "rmse": [0.0, 1.0, 2.0, 3.0, 4.0],
        "kendalls_tau": [-0.25, 0, 0.25, 0.5, 0.75, 1],
        "window_tau": [-0.25 ,0, 0.25, 0.5, 0.75, 1],
        "slope_diff": [0.0, 0.25, 0.5, 0.75, 1, 1.25],
        "mcc": [-0.25 ,0.0, 0.25, 0.5, 0.75, 1],
        "spearman": [-0.25, 0, 0.25, 0.5, 0.75, 1],   
    }
    lims = {
        "rmse": (0.0, 4),
        "kendalls_tau": (-0.25, 1.0),
        "window_tau": (-0.25, 1.0),
        "slope_diff": (0.0, 1.25),
        "mcc": (-0.25, 1),
        "spearman": (-0.25, 1.0),   
    }
    
    titles = {
        "rmse": "RMSE [kcal/mol]",
        "kendalls_tau": "Kendall’s τ",
        "window_tau": "Window τ",
        "slope_diff": "|slope − 1|",
        "mcc": "MCC",
        "spearman": "Spearman’s ρ",  
    }
    xlabels = titles  # reuse titles

    SKIP_TARGETS = {"galectin", "p2", "thrombin"}

    with open(target_counts_path) as f:
        target_counts = json.load(f)

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

    for met in METRICS:
        if COUNT_EFFECTS:
            effect_counts = {m: [] for m in METHODS}

        n_tgt = len(sorted_targets)
        n_col = n_col_subplots
        n_row = int(np.ceil(n_tgt / n_col))
        plt.style.use('tableau-colorblind10')
        fig, axes = plt.subplots(
            n_row, n_col,
            figsize=(figsize_width_mult * n_col, figsize_height_mult * len(DISPLAY_ORDER) * n_row),
            constrained_layout=False
        )
        if show_titles:
            fig.suptitle(titles[met], y=0.98, fontsize=fontsize_suptitle)
        axes = axes.flatten()
        for ax in axes[n_tgt:]:
            ax.axis("off")

        for i, tgt in enumerate(sorted_targets):
            ax = axes[i]
            sub = df[(df.metric == met) & (df.target == tgt)]
            if sub.empty:
                ax.axis("off")
                continue

            common = sorted(set.intersection(*[
                set(sub.query("method == @m").split) for m in METHODS
            ]))
            if len(common) < 95:
                print(f"{len(common)} splits left")
                raise ValueError(f"too many splits removed: <95 for target {tgt}")
            d = sub[sub.split.isin(common)].copy()

            mean_vals = d.groupby("method").value.mean()
            
            # best model for coloring point blue
            best_all = mean_vals.idxmin() if met in ["rmse","slope_diff"] else mean_vals.idxmax()
            
            # set dashed black line for best model
            if fusion is not None:
                # only use methods in FUSION list
                fusion_means = mean_vals[mean_vals.index.isin(fusion)]
                best = fusion_means.idxmin() if met in ["rmse","slope_diff"] else fusion_means.idxmax()
            else:
                # normal behaviour
                best = mean_vals.idxmin() if met in ["rmse","slope_diff"] else mean_vals.idxmax()

            best_mean = mean_vals[best]

            if COUNT_EFFECTS:
                thresh = EFFECT_THRESHOLDS.get(met, 0)
                for m in METHODS:
                    if m == best:
                        continue
                    if met in ["rmse", "slope_diff"]:
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

            y_pos = np.arange(1, len(DISPLAY_ORDER) + 1)
            for y, m in zip(y_pos, DISPLAY_ORDER):
                if m in mean_vals:
                    ax.scatter(
                        mean_vals[m], y,
                        color=colors[m],
                        s=scatter_size,
                        alpha=alpha_scatter
                    )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(DISPLAY_ORDER, fontsize=fontsize_ylabel)
            for label in ax.get_yticklabels():
                label.set_fontsize(fontsize_ylabel)

            # when 1: remove y axis ticks on plots that are not in the verry left column
            if 1:
                if i % n_col != 0:
                    #ax.set_yticks([])
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

            ax.axvline(best_mean, ls=linestyle_axvline, color="black", lw=linewidth_axvline)
            ax.set_xlim(*lims[met])
            ax.set_xticks(ticks[met])
            ax.xaxis.set_major_formatter(tick_formatter)
            for label in ax.get_xticklabels():
                label.set_fontsize(fontsize_tickmark)
                label.set_fontweight('bold' if xticklabels_bold else 'normal')
            disp_name = NAME_MAP.get(tgt, tgt)
            ax.set_title(f"{disp_name} (n={target_counts.get(tgt, '?')})", fontweight='bold', fontsize=fontsize_title, pad=title_pad)
            ax.grid(True, axis="x", ls=linestyle_grid, lw=linewidth_grid)
            ax.tick_params(axis='both', length=ticklength, width=tickwidth)
            for spine in ax.spines.values():
                spine.set_linewidth(linewidth_box)

        fig.subplots_adjust(bottom=bottom_margin, hspace=hspace_between, wspace=wspace_between)
        fig.text(0.5, xlabel_y_pos, xlabels[met], ha='center', fontweight='bold', fontsize=fontsize_xlabel)
        fig.savefig(prefix + met + ".png", dpi=dpi_save, pad_inches=pad_inches_save, bbox_inches='tight')
        plt.show()

        if COUNT_EFFECTS:
            print(f"\nEffect-size counts for metric '{met}':")
            for m in METHODS:
                targets_list = [t for t in sorted_targets if t in effect_counts[m]]
                print(f"- {m}: {len(targets_list)} targets → {targets_list}")

#------------------------------------------------------------------------------------------------------------------------------
