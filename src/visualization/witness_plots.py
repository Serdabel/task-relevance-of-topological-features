"""Figures for witness statistics (extracted from mass_training.ipynb).

All functions consume the tables produced by ``analysis.witness_stats`` and
save the paper's figures under analysis/results/figures/.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from analysis.witness_stats import DEFAULT_DATASET_LABELS


@contextmanager
def save_figure_instead_of_show(
    path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    close=True,
):
    """Temporarily redirects plt.show() to fig.savefig(path).

    Useful for saving figures produced by plotting functions that call
    plt.show() internally.
    """
    old_show = plt.show

    def _save_and_optionally_close(*args, **kwargs):
        fig = plt.gcf()
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor=facecolor,
        )
        if close:
            plt.close(fig)

    plt.show = _save_and_optionally_close

    try:
        yield
    finally:
        plt.show = old_show


def plot_layerwise_witness_bars(
    layer_summary,
    dataset_types,
    feature_names,
    dataset_labels=None,
    feature_labels=None,
    title=None,
    save_path=None,
    figsize=None,
    y_max=None,
):
    """
    Plots layer-wise mean structural witness counts with bootstrap CI.

    Parameters
    ----------
    layer_summary:
        output tables["layer_summary"] from build_witness_aggregate_tables.

    dataset_types:
        list of datasets to show as columns.

    feature_names:
        list of features to show as rows.

    dataset_labels:
        optional dict dataset_type -> short label.

    feature_labels:
        optional dict feature_name -> label.

    save_path:
        if not None, saves the figure.

    Returns
    -------
    fig, axes
    """

    dataset_labels = dataset_labels or DEFAULT_DATASET_LABELS
    feature_labels = feature_labels or {}

    n_rows = len(feature_names)
    n_cols = len(dataset_types)

    if figsize is None:
        figsize = (4.2 * n_cols, 3.4 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )

    # Global y-limit, so panels are comparable.
    selected = layer_summary[
        layer_summary["dataset_type"].isin(dataset_types)
        & layer_summary["feature_name"].isin(feature_names)
    ].copy()

    if y_max is None:
        if len(selected) > 0:
            y_max = selected["ci_high_witness_count"].max()
            if pd.isna(y_max):
                y_max = selected["mean_witness_count"].max()
            y_max = max(1.0, float(y_max) * 1.15)
        else:
            y_max = 1.0

    for row_idx, feature_name in enumerate(feature_names):
        for col_idx, dataset_type in enumerate(dataset_types):
            ax = axes[row_idx, col_idx]

            cur = layer_summary[
                (layer_summary["dataset_type"] == dataset_type)
                & (layer_summary["feature_name"] == feature_name)
            ].copy()

            if len(cur) == 0:
                ax.set_title(f"{dataset_labels.get(dataset_type, dataset_type)}\nno data")
                ax.set_axis_off()
                continue

            cur = cur.sort_values("layer_id")

            x = np.arange(len(cur))
            y = cur["mean_witness_count"].to_numpy(dtype=float)

            ci_low = cur["ci_low_witness_count"].to_numpy(dtype=float)
            ci_high = cur["ci_high_witness_count"].to_numpy(dtype=float)

            yerr_lower = np.maximum(y - ci_low, 0.0)
            yerr_upper = np.maximum(ci_high - y, 0.0)

            ax.bar(
                x,
                y,
                yerr=np.vstack([yerr_lower, yerr_upper]),
                capsize=4,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(cur["layer_id"].astype(int).tolist())

            ax.set_ylim(0, y_max)
            ax.grid(axis="y", alpha=0.3)

            dataset_label = dataset_labels.get(dataset_type, dataset_type)
            feature_label = feature_labels.get(feature_name, feature_name)

            if row_idx == 0:
                ax.set_title(dataset_label, fontsize=13)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{feature_label}\nmean witness count",
                    fontsize=11,
                )

            if row_idx == n_rows - 1:
                ax.set_xlabel("Layer")

            # Put n_models in the panel.
            n_models = int(cur["n_models"].max())
            ax.text(
                0.98,
                0.95,
                f"n={n_models}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

    if title is not None:
        fig.suptitle(title, fontsize=15, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, axes


def plot_total_witness_summary(
    pair_summary,
    dataset_types,
    feature_name,
    dataset_labels=None,
    title=None,
    save_path=None,
    figsize=(7, 4),
):
    """
    Plots total witness count aggregated over all layers.
    One bar per dataset.
    """

    dataset_labels = dataset_labels or DEFAULT_DATASET_LABELS

    cur = pair_summary[
        (pair_summary["dataset_type"].isin(dataset_types))
        & (pair_summary["feature_name"] == feature_name)
    ].copy()

    assert len(cur) > 0, "No data for selected datasets and feature."

    order = {name: i for i, name in enumerate(dataset_types)}
    cur["_order"] = cur["dataset_type"].map(order)
    cur = cur.sort_values("_order")

    x = np.arange(len(cur))

    y = cur["mean_total_witness_count"].to_numpy(dtype=float)
    ci_low = cur["ci_low_total_witness_count"].to_numpy(dtype=float)
    ci_high = cur["ci_high_total_witness_count"].to_numpy(dtype=float)

    yerr_lower = np.maximum(y - ci_low, 0.0)
    yerr_upper = np.maximum(ci_high - y, 0.0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x,
        y,
        yerr=np.vstack([yerr_lower, yerr_upper]),
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [dataset_labels.get(d, d) for d in cur["dataset_type"]],
    )

    ax.set_ylabel("Mean total witness count")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)

    if title is None:
        title = f"Total structural witnesses for {feature_name}"

    ax.set_title(title)

    for i, (_, row) in enumerate(cur.iterrows()):
        ax.text(
            i,
            y[i],
            f"n={int(row['n_models'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, ax


def plot_overlap_category_bars(
    overlap_summary,
    dataset_types,
    dataset_labels=None,
    category_labels=None,
    title=None,
    save_path=None,
    figsize=None,
):
    """
    Plots mean counts of the three witness-overlap categories per dataset:

        feature1_only — neurons witnessing only feature1 (e.g. outer circle)
        both          — neurons witnessing both features
        feature2_only — neurons witnessing only feature2 (e.g. C_mid)

    with bootstrap CI error bars.

    Parameters
    ----------
    overlap_summary:
        tables["overlap_summary"] from build_feature_overlap_tables.

    dataset_types:
        datasets to show as bar groups, in order.

    Returns
    -------
    fig, ax
    """

    dataset_labels = dataset_labels or DEFAULT_DATASET_LABELS

    feature1 = str(overlap_summary["feature1"].iloc[0])
    feature2 = str(overlap_summary["feature2"].iloc[0])

    categories = [
        ("n_feature1_only", f"only {feature1}"),
        ("n_both", "both"),
        ("n_feature2_only", f"only {feature2}"),
    ]

    if category_labels is not None:
        categories = [
            (metric, category_labels.get(metric, default_label))
            for metric, default_label in categories
        ]

    cur = overlap_summary[
        overlap_summary["dataset_type"].isin(dataset_types)
    ].copy()

    assert len(cur) > 0, "No data for selected datasets."

    order = {name: i for i, name in enumerate(dataset_types)}
    cur["_order"] = cur["dataset_type"].map(order)
    cur = cur.sort_values("_order")

    n_groups = len(cur)
    n_cats = len(categories)
    bar_width = 0.8 / n_cats

    if figsize is None:
        figsize = (max(6, 2.2 * n_groups), 4.2)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_groups)

    for cat_idx, (metric, label) in enumerate(categories):
        y = cur[f"mean_{metric}"].to_numpy(dtype=float)
        ci_low = cur[f"ci_low_{metric}"].to_numpy(dtype=float)
        ci_high = cur[f"ci_high_{metric}"].to_numpy(dtype=float)

        yerr_lower = np.maximum(y - ci_low, 0.0)
        yerr_upper = np.maximum(ci_high - y, 0.0)

        offset = (cat_idx - (n_cats - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            y,
            width=bar_width,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            capsize=3,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [dataset_labels.get(d, d) for d in cur["dataset_type"]],
    )

    ax.set_ylabel("Mean witness count per model")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    if title is None:
        title = f"Witness overlap: {feature1} vs {feature2}"

    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, ax


def plot_witness_count_heatmap(
    witness_count_by_model_layer: pd.DataFrame,
    figsize=None,
    annotate=True,
    title="Structural witnesses by model and layer",
    save_path=None,
):
    """
    Visualizes witness_counts_by_model_layer output as a heatmap.

    Expected input:
        rows    = model_id / model_path / model_index
        columns = layer_id
        values  = witness counts
    """

    table = witness_count_by_model_layer.copy()

    if figsize is None:
        figsize = (
            max(6, 1.2 * table.shape[1]),
            max(4, 0.45 * table.shape[0]),
        )

    values = table.values

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(values, aspect="auto")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns)

    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Witness count")

    if annotate:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = values[i, j]
                ax.text(
                    j,
                    i,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, ax


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_dataset_collage_2x3(
    centered_loaded_datasets,
    shifted_loaded_datasets,
    centered_keys=(
        "a_empty_irrelevant",
        "b_filled_same_irrelevant",
        "c_filled_opposite_relevant",
    ),
    shifted_keys=(
        "a1_empty_irrelevant",
        "b1_filled_same_irrelevant",
        "c1_filled_opposite_relevant",
    ),
    centered_panel_labels=("a", "b", "c"),
    shifted_panel_labels=("a.1", "b.1", "c.1"),
    save_path=None,
    point_size=4,
    alpha=0.85,
    figsize=(15, 8),
    show_legend=True,
    show_row_titles=True,
):
    """
    Draws a 2x3 collage:

        row 1: a, b, c        (centered datasets)
        row 2: a.1, b.1, c.1  (shifted / asymmetric datasets)

    Both dataset dicts are keyed by dataset_type (as returned by
    load_dataset_instance); the *_keys arguments select which entries fill
    each row.
    """

    panels = [
        [
            (label, centered_loaded_datasets[key])
            for label, key in zip(centered_panel_labels, centered_keys)
        ],
        [
            (label, shifted_loaded_datasets[key])
            for label, key in zip(shifted_panel_labels, shifted_keys)
        ],
    ]

    class_colors = {
        0: "tab:blue",
        1: "tab:orange",
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
        facecolor="white",
    )

    for row_id in range(2):
        for col_id in range(3):
            ax = axes[row_id, col_id]

            panel_name, loaded = panels[row_id][col_id]

            X = _to_numpy(loaded["X"])
            y = _to_numpy(loaded["y"])

            for cls in np.unique(y):
                mask = y == cls

                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    s=point_size,
                    alpha=alpha,
                    color=class_colors.get(int(cls), None),
                    linewidths=0,
                    label=f"class {int(cls)}",
                )

            ax.set_title(
                panel_name,
                fontsize=16,
                color="black",
                pad=6,
            )

            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    if show_row_titles:
        fig.text(
            0.5,
            0.975,
            "Centered motifs",
            ha="center",
            va="top",
            fontsize=17,
            fontweight="bold",
            color="black",
        )

        fig.text(
            0.5,
            0.505,
            "Asymmetric / shifted motifs",
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="black",
        )

    if show_legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(0.965, 0.035),
            frameon=False,
            fontsize=12,
        )

    plt.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.06,
        top=0.92,
        wspace=0.10,
        hspace=0.34,
    )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return fig, axes
