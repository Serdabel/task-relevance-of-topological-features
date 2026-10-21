"""Figures for the PLAN.md experiment analyses (stage 4).

One plotting function per aggregate produced by ``experiments.runner
.aggregate``; every function takes the aggregate table(s), draws on a fresh
figure, and either shows it (interactive, returns None — returning the figure
would make Jupyter draw it a second time) or saves it to ``save_path`` and
returns it. Keep figure logic here, not in the runner and not in notebooks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_or_show(fig, save_path=None):
    if save_path is None:
        # Returning the figure here would make Jupyter render it a SECOND
        # time, below the one plt.show() already drew.
        plt.show()
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_dataset_instances(
    spec,
    instance_id: int = 0,
    max_cols: int = 3,
    size: float = 8.0,
    save_path=None,
):
    """Preview every stored dataset instance an ExperimentSpec trains on.

    One panel per DatasetSpec: the point cloud coloured by class, the target
    feature outlined in crimson, the A.1 null features in grey. This is the
    "look at the data before spending hours on it" check for the mass-run
    notebooks — a mislabelled target or nulls that landed on top of the real
    feature are obvious here and invisible in the aggregate tables.

    Datasets whose instance is not on disk yet are reported and skipped
    (the data stage may not have run, or may be scheduled for a server), so
    the call is safe to leave in a notebook that only prints CLI commands.

    Ambient dimension > 2 is drawn as the first two coordinates, flagged in
    the panel title — a projection, not the geometry itself.

    Returns the figure, or None when no instance exists.
    """
    from experiments.runner import dataset_instance_path
    from management.experiment_management import (
        load_dataset_instance,
        get_feature_points,
    )
    from analysis.stat_aggregation import feature_role_from_metadata
    from visualization.plot_point_clouds import get_color, to_numpy

    loaded_datasets = []

    for ds in spec.datasets:
        path = Path(dataset_instance_path(ds.name, instance_id))

        if not path.exists():
            print(f"[plot_dataset_instances] {ds.name}: no instance at "
                  f"{path} — skipped (run the data stage first)")
            continue

        loaded_datasets.append(
            (ds.name, load_dataset_instance(str(path), map_location="cpu",
                                            reconstruct_data=False))
        )

    if not loaded_datasets:
        print("[plot_dataset_instances] nothing to draw")
        return None

    n = len(loaded_datasets)
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False
    )

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, (name, loaded) in zip(axes.flat, loaded_datasets):
        X = to_numpy(loaded["X"])
        y = to_numpy(loaded["y"]).reshape(-1)
        dim = X.shape[1]

        for cls in sorted(set(y.tolist())):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=size, alpha=0.65, linewidths=0,
                       color=get_color(cls), label=f"class {int(cls)}")

        n_nulls = 0

        for feature_name in loaded.get("features", {}):
            role = feature_role_from_metadata(loaded, feature_name)
            points = to_numpy(get_feature_points(loaded, feature_name))

            if role == "null":
                n_nulls += 1
                ax.scatter(
                    points[:, 0], points[:, 1], s=size, facecolors="none",
                    edgecolors="#777777", linewidths=0.6, alpha=0.7,
                    label="null features (A.1)" if n_nulls == 1 else None,
                )
            elif role == "target":
                ax.scatter(
                    points[:, 0], points[:, 1], s=size + 12, facecolors="none",
                    edgecolors="crimson", linewidths=1.1,
                    label=f"target: {feature_name}",
                )
            else:
                ax.scatter(
                    points[:, 0], points[:, 1], s=size + 6, facecolors="none",
                    edgecolors="#1f77b4", linewidths=0.9,
                    label=f"{role}: {feature_name}",
                )

        projected = "  [first 2 of %d dims]" % dim if dim > 2 else ""
        ax.set_title(f"{name}\n{len(X)} points, {n_nulls} nulls{projected}",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()

    return _save_or_show(fig, save_path)


def plot_rate_gap_bars(
    per_model: pd.DataFrame,
    summary: dict,
    title: str = "Witness rate: feature vs null baseline",
    save_path=None,
):
    """
    Experiment 1 headline: per-model rate[A] and rate[null] side by side,
    plus the mean gap with its bootstrap CI in the title.
    """
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(per_model)), 4))

    x = np.arange(len(per_model))
    width = 0.4

    ax.bar(x - width / 2, per_model["rate_feature"], width,
           label="rate[feature]", color="#1f77b4")
    ax.bar(x + width / 2, per_model["rate_null"], width,
           label="rate[null]", color="#bbbbbb")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(i) for i in range(len(per_model))], fontsize=7
    )
    ax.set_xlabel("model")
    ax.set_ylabel("witness rate over neurons")
    ax.legend()

    ax.set_title(
        f"{title}\nmean gap = {summary['mean_gap']:.3f} "
        f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}] "
        f"(n={summary['n_groups']})"
    )

    return _save_or_show(fig, save_path)


def plot_rate_vs_width(
    curve: pd.DataFrame,
    x_col: str = "width",
    title: str = "Incidental witness rate vs width (gap over null)",
    save_path=None,
):
    """
    Experiment 2 headline: mean gap (rate[A] - rate[null]) per architecture
    width, with bootstrap CI band. ``curve`` columns: x_col, mean_gap,
    ci_low, ci_high, mean_rate_feature, mean_rate_null.
    """
    curve = curve.sort_values(x_col)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(curve[x_col], curve["mean_gap"], "o-", color="#1f77b4",
            label="rate[A] − rate[null]")
    ax.fill_between(curve[x_col], curve["ci_low"], curve["ci_high"],
                    alpha=0.2, color="#1f77b4")

    ax.plot(curve[x_col], curve["mean_rate_feature"], "s--", ms=4,
            color="#2ca02c", label="rate[A]")
    ax.plot(curve[x_col], curve["mean_rate_null"], "s--", ms=4,
            color="#999999", label="rate[null]")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel("witness rate")
    ax.set_title(title)
    ax.legend()

    return _save_or_show(fig, save_path)


def plot_basin_ridge_bars(
    typing_table: pd.DataFrame,
    group_col: str = "dataset_type",
    title: str = "Witness encoding strategy: basin vs ridge",
    save_path=None,
):
    """
    Experiments 4/6: stacked counts of witness types per group (dataset /
    density ratio). ``typing_table`` needs columns [group_col, witness_type].
    """
    counts = (
        typing_table.groupby([group_col, "witness_type"])
        .size()
        .unstack(fill_value=0)
    )

    order = [c for c in ("basin", "ridge", "ambiguous") if c in counts.columns]
    counts = counts[order]

    colors = {"basin": "#1f77b4", "ridge": "#d62728", "ambiguous": "#bbbbbb"}

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(counts)), 4))
    bottom = np.zeros(len(counts))

    for col in counts.columns:
        ax.bar(range(len(counts)), counts[col], bottom=bottom,
               label=col, color=colors.get(col, None))
        bottom += counts[col].to_numpy()

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("witness neurons")
    ax.set_title(title)
    ax.legend()

    return _save_or_show(fig, save_path)


def plot_layer_persistence(
    layer_table: pd.DataFrame,
    title: str = "Witness persistence across layers",
    save_path=None,
):
    """
    Experiment 3: per-layer witness fraction (mean over models with CI) plus
    the layer-monotonicity violation rate in the title. ``layer_table``
    columns: layer_id, mean_fraction, ci_low, ci_high; attrs may carry
    "violation_rate".
    """
    layer_table = layer_table.sort_values("layer_id")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(layer_table["layer_id"], layer_table["mean_fraction"], "o-",
            color="#1f77b4")
    ax.fill_between(layer_table["layer_id"], layer_table["ci_low"],
                    layer_table["ci_high"], alpha=0.2, color="#1f77b4")

    ax.set_xlabel("layer")
    ax.set_ylabel("witness fraction")

    violation = layer_table.attrs.get("violation_rate", None)
    if violation is not None:
        title = f"{title}\n'once witnessed, stays witnessed' violations: " \
                f"{violation:.2%} of (model, layer) transitions"

    ax.set_title(title)

    return _save_or_show(fig, save_path)


def plot_consistency_heatmap(
    table: pd.DataFrame,
    criteria: list[str] | None = None,
    title: str = "Criteria agreement",
    save_path=None,
):
    """
    PLAN.md consistency table as a heatmap: rows = verdict combinations,
    columns = (dataset, feature) groups, cells = fraction of neurons.
    """
    if criteria is None:
        criteria = [
            c for c in ("structural", "extremum", "homological")
            if c in table.columns
        ]

    table = table.copy()
    table["combo"] = table[criteria].apply(
        lambda r: " / ".join(
            f"{name}:{'Y' if r[name] else 'n'}" for name in criteria
        ),
        axis=1,
    )

    group_cols = [
        c for c in ("dataset_type", "feature_name") if c in table.columns
    ]
    table["group"] = table[group_cols].astype(str).agg("\n".join, axis=1)

    pivot = table.pivot_table(
        index="combo", columns="group", values="fraction", fill_value=0.0
    )

    fig, ax = plt.subplots(
        figsize=(max(6, 1.1 * pivot.shape[1]), max(4, 0.45 * pivot.shape[0]))
    )

    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=1,
                   aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if value > 0:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if value > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="fraction of neurons")
    ax.set_title(title)

    return _save_or_show(fig, save_path)


def plot_persistence_diagram(
    diagrams: dict,
    title: str = "Superlevel persistence",
    save_path=None,
):
    """
    Persistence diagram of ``analysis.homological`` output ({dim: [n, 2]
    (birth, death) in level units, birth >= death for superlevel}).
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    colors = {0: "#999999", 1: "#d62728", 2: "#1f77b4"}
    finite_vals = []

    for dim, bars in diagrams.items():
        bars = np.asarray(bars, dtype=float).reshape(-1, 2)
        if len(bars) == 0:
            continue
        finite = bars[np.isfinite(bars).all(axis=1)]
        finite_vals.extend(finite.reshape(-1).tolist())
        ax.scatter(finite[:, 0], finite[:, 1], s=18,
                   color=colors.get(int(dim), None), label=f"H{dim}")

    if finite_vals:
        lo, hi = min(finite_vals), max(finite_vals)
        pad = 0.05 * max(hi - lo, 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="black", lw=0.8)

    ax.set_xlabel("birth level")
    ax.set_ylabel("death level")
    ax.set_title(title)
    ax.legend()

    return _save_or_show(fig, save_path)


def plot_seed_bimodality(
    per_model: pd.DataFrame,
    value_col: str = "n_witnesses",
    group_col: str = "dataset_type",
    title: str = "Per-seed witness distribution (A.7)",
    save_path=None,
):
    """
    A.7: histogram of a per-model quantity (witness count / basin share)
    across seeds, one panel per group — the anti-averaging view.
    """
    groups = list(per_model[group_col].unique())

    fig, axes = plt.subplots(
        1, len(groups), figsize=(4 * len(groups), 3.2), squeeze=False
    )

    for ax, group in zip(axes[0], groups):
        values = per_model.loc[per_model[group_col] == group, value_col]
        ax.hist(values, bins=max(5, min(15, values.nunique())),
                color="#1f77b4", edgecolor="white")
        ax.set_title(str(group), fontsize=9)
        ax.set_xlabel(value_col)
        ax.set_ylabel("models")

    fig.suptitle(title)
    fig.tight_layout()

    return _save_or_show(fig, save_path)
