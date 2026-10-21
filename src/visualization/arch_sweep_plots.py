"""Figures for experiment 2 (``analysis/arch_sweep.py``).

Only what the calibration figures cannot already do. ``plot_unit_strips`` and
``plot_populations`` in ``calibration_plots`` draw score populations and per-unit
bests, and ``arch_sweep.to_scores`` hands them experiment 2's data unchanged;
what is missing there is the ARCHITECTURE AXIS, which is what all four
functions here draw.

Same convention as the rest of ``visualization/``: show interactively (return
None) or save to ``save_path`` and return the figure.

One rule runs through all of them: nothing about the target is ever drawn
without the floor of the SAME cell beside it. The floor falls with the number
of neurons on its own (about ``N^(-0.28)``, experiment 1), so a target curve
that improves with width says nothing until it is read against a floor that
improves too.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

#: What the second line of each x tick actually is. Getting this wrong is
#: silent: the tick labels are architectures either way, and only the number
#: under them changes meaning.
_AXIS_LABELS = {
    "n_neurons": "architecture and number of neurons",
    "n_capable": "architecture and number of capable neurons",
    "width": "architecture and width",
    "depth": "architecture and depth",
}

TARGET_COLOUR = "tab:blue"
FLOOR_COLOUR = "0.45"
TAU_COLOUR = "crimson"


def _save_or_show(fig, save_path=None):
    if save_path is None:
        plt.show()
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _cell_rows(cells, dataset_type, want):
    """Rows of ``cell_table`` for the requested (width, depth) list, in order."""
    block = cells[cells["dataset_type"] == dataset_type]
    index = {(int(r.width), int(r.depth)): r for r in block.itertuples()}
    missing = [c for c in want if c not in index]
    assert not missing, f"нет ячеек {missing} у {dataset_type}"
    return [index[c] for c in want]


def plot_axis_panels(cells, panels, x="n_neurons", suptitle=None, save_path=None):
    """One panel per question, each a statistic against an architecture axis.

    ``panels`` is a list of dicts:

        {"title": str,
         "dataset_type": str,
         "cells": [(width, depth), ...],          # the axis, in order
         "series": [(statistic, label, colour), ...],
         "hline": (value, label) | None,          # tau, or 0.5 for p_superior
         "band": (lo, hi, label) | None,          # an indifference band
         "ylim": (lo, hi) | None,                 # pin it when panels must compare
         "log_y": bool}

    Pin ``ylim`` across panels whenever the comparison BETWEEN them is the
    point. Autoscaling a panel whose values are all near zero turns sampling
    noise into a visible trend, and the reader has no way to tell that the
    neighbouring panel is on a scale seventy times larger.

    Every series is drawn with the shaded bootstrap interval its
    ``*_lo``/``*_hi`` columns carry, because the whole point of these curves is
    whether a difference between two of them survives the spread.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.4), squeeze=False)

    for ax, panel in zip(axes[0], panels):
        # Per-panel x, so a width row and a depth row can share one figure —
        # the width row would collapse to a single point on a depth axis.
        px = panel.get("x", x)
        rows = _cell_rows(cells, panel["dataset_type"], panel["cells"])
        xs = np.array([getattr(r, px) for r in rows], dtype=float)

        # Cells with no capable neuron at all — depth 1, or width <= 2 — are
        # kept on the axis (they are the theorem's demonstration) but marked,
        # so a reader never reads their value as a measurement of anything.
        incapable = [xi for xi, r in zip(xs, rows) if r.n_capable == 0]
        for i, xi in enumerate(incapable):
            ax.axvspan(xi / 1.18, xi * 1.18, color="0.92", zorder=0,
                       label="witness impossible (theorem)" if i == 0 else None)

        if panel.get("band"):
            lo, hi, label = panel["band"]
            ax.axhspan(lo, hi, color="0.85", zorder=0, label=label)
        if panel.get("hline"):
            value, label = panel["hline"]
            ax.axhline(value, color=TAU_COLOUR, ls=":", lw=1.5, zorder=1,
                       label=label)

        for statistic, label, colour in panel["series"]:
            y = np.array([getattr(r, statistic) for r in rows], dtype=float)
            lo = np.array([getattr(r, f"{statistic}_lo") for r in rows], dtype=float)
            hi = np.array([getattr(r, f"{statistic}_hi") for r in rows], dtype=float)
            ok = np.isfinite(y)
            ax.fill_between(xs[ok], lo[ok], hi[ok], color=colour, alpha=0.18,
                            linewidth=0, zorder=2)
            ax.plot(xs[ok], y[ok], "o-", color=colour, lw=2.0, ms=6,
                    label=label, zorder=3)

        # Log scale so geometric width spacing reads evenly, but the ticks are
        # the architectures themselves — matplotlib's decade minor labels would
        # otherwise overprint them.
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r.arch}\n{int(getattr(r, px))}" for r in rows],
                           fontsize=8)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        if panel.get("ylim"):
            ax.set_ylim(*panel["ylim"])
        if panel.get("log_y"):
            ax.set_yscale("log")
        ax.set_xlabel(_AXIS_LABELS.get(px, f"architecture and {px}"), fontsize=9)
        ax.set_title(panel["title"], fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, framealpha=0.9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if suptitle else 1.0))
    return _save_or_show(fig, save_path)


def plot_count_vs_neurons(series, fraction=None, title=None, save_path=None):
    """Witness count against model size, one series per architecture row.

    This is an ILLUSTRATION of what the fraction plot already said, not an
    independent result: a constant fraction and a linearly growing count are
    the same statement written twice, since count = fraction x N by
    definition. It earns its place by showing the spread — how much the count
    varies BETWEEN models at a fixed size — which a cell-level fraction hides.

    ``series`` is a list of ``(label, DataFrame, colour)``; each frame is the
    per-model output of ``arch_sweep.witness_counts``. ``fraction`` draws the
    pooled ``fraction x N`` line for reference.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    rng = np.random.default_rng(0)

    for label, block, colour in series:
        xs = block["n_neurons"].to_numpy(float)
        ax.scatter(xs + xs * rng.uniform(-0.02, 0.02, len(xs)), block["count"],
                   s=22, alpha=0.45, color=colour, edgecolors="none", zorder=3,
                   label=f"{label}: models ({len(block)})")
        med = block.groupby("n_neurons")["count"].median()
        ax.plot(med.index, med.to_numpy(), "D-", color=colour, ms=7, lw=1.4,
                zorder=5, label=f"{label}: cell medians")

    if fraction is not None:
        top = max(b["n_neurons"].max() for _, b, _ in series)
        grid = np.linspace(0, top * 1.05, 100)
        ax.plot(grid, fraction * grid, "-", color="0.35", lw=1.6, zorder=2,
                label=f"constant fraction {fraction:.3f}·N")

    ax.set_xlabel("neurons in the model  N = width × depth", fontsize=9.5)
    ax.set_ylabel("witnesses per model", fontsize=9.5)
    ax.set_title(title or "number of witnesses against model size", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper left")
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_witness_count_hist(counts, archs, ncols=4, title=None, save_path=None):
    """How many witnesses one model has — the distribution over an ensemble.

    One panel per architecture. Inside a panel each of the 50 models of that
    ensemble contributes ONE number: how many of its neurons scored below tau
    on the target. The bar at k is therefore "this many models had exactly k
    witness neurons", and the x axis runs over neurons-per-model, not over
    models and not over score.

    The red line at 1 marks the hypothesis the panel exists to test — that a
    feature is carried by a single dedicated neuron, so the mass should pile up
    at k = 1. Where the mass actually sits is the answer.
    """
    nrows = int(np.ceil(len(archs) / ncols))
    fig, axes = plt.subplots(nrows, min(ncols, len(archs)),
                             figsize=(4.3 * min(ncols, len(archs)), 3.3 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(archs):]:
        ax.axis("off")

    for ax, arch in zip(axes.flat, archs):
        block = counts[counts["arch"] == arch]
        assert len(block), f"нет ячейки {arch}"
        values = block["count"].to_numpy(int)
        total = int(block["n_neurons"].iat[0])
        capable = int(block["n_capable"].iat[0])
        ax.hist(values, bins=np.arange(-0.5, values.max() + 1.5),
                color=TARGET_COLOUR, alpha=0.75, edgecolor="white", linewidth=0.6)
        ax.axvline(np.median(values), color="k", lw=2.0,
                   label=f"median {np.median(values):.0f}")
        ax.axvline(1, color=TAU_COLOUR, ls=":", lw=1.6,
                   label="one distinguished neuron")
        ax.set_title(f"{arch}: {total} neurons, {capable} capable", fontsize=10)
        ax.set_xlabel("witnesses in the model", fontsize=8.5)
        ax.set_ylabel("models", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1.0))
    return _save_or_show(fig, save_path)


def plot_best_vs_floor(check, cells_width, cells_depth=None, exponents=None,
                       title=None, save_path=None):
    """Best neuron against model size, with the floor of the same cells.

    The question is whether width bought a better representation or just more
    draws, and the floor answers it without importing anything: the null
    placements of the same cell offer the same number of neurons and carry no
    signal, so their own decline IS the multiplicity effect, measured here.

    Three things are drawn, and the dashed line is the one that needs saying
    out loud. It is NOT a fitted law and NOT taken from another experiment: it
    is the reference cell's target scaled point by point by how much the FLOOR
    moved, ``target(ref) x floor(cell) / floor(ref)``. So it reads "where the
    best neuron would sit if the only thing width changed were the number of
    tries". Measured points below it mean the neurons themselves got closer.

    ``check`` is the output of ``arch_sweep.floor_scaled_check``.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    def rows_for(cells):
        want = [f"w{w} d{d}" for w, d in cells]
        return check[check["arch"].isin(want)].sort_values("n_neurons")

    def series(rows, column, label, colour, marker, annotate=False):
        x = rows["n_neurons"].to_numpy(float)
        y = rows[column].to_numpy(float)
        ax.plot(x, y, marker, color=colour, ms=7, lw=1.8, label=label, zorder=4)
        if annotate:
            for xi, yi, arch in zip(x, y, rows["arch"]):
                ax.annotate(arch, (xi, yi), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color=colour)

    width_rows = rows_for(cells_width)
    x = width_rows["n_neurons"].to_numpy(float)
    ax.fill_between(x, width_rows["median_best_lo"], width_rows["median_best_hi"],
                    color=TARGET_COLOUR, alpha=0.18, linewidth=0, zorder=2)
    series(width_rows, "median_best", "target, width row", TARGET_COLOUR, "o-", True)
    series(width_rows, "median_best_null", "floor of the same cells", FLOOR_COLOUR, "s-")
    ax.plot(x, width_rows["predicted_median_best"], "--", color="k", lw=1.6, zorder=3,
            label="if width bought only attempts\n(reference target × cell floor / reference floor)")

    if cells_depth:
        series(rows_for(cells_depth), "median_best", "target, depth row",
               "tab:orange", "^-", True)

    if exponents:
        text = "   ".join(f"{k}: N^({v:+.2f})" for k, v in exponents.items())
        fig.text(0.5, 0.005, text, ha="center", fontsize=9, color="0.3")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("neurons in the model  N", fontsize=9.5)
    ax.set_ylabel("median best neuron (score / diam A)", fontsize=9.5)
    ax.set_title(title or "best neuron against the floor of the same cell", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, framealpha=0.92)
    fig.tight_layout(rect=(0, 0.035, 1, 1) if exponents else None)
    return _save_or_show(fig, save_path)


def plot_neuron_fields(items, data_points=None, title=None, ncols=None,
                       save_path=None):
    """One panel per neuron: its field, the component scored, the feature.

    Task A's last question is not statistical. Four models of fifty produce a
    structural witness for a circle their task never needed; a count cannot say
    whether that is a level set genuinely tracing the circle or a fragment that
    landed near it, and only the picture can. Each panel therefore draws the
    thing the criterion actually matched — ``best_component_points`` — over the
    field it came from, with the feature on top.

    ``items`` are the dicts returned by ``arch_sweep.neuron_field_and_component``.
    """
    n = len(items)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows),
                             squeeze=False)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, item in zip(axes.flat, items):
        field = item["field"]
        (x0, x1), (y0, y1) = item["bounds"]
        values = field.values_nd.detach().cpu().numpy()
        # make_grid is indexing="ij": rows are x, columns are y, so the image
        # needs a transpose to put x on the horizontal axis.
        im = ax.imshow(values.T, origin="lower", extent=(x0, x1, y0, y1),
                       cmap="RdBu_r", aspect="equal",
                       vmin=-np.abs(values).max(), vmax=np.abs(values).max())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        # Per-item points win over the shared ones: a panel contrasting the two
        # datasets must draw each neuron over the cloud it was trained on.
        points = item.get("data_points", data_points)
        if points is not None:
            P = np.asarray(points)
            ax.scatter(P[:, 0], P[:, 1], s=1.2, color="0.35", alpha=0.25,
                       linewidths=0, zorder=2)

        # The component is drawn WIDE and under, the feature thin and over, so
        # agreement reads as a green halo around a black curve. Drawing them at
        # the same size hides exactly the agreement the panel exists to show.
        component = item.get("component_points")
        if component is not None and len(component):
            C = np.asarray(component)
            ax.scatter(C[:, 0], C[:, 1], s=16, color="lime", alpha=0.55,
                       linewidths=0, zorder=4, label="what the criterion matched")

        A = np.asarray(item["feature_points"])
        ax.scatter(A[:, 0], A[:, 1], s=2.5, color="k", alpha=0.95, linewidths=0,
                   zorder=5, label="feature A")

        ax.set_title(item["label"], fontsize=9.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1.0))
    return _save_or_show(fig, save_path)


# ----------------------------------------------------------------------
# Task C figures
# ----------------------------------------------------------------------

LAYER_COORDS = {
    "layer_id": "layer index (0 = first hidden)",
    "from_output": "distance to the output (0 = last hidden)",
    "rel_pos": "relative position  l / (d − 1)",
    "rel_pos_head": "relative position with the head  l / d",
}


def plot_layer_profiles(panels, suptitle=None, save_path=None):
    """Witness fraction along the layer axis, one curve per architecture.

    ``panels`` is a list of dicts:

        {"title": str,
         "profile": DataFrame from arch_sweep.layer_profile (target),
         "floor": DataFrame | None,     # the same for role="null"
         "x": "layer_id" | "from_output" | "rel_pos",
         "by": "depth" | "width"}       # what separates the curves

    The floor is drawn dashed in the same colour as its target, because the
    layer axis has structure on the nulls too — a target profile without it
    cannot be told apart from "everything gets curvier with depth".
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6),
                             squeeze=False)

    for ax, panel in zip(axes[0], panels):
        profile, floor = panel["profile"], panel.get("floor")
        x, by = panel.get("x", "layer_id"), panel.get("by", "depth")
        value = panel.get("value", "fraction")
        keys = sorted(profile[by].unique())
        colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(keys)))

        for key, colour in zip(keys, colours):
            block = profile[profile[by] == key].sort_values(x)
            label = block["arch"].iat[0]
            if f"{value}_lo" in block:
                ax.fill_between(block[x], block[f"{value}_lo"], block[f"{value}_hi"],
                                color=colour, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(block[x], block[value], "o-", color=colour, lw=2.0,
                    ms=5, label=label, zorder=3)
            if floor is not None:
                fb = floor[floor[by] == key].sort_values(x)
                if len(fb):
                    ax.plot(fb[x], fb[value], ":", color=colour, lw=1.6, zorder=2)

        if floor is not None:
            ax.plot([], [], ":", color="0.4", lw=1.6, label="floor of the same cells")
        if x in ("layer_id", "from_output"):
            # Номер слоя целочислен; дробные подписи на этой оси бессмысленны.
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel(LAYER_COORDS.get(x, x), fontsize=9)
        ax.set_ylabel(panel.get("ylabel", "fraction of the layer's neurons that pass τ"),
                      fontsize=9)
        ax.set_title(panel["title"], fontsize=10.5)
        if panel.get("ylim") is not None:
            ax.set_ylim(*panel["ylim"])
        elif value.endswith("fraction"):
            ax.set_ylim(-0.03, 1.03)
        if panel.get("hline") is not None:
            v, lab = panel["hline"]
            # Красный пунктир зарезервирован за tau; всё остальное рисуется иначе,
            # иначе глаз читает любую горизонталь как порог.
            ax.axhline(v, color=panel.get("hline_colour", TAU_COLOUR), ls=":",
                       lw=1.5, label=lab, zorder=1)
        if panel.get("log_y"):
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, framealpha=0.9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if suptitle else 1.0))
    return _save_or_show(fig, save_path)


def plot_filter_rates(rates, title=None, save_path=None):
    """Share of witnesses the continuity filter flags, per ensemble.

    ``rates`` needs columns ``label``, ``flagged``, ``n_witnesses`` and
    ``colour``. Horizontal bars, because the labels are architecture names and
    the eye compares lengths against a common zero.
    """
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rates) + 2.2))
    y = np.arange(len(rates))[::-1]

    ax.barh(y, rates["flagged"], color=rates["colour"], alpha=0.85, height=0.62)
    for yi, row in zip(y, rates.itertuples()):
        ax.text(row.flagged + 0.012, yi,
                f"{row.flagged:.1%}  ({int(round(row.flagged * row.n_witnesses))}"
                f"/{int(row.n_witnesses)})",
                va="center", fontsize=8.5, color="0.25")

    ax.set_yticks(y)
    ax.set_yticklabels(rates["label"], fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("fraction of witnesses flagged by the filter "
                  "(no witness deeper and the layer is not the last)", fontsize=9)
    ax.set_title(title or "continuity filter across layers", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_layer_multiplicity(counts, archs, ncols=4, title=None, save_path=None):
    """How many neurons of ONE layer witness — read row by row.

    One panel per architecture. A row is a layer; a column is a number of
    witnesses, from none to the whole layer; the cell is the share of the 50
    models whose layer had exactly that many. Every row sums to 1, so rows are
    comparable to each other even though layers differ in how often they
    witness at all.

    A pooled histogram over all layers hides the thing worth seeing: the
    distribution is bimodal, and the two modes belong to DIFFERENT layers —
    early layers pile up at zero, late layers at the full layer. Splitting by
    row is what makes that visible.
    """
    nrows = int(np.ceil(len(archs) / ncols))
    fig, axes = plt.subplots(nrows, min(ncols, len(archs)),
                             figsize=(4.6 * min(ncols, len(archs)), 3.4 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(archs):]:
        ax.axis("off")

    for ax, arch in zip(axes.flat, archs):
        block = counts[counts["arch"] == arch]
        assert len(block), f"нет ячейки {arch}"
        width = int(block["width"].iat[0])
        layers = sorted(block["layer_id"].unique())

        grid = np.zeros((len(layers), width + 1), dtype=float)
        for i, layer in enumerate(layers):
            values = block.loc[block["layer_id"] == layer, "count"].to_numpy(int)
            for k in range(width + 1):
                grid[i, k] = (values == k).mean()

        im = ax.imshow(grid, cmap="magma_r", vmin=0, vmax=1, aspect="auto",
                       origin="lower")
        if width <= 10:
            for i in range(len(layers)):
                for k in range(width + 1):
                    if grid[i, k] >= 0.02:
                        ax.text(k, i, f"{grid[i, k]:.0%}".replace("%", ""),
                                ha="center", va="center", fontsize=7.5,
                                color="white" if grid[i, k] > 0.55 else "0.2")

        ax.set_xticks(range(0, width + 1, max(1, width // 5)))
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"layer {l}" for l in layers], fontsize=8.5)
        ax.set_xlabel("witnesses in the layer (0 … whole layer)", fontsize=8.5)
        ax.set_title(f"{arch}: {width} neurons in a layer", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label="fraction of models" if arch == archs[-1] else None)

    if title:
        fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if title else 1.0))
    return _save_or_show(fig, save_path)


RUNG_LABELS = {"x1": "gate\n(×1)", "x3": "×3 gate", "x10": "×10 gate",
               "end": "3000 epochs"}


def plot_rung_ladder(panels, rung_order=("x1", "x10", "end"),
                     suptitle=None, save_path=None):
    """A statistic along the overtraining ladder, one line per architecture.

    ``panels`` is a list of dicts:

        {"title": str,
         "table": DataFrame from arch_sweep.paired_rungs, concatenated over
                  cells; needs columns base, rung, <value>[, <value>_lo/_hi],
         "value": column to draw,
         "ylabel": str, "log_y": bool, "ylim": (lo, hi),
         "hline": (value, label) | None, "hline_colour": str,
         "highlight": base drawn dashed and grey — the instrument control}

    The x axis is the rung, not the epoch, because the rungs are multiples of
    each run's OWN gate: at a fixed rung different cells sit at different
    absolute epochs on purpose, and plotting epochs would hide the pairing.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4),
                             squeeze=False)
    positions = {r: i for i, r in enumerate(rung_order)}

    for ax, panel in zip(axes[0], panels):
        table, value = panel["table"], panel["value"]
        bases = list(dict.fromkeys(table["base"]))
        highlight = panel.get("highlight")
        ordinary = [b for b in bases if b != highlight]
        colours = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(ordinary), 1)))

        for base in bases:
            block = table[table["base"] == base]
            block = block[block["rung"].isin(positions)].copy()
            block["_x"] = block["rung"].map(positions)
            block = block.sort_values("_x")
            control = base == highlight
            colour = "0.45" if control else colours[ordinary.index(base)]
            label = base.replace("tanh_w", "w").replace("_d", " d")
            if control:
                label += " — control (witness impossible)"
            if f"{value}_lo" in block:
                ax.fill_between(block["_x"], block[f"{value}_lo"], block[f"{value}_hi"],
                                color=colour, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(block["_x"], block[value], "o--" if control else "o-",
                    color=colour, lw=1.6 if control else 2.0, ms=5,
                    label=label, zorder=3)

        if panel.get("hline") is not None:
            v, lab = panel["hline"]
            ax.axhline(v, color=panel.get("hline_colour", TAU_COLOUR), ls=":",
                       lw=1.5, label=lab, zorder=1)
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([RUNG_LABELS.get(r, r) for r in rung_order], fontsize=9)
        ax.set_xlabel("rung of the over-training ladder", fontsize=9)
        ax.set_ylabel(panel.get("ylabel", value), fontsize=9)
        ax.set_title(panel["title"], fontsize=10.5)
        if panel.get("ylim") is not None:
            ax.set_ylim(*panel["ylim"])
        if panel.get("log_y"):
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if suptitle else 1.0))
    return _save_or_show(fig, save_path)
