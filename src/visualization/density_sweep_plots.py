"""Figures for experiment 4 — density versus topology.

Every figure here answers one question of the experiment and is called from
``notebooks/experiments/exp4_density_sweep.ipynb``; the numbers behind them come
from ``analysis.density_sweep``.

Three conventions run through the file.

**Scores are DISTANCES** — smaller is better — and the normalised score
(``chamfer / diam(A)``) is the working unit everywhere, as in experiments 1
and 2. Where a figure needs another unit it says so in its own docstring.

**A score is only ever shown against the floor of its own cell**: the null
placements of that same target in that same cell. The floor moves with the
target's shape and with where the nulls land, and a bare score cannot be read
without it.

**The unit of observation is one (model, feature) pair scored by its best
neuron.** Strip plots draw one dot per unit rather than a summary, because the
shape of the population — bimodal, long-tailed, or tight — is itself a result,
and a median hides it.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

C0, C1 = "#d62728", "#1f77b4"          # класс 0 (кольцо) и класс 1 (диск)
CT, CN = "#2ca02c", "#8c8c8c"          # цель и нули
FAMILIES = ("t_inner", "t_mid", "t_outer", "t_ring")
FAM_COLOUR = {"t_inner": "#2ca02c", "t_mid": "#1f77b4",
              "t_outer": "#9467bd", "t_ring": "#d62728"}
FAM_LABEL = {"t_inner": "inner circle", "t_mid": "middle circle",
             "t_outer": "outer circle", "t_ring": "ring"}


@contextmanager
def save_figure_instead_of_show(path):
    """Write figures to disk instead of displaying them."""
    real_show = plt.show
    plt.show = lambda *a, **k: plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    try:
        yield
    finally:
        plt.show = real_show


def _strip(ax, y, values, colour, label_right=True, jitter=0.16, rng=None, extra=""):
    """One row of a strip plot: every unit as a dot, plus median and IQR."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray([v for v in values if v == v], dtype=float)
    if not len(values):
        return
    ax.scatter(values, y + rng.uniform(-jitter, jitter, len(values)), s=13,
               color=colour, alpha=.55, edgecolors="none", zorder=3)
    q1, med, q3 = np.quantile(values, [.25, .5, .75])
    ax.plot([q1, q3], [y, y], color=colour, lw=3.0, alpha=.85, zorder=4)
    ax.plot([med, med], [y - .30, y + .30], color="k", lw=2.0, zorder=5)
    if label_right:
        ax.text(1.005, y, f"med {med:.3f}  min {values.min():.3f}  n={len(values)}{extra}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                color="0.25")


# ----------------------------------------------------------------------
# The datasets themselves
# ----------------------------------------------------------------------

def plot_dataset_grid(cells, loader, ncols: int = 6, size: float = 2.2,
                      title: str | None = None, xlim=None, ylim=None,
                      label_of=None, row_labels=None, col_labels=None):
    """Every dataset of a block, one panel each, in the order given.

    ``cells`` is a list of dataset names — the ORDER IS THE LAYOUT, so the
    caller decides what a row and a column mean and says so through
    ``row_labels`` / ``col_labels``. The rule this enforces is that no block of
    the experiment is discussed without its datasets on screen.
    """
    n = len(cells)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 0.9, size * nrows + 0.6),
                             squeeze=False)

    for k, name in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        X, y, _, _ = loader(name)
        X, y = X.numpy(), y.numpy()
        ax.scatter(*X[y == 0].T, s=1.2, c=C0, linewidths=0)
        ax.scatter(*X[y == 1].T, s=1.2, c=C1, linewidths=0)
        ax.set_aspect("equal")
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if label_of:
            ax.set_title(label_of(name), fontsize=7.5)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    if row_labels:
        for r, text in enumerate(row_labels):
            axes[r][0].set_ylabel(text, fontsize=9)
    if col_labels:
        for c, text in enumerate(col_labels):
            axes[0][c].annotate(text, xy=(0.5, 1.28), xycoords="axes fraction",
                                ha="center", fontsize=9.5, weight="bold")
    if title:
        fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


def plot_targets_and_nulls(cell, loader, size: float = 6.5, title=None):
    """One cell with everything the criteria actually see.

    The data in colour, the measurement targets and their A.1 nulls on top —
    the picture that makes "each target carries its own nulls" concrete.
    """
    X, y, features, roles = loader(cell)
    X, y = X.numpy(), y.numpy()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.scatter(*X[y == 0].T, s=3, c=C0, linewidths=0, label="class 0 (ring)")
    ax.scatter(*X[y == 1].T, s=3, c=C1, linewidths=0, label="class 1 (disk)")

    for name, pts in features.items():
        pts = pts.numpy()
        if roles[name] == "target":
            ax.scatter(*pts.T, s=3.0, c=CT, linewidths=0)
        else:
            ax.scatter(*pts.T, s=1.4, c=CN, linewidths=0)
    ax.scatter([], [], s=12, c=CT, label="targets (what the level set is compared with)")
    ax.scatter([], [], s=12, c=CN, label="A.1 nulls — the same targets, moved to empty places")

    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(title or cell, fontsize=10)
    fig.tight_layout()
    return fig


def plot_target_shapes(loader, cell, size: float = 5.6):
    """The four measurement targets of one cell, drawn apart from the data."""
    X, y, features, roles = loader(cell)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.scatter(*X[y == 0].numpy().T, s=2, c=C0, alpha=.30, linewidths=0, label="data: ring")
    ax.scatter(*X[y == 1].numpy().T, s=2, c=C1, alpha=.30, linewidths=0, label="data: disk")
    for fam in ("t_ring", "t_outer", "t_mid", "t_inner"):
        if fam not in features:
            continue
        pts = features[fam].numpy()
        ax.scatter(*pts.T, s=4.5 if fam == "t_ring" else 3.0,
                   c=FAM_COLOUR[fam], linewidths=0, label=FAM_LABEL[fam])
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("four candidate targets on the same data", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Populations: the workhorse figures
# ----------------------------------------------------------------------

def plot_strips(rows, threshold=None, title=None, caption=None,
                xlim=None, xlabel="best-neuron score (smaller = better)",
                seed: int = 0, height: float = 0.62, row_thresholds=None,
                thumbs=None, loader=None, groups=None, ratios=None,
                null_medians=None, vline=None):
    """One row per population, every (model, feature) unit as a dot.

    ``rows`` is a list of (label, values, colour). This is the figure to reach
    for whenever a verdict is being read: it shows the whole population, so a
    bimodal split or a long tail cannot hide behind a median.

    **A row must be one configuration**, never several pooled together: a null
    belongs to the configuration it was drawn in, and a mixture of
    configurations is a different quantity that can separate where no single
    configuration does. ``row_thresholds`` takes that configuration's own
    threshold per row and draws it as a tick, which is what makes a stack of
    configurations readable without pooling them.

    ``thumbs`` names the dataset behind each row and ``loader`` draws it, so a
    row is identified by the picture of its dataset rather than by decoding
    ``w = 0.5, ρ = 4`` from a label. ``groups`` is a list of ``(label, count)``
    over consecutive rows: a comparison inside a figure is always inside one
    such block — the block is bracketed and named, so what is held fixed and
    what varies is visible instead of stated in the caption. ``ratios`` puts
    floor/target on the row itself as a ``×N`` badge — the dimensionless reading
    of the same population, and the one number that survives a change of target
    size — and replaces the per-row median/min/n text, which is unreadable in a
    stack of eighteen rows.

    ``null_medians`` must be passed whenever ``ratios`` is: the badge compares
    the two MEDIANS, while ``row_thresholds`` draws the 1st percentile of the
    same nulls. Those are different reference points, and a row whose median
    sits right of the threshold can still be far better than its floor. Drawing
    only one of them invites exactly that misreading, so both are drawn.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    has_thumbs = thumbs is not None and loader is not None
    has_groups = bool(groups)

    widths = ([1.15] if has_thumbs else []) + ([0.5] if has_groups else []) + [9.0]
    extra = (1.5 if has_thumbs else 0) + (0.5 if has_groups else 0)
    show_stats = ratios is None
    W, H = 11.5 + extra, height * n + 2.2
    fig = plt.figure(figsize=(W, H))
    # Миниатюры держат равные оси, а tight_layout с такими не работает (и честно
    # об этом предупреждает), поэтому поля считаются явно, в дюймах.
    # Поля считаются явно: tight_layout не работает с осями равного масштаба
    # (миниатюрами) и честно об этом предупреждает.
    margins = dict(left=(1.45 if has_thumbs else 2.8) / W,
                   right=1 - (2.05 if show_stats else 0.55) / W,
                   top=1 - 0.85 / H, bottom=0.72 / H)
    gs = fig.add_gridspec(n, len(widths), width_ratios=widths,
                          hspace=0.12, wspace=0.05, **margins)
    ax = fig.add_subplot(gs[:, -1])

    if has_thumbs:
        for i, name in enumerate(thumbs):
            axt = fig.add_subplot(gs[i, 0])
            axt.set_xticks([]); axt.set_yticks([])
            if name is not None:
                X, y, _, _ = loader(name)
                X, y = X.numpy(), y.numpy()
                axt.scatter(*X[y == 0].T, s=.55, c=C0, linewidths=0)
                axt.scatter(*X[y == 1].T, s=.55, c=C1, linewidths=0)
                axt.set_aspect("equal")
            else:
                for sp in axt.spines.values():
                    sp.set_visible(False)
            axt.set_ylabel(rows[i][0], fontsize=8, rotation=0, ha="right",
                           va="center", labelpad=6)

    for i, (label, values, colour) in enumerate(rows):
        _strip(ax, n - 1 - i, values, colour, rng=rng, label_right=show_stats)
        r = None if show_stats else ratios[i]
        if r is not None and r == r:
            ax.text(0.995, n - 1 - i, f"×{r:.1f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=10, color=colour, weight="bold")
    if null_medians is not None:
        for i, m in enumerate(null_medians):
            if m is not None and m == m:
                ax.plot([m, m], [n - 1 - i - .30, n - 1 - i + .30],
                        color=CN, lw=2.4, zorder=5)
    if row_thresholds is not None:
        for i, t in enumerate(row_thresholds):
            if t is not None and t == t:
                ax.plot([t, t], [n - 1 - i - .34, n - 1 - i + .34],
                        color="crimson", ls=":", lw=1.7, zorder=6)
    if row_thresholds is not None or null_medians is not None:
        xlabel += ("\nblack tick — target median"
                   + (",  grey tick — median of its nulls (×N — their ratio)"
                      if null_medians is not None else "")
                   + (",  red dotted — threshold of its own configuration"
                      if row_thresholds is not None else ""))
    if threshold is not None:
        ax.axvline(threshold, color="crimson", ls=":", lw=1.7,
                   label=f"threshold from its own nulls = {threshold:.3f}")
    if vline is not None:
        ax.axvline(vline[0], color="crimson", ls="--", lw=1.5, label=vline[1])

    if has_groups:
        axb = fig.add_subplot(gs[:, -2])
        axb.axis("off")
        axb.set_xlim(0, 1); axb.set_ylim(-0.5, n - 0.5)
        i0 = 0
        for k, (glabel, count) in enumerate(groups):
            top, bot = n - 1 - i0 + .44, n - 1 - (i0 + count - 1) - .44
            axb.plot([.82, .82], [bot, top], color="0.35", lw=1.4)
            for yy in (bot, top):
                axb.plot([.66, .82], [yy, yy], color="0.35", lw=1.4)
            # подпись группы вертикальная, поэтому её кегль ограничен ВЫСОТОЙ группы:
            # у пары строк места вдвое меньше, чем у тройки, и 8.5 пт там налезают
            fs = max(6.0, min(8.5, 62.0 * count * height / max(len(glabel), 1)))
            axb.text(.52, (bot + top) / 2, glabel, rotation=90, ha="center",
                     va="center", fontsize=fs, color="0.25")
            if k:
                ax.axhline(n - 1 - i0 + .5, color="0.8", lw=1.0, zorder=1)
            i0 += count

    ax.set_yticks(range(n))
    ax.set_yticklabels([] if has_thumbs else [r[0] for r in rows][::-1], fontsize=9)
    ax.set_ylim(-0.5, n - 0.5)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.grid(axis="x", alpha=.25)
    if threshold is not None or vline is not None:
        ax.legend(fontsize=8.5, loc="lower right", framealpha=.92)
    ax.set_title(title or "", fontsize=12, pad=14)
    if caption:
        fig.text(0.5 * (margins["left"] + margins["right"]), 1 - 0.22 / H,
                 caption, ha="center", va="top", fontsize=8.5, color="0.35")
    return fig


def plot_score_histograms(best, families=("t_inner", "t_mid", "t_outer", "t_ring"),
                          task="relevant", bins: int = 40, cells=None,
                          title=None, xmax: float = 1.0):
    """Target and null score distributions, one panel per target family.

    The panel is the whole evidence for that family: where the target sits,
    where its own floor sits, and how much they overlap.
    """
    fams = [f for f in families
            if len(best[(best["task"] == task) & (best["family"] == f)])]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.5 * len(fams), 3.4),
                             sharey=True, squeeze=False)
    edges = np.linspace(0, xmax, bins)
    for ax, fam in zip(axes[0], fams):
        sel = best[(best["task"] == task) & (best["family"] == fam)]
        if cells is not None:
            sel = sel[sel["dataset_type"].isin(cells)]
        tgt = sel[sel["feature_role"] == "target"]["score"]
        nul = sel[sel["feature_role"] == "null"]["score"]
        ax.hist(nul, bins=edges, color=CN, alpha=.85, label="nulls (floor)", density=True)
        ax.hist(tgt, bins=edges, color=FAM_COLOUR[fam], alpha=.70, label="target", density=True)
        ax.axvline(0.1, color="k", ls="--", lw=1.0)
        ax.set_title(f"{FAM_LABEL[fam]}\ntargets {len(tgt)}, nulls {len(nul)}", fontsize=9.5)
        ax.set_xlabel("best-neuron score", fontsize=8.5)
        ax.grid(alpha=.2)
    axes[0][0].set_ylabel("density (normalised)", fontsize=9)
    axes[0][0].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_axis_panels(cells, axis, panels, family="t_mid", task="relevant",
                     suptitle=None, xlabel=None, logx=False, y_from_zero=True,
                     value="med_target", value_label=None):
    """Score AND witness rate along one axis, one line per held-fixed group.

    ``panels`` is a list of dicts ``{"label":…, "sel":{col: value}, "colour":…}``
    naming the lines; inside a line only ``axis`` varies, so a flat line means
    "this variable does nothing" and can be read straight off the figure. Both
    quantities are drawn because they can disagree: a score that improves
    without crossing the threshold moves the left panel and not the right one.

    ``family="edge"`` selects the verdict target — the ring's inner edge, which
    is ``t_mid`` at zero width (see ``analysis.density_sweep.inner_edge``).
    Because that target's diameter moves with ``w``, a line may only run along
    ``w`` on a target of fixed radius; the width axis is therefore read on
    ``t_mid``.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.0))
    fam_rows = (((cells["w"] > 0) & (cells["family"] == "t_inner"))
                | ((cells["w"] == 0) & (cells["family"] == "t_mid"))
                if family == "edge" else cells["family"] == family)
    sel0 = cells[(cells["task"] == task) & fam_rows]

    for spec in panels:
        block = sel0
        for key, want in spec["sel"].items():      # не `value`: так зовётся параметр
            block = block[block[key] == want]
        block = block.sort_values(axis)
        if not len(block):
            continue
        colour = spec.get("colour")
        ls = spec.get("ls", "-")
        ax1.plot(block[axis], block[value], marker="o", ls=ls, color=colour,
                 label=spec["label"], lw=1.8, ms=5)
        ax2.plot(block[axis], block["rate"], marker="o", ls=ls, color=colour,
                 label=spec["label"], lw=1.8, ms=5)
        ax2.fill_between(block[axis], block["rate_lo"], block["rate_hi"],
                         color=colour, alpha=.13)

    if value == "med_target":
        # порог сравним со скором только в тех же единицах; для производных
        # величин (абсолютный Чамфер, отношение «пол / цель») он бессмыслен
        floor = sel0["threshold"].median()
        ax1.axhline(floor, color=CN, ls="--", lw=1.2)
        ax1.text(0.015, floor, " median threshold over these cells", va="bottom",
                 fontsize=7.5, color="0.35", transform=ax1.get_yaxis_transform())
    ax2.axhline(0.01, color="crimson", ls=":", lw=1.2)
    ax2.text(0.015, 0.01, " 1% — the built-in false-positive rate", va="bottom",
             fontsize=7.5, color="crimson", transform=ax2.get_yaxis_transform())

    for ax, ylab in ((ax1, value_label or "median best-neuron score"),
                     (ax2, "fraction of models with a witness")):
        ax.set_xlabel(xlabel or axis, fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(alpha=.25)
        if logx:
            ax.set_xscale("log")
    ax2.set_ylim(-0.04, 1.04)
    if y_from_zero:
        # Иначе автомасштаб растягивает разброс внутри линии на всю ось, и
        # плоская зависимость выглядит как заметный тренд.
        bottom = max(ax1.get_ylim()[1],
                     float(sel0["threshold"].median()) * 1.12 if value == "med_target" else 0)
        ax1.set_ylim(0, bottom)
    ax1.legend(fontsize=9, ncol=2, framealpha=.92, handlelength=3.4,
               labelspacing=0.35, columnspacing=1.1)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def plot_paired_scatter(best, fam_x, fam_y, task="relevant", cells=None,
                        thresholds=None, title=None):
    """The same model scored on two targets, one dot per model.

    A between-cell comparison of medians cannot say whether the two targets
    agree MODEL BY MODEL; this can. Points below the diagonal are models the
    y-target scores better on.
    """
    sel = best[(best["task"] == task) & (best["feature_role"] == "target")]
    if cells is not None:
        sel = sel[sel["dataset_type"].isin(cells)]
    piv = sel.pivot_table(index=["dataset_type", "model_path"], columns="family",
                          values="score")
    piv = piv.dropna(subset=[fam_x, fam_y])

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(piv[fam_x], piv[fam_y], s=12, alpha=.45, color="#1f77b4", edgecolors="none")
    lim = float(max(piv[fam_x].max(), piv[fam_y].max())) * 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--", label="equality")
    if thresholds:
        ax.axvline(thresholds[0], color=FAM_COLOUR[fam_x], ls=":", lw=1.4,
                   label=f"threshold, {FAM_LABEL[fam_x]}")
        ax.axhline(thresholds[1], color=FAM_COLOUR[fam_y], ls=":", lw=1.4,
                   label=f"threshold, {FAM_LABEL[fam_y]}")
    below = float((piv[fam_y] < piv[fam_x]).mean())
    ax.set_xlabel(f"score on the {FAM_LABEL[fam_x]}", fontsize=9.5)
    ax.set_ylabel(f"score on the {FAM_LABEL[fam_y]}", fontsize=9.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title((title or "") + f"\nbelow the diagonal: {below:.0%} of models "
                 f"(n = {len(piv)})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_multiplicity(counts_by_group, n_neurons: int = 25, title=None, ncols: int = 3):
    """How many neurons of one model witness the target, as a histogram.

    ``counts_by_group`` maps a label to a Series of per-model counts. A verdict
    says whether a model has ANY witness; this says how many, which separates
    "one specialised neuron" from "the whole layer".
    """
    items = list(counts_by_group.items())
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    edges = np.arange(-0.5, n_neurons + 1.5)
    for k, (label, counts) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        ax.hist(counts, bins=edges, color="#1f77b4", alpha=.8)
        ax.axvline(float(np.median(counts)), color="crimson", lw=1.5,
                   label=f"median {np.median(counts):.0f}")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7.5)
    for k in range(len(items), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("witness neurons per model (of 25)", fontsize=8.5)
    for row in axes:
        row[0].set_ylabel("models", fontsize=8.5)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_layer_profile(profiles, title=None):
    """Share of neurons clearing the bar, layer by layer.

    ``profiles`` maps a label to a DataFrame with ``layer_id`` and
    ``hit_rate``. Layer 0 is the theorem control and must be flat zero.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for (label, prof), colour in zip(profiles.items(),
                                     plt.cm.tab10(np.linspace(0, .9, len(profiles)))):
        ax.plot(prof["layer_id"], prof["hit_rate"], marker="o",
                color=colour, label=label, lw=1.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("layer (0 is the first hidden layer; its level sets are straight lines by theorem)", fontsize=9)
    ax.set_ylabel("fraction of the layer's neurons passing their cell's threshold", fontsize=9)
    ax.set_xticks(sorted({int(v) for p in profiles.values() for v in p["layer_id"]}))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_spacing(table, grid_step: float, title=None):
    """How tightly the ring's points lie, measured, in two readable units.

    Left: mean distance to the nearest neighbour, in the same length unit for
    every cell — this is the quantity the whole experiment manipulates, and it
    is finite and ordinary at w = 0, where an areal density would not be.
    Right: the same divided by the grid step, which says whether the sampling
    or the discretisation is the coarser of the two.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for n_ring, block in table.groupby("n_ring"):
        block = block.sort_values("w")
        colour = plt.cm.plasma(0.15 + 0.7 * (n_ring / max(table["n_ring"])))
        ax1.plot(block["w"], block["spacing"], marker="o", color=colour,
                 label=f"{n_ring} points", lw=1.8)
        ax2.plot(block["w"], block["spacing"] / grid_step, marker="o", color=colour,
                 label=f"{n_ring} points", lw=1.8)
    ax2.axhline(1.0, color="crimson", ls="--", lw=1.3)
    ax2.text(0.02, 1.0, " grid step", va="bottom", fontsize=8, color="crimson",
             transform=ax2.get_yaxis_transform())
    ax1.set_ylabel("mean distance to the nearest neighbour", fontsize=9)
    ax2.set_ylabel("the same, in grid steps", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xlabel("ring width w", fontsize=9.5)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8.5, title="points on the ring", title_fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_trainability(table):
    """Share of initialisations reaching the 100 % gate, cell by cell."""
    t = table.sort_values("success rate")
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.22 * len(t))))
    colours = [C0 if task == "relevant" else C1 for task in t["task"]]
    ax.barh(range(len(t)), t["success rate"], color=colours, height=.75)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["cell"], fontsize=6)
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xlabel("fraction of the 50 initialisations that reached 100% test accuracy", fontsize=9.5)
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=.25, axis="x")
    ax.scatter([], [], c=C0, label="task-relevant")
    ax.scatter([], [], c=C1, label="task-irrelevant")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    return fig


def plot_two_outcomes(merged, title=None):
    """Witness rate against training success rate, one dot per cell.

    Two instruments measured on the same grid. If they respond to the same
    thing, the cloud is ordered; if one is noise, it is not. The median success
    rate along each design axis is annotated on the figure — that annotation is
    the whole of what a separate trainability table would say, so there is no
    separate table.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    MK = {0.25: "o", 0.5: "s"}
    for (rho, R), block in merged.groupby(["rho", "R"]):
        ax.scatter(block["success rate"], block["med_target"], s=46,
                   marker=MK.get(R, "^"), alpha=.85, edgecolors="none",
                   color=plt.cm.tab10([0.05, 0.15, 0.25][sorted(merged["rho"].unique()).index(rho)]))
    for rho in sorted(merged["rho"].unique()):
        ax.scatter([], [], s=46, color=plt.cm.tab10(
            [0.05, 0.15, 0.25][sorted(merged["rho"].unique()).index(rho)]), label=f"ρ = {rho}")
    for R, mk in MK.items():
        ax.scatter([], [], s=46, marker=mk, color="0.45", label=f"R = {R}")

    lines = []
    for col, name in (("rho", "along the balance ρ"), ("R", "along the disk radius R"),
                      ("w", "along the ring width w")):
        med = merged.groupby(col)["success rate"].median()
        lines.append(f"{name}:  " + "   ".join(f"{k}→{v:.2f}" for k, v in med.items()))
    ax.text(0.02, 0.03, "median training success rate\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=8, va="bottom", color="0.25",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"))

    ax.set_xlabel("fraction of initialisations that reached the gate", fontsize=9.5)
    ax.set_ylabel("median best-neuron score", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_agreement(matrix, title=None):
    """Do two targets call the same MODELS witnesses? Pairwise agreement.

    Each target is thresholded by its own nulls, so this asks whether the
    choice of target changes the verdict, not whether the numbers differ.
    """
    fams = list(matrix.index)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix.values.astype(float), cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(fams)))
    ax.set_yticks(range(len(fams)))
    ax.set_xticklabels([FAM_LABEL[f] for f in fams], rotation=30, ha="right", fontsize=8.5)
    ax.set_yticklabels([FAM_LABEL[f] for f in fams], fontsize=8.5)
    for i in range(len(fams)):
        for j in range(len(fams)):
            v = matrix.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="w" if v < 0.8 else "k")
    fig.colorbar(im, ax=ax, label="fraction of models with the same verdict", shrink=.85)
    ax.set_title(title or "agreement of verdicts between targets", fontsize=10.5, pad=12)
    fig.tight_layout()
    return fig


def plot_discrimination(table, title=None):
    """Each target's floor and target medians side by side, plus their ratio.

    The ratio floor/target is dimensionless and identical whether the scores
    are normalised by the target's diameter or not — the diameter cancels — so
    it is the one number that compares targets of different size without the
    comparison depending on the unit.
    """
    fams = list(table.index)
    x = np.arange(len(fams))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax1.bar(x - 0.19, table["floor_median"], width=.36, color=CN, label="floor (median of the nulls)")
    ax1.bar(x + 0.19, table["target_median"], width=.36,
            color=[FAM_COLOUR[f] for f in fams], label="target (median)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([FAM_LABEL[f] for f in fams], rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("score (normalised)", fontsize=9)
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=.25, axis="y")

    ax2.bar(x, table["floor/target"], color=[FAM_COLOUR[f] for f in fams])
    for i, v in enumerate(table["floor/target"]):
        ax2.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=9)
    ax2.axhline(1.0, color="crimson", ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([FAM_LABEL[f] for f in fams], rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylabel("how many times the target beats its own floor", fontsize=9)
    ax2.grid(alpha=.25, axis="y")
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_config_panels(config_cells, best, cells, loader, xmax=None, seed: int = 0,
                       title=None, row_label=None, row_height: float = 1.15,
                       show_dataset: bool = True, show_gap: bool = True):
    """One ROW per dataset configuration, one column per measurement target.

    This is the figure the choice of target has to be made on, and its layout is
    the methodology: nulls belong to a configuration, so a distribution may only
    ever be looked at inside one. Pooling configurations before comparing target
    to floor can manufacture a separation that no single configuration has, or
    hide one that all of them do — both were observed here before the figure was
    drawn this way.

    The leftmost column draws the dataset itself, so a row can be read without
    decoding its name. Each target column shows the whole population — nulls
    above in grey, target below in the target's colour — with that cell's own
    threshold as the dotted line and its witness rate printed.
    """
    rng = np.random.default_rng(seed)
    fams = list(FAMILIES)
    ncols = len(fams) + (1 if show_dataset else 0)
    nrows = len(config_cells)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(2.05 * ncols + 1.4, row_height * nrows + 1.5),
                             gridspec_kw={"width_ratios": ([1.0] if show_dataset else []) + [1.5] * len(fams)})

    if xmax is None:
        sel = best[best["dataset_type"].isin(config_cells)]
        xmax = float(np.quantile(sel["score"].dropna(), 0.995)) * 1.05

    thr = {(r.cell, r.family): r.threshold for r in cells.itertuples()}
    rate = {(r.cell, r.family): r.rate for r in cells.itertuples()}

    for i, name in enumerate(config_cells):
        col = 0
        if show_dataset:
            ax = axes[i][0]
            X, y, _, _ = loader(name)
            X, y = X.numpy(), y.numpy()
            ax.scatter(*X[y == 0].T, s=.7, c=C0, linewidths=0)
            ax.scatter(*X[y == 1].T, s=.7, c=C1, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(row_label(name) if row_label else name, fontsize=7.5, rotation=0,
                          ha="right", va="center", labelpad=6)
            col = 1

        block = best[best["dataset_type"] == name]
        for j, fam in enumerate(fams):
            ax = axes[i][col + j]
            sub = block[block["family"] == fam]
            tgt = sub[sub["feature_role"] == "target"]["score"].to_numpy()
            nul = sub[sub["feature_role"] == "null"]["score"].to_numpy()
            if len(nul):
                ax.scatter(nul, 1 + rng.uniform(-.22, .22, len(nul)), s=4.5,
                           c=CN, alpha=.5, edgecolors="none")
            if len(tgt):
                ax.scatter(tgt, 0 + rng.uniform(-.22, .22, len(tgt)), s=4.5,
                           c=FAM_COLOUR[fam], alpha=.6, edgecolors="none")
                ax.plot([np.median(tgt)] * 2, [-0.32, 0.32], color="k", lw=1.4, zorder=5)
            if len(nul):
                ax.plot([np.median(nul)] * 2, [0.68, 1.32], color="k", lw=1.4, zorder=5)
            t = thr.get((name, fam))
            if t is not None and t == t:
                ax.axvline(t, color="crimson", ls=":", lw=1.2)
                ax.text(.985, .06, f"fraction {rate.get((name, fam), float('nan')):.2f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                        color="0.25")
            if show_gap and len(tgt) and len(nul):
                ratio = float(np.median(nul)) / float(np.median(tgt))
                ax.text(.985, .70, f"×{ratio:.1f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color=FAM_COLOUR[fam],
                        weight="bold")
            ax.set_xlim(0, xmax); ax.set_ylim(-0.55, 1.55)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["target", "floor"] if j == 0 else ["", ""], fontsize=6.5)
            ax.grid(axis="x", alpha=.2)
            if i == 0:
                ax.set_title(FAM_LABEL[fam], fontsize=8.5)
            if i < nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("best-neuron score", fontsize=7.5)
            ax.tick_params(labelsize=6.5)

    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_ceiling(table, real_best=None, grid_step=None, title=None):
    """Achievable score per target, in the unit that makes it comparable.

    Left panel: the ceiling in ABSOLUTE units, against ``h/2``. This is the
    reading in which the ceiling is a property of the target's SHAPE and of the
    instrument, and of nothing else — a curve costs half a grid step whatever
    its radius or position.

    Right panel: how far the best real neuron of the cell sits from that
    ceiling, as a ratio. The ratio is dimensionless, so it compares targets of
    different size AND different shape without any normalisation choice at all
    — which is what makes it the honest way to ask where the structure is.
    """
    fams = [c for c in FAMILIES if c in table.columns]
    ceil_abs = table[fams].min()
    x = np.arange(len(fams))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2))

    ax1.bar(x, ceil_abs.values, color=[FAM_COLOUR[f] for f in fams], alpha=.85)
    for i, f in enumerate(fams):
        ax1.text(i, ceil_abs[f] + .0012, f"{ceil_abs[f]:.4f}", ha="center", fontsize=9)
    if grid_step is not None:
        ax1.axhline(grid_step / 2, color="crimson", ls="--", lw=1.4)
        ax1.text(.99, grid_step / 2, "h/2 — half the grid step ", va="bottom", ha="right",
                 fontsize=8.5, color="crimson", transform=ax1.get_yaxis_transform())
    ax1.set_xticks(x)
    ax1.set_xticklabels([FAM_LABEL[f].replace(" ", "\n") for f in fams], fontsize=8.5)
    ax1.set_ylabel("ceiling, ABSOLUTE Chamfer distance", fontsize=9.5)
    ax1.grid(axis="y", alpha=.25)
    ax1.set_title("the ceiling is set by the shape and the grid step", fontsize=10.5)

    if real_best is not None:
        norm = table[[f + "_norm" for f in fams]].min()
        ratio = [real_best.get(f, np.nan) / norm[f + "_norm"] for f in fams]
        ax2.bar(x, ratio, color=[FAM_COLOUR[f] for f in fams], alpha=.85)
        for i, v in enumerate(ratio):
            if v == v:
                ax2.text(i, v + .25, f"{v:.1f}x", ha="center", fontsize=9.5)
        ax2.axhline(1.0, color="crimson", ls="--", lw=1.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels([FAM_LABEL[f].replace(" ", "\n") for f in fams], fontsize=8.5)
        ax2.set_ylabel("how many times further from its ceiling\nthe best real neuron is", fontsize=9.5)
        ax2.grid(axis="y", alpha=.25)
        ax2.set_title("the ratio is dimensionless: the targets are comparable", fontsize=10.5)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_single_network_threshold(table, title=None):
    """Can the threshold be set from ONE trained network?

    The research statistics need 50 models, but the THRESHOLD need not: a null
    is a rigid motion of the target scored on the same network, so one network
    supplies its own floor from its own neurons. This compares the verdict that
    per-network floor gives against the verdict from the 50-model floor.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for task, block in table.groupby("task"):
        colour = CT if task == "relevant" else C1
        ax1.scatter(block["rate: threshold from 50 models"], block["rate: threshold from its OWN network"],
                    s=45, color=colour, alpha=.85, label=task, edgecolors="none")
    ax1.plot([0, 1], [0, 1], color="k", ls="--", lw=1)
    ax1.set_xlabel("witness fraction with the threshold from 50 models", fontsize=9.5)
    ax1.set_ylabel("witness fraction with the threshold from its OWN network", fontsize=9.5)
    ax1.set_aspect("equal"); ax1.set_xlim(-.03, 1.03); ax1.set_ylim(-.03, 1.03)
    ax1.grid(alpha=.25); ax1.legend(fontsize=9)
    ax1.set_title("one point — one configuration", fontsize=10.5)

    for task, block in table.groupby("task"):
        colour = CT if task == "relevant" else C1
        ax2.hist(block["agreement"], bins=np.linspace(.5, 1.0, 16), color=colour,
                 alpha=.7, label=task)
    ax2.set_xlabel("fraction of models for which both thresholds give the SAME verdict", fontsize=9.5)
    ax2.set_ylabel("configurations", fontsize=9.5)
    ax2.grid(alpha=.25); ax2.legend(fontsize=9)
    ax2.set_title("agreement of verdicts, model by model", fontsize=10.5)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_grouping_options(tables, title=None):
    """Which groupings of configurations may be pooled, measured.

    ``tables`` maps a grouping's name to the list of its per-block ratios
    (spread of medians between configurations / typical spread inside one).
    Below 1 the configurations differ less than one of them fluctuates, and
    pooling changes nothing; above 1 it does.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    rng = np.random.default_rng(0)
    names = list(tables)
    for i, name in enumerate(names):
        v = np.asarray(tables[name], dtype=float)
        ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=42, alpha=.8,
                   edgecolors="none")
        ax.plot([i - .28, i + .28], [np.median(v)] * 2, color="k", lw=2.2, zorder=5)
        ax.text(i, max(v) + .08, f"max {max(v):.2f}", ha="center", fontsize=8.5)
    ax.axhline(1.0, color="crimson", ls="--", lw=1.4)
    ax.text(.01, 1.02, " above — must not be pooled", transform=ax.get_yaxis_transform(),
            fontsize=8.5, color="crimson", va="bottom")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("spread BETWEEN configurations / spread WITHIN one", fontsize=9.5)
    ax.grid(alpha=.25, axis="y")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_bracket(bracket, title=None):
    """Where the structure sits, as a function of the PROBING RADIUS.

    A probing circle is one of the three concentric measurement targets: the
    ring's inner edge (r = 1.5 - w/2), its midline (1.5) and its outer edge
    (1.5 + w/2). Scoring the same neurons against each asks, in effect, "at
    which radius does the network's structure actually lie" — a question a
    single target cannot pose.

    The x axis is that radius. Each panel plots the same comparison in a
    different unit, so the conclusion can be checked against the choice of unit
    instead of resting on it: the working normalised score, the raw distance,
    and the dimensionless ratio of the target to its own floor.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    cols = [("norm_median", "median target score\n(Chamfer / target diameter)", False),
            ("abs_median", "median Chamfer distance\n(absolute)", False),
            ("times better than floor", "how many times the target beats ITS OWN floor\n(dimensionless)", True)]
    for ax, (col, ylab, is_ratio) in zip(axes, cols):
        for w, block in bracket.groupby("w"):
            block = block.sort_values("radius")
            ax.plot(block["radius"], block[col], marker="o" if w == 0.5 else "s",
                    lw=1.9, ms=7, label=f"ring of width w = {w}")
            for _, r in block.iterrows():
                ax.annotate({"t_inner": "inner", "t_mid": "middle", "t_outer": "outer"}
                            [r["target"]], (r["radius"], r[col]),
                            textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=7, color="0.35")
        if is_ratio:
            ax.axhline(1.0, color="crimson", ls="--", lw=1.2)
            ax.text(.98, 1.0, "chance level ", transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color="crimson", va="top", ha="right")
        ax.set_xlabel("radius of the probing circle", fontsize=9)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.margins(x=.16, y=.16)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_paired_scatters(best, pairs, task="relevant", cells=None, thresholds=None,
                         suptitle=None):
    """Several paired-target scatters side by side, one panel per pair.

    Each dot is ONE MODEL, scored on both targets of the pair, so the comparison
    is within a model and never across configurations.
    """
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.9 * len(pairs), 4.8), squeeze=False)
    sel = best[(best["task"] == task) & (best["feature_role"] == "target")]
    if cells is not None:
        sel = sel[sel["dataset_type"].isin(cells)]
    piv_all = sel.pivot_table(index=["dataset_type", "model_path"], columns="family",
                              values="score")
    for ax, (fam_x, fam_y) in zip(axes[0], pairs):
        piv = piv_all[[fam_x, fam_y]].dropna()
        ax.scatter(piv[fam_x], piv[fam_y], s=11, alpha=.4, color="#1f77b4",
                   edgecolors="none")
        lim = float(max(piv[fam_x].max(), piv[fam_y].max())) * 1.05
        ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--", label="equality")
        if thresholds:
            tx, ty = thresholds[(fam_x, fam_y)]
            ax.axvline(tx, color=FAM_COLOUR[fam_x], ls=":", lw=1.4)
            ax.axhline(ty, color=FAM_COLOUR[fam_y], ls=":", lw=1.4)
        below = float((piv[fam_y] < piv[fam_x]).mean())
        rho = float(piv[fam_x].corr(piv[fam_y], method="spearman"))
        ax.set_xlabel(f"score on the {FAM_LABEL[fam_x]}", fontsize=9)
        ax.set_ylabel(f"score on the {FAM_LABEL[fam_y]}", fontsize=9)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"below the diagonal: {below:.0%} of models\nrank correlation {rho:+.2f}"
                     f"   (n = {len(piv)})", fontsize=9.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_chromatic_ground_truth(chrom, cells, title=None):
    """Task relevance measured on the DATA, and each target's error rates.

    Left: the persistence of the chromatic kernel class per configuration — a
    cycle of one class filled by the other. It exists in a configuration or it
    does not, and that is the ground truth every witness criterion is trying to
    recover, established before any network is trained.

    Right: given that ground truth, what each target gets wrong. A miss on a
    configuration known to be task-relevant is a FALSE NEGATIVE; a hit on one
    known not to be is a FALSE POSITIVE. Both are read as shares of models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.3))
    rng = np.random.default_rng(0)
    for i, (task, colour) in enumerate((("relevant", CT), ("irrelevant", C1))):
        v = chrom[chrom["task"] == task]["kernel_persistence"].to_numpy()
        ax1.scatter(i + rng.uniform(-.2, .2, len(v)), v, s=48, color=colour,
                    alpha=.85, edgecolors="none")
        ax1.plot([i - .3, i + .3], [np.median(v)] * 2, color="k", lw=2.2, zorder=5)
        ax1.text(i, max(v) + .06, f"{(v > 0).sum()} of {len(v)}\nwith a kernel",
                 ha="center", fontsize=9)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["task-relevant", "task-irrelevant"], fontsize=10)
    ax1.set_ylabel("persistence of the kernel class\n(chromatic alpha complex)", fontsize=9)
    ax1.set_ylim(-0.08, 1.45)
    ax1.grid(alpha=.25, axis="y")
    ax1.set_title("one point — one dataset configuration", fontsize=10.5)

    fams = list(FAMILIES)
    x = np.arange(len(fams))
    fn, fp = [], []
    for f in fams:
        r = cells[(cells["family"] == f) & (cells["task"] == "relevant")]["rate"]
        i_ = cells[(cells["family"] == f) & (cells["task"] == "irrelevant")]["rate"]
        fn.append(1 - float(r.median())); fp.append(float(i_.median()))
    ax2.bar(x - .19, fn, width=.36, color=[FAM_COLOUR[f] for f in fams],
            label="misses (median over configurations)")
    ax2.bar(x + .19, fp, width=.36, color=CN, label="false firings")
    for i, (a, b) in enumerate(zip(fn, fp)):
        ax2.text(i - .19, a + .015, f"{a:.2f}", ha="center", fontsize=8.5)
        ax2.text(i + .19, b + .015, f"{b:.2f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([FAM_LABEL[f].replace(" ", "\n") for f in fams], fontsize=8.5)
    ax2.set_ylabel("fraction of models", fontsize=9.5)
    ax2.set_ylim(0, max(max(fn), max(fp)) * 1.35)
    ax2.grid(alpha=.25, axis="y")
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    ax2.set_title("errors against the formal ground truth", fontsize=10.5)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_fpr_calibration(table, title=None):
    """What the percentile actually buys, measured OUT OF SAMPLE.

    The threshold is a percentile of the null scores, so measuring the
    false-positive rate on the same nulls returns the nominal value by
    construction and says nothing. Here the threshold comes from one half of
    the models and every rate is measured on the other half, which is the only
    way the number means anything.

    Three panels: what firing on nulls actually costs (against the nominal), what
    it costs in sensitivity on the verdict target, and — the panel that decides
    the matter — the rate on the task-irrelevant TARGET against the realised
    chance level. A threshold controls the middle panel and the left one; it
    does not control the right one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    fams = sorted(table["target"].unique())

    ax = axes[0]
    for fam in fams:
        b = table[(table["target"] == fam) & (table["task"] == "relevant")].sort_values("FPR")
        ax.plot(b["FPR"], b["firing on nulls (out of sample)"], marker="o",
                color=FAM_COLOUR[fam], label=FAM_LABEL[fam])
    lim = float(table["FPR"].max())
    ax.plot([0, lim], [0, lim], color="k", ls="--", lw=1, label="nominal")
    ax.set_xlabel("null percentile", fontsize=9)
    ax.set_ylabel("firing rate on the nulls,\nmeasured out of sample", fontsize=9)
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title("the threshold cuts firing on the nulls", fontsize=10)

    ax = axes[1]
    for fam in fams:
        b = table[(table["target"] == fam) & (table["task"] == "relevant")].sort_values("FPR")
        ax.plot(b["FPR"], b["rate on target"], marker="o", color=FAM_COLOUR[fam],
                label=FAM_LABEL[fam])
    ax.set_xlabel("null percentile", fontsize=9)
    ax.set_ylabel("witness fraction on the task-relevant target", fontsize=9)
    ax.set_ylim(0.7, 1.02)
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title("and this costs sensitivity", fontsize=10)

    ax = axes[2]
    for fam in fams:
        b = table[(table["target"] == fam) & (table["task"] == "irrelevant")].sort_values("FPR")
        ax.plot(b["FPR"], b["rate on target"], marker="o", color=FAM_COLOUR[fam],
                label=f"{FAM_LABEL[fam]}: target")
        ax.plot(b["FPR"], b["firing on nulls (out of sample)"], marker="s", ls="--",
                color=FAM_COLOUR[fam], alpha=.55, label=f"{FAM_LABEL[fam]}: its nulls")
    ax.set_xlabel("null percentile", fontsize=9)
    ax.set_ylabel("fraction on the task-IRRELEVANT dataset", fontsize=9)
    ax.grid(alpha=.25); ax.legend(fontsize=7.5)
    ax.set_title("but this is not removed by the threshold", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_continuity(test, title=None):
    """How many false positives survive being carried through the composition.

    A witness at layer l is only something the network kept if every later
    layer — and the head — also witnesses the same feature in the same model.
    Left: the share surviving, per error class. Right: where the chain broke.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    test = test.copy()
    # survives приходит как object/float (NaN у неопределённых); приводим явно
    test["survives"] = test["survives"].astype(float)
    classes = list(test.groupby("class"))
    labels = [c for c, _ in classes]
    surv = [float(b["survives"].mean()) for _, b in classes]
    tot = [len(b) for _, b in classes]
    colours = [CN if "null" in c else C1 if "irrelevant" in c else CT for c in labels]

    ax1.bar(range(len(labels)), surv, color=colours)
    for i, (s, n) in enumerate(zip(surv, tot)):
        ax1.text(i, s + .012, f"{s:.1%}\nof {n}", ha="center", fontsize=9)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8.5)
    ax1.set_ylabel("fraction of witnesses that survived the composition", fontsize=9.5)
    ax1.set_ylim(0, min(max(surv + [0.05]) * 1.45, 1.16))
    ax1.grid(alpha=.25, axis="y")
    ax1.set_title("is the witness kept up to the head?", fontsize=10.5)

    failed = test[test["survives"] == 0.0]
    broke = (failed.groupby(["class", "broke at"]).size().unstack(fill_value=0)
             if len(failed) else None)
    if broke is not None:
        broke = broke.reindex(labels, fill_value=0)
        bottom = np.zeros(len(broke))
        for c, colour in zip(broke.columns,
                             plt.cm.viridis(np.linspace(.15, .9, len(broke.columns)))):
            lab = f"layer {c}" if str(c).isdigit() else str(c)
            ax2.bar(range(len(broke)), broke[c].values, bottom=bottom, color=colour, label=lab)
            bottom += broke[c].values
        ax2.set_xticks(range(len(broke)))
        ax2.set_xticklabels([str(i).replace(" ", "\n") for i in broke.index], fontsize=8.5)
    ax2.set_ylabel("witnesses", fontsize=9.5)
    ax2.grid(alpha=.25, axis="y")
    if broke is not None:
        ax2.legend(fontsize=8, title="first layer without a witness", title_fontsize=8)
    ax2.set_title("where the chain breaks", fontsize=10.5)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_criteria_axes(struct, homo, axes, title=None):
    """Two criteria over the same axes: one row each, one column per axis.

    ``struct`` and ``homo`` are cell-level tables keyed the same way; ``axes``
    is a list of ``{"axis", "panels", "xlabel"}`` in the format of
    :func:`plot_axis_panels`. The figure exists to be read as a whole: if the
    top row moves along an axis and the bottom row does not, the two criteria
    are answering different questions about the same networks, and that is a
    result rather than a discrepancy.
    """
    ncols = len(axes)
    fig, ax = plt.subplots(2, ncols, figsize=(4.4 * ncols, 7.0), squeeze=False)
    for j, spec in enumerate(axes):
        for row, (table, col, ylab) in enumerate((
                (struct, "med_target", "structural score\n(smaller = better)"),
                (homo, "med_margin", "homological margin\n(larger = better)"))):
            a = ax[row][j]
            for pan in spec["panels"]:
                block = table
                for key, want in pan["sel"].items():
                    block = block[block[key] == want]
                block = block.sort_values(spec["axis"])
                if not len(block):
                    continue
                a.plot(block[spec["axis"]], block[col], marker="o", lw=1.8, ms=5,
                       color=pan.get("colour"), ls=pan.get("ls", "-"),
                       label=pan["label"])
            a.grid(alpha=.25)
            a.set_xlabel(spec["xlabel"], fontsize=9)
            if j == 0:
                a.set_ylabel(ylab, fontsize=9.5)
            if row == 0:
                a.set_title(spec.get("title", spec["axis"]), fontsize=10.5)
            # обе строки в своих единицах, но с нуля: иначе плоское выглядит крутым
            lo, hi = a.get_ylim()
            a.set_ylim(0, hi * 1.05)
    ax[0][0].legend(fontsize=7.5, ncol=2, framealpha=.9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_criteria_scatter(units, title=None):
    """One dot per (cell, model): structural score vs homological margin.

    The x axis is the score divided by ITS OWN cell's threshold, so x < 1 means
    "structural witness" in every cell at once; y > 0 means "homological
    witness". The quadrants are therefore the agreement table, drawn.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    styles = {("relevant", "target"): (CT, "relevant target", 16),
              ("relevant", "null"): ("#9ecae1", "nulls of relevant cells", 8),
              ("irrelevant", "target"): (C0, "irrelevant target", 16),
              ("irrelevant", "null"): (CN, "nulls of irrelevant cells", 8)}
    for (task, role), (colour, label, size) in styles.items():
        b = units[(units["task"] == task) & (units["feature_role"] == role)]
        if not len(b):
            continue
        ax.scatter(b["score_over_threshold"], b["margin"].fillna(-0.25), s=size,
                   color=colour, alpha=.55, edgecolors="none", label=label)
    ax.axhspan(-0.30, -0.20, color="0.90", zorder=0)
    ax.axvline(1.0, color="crimson", ls=":", lw=1.6)
    ax.axhline(0.0, color="0.35", ls="--", lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("score / threshold of ITS OWN cell   (left of 1 — structural witness)", fontsize=9.5)
    ax.set_ylabel("persistence margin   (above 0 — homological witness)", fontsize=9.5)
    ax.text(0.012, 0.045, "above the dashed line — a loop is present", transform=ax.get_yaxis_transform(),
            fontsize=8, color="0.4")
    ax.text(0.012, -0.25, "  diagram is empty (NaN)", transform=ax.get_yaxis_transform(),
            fontsize=8, color="0.45", va="center")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=.92)
    if title:
        ax.set_title(title, fontsize=11.5)
    fig.tight_layout()
    return fig
