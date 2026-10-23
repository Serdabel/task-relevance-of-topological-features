"""Figures for experiment 5 — the geometry sweep.

Every figure here answers one question of ``notebooks/experiments/exp5_geometry.ipynb``
and is labelled to be read without the surrounding text: the colours mean the
same thing everywhere (the FEATURE is the dark curve of class 0, the INTERIOR
the warm cloud of class 1, a NULL placement is grey, a level CURVE is crimson),
and every axis carries its units.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Цвета взяты из exp5_geometry_plan, где датасеты читались лучше всего:
# класс 0 синий, класс 1 красный. Линия уровня — чёрная, чтобы не спорить ни с одним
# из двух классов; пороговые линии — тёмно-серые.
C_FEATURE = "#1f77b4"
C_INTERIOR = "#d62728"
C_NULL = "#9a9a9a"
C_CURVE = "#111111"
C_GOOD = "#2c7a4b"
C_BAD = "#a03030"

__all__ = [
    "cells_extent", "draw_cell", "field_background", "draw_curve", "plot_cell_gallery", "plot_grid_heatmap", "plot_score_panels",
    "plot_floor_bars", "plot_margin_panels", "plot_shape_scale", "plot_elongation_grid",
    "plot_phase_pairs", "plot_balance", "plot_position", "plot_level_curve",
    "plot_summary",
]


# ----------------------------------------------------------------------
# the neuron's own field, under everything else
# ----------------------------------------------------------------------

def field_background(ax, field, window=None, alpha: float = 0.5,
                     levels: int = 25, cmap: str = "RdBu_r",
                     quantile: bool = True, value_range=None):
    """The scalar field of one neuron as a filled contour map.

    A level curve drawn on white is a shape; drawn on the field it is a shape
    IN something, and the reader can see at once whether the curve sits on a
    steep fold or in a flat plateau, and whether the neuron has anything at all
    going on where the objects are. Colour is only the sign and size of the
    value -- blue low, red high -- and it is washed out so the curves and the
    points stay on top.

    ``value_range`` pins the colour scale to an interval given by the caller --
    normally the ladder's two anchors. Prefer it whenever the ladder is drawn on
    top: see the note in the code for why ranks are the wrong scale here.

    ``window`` = (half-width, half-height) of the part that will be shown. It
    is not cosmetic. The evaluation box of a cell is much larger than the cell
    itself -- it has to hold the null placements -- and the field's extremes
    usually sit out there, so colouring by the whole box paints the visible
    part in one flat tone. The colour scale is therefore built from the values
    INSIDE the window.
    """
    pts = field.points_nd.numpy()
    vals = field.values_nd.numpy()
    if window is not None:
        hx, hy = window
        inside = (np.abs(pts[..., 0]) <= hx) & (np.abs(pts[..., 1]) <= hy)
        seen = vals[inside] if inside.any() else vals
    else:
        seen = vals
    if value_range is not None:
        # Colour spans exactly the interval the ladder works in. A tanh unit
        # spends most of the window on one of its two plateaus -- here 84 % of
        # the nodes sit at |f| > 0.9 -- so a scale built on the full range, or
        # on ranks, spends its colours on ground where nothing happens. Pinning
        # it to the ladder's own interval gives the whole colormap to the fold
        # and paints both plateaus in the two end colours, which is exactly the
        # distinction the reader needs.
        lo, hi = float(min(value_range)), float(max(value_range))
        edges = np.linspace(lo, hi, levels)
    elif quantile:
        # A tanh unit saturates: over most of the window the field sits at one
        # of its two plateaus, and equally spaced colours paint nearly the whole
        # picture in a single tone -- the fold, which is the only part with any
        # geometry in it, gets two bands out of twenty-five. Levels placed at
        # equal QUANTILES of the values in the window give every colour the same
        # number of grid nodes, so the fold is where the colour changes. Colour
        # then reads as rank, not as value; the value is what the level curves
        # drawn on top are labelled by.
        edges = np.unique(np.quantile(seen, np.linspace(0, 1, levels)))
    else:
        lo, hi = float(np.min(seen)), float(np.max(seen))
        edges = np.linspace(lo, hi + max(1e-9, 1e-6 * abs(hi)), levels)
    if len(edges) < 2:
        return
    ax.contourf(pts[..., 0], pts[..., 1], vals, levels=edges, cmap=cmap,
                alpha=alpha, zorder=0, extend="both")


def draw_curve(ax, curve, color, lw: float = 1.9, zorder: int = 5,
               halo: float = 1.3):
    """A closed level curve with a white halo, so it reads over any background."""
    import matplotlib.patheffects as pe

    pts = np.asarray(curve)
    pts = np.vstack([pts, pts[:1]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=zorder,
            solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=lw + halo, foreground="white"),
                          pe.Normal()])


# ----------------------------------------------------------------------
# the datasets themselves
# ----------------------------------------------------------------------

def cells_extent(names, nulls: bool = False) -> tuple[float, float]:
    """Half-width and half-height of one window holding every listed cell.

    Panels drawn to their own data would put a 4:1 ellipse and a circle at
    different magnifications, which is exactly the comparison the reader is
    being asked to make by eye. One frame for all of them keeps it honest.
    """
    from analysis.geometry_sweep import dataset_points
    from experiments.exp5_plan import derived_plan

    plan = {c["name"]: c for c in derived_plan()}
    hx = hy = 0.0
    for name in names:
        if nulls:
            b = plan[name]["bounds"]
            hx = max(hx, max(abs(v) for v in b[0]))
            hy = max(hy, max(abs(v) for v in b[1]))
        else:
            X, _, _, _ = dataset_points(name)
            hx = max(hx, float(X[:, 0].abs().max()))
            hy = max(hy, float(X[:, 1].abs().max()))
    return hx * 1.10, hy * 1.18


def draw_cell(ax, cell: str, title: str | None = None, nulls: bool = False,
              point_size: float = 1.1, legend: bool = False, half=None):
    """One cell drawn: the feature curve, the interior region, optionally the
    null placements that give this cell its floor."""
    from analysis.geometry_sweep import dataset_points
    from experiments.exp5_plan import derived_plan
    from analysis.null_placement import build_null_features

    X, y, A, _ = dataset_points(cell)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=point_size, c=C_INTERIOR,
               label="interior (class 1)", zorder=2)
    ax.scatter(A[:, 0], A[:, 1], s=point_size, c=C_FEATURE,
               label="feature (class 0)", zorder=3)

    if nulls:
        plan = {c["name"]: c for c in derived_plan()}
        from analysis.null_placement import at
        pls = [at(dx, dy, rot=rot or None, name=nm)
               for dx, dy, rot, nm in plan[cell]["placements"]]
        for i, (name, pts) in enumerate(build_null_features(A, pls).items()):
            ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=C_NULL, zorder=1,
                       label="null placements" if i == 0 else None)

    ax.set_aspect("equal")
    if half is not None:
        hx, hy = half
        ax.set_xlim(-hx, hx); ax.set_ylim(-hy, hy)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(loc="upper right", fontsize=7, markerscale=4, framealpha=0.9)
    return ax


def plot_cell_gallery(names, titles=None, ncols: int = 4, nulls: bool = False,
                      suptitle: str = "", panel: float = 3.1, dpi: int = 150):
    """A row of cells, drawn to the same rules — the picture that makes the
    table above it mean something."""
    names = list(names)
    titles = list(titles) if titles is not None else names
    nrows = math.ceil(len(names) / ncols)
    half = cells_extent(names, nulls=nulls)
    ratio = max(0.35, min(1.0, half[1] / half[0]))
    fig, axes = plt.subplots(nrows, ncols, dpi=dpi,
                             figsize=(panel * ncols, panel * ratio * nrows + 0.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name, title in zip(axes, names, titles):
        draw_cell(ax, name, title, nulls=nulls, half=half)
    for ax in axes[len(names):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=l)
               for c, l in [(C_FEATURE, "feature (class 0)"),
                            (C_INTERIOR, "interior (class 1)")]
               + ([(C_NULL, "null placements")] if nulls else [])]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    return fig


# ----------------------------------------------------------------------
# the 4x4 square of block A
# ----------------------------------------------------------------------

def plot_grid_heatmap(table: pd.DataFrame, value: str, title: str,
                      fmt: str = "{:.2f}", cmap: str = "RdYlGn",
                      vmin=None, vmax=None, block: str = "A", cbar_label: str = ""):
    """Block A as the square it is: the feature's aspect down, the interior's
    across, one number per cell written into its box."""
    A = table[table.block == block]
    piv = A.pivot(index="feature_aspect", columns="interior_aspect", values=value)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}:1" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{r:g}:1" for r in piv.index])
    ax.set_xlabel("aspect of the INTERIOR"); ax.set_ylabel("aspect of the FEATURE")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=11,
                        color="black")
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    fig.tight_layout()
    return fig


def plot_score_panels(best: pd.DataFrame, names, titles, floors, ncols: int = 4,
                      suptitle: str = "", xmax: float = 0.45):
    """Target against its own nulls, cell by cell.

    One panel per cell so a shifted distribution cannot hide inside an average:
    the target's best-neuron scores in colour, the cell's null placements in
    grey, and the cell's own floor as the dashed line the witness rate counts
    against."""
    names = list(names); titles = list(titles)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(0, xmax, 34)
    for ax, name, title in zip(axes, names, titles):
        g = best[best.dataset_type == name]
        tgt = g[g.feature_role == "target"]["score"]
        nul = g[g.feature_role != "target"]["score"]
        ax.hist(nul, bins=bins, color=C_NULL, alpha=0.75, density=True,
                label="the cell's nulls")
        ax.hist(tgt, bins=bins, color=C_FEATURE, alpha=0.75, density=True,
                label="target")
        f = floors.get(name, np.nan)
        ax.axvline(f, color=C_CURVE, ls="--", lw=1.4, label="the cell's floor")
        rate = float((tgt < f).mean())
        ax.set_title(f"{title}\nwitness fraction {rate:.2f}", fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.supxlabel("best-neuron score of the model (Chamfer / diameter)", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def plot_margin_panels(table: pd.DataFrame, names, titles, ncols: int = 4,
                       suptitle: str = "", value: str = "margin"):
    """The homological margin, target against its own nulls, cell by cell.

    Same shape of picture as ``plot_score_panels`` and for the same reason: a
    criterion that fires everywhere and a criterion that separates look
    identical in a table of medians and unmistakable as two distributions. The
    margin is read the other way round from the structural score -- more is
    better -- so the target sitting to the RIGHT of the nulls is the criterion
    working.
    """
    names = list(names); titles = list(titles)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    finite = table[value].replace([np.inf, -np.inf], np.nan).dropna()
    lo = min(0.0, float(np.nanquantile(finite, 0.01)))
    bins = np.linspace(lo, float(np.nanquantile(finite, 0.99)), 34)
    for ax, name, title in zip(axes, names, titles):
        g = table[table.dataset_type == name].replace([np.inf, -np.inf], np.nan)
        # an empty diagram is NaN, not a zero margin: no loop was found at all,
        # which is a different statement from "a loop with no persistence". It
        # is dropped here and counted in the tables instead.
        tgt = g[g.feature_role == "target"][value].dropna()
        nul = g[g.feature_role != "target"][value].dropna()
        ax.hist(nul, bins=bins, color=C_NULL, alpha=0.75, density=True,
                label="the cell's nulls")
        ax.hist(tgt, bins=bins, color=C_GOOD, alpha=0.75, density=True,
                label="target")
        over = float((tgt.values[:, None] > nul.values[None, :]).mean()) if len(nul) else np.nan
        ax.set_title(f"{title}\nP(target > null) = {over:.2f}", fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.supxlabel("homological margin of the model's best neuron", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def plot_floor_bars(cells: pd.DataFrame):
    """Every cell's floor, sorted — the reason a threshold cannot be inherited."""
    c = cells.sort_values("floor")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    colors = {"A": C_FEATURE, "A⊥": "#7b4bb7", "C": C_INTERIOR, "D": C_GOOD}
    ax.bar(range(len(c)), c["floor"], color=[colors[b] for b in c["block"]])
    if "floor_p5" in c:
        ax.plot(range(len(c)), c["floor_p5"], ".", color="#444", ms=4,
                label="the same floor at the 5th percentile")
    ax.axhline(0.1, color=C_CURVE, ls="--", lw=1.4,
               label="tau = 0.1, the threshold of experiments 1 and 2")
    ax.set_xticks([]); ax.set_ylabel("the cell's floor\n(1st percentile of its own nulls)")
    ax.set_xlabel("56 cells, sorted by floor")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[b]) for b in colors]
    ax.legend(handles + [ax.lines[0]], list(colors) + ["tau = 0.1"], ncol=5, fontsize=9)
    ax.set_title(f"The floor varies {c['floor'].max()/c['floor'].min():.1f}-fold between cells — "
                 "so every cell takes its own threshold", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# the descriptor's scale
# ----------------------------------------------------------------------

def plot_shape_scale():
    """What the shape distance means: reference ellipses, their spectra, and
    the distances the experiment must be able to tell apart."""
    import torch
    from analysis.shape_descriptors import fourier_descriptor, shape_distance

    aspects = (1.0, 1.5, 2.5, 4.0)
    th = torch.linspace(0, 2 * math.pi, 2001)[:-1]
    desc = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for a in aspects:
        pts = torch.stack([a * torch.cos(th), torch.sin(th)], 1)
        desc[a] = fourier_descriptor(pts)
        axes[0].plot(pts[:, 0], pts[:, 1], label=f"{a:g}:1")
    axes[0].set_aspect("equal"); axes[0].legend(fontsize=8)
    axes[0].set_title("reference shapes\n(minor semi-axis fixed)", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].bar([f"{a:g}:1" for a in aspects],
                [desc[a]["amplitude"][2] for a in aspects], color=C_FEATURE)
    axes[1].set_title("second harmonic a2 —\na measure of elongation", fontsize=10)
    axes[1].set_ylabel("a2")

    M = np.array([[shape_distance(desc[a], desc[b]) for b in aspects] for a in aspects])
    im = axes[2].imshow(M, cmap="viridis")
    axes[2].set_xticks(range(4), [f"{a:g}:1" for a in aspects])
    axes[2].set_yticks(range(4), [f"{a:g}:1" for a in aspects])
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                         color="white" if M[i, j] < 0.35 else "black", fontsize=10)
    axes[2].set_title("shape distance between the references\n"
                      "(smallest to be resolved: 0.144)", fontsize=10)
    fig.colorbar(im, ax=axes[2])
    fig.tight_layout()
    return fig


def plot_elongation_grid(curves: pd.DataFrame, block: str = "A"):
    """The curve's own elongation against the objects' — on one axis.

    Left: the aspect ratio the level curve actually has, cell by cell. Right:
    the same numbers read along the two axes of the square, which is what says
    WHICH object the curve is following."""
    A = curves[curves.block == block]
    piv = A.pivot(index="feature_aspect", columns="interior_aspect", values="curve_aspect")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    im = axes[0].imshow(piv.values, cmap="magma_r", vmin=1.0, vmax=4.0, aspect="auto")
    axes[0].set_xticks(range(len(piv.columns)), [f"{c:g}:1" for c in piv.columns])
    axes[0].set_yticks(range(len(piv.index)), [f"{r:g}:1" for r in piv.index])
    axes[0].set_xlabel("aspect of the INTERIOR"); axes[0].set_ylabel("aspect of the FEATURE")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                         fontsize=11, color="black")
    axes[0].set_title("aspect of the LEVEL CURVE itself\n(the same scale as the objects: 1…4)",
                      fontsize=10)
    cb = fig.colorbar(im, ax=axes[0]); cb.set_label("curve aspect")

    for axis, color, lab in (("interior_aspect", C_INTERIOR, "aspect of the INTERIOR"),
                             ("feature_aspect", C_FEATURE, "aspect of the FEATURE")):
        m = A.groupby(axis)["curve_aspect"].mean()
        axes[1].plot(m.index, m.values, "o-", color=color, lw=2.2, ms=8,
                     label=f"{lab} (range {m.max()-m.min():.2f})")
    axes[1].plot([1, 4], [1, 4], ls=":", color="k", lw=1.2,
                 label="if the curve copied the object")
    axes[1].set_xlabel("aspect of the object"); axes[1].set_ylabel("aspect of the level curve")
    axes[1].set_ylim(0.9, 4.2); axes[1].legend(fontsize=9)
    axes[1].set_title("the curve follows the interior, not the feature —\n"
                      "and reaches neither", fontsize=10)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# A against A-perp
# ----------------------------------------------------------------------

def plot_phase_pairs(pairs: pd.DataFrame):
    """Nine pairs that share their feature and differ only in the interior's
    orientation: what the rotation does to witnessing and to the curve's shape."""
    labels = [f"f {r.feature_aspect:g}:1\ni {r.interior_aspect:g}:1"
              for r in pairs.itertuples()]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))

    axes[0].bar(x - 0.2, pairs.rate_phase, 0.4, color=C_GOOD, label="in phase")
    axes[0].bar(x + 0.2, pairs.rate_cross, 0.4, color=C_BAD, label="across")
    axes[0].axhline(0.05, ls="--", color="k", lw=1.2, label="chance level (0.05)")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("fraction of models with a witness")
    axes[0].set_title("the INTERIOR turned by 90° — the feature untouched", fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].bar(x - 0.2, pairs.lam_phase, 0.4, color=C_GOOD, label="in phase")
    axes[1].bar(x + 0.2, pairs.lam_cross, 0.4, color=C_BAD, label="across")
    axes[1].axhline(0.5, ls="--", color="k", lw=1.2)
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylabel("lam:  0 = shape of the feature,  1 = shape of the interior")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("whose shape the curve carries", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# balance and position
# ----------------------------------------------------------------------

def plot_balance(rows: pd.DataFrame):
    """Block C: the same four geometries at three class balances."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for geom, g in rows.groupby("geometry", sort=False):
        for ax, col, lab in zip(axes,
                                ["witness_rate", "score", "a2_curve"],
                                ["fraction of models with a witness",
                                 "median best-neuron score",
                                 "elongation of the curve, a2"]):
            ax.plot(g["balance"], g[col], "o-", lw=2, ms=7, label=geom)
            ax.set_xlabel("fraction of the points in the interior")
            ax.set_ylabel(lab)
    axes[0].axhline(0.05, ls="--", color="k", lw=1.2)
    axes[0].set_title("witnessing", fontsize=11)
    axes[1].set_title("score", fontsize=11)
    axes[2].set_title("shape of the curve", fontsize=11)
    axes[2].legend(fontsize=8, loc="upper left", framealpha=0.85)
    fig.suptitle("The class balance changes fourfold; the geometry holds the result",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_position(rows: pd.DataFrame):
    """Block D, one column per pair family.

    Ten curves in one frame would hide exactly the thing worth seeing -- that
    the answer differs by AXIS within each family -- so each family gets its
    own pair of panels: where the curve sits (top) and whether the feature is
    witnessed at all (bottom).
    """
    pairs = list(dict.fromkeys(rows["pair"]))
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.9 * len(pairs), 6.6),
                             sharey="row", squeeze=False)
    for k, pair in enumerate(pairs):
        g = rows[rows["pair"] == pair]
        for r, (col, ref) in enumerate((("lam_pos", None), ("witness_rate", 0.05))):
            ax = axes[r][k]
            for axis, color, mk in (("x", C_FEATURE, "o"), ("y", C_INTERIOR, "s")):
                gg = g[g["axis"] == axis].sort_values("offset_fraction")
                if col == "lam_pos":
                    gg = gg[gg["offset_fraction"] > 0]
                if len(gg) >= 1:
                    ax.plot(gg["offset_fraction"], gg[col], "-", marker=mk, color=color,
                            lw=2.2, ms=8, label=f"offset along {axis}")
            if col == "lam_pos":
                ax.axhline(1.0, ls=":", color=C_INTERIOR, lw=1.6)
                ax.axhline(0.0, ls=":", color=C_FEATURE, lw=1.6)
                ax.set_ylim(-0.15, 1.25)
                ax.set_title(pair, fontsize=10)
            else:
                ax.axhline(ref, ls="--", color="k", lw=1.2)
                ax.set_ylim(-0.05, 1.0)
                ax.set_xlabel("fraction of the largest offset", fontsize=9)
            ax.set_xticks([0.0, 0.5, 1.0])
    axes[0][0].set_ylabel("lam_pos\n0 = centre of the feature,  1 = centre of the interior", fontsize=9)
    axes[1][0].set_ylabel("fraction of models with a witness", fontsize=9)
    fig.suptitle("Block D: what the curve follows when the interior is displaced — "
                 "by family of pairs and by axis", fontsize=12, y=1.0)
    fig.tight_layout()
    handles = [plt.Line2D([], [], marker=m, color=c, lw=2.2, ms=8, label=l)
               for m, c, l in (("o", C_FEATURE, "offset along x"),
                               ("s", C_INTERIOR, "offset along y"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    return fig


# ----------------------------------------------------------------------
# the illustration: a real level curve on real data
# ----------------------------------------------------------------------

def plot_level_curve(cells, titles=None, best=None, ncols: int = 3,
                     suptitle: str = "", which: str = "median",
                     model_index: int = 0, layer_id: int = 3, coord_id: int = 0):
    """The middle-rung level curve of one neuron, drawn over the data it lives in.

    This is what every ``lam`` in the tables is a number for. The frame is
    widened to hold the CURVE as well as the data: the curve lives in the
    evaluation window, which is larger than the data (it has to hold the null
    placements too), and a clipped curve reads as an open one.
    """
    from analysis.geometry_sweep import enclosing_curve_at_mid, best_neuron_of

    cells = list(cells)
    titles = list(titles) if titles is not None else cells
    which_label = "typical model" if which == "median" else "best of 50"

    drawn = []
    for cell in cells:
        if best is not None:
            n = best_neuron_of(best, cell, which=which)
            curve = enclosing_curve_at_mid(cell, layer_id=n["layer_id"],
                                           coord_id=n["coord_id"],
                                           model_path=n["model_path"])
        else:
            n = None
            curve = enclosing_curve_at_mid(cell, model_index, layer_id, coord_id)
        drawn.append((cell, n, curve))

    # Each panel gets its OWN frame here, unlike the galleries. A panel of this
    # figure is about the relation between three things inside it, and one
    # runaway curve under a shared frame would shrink every other panel to a
    # dot; the cross-cell comparison of size is carried by the tables instead.
    halves = []
    for cell, _, curve in drawn:
        hx, hy = cells_extent([cell])
        if curve is not None:
            c = curve.numpy()
            hx = max(hx, float(np.abs(c[:, 0]).max()) * 1.06)
            hy = max(hy, float(np.abs(c[:, 1]).max()) * 1.12)
        # Square frame per panel: the panels then come out the same size in the
        # grid without any shape being stretched to fill its box.
        m = max(hx, hy)
        halves.append((m, m))

    nrows = math.ceil(len(cells) / ncols)
    fig, axes = plt.subplots(nrows, ncols, dpi=150,
                             figsize=(4.0 * ncols, 4.0 * nrows + 0.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, title, (cell, n, curve), half in zip(axes, titles, drawn, halves):
        draw_cell(ax, cell, title, half=half)
        if n is not None:
            ax.set_title(f"{title}\n{which_label}, score {n['score']:.3f}", fontsize=9)
        if curve is not None:
            closed = np.vstack([curve.numpy(), curve.numpy()[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=C_CURVE, lw=2.4, zorder=5)
    for ax in axes[len(cells):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_FEATURE, ms=7,
                          label="feature (class 0)"),
               plt.Line2D([], [], marker="o", ls="", color=C_INTERIOR, ms=7,
                          label="interior (class 1)"),
               plt.Line2D([], [], color=C_CURVE, lw=2.4, label="the neuron's level curve")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    return fig


def plot_summary(cells: pd.DataFrame, shape_mid: pd.DataFrame):
    """All 56 cells on one pair of axes: how far apart the two objects are in
    shape, against whether the feature is witnessed at all."""
    m = cells.merge(shape_mid[["dataset_type", "objects_shape", "lam"]],
                    on="dataset_type", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"A": C_FEATURE, "A⊥": "#7b4bb7", "C": C_INTERIOR, "D": C_GOOD}
    for blk, g in m.groupby("block"):
        axes[0].scatter(g.objects_shape, g.rate_floor, s=52, color=colors[blk],
                        label=blk, alpha=0.85, edgecolor="white")
        axes[1].scatter(g.feature_aspect, g.rate_floor, s=52, color=colors[blk],
                        label=blk, alpha=0.85, edgecolor="white")
    for ax, xl in zip(axes, ["SHAPE distance between the feature and the interior",
                             "aspect of the FEATURE"]):
        ax.axhline(0.05, ls="--", color="k", lw=1.2, label="chance level")
        ax.set_xlabel(xl); ax.set_ylabel("fraction of models with a witness")
    axes[0].legend(fontsize=9)
    axes[0].set_title("a difference in shape does not decide by itself:\nat zero difference there are both 1.00 and 0.04", fontsize=11)
    axes[1].set_title("the elongation of the feature decides:\nat 4:1 everything lies on the floor", fontsize=11)
    fig.tight_layout()
    return fig
