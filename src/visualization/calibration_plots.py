"""Figures for the detector calibration (``analysis/calibration.py``).

Three kinds of picture, one per question of ``CALIBRATION.md``:

``plot_detection_curve``
    measured score against a measured amount of degradation, with the ideal
    geometry drawn alongside and the threshold marked. Reads as: how much
    damage does the instrument tolerate, and how much of the tolerance comes
    from the instrument rather than from the geometry.

``plot_stability``
    score against one discretisation parameter, one line per probe, with the
    plateau shaded. Reads as: is there a region where the number does not move.

``plot_score_distributions``
    target and null score distributions with both candidate decision rules on
    them. Reads as: does the verdict follow from the numbers or from tau.

Same convention as ``experiment_plots``: show interactively (return None) or
save to ``save_path`` and return the figure.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _save_or_show(fig, save_path=None):
    if save_path is None:
        plt.show()
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_detection_curve(df, axis, threshold=None, criteria=("structural",),
                         title=None, xlabel=None, save_path=None):
    """Measured vs ideal score along one degradation axis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for name in criteria:
        sub = df[df["criterion"] == name].sort_values(axis)
        ax.plot(sub[axis], sub["score"], "o-", label=f"{name} (measured)")

    ideal = df[df["criterion"] == criteria[0]].sort_values(axis)
    if ideal["ideal_score"].notna().any():
        ax.plot(ideal[axis], ideal["ideal_score"], "k--", alpha=0.65,
                label="ideal geometry")

    if threshold is not None:
        ax.axhline(threshold, color="crimson", lw=1.2, ls=":",
                   label=f"tau = {threshold}")

    ax.set_xlabel(xlabel or axis)
    ax.set_ylabel("score / diam(A)")
    ax.set_title(title or f"detection curve along {axis}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_stability(df, param, stability=None, criterion="structural",
                   threshold=None, logx=False, title=None, save_path=None):
    """Score against one parameter, one line per probe, plateau shaded.

    ``stability`` is the matching ``calibration.stability_table`` frame; when
    given, each probe's plateau is shaded in the probe's own colour, which is
    what turns a set of curves into a statement about a region.
    """
    sub = df[df["criterion"] == criterion]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for probe, block in sub.groupby("probe", sort=False):
        block = block.sort_values(param)
        line, = ax.plot(block[param], block["score"], "o-", label=probe)
        if stability is not None:
            row = stability[stability["probe"] == probe]
            if len(row) and row.iloc[0]["plateau_n"] > 1:
                ax.axvspan(row.iloc[0]["plateau_from"], row.iloc[0]["plateau_to"],
                           color=line.get_color(), alpha=0.08)

    if threshold is not None:
        ax.axhline(threshold, color="crimson", lw=1.2, ls=":", label="tau")
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(param)
    ax.set_ylabel("score / diam(A)")
    ax.set_title(title or f"{criterion}: stability in {param}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_score_distributions(target_scores, null_scores, threshold=None,
                             null_threshold=None, bins=25, title=None,
                             save_path=None):
    """Where the target sits inside its own null distribution."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    target = np.asarray([s for s in target_scores if s == s], dtype=float)
    null = np.asarray([s for s in null_scores if s == s], dtype=float)
    edges = np.histogram_bin_edges(np.concatenate([target, null]), bins=bins)

    ax.hist(null, bins=edges, alpha=0.55, label=f"null (n={len(null)})",
            color="tab:gray")
    ax.hist(target, bins=edges, alpha=0.65, label=f"target (n={len(target)})",
            color="tab:blue")

    if threshold is not None:
        ax.axvline(threshold, color="crimson", ls=":", lw=1.4,
                   label=f"tau = {threshold} (hand-picked)")
    if null_threshold is not None:
        ax.axvline(null_threshold, color="darkgreen", ls="--", lw=1.4,
                   label=f"5th pct of nulls = {null_threshold:.3f}")

    ax.set_xlabel("score / diam(A)   (lower = closer to the feature)")
    ax.set_ylabel("neurons")
    ax.set_title(title or "target vs null score distributions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# ----------------------------------------------------------------------
# A parameter sweep: the distributions, and what they do along the axis
# ----------------------------------------------------------------------
#
# ``axis`` names the swept column ("resolution", "n_levels", ...). It is a
# parameter and not a hard-coded name because the plateau is read the same way
# whatever was swept, and a second copy of these figures per parameter would
# drift from the first.

#: One colour per population, kept identical across every sweep figure so a
#: panel can be read without its legend.
ROLE_COLORS = {
    ("irrelevant", "target"): "tab:orange",
    ("relevant", "target"): "tab:blue",
    ("irrelevant", "null"): "0.55",
    ("relevant", "null"): "0.55",
}
DATASET_LABELS = {
    "irrelevant": "task-irrelevant circle",
    "relevant": "task-relevant circle",
}


#: How the swept parameter is spelled on an axis; anything not listed is
#: printed under its own column name.
AXIS_LABELS = {"resolution": "R"}


def _axis_label(axis: str) -> str:
    return AXIS_LABELS.get(axis, axis)


def _finite(a):
    return np.asarray([v for v in np.asarray(a, dtype=float) if v == v])


def _planted_lookup(planted, criterion, axis: str = "resolution"):
    """``{axis value: score}`` from a ``calibration.scan`` frame or a dict."""
    if planted is None:
        return {}
    if isinstance(planted, dict):
        return {int(k): float(v) for k, v in planted.items()}
    sub = planted[planted["criterion"] == criterion]
    if axis not in sub.columns:
        return {}
    return {int(r): float(s) for r, s in zip(sub[axis], sub["score"])}


def _best_medians(scores, criterion, lower_is_better, axis: str = "resolution"):
    """Median over units of the per-(model, feature) best, per cell."""
    from analysis.calibration import best_per_unit

    sub = scores[scores["criterion"] == criterion]
    best = best_per_unit(sub, lower_is_better=lower_is_better,
                         unit=("model_path", "feature", axis, "dataset"))
    return best


def _direction_texts(lower_is_better, mark_note: bool = True):
    """Axis label and caption that say which way "detected" is."""
    if lower_is_better:
        xlabel = ("<-- detection      score / diam(A)      "
                  "(LOWER = the level set traces A more closely)")
        direction = ("LOWER is better: the score is a distance, so a detected "
                     "feature sits on the LEFT")
    else:
        xlabel = ("persistence margin      (HIGHER = the loop survives a wider "
                  "band of levels)      detection -->")
        direction = ("HIGHER is better: the score is a margin, so a detected "
                     "feature sits on the RIGHT")
    unit = ("unit = one (model, feature) pair, scored by that pair's BEST "
            "neuron")
    if mark_note:
        unit += "; the thick vertical line is the median over units"
    return xlabel, f"{direction}.   {unit}."


def plot_sweep_distributions(scores, criterion, planted=None, threshold=None,
                             lower_is_better=None, bins=45, clip=(0.005, 0.995),
                             axis="resolution", values=None,
                             datasets=("irrelevant", "relevant"),
                             title=None, save_path=None):
    """The sweep as it actually is: one score distribution per (axis, dataset).

    Rows are the swept parameter ``axis``, columns are the two datasets, so a
    column is one task read at every setting and a row is the two tasks read at
    one.
    Each panel carries every neuron of every model — grey for the A.1 nulls,
    coloured for the target — plus three reference marks:

    * the **planted neuron**, a field whose correct verdict is known before the
      run. For the structural criterion its true geometric error is ~0, so
      whatever it scores is the instrument's own overhead and the mark is a
      CEILING. For the homological criterion it is only a known positive: the
      margin measures the ridge's contrast, and a trained tanh neuron saturates
      and so beats the planted probe's soft bump — that is a fact about the
      probe, not about the criterion.
    * **tau**, the hand-picked threshold (structural only).
    * the **median best-per-unit**, one line per population. A unit is a
      (model, feature) pair and its value is that pair's best neuron; this is
      the only one of the three marks a verdict is allowed to use, because a
      per-neuron count inherits a multiplicity that grows with width by
      construction (CALIBRATION.md 3.1).

    Bars are fractions of all neurons scored, not a density, so a criterion
    that returns no score for most neurons (the homological one, whose NaN
    means "no matched bar") shows that as missing area instead of hiding it in
    a renormalisation.
    """
    from analysis.calibration import DIRECTIONS

    if lower_is_better is None:
        lower_is_better = DIRECTIONS.get(criterion, True)

    sub = scores[scores["criterion"] == criterion]
    if values is None:
        values = sorted(sub[axis].unique())
    datasets = [d for d in datasets if d in set(sub["dataset"])]

    pool = _finite(sub["score"])
    lo, hi = np.quantile(pool, clip[0]), np.quantile(pool, clip[1])
    marks = _planted_lookup(planted, criterion, axis)
    if marks:
        lo, hi = min(lo, min(marks.values())), max(hi, max(marks.values()))
    edges = np.linspace(lo, hi, bins + 1)

    best = _best_medians(sub, criterion, lower_is_better, axis)

    fig, axes = plt.subplots(len(values), len(datasets), sharex=True,
                             figsize=(5.9 * len(datasets), 2.05 * len(values)),
                             squeeze=False)

    for i, res in enumerate(values):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            cell = sub[(sub[axis] == res) & (sub["dataset"] == ds)]
            notes = []

            for role in ("null", "target"):
                part = cell[cell["role"] == role]
                if not len(part):
                    continue
                population = ("A.1 nulls (same circle, moved)" if role == "null"
                              else DATASET_LABELS[ds])
                v = _finite(part["score"])
                w = np.full(len(v), 1.0 / max(len(part), 1))
                ax.hist(v, bins=edges, weights=w,
                        color=ROLE_COLORS[(ds, role)],
                        alpha=0.50 if role == "null" else 0.65,
                        label=f"all neurons — {population}")
                if len(v) < len(part):
                    notes.append(f"{role}: {len(v) / len(part):.0%} scored")

                b = best[(best[axis] == res) & (best["dataset"] == ds)
                         & (best["role"] == role)]["best_score"]
                if len(b):
                    ax.axvline(float(np.median(b)), color=ROLE_COLORS[(ds, role)],
                               lw=2.2, alpha=0.95,
                               label=f"median best-per-unit — {population}")

            if res in marks:
                ax.axvline(marks[res], color="k", ls="--", lw=1.3,
                           label="planted neuron (known positive)")
            if threshold is not None:
                ax.axvline(threshold, color="crimson", ls=":", lw=1.4,
                           label=f"tau = {threshold}")

            if notes:
                ax.text(0.99, 0.92, "   ".join(notes), transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5, color="0.25")
            if j == 0:
                ax.set_ylabel(f"{_axis_label(axis)} = {res}\nfraction", fontsize=9)
            if i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=10)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)

    xlabel, caption = _direction_texts(lower_is_better)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=8.5)

    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(title or f"{criterion}: score distributions across the {axis} sweep",
                 fontsize=12, y=0.995)
    fig.text(0.5, 0.972, caption, ha="center", va="top", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.01, 1, 0.962))
    return _save_or_show(fig, save_path)


def plot_plateau_curves(scores, criterion, planted=None, threshold=None,
                        lower_is_better=None, contrast=None, axis="resolution",
                        title=None, save_path=None):
    """The same sweep collapsed to the three numbers a plateau is read from.

    Left: the per-unit best (a unit is one (model, feature) pair, scored by its
    best neuron), median with a q10-q90 band, one line per (dataset, role),
    with the planted neuron drawn alongside — a target curve that falls in step
    with the planted one is not improving, it is following the instrument's own
    overhead down. Middle: ``P(target beats null)``, the verdict as a
    scale-free quantity, which is what has to stop moving. Right (only when
    something went undetected): the fraction of units for which the criterion
    returned any score at all.
    """
    from analysis.calibration import DIRECTIONS, sweep_contrast

    if lower_is_better is None:
        lower_is_better = DIRECTIONS.get(criterion, True)

    sub = scores[scores["criterion"] == criterion]
    best = _best_medians(sub, criterion, lower_is_better, axis)
    marks = _planted_lookup(planted, criterion, axis)
    if contrast is None:
        contrast = sweep_contrast(sub, by=("criterion", axis, "dataset"),
                                  threshold=threshold)
    contrast = contrast[contrast["criterion"] == criterion]

    detection = (sub.assign(hit=sub["score"].notna())
                 .groupby([axis, "dataset", "role"])["hit"].mean()
                 .reset_index())
    show_detection = float(detection["hit"].min()) < 0.999

    n = 3 if show_detection else 2
    fig, axes = plt.subplots(1, n, figsize=(5.1 * n, 4.2))

    ax = axes[0]
    for (ds, role), block in best.groupby(["dataset", "role"], sort=False):
        agg = (block.groupby(axis)["best_score"]
               .agg(med="median", lo=lambda s: s.quantile(0.10),
                    hi=lambda s: s.quantile(0.90))
               .sort_index())
        color = ROLE_COLORS[(ds, role)]
        # The two null curves are both grey by convention, so they are told
        # apart by dash pattern rather than by colour.
        style = {"target": "o-", "null": "s--" if ds == "irrelevant" else "^:"}[role]
        label = (DATASET_LABELS[ds] if role == "target"
                 else f"A.1 nulls ({ds} models)")
        ax.plot(agg.index, agg["med"], style, color=color, label=label)
        ax.fill_between(agg.index, agg["lo"], agg["hi"], color=color, alpha=0.12)
    if marks:
        rs = sorted(marks)
        ax.plot(rs, [marks[r] for r in rs], "k--", lw=1.3, marker="x",
                label="planted neuron (known positive)")
    if threshold is not None:
        ax.axhline(threshold, color="crimson", ls=":", lw=1.3, label="tau")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(best[axis].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(axis)
    arrow = "lower is better" if lower_is_better else "higher is better"
    ax.set_ylabel(f"per-unit best score, median + q10-q90\n({arrow})")
    ax.set_title("best neuron per (model, feature)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for ds, block in contrast.groupby("dataset", sort=False):
        block = block.sort_values(axis)
        ax.plot(block[axis], block["p_superior"], "o-",
                color=ROLE_COLORS[(ds, "target")], label=DATASET_LABELS.get(ds, ds))
    ax.axhline(0.5, color="0.4", ls=":", lw=1.2, label="indistinguishable from a null")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(contrast[axis].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(axis)
    ax.set_ylabel("P(a target unit beats a null unit)")
    ax.set_title("separation from the floor", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if show_detection:
        ax = axes[2]
        for (ds, role), block in detection.groupby(["dataset", "role"], sort=False):
            block = block.sort_values(axis)
            style = {"target": "o-",
                     "null": "s--" if ds == "irrelevant" else "^:"}[role]
            label = (DATASET_LABELS[ds] if role == "target"
                     else f"A.1 nulls ({ds} models)")
            ax.plot(block[axis], block["hit"], style,
                    color=ROLE_COLORS[(ds, role)], label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(detection[axis].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(axis)
        ax.set_ylabel("neurons that returned any score")
        ax.set_title("detection rate (censoring)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    _, caption = _direction_texts(lower_is_better, mark_note=False)
    fig.suptitle(title or f"{criterion}: is the sweep on a plateau?", fontsize=12,
                 y=0.995)
    fig.text(0.5, 0.955, caption, ha="center", va="top", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return _save_or_show(fig, save_path)


def plot_probe_field(probe, bounds, resolution, criteria_specs=None,
                     threshold=None, X=None, y=None, vmin=-1.0, vmax=1.0,
                     cmap="viridis", title=None, save_path=None):
    """One scalar field, drawn as the criterion sees it.

    Used on a probe whose verdict is known before the measurement (a planted
    ridge above all), where the picture answers a different question than the
    number does: not "is it a witness" but "what exactly is the criterion
    looking at" — where the crest runs, which connected component of which
    level slab it settles on, and how that component sits against A.

    ``criteria_specs`` is the ``calibration.criteria()`` dict; the structural
    one is re-run with ``return_best_component_points=True`` so the chosen
    component can be drawn, and every criterion's score is printed in the
    corner. ``vmin``/``vmax`` default to the post-activation range of tanh, the
    same fixed scale ``plot_all_hidden_neuron_fields_2d`` uses, so two fields
    are comparable by colour.
    """
    import torch
    from analysis.geometry import make_grid, evaluate_scalar_function_on_points

    A = probe.feature_points.detach().cpu().numpy()
    grid, shape = make_grid(bounds=bounds, resolution=resolution)
    values = evaluate_scalar_function_on_points(probe.function, grid)
    Z = values.reshape(shape).T.detach().cpu().numpy()

    (x0, x1), (y0, y1) = bounds[0], bounds[1]
    fig, ax = plt.subplots(figsize=(9.0, 9.0 * (y1 - y0) / (x1 - x0) + 1.0))
    im = ax.imshow(Z, extent=[x0, x1, y0, y1], origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="bilinear")
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, label="activation")

    if X is not None:
        Xn = np.asarray(X.detach().cpu() if hasattr(X, "detach") else X)
        yn = (np.asarray(y.detach().cpu() if hasattr(y, "detach") else y)
              if y is not None else np.zeros(len(Xn)))
        for cls in np.unique(yn):
            m = yn.reshape(-1) == cls
            ax.scatter(Xn[m, 0], Xn[m, 1], s=3, alpha=0.35, linewidths=0,
                       label=f"data, class {int(cls)}")

    lines = []
    if criteria_specs:
        for name, spec in criteria_specs.items():
            # Only the structural criterion returns a component to draw; the
            # others take **kwargs and would swallow the flag in silence.
            kwargs = ({"return_best_component_points": True}
                      if name == "structural" else {})
            out = spec.run(probe.function, probe.feature_points, bounds,
                           resolution, **kwargs)
            score = out.get(spec.score_key, float("nan"))
            lines.append(f"{name}: {score:.4f}   witness={bool(out.get(spec.witness_key, False))}")
            pts = out.get("best_component_points")
            if pts is not None and len(pts):
                pts = np.asarray(pts.detach().cpu() if hasattr(pts, "detach") else pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=4, c="red", linewidths=0,
                           alpha=0.9, label="best level-set component")

    ax.scatter(A[:, 0], A[:, 1], s=6, facecolors="none", edgecolors="white",
               linewidths=0.7, label="target feature A")

    if lines:
        ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8, lw=0))
    if threshold is not None:
        ax.set_xlabel(f"tau = {threshold} (structural)")

    ax.set_title(title or f"{probe.name} — field at resolution {resolution}",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right", markerscale=2.0, framealpha=0.85)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# ----------------------------------------------------------------------
# One frozen configuration: the populations side by side
# ----------------------------------------------------------------------
#
# The sweep figures above answer "does the answer depend on the parameter".
# These two answer "what IS the answer, at the parameter we froze" — so they
# take a table already restricted to one configuration and never key on a
# swept column.


def _select(df, selector: dict):
    """Rows matching every column -> value (or value list) in ``selector``."""
    mask = np.ones(len(df), dtype=bool)
    for column, want in selector.items():
        col = df[column]
        mask &= col.isin(want) if isinstance(want, (list, tuple, set)) else (col == want)
    return df[mask]


def _unit_bests(block, lower_is_better):
    from analysis.calibration import best_per_unit

    return best_per_unit(block, lower_is_better=lower_is_better,
                         unit=("model_path", "feature"))["best_score"].to_numpy(float)


def plot_populations(scores, criterion, populations, planted=None, threshold=None,
                     lower_is_better=None, bins=45, clip=(0.005, 0.995),
                     title=None, caption=None, save_path=None):
    """Score distributions of several populations in one panel.

    ``populations`` is a list of ``(label, selector, colour)``, where a selector
    is a dict of column -> value(s) — ``{"dataset": "relevant", "role": "null"}``.

    Every population is drawn twice, deliberately: the pale histogram is EVERY
    neuron, which is what the eye reads, and the thick vertical line is the
    median of the per-unit best, which is the only one of the two a verdict may
    use (a per-neuron count inherits a multiplicity that grows with width by
    construction, CALIBRATION.md 3.1). Bars are fractions of that population,
    so populations of different size stay comparable.
    """
    from analysis.calibration import DIRECTIONS

    if lower_is_better is None:
        lower_is_better = DIRECTIONS.get(criterion, True)

    sub = scores[scores["criterion"] == criterion]
    blocks = [(label, _select(sub, sel), colour) for label, sel, colour in populations]
    blocks = [(l, b, c) for l, b, c in blocks if len(b)]

    pool = _finite(np.concatenate([b["score"].to_numpy(float) for _, b, _ in blocks]))
    lo, hi = np.quantile(pool, clip[0]), np.quantile(pool, clip[1])
    mark = None
    if planted is not None:
        marks = _planted_lookup(planted, criterion, "resolution")
        if marks:
            mark = float(list(marks.values())[0]) if len(marks) == 1 else None
    if mark is not None:
        lo, hi = min(lo, mark), max(hi, mark)
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for label, block, colour in blocks:
        v = _finite(block["score"])
        ax.hist(v, bins=edges, weights=np.full(len(v), 1.0 / max(len(block), 1)),
                color=colour, alpha=0.45, label=f"{label} — all neurons ({len(block)})")
        best = _unit_bests(block, lower_is_better)
        if len(best):
            ax.axvline(float(np.median(best)), color=colour, lw=2.4, alpha=0.95,
                       label=f"{label} — median best-per-unit ({len(best)} units)")
        if len(v) < len(block):
            ax.text(0.99, 0.05, f"{label}: score returned for {len(v) / len(block):.0%}",
                    transform=ax.transAxes, ha="right", fontsize=8, color="0.3")

    if mark is not None:
        ax.axvline(mark, color="k", ls="--", lw=1.3, label="planted neuron")
    if threshold is not None:
        ax.axvline(threshold, color="crimson", ls=":", lw=1.5, label=f"tau = {threshold}")

    xlabel, direction = _direction_texts(lower_is_better)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("fraction of the population's neurons")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_title(title or f"{criterion}: distributions at the frozen configuration",
                 fontsize=12)
    fig.text(0.5, 0.965, caption or direction, ha="center", va="top",
             fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save_or_show(fig, save_path)


def plot_unit_strips(scores, criterion, groups, threshold=None, lower_is_better=None,
                     planted=None, title=None, caption=None, seed: int = 0,
                     save_path=None):
    """One row per group: every unit as a dot, with median and IQR.

    The unit is the (model, feature) pair scored by its best neuron, so a row
    is exactly the population a verdict is read from — 50 dots for a target, 50
    per placement for a floor. Rows are drawn in the order given, not sorted, so
    a figure keeps its layout when the numbers move.
    """
    from analysis.calibration import DIRECTIONS

    if lower_is_better is None:
        lower_is_better = DIRECTIONS.get(criterion, True)
    rng = np.random.default_rng(seed)

    sub = scores[scores["criterion"] == criterion]
    rows = []
    for label, sel, colour in groups:
        block = _select(sub, sel)
        if len(block):
            rows.append((label, _unit_bests(block, lower_is_better), colour))

    fig, ax = plt.subplots(figsize=(11.0, 0.62 * len(rows) + 2.4))
    for i, (label, values, colour) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(values, y + rng.uniform(-0.16, 0.16, len(values)), s=15,
                   color=colour, alpha=0.55, edgecolors="none", zorder=3)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([q1, q3], [y, y], color=colour, lw=3.0, alpha=0.85, zorder=4)
        ax.plot([med, med], [y - 0.30, y + 0.30], color="k", lw=2.0, zorder=5)
        ax.text(1.005, y, f"med {med:.3f}   min {values.min():.3f}   n={len(values)}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5,
                color="0.25")

    if threshold is not None:
        ax.axvline(threshold, color="crimson", ls=":", lw=1.6, zorder=2,
                   label=f"tau = {threshold}")
    if planted is not None:
        ax.axvline(planted, color="k", ls="--", lw=1.3, zorder=2,
                   label="planted neuron")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in rows][::-1], fontsize=9.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    xlabel, direction = _direction_texts(lower_is_better, mark_note=False)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    if threshold is not None or planted is not None:
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.set_title(title or f"{criterion}: each point is one (model, feature) pair",
                 fontsize=12)
    fig.text(0.5, 0.985, caption or direction, ha="center", va="top",
             fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 0.80, 0.94))
    return _save_or_show(fig, save_path)


def plot_detection_curves(curves, threshold=None, criterion="structural",
                          axes_order=None, title=None, save_path=None):
    """One panel per degradation axis: measured score, ideal geometry, crossing.

    The single-positive figure says the criterion CAN fire; these say where it
    stops firing. Each panel sweeps one way of breaking the planted circle, and
    the annotated crossing is the answer to "how partial can an encoding be and
    still clear tau" (CALIBRATION.md 1.4) — in the units of that axis, which is
    why the panels are not merged.
    """
    from analysis.calibration import DEGRADATION_AXES, crossing

    sub = curves[curves["criterion"] == criterion]
    order = axes_order or list(dict.fromkeys(sub["axis"]))
    fig, axs = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.3),
                            squeeze=False)

    for ax, name in zip(axs[0], order):
        block = sub[sub["axis"] == name].sort_values("value")
        ax.plot(block["value"], block["score"], "o-", color="tab:blue",
                label="measured score")
        if block["ideal_score"].notna().any():
            ax.plot(block["value"], block["ideal_score"], "k--", alpha=0.6,
                    label="true geometry")
        if threshold is not None:
            ax.axhline(threshold, color="crimson", ls=":", lw=1.4,
                       label=f"tau = {threshold}")
            x = crossing(block, threshold, x="value", y="score")
            if x is not None:
                ax.axvline(x, color="crimson", lw=1.0, alpha=0.45)
                ax.annotate(f"{x:.3f}", (x, threshold), textcoords="offset points",
                            xytext=(6, 8), color="crimson", fontsize=9)
        ax.set_xlabel(f"{name}\n({DEGRADATION_AXES[name][1]})", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axs[0][0].set_ylabel("score / diam(A)   (lower = closer to A)")
    axs[0][0].legend(fontsize=8.5)
    fig.suptitle(title or f"{criterion}: where the detector stops firing",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_or_show(fig, save_path)
