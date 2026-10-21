"""Reader for experiment 5 — the geometry sweep.

Experiment 5 asks a question experiments 1 and 2 could not: when a neuron's
level set separates a closed curve from the region inside it, WHOSE shape does
it carry? In the earlier experiments both objects were round and concentric, so
"follows the feature" and "follows the interior" predicted the same picture.
Here the pair is pulled apart along four axes, and each axis is a block of cells:

    A     aspect of the feature x aspect of the interior (4x4)
    A-perp  the same, with the interior turned across the feature
    C     four pair geometries x class balance
    D     the interior displaced, in x and in y

The dataset name encodes the whole cell, but this module never parses it: the
coordinates come from ``experiments.exp5_plan``, which is what BUILT the cells,
so the reader cannot drift from the plan.

Two families of number live in the table and they answer different questions:

``structural_score_normalized_by_diameter``
    how close the best level set comes to the FEATURE. This is the detector of
    experiments 1, 2 and 4, unchanged, and it is read the same way: against the
    floor measured from that cell's own null placements, never against an
    inherited threshold.

``harmonic_*``
    the shape descriptor. For five levels spread between the two objects it
    reports the level curve's distance in SHAPE to the feature (``dfeat``) and
    to the interior (``dint``). Their ratio is the answer to "whose shape",
    and it is defined here as

        lam = dfeat / (dfeat + dint)        0 = the feature's shape,
                                            1 = the interior's shape.

    ``lam`` only means something when the two objects HAVE different shapes:
    on the diagonal of block A they are similar, ``harmonic_objects_shape``
    is near zero, and the ratio is noise over noise. Every shape table
    therefore carries ``objects_shape`` so the degenerate cells stay visible
    instead of being averaged in.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = Path("analysis/results/experiments/exp5_geometry")

SCORE = "structural_score_normalized_by_diameter"
MARGIN = "homological_persistence_margin"
TAU = 0.1                 # legal only for a circle-shaped target; see cell_table
NULL_PERCENTILE = 1.0     # the floor rule frozen in experiment 4
ROBUST_PERCENTILE = 5.0   # reported alongside, so the verdict's robustness is visible
N_RUNGS = 5

_CONVERTERS = {"feature_role": lambda v: v}     # bare `null` is a string, not NaN

__all__ = [
    "SCORE", "TAU", "NULL_PERCENTILE", "N_RUNGS",
    "cell_coordinates", "load_neuron_stats", "best_by_model", "cell_table",
    "ladder_long", "shape_by_cell", "position_table", "dataset_points",
    "curve_shape_table", "marginal_effect", "phase_pairs", "a2_to_aspect",
    "best_neuron_of", "neuron_field", "enclosing_curve_at_mid",
]


# ----------------------------------------------------------------------
# the cells, straight from the plan that built them
# ----------------------------------------------------------------------

def cell_coordinates() -> pd.DataFrame:
    """One row per cell: its block, its geometry, its window, its nulls.

    ``offset_frac`` is the displacement as a fraction of the largest one this
    pair admits, which is what makes block D's three steps comparable across
    pairs whose geometry allows very different absolute room.
    """
    from experiments.exp5_plan import derived_plan

    rows = []
    for cell in derived_plan():
        p = cell["params"]
        ox, oy = p.get("interior_offset", (0.0, 0.0))
        rows.append({
            "dataset_type": cell["name"],
            "block": cell["block"],
            "label": cell["label"],
            "feature_aspect": p.get("feature_aspect", 1.0),
            "interior_aspect": p.get("interior_aspect", 1.0),
            "interior_turned": bool(p.get("interior_rotation", 0.0)),
            "interior_scale": p.get("interior_scale", 1.0),
            "offset_x": float(ox),
            "offset_y": float(oy),
            "offset_norm": float(math.hypot(ox, oy)),
            "offset_axis": "x" if abs(ox) > abs(oy) else ("y" if abs(oy) > 0 else "-"),
            "n_interior": p.get("n_interior", 500),
            "balance": p.get("n_interior", 500) / 1000.0,
            "resolution": cell["resolution"],
            "diameter": cell["diameter"],
            "interior_a": cell["interior"]["a"],
            "interior_b": cell["interior"]["b"],
            "n_nulls": len(cell["placements"]),
        })
    out = pd.DataFrame(rows)
    out["grid_step"] = out["diameter"] / out["resolution"] * 0  # filled below
    return out.drop(columns=["grid_step"])


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def load_neuron_stats(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """The per-neuron table, with the cell's coordinates merged in.

    The gzipped twin exists because this run's table is 216 MB and GitHub
    refuses a file over 100 MB, so only ``neuron_stats_*.csv.gz`` (34 MB) is in
    the repository. pandas reads either by extension; the raw CSV wins when
    both are present, so a local run is never slowed by decompression.
    """
    results_dir = Path(results_dir)
    found: dict[str, Path] = {}
    for pattern in ("neuron_stats_*.csv.gz", "*/neuron_stats_*.csv.gz",
                    "neuron_stats_*.csv", "*/neuron_stats_*.csv"):
        for path in sorted(results_dir.glob(pattern)):
            found[path.name.removesuffix(".gz")] = path
    paths = [found[k] for k in sorted(found)]
    assert paths, f"no neuron_stats_*.csv[.gz] under {results_dir}"

    df = pd.concat([pd.read_csv(p, converters=_CONVERTERS) for p in paths],
                   ignore_index=True)
    df["score"] = df[SCORE]
    df = df.merge(cell_coordinates(), on="dataset_type", how="left")
    assert df["block"].notna().all(), "a dataset in the table is not in the plan"

    # The step of the grid the field was sampled on. Needed wherever a length
    # has to be read as "how many grid steps", which is how the descriptor's
    # reliability is judged.
    df["grid_step"] = df["diameter"] / 54.5
    return df


def load_harmonic_ladder(ladder: str | None = "value", path=None) -> pd.DataFrame:
    """The per-rung shape spectrum of every neuron (see analysis.harmonic_ladder).

    The run's CSV stores the spectrum for the middle rung only. This is the
    recomputation that keeps it at every rung, built once by
    ``scripts/build_harmonic_ladder.py`` and read from disk here.
    """
    from analysis import harmonic_ladder

    return harmonic_ladder.load(path or harmonic_ladder.CACHE, ladder=ladder)


def best_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, model, feature), valued by its BEST neuron.

    The unit of observation frozen in experiment 1: a model witnesses a feature
    or it does not, and the neuron that does it is the model's best. Bootstrap
    resampling later resamples THESE rows -- models -- never neurons.
    """
    keep = ["dataset_type", "block", "label", "feature_aspect", "interior_aspect",
            "interior_turned", "interior_scale", "offset_x", "offset_y",
            "offset_norm", "offset_axis", "balance", "resolution", "diameter",
            "n_nulls", "model_path", "feature_name", "feature_role"]
    idx = df.groupby(["dataset_type", "model_path", "feature_name"])["score"].idxmin()
    best = df.loc[idx, keep + ["score", "layer_id", "coord_id", MARGIN]].copy()
    return best.reset_index(drop=True)


# ----------------------------------------------------------------------
# scores against the cell's own floor
# ----------------------------------------------------------------------

def _boot_median(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True)
    meds = np.median(draws, axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def cell_table(best: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per cell: the target's best-neuron score, the cell's own null floor, and
    the witness rate read against THAT floor.

    Why not tau = 0.1: tau normalises by diam(A), and a 4:1 ellipse has four
    times the diameter of the circle, so the same tau is four times the
    absolute tolerance. Experiment 4 froze the alternative -- a percentile of
    the cell's own nulls -- and that is what is used here.

    Which percentile is a decision about the cost of the two errors. A false
    positive is expensive: a criterion that fires on a feature the task cannot
    see puts the notion of relevance itself in doubt. A false negative is
    cheap by comparison. Experiment 4 settled on the 1st percentile for that
    reason, and the same reason applies here, so ``NULL_PERCENTILE = 1``.

    The 5th is reported next to it (``floor_p5``, ``rate_p5``) rather than
    dropped, for two reasons: every conclusion of the experiment must be shown
    to survive the choice, and in the eight cells with one or two null
    placements the floor rests on 50-100 values, where the 1st percentile is
    an extreme order statistic and the 5th is not. ``n_null_values`` is
    reported so that is visible per cell.
    """
    rows = []
    for name, g in best.groupby("dataset_type"):
        tgt = g[g.feature_role == "target"]
        nul = g[g.feature_role != "target"]
        floor = float(np.percentile(nul["score"], NULL_PERCENTILE)) if len(nul) else float("nan")
        floor5 = float(np.percentile(nul["score"], ROBUST_PERCENTILE)) if len(nul) else float("nan")
        med, lo, hi = _boot_median(tgt["score"], n_boot, seed)
        meta = tgt.iloc[0]
        rows.append({
            "dataset_type": name,
            "block": meta["block"], "label": meta["label"],
            "feature_aspect": meta["feature_aspect"],
            "interior_aspect": meta["interior_aspect"],
            "interior_turned": meta["interior_turned"],
            "interior_scale": meta["interior_scale"],
            "offset_norm": meta["offset_norm"], "offset_axis": meta["offset_axis"],
            "balance": meta["balance"], "resolution": meta["resolution"],
            "n_models": len(tgt), "n_nulls": int(meta["n_nulls"]),
            "target_median": med, "target_lo": lo, "target_hi": hi,
            "null_median": float(nul["score"].median()) if len(nul) else float("nan"),
            "floor": floor,
            "rate_floor": float((tgt["score"] < floor).mean()) if len(nul) else float("nan"),
            "floor_p5": floor5,
            "rate_p5": float((tgt["score"] < floor5).mean()) if len(nul) else float("nan"),
            "n_null_values": len(nul),
            "rate_tau": float((tgt["score"] <= TAU).mean()),
            "p_superior": float(np.mean([
                (tgt["score"].values[:, None] < nul["score"].values[None, :]).mean()
            ])) if len(nul) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["block", "dataset_type"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# shape: whose shape does the level curve carry
# ----------------------------------------------------------------------

def ladder_long(df: pd.DataFrame, ladder: str = "value") -> pd.DataFrame:
    """The descriptor's rungs, one row per (neuron, rung), targets only.

    ``lam`` is the answer to "whose shape": 0 = the feature's, 1 = the
    interior's. It is NaN where the rung produced no closed curve.
    """
    assert ladder in ("value", "radial")
    t = df[df.feature_role == "target"]
    base = ["dataset_type", "block", "label", "feature_aspect", "interior_aspect",
            "interior_turned", "interior_scale", "offset_norm", "offset_axis",
            "balance", "model_path", "layer_id", "coord_id", "score",
            "harmonic_objects_shape", "grid_step"]
    frames = []
    for j in range(N_RUNGS):
        cols = {f"harmonic_{ladder}{j}_r0": "r0",
                f"harmonic_{ladder}{j}_dfeat": "dfeat",
                f"harmonic_{ladder}{j}_dint": "dint"}
        part = t[base + list(cols)].rename(columns=cols)
        part["rung"] = j
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["lam"] = out["dfeat"] / (out["dfeat"] + out["dint"])
    out["r0_steps"] = out["r0"] / out["grid_step"]
    out = out.rename(columns={"harmonic_objects_shape": "objects_shape"})
    return out


def shape_by_cell(long: pd.DataFrame, min_objects_shape: float = 0.10,
                  min_rung_steps: float = 4.0) -> pd.DataFrame:
    """Per (cell, rung): the median ``lam`` over neurons, with its support.

    Two gates, both reported rather than applied silently:

    ``min_objects_shape``  the two objects must actually differ in shape, or
                           "whose shape" has no answer. 0.10 sits well under
                           the smallest difference the design has to resolve
                           (0.144, a circle against a 1.5:1 ellipse).
    ``min_rung_steps``     the curve must be big enough for the grid to
                           describe it. At r0 = 4 grid steps the descriptor's
                           error is 0.010, fourteen times under that 0.144.
    """
    ok = long[long["r0_steps"] >= min_rung_steps]
    rows = []
    for (name, rung), g in ok.groupby(["dataset_type", "rung"]):
        usable = g["lam"].notna()
        meta = g.iloc[0]
        rows.append({
            "dataset_type": name, "rung": int(rung),
            "block": meta["block"], "label": meta["label"],
            "feature_aspect": meta["feature_aspect"],
            "interior_aspect": meta["interior_aspect"],
            "interior_turned": meta["interior_turned"],
            "balance": meta["balance"], "offset_norm": meta["offset_norm"],
            "objects_shape": meta["objects_shape"],
            "separable": bool(meta["objects_shape"] >= min_objects_shape),
            "n_neurons": int(usable.sum()),
            "n_models": int(g.loc[usable, "model_path"].nunique()),
            "lam": float(g["lam"].median()),
            "lam_q25": float(g["lam"].quantile(0.25)),
            "lam_q75": float(g["lam"].quantile(0.75)),
            "dfeat": float(g["dfeat"].median()),
            "dint": float(g["dint"].median()),
            "r0": float(g["r0"].median()),
        })
    return pd.DataFrame(rows).sort_values(["block", "dataset_type", "rung"]).reset_index(drop=True)


def position_table(df: pd.DataFrame, rung: str = "mid") -> pd.DataFrame:
    """Block D's question: does the level curve sit on the feature's centre or
    on the displaced interior's?

    ``harmonic_offset_x/y`` is the middle-rung curve's centroid MINUS the
    interior's centre, so the curve's own position is that plus the offset, and

        lam_pos = (curve centre . offset direction) / |offset|

    reads 0 at the feature's centre and 1 at the interior's.
    """
    t = df[(df.feature_role == "target") & (df.offset_norm > 0)].copy()
    ux = t["offset_x"] / t["offset_norm"]
    uy = t["offset_y"] / t["offset_norm"]
    cx = t["harmonic_offset_x"] + t["offset_x"]
    cy = t["harmonic_offset_y"] + t["offset_y"]
    t["lam_pos"] = (cx * ux + cy * uy) / t["offset_norm"]
    return t


# ----------------------------------------------------------------------
# the datasets themselves, for drawing
# ----------------------------------------------------------------------

def _interior_boundary(interior: dict):
    """The interior's boundary as points -- the ladder's inner anchor.

    NOT the filled class-1 cloud. The ladder's ends are the field's mean value
    on each object, the run computed the inner one on the ellipse's boundary,
    and the field varies across the disk -- so anchoring on the cloud would put
    the rungs at different levels than the numbers in the CSV describe.
    """
    from analysis.harmonic_ladder import _interior_boundary as boundary
    return boundary(interior)


def dataset_points(cell: str, instance_id: int = 0):
    """(X, y, feature points, null features) of a stored cell, for a picture."""
    from management.experiment_management import (load_dataset_instance,
                                                  get_feature_points)
    inst = load_dataset_instance(
        f"data/dataset_instances/{cell}/{cell}_{instance_id}")
    X, y = inst["X"], inst["y"]
    A = get_feature_points(inst, "ring")
    return X, y, A, inst


# ----------------------------------------------------------------------
# elongation on a common scale
# ----------------------------------------------------------------------

def _ellipse_a2(aspect: float) -> float:
    """The second harmonic of an ellipse of the given aspect ratio."""
    import torch
    from analysis.shape_descriptors import fourier_descriptor

    th = torch.linspace(0, 2 * math.pi, 4001)[:-1]
    pts = torch.stack([aspect * torch.cos(th), torch.sin(th)], dim=1)
    return fourier_descriptor(pts)["amplitude"][2]


_A2_GRID = None


def a2_to_aspect(a2):
    """Read an ``a2`` amplitude back as an aspect ratio.

    The descriptor's ``a2`` is the natural measure of "how elongated", but it
    is not a ratio and nobody has an intuition for it. Inverting the map
    aspect -> a2 puts the level CURVE on the same axis as the objects, which is
    the only way to say "the curve is rounder than either of them" and be
    believed.
    """
    global _A2_GRID
    if _A2_GRID is None:
        xs = np.array([1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
                       6.0, 8.0, 10.0, 14.0])
        _A2_GRID = (xs, np.array([_ellipse_a2(float(x)) for x in xs]))
    xs, ys = _A2_GRID
    # np.interp clamps, so anything past the end of the grid comes back AS the
    # end. The grid reaches 14:1, well beyond anything block A produces.
    return np.interp(a2, ys, xs)


def curve_shape_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: the level curve's own elongation, in ``a2`` and in aspect.

    Taken at the MIDDLE rung of the value ladder -- the level chosen
    symmetrically between the two objects, so neither is favoured -- and over
    every neuron that produced a closed curve there, not only the best-scoring
    one (which would beg the question).
    """
    t = df[df.feature_role == "target"]
    rows = []
    for name, g in t.groupby("dataset_type"):
        a2 = g["harmonic_a2_mid"]
        meta = g.iloc[0]
        rows.append({
            "dataset_type": name, "block": meta["block"], "label": meta["label"],
            "feature_aspect": meta["feature_aspect"],
            "interior_aspect": meta["interior_aspect"],
            "interior_turned": meta["interior_turned"],
            "balance": meta["balance"], "offset_norm": meta["offset_norm"],
            "offset_axis": meta["offset_axis"],
            "n_curves": int(a2.notna().sum()),
            "a2": float(a2.median()),
            "curve_aspect": float(a2_to_aspect(a2.median())),
            "a2_q25": float(a2.quantile(0.25)), "a2_q75": float(a2.quantile(0.75)),
        })
    return pd.DataFrame(rows).sort_values(["block", "dataset_type"]).reset_index(drop=True)


def marginal_effect(table: pd.DataFrame, value: str = "curve_aspect") -> pd.DataFrame:
    """How far ``value`` moves along each axis of block A, averaged over the other.

    The comparison the block exists to make: the feature's elongation and the
    interior's elongation are two axes of the same square, so the honest way to
    say which one drives the curve is to average out the other and compare the
    two ranges.
    """
    A = table[table.block == "A"]
    out = []
    for axis, other in (("interior_aspect", "feature_aspect"),
                        ("feature_aspect", "interior_aspect")):
        m = A.groupby(axis)[value].mean()
        out.append({"ось": "внутренность" if axis == "interior_aspect" else "фича",
                    "1.0": m.get(1.0), "1.5": m.get(1.5), "2.5": m.get(2.5),
                    "4.0": m.get(4.0), "размах": float(m.max() - m.min()),
                    "монотонно": bool(m.is_monotonic_increasing or m.is_monotonic_decreasing)})
    return pd.DataFrame(out)


def phase_pairs(cells: pd.DataFrame, shape_mid: pd.DataFrame,
                curves: pd.DataFrame) -> pd.DataFrame:
    """Block A against block A-perp, cell by matching cell.

    The pair shares its FEATURE bit for bit -- same aspect, same points, same
    null placements -- and differs only in whether the interior lies along it
    or across it. Any difference between the two columns is therefore
    attributable to the interior alone, which is the sharpest statement the
    experiment can make.
    """
    rows = []
    for fa in (1.5, 2.5, 4.0):
        for ia in (1.5, 2.5, 4.0):
            sel = lambda t, b: t[(t.block == b) & (t.feature_aspect == fa)
                                 & (t.interior_aspect == ia)]
            ca, cp = sel(cells, "A"), sel(cells, "A⊥")
            sa, sp = sel(shape_mid, "A"), sel(shape_mid, "A⊥")
            va, vp = sel(curves, "A"), sel(curves, "A⊥")
            if not (len(ca) and len(cp)):
                continue
            rows.append({
                "feature_aspect": fa, "interior_aspect": ia,
                "objects_phase": float(sa.objects_shape.iloc[0]),
                "objects_cross": float(sp.objects_shape.iloc[0]),
                "rate_phase": float(ca.rate_floor.iloc[0]),
                "rate_cross": float(cp.rate_floor.iloc[0]),
                "score_phase": float(ca.target_median.iloc[0]),
                "score_cross": float(cp.target_median.iloc[0]),
                "lam_phase": float(sa.lam.iloc[0]), "lam_cross": float(sp.lam.iloc[0]),
                "aspect_phase": float(va.curve_aspect.iloc[0]),
                "aspect_cross": float(vp.curve_aspect.iloc[0]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# the level curves themselves, for a picture
# ----------------------------------------------------------------------

def best_neuron_of(best: pd.DataFrame, cell: str, which: str = "median") -> dict:
    """The (model, layer, neuron) to draw for a cell.

    ``which="median"`` picks the model whose best neuron is the cell's MEDIAN
    model -- the typical case, which is what the tables report. ``"best"``
    picks the best of the fifty, which is the strongest case the cell has and
    is worth showing only when the point is that even it fails.
    """
    g = best[(best.dataset_type == cell) & (best.feature_role == "target")]
    if which == "best":
        row = g.loc[g["score"].idxmin()]
    else:
        order = g["score"].sort_values()
        row = g.loc[order.index[len(order) // 2]]
    return {"model_path": row["model_path"], "layer_id": int(row["layer_id"]),
            "coord_id": int(row["coord_id"]), "score": float(row["score"])}


def neuron_field(cell: str, model_index: int = 0, layer_id: int = 3,
                 coord_id: int = 0, instance_id: int = 0, model_path=None):
    """(FieldSample, model, dataset) for one neuron of one stored model.

    Used only for illustration: the tables never re-evaluate a field, they read
    the run. The window and resolution come from the plan, so a curve drawn
    here is the same curve the descriptor measured.
    """
    from analysis.fields import evaluate_field_on_grid
    from analysis.stat_aggregation import find_model_paths_for_dataset
    from management.experiment_management import load_model_instance
    from models.NN_analysis import ScalarProjection, TruncatedMLPFunction
    from experiments.exp5_plan import eval_window
    from experiments.exp5_plan import derived_plan

    plan = {c["name"]: c for c in derived_plan()}
    window = {"bounds": plan[cell]["bounds"], "resolution": plan[cell]["resolution"]}

    if model_path is None:
        paths = sorted(find_model_paths_for_dataset(cell, instance_id=instance_id))
        assert paths, f"no stored models for {cell}"
        model_path = paths[model_index % len(paths)]
    model, _ = load_model_instance(model_path, eval_mode=True)
    fn = ScalarProjection(TruncatedMLPFunction(model, layer_id, kind="post"),
                          mode="coord", coord_id=coord_id)
    field = evaluate_field_on_grid(fn, window["bounds"], window["resolution"])
    return field, model, window


def enclosing_curve_at_mid(cell: str, model_index: int = 0, layer_id: int = 3,
                           coord_id: int = 0, model_path=None):
    """The level curve the descriptor's middle rung describes, as points."""
    import torch
    from analysis.shape_descriptors import (level_ladder, level_curves,
                                            enclosing_curve)
    from experiments.exp5_plan import derived_plan

    plan = {c["name"]: c for c in derived_plan()}
    cellrec = plan[cell]
    field, _, _ = neuron_field(cell, model_index, layer_id, coord_id,
                               model_path=model_path)
    X, y, A, inst = dataset_points(cell)
    centre = torch.tensor(cellrec["interior"]["offset"], dtype=torch.float32)
    levels = level_ladder(field, A, _interior_boundary(cellrec["interior"]),
                          n_steps=N_RUNGS)
    return enclosing_curve(level_curves(field, float(levels[N_RUNGS // 2])), centre)


# ----------------------------------------------------------------------
# the two blocks that need their own shape of table
# ----------------------------------------------------------------------

C_GEOMETRIES = [
    ("g5_f040_i010_s100", "фича 4:1, внутренность круглая"),
    ("g5_f010_i040_s100", "фича круглая, внутренность 4:1"),
    ("g5_f040_i040_s100", "оба 4:1, в фазе"),
    ("g5_f040_i040T_s100", "оба 4:1, поперёк"),
]


def balance_rows(cells: pd.DataFrame, curves: pd.DataFrame,
                 shape_mid: pd.DataFrame) -> pd.DataFrame:
    """Block C: four geometries at three class balances.

    The middle balance is not a cell of its own -- it IS the corresponding cell
    of block A, reused. That is deliberate: the balance axis then passes
    through a point whose geometry has already been measured, so a change
    along it cannot be confused with a change of cell.
    """
    rows = []
    for prefix, label in C_GEOMETRIES:
        for balance, suffix in ((0.2, "b0200"), (0.5, "b0500"), (0.8, "b0800")):
            name = f"{prefix}_o+0000+0000_{suffix}"
            c = cells[cells.dataset_type == name]
            v = curves[curves.dataset_type == name]
            s = shape_mid[shape_mid.dataset_type == name]
            if not len(c):
                continue
            rows.append({
                "геометрия": label, "ячейка": name, "баланс": balance,
                "n_внутр": int(balance * 1000), "n_фичи": int((1 - balance) * 1000),
                "скор": float(c.target_median.iloc[0]), "пол": float(c.floor.iloc[0]),
                "доля_св": float(c.rate_floor.iloc[0]),
                "p_sup": float(c.p_superior.iloc[0]),
                "a2_кривой": float(v.a2.iloc[0]) if len(v) else np.nan,
                "аспект_кривой": float(v.curve_aspect.iloc[0]) if len(v) else np.nan,
                "lam": float(s.lam.iloc[0]) if len(s) else np.nan,
            })
    return pd.DataFrame(rows)


def position_rows(cells: pd.DataFrame, pos: pd.DataFrame,
                  shape_mid: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Block D: the pair, the axis it was displaced along, and how far.

    The displacement is reported as a FRACTION of the largest one that pair
    admits, because the absolute room differs by a factor of seven between a
    round pair and a 4:1 one; the fraction is what makes the three steps the
    same experiment in every family.
    """
    lam_pos = pos.groupby("dataset_type")["lam_pos"].median()
    rows = []
    for name, g in cells[cells.block == "D"].groupby("dataset_type"):
        c = g.iloc[0]
        parts = [p.strip() for p in str(c.label).split(",")]
        pair = parts[0]
        axis = parts[1] if len(parts) > 2 else "-"
        frac = float(parts[-1].rstrip("%")) / 100 if parts[-1].endswith("%") else 0.0
        s = shape_mid[shape_mid.dataset_type == name]
        v = curves[curves.dataset_type == name]
        rows.append({
            "пара": pair, "ось": axis, "доля_сдвига": frac, "ячейка": name,
            "сдвиг": float(c.offset_norm),
            "скор": float(c.target_median), "пол": float(c.floor),
            "доля_св": float(c.rate_floor),
            "lam_pos": float(lam_pos.get(name, np.nan)),
            "lam_shape": float(s.lam.iloc[0]) if len(s) else np.nan,
            "аспект_кривой": float(v.curve_aspect.iloc[0]) if len(v) else np.nan,
        })
    out = pd.DataFrame(rows)
    # a zero-offset cell is shared by both axes: it is the same dataset
    zero = out[out["доля_сдвига"] == 0].copy()
    extra = []
    for _, r in zero.iterrows():
        for ax in ("x", "y"):
            if not ((out["пара"] == r["пара"]) & (out["ось"] == ax)
                    & (out["доля_сдвига"] == 0)).any():
                e = r.copy(); e["ось"] = ax; extra.append(e)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    return out.sort_values(["пара", "ось", "доля_сдвига"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# how much the task determines the boundary
# ----------------------------------------------------------------------

def _ellipse_points(a: float, b: float, rotation: float = 0.0,
                    offset=(0.0, 0.0), n: int = 1201):
    import torch

    th = torch.linspace(0, 2 * math.pi, n)[:-1]
    p = torch.stack([a * torch.cos(th), b * torch.sin(th)], dim=1)
    if rotation:
        c, s = math.cos(rotation), math.sin(rotation)
        p = p @ torch.tensor([[c, -s], [s, c]]).T
    return p + torch.tensor(offset, dtype=torch.float32)


def determinacy_index(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Per cell: ``D``, the worst score an ADMISSIBLE boundary can have.

    A network only has to separate the two classes, so any closed curve lying
    in the gap between the interior and the feature is a valid decision
    boundary. The criterion, though, measures the distance to the FEATURE, and
    nothing in the task requires a boundary to hug it. ``D`` is the size of
    that slack, in the criterion's own units:

        D = maxChamfer(interior boundary, feature) / diam(feature)

    -- the score of the worst boundary the task still accepts. It is the
    ceiling on how badly a correct network can do, and it makes the tautology
    objection measurable instead of rhetorical:

    * ``D <= tau``          every admissible boundary passes: the criterion is
                            VACUOUS in that cell and its witness rate carries
                            no information about the network.
    * ``D <= floor``        the same, against the cell's own floor.
    * ``D >> tau``          admissible boundaries can and do fail, so passing
                            says something.

    ``score / D`` then reads as a position in the corridor: 0 on the feature,
    1 at the far wall. That quantity is comparable across geometries in a way
    the raw normalised score is not.
    """
    from analysis.geometry import max_chamfer_distance
    from experiments.exp5_plan import derived_plan

    rows = []
    for cell in derived_plan():
        p = cell["params"]
        fa = p.get("feature_aspect", 1.0)
        feature = _ellipse_points(1.5 * fa, 1.5)
        scale = p.get("interior_scale", 1.0)
        interior = _ellipse_points(
            scale, scale / p.get("interior_aspect", 1.0),
            p.get("interior_rotation", 0.0), p.get("interior_offset", (0.0, 0.0)))
        inradius = 1.5                      # the feature's smallest half-width
        reach = float(interior.norm(dim=1).max())
        rows.append({
            "dataset_type": cell["name"],
            "D": float(max_chamfer_distance(interior, feature)["score"]) / (3.0 * fa),
            "circle_suffices": bool(reach <= inradius),
            "interior_reach": reach,
            "feature_inradius": inradius,
        })
    return pd.DataFrame(rows)


def ladder_curves(cell: str, model_path=None, layer_id: int = 3, coord_id: int = 0,
                  n_steps: int = N_RUNGS, model_index: int = 0):
    """The level curve at every rung of the value ladder, as point sets.

    The tables report numbers about these curves; this returns the curves
    themselves, so a reader can see what a given ``lam`` or ``a2`` is a number
    FOR. Rungs that produced no closed curve come back as ``None``, which is
    itself informative -- the outer rung fails for three neurons in four.
    """
    import torch
    from analysis.shape_descriptors import (level_ladder, level_curves,
                                            enclosing_curve, densify)
    from experiments.exp5_plan import derived_plan

    rec = {c["name"]: c for c in derived_plan()}[cell]
    field, _, _ = neuron_field(cell, model_index, layer_id, coord_id,
                               model_path=model_path)
    X, y, A, _ = dataset_points(cell)
    centre = torch.tensor(rec["interior"]["offset"], dtype=torch.float32)
    levels = level_ladder(field, A, _interior_boundary(rec["interior"]),
                          n_steps=n_steps)
    out = []
    for c in levels:
        curve = enclosing_curve(level_curves(field, float(c)), centre)
        out.append(None if curve is None
                   else densify(curve, 0.5 * float(field.grid_step)))
    return {"levels": [float(c) for c in levels], "curves": out,
            "field": field, "feature": A, "interior_boundary": None}


def harmonic_terms(first: dict, second: dict, max_k: int = 6):
    """Per-harmonic contribution to the shape distance between two descriptors.

    The distance is a single number, and a single number invites the wrong
    picture -- that it is "the difference in elongation". It is not: every
    harmonic contributes, with its PHASE, and the odd ones carry the asymmetry
    that an ellipse does not have but a real level curve does. This returns the
    terms so that can be shown rather than asserted.
    """
    import numpy as np

    terms = []
    for k in range(1, max_k + 1):
        a, pa = first["amplitude"][k], first["phase"][k]
        b, pb = second["amplitude"][k], second["phase"][k]
        u = np.array([a * math.cos(k * pa), a * math.sin(k * pa)])
        v = np.array([b * math.cos(k * pb), b * math.sin(k * pb)])
        terms.append(0.5 * float(((u - v) ** 2).sum()))
    return terms


# ----------------------------------------------------------------------
# what the gap itself allows, forces and suggests
# ----------------------------------------------------------------------

def _radial_function(points, centre, grid):
    """r(theta) of a closed curve about `centre`, sampled on `grid`."""
    import numpy as np

    rel = np.asarray(points) - np.asarray(centre)
    r = np.linalg.norm(rel, axis=1)
    th = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(th)
    return np.interp(grid, th[order], r[order], period=2 * math.pi)


def separating_geometry(n_theta: int = 721, n_centre: int = 41) -> pd.DataFrame:
    """Per cell: what shape the geometry FORCES, and what it SUGGESTS.

    The experiment asks whose shape the level curve carries, and a third
    answer has to be excluded before either of the first two can be believed:
    that the curve is simply the shape the GAP leaves room for. So the gap is
    measured on its own terms, from the plan's geometry alone, with no network
    in sight.

    ``circle_fits``
        is there a disk that contains the interior and lies inside the
        feature? If yes the geometry FORCES nothing: a round level curve
        separates the classes, and any elongation the network shows is its
        own. Searched over centres, not just the concentric one.
    ``mid_aspect``
        the aspect of the gap's MIDLINE -- the curve halfway between the two
        objects in every direction. This is the shape the gap "suggests": the
        most comfortable separator, equidistant from both walls. If the level
        curve followed the gap rather than either object, this is what it
        would look like.
    ``gap_anisotropy``
        max over min of the gap width around the circle. One means an annulus
        of constant width, large means a crescent.
    """
    import numpy as np
    from experiments.exp5_plan import derived_plan

    grid = np.linspace(-math.pi, math.pi, n_theta)
    rows = []
    for cell in derived_plan():
        interior = cell["interior"]
        a_f = 0.5 * float(cell["diameter"])
        b_f = a_f / float(cell["params"].get("feature_aspect", 1.0))
        centre = np.asarray(interior.get("offset", (0.0, 0.0)), dtype=float)

        t = np.linspace(0, 2 * math.pi, 2001)[:-1]
        feature = np.stack([a_f * np.cos(t), b_f * np.sin(t)], axis=1)
        inner = np.stack([interior["a"] * np.cos(t), interior["b"] * np.sin(t)], axis=1)
        if interior.get("rotation"):
            c, s = math.cos(interior["rotation"]), math.sin(interior["rotation"])
            inner = inner @ np.array([[c, -s], [s, c]]).T
        inner = inner + centre

        # the roundest admissible separator: a disk containing the interior and
        # contained in the feature, over a grid of centres
        cx = np.linspace(-0.6 * b_f, 0.6 * b_f, n_centre) + centre[0]
        cy = np.linspace(-0.6 * b_f, 0.6 * b_f, n_centre) + centre[1]
        best, best_lo, best_hi = -np.inf, np.nan, np.nan
        for x in cx:
            for y in cy:
                c0 = np.array([x, y])
                need = np.linalg.norm(inner - c0, axis=1).max()
                room = np.linalg.norm(feature - c0, axis=1).min()
                if room - need > best:
                    best, best_lo, best_hi = room - need, need, room

        # the same question with the centre PINNED to the feature's own centre:
        # a different and stricter test, and the one the earlier draft used
        need0 = np.linalg.norm(inner, axis=1).max()
        room0 = np.linalg.norm(feature, axis=1).min()

        r_f = _radial_function(feature, centre, grid)
        r_i = _radial_function(inner, centre, grid)
        gap = r_f - r_i
        mid = np.stack([0.5 * (r_f + r_i) * np.cos(grid),
                        0.5 * (r_f + r_i) * np.sin(grid)], axis=1)
        rows.append({
            "dataset_type": cell["name"],
            "block": cell["block"],
            "circle_fits": bool(best >= 0),
            "circle_margin": float(best),
            "circle_r_lo": float(best_lo),
            "circle_r_hi": float(best_hi),
            "circle_fits_concentric": bool(room0 >= need0),
            "circle_margin_concentric": float(room0 - need0),
            "mid_aspect": float(curve_aspect(mid)),
            "gap_min": float(gap.min()),
            "gap_max": float(gap.max()),
            "gap_anisotropy": float(gap.max() / max(gap.min(), 1e-6)),
        })
    return pd.DataFrame(rows)


def curve_aspect(points) -> float:
    """The aspect ratio a closed curve reads as, through the descriptor."""
    import torch
    from analysis.shape_descriptors import fourier_descriptor

    desc = fourier_descriptor(torch.tensor(np.asarray(points), dtype=torch.float32))
    return float(a2_to_aspect(np.array([desc["amplitude"][2]]))[0])


# ----------------------------------------------------------------------
# is a bump a mode?
# ----------------------------------------------------------------------

def _kde(x, h, grid):
    import numpy as np

    z = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * z * z).sum(axis=1) / (len(x) * h * math.sqrt(2 * math.pi))


def _n_modes(x, h, grid):
    import numpy as np

    d = _kde(x, h, grid)
    return int(((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])).sum())


def modality(values, n_boot: int = 200, subsample: int = 1200, seed: int = 0,
             n_grid: int = 512) -> dict:
    """Silverman's critical-bandwidth test: is one mode enough?

    "Two groups" is the kind of claim a picture makes easily and data support
    rarely, so it gets a test rather than an adjective. The critical bandwidth
    ``h_crit`` is the smallest smoothing at which the density stops having more
    than one mode; a genuinely bimodal sample needs a lot of smoothing and so
    has a large ``h_crit``. How large is "large" is calibrated by a smoothed
    bootstrap from the unimodal density at ``h_crit`` itself: ``p`` is the share
    of resamples that need at least as much smoothing.

    Small ``p`` = one mode is not enough. Large ``p`` = the bump is inside what
    a single mode produces by chance, and the honest word is "tail", not "mode".
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) > subsample:
        x = rng.choice(x, subsample, replace=False)
    if len(x) < 50:
        return {"n": len(x), "h_crit": float("nan"), "h_rot": float("nan"),
                "modes_at_rot": 0, "p_unimodal": float("nan")}

    sd = float(x.std(ddof=1))
    grid = np.linspace(x.min() - 3 * sd, x.max() + 3 * sd, n_grid)
    h_rot = 1.06 * sd * len(x) ** (-0.2)             # Silverman's rule of thumb

    lo, hi = 1e-4 * sd, 3.0 * sd
    for _ in range(40):                               # bisect for the critical h
        mid = 0.5 * (lo + hi)
        if _n_modes(x, mid, grid) > 1:
            lo = mid
        else:
            hi = mid
    h_crit = hi

    over = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        y = x[idx] + h_crit * rng.standard_normal(len(x))
        y = x.mean() + (y - y.mean()) / math.sqrt(1 + h_crit ** 2 / sd ** 2)
        over += int(_n_modes(y, h_crit, grid) > 1)
    return {"n": len(x), "h_crit": float(h_crit), "h_rot": float(h_rot),
            "h_ratio": float(h_crit / h_rot),
            "modes_at_rot": _n_modes(x, h_rot, grid),
            "p_unimodal": float(over / n_boot)}


def model_index_map(cells) -> pd.DataFrame:
    """(dataset_type, model_index) -> model_path, the ordering the ladder used.

    ``analysis.harmonic_ladder`` keys models by position in
    ``sorted(find_model_paths_for_dataset(...))`` while the run's CSV keys them
    by path. Joining the two tables needs this, and it has to be the SAME sort
    or a neuron's spectrum would be attached to another model's score.
    """
    from analysis.stat_aggregation import find_model_paths_for_dataset

    rows = []
    for cell in cells:
        for i, path in enumerate(sorted(find_model_paths_for_dataset(cell, instance_id=0))):
            rows.append({"dataset_type": cell, "model_index": i, "model_path": str(path)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# the level curve as a dilation of the interior
# ----------------------------------------------------------------------

def interior_boundary_points(interior: dict, n: int = 721):
    """The interior's boundary as a numpy array."""
    import numpy as np

    t = np.linspace(0, 2 * math.pi, n)[:-1]
    b = np.stack([interior["a"] * np.cos(t), interior["b"] * np.sin(t)], axis=1)
    if interior.get("rotation"):
        c, s = math.cos(interior["rotation"]), math.sin(interior["rotation"])
        b = b @ np.array([[c, -s], [s, c]]).T
    return b + np.asarray(interior.get("offset", (0.0, 0.0)), dtype=float)


def dilated_interior(interior: dict, delta: float, n_theta: int = 721,
                     n_bisect: int = 40, n_boundary: int = 721):
    """The curve at distance ``delta`` OUTSIDE the interior's boundary.

    Not the ellipse with both semi-axes grown by ``delta`` -- that is a
    different and wrong object once ``delta`` exceeds the smallest radius of
    curvature, which for a 4:1 ellipse happens almost immediately. This is the
    true outward morphological dilation.

    The construction matters because dilation is a ROUNDING operation: the
    aspect of the dilated body is always closer to 1 than the aspect of the
    body, and the more room there is the rounder it gets. That is the shape of
    the effect experiment 5 measures, so it is worth having as an explicit
    prediction rather than as a metaphor.

    Implementation note, because the obvious version is wrong. The unsigned
    distance to the boundary is large both INSIDE and outside, so bisecting
    "is d > delta" along a ray searches a non-monotone predicate and can
    converge to the inner offset -- or, if the halving sequence steps over the
    band entirely, to the centre. The search is therefore started at the
    boundary itself, where d is genuinely increasing outward.
    """
    import numpy as np

    boundary = interior_boundary_points(interior, n_boundary)
    centre = np.asarray(interior.get("offset", (0.0, 0.0)), dtype=float)
    angles = np.linspace(-math.pi, math.pi, n_theta)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    r_boundary = _radial_function(boundary, centre, angles)
    lo = r_boundary.copy()                       # on the boundary, d = 0 <= delta
    hi = r_boundary + 2.0 * delta + 1e-6         # far enough out that d > delta
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        pts = centre + mid[:, None] * u
        d = np.sqrt(((pts[:, None, :] - boundary[None, :, :]) ** 2).sum(-1)).min(1)
        inside = d <= delta
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return centre + (0.5 * (lo + hi))[:, None] * u


def dilation_model(observed: pd.DataFrame, n_theta: int = 721,
                   n_grid: int = 14) -> pd.DataFrame:
    """Fit the dilation to the curve's SIZE, then predict its SHAPE.

    ``observed`` is indexed by ``dataset_type`` and carries the median ``r0``
    and ``aspect`` of the measured level curves. For each cell the dilation
    radius is chosen so that the dilated interior has the observed ``r0`` --
    one number, fitted to size alone -- and the aspect that comes out is then a
    PREDICTION, compared against the observed aspect it never saw.
    """
    import numpy as np
    import torch
    from analysis.shape_descriptors import fourier_descriptor
    from experiments.exp5_plan import derived_plan

    plan = {c["name"]: c for c in derived_plan()}
    rows = []
    for name, row in observed.iterrows():
        interior = plan[name]["interior"]
        # r0(delta) is monotone, so a small grid plus interpolation beats
        # bisecting the expensive function twenty times over
        grid = np.linspace(0.01, 3.0, n_grid)
        radii = [fourier_descriptor(torch.tensor(dilated_interior(interior, d, n_theta),
                                                 dtype=torch.float32))["r0"] for d in grid]
        delta = float(np.interp(row["r0"], radii, grid))
        desc = fourier_descriptor(torch.tensor(dilated_interior(interior, delta, n_theta),
                                               dtype=torch.float32))
        hi = delta
        rows.append({"dataset_type": name, "delta": hi,
                     "predicted_aspect": float(a2_to_aspect(
                         np.array([desc["amplitude"][2]]))[0]),
                     "observed_aspect": float(row["aspect"]),
                     "observed_r0": float(row["r0"])})
    return pd.DataFrame(rows)
