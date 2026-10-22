"""Read-out for experiment 4 — density versus topology.

A thin layer over the per-neuron CSV, the same shape as ``analysis.arch_sweep``
is for experiment 2. Everything that produces a number or a figure for the
report lives here; the notebook only calls it and reads the answer.

Experiment 4 varies density by GEOMETRY at a constant point count. The circle
of experiments 1 and 2 becomes a ring of width ``w`` (uniform by area over
``[1.5 - w/2, 1.5 + w/2]``, so its area is exactly ``3*pi*w`` and its density is
``n_ring / (3*pi*w)``); its interior becomes a disk of radius ``R``. Class
balance therefore never moves with density and the support never degenerates.
``w = 0`` reproduces experiment 1's circle exactly.

Four things follow from that construction and shape every function here.

**A cell is (task, w, R, rho, T)** — task relevance, ring width, disk radius,
class balance ``n_ring/n_disk`` and total point count. The dataset name encodes
all five, and ``cell_of`` parses it back.

**Every cell carries up to four measurement targets.** Where the ring has width
there are four (the ring itself and the circles at its inner edge, midline and
outer edge); where ``w = 0`` they coincide and only the midline is registered.
``family`` names which of the four a row belongs to, for target and null rows
alike.

**Each target owns its nulls, so each target owns its threshold.** A null is a
rigid motion of the target it is the floor for, so a ring-shaped target has
ring-shaped nulls carrying the same shape-induced floor, and the contrast
cancels it. The threshold is therefore read off that target's own nulls in that
same cell (``null_percentile_threshold``, a 5 % false-positive rate) and never
inherited from experiment 1's tau = 0.1. Both are reported: ``rate`` uses the
target's own threshold, ``rate_tau`` the frozen 0.1, which is comparable across
experiments but only meaningful for a circle-shaped target.

**The unit of observation is (model, feature), valued by its BEST neuron**, and
intervals bootstrap over MODELS. Both are frozen conventions from experiment 1.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.calibration import (
    bootstrap_over_models,
    null_percentile_threshold,
    superiority_probability,
)

DEFAULT_RESULTS_DIR = "analysis/results/experiments/exp4_density_sweep"

#: The normalised structural score is the verdict column; smaller is better.
SCORE = "structural_score_normalized_by_diameter"

#: The homological margin runs the other way: larger is better.
MARGIN = "homological_persistence_margin"

#: Experiment 1's threshold. Kept for comparability, not used as THE rule --
#: it was tuned for a circle-shaped target and a ring does not inherit it.
TAU = 0.1

#: False-positive rate the per-target threshold is built to have.
#
# 1 %, not 5 %. Measured out of sample (threshold from one half of the models,
# rate on the other), the 5th percentile fires on 6-7 % of null placements while
# the 1st fires on 2-3 %, and the sensitivity that buys costs 2.5 points on the
# verdict target (0.993 -> 0.968). False positives are the expensive error here,
# so the trade is taken. What tightening does NOT touch is the 2 % of models
# that witness the task-irrelevant TARGET: those clear even the strictest
# threshold, and they are adjudicated by the continuity test, not by tau.
FPR = 0.01

#: pandas parses a bare `null` cell as NaN; feature_role must be read verbatim.
_CONVERTERS = {"feature_role": lambda v: v}

#: The ring's midline. Its diameter, 3.0, is what N_A is measured against.
R_MID = 1.5

#: The four measurement targets, ordered from the disk outwards.
FAMILIES = ("t_inner", "t_mid", "t_outer", "t_ring")

_CELL = re.compile(r"^d4_(rel|irr)_w(\d{2})_R(\d{3})_r(\d{3})_T(\d+)$")


# ----------------------------------------------------------------------
# What a cell is
# ----------------------------------------------------------------------

def cell_of(dataset_type: str) -> dict:
    """The five coordinates of a cell, parsed back out of its name."""
    m = _CELL.match(dataset_type)
    assert m, f"not an experiment 4 dataset name: {dataset_type}"
    task, w, disk_r, rho, total = m.groups()
    return {
        "task": "relevant" if task == "rel" else "irrelevant",
        "w": int(w) / 10.0,
        "R": int(disk_r) / 100.0,
        "rho": int(rho) / 100.0,
        "T": int(total),
    }


def family_of(feature_name: str) -> str:
    """Which of the four targets a feature belongs to, target or null alike.

    Null features are named ``n_<target>_<placement>``, and both parts can
    contain underscores, so the family is matched from the front rather than
    split off the back.
    """
    stem = feature_name[2:] if feature_name.startswith("n_") else feature_name
    for fam in FAMILIES:
        if stem == fam or stem.startswith(fam + "_"):
            return fam
    raise ValueError(f"no target family in {feature_name!r}")


def inner_edge(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the VERDICT target — the ring's inner edge (notebook §2.6).

    The inner edge sits at radius ``1.5 - w/2``, so at ``w = 0`` the ring
    degenerates into the circle itself and the inner edge IS that circle; the
    data records it under ``t_mid``, because at zero width the three
    concentric probes coincide. Selecting the verdict target therefore means
    ``t_inner`` wherever a ring exists and ``t_mid`` where it does not, and
    this helper is the one place that rule is written down.

    Takes any table carrying ``w`` and ``family`` — ``best_by_model`` output,
    ``cell_table`` output, or the raw per-neuron frame.
    """
    w, fam = table["w"], table["family"]
    return table[((w > 0) & (fam == "t_inner")) | ((w == 0) & (fam == "t_mid"))]


def ring_density(w: float, n_ring: int) -> float:
    """Points per unit area on the ring; infinite at w = 0, which is a curve."""
    return math.inf if w == 0 else n_ring / (3.0 * math.pi * w)


def disk_density(disk_r: float, n_disk: int) -> float:
    return n_disk / (math.pi * disk_r * disk_r)


def clearance(w: float, disk_r: float) -> float:
    """Empty band between the disk's edge and the ring's inner edge."""
    return (R_MID - w / 2.0) - disk_r


def n_ring_of(rho: float, total: int) -> int:
    """How the recipe splits ``T`` points between the two classes."""
    return int(round(total * rho / (1.0 + rho)))


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_neuron_stats(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Every ``neuron_stats_*.csv`` of the run, with the derived columns.

    Added: the five cell coordinates, ``family``, ``score`` and ``margin`` under
    stable names, and the two densities and the clearance of the cell.
    """
    results_dir = Path(results_dir)
    # With per-dataset eval_overrides the runner writes one CSV per
    # architecture x dataset instead of one per architecture; glob both layouts.
    #
    # The gzipped twin exists because this run's table is 272 MB and GitHub
    # refuses a file over 100 MB, so only `neuron_stats_*.csv.gz` (38 MB) is in
    # the repository. pandas reads either by extension; the raw CSV wins when
    # both are present, so a local run is never slowed down by decompression.
    found: dict[str, Path] = {}
    for pattern in ("neuron_stats_*.csv.gz", "*/neuron_stats_*.csv.gz",
                    "neuron_stats_*.csv", "*/neuron_stats_*.csv"):
        for path in sorted(results_dir.glob(pattern)):
            found[path.name.removesuffix(".gz")] = path
    paths = [found[k] for k in sorted(found)]
    assert paths, f"no neuron_stats_*.csv[.gz] under {results_dir}"

    df = pd.concat([pd.read_csv(p, converters=_CONVERTERS) for p in paths],
                   ignore_index=True)

    cells = pd.DataFrame([{"dataset_type": name, **cell_of(name)}
                          for name in df["dataset_type"].unique()])
    df = df.merge(cells, on="dataset_type", how="left")

    df["family"] = df["feature_name"].map(family_of)
    df["score"] = df[SCORE]
    df["margin"] = df[MARGIN] if MARGIN in df else np.nan

    n_ring = df.apply(lambda r: n_ring_of(r["rho"], r["T"]), axis=1)
    df["n_ring"] = n_ring
    df["n_disk"] = df["T"] - n_ring
    df["ring_density"] = [ring_density(w, n) for w, n in zip(df["w"], n_ring)]
    df["disk_density"] = [disk_density(R, n) for R, n in zip(df["R"], df["n_disk"])]
    df["clearance"] = [clearance(w, R) for w, R in zip(df["w"], df["R"])]
    return df


def best_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse neurons: one row per (cell, model, feature), its BEST neuron.

    Best means the smallest structural score and, independently, the largest
    homological margin — the two criteria are allowed to pick different
    neurons, exactly as in experiments 1 and 2.
    """
    keys = ["dataset_type", "task", "w", "R", "rho", "T", "n_ring", "n_disk",
            "ring_density", "disk_density", "clearance",
            "model_path", "feature_name", "family", "feature_role"]
    return (df.groupby(keys, as_index=False)
              .agg(score=("score", "min"), margin=("margin", "max")))


# ----------------------------------------------------------------------
# The sweep as a table
# ----------------------------------------------------------------------

def _median(values) -> float:
    """Median that returns NaN for an all-NaN input without warning.

    The homological criterion legitimately returns no bar at all — on the
    task-irrelevant cells whole groups have none — and taking a median of that
    is a NaN, not a mistake. pandas gets there via ``nanmedian`` and warns
    about an empty slice on the way, which would put a scary RuntimeWarning in
    front of a correct answer.
    """
    values = pd.Series(values).dropna()
    return float(values.median()) if len(values) else float("nan")


def cell_table(best: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per (cell, target family): the verdict and how sure it is.

    Columns:
      ``threshold``   ``FPR``-percentile (1 %) of THIS target's nulls in THIS cell
      ``rate``        share of models whose best neuron clears that threshold
      ``rate_lo/hi``  90 % bootstrap interval over models for that share
      ``rate_tau``    the same share at experiment 1's fixed tau = 0.1
      ``med_target``  median best-neuron score over models (smaller is better)
      ``med_null``    median over the null placements, i.e. where chance sits
      ``p_superior``  P(target scores better than a congruent copy elsewhere);
                      0.5 is indistinguishable, 1.0 is no overlap at all
      ``med_margin``  median homological margin (larger is better)
    """
    rows = []
    for keys, block in best.groupby(
            ["dataset_type", "task", "w", "R", "rho", "T", "family"], sort=True):
        target = block[block["feature_role"] == "target"]
        nulls = block[block["feature_role"] == "null"]
        if not len(target) or not len(nulls):
            continue

        threshold = null_percentile_threshold(
            nulls["score"], false_positive_rate=FPR, lower_is_better=True)

        hits = (target["score"] <= threshold).to_numpy()
        est, lo, hi = bootstrap_over_models(
            hits, lambda u: float(np.mean(u)), n_boot=n_boot, seed=seed)

        ds, task, w, disk_r, rho, total, fam = keys
        rows.append({
            "cell": ds, "task": task, "w": w, "R": disk_r, "rho": rho, "T": total,
            "family": fam,
            "n_models": len(target),
            "n_ring": int(block["n_ring"].iloc[0]),
            "n_disk": int(block["n_disk"].iloc[0]),
            "ring_density": float(block["ring_density"].iloc[0]),
            "disk_density": float(block["disk_density"].iloc[0]),
            "clearance": float(block["clearance"].iloc[0]),
            "threshold": threshold,
            "rate": est, "rate_lo": lo, "rate_hi": hi,
            "rate_tau": float((target["score"] <= TAU).mean()),
            "med_target": float(target["score"].median()),
            "min_target": float(target["score"].min()),
            "med_null": float(nulls["score"].median()),
            "p_superior": superiority_probability(target["score"], nulls["score"]),
            "med_margin": _median(target["margin"]),
            "med_margin_null": _median(nulls["margin"]),
        })
    return pd.DataFrame(rows).sort_values(["task", "family", "w", "R", "rho", "T"])


def axis_table(cells: pd.DataFrame, axis: str, family: str = "t_mid",
               task: str = "relevant", **fixed) -> pd.DataFrame:
    """Collapse the grid onto one axis, holding the others fixed.

    ``fixed`` pins the other coordinates (e.g. ``R=0.5, T=1000``); whatever is
    left varies along ``axis`` and is averaged over. Use it to answer "does the
    verdict move along w?" without eyeballing a pivot table.
    """
    sel = cells[(cells["task"] == task) & (cells["family"] == family)]
    for key, value in fixed.items():
        sel = sel[sel[key] == value]
    return (sel.groupby(axis)
               .agg(cells=("cell", "nunique"),
                    threshold=("threshold", "median"),
                    rate=("rate", "mean"),
                    rate_lo=("rate_lo", "mean"), rate_hi=("rate_hi", "mean"),
                    median=("med_target", "median"),
                    p_superior=("p_superior", "median"),
                    margin=("med_margin", "median"))
               .reset_index())


def task_contrast(cells: pd.DataFrame, family: str = "t_mid") -> pd.DataFrame:
    """The headline comparison: the same target under both tasks."""
    sel = cells[cells["family"] == family]
    return (sel.groupby("task")
               .agg(cells=("cell", "nunique"),
                    rate=("rate", "median"), rate_min=("rate", "min"),
                    rate_max=("rate", "max"),
                    rate_tau=("rate_tau", "median"),
                    median=("med_target", "median"),
                    best=("min_target", "min"),
                    floor=("med_null", "median"),
                    p_superior=("p_superior", "median"),
                    margin=("med_margin", "median"))
               .reset_index())


# ----------------------------------------------------------------------
# The controls that fall out of the grid for free
# ----------------------------------------------------------------------

def matched_clearance_pairs(cells: pd.DataFrame, family: str = "t_mid",
                            total: int = 1000) -> pd.DataFrame:
    """Cells that share a clearance while differing in (w, R).

    The clearance ``1.5 - w/2 - R`` moves with both manipulated variables, so a
    disk effect and a clearance effect are not separated by the grid alone.
    These pairs separate them: inside a pair the clearance is identical, so a
    difference cannot be the clearance.
    """
    sel = cells[(cells["task"] == "relevant") & (cells["family"] == family)
                & (cells["T"] == total) & (cells["R"] < 1.0)].copy()
    sel["clearance"] = sel["clearance"].round(3)
    out = []
    for (gap, rho), block in sel.groupby(["clearance", "rho"]):
        if block["cell"].nunique() < 2:
            continue
        for _, row in block.iterrows():
            out.append({"gap": gap, "rho": rho, "w": row["w"], "R": row["R"],
                        "ring density": row["ring_density"],
                        "disk density": row["disk_density"],
                        "median": row["med_target"], "rate": row["rate"]})
    return pd.DataFrame(out)


def matched_density_pairs(cells: pd.DataFrame, family: str = "t_mid",
                          tol: float = 0.5) -> pd.DataFrame:
    """Cells whose RING density agrees while their point counts differ.

    Note what this pair does and does not control. It matches the ring's
    density and nothing else — the disk's density and the class balance move
    with it — so it tests whether ring density ALONE predicts the verdict, not
    whether point count matters with everything else held still.
    """
    sel = cells[(cells["task"] == "relevant") & (cells["family"] == family)
                & (cells["w"] > 0)].copy()
    sel["dens"] = sel["ring_density"].round(0)
    out = []
    for dens, block in sel.groupby("dens"):
        if block["cell"].nunique() < 2 or block["T"].nunique() < 2:
            continue
        for _, row in block.iterrows():
            out.append({"ring density": dens, "T": row["T"], "rho": row["rho"],
                        "w": row["w"], "R": row["R"], "n_ring": row["n_ring"],
                        "disk density": row["disk_density"],
                        "median": row["med_target"], "rate": row["rate"]})
    return pd.DataFrame(out)


def _bar(df: pd.DataFrame, threshold):
    """Resolve a bar into one value per row.

    ``threshold`` is either a number (the same bar everywhere) or a
    ``cell_table``-shaped frame, in which case every row is judged against the
    threshold of ITS OWN cell and target — the rule of §2.7, which is why no
    figure downstream needs a global tau.
    """
    if isinstance(threshold, pd.DataFrame):
        by = {(r.cell, r.family): r.threshold for r in threshold.itertuples()}
        return np.array([by.get((c, f), np.nan)
                         for c, f in zip(df["dataset_type"], df["family"])])
    return float(threshold)


def theorem_control(df: pd.DataFrame, threshold=TAU) -> pd.DataFrame:
    """Layer 0 must give exactly zero witnesses, in every cell.

    A first-layer neuron is an affine function composed with a monotone
    activation, so its level sets are straight lines and cannot reproduce a
    closed curve. Any hit here is an instrument failure, not a finding.
    """
    target = df[df["feature_role"] == "target"].assign(
        _hit=lambda t: t["score"] <= _bar(t, threshold))
    return (target.groupby("layer_id")
                  .agg(neurons=("score", "size"),
                       hits=("_hit", "sum"),
                       min_score=("score", "min"))
                  .reset_index())


def layer_profile(df: pd.DataFrame, task: str = "relevant",
                  threshold=TAU) -> pd.DataFrame:
    """Where in the depth the witnesses sit."""
    sel = df[(df["feature_role"] == "target") & (df["task"] == task)]
    sel = sel.assign(_hit=lambda t: t["score"] <= _bar(t, threshold))
    return (sel.groupby("layer_id")
               .agg(neurons=("score", "size"),
                    hit_rate=("_hit", "mean"),
                    median=("score", "median"))
               .reset_index())


def witness_multiplicity(df: pd.DataFrame, cell: str, family: str = "t_mid",
                         threshold: float | None = None) -> pd.Series:
    """How many neurons of one model witness the target, model by model."""
    sel = df[(df["dataset_type"] == cell) & (df["family"] == family)
             & (df["feature_role"] == "target")]
    bar = TAU if threshold is None else threshold
    return (sel.assign(hit=sel["score"] <= bar)
               .groupby("model_path")["hit"].sum())


# ----------------------------------------------------------------------
# Trainability, which this experiment turned into an outcome
# ----------------------------------------------------------------------

#: Ensembles as the server run first returned them, BEFORE the top-up: models
#: that reached 100 % test accuracy out of 50 attempts. Recorded here because
#: nothing else keeps it — the stored artifacts hold only the models that
#: succeeded, so the failures survive as this table or not at all. Cells absent
#: from the mapping filled all 50 slots on the first pass.
FIRST_PASS_SUCCESSES = {
    "d4_rel_w00_R025_r025_T1000": 43, "d4_rel_w00_R025_r400_T1000": 45,
    "d4_rel_w00_R050_r025_T1000": 42, "d4_rel_w00_R050_r400_T1000": 47,
    "d4_rel_w05_R025_r025_T1000": 36, "d4_rel_w05_R025_r100_T1000": 49,
    "d4_rel_w05_R025_r400_T1000": 45, "d4_rel_w05_R050_r025_T1000": 45,
    "d4_rel_w05_R050_r400_T1000": 44, "d4_rel_w05_R050_r400_T5000": 46,
    "d4_rel_w10_R025_r025_T1000": 36, "d4_rel_w10_R025_r100_T1000": 48,
    "d4_rel_w10_R025_r400_T1000": 44, "d4_rel_w10_R050_r025_T1000": 34,
    "d4_rel_w10_R050_r025_T5000": 48, "d4_rel_w10_R050_r100_T1000": 49,
    "d4_rel_w10_R050_r400_T1000": 46, "d4_rel_w10_R050_r400_T5000": 48,
}


# ----------------------------------------------------------------------
# Criterion 4 (homological) over the same grid
# ----------------------------------------------------------------------

def homology_units(df: pd.DataFrame, layer_min: int = 1) -> pd.DataFrame:
    """One row per (cell, feature, model): what criterion 4 saw in that model.

    ``persistence_margin`` is how far the longest H_1 bar near the target
    dominates the next one (and an absolute floor); positive means a persistent
    loop is there, NaN means the diagram held no bars at all. The criterion
    reads a NEIGHBOURHOOD of the target — 1.5 x its radius — so it answers
    "is there a loop around here", not "does the loop lie on the target". That
    is exactly what makes it complementary to the structural criterion, and
    also why margins may only be compared across cells at a FIXED target: the
    neighbourhood grows with the target's radius.

    Layer 0 is dropped: its level sets are straight lines, so it can carry no
    loop by construction and would only dilute the rates.

    The unit is one (cell, FEATURE, model) — a null placement counts as its own
    feature, exactly as in :func:`best_by_model`, so the two criteria are read
    on the same rows.
    """
    sel = df[df["layer_id"] >= layer_min]
    out = (sel.groupby(["dataset_type", "task", "w", "R", "rho", "T",
                        "family", "feature_name", "feature_role", "model_path"], sort=False)
              .agg(margin=("margin", "max"),
                   n_positive=("margin", lambda s: int((s > 0).sum())),
                   n_diagram=("margin", lambda s: int(s.notna().sum())),
                   n_neurons=("margin", "size"))
              .reset_index())
    out["witness"] = out["margin"].fillna(-np.inf) > 0
    return out


def homology_cells(units: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, target family, role): the same three readings, per cell.

    ``rate`` saturates on relevant cells, so ``med_n_positive`` — how many of a
    model's neurons carry the loop — is kept beside it as the reading that
    still has room to move.
    """
    g = units.groupby(["dataset_type", "task", "w", "R", "rho", "T",
                       "family", "feature_role"], sort=True)
    out = g.agg(n_models=("witness", "size"),
                rate=("witness", "mean"),
                med_margin=("margin", "median"),
                med_n_positive=("n_positive", "median"),
                med_n_diagram=("n_diagram", "median")).reset_index()
    return out.rename(columns={"dataset_type": "cell"})


def trainability_table(cells: pd.DataFrame, attempts: int = 50) -> pd.DataFrame:
    """Share of initialisations that reached the gate, per cell.

    Trainability is not a nuisance here, it is a second outcome measured on the
    same grid: the gate is 100 % test accuracy, so a cell where fewer seeds
    reach it is a cell where the task pins the boundary less firmly. Read it
    beside the witness rate, not instead of it.
    """
    seen = cells[["cell", "task", "w", "R", "rho", "T", "clearance",
                  "ring_density", "disk_density"]].drop_duplicates("cell")
    seen = seen.copy()
    seen["successes"] = [FIRST_PASS_SUCCESSES.get(c, attempts) for c in seen["cell"]]
    seen["attempts"] = attempts
    seen["success rate"] = seen["successes"] / attempts
    return seen.sort_values("success rate")


# ----------------------------------------------------------------------
# Getting the datasets themselves back, for the figures
# ----------------------------------------------------------------------

def dataset_points(cell: str, instance_id: int = 0):
    """Points, labels and registered features of one stored cell.

    The instances are rebuilt from the recipe rather than shipped back from the
    server; the recipe is seeded, and a spot check reproduces a stored score to
    5e-9 (float32 round-off), so they are the same datasets the models saw.
    """
    from management.experiment_management import (
        load_dataset_instance, reconstruct_data_from_structure, get_feature_points,
    )

    path = f"data/dataset_instances/{cell}/{cell}_{instance_id}/"
    stored = load_dataset_instance(path)
    data = reconstruct_data_from_structure(stored)
    features = {name: get_feature_points(stored, name) for name in stored["features"]}
    roles = {name: stored["features"][name]["metadata"]["feature_role"]
             for name in stored["features"]}
    return data.get_data(), data.get_labels(), features, roles


# ----------------------------------------------------------------------
# How tightly the points lie — the quantity this experiment manipulates
# ----------------------------------------------------------------------

def spacing_table(cell_names, instance_id: int = 0) -> pd.DataFrame:
    """Mean distance to the nearest neighbour inside each class, MEASURED.

    This is the experiment's density variable, and it is stated this way rather
    than as points per unit area for a reason. An areal density is undefined on
    the ``w = 0`` end of the sweep, where the ring degenerates to a curve — but
    that endpoint is not singular for anything we do with it: it is experiment
    1's circle, its points have a perfectly ordinary spacing, and the criterion
    treats it like any other cloud. Nearest-neighbour spacing is defined for
    every cell, carries the same unit (length) everywhere, and is directly the
    quantity the criteria care about, since it is what a level-set band width
    and a Chamfer distance are compared against.

    The measured value is not the textbook one. For points scattered uniformly
    at random the mean nearest-neighbour distance is about half of
    ``sqrt(area/n)``, because random points clump where a lattice would not, so
    the formula is reported beside it rather than instead of it.
    """
    from scipy.spatial import cKDTree

    rows = []
    for name in cell_names:
        X, y, _, _ = dataset_points(name, instance_id)
        cell = cell_of(name)
        for label, mask in (("ring", y == 0), ("disk", y == 1)):
            pts = X[mask].numpy()
            if len(pts) < 2:
                continue
            dist, _ = cKDTree(pts).query(pts, k=2)
            area = (3.0 * math.pi * cell["w"] if label == "ring"
                    else math.pi * cell["R"] ** 2)
            rows.append({
                "cell": name, "object": label, **cell,
                "n_ring": n_ring_of(cell["rho"], cell["T"]),
                "points": len(pts),
                "spacing": float(dist[:, 1].mean()),
                "formula": (2 * math.pi * R_MID / len(pts)
                            if label == "ring" and cell["w"] == 0
                            else math.sqrt(area / len(pts)) if area > 0 else float("nan")),
            })
    return pd.DataFrame(rows)


def chamfer_reference_probe(cell: str, n_grid=(250, 1000, 4000, 16000),
                            seed: int = 777) -> dict:
    """Does a band-shaped target carry a floor of its own? Measured, not argued.

    The worry is that ``max_chamfer`` demands set equality, so comparing a
    two-dimensional ring against a one-dimensional curve is bounded below by
    roughly ``w/4`` no matter how good the curve is — which would make the ring
    a degraded target. The probe checks each half of that:

    * two INDEPENDENT samples of the same ring — if the ring carried a floor of
      its own this would not go to zero, and it does, as ``n**-0.5``;
    * the ring against the ideal middle circle — this is the ``w/4`` bound, and
      it is real, but it is a statement about comparing a band to a CURVE;
    * the same for the circle target, as the control.

    What settles it is comparing that bound against the best score the real
    neurons actually reach on the ring: if they get below it, the extracted
    level-set components are not curves but bands, and the bound never binds.
    """
    import torch

    X, y, features, _ = dataset_points(cell)
    params = cell_of(cell)
    w = params["w"]
    assert w > 0, "the probe needs a ring of nonzero width"
    r_in, r_out = R_MID - w / 2, R_MID + w / 2

    def ring_sample(n, s):
        g = torch.Generator().manual_seed(int(s))
        u = torch.rand(n, generator=g)
        r = torch.sqrt(u * (r_out ** 2 - r_in ** 2) + r_in ** 2)
        th = 2 * math.pi * torch.rand(n, generator=g)
        return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1)

    def circle_sample(n, s):
        g = torch.Generator().manual_seed(int(s))
        th = 2 * math.pi * torch.rand(n, generator=g)
        return torch.stack([R_MID * torch.cos(th), R_MID * torch.sin(th)], dim=1)

    def chamfer(a, b):
        from analysis.geometry import max_chamfer_distance
        d = max_chamfer_distance(a, b)
        return float(d["max_chamfer"] if isinstance(d, dict) and "max_chamfer" in d
                     else list(d.values())[0])

    ring, mid = features["t_ring"], features["t_mid"]
    return {
        "w": w,
        "ring vs an independent sample of the ring": chamfer(ring, ring_sample(len(ring), seed)),
        "circle vs an independent sample of the circle":
            chamfer(mid, circle_sample(len(mid), seed)),
        "ring vs the ideal middle circle": chamfer(ring, mid),
        "prediction w/4": w / 4.0,
        "convergence in the number of points": {
            n: chamfer(ring_sample(n, seed), ring_sample(n, seed + 1)) for n in n_grid
        },
    }


def bracket_table(best: pd.DataFrame, cells: pd.DataFrame,
                  task: str = "relevant") -> pd.DataFrame:
    """Where the structure sits: three probing radii in three units.

    Computed PER CONFIGURATION and only then summarised, because a target's
    score varies more between configurations than inside one (see the
    poolability check) and a median over pooled scores would not be the median
    of anything. The floor may be pooled inside a block, but it is kept
    per-configuration here too so that every ratio is formed within one cell.

    The three concentric circles differ in radius, so their scores are not
    automatically comparable — the normalised score divides by each target's own
    diameter. The table therefore carries all three readings (normalised, raw,
    and relative to that target's own floor). The last is dimensionless and
    identical in either unit, since the diameter cancels, so a conclusion drawn
    from it does not depend on the choice of unit at all.
    """
    sel = best[(best["task"] == task) & (best["w"] > 0) & (best["R"] < 1.0)
               & (best["family"] != "t_ring")]
    per_cell = []
    for (ds, w, fam), block in sel.groupby(["dataset_type", "w", "family"]):
        tgt = block[block["feature_role"] == "target"]["score"]
        nul = block[block["feature_role"] == "null"]["score"]
        if not len(tgt) or not len(nul):
            continue
        radius = {"t_inner": R_MID - w / 2, "t_mid": R_MID,
                  "t_outer": R_MID + w / 2}[fam]
        norm, floor = float(tgt.median()), float(nul.median())
        per_cell.append({"cell": ds, "w": w, "target": fam, "radius": radius,
                         "norm_median": norm, "abs_median": norm * 2 * radius,
                         "norm_floor": floor, "abs_floor": floor * 2 * radius,
                         "times better than floor": floor / norm})
    per_cell = pd.DataFrame(per_cell)
    out = (per_cell.groupby(["w", "target", "radius"])
                   [["norm_median", "abs_median", "norm_floor", "abs_floor",
                     "times better than floor"]]
                   .median().reset_index())
    out["diameter"] = 2 * out["radius"]
    return out.sort_values(["w", "radius"]), per_cell


def multiplicity(df: pd.DataFrame, cells, family: str | None = "t_mid",
                 threshold=TAU) -> pd.Series:
    """Neurons per model clearing the bar, for one group of cells.

    ``family=None`` reads the verdict target (:func:`inner_edge`), and
    ``threshold`` may be a ``cell_table`` frame, so the count is made against
    each cell's own bar rather than a shared one.
    """
    sel = df[(df["dataset_type"].isin(cells)) & (df["feature_role"] == "target")]
    sel = inner_edge(sel) if family is None else sel[sel["family"] == family]
    return (sel.assign(hit=sel["score"] <= _bar(sel, threshold))
               .groupby(["dataset_type", "model_path"])["hit"].sum())


def target_agreement(best: pd.DataFrame, cells: pd.DataFrame,
                     task: str = "relevant") -> pd.DataFrame:
    """Do the four targets call the same MODELS witnesses?

    Medians can agree while verdicts disagree model by model. Each target is
    thresholded by its own nulls in its own cell, then the verdicts are compared
    pairwise on the models both saw.
    """
    thr = {(r.cell, r.family): r.threshold for r in cells.itertuples()}
    sel = best[(best["task"] == task) & (best["feature_role"] == "target")
               & (best["w"] > 0)].copy()
    sel["hit"] = [s <= thr.get((c, f), np.nan)
                  for s, c, f in zip(sel["score"], sel["dataset_type"], sel["family"])]
    piv = sel.pivot_table(index=["dataset_type", "model_path"], columns="family",
                          values="hit")
    fams = [f for f in FAMILIES if f in piv]
    out = pd.DataFrame(index=fams, columns=fams, dtype=float)
    for a in fams:
        for b in fams:
            both = piv[[a, b]].dropna() if a != b else piv[[a]].dropna()
            if not len(both):
                out.loc[a, b] = np.nan
            elif a == b:
                out.loc[a, b] = 1.0
            else:
                out.loc[a, b] = float((both.iloc[:, 0] == both.iloc[:, 1]).mean())
    return out


def placement_floors(best: pd.DataFrame) -> pd.DataFrame:
    """The floor, split by where the null was placed.

    Experiment 1 measured that the floor is not a number but a distribution over
    placements (0.165 against 0.368). Keeping the split visible is what stops a
    single pooled percentile from being read as "the" chance level.
    """
    nulls = best[best["feature_role"] == "null"].copy()
    nulls["placement"] = nulls["feature_name"].str.extract(
        r"_(far_right|above_target|in_corridor)$")[0]
    return (nulls.groupby(["task", "family", "placement"])["score"]
                 .agg(["count", "median", "min"]).reset_index())


# ----------------------------------------------------------------------
# The ceiling: what score is achievable on each target at all
# ----------------------------------------------------------------------

def ideal_fields(w: float) -> dict:
    """Analytic scalar fields built to be the best possible detector of each
    target shape, for measuring how good a score is even reachable.

    The criterion takes an arbitrary callable, so the ceiling does not need a
    trained network: it needs the field a perfect neuron would have.

    Which field is ideal depends on the target, and the reason is how the level
    set is extracted. A component is ``{x : |f - c| <= eps}`` with eps chosen
    adaptively as a quantile of the neighbouring grid-value jumps near the
    target — so on a sloped field the band is about two grid steps wide no
    matter what, and a band that wide cannot cover a ring of width 1.0. The
    only way to cover a ring is a field that is FLAT on it: then ``|f - c|``
    stays under any eps across the whole annulus. Hence the plateau family.

    * ``|x|`` — level sets are thin circles: the ideal detector of a CIRCLE.
    * ``tanh(k(r - r_in)) - tanh(k(r - r_out))`` — a flat-topped radial
      plateau, saturated across the annulus: the ideal detector of a RING.
      Sharpness ``k`` trades the plateau's flatness against how much of the
      edge is lost to the grid.
    * ``-dist(x, ring)`` — exactly zero on the annulus, the sharpest possible
      plateau; included as the limiting case.
    """
    import torch

    r_in, r_out = R_MID - w / 2.0, R_MID + w / 2.0

    def radial(x):
        return x.norm(dim=1)

    def neg_dist(x):
        r = x.norm(dim=1)
        return -torch.clamp(torch.maximum(r_in - r, r - r_out), min=0.0)

    def plateau(k):
        def f(x):
            r = x.norm(dim=1)
            return torch.tanh(k * (r - r_in)) - torch.tanh(k * (r - r_out))
        return f

    out = {"|x| — thin circle": radial,
           "−dist to the ring — exact ring": neg_dist}
    for k in (1, 2, 4, 8, 16, 32):
        out[f"tanh plateau, steepness {k}"] = plateau(k)
    return out


#: Characteristic radius of each target family, as a function of the ring width.
def target_radius(family: str, w: float) -> float:
    return {"t_inner": R_MID - w / 2, "t_mid": R_MID, "t_outer": R_MID + w / 2,
            "t_ring": R_MID}[family]


def ceiling_probe(cell: str, spec=None, targets=FAMILIES,
                  exact_level: bool = True) -> pd.DataFrame:
    """Score every ideal field of ``ideal_fields`` against every target.

    The best entry of each column is the CEILING for that target: no neuron can
    do better under this criterion, because the field being scored already is
    the shape.

    **Read the ceiling in ABSOLUTE units.** The score is
    ``chamfer / diam(A)``, and the chamfer a perfect field leaves behind is set
    by the grid step — an absolute length that knows nothing about the target.
    Dividing it by the target's diameter therefore makes the NORMALISED ceiling
    fall like ``1/diam`` as a matter of arithmetic, not of geometry. Measured on
    circles of radius 1.0, 1.5 and 2.0 against a dense reference: the absolute
    ceiling is 0.0333-0.0349 (a 5 % spread around ``h/2 = 0.0339``) while the
    normalised one runs 0.0167-0.0087. Shape sets the ceiling; scale and
    position do not — but only the absolute reading shows that.

    ``exact_level=True`` hands the criterion the one level the ideal field would
    have on the target, instead of letting it scan ``n_levels`` values across
    the field's whole range. That scan is the third contaminant and the least
    obvious: at 101 levels the step for ``|x|`` on this box is 0.106, three
    times the grid step, so where a target's radius happens to fall between two
    scanned levels adds up to 49 % to its measured ceiling — an accident of
    alignment that has nothing to do with the target. The experiment itself runs
    at 101 levels and pays that lottery like every neuron does; the ceiling is
    measured without it so the two contributions stay separable.
    """
    import torch
    from analysis.criterions import structural_criterion

    if spec is None:
        from experiments.registry import EXPERIMENTS
        spec = EXPERIMENTS["exp4_density_sweep"]

    params = cell_of(cell)
    w = params["w"]
    _, _, features, _ = dataset_points(cell)
    base = dict(bounds=spec.eval.bounds, resolution=spec.eval.resolution,
                level_eps=spec.eval.structural_kwargs.get("level_eps", "adaptive"),
                activation=None, threshold=TAU)

    rows = []
    for name, fn in ideal_fields(w).items():
        row = {"field": name}
        for fam in targets:
            if fam not in features:
                continue
            if exact_level:
                radius = target_radius(fam, w)
                level = float(fn(torch.tensor([[radius, 0.0]])).item())
                kwargs = dict(base, levels=[level])
            else:
                kwargs = dict(base, n_levels=spec.eval.structural_kwargs.get("n_levels", 101))
            res = structural_criterion(fn, feature_points=features[fam], **kwargs)
            row[fam] = res["score"]                       # АБСОЛЮТНЫЙ, см. докстроку
            row[fam + "_norm"] = res["score_normalized_by_diameter"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("field")


def per_cell_overlap(best: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Does the target population overlap its own floor, CELL BY CELL?

    Pooling cells before asking this is not allowed: each configuration has its
    own trained models and its own nulls, and a mixture of configurations can
    look separated when no single configuration is, or the reverse. So the
    overlap is computed inside each cell and only then summarised.
    """
    rows = []
    for (ds, fam), block in best.groupby(["dataset_type", "family"]):
        tgt = block[block["feature_role"] == "target"]["score"]
        nul = block[block["feature_role"] == "null"]["score"]
        if not len(tgt) or not len(nul):
            continue
        params = cell_of(ds)
        rows.append({"cell": ds, "family": fam, **params,
                     "target_max": float(tgt.max()), "target_med": float(tgt.median()),
                     "floor_min": float(nul.min()), "floor_med": float(nul.median()),
                     "overlap": float(tgt.max() > nul.min()),
                     "fraction of target above floor_min": float((tgt > nul.min()).mean()),
                     "p_superior": superiority_probability(tgt, nul)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Task relevance measured on the DATA, before any network exists
# ----------------------------------------------------------------------

CHROMATIC_CACHE = "aggregates/chromatic_task_relevance.csv"


def chromatic_task_relevance(cell_names, results_dir: str | Path = DEFAULT_RESULTS_DIR,
                             use_cache: bool = True, instance_id: int = 0) -> pd.DataFrame:
    """Is each configuration task-relevant AT THE LEVEL OF THE DATA?

    Everything else in this experiment measures a trained network. This does
    not: the chromatic alpha complex is built on the coloured point cloud
    itself, so it answers whether the pair (feature, interior) makes the
    feature task-relevant *before* any model exists, and independently of
    whether a model found it.

    The quantity is the KERNEL of the map from the class-0 subcomplex into the
    full complex: a cycle made of ring points that DIES once the disk's points
    are added, i.e. a hole of one class filled by the other. That is exactly
    the definition of a task-relevant topological feature (PLAN.md A.8), and it
    is the ground truth the witness criteria are supposed to be detecting.

    Note the subcomplex class: 0, the ring. The default in
    ``analysis.chromatic`` is 1; here the cycle belongs to class 0 and the
    filling to class 1, so passing the default would ask the opposite question.

    Cached to CSV, because the alpha complex costs ~2 s at 1000 points and
    ~20 s at 5000, and nothing about it changes between runs.
    """
    from analysis.chromatic import kernel_task_relevant_features

    path = Path(results_dir) / CHROMATIC_CACHE
    if use_cache and path.exists():
        cached = pd.read_csv(path)
        if set(cell_names) <= set(cached["cell"]):
            return cached[cached["cell"].isin(cell_names)].reset_index(drop=True)

    rows = []
    for name in cell_names:
        X, y, _, _ = dataset_points(name, instance_id)
        res = kernel_task_relevant_features(X, y, sub_complex_class=0, homology_dims=(1,))
        rows.append({"cell": name, **cell_of(name),
                     "kernel_persistence": res["max_kernel_persistence"],
                     "n_kernel_classes": res["n_task_relevant"]})
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


# ----------------------------------------------------------------------
# The head, and whether a witness survives the composition
# ----------------------------------------------------------------------

HEAD_CACHE = "aggregates/head_scores.csv"


def head_scores(pairs, spec=None, results_dir: str | Path = DEFAULT_RESULTS_DIR,
                output: str = "prob", use_cache: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """Score the network's OUTPUT against a feature, exactly as a neuron is.

    The output layer is a single linear unit, so the whole network is itself a
    scalar field on R^2 and the structural criterion applies to it unchanged.
    That makes the head one more position on the layer axis — and the position
    that matters, because the classification is read off exactly this function.

    ``output="prob"`` scores ``sigmoid(logit)``, the function the classifier
    actually is, which is the analogue of scoring hidden neurons at
    ``kind="post"`` (after their nonlinearity). Sigmoid is strictly monotone, so
    the level SETS are identical to the logit's; what changes is which levels
    the 101-point scan visits, and in probability units that scan is dense
    exactly where the decision boundary is and means the same thing in every
    model.

    ``pairs`` is an iterable of (dataset_type, model_path, feature_name).
    Cached, because each pair costs a couple of seconds.
    """
    import torch
    from analysis.criterions import structural_criterion
    from management.experiment_management import (
        load_dataset_instance, load_model_instance, get_feature_points,
    )

    if spec is None:
        from experiments.registry import EXPERIMENTS
        spec = EXPERIMENTS["exp4_density_sweep"]

    path = Path(results_dir) / HEAD_CACHE
    pairs = [tuple(p) for p in pairs]
    if use_cache and path.exists():
        cached = pd.read_csv(path)
        have = set(map(tuple, cached[["dataset_type", "model_path", "feature_name"]].values))
        if set(pairs) <= have:
            keep = cached.apply(
                lambda r: (r.dataset_type, r.model_path, r.feature_name) in set(pairs), axis=1)
            return cached[keep].reset_index(drop=True)

    kwargs = dict(bounds=spec.eval.bounds, resolution=spec.eval.resolution,
                  n_levels=spec.eval.structural_kwargs.get("n_levels", 101),
                  level_eps=spec.eval.structural_kwargs.get("level_eps", "adaptive"),
                  activation=None, threshold=TAU)

    feature_cache: dict = {}
    model_cache: dict = {}
    rows = []
    for i, (ds, model_path, feature_name) in enumerate(pairs, start=1):
        if ds not in feature_cache:
            stored = load_dataset_instance(
                f"data/dataset_instances/{ds}/{ds}_0/")
            feature_cache[ds] = stored
        A = get_feature_points(feature_cache[ds], feature_name)

        if model_path not in model_cache:
            model_cache[model_path] = load_model_instance(model_path)[0]
        model = model_cache[model_path]

        def field(x, _m=model):
            with torch.no_grad():
                z = _m(x).reshape(-1)
            return torch.sigmoid(z) if output == "prob" else z

        res = structural_criterion(field, feature_points=A, **kwargs)
        rows.append({"dataset_type": ds, "model_path": model_path,
                     "feature_name": feature_name, "family": family_of(feature_name),
                     "head_score": res["score_normalized_by_diameter"]})
        if verbose and i % 50 == 0:
            print(f"  [head] {i}/{len(pairs)}", flush=True)

    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True) \
                  .drop_duplicates(["dataset_type", "model_path", "feature_name"], keep="last")
    table.to_csv(path, index=False)
    return table


def load_head_scores(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Every head score computed so far, or an empty frame."""
    path = Path(results_dir) / HEAD_CACHE
    if not path.exists():
        return pd.DataFrame(columns=["dataset_type", "model_path", "feature_name",
                                     "family", "head_score"])
    return pd.read_csv(path)


def continuity_test(df: pd.DataFrame, cells: pd.DataFrame,
                    head: pd.DataFrame | None = None) -> pd.DataFrame:
    """Did a witness at layer l survive into every deeper layer, and the head?

    The network is the composition of its layers. If a feature is picked out at
    layer ``l`` and some later layer picks out nothing, the composition did not
    carry it forward — the head cannot be reading it there, and the witness at
    ``l`` was not something the network kept. This is the check experiment 2
    introduced, in its strict form: EVERY layer after ``l``, the head included,
    must contain at least one witness of the same feature in the same model.

    It is a NECESSARY condition and not a sufficient one. A feature can be
    linearly decodable from a layer without any single neuron's level set
    tracing it, so surviving the test does not make a witness "used"; failing
    it does show the witness was not propagated.

    Returns one row per witnessing neuron with ``survives`` and the layer at
    which the chain broke. Where the chain reached the head but no head score
    was computed for that pair, ``decided`` is False and ``survives`` is NaN —
    an undecided test, never a failed one. Head scores are expensive (a couple
    of seconds per pair), so they are computed for the pairs that matter and
    the rest stay undecided honestly.
    """
    thr = {(r.cell, r.family): r.threshold for r in cells.itertuples()}
    work = df.copy()
    work["threshold"] = [thr.get((c, f), np.nan)
                         for c, f in zip(work["dataset_type"], work["family"])]
    work["hit"] = work["score"] <= work["threshold"]

    by_layer = (work.groupby(["dataset_type", "model_path", "feature_name", "layer_id"])
                    ["hit"].any().unstack(fill_value=False))
    layers = sorted(int(c) for c in by_layer.columns)
    by_layer.columns = layers

    head_hit = {}
    if head is not None and len(head):
        for r in head.itertuples():
            key = (r.dataset_type, r.model_path, r.feature_name)
            t = thr.get((r.dataset_type, family_of(r.feature_name)), np.nan)
            head_hit[key] = bool(r.head_score <= t)

    # .loc внутри цикла по десяткам тысяч срабатываний стоит минуты; словарь — секунды.
    chains = {k: v for k, v in zip(by_layer.index, by_layer.to_numpy())}
    pos = {l: i for i, l in enumerate(layers)}

    rows = []
    for r in work[work["hit"]].itertuples():
        key = (r.dataset_type, r.model_path, r.feature_name)
        chain = chains[key]
        broke = None
        for l in layers:
            if l <= r.layer_id:
                continue
            if not chain[pos[l]]:
                broke = l
                break
        decided = True
        if broke is None:
            hh = head_hit.get(key)
            if hh is False:
                broke = "head"
            elif hh is None:
                # Голова для этой пары не считана: цепочка дошла до неё, но
                # последний шаг неизвестен. Это НЕ провал теста — это его
                # неопределённость, и такие строки не должны попадать в долю.
                broke = "head not scored"
                decided = False
        rows.append({"dataset_type": r.dataset_type, "model_path": r.model_path,
                     "feature_name": r.feature_name, "family": r.family,
                     "feature_role": r.feature_role, "task": r.task,
                     "layer_id": int(r.layer_id), "score": float(r.score),
                     "decided": decided,
                     "survives": (broke is None) if decided else np.nan,
                     "broke at": broke})
    return pd.DataFrame(rows)


def reference_density_probe(cell: str, model_path: str, sizes=(250, 500, 1000, 2000, 4000),
                            shape: str = "ring", spec=None, seed: int = 0) -> pd.DataFrame:
    """Does the reference's own point count change a neuron's score?

    The measurement targets are reference clouds of a FIXED size — 500 points on
    a circle, 1000 on the ring — the same in all forty configurations, and
    deliberately not following how densely the training data samples the object.
    The reason is that the score is meant to measure the distance from a level
    set to a GEOMETRIC OBJECT; the point count is only how finely that object is
    discretised for the Chamfer sum. Letting it follow the data would make the
    score move between cells for a reason that has nothing to do with the
    network — exactly the confound the experiment exists to remove.

    That argument is only worth anything if the score is in fact insensitive to
    the count, so this probe measures it: the same neurons, the same field, a
    reference swept over sixteenfold in size.

    Note what makes the question secondary in any case: a target and its nulls
    carry the SAME count by construction (verified in the notebook — the null is
    the target's own cloud, rigidly translated), so whatever residual
    sensitivity exists cancels in the target-against-floor contrast.
    """
    import math
    import torch
    from analysis.criterions import evaluate_neurons_on_feature
    from management.experiment_management import load_model_instance

    if spec is None:
        from experiments.registry import EXPERIMENTS
        spec = EXPERIMENTS["exp4_density_sweep"]

    w = cell_of(cell)["w"]
    model = load_model_instance(model_path)[0]

    def reference(n):
        g = torch.Generator().manual_seed(seed)
        theta = 2 * math.pi * torch.rand(n, generator=g)
        if shape == "ring":
            r_in, r_out = R_MID - w / 2, R_MID + w / 2
            u = torch.rand(n, generator=g)
            r = torch.sqrt(u * (r_out ** 2 - r_in ** 2) + r_in ** 2)
        else:
            r = torch.full((n,), R_MID)
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

    rows = []
    for n in sizes:
        res = evaluate_neurons_on_feature(
            model=model, feature_points=reference(n), bounds=spec.eval.bounds,
            resolution=spec.eval.resolution,
            structural_kwargs=dict(spec.eval.structural_kwargs),
            run_tubular=False, verbose=False)
        scores = np.array([r["structural_score_normalized_by_diameter"] for r in res])
        rows.append({"points in the reference": n, "median score": float(np.median(scores)),
                     "best neuron": float(scores.min())})
    return pd.DataFrame(rows)
