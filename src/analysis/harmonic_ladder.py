"""Per-rung shape spectrum of experiment 5's level curves.

The run's CSV answers, for every neuron and every rung, WHOSE shape the curve
carries: it stores ``r0`` and the two shape distances. It does not answer WHAT
shape the curve is -- the harmonics themselves are stored for the middle rung
only, because a full spectrum at every rung would have tripled the column set
of a 320 000-row table for a question nobody had yet asked.

Section 2 of the notebook asks it. So this module recomputes the ladder for
every neuron of every model and keeps the whole shape vector at every rung.
It is a recomputation, not a second measurement: the field window, the ladder
anchors, the contour extraction and the descriptor are the ones
``analysis.shape_descriptors.harmonic_criterion`` used during the run, so
``dfeat``/``dint``/``r0`` here must reproduce the CSV column for column. They
are recomputed anyway and the notebook checks them against it -- a silent
divergence between the figures and the tables is exactly the failure this
table would otherwise introduce.

One trap is worth naming, because the notebook fell into it before this module
existed. The ladder's inner anchor is the mean field value on the interior's
BOUNDARY (1200 points on the ellipse), not on the filled disk of class-1
points. The field varies across the disk, so the two anchors give different
levels and therefore different curves. Everything here goes through
``_interior_boundary``.

Build once (~10 min on 20 cores), read many times:

    python scripts/build_harmonic_ladder.py --jobs 18
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd

CACHE = Path("analysis/results/experiments/exp5_geometry/harmonic_ladder.csv.gz")

N_RUNGS = 5
N_PROBE = 13
MAX_K = 6
LAYERS = (1, 2, 3, 4)      # layer 0 is a theorem: its level sets are straight lines
LADDERS = ("value", "radial")


# ----------------------------------------------------------------------
# one model
# ----------------------------------------------------------------------

def _interior_boundary(interior: dict, n: int = 1201):
    """The interior's boundary as points -- the run's ladder anchor."""
    import torch

    angles = torch.linspace(0, 2 * math.pi, n)[:-1]
    boundary = torch.stack([interior["a"] * torch.cos(angles),
                            interior["b"] * torch.sin(angles)], dim=1)
    if interior.get("rotation"):
        c, s = math.cos(interior["rotation"]), math.sin(interior["rotation"])
        boundary = boundary @ torch.tensor([[c, -s], [s, c]]).T
    return boundary + torch.tensor(interior.get("offset", (0.0, 0.0)),
                                   dtype=torch.float32)


def _shape_row(desc, max_k: int = MAX_K) -> dict:
    """The descriptor's shape vector, phases included, as flat columns."""
    row = {"r0": desc["r0"], "coverage": desc["coverage"],
           "cx": float(desc["center"][0]), "cy": float(desc["center"][1])}
    for k in range(1, max_k + 1):
        a, phi = desc["amplitude"][k], desc["phase"][k]
        row[f"u{k}c"] = a * math.cos(k * phi)
        row[f"u{k}s"] = a * math.sin(k * phi)
    return row


def ladder_rows_for_model(cell: str, model_index: int, layers=LAYERS,
                          n_rungs: int = N_RUNGS, max_k: int = MAX_K,
                          ladders=LADDERS) -> list[dict]:
    """Every rung of every neuron of one stored model, with its spectrum."""
    import torch
    from analysis.fields import evaluate_field_on_grid
    from analysis.shape_descriptors import (fourier_descriptor, level_ladder,
                                            spatial_ladder, level_curves,
                                            enclosing_curve, densify,
                                            shape_distance)
    from analysis.stat_aggregation import find_model_paths_for_dataset
    from management.experiment_management import load_model_instance
    from models.NN_analysis import ScalarProjection, TruncatedMLPFunction
    from experiments.exp5_plan import derived_plan
    from analysis.geometry_sweep import dataset_points

    rec = {c["name"]: c for c in derived_plan()}[cell]
    paths = sorted(find_model_paths_for_dataset(cell, instance_id=0))
    if model_index >= len(paths):
        return []
    model, _ = load_model_instance(paths[model_index], eval_mode=True)

    _, _, feature_points, _ = dataset_points(cell)
    boundary = _interior_boundary(rec["interior"])
    centre = torch.tensor(rec["interior"]["offset"], dtype=torch.float32)
    desc_feature = fourier_descriptor(feature_points, max_k=max_k)
    desc_interior = fourier_descriptor(boundary, max_k=max_k)

    out = []
    for layer in layers:
        for coord in range(model.blocks[layer].linear.out_features):
            fn = ScalarProjection(TruncatedMLPFunction(model, layer, kind="post"),
                                  mode="coord", coord_id=coord)
            field = evaluate_field_on_grid(fn, rec["bounds"], rec["resolution"])
            step = 0.5 * float(field.grid_step)

            def describe(level):
                if level != level:
                    return None
                curve = enclosing_curve(level_curves(field, float(level)), centre)
                if curve is None:
                    return None
                return fourier_descriptor(densify(curve, step), max_k=max_k)

            value_levels = level_ladder(field, feature_points, boundary,
                                        n_steps=n_rungs)
            families = {"value": [float(v) for v in value_levels]}
            monotone = float("nan")
            if "radial" in ladders:
                spatial = spatial_ladder(
                    field, centre, desc_interior["r0"], desc_feature["r0"],
                    (float(value_levels[0]), float(value_levels[-1])),
                    n_steps=n_rungs, n_probe=N_PROBE, curve_step=step)
                monotone = float(bool(spatial["monotone"]))
                families["radial"] = [float(v) for v in spatial["levels"]]

            base = dict(dataset_type=cell, model_index=model_index,
                        layer_id=layer, coord_id=coord, monotone=monotone)
            for name in ladders:
                for j, level in enumerate(families[name]):
                    desc = describe(level)
                    if desc is None:
                        continue
                    out.append({**base, "ladder": name, "rung": j, "level": level,
                                **_shape_row(desc, max_k),
                                "dfeat": shape_distance(desc, desc_feature, max_k),
                                "dint": shape_distance(desc, desc_interior, max_k)})
    return out


# ----------------------------------------------------------------------
# the whole table
# ----------------------------------------------------------------------

def _task(args):
    cell, model_index, layers, ladders = args
    try:
        return ladder_rows_for_model(cell, model_index, layers=layers,
                                     ladders=ladders)
    except Exception as exc:                      # a missing model is not fatal
        return [{"dataset_type": cell, "model_index": model_index,
                 "ladder": "ERROR", "rung": -1, "level": float("nan"),
                 "coverage": str(exc)[:200]}]


def build(cells=None, n_models: int = 50, layers=LAYERS, ladders=LADDERS,
          jobs: int | None = None, out: Path | str = CACHE,
          progress: bool = True) -> pd.DataFrame:
    """Build the table over the given cells and write it to ``out``."""
    import multiprocessing as mp
    from experiments.exp5_plan import derived_plan

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if cells is None:
        cells = [c["name"] for c in derived_plan()]
    tasks = [(cell, i, tuple(layers), tuple(ladders))
             for cell in cells for i in range(n_models)]
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)

    rows, done = [], 0
    with mp.Pool(jobs) as pool:
        for part in pool.imap_unordered(_task, tasks, chunksize=1):
            rows.extend(part)
            done += 1
            if progress and done % 50 == 0:
                print(f"  {done}/{len(tasks)} моделей, строк {len(rows)}",
                      flush=True)

    table = pd.DataFrame(rows)
    bad = table[table.ladder == "ERROR"] if "ladder" in table else table.iloc[:0]
    assert bad.empty, f"{len(bad)} моделей упало, например: {bad.iloc[0].to_dict()}"
    table = table.sort_values(["dataset_type", "model_index", "layer_id",
                               "coord_id", "ladder", "rung"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.5g")
    return table


def load(path: Path | str = CACHE, ladder: str | None = "value") -> pd.DataFrame:
    """Read the table and add the columns every figure derives from it."""
    import numpy as np
    from analysis.geometry_sweep import a2_to_aspect, cell_coordinates

    table = pd.read_csv(path)
    if ladder is not None:
        table = table[table.ladder == ladder].copy()
    for k in range(1, MAX_K + 1):
        table[f"a{k}"] = np.hypot(table[f"u{k}c"], table[f"u{k}s"])
    # the major-axis direction: fourier_descriptor already stores phase[k] as
    # the angle of the k-th harmonic DIVIDED by k, so this is the axis itself
    # and must not be doubled again.
    table["angle2"] = np.degrees(
        np.arctan2(table.u2s, table.u2c) / 2.0 % math.pi)
    table["aspect"] = a2_to_aspect(table.a2.values)
    table["d_circle"] = np.sqrt(
        0.5 * sum(table[f"u{k}c"] ** 2 + table[f"u{k}s"] ** 2
                  for k in range(1, MAX_K + 1)))
    table["lam"] = table.dfeat / (table.dfeat + table.dint)
    coords = cell_coordinates()
    return table.merge(coords, on="dataset_type", how="left")
