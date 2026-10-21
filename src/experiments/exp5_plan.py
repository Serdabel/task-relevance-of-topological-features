"""Plan of experiment 5: the cells, their nulls, and the window each is scored in.

Kept out of ``registry.py`` because it is derived, not declared: 56 cells, 200
null placements and 56 evaluation windows are computed from four rules, and a
list that long written out by hand would be a list nobody can check. The rules
are in one place here; the registry only asks for the result.

The design these rules come from is ``notebooks/experiments/exp5_geometry_plan.ipynb``,
which draws every cell and every null.
"""

import functools
import hashlib
import json
import math
import pathlib

import torch

from data.dataset_recipes import make_geometry_pair_dataset, _feature_semiaxes

__all__ = [
    "ASPECTS", "ELL", "TOTAL_POINTS", "GAP_MIN", "SEPARATION", "N_A",
    "cell_name", "cell_plan", "unique_cells",
    "n_nulls", "null_placements", "eval_window",
]

ASPECTS = (1.0, 1.5, 2.5, 4.0)
ELL = 4.0                 # "the elongated one" wherever a single value is meant
TOTAL_POINTS = 1000       # class sizes always add up to this
GAP_MIN = 1.0             # a null never comes closer than this to the feature
SEPARATION = 0.32         # and never closer than this in chamfer/diam either
N_A = 54.5                # grid steps across the feature's diameter (experiment 1)
INTERIOR_SCALE_D = 0.5    # block D shrinks the interior so it has room to move
CLEARANCE_FLOOR = 0.25


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------

def _key(params: dict):
    """Hashable form of a params dict, used as a cache key."""
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in params.items()))


@functools.lru_cache(maxsize=None)
def _dataset(key):
    """Built datasets, cached.

    Deriving the plan asks for the same cell many times over — the offset
    bisection of block D alone rebuilds one cell fifty times — and building it
    is the expensive part, not the arithmetic around it. The registry
    constructs the whole spec at import, and every worker subprocess imports
    the registry, so this cache is what keeps that import cheap.
    """
    return make_geometry_pair_dataset(**dict(key))


def cell_name(params: dict) -> str:
    """Canonical name: equal parameters give an equal name, so the duplicates
    a plan contains between blocks collapse on their own instead of being
    spotted by eye."""
    fa = params.get("feature_aspect", 1.0)
    ia = params.get("interior_aspect", 1.0)
    rot = "T" if params.get("interior_rotation", 0.0) else ""
    scale = params.get("interior_scale", 1.0)
    ox, oy = params.get("interior_offset", (0.0, 0.0))
    n_int = params.get("n_interior", 500)
    return (f"g5_f{int(fa * 10):03d}_i{int(ia * 10):03d}{rot}_s{int(scale * 100):03d}"
            f"_o{int(round(ox * 100)):+05d}{int(round(oy * 100)):+05d}_b{n_int:04d}")


def _admissible(value, params, axis):
    off = (value, 0.0) if axis == "x" else (0.0, value)
    meta = _dataset(_key(dict(interior_scale=INTERIOR_SCALE_D, interior_offset=off,
                              **params))).features["ring"]["metadata"]
    return meta["interior_inside"] and meta["min_clearance"] >= CLEARANCE_FLOOR


def _max_offset(params, axis, hi=12.0):
    lo = 0.0
    for _ in range(24):        # 24 шагов дают точность 1e-6 на диапазоне 12
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _admissible(mid, params, axis) else (lo, mid)
    return lo


def cell_plan() -> list[dict]:
    """Every position of the plan, in order, with the block it belongs to.

    Positions repeat between blocks — the middle balance of C is a cell of A,
    the zero offset of D is shared by two directions — and each carries
    ``first`` saying whether this position is where the cell is introduced.
    """
    plan = []

    for fa in ASPECTS:                                   # A: shape against shape
        for ia in ASPECTS:
            plan.append(("A", f"фича {fa}:1, внутр. {ia}:1",
                         dict(feature_aspect=fa, interior_aspect=ia)))

    for fa in ASPECTS:                                   # A-perp: interior across
        for ia in ASPECTS:
            if fa == 1.0 or ia == 1.0:
                continue      # turning a round object produces no new configuration
            plan.append(("A⊥", f"фича {fa}:1, внутр. {ia}:1 поперёк",
                         dict(feature_aspect=fa, interior_aspect=ia,
                              interior_rotation=math.pi / 2)))

    geometries = [("вытянута фича", dict(feature_aspect=ELL)),
                  ("вытянута внутр.", dict(interior_aspect=ELL)),
                  ("оба, в фазе", dict(feature_aspect=ELL, interior_aspect=ELL)),
                  ("оба, противофаза", dict(feature_aspect=ELL, interior_aspect=ELL,
                                            interior_rotation=math.pi / 2))]
    for label, geometry in geometries:                   # C: geometry x balance
        for balance in (0.2, 0.5, 0.8):
            n_int = int(round(TOTAL_POINTS * balance))
            plan.append(("C", f"{label}, баланс {balance:.1f}",
                         dict(n_feature=TOTAL_POINTS - n_int, n_interior=n_int,
                              **geometry)))

    shapes = {"круг в круге": dict(),
              "круг в эллипсе": dict(feature_aspect=ELL),
              "эллипс в круге": dict(interior_aspect=ELL),
              "эллипс в эллипсе": dict(feature_aspect=ELL, interior_aspect=ELL),
              "эллипс ⊥ эллипса": dict(feature_aspect=ELL, interior_aspect=ELL,
                                       interior_rotation=math.pi / 2)}
    rows = [("круг в круге", "x"), ("круг в эллипсе", "x"), ("круг в эллипсе", "y"),
            ("эллипс в круге", "x"), ("эллипс в круге", "y"),
            ("эллипс в эллипсе", "x"), ("эллипс в эллипсе", "y"),
            ("эллипс ⊥ эллипса", "x"), ("эллипс ⊥ эллипса", "y")]
    for shape, axis in rows:                             # D: position x shape
        geometry = shapes[shape]
        top = _max_offset(geometry, axis)
        for fraction in (0.0, 0.5, 1.0):
            value = fraction * top
            offset = (value, 0.0) if axis == "x" else (0.0, value)
            plan.append(("D", f"{shape}, {axis}, {fraction:.0%}",
                         dict(interior_scale=INTERIOR_SCALE_D, interior_offset=offset,
                              **geometry)))

    seen = set()
    out = []
    for block, label, params in plan:
        name = cell_name(params)
        out.append(dict(block=block, label=label, name=name, params=params,
                        first=name not in seen))
        seen.add(name)
    return out


def unique_cells() -> list[dict]:
    """The cells that are actually built and trained — one per canonical name."""
    return [row for row in cell_plan() if row["first"]]


# --------------------------------------------------------------------------
# nulls
# --------------------------------------------------------------------------

def n_nulls(params: dict) -> int:
    """How many placements the cell's symmetry does not collapse.

    A round feature with a concentric round interior is invariant under all of
    O(2): every direction is the same and a circle has no orientation, so one
    placement is the whole control. Elongate the interior and the two axes stop
    being equivalent (two placements); displace it and left stops being right
    (three); elongate the FEATURE and the copy acquires an orientation of its
    own, which doubles the count again (four).
    """
    fa = params.get("feature_aspect", 1.0)
    ia = params.get("interior_aspect", 1.0)
    shifted = any(abs(v) > 1e-9 for v in params.get("interior_offset", (0.0, 0.0)))
    if fa > 1.0:
        return 4
    if shifted:
        return 3
    if ia > 1.0:
        return 2
    return 1


_DIRECTIONS = (((1.0, 0.0), 0, "справа"), ((0.0, 1.0), 0, "сверху"),
               ((-1.0, 0.0), 90, "слева_поперёк"), ((0.0, -1.0), 90, "снизу_поперёк"))


def null_placements(params: dict) -> tuple:
    """Where this cell's nulls go, as ``(dx, dy, rotation_deg, name)``.

    A null is the cell's own target, turned and moved, so congruence is not
    something to arrange -- it is what a null IS here.

    The distance is not a constant. The separation contract is stated in units
    of the feature's DIAMETER, and a 4:1 ellipse is four times the diameter of
    the base circle, so a fixed gap of one unit satisfies it for the circle and
    fails it (0.17 against 0.3) for the ellipse. Each copy is therefore pushed
    out by bisection to the smallest distance that meets the contract, so it
    sits as close as it is allowed to and the evaluation window stays as small
    as it can.
    """
    return _null_placements_cached(_key(params))


@functools.lru_cache(maxsize=None)
def _null_placements_cached(key) -> tuple:
    """Cached because the registry builds the whole spec at import time and every
    worker subprocess imports it: 200 bisections cost seven seconds, which over
    the ~350 shards of a run would be forty minutes of nothing.

    The bisection runs on a SUBSAMPLE of the curve — the placement only has to
    be found, and a few hundred points locate it to far better than the grid
    step — but the separation it reports is measured on the full point set.
    """
    from analysis.geometry import max_chamfer_distance, feature_diameter

    params = dict(key)
    target = _dataset(key).get_feature_points("ring")
    diameter = float(feature_diameter(target, method="exact"))
    probe = target[::7]                      # ~285 points, enough to place a copy
    out = []
    for direction, rotation, label in _DIRECTIONS[:n_nulls(params)]:
        turned = (torch.stack([-target[:, 1], target[:, 0]], dim=1)
                  if rotation else target)
        turned_probe = turned[::7]
        axis = torch.tensor(direction, dtype=torch.float32)
        lo = (float((target * axis).sum(1).max()) + GAP_MIN
              + float((turned * axis).sum(1).max()))
        hi = lo + 4 * diameter
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            separation = float(
                max_chamfer_distance(probe, turned_probe + mid * axis)["score"]) / diameter
            lo, hi = (lo, mid) if separation >= SEPARATION else (mid, hi)
        out.append((round(hi * direction[0], 4), round(hi * direction[1], 4),
                    rotation, label))
    return tuple(out)


# --------------------------------------------------------------------------
# evaluation window
# --------------------------------------------------------------------------

def eval_window(params: dict) -> dict:
    return _eval_window_cached(_key(params))


@functools.lru_cache(maxsize=None)
def _eval_window_cached(key) -> dict:
    """The box the field is evaluated on, and the resolution that keeps the
    instrument fixed.

    The box is set by the NULLS, not by the data: a congruent copy of a 4:1
    ellipse needs twelve more units of room. It is made square so the grid step
    is the same along both axes, which every criterion assumes.

    The resolution is chosen so that the feature's diameter always spans the
    same number of grid steps (``N_A``, read off experiment 1: an 11-unit box
    at resolution 200 puts 54.5 steps across a diameter of 3). Without that a
    long feature would be measured by a coarser instrument than a round one and
    the sweep would partly measure the grid.
    """
    from analysis.geometry import feature_diameter

    params = dict(key)
    data = _dataset(key)
    target = data.get_feature_points("ring")
    points = [target] + [data.get_component_data(name) for name in data.object_names]
    for dx, dy, rotation, _ in null_placements(params):
        turned = (torch.stack([-target[:, 1], target[:, 0]], dim=1)
                  if rotation else target)
        points.append(turned + torch.tensor([dx, dy], dtype=torch.float32))
    cloud = torch.cat(points, dim=0)

    lo = cloud.min(dim=0).values
    hi = cloud.max(dim=0).values
    centre = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max()) * 1.06        # 6 % margin, as elsewhere
    bounds = [(float(centre[i] - half), float(centre[i] + half)) for i in range(2)]

    diameter = float(feature_diameter(target, method="exact"))
    step = diameter / N_A
    resolution = int(round(2 * half / step))
    return {"bounds": bounds, "resolution": resolution, "step": step,
            "diameter": diameter, "side": 2 * half}


# --------------------------------------------------------------------------
# the spec, assembled
# --------------------------------------------------------------------------

_CACHE = pathlib.Path(__file__).with_name("exp5_plan_cache.json")


def _fingerprint() -> str:
    """Identity of the rules, so a stale cache cannot survive a change of them."""
    source = pathlib.Path(__file__).read_bytes()
    constants = repr((ASPECTS, ELL, TOTAL_POINTS, GAP_MIN, SEPARATION, N_A,
                      INTERIOR_SCALE_D, CLEARANCE_FLOOR)).encode()
    return hashlib.sha1(source + constants).hexdigest()[:16]


def derived_plan(use_cache: bool = True) -> list[dict]:
    """Cells with their nulls, windows and interiors — the whole derivation.

    Cached to JSON next to this module. Not an optimisation for its own sake:
    the registry builds this spec at IMPORT, and every training and evaluation
    subprocess imports the registry, so ten seconds of derivation would become
    an hour of nothing across a run. The cache is keyed by a hash of this file
    and its constants, so editing a rule invalidates it automatically.
    """
    if use_cache and _CACHE.exists():
        blob = json.loads(_CACHE.read_text())
        if blob.get("fingerprint") == _fingerprint():
            return blob["cells"]

    out = []
    for cell in unique_cells():
        params = cell["params"]
        window = eval_window(params)
        a, b = _feature_semiaxes(params.get("interior_aspect", 1.0), "major",
                                 params.get("interior_scale", 1.0))
        out.append(dict(
            name=cell["name"], block=cell["block"], label=cell["label"],
            params={k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in params.items()},
            placements=[list(p) for p in null_placements(params)],
            bounds=window["bounds"], resolution=window["resolution"],
            diameter=window["diameter"],
            interior=dict(a=a, b=b,
                          rotation=float(params.get("interior_rotation", 0.0)),
                          offset=[float(v) for v in
                                  params.get("interior_offset", (0.0, 0.0))]),
        ))
    _CACHE.write_text(json.dumps({"fingerprint": _fingerprint(), "cells": out},
                                 ensure_ascii=False, indent=1))
    return out


def dataset_specs():
    """One DatasetSpec per cell, each with its own window, nulls and interior.

    Everything a cell needs that differs from its neighbours travels in
    ``eval_overrides``: the box and resolution its geometry demands, the null
    placements its symmetry allows, and the interior's exact shape, which the
    harmonic descriptor cannot get any other way (the evaluation loop hands a
    criterion the feature and the field, never the second object).

    ``n_null_features=0`` on purpose — nulls are not baked into the stored
    instance but placed from the spec. Random placement was measured to be
    infeasible here: the target is nearly as large as its own box, and a
    congruent copy has nowhere to land that satisfies the separation contract.
    """
    from experiments.configs import DatasetSpec
    from analysis.null_placement import at

    specs = []
    for cell in derived_plan():
        params = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in cell["params"].items()}
        interior = dict(cell["interior"])
        interior["offset"] = tuple(interior["offset"])
        placements = tuple(at(dx, dy, rot=rot or None, name=name)
                           for dx, dy, rot, name in cell["placements"])
        specs.append(DatasetSpec(
            name=cell["name"],
            recipe="make_geometry_pair_dataset",
            params=params,
            target_feature="ring",
            n_null_features=0,
            eval_overrides=dict(
                bounds=[tuple(b) for b in cell["bounds"]],
                resolution=cell["resolution"],
                null_placements=placements,
                extra_criteria_kwargs={
                    "homological": {"backend": "delaunay", "max_sample_points": 7000},
                    "harmonic": {"interior": interior,
                                 "target_center": (0.0, 0.0),
                                 "target_diameter": cell["diameter"]},
                },
            ),
        ))
    return specs
