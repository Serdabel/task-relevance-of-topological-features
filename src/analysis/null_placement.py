"""
Manual, explicit placement of A.1 null control features.

WHY THIS EXISTS
---------------
A null feature is a NEGATIVE CONTROL: the same shape as the target, put
somewhere else, so that ``rate[target] - rate[null]`` isolates task-relevance
from "the criterion fires on any curve of this size". For that subtraction to
mean what it says, the null must differ from the target ONLY in placement.

``analysis.controls.make_null_features`` draws placements at random under a
separation contract. That is the right default in high dimensions, where
placing anything by hand is hopeless, but in 2D/3D it produced controls that
all landed in the same uninformative region: on ``halfplane_base`` every null
ended up off the data (0.13-0.56 away from the nearest training point) while
the target IS the class-1 point set, and the network's field over the two is
qualitatively different (mean |grad f| 0.068 on the target vs 0.40-0.47 on the
nulls). The gap then partly measures "on data vs off data" rather than
relevance. See ``AUDIT/08_independent_review.md`` section 5.1.

This module takes the opposite approach: you place the nulls yourself, so that
position becomes a SWEPT VARIABLE instead of a nuisance. A representative set
covers the problem deliberately — on a data cluster, far from all data, in the
gap between clusters, intersecting the target, rotated in place — and then the
witness rate can be read AS A FUNCTION of where the feature sits.

WHY RIGID MOTIONS ONLY
----------------------
Every placement is an element of E(n): a rotation about the shape's own
centroid followed by a translation. Rigid motions are isometries, so the
matched-diameter and matched-point-count requirements that the control
contract cares about hold BY CONSTRUCTION rather than by check, and the null is
guaranteed congruent to the target - same shape, same size, same topology.

SPECIFYING A PLACEMENT
----------------------
Two constructors, both taking coordinates as plain positional numbers:

    at(x, y)            put the shape's centroid AT this point (absolute)
    shift(dx, dy)       DISPLACE the shape by this vector (relative)

with an optional ``rot=`` and an optional ``name=``:

    at(-2.5, 0.0, name="on_blue_disk")
    shift(0.0, 1.2, rot=30, name="lifted_and_turned")
    at(1.0, 0.0, 2.0, rot=(0, 45, 0), name="3d_example")

``rot`` is in DEGREES and its accepted forms depend on the dimension:

    dim 2  scalar                       angle in the plane
    dim 3  scalar                       angle about +z
           (rx, ry, rz)                 intrinsic X-Y-Z Euler angles
           ((ax, ay, az), angle)        axis-angle
    any    (d, d) array/tensor          the rotation matrix itself, used as-is

For a rotationally symmetric shape (a circle in 2D, a sphere in 3D) rotation is
a no-op; it matters for ellipses, wavy circles and linked components.

TYPICAL NOTEBOOK USE
--------------------
    from analysis.null_placement import at, shift, build_null_features, preview

    placements = [
        at(-2.5,  0.0, name="on_blue_disk"),
        at( 0.0,  0.0, name="empty_corridor"),
        at( 2.5,  0.0, name="on_target"),      # deliberately intersecting
        at(-6.0,  0.0, name="far_outside"),
    ]
    nulls = build_null_features(target_points, placements)
    preview(X, y, target_points, nulls)        # picture + diagnostics table

Nulls built this way are NOT persisted into the dataset instance - they are an
evaluation-time control, constructed per run. What makes a run reproducible is
that the placement list is recorded next to the results; see
``placements_to_records`` / ``placements_from_records``, which the eval-config
sidecar uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _dc_field

import torch

from analysis.geometry import (
    feature_diameter,
    max_chamfer_distance,
    nearest_distances_to_set,
)

__all__ = [
    "Placement",
    "at",
    "shift",
    "rotation_matrix",
    "apply_placement",
    "build_null_features",
    "placement_report",
    "bounds_covering",
    "random_placements",
    "null_features_provider",
    "placements_to_records",
    "placements_from_records",
    "as_placement",
    "preview",
]


# --------------------------------------------------------------------------
# the placement object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """One rigid motion applied to a reference shape.

    ``vector`` is read as an absolute centroid position when ``absolute`` is
    True and as a displacement otherwise. ``rotation`` is applied about the
    shape's own centroid FIRST, then the translation - so rotating never moves
    the shape, which is what makes the two knobs independent.
    """

    vector: tuple[float, ...]
    rotation: object | None = None
    name: str | None = None
    absolute: bool = False
    meta: dict = _dc_field(default_factory=dict, compare=False)

    @property
    def dim(self) -> int:
        return len(self.vector)


def at(*coords, rot=None, name: str | None = None, **meta) -> Placement:
    """Place the shape's centroid AT ``coords`` (absolute position)."""
    return Placement(
        vector=tuple(float(c) for c in coords),
        rotation=rot,
        name=name,
        absolute=True,
        meta=dict(meta),
    )


def shift(*delta, rot=None, name: str | None = None, **meta) -> Placement:
    """DISPLACE the shape by ``delta`` from where it currently is."""
    return Placement(
        vector=tuple(float(c) for c in delta),
        rotation=rot,
        name=name,
        absolute=False,
        meta=dict(meta),
    )


def as_placement(pl) -> Placement:
    """Coerce whatever a caller has to a ``Placement``.

    A spec that has been round-tripped through JSON hands back plain dicts,
    and a hand-written plan may hold ``(dx, dy, rot, name)`` tuples. Both are
    unambiguous, and refusing them cost an experiment a server run: the
    AttributeError fired in the shard subprocess, after the models had
    trained. Accepting them here means no path into the null machinery can
    fail on the container its placements arrived in.
    """
    if isinstance(pl, Placement):
        return pl
    if isinstance(pl, dict):
        return Placement(
            vector=tuple(float(v) for v in pl["vector"]),
            rotation=pl.get("rotation"),
            name=pl.get("name"),
            absolute=bool(pl.get("absolute", False)),
            meta=dict(pl.get("meta", {})),
        )
    if isinstance(pl, (list, tuple)) and len(pl) == 4:
        dx, dy, rot, name = pl
        return Placement(vector=(float(dx), float(dy)), rotation=rot or None,
                         name=name, absolute=True)
    raise TypeError(
        f"cannot read {pl!r} as a Placement; build one with at()/shift()"
    )


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------

def _rot2(angle_deg: float) -> torch.Tensor:
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def _rot3_axis_angle(axis, angle_deg: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float32).reshape(3)
    norm = float(axis.norm())
    assert norm > 1e-12, "rotation axis must be non-zero"
    axis = axis / norm

    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    x, y, z = (float(v) for v in axis)

    # Rodrigues
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float32
    )
    return torch.eye(3) + s * K + (1.0 - c) * (K @ K)


def rotation_matrix(dim: int, rotation) -> torch.Tensor:
    """Resolve a human-friendly ``rot=`` spec into a (dim, dim) matrix.

    Accepted forms are documented in the module docstring. Angles are DEGREES.
    A ``None`` rotation returns the identity.
    """
    if rotation is None:
        return torch.eye(dim, dtype=torch.float32)

    # An explicit matrix always wins, in any dimension.
    if torch.is_tensor(rotation) or (
        hasattr(rotation, "__len__")
        and len(rotation) == dim
        and all(hasattr(r, "__len__") and len(r) == dim for r in rotation)
    ):
        R = torch.as_tensor(rotation, dtype=torch.float32).reshape(dim, dim)
        _assert_rotation(R)
        return R

    if dim == 2:
        assert not hasattr(rotation, "__len__"), (
            "in 2D `rot` is a single angle in degrees, got "
            f"{rotation!r}"
        )
        return _rot2(rotation)

    if dim == 3:
        if not hasattr(rotation, "__len__"):
            return _rot3_axis_angle((0.0, 0.0, 1.0), rotation)

        # axis-angle: ((ax, ay, az), angle)
        if len(rotation) == 2 and hasattr(rotation[0], "__len__"):
            return _rot3_axis_angle(rotation[0], rotation[1])

        # intrinsic X-Y-Z Euler angles
        assert len(rotation) == 3, (
            "in 3D `rot` is a scalar (about +z), a 3-tuple of Euler angles, "
            "an (axis, angle) pair, or a 3x3 matrix; got "
            f"{rotation!r}"
        )
        rx, ry, rz = rotation
        return (
            _rot3_axis_angle((0.0, 0.0, 1.0), rz)
            @ _rot3_axis_angle((0.0, 1.0, 0.0), ry)
            @ _rot3_axis_angle((1.0, 0.0, 0.0), rx)
        )

    raise AssertionError(
        f"in dim {dim} pass `rot` as an explicit ({dim}, {dim}) matrix "
        "(or None); angle shortcuts are only defined for 2D and 3D"
    )


def _assert_rotation(R: torch.Tensor, tol: float = 1e-4):
    d = R.shape[0]
    err = float((R @ R.T - torch.eye(d, dtype=R.dtype)).abs().max())
    assert err <= tol, (
        f"rotation matrix is not orthonormal (max |R R^T - I| = {err:.2e}). "
        "A non-rigid transform would change the null's shape or diameter, "
        "which breaks the matched-size requirement of the A.1 control."
    )
    det = float(torch.det(R))
    assert det > 0, (
        f"rotation matrix has determinant {det:.3f} < 0 — that is a reflection, "
        "not a rotation. Pass a proper rotation, or the null is a mirror image "
        "of the target rather than a congruent copy."
    )


# --------------------------------------------------------------------------
# applying placements
# --------------------------------------------------------------------------

def _chamfer(X: torch.Tensor, Y: torch.Tensor) -> float:
    """max-symmetrised Chamfer as a scalar (the criterion returns a dict)."""
    return float(max_chamfer_distance(X, Y)["score"])


def _diam(P: torch.Tensor) -> float:
    """Feature diameter, matching what the CRITERIA normalise by.

    ``method="exact"`` (max pairwise distance), not the cheaper ``"bbox"``
    default of ``feature_diameter`` — the structural criterion
    (``criterions.py:524``), the extremum-anchored criterion
    (``extremum_anchored.py:241``) and ``controls.make_null_features``
    (``controls.py:117``) all use "exact". Using anything else here would make
    the ``separation`` column silently incomparable with the witness threshold
    it is meant to be read against (for a radius-1.5 circle the two differ by
    sqrt(2): 3.00 vs 4.24).
    """
    return float(feature_diameter(P, method="exact").item())


def _as_points(points) -> torch.Tensor:
    P = torch.as_tensor(points).detach().cpu()
    if not torch.is_floating_point(P):
        P = P.float()
    assert P.ndim == 2, f"points must be [n, dim], got shape {tuple(P.shape)}"
    return P


def apply_placement(points, placement: Placement) -> torch.Tensor:
    """Rotate ``points`` about their centroid, then translate. Returns a copy."""
    P = _as_points(points)
    dim = P.shape[1]

    assert placement.dim == dim, (
        f"placement {placement.name or ''!r} is {placement.dim}-dimensional but "
        f"the feature lives in dim {dim}"
    )

    centroid = P.mean(dim=0)
    R = rotation_matrix(dim, placement.rotation).to(P.dtype)
    rotated = (P - centroid) @ R.T + centroid

    v = torch.tensor(placement.vector, dtype=P.dtype)
    if placement.absolute:
        return rotated + (v - centroid)
    return rotated + v


def build_null_features(
    target_points,
    placements,
    prefix: str = "null",
) -> dict:
    """{name: points} for each placement, all congruent to ``target_points``.

    Unnamed placements get ``{prefix}_{i}``; named ones get
    ``{prefix}_{name}`` so that the ``feature_role`` inference downstream still
    recognises them as controls by their prefix.
    """
    P = _as_points(target_points)
    out = {}

    for i, pl in enumerate(placements):
        pl = as_placement(pl)
        key = f"{prefix}_{pl.name}" if pl.name else f"{prefix}_{i}"
        assert key not in out, f"duplicate null feature name {key!r}"
        out[key] = apply_placement(P, pl)

    return out


# --------------------------------------------------------------------------
# diagnostics — the point of placing by hand is to SEE where things landed
# --------------------------------------------------------------------------

def placement_report(
    target_points,
    nulls: dict,
    X=None,
    y=None,
    bounds=None,
):
    """A table describing where each null sits. Returns a pandas DataFrame.

    Columns, and why each is here:

    ``separation``      ``max_chamfer(target, null) / diam(target)``. This is
                        the criterion's own distance, so it is directly
                        comparable to the witness threshold (0.1). Below ~0.1
                        the null is close enough to inherit the target's
                        verdict; that is legitimate if you placed it there ON
                        PURPOSE as an intersecting case, and a mistake
                        otherwise.
    ``min_pair_sep``    the same quantity against the closest OTHER null. Two
                        nulls sitting on top of each other are one control, not
                        two, and the effective sample size is smaller than it
                        looks.
    ``dist_to_data``    mean distance from the null's points to the nearest
                        training point, in data units. The target's value is
                        the reference: on ``halfplane_base`` the target IS the
                        data, so its value is ~0 and any null with a large
                        value is being scored in a different regime.
    ``nearest_class``   which class the null mostly sits over (needs ``y``).
    ``in_bounds``       fraction of the null inside ``bounds``. Anything below
                        1.0 is partly outside the evaluated domain and will be
                        scored against a truncated level set - use
                        ``bounds_covering`` to fix it.
    """
    import pandas as pd

    A = _as_points(target_points)
    diam = _diam(A)
    assert diam > 0, "target feature has zero diameter"

    Xt = _as_points(X) if X is not None else None
    yt = torch.as_tensor(y).flatten() if y is not None else None

    names = list(nulls.keys())
    rows = []

    for name in ["__target__", *names]:
        P = A if name == "__target__" else _as_points(nulls[name])

        row = {
            "feature": "TARGET" if name == "__target__" else name,
            "centre": tuple(round(float(v), 3) for v in P.mean(dim=0)),
            "diam": round(_diam(P), 4),
        }

        row["separation"] = (
            0.0 if name == "__target__"
            else round(_chamfer(A, P) / diam, 4)
        )

        others = [n for n in names if n != name and name != "__target__"]
        row["min_pair_sep"] = (
            round(
                min(
                    _chamfer(_as_points(nulls[o]), P) / diam
                    for o in others
                ),
                4,
            )
            if others else float("nan")
        )

        if Xt is not None:
            d = nearest_distances_to_set(P, Xt)
            row["dist_to_data"] = round(float(d.mean()), 4)

            if yt is not None:
                # class of the nearest training point, per null point
                nn = torch.cdist(P, Xt).argmin(dim=1)
                lab = yt[nn]
                vals, counts = torch.unique(lab, return_counts=True)
                row["nearest_class"] = int(vals[counts.argmax()])

        if bounds is not None:
            inside = torch.ones(len(P), dtype=torch.bool)
            for k, (lo, hi) in enumerate(bounds):
                inside &= (P[:, k] >= lo) & (P[:, k] <= hi)
            row["in_bounds"] = round(float(inside.float().mean()), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def bounds_covering(bounds, point_sets, margin: float = 0.5):
    """Expand ``bounds`` minimally so every set in ``point_sets`` fits, + margin.

    Deliberately MINIMAL rather than, say, doubling every side. The evaluation
    box is not cosmetic: it sets where the field is sampled, the range the
    level sweep covers, and the statistics the adaptive level_eps is chosen
    from - for the TARGET as much as for the nulls. Enlarging it therefore
    changes the target's own numbers and makes runs with different boxes
    incomparable (``AUDIT/05_post_repair_validation.md`` section E.3, option 4).
    Growing only as far as the placements actually require keeps that cost at
    its minimum, and returns ``bounds`` unchanged when everything already fits.
    """
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]

    for P in point_sets:
        P = _as_points(P)
        for k in range(P.shape[1]):
            lo[k] = min(lo[k], float(P[:, k].min()) - margin)
            hi[k] = max(hi[k], float(P[:, k].max()) + margin)

    return [(lo[k], hi[k]) for k in range(len(lo))]


# --------------------------------------------------------------------------
# random placements — the d >= 4 default, expressed in the same language
# --------------------------------------------------------------------------

def random_placements(
    target_points,
    bounds,
    n: int = 5,
    seed: int = 0,
    min_separation_from_target: float = 0.3,
    min_separation_between_nulls: float = 0.1,
    max_attempts: int = 20000,
    rotate: bool = True,
) -> list:
    """Rejection-sampled placements, returned as explicit ``Placement`` objects.

    Same contract as ``analysis.controls.make_null_features`` (matched shape by
    construction, separated from the target and from each other), but the
    OUTPUT is a placement list rather than raw point sets. That matters: the
    random draw becomes inspectable, printable, reproducible and editable — you
    can take what the sampler produced, look at it with ``placement_report``,
    and hand-adjust the two placements you dislike instead of re-rolling the
    seed. Use this in d >= 4, where placing by hand is not feasible.
    """
    A = _as_points(target_points)
    dim = A.shape[1]
    diam = _diam(A)
    centroid = A.mean(dim=0)

    gen = torch.Generator().manual_seed(int(seed))

    # the centroid may travel anywhere that keeps the shape inside `bounds`
    radius = float((A - centroid).norm(dim=1).max())
    lo = torch.tensor([b[0] + radius for b in bounds], dtype=torch.float32)
    hi = torch.tensor([b[1] - radius for b in bounds], dtype=torch.float32)
    assert bool((hi > lo).all()), (
        "the feature is too large for these bounds — no placement keeps it "
        "inside. Widen `bounds` (see bounds_covering) or shrink the feature."
    )

    accepted: list[Placement] = []
    accepted_pts: list[torch.Tensor] = []

    for _ in range(max_attempts):
        if len(accepted) == n:
            break

        u = torch.rand(dim, generator=gen, dtype=torch.float32)
        centre = lo + u * (hi - lo)

        rot = None
        if rotate and dim >= 2:
            R = _random_rotation(dim, gen)
            rot = R

        cand = at(*centre.tolist(), rot=rot)
        pts = apply_placement(A, cand)

        if _chamfer(A, pts) / diam <= min_separation_from_target:
            continue
        if any(
            _chamfer(q, pts) / diam <= min_separation_between_nulls
            for q in accepted_pts
        ):
            continue

        accepted.append(
            Placement(
                vector=cand.vector,
                rotation=cand.rotation,
                name=f"rand{len(accepted)}",
                absolute=True,
            )
        )
        accepted_pts.append(pts)

    assert len(accepted) == n, (
        f"only {len(accepted)} of {n} placements satisfied the separation "
        f"contract in {max_attempts} attempts. This is the geometry refusing, "
        "not a bug: a diameter-matched null is infeasible when the feature is "
        "large relative to its domain. Lower `n`, lower the separation floors, "
        "widen `bounds`, or place the nulls by hand with `at`/`shift`."
    )
    return accepted


def _random_rotation(dim: int, gen) -> torch.Tensor:
    """Haar-ish random rotation via QR of a Gaussian matrix, det forced to +1."""
    M = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if float(torch.det(Q)) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# --------------------------------------------------------------------------
# recording — what keeps an unpersisted control reproducible
# --------------------------------------------------------------------------

def placements_to_records(placements) -> list:
    """JSON-able description of a placement list, for the eval-config sidecar.

    Nulls built here are not stored in the dataset instance, so this record is
    the ONLY thing that lets a result be reconstructed later. Every run that
    uses manual placements must write it next to its CSV.
    """
    out = []
    for pl in placements:
        pl = as_placement(pl)
        rot = pl.rotation
        if torch.is_tensor(rot):
            rot = rot.tolist()
        elif rot is not None and hasattr(rot, "__len__"):
            rot = [
                r.tolist() if torch.is_tensor(r) else
                (list(r) if hasattr(r, "__len__") else r)
                for r in rot
            ]
        out.append({
            "name": pl.name,
            "vector": list(pl.vector),
            "absolute": bool(pl.absolute),
            "rotation": rot,
            **({"meta": pl.meta} if pl.meta else {}),
        })
    return out


def placements_from_records(records) -> list:
    """Inverse of ``placements_to_records``."""
    return [
        Placement(
            vector=tuple(float(v) for v in r["vector"]),
            rotation=r.get("rotation"),
            name=r.get("name"),
            absolute=bool(r.get("absolute", False)),
            meta=dict(r.get("meta", {})),
        )
        for r in records
    ]


# --------------------------------------------------------------------------
# provider hook — how the evaluation stage picks these up
# --------------------------------------------------------------------------

def null_features_provider(placements, prefix: str = "null"):
    """Build the ``extra_features_fn`` that the evaluation stage calls.

    Returns ``fn(loaded_dataset, dataset_type) -> {name: (points, role)}``,
    transforming whichever feature is marked ``feature_role == "target"``.
    Passed to ``analysis.stat_aggregation.collect_compact_neuron_statistics``.
    """
    from management.experiment_management import get_feature_points
    from analysis.stat_aggregation import feature_role_from_metadata

    def fn(loaded_dataset, dataset_type=None):
        names = list(loaded_dataset.get("features", {}).keys())
        targets = [
            n for n in names
            if feature_role_from_metadata(loaded_dataset, n) == "target"
        ]
        assert len(targets) == 1, (
            f"need exactly one feature with role 'target' to build nulls from, "
            f"found {targets} among {names}"
        )
        A = get_feature_points(loaded_dataset, targets[0])
        return {
            name: (pts, "null")
            for name, pts in build_null_features(A, placements, prefix=prefix).items()
        }

    # Carried on the callable so the evaluation stage can write it into the
    # eval-config sidecar without knowing anything about placements. These
    # nulls exist nowhere on disk; this record is what makes the run
    # reconstructible.
    fn.placement_records = placements_to_records(placements)
    return fn


# --------------------------------------------------------------------------
# preview (lazy viz import, as elsewhere in the project)
# --------------------------------------------------------------------------

def preview(X, y, target_points, nulls, bounds=None, **kwargs):
    """Plot the data with the target and the placed nulls, and print the report.

    2D uses matplotlib, 3D uses plotly; d >= 4 skips the picture and returns
    the table only. Returns the ``placement_report`` DataFrame.
    """
    from visualization.null_plots import plot_null_placements

    report = placement_report(target_points, nulls, X=X, y=y, bounds=bounds)
    plot_null_placements(X, y, target_points, nulls, bounds=bounds, **kwargs)
    return report
