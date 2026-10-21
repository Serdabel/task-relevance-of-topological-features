"""Describing the SHAPE of a level-set component (experiment 5).

Every criterion in the project answers one question: does this neuron's level
set match a target that was fixed in advance? That is a detector, and it takes
the target as given. Experiment 5 asks a different question — *what shape is the
level set*, and which of the two objects of the configuration (the feature curve
or the opposite-class interior) it resembles — and a detector cannot answer it.
This module supplies the descriptive measurement instead.

The description is polar. A closed curve around a centre is written as
``r(theta)`` and expanded in harmonics:

    r(theta) = r_0 * (1 + sum_k a_k cos(k*theta - phi_k))

``r_0`` says WHERE the curve sits, ``a_2`` how elliptic it is, ``a_4`` how
square, and so on. Both objects of the configuration are described the same way,
so "whose shape did the level set take" becomes a comparison of coefficients
rather than a verbal judgement.

TWO CAUTIONS, both of which change results rather than presentation.

*Which level.* The structural criterion picks the level that minimises the
distance to the FEATURE. Describing that level and then asking whether it looks
like the feature or like the interior is circular by construction. ``neutral_level``
therefore picks a level by a rule symmetric in the two objects — halfway between
the field's mean value on each — and ``describe_level_family`` reports the whole
sweep so the choice can be inspected rather than trusted.

*Which component.* A level set is a union of components, and only the one that
encircles the configuration is the separating curve. ``coverage`` (the fraction
of angular bins the component occupies) is reported with every descriptor; a
descriptor with low coverage describes an arc, and its harmonics are not
comparable with a closed curve's.
"""

from __future__ import annotations

import math

import torch

from analysis.criterions import level_set_component_points
from analysis.fields import FieldSample


def radial_profile(
    points: torch.Tensor,
    center: torch.Tensor | None = None,
    n_bins: int = 180,
) -> dict:
    """Bins a planar point set by angle and returns the mean radius per bin.

    Returns theta (bin centres), r (mean radius, NaN in empty bins), the centre
    used, and ``coverage`` = fraction of non-empty bins.
    """
    assert points.ndim == 2 and points.shape[1] == 2, "points must be [N, 2]"
    assert len(points) > 0, "points must be non-empty"

    center = points.mean(dim=0) if center is None else center.to(points.dtype)
    offset = points - center
    angles = torch.atan2(offset[:, 1], offset[:, 0]) % (2 * math.pi)
    radii = torch.linalg.norm(offset, dim=1)

    idx = torch.clamp((angles / (2 * math.pi) * n_bins).long(), 0, n_bins - 1)
    total = torch.zeros(n_bins, dtype=radii.dtype).index_add_(0, idx, radii)
    count = torch.zeros(n_bins, dtype=radii.dtype).index_add_(
        0, idx, torch.ones_like(radii))

    r = torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, float("nan")))
    theta = (torch.arange(n_bins, dtype=radii.dtype) + 0.5) * (2 * math.pi / n_bins)

    return {
        "theta": theta,
        "r": r,
        "center": center,
        "coverage": float((count > 0).float().mean()),
        "n_points": int(len(points)),
    }


def fourier_descriptor(
    points: torch.Tensor,
    center: torch.Tensor | None = None,
    max_k: int = 6,
    n_bins: int = 180,
) -> dict:
    """Harmonic description of a closed planar point set.

    ``r0`` is the mean radius; ``amplitude[k]`` is the RELATIVE amplitude of
    harmonic k (dimensionless, so shapes of different size are comparable) and
    ``phase[k]`` its orientation. ``aspect`` converts the second harmonic into
    the familiar major:minor ratio of an ellipse, ``(1+a2)/(1-a2)``.

    Empty angular bins are filled by circular interpolation before the
    transform; ``coverage`` says how much filling was needed.
    """
    profile = radial_profile(points, center=center, n_bins=n_bins)
    theta, r = profile["theta"], profile["r"]

    finite = torch.isfinite(r)
    assert int(finite.sum()) >= 8, "too few occupied angular bins to describe"
    if not bool(finite.all()):
        # Circular linear interpolation over the gaps.
        known_theta = theta[finite]
        known_r = r[finite]
        extended_theta = torch.cat([known_theta - 2 * math.pi, known_theta,
                                    known_theta + 2 * math.pi])
        extended_r = torch.cat([known_r, known_r, known_r])
        idx = torch.searchsorted(extended_theta, theta.contiguous()).clamp(1, len(extended_theta) - 1)
        t0, t1 = extended_theta[idx - 1], extended_theta[idx]
        r0_, r1_ = extended_r[idx - 1], extended_r[idx]
        r = r0_ + (theta - t0) / (t1 - t0).clamp(min=1e-12) * (r1_ - r0_)

    r0 = float(r.mean())
    amplitude, phase = {}, {}
    for k in range(1, max_k + 1):
        cos_k = float((r * torch.cos(k * theta)).mean()) * 2.0
        sin_k = float((r * torch.sin(k * theta)).mean()) * 2.0
        magnitude = math.hypot(cos_k, sin_k)
        amplitude[k] = magnitude / max(r0, 1e-12)
        phase[k] = math.atan2(sin_k, cos_k) / k

    a2 = amplitude.get(2, 0.0)
    aspect = (1.0 + a2) / (1.0 - a2) if a2 < 1.0 else float("inf")

    return {
        "center": profile["center"],
        "r0": r0,
        "amplitude": amplitude,
        "phase": phase,
        "aspect": aspect,
        "coverage": profile["coverage"],
        "n_points": profile["n_points"],
    }


SHAPE_ORDER = 6
"""Harmonics kept when two shapes are compared.

Where the shape ends and the discretisation noise begins was measured, not
guessed. On a perfectly circular field sampled by the same 200x200 grid every
coefficient comes out below 0.004 (the grid's own four-fold bias shows up as
a4 ~ 0.001-0.004 and nothing else), while real level sets of experiment 1 carry
a1 ~ 0.07, a2-a5 in the range 0.01-0.04, and only from k ~ 6 settle onto that
same 0.002-0.004 floor. So the boundary sits near k = 5, and the measurement
wobbles with which models are looked at.

Six, not five, because the cost is asymmetric. Keeping a coordinate that is
pure noise adds ~0.004 in quadrature to a distance of order 0.05-0.25 -- a
change in the fourth decimal. Dropping a coordinate that carries real shape
(a4 ~ 0.04 is routine) changes the distance outright. When in doubt, keep.

The objects are richer still -- a 4:1 ellipse has a6 = 0.147 -- so going higher
means raising the grid resolution first, not raising this number.
"""


def shape_vector(descriptor: dict, max_k: int = SHAPE_ORDER) -> torch.Tensor:
    """The curve's shape as a plain vector of numbers, size independent.

    Harmonic k contributes the pair ``a_k*(cos k*phi_k, sin k*phi_k)`` -- the
    cosine and sine coefficients of the profile normalised by ``r0``. Keeping
    both components (rather than the magnitude alone) keeps the ORIENTATION:
    an ellipse along x and the same ellipse across differ here, and in the
    anti-phase cells of experiment 5 that difference is the whole question.
    """
    assert max_k >= 1, "max_k must be at least 1"
    available = max(descriptor["amplitude"])
    assert max_k <= available, (
        f"descriptor carries {available} harmonics, {max_k} requested"
    )
    parts = []
    for k in range(1, max_k + 1):
        a, phi = descriptor["amplitude"][k], descriptor["phase"][k]
        parts += [a * math.cos(k * phi), a * math.sin(k * phi)]
    return torch.tensor(parts, dtype=torch.float32)


def shape_distance(first: dict, second: dict, max_k: int = SHAPE_ORDER) -> float:
    """How far apart two shapes are, ignoring size and position.

    The Euclidean distance between the coefficient vectors is not an arbitrary
    choice: by Parseval it IS the L2 distance between the normalised radial
    profiles ``rho(theta) = r(theta)/r0``, up to the 1/sqrt(2) below. So the
    number reads directly as the RMS relative radial disagreement between the
    two shapes -- 0.10 means they differ by 10 % of r0 in RMS.
    """
    delta = shape_vector(first, max_k) - shape_vector(second, max_k)
    return float(torch.linalg.norm(delta) / math.sqrt(2.0))


def compare_to_pair(level_set: dict, feature: dict, interior: dict,
                    max_k: int = SHAPE_ORDER) -> dict:
    """Describes a level set against the two objects of its cell.

    Three parts, kept apart on purpose because they answer different questions
    and can disagree:

        WHERE     the centroid, reported as the offset from each object;
        HOW BIG   ``r0``, the mean radius;
        WHAT SHAPE the coefficient vector, compared by ``shape_distance``.

    Each curve is expanded about ITS OWN centroid. That is what makes the
    description valid at all: the polar form needs the centre inside the curve,
    and in the displaced cells of experiment 5 a small pocket around the
    interior need not contain the feature's centre. Position is then carried by
    the offsets rather than smuggled into the first harmonic.

    Both objects must be given as CURVES. A filled region has to be passed as
    its boundary, not as its own points: binning a uniform disk by angle and
    averaging the radius returns 2R/3, not R, so a disk of radius 1 would be
    described as a circle of radius 0.67 and would carry spurious harmonics.
    Experiment 5's recipes keep the interior's exact boundary in the metadata
    for this reason.

    ``objects_shape`` is the distance between the two OBJECTS. It is the scale
    the other two shape numbers are read against: where it is near zero the two
    objects agree about shape, and shape cannot tell them apart -- not a defect
    but the question being empty there, with size and position still able to.
    """
    return {
        "offset_from_feature": level_set["center"] - feature["center"],
        "offset_from_interior": level_set["center"] - interior["center"],
        "r0": level_set["r0"],
        "r0_feature": feature["r0"],
        "r0_interior": interior["r0"],
        "shape_to_feature": shape_distance(level_set, feature, max_k),
        "shape_to_interior": shape_distance(level_set, interior, max_k),
        "objects_shape": shape_distance(feature, interior, max_k),
        "objects_r0_gap": abs(feature["r0"] - interior["r0"]),
        "max_k": max_k,
        "coverage": level_set["coverage"],
    }


def neutral_level(field: FieldSample, feature_points: torch.Tensor,
                  interior_points: torch.Tensor) -> float:
    """A level chosen symmetrically in the two objects.

    Halfway between the field's mean value on the feature and on the interior.
    Symmetric by construction, so it cannot favour either object — unlike the
    structural criterion's level, which is chosen to fit the feature.
    """
    on_feature = _nearest_values(field, feature_points).mean()
    on_interior = _nearest_values(field, interior_points).mean()
    return float(0.5 * (on_feature + on_interior))


def level_ladder(field: FieldSample, feature_points: torch.Tensor,
                 interior_points: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
    """Levels to describe, spread evenly between the two objects.

    The ends are the field's mean value ON the interior and ON the feature, so
    the ladder is anchored by the pair itself and is symmetric in the two -- it
    cannot favour either. ``neutral_level`` is its midpoint; the ladder is what
    you want whenever one number per neuron is not enough.

    Seven steps by default, and the number is bounded from both sides. Too few
    and there is no trend to read, and a rung lost to an unclosed curve costs
    too much. Too many and consecutive curves sit closer together than the
    instrument can tell apart: across the ladder the curve travels roughly the
    width of the gap, so the spacing is gap/(n-1), and it has to stay above
    both the grid step and the descriptor's radial resolution (~0.03*r0). At
    the base cell of experiment 5 -- gap 0.5, grid step 0.025, r0 ~ 1.2 --
    that caps n at about 13. Seven is odd, so the midpoint is a rung rather
    than falling between two, and it leaves five usable rungs when one is lost.
    """
    assert n_steps >= 3, "a ladder needs at least both ends and a middle"
    on_feature = float(_nearest_values(field, feature_points).mean())
    on_interior = float(_nearest_values(field, interior_points).mean())
    return torch.linspace(on_interior, on_feature, n_steps)


def spatial_ladder(field: FieldSample, center: torch.Tensor, r0_interior: float,
                   r0_feature: float, value_range: tuple[float, float],
                   n_steps: int = 5, n_probe: int = 13,
                   curve_step: float | None = None) -> dict:
    """The same family, spaced evenly in SPACE instead of in field value.

    A ladder uniform in the field's value is not uniform where the data live,
    and worse, it is not a COMMON COORDINATE: the radius its k-th rung lands on
    differs from neuron to neuron and from cell to cell, so "rung 3" means
    different geometry in different places -- and experiment 5 compares trends
    across cells. Walking a ray uniformly would fix the geometry but introduces
    the confound of which ray.

    No ray is needed. The family is already a one-parameter curve, and the map
    level -> r0 of the enclosing curve is monotone wherever the level set moves
    outward. Probing it and inverting it gives levels whose CURVES sit at
    prescribed radii, the same radii for every neuron.

    The targets are FIXED at ``linspace(r0_interior, r0_feature, n_steps)`` and
    are not clipped to what a given neuron happens to reach: a rung outside a
    neuron's achievable range comes back as ``nan`` in ``levels``. Clipping
    would keep every rung filled at the cost of the common frame, which is the
    whole point of the construction.

    Returns the levels, the target radii, ``monotone``, and ``reached`` -- a
    boolean mask of the rungs the neuron can actually produce.
    """
    low, high = value_range
    probe = torch.linspace(low, high, n_probe)
    levels_seen, radii_seen = [], []
    for c in probe:
        curve = enclosing_curve(level_curves(field, float(c)), center)
        if curve is None:
            continue
        step = curve_step if curve_step is not None else 0.5 * (field.grid_step or 0.01)
        levels_seen.append(float(c))
        # r0 about the curve's OWN centroid: that is its size, and it is what
        # `compare_to_pair` compares. Measuring it about a fixed point would
        # mix size with how far the curve has drifted from that point.
        radii_seen.append(fourier_descriptor(densify(curve, step))["r0"])

    targets = torch.linspace(r0_interior, r0_feature, n_steps)
    if len(levels_seen) < 3:
        return {"levels": torch.full((n_steps,), float("nan")), "r0": targets,
                "monotone": False, "reached": torch.zeros(n_steps, dtype=torch.bool),
                "n_usable": len(levels_seen), "r0_span": (float("nan"), float("nan"))}

    levels = torch.tensor(levels_seen, dtype=torch.float32)
    radii = torch.tensor(radii_seen, dtype=torch.float32)
    monotone = bool((radii[1:] - radii[:-1] >= -1e-6).all())
    lo, hi = float(radii.min()), float(radii.max())
    reached = (targets >= lo) & (targets <= hi)

    order = torch.argsort(radii)
    chosen = _interp_monotone(targets, radii[order], levels[order])
    chosen = torch.where(reached, chosen, torch.full_like(chosen, float("nan")))
    return {"levels": chosen, "r0": targets, "monotone": monotone,
            "reached": reached, "n_usable": len(levels_seen), "r0_span": (lo, hi)}


def _interp_monotone(targets: torch.Tensor, xs: torch.Tensor,
                     ys: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ys(xs) at `targets`; xs must be non-decreasing."""
    idx = torch.searchsorted(xs.contiguous(), targets.contiguous()).clamp(1, len(xs) - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    return y0 + (targets - x0) / (x1 - x0).clamp(min=1e-12) * (y1 - y0)


def _nearest_values(field: FieldSample, query: torch.Tensor) -> torch.Tensor:
    """Field values at the sample points nearest to `query`."""
    idx = torch.cdist(query.to(field.points.dtype), field.points).argmin(dim=1)
    return field.values[idx]


def level_curves(field: FieldSample, level: float, min_points: int = 8) -> list[dict]:
    """The level set as CURVES, not as a band.

    Thresholding ``|f - c| <= eps`` cannot give a thin ring: eps has to be at
    least half the largest neighbour jump ALONG the level set or the band tears,
    and where the field is flat that same eps inflates the band to half a unit
    wide. Consecutive levels then overlap and the "curve" is a blob. Marching
    squares has no such parameter -- it interpolates the crossing inside each
    cell and returns a polyline that is one cell thick by construction and
    accurate to a fraction of a cell.

    Each entry is ``{"points": [N, 2], "closed": bool}``. Closedness is exact
    (the polyline returns to its first vertex), so "is there a closed curve at
    this level" stops being a threshold on angular coverage and becomes a fact.
    """
    from skimage import measure

    assert field.grid_shape is not None, "curves need a grid field"
    values = field.values.reshape(field.grid_shape).numpy()
    grid = field.points.reshape(*field.grid_shape, 2)
    origin = grid[0, 0]
    step_x = float(grid[1, 0, 0] - grid[0, 0, 0])
    step_y = float(grid[0, 1, 1] - grid[0, 0, 1])

    out = []
    for segment in measure.find_contours(values, float(level)):
        if len(segment) < min_points:
            continue
        points = torch.stack([
            origin[0] + torch.tensor(segment[:, 0], dtype=torch.float32) * step_x,
            origin[1] + torch.tensor(segment[:, 1], dtype=torch.float32) * step_y,
        ], dim=1)
        closed = bool(torch.allclose(points[0], points[-1], atol=1e-5))
        out.append({"points": points, "closed": closed})
    return out


def enclosing_curve(curves: list[dict], center: torch.Tensor) -> torch.Tensor | None:
    """The closed curve that goes around `center` -- the separating curve.

    Membership is decided by the crossing-number test, not by angular coverage:
    a closed polyline either contains the point or it does not.
    """
    inside = [c for c in curves if c["closed"] and _contains(c["points"], center)]
    if not inside:
        return None
    # If several nest around the centre, the innermost is the one that separates
    # it from everything else.
    return min(inside, key=lambda c: float(torch.linalg.norm(c["points"] - center,
                                                             dim=1).mean()))["points"]


def _contains(polygon: torch.Tensor, point: torch.Tensor) -> bool:
    """Crossing-number test for a closed polyline."""
    import numpy as np

    x, y = float(point[0]), float(point[1])
    px, py = polygon[:, 0].numpy(), polygon[:, 1].numpy()
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    straddles = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = px + (y - py) / (qy - py) * (qx - px)
    return bool(np.sum(straddles & (x_cross > x)) % 2 == 1)


def densify(points: torch.Tensor, step: float) -> torch.Tensor:
    """Resamples a polyline so no gap exceeds `step`.

    A contour's vertices sit where it crosses cell edges, so in some directions
    they are sparse; binning such a polyline by angle would leave empty bins for
    a curve that geometrically covers every angle. Densifying first makes the
    radial profile of a closed curve complete by construction.
    """
    segments = points[1:] - points[:-1]
    lengths = torch.linalg.norm(segments, dim=1)
    pieces = []
    for start, seg, length in zip(points[:-1], segments, lengths):
        n = max(1, int(math.ceil(float(length) / step)))
        t = torch.arange(n, dtype=torch.float32).unsqueeze(1) / n
        pieces.append(start.unsqueeze(0) + t * seg.unsqueeze(0))
    pieces.append(points[-1:])
    return torch.cat(pieces, dim=0)


def level_components(
    field: FieldSample,
    level: float,
    eps: float | None = None,
    min_component_size: int = 5,
    connectivity: str = "full",
) -> list[torch.Tensor]:
    """Connected components of ``{|f - level| <= eps}``, as point sets."""
    if eps is None:
        assert field.grid_shape is not None, "eps required for a gridless field"
        # The band has to be thick enough that two neighbouring grid points
        # cannot BOTH fall outside it while the level set passes between them.
        # That happens exactly when |f(x_next) - f(x)| <= 2*eps, so half the
        # largest neighbour jump is the smallest eps that cannot tear the band.
        # A thinner band shatters the level set into arcs, and then there is no
        # closed curve left to describe -- the same reason the structural
        # criterion sizes its level_eps from local variation rather than from
        # the level step.
        eps = 0.5 * _max_neighbor_jump(field)
    mask = torch.abs(field.values - level) <= eps
    return level_set_component_points(
        field, mask, connectivity=connectivity,
        min_component_size=min_component_size, component_radius=None,
    )


def _max_neighbor_jump(field: FieldSample) -> float:
    """Largest |f| difference between neighbouring points of a grid field."""
    values = field.values.reshape(field.grid_shape)
    jumps = []
    for axis in range(values.ndim):
        lo = values.narrow(axis, 0, values.shape[axis] - 1)
        hi = values.narrow(axis, 1, values.shape[axis] - 1)
        jumps.append((hi - lo).abs().flatten())
    return float(torch.cat(jumps).max())


def enclosing_component(components: list[torch.Tensor], center: torch.Tensor,
                        n_bins: int = 180, min_coverage: float = 0.0
                        ) -> torch.Tensor | None:
    """The component that best surrounds `center` — the separating curve.

    Chosen by angular coverage around the centre, not by size: a large blob
    somewhere else is not the curve we are describing. ``min_coverage`` rejects
    everything below a floor, so a caller that needs a closed curve gets None
    rather than an arc it would then have to detect.
    """
    best, best_coverage = None, 0.0
    for component in components:
        if len(component) < 8:
            continue
        coverage = radial_profile(component, center=center, n_bins=n_bins)["coverage"]
        if coverage > best_coverage:
            best, best_coverage = component, coverage
    if best_coverage < min_coverage:
        return None
    return best


def describe_level_family(
    field: FieldSample,
    center: torch.Tensor,
    levels=None,
    n_levels: int = 41,
    eps: float | None = None,
    max_k: int = 6,
    min_coverage: float = 0.3,
) -> list[dict]:
    """Descriptor of the enclosing component at every level — the diagnostic.

    The neutral level is one point of this curve; seeing the whole family shows
    whether the description is stable in the level or an artefact of one choice.
    """
    if levels is None:
        low, high = float(field.values.min()), float(field.values.max())
        levels = torch.linspace(low, high, n_levels + 2)[1:-1].tolist()

    rows = []
    for level in levels:
        component = enclosing_component(
            level_components(field, level, eps=eps), center, min_coverage=min_coverage)
        row = {"level": float(level)}
        if component is None:
            # Below `min_coverage` the level has no closed curve around the
            # configuration; an arc's harmonics are not comparable with a
            # closed curve's, so the row is NaN rather than a smaller number.
            row.update({"coverage": 0.0, "r0": float("nan"), "aspect": float("nan"),
                        **{f"a{k}": float("nan") for k in range(1, max_k + 1)}})
        else:
            descriptor = fourier_descriptor(component, center=center, max_k=max_k)
            row.update({
                "coverage": descriptor["coverage"],
                "r0": descriptor["r0"],
                "aspect": descriptor["aspect"],
                **{f"a{k}": descriptor["amplitude"][k] for k in range(1, max_k + 1)},
            })
        rows.append(row)
    return rows


def ray_profile(
    function,
    center: torch.Tensor,
    radius: float,
    n_directions: int = 32,
    n_samples: int = 200,
    spacing: str = "space",
) -> dict:
    """Field values along rays from `center` outward — the 1-D view of a neuron.

    A level set is the pre-image of one value; a ray shows all of them at once,
    so it tells you how the field is arranged between the interior and the
    feature without committing to a level first.

    ``spacing`` is the question of where to spend the samples:

        "space"  equal steps in radius — every part of the gap gets the same
                 attention, and a plateau in the field looks as long as it is.
        "value"  equal steps in the field's own value — samples concentrate
                 where the field CHANGES, which is where the level sets are
                 dense and where the separating surface lives.

    Both are returned (the "value" resampling is derived from the "space" scan),
    because which is right depends on the question: the geometry of the gap is
    read in space, the arrangement of level sets is read in value.
    """
    directions = torch.linspace(0, 2 * math.pi, n_directions + 1)[:-1]
    steps = torch.linspace(0.0, radius, n_samples)

    points = (center.view(1, 1, 2)
              + steps.view(1, -1, 1)
              * torch.stack([torch.cos(directions), torch.sin(directions)], dim=1).view(-1, 1, 2))
    with torch.no_grad():
        values = function(points.reshape(-1, 2)).reshape(n_directions, n_samples)

    out = {"directions": directions, "radius_grid": steps, "values_by_space": values}

    if spacing in ("value", "both"):
        # Per ray, invert value -> radius at equal value steps.
        radii_by_value = torch.empty_like(values)
        for i in range(n_directions):
            v = values[i]
            monotone_v, order = torch.sort(v)
            targets = torch.linspace(float(v.min()), float(v.max()), n_samples)
            idx = torch.searchsorted(monotone_v, targets.contiguous()).clamp(1, n_samples - 1)
            v0, v1 = monotone_v[idx - 1], monotone_v[idx]
            r0_, r1_ = steps[order][idx - 1], steps[order][idx]
            w = torch.where(v1 > v0, (targets - v0) / (v1 - v0), torch.zeros_like(targets))
            radii_by_value[i] = r0_ + w * (r1_ - r0_)
        out["radii_by_value"] = radii_by_value

    return out


# --------------------------------------------------------------------------
# extra_criteria adapter
# --------------------------------------------------------------------------

def _harmonic_keys(n_steps: int, max_k: int) -> list[str]:
    keys = ["n_closed", "monotone", "objects_shape", "objects_r0_gap",
            "offset_x", "offset_y", "r0_mid", "coverage_mid"]
    keys += [f"a{k}_mid" for k in range(1, max_k + 1)]
    for ladder in ("value", "radial"):
        for j in range(n_steps):
            keys += [f"{ladder}{j}_r0", f"{ladder}{j}_dfeat", f"{ladder}{j}_dint"]
    return keys


def harmonic_criterion(*, field, feature_points, interior=None, target_center=None,
                       target_diameter=None, n_steps: int = 5, n_probe: int = 13,
                       max_k: int = SHAPE_ORDER, **kwargs) -> dict:
    """The harmonic descriptor as an extra criterion: flat numbers for the CSV.

    Bound per cell through ``EvalSpec.extra_criteria_kwargs`` with the
    interior's exact geometry, because the evaluation loop passes only the
    feature and the field — and this descriptor is about the PAIR.

    NULLS are skipped. The loop calls every criterion on every feature, but a
    null has no interior of its own to be described against; the descriptor's
    reference is the distribution over neurons, not the null placements. A
    feature whose centroid has moved away from the target's is therefore
    answered with NaNs, which costs nothing and keeps the column set constant.

    Both ladders are reported. Sampling by field value always works; the radial
    one fixes WHERE each rung sits and so removes the position from the
    comparison — measured to shrink the between-neuron spread of ``a2`` by 13 %
    — but needs the map level -> r0 to be monotone, which holds for 87 % of
    neurons. Neither dominates and both are cheap next to the criteria, so the
    choice is left to the analysis.
    """
    keys = _harmonic_keys(n_steps, max_k)
    blank = {key: float("nan") for key in keys}

    if interior is None or field.grid_shape is None:
        return blank

    centre = torch.tensor(interior.get("offset", (0.0, 0.0)), dtype=torch.float32)
    if target_center is not None and target_diameter:
        moved = float(torch.linalg.norm(feature_points.mean(dim=0)
                                        - torch.tensor(target_center,
                                                       dtype=torch.float32)))
        if moved > 0.25 * float(target_diameter):
            return blank                      # a null placement, not the target

    angles = torch.linspace(0, 2 * math.pi, 1201)[:-1]
    boundary = torch.stack([interior["a"] * torch.cos(angles),
                            interior["b"] * torch.sin(angles)], dim=1)
    if interior.get("rotation"):
        c, s = math.cos(interior["rotation"]), math.sin(interior["rotation"])
        boundary = boundary @ torch.tensor([[c, -s], [s, c]]).T
    boundary = boundary + centre

    step = 0.5 * float(field.grid_step)
    desc_feature = fourier_descriptor(feature_points, max_k=max_k)
    desc_interior = fourier_descriptor(boundary, max_k=max_k)

    def describe(level):
        curve = enclosing_curve(level_curves(field, float(level)), centre)
        if curve is None:
            return None
        return fourier_descriptor(densify(curve, step), max_k=max_k)

    row = dict(blank)
    row["objects_shape"] = shape_distance(desc_feature, desc_interior, max_k)
    row["objects_r0_gap"] = abs(desc_feature["r0"] - desc_interior["r0"])

    ladder = level_ladder(field, feature_points, boundary, n_steps=n_steps)
    value_range = (float(ladder[0]), float(ladder[-1]))
    described = [describe(level) for level in ladder]
    row["n_closed"] = sum(d is not None for d in described)

    spatial = spatial_ladder(field, centre, desc_interior["r0"], desc_feature["r0"],
                             value_range, n_steps=n_steps, n_probe=n_probe,
                             curve_step=step)
    row["monotone"] = bool(spatial["monotone"])
    radial = [describe(level) if level == level else None
              for level in spatial["levels"]]

    for name, family in (("value", described), ("radial", radial)):
        for j, desc in enumerate(family):
            if desc is None:
                continue
            row[f"{name}{j}_r0"] = desc["r0"]
            row[f"{name}{j}_dfeat"] = shape_distance(desc, desc_feature, max_k)
            row[f"{name}{j}_dint"] = shape_distance(desc, desc_interior, max_k)

    middle = described[n_steps // 2]
    if middle is not None:
        row["r0_mid"] = middle["r0"]
        row["coverage_mid"] = middle["coverage"]
        offset = middle["center"] - centre
        row["offset_x"], row["offset_y"] = float(offset[0]), float(offset[1])
        for k in range(1, max_k + 1):
            row[f"a{k}_mid"] = middle["amplitude"][k]
    return row
