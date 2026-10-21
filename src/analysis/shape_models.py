"""Candidate laws for the SHAPE of a neuron's separating level curve.

Experiment 5 measures one number per cell -- how elongated the level curve is --
and the question this module exists for is whether that number is a function of
the cell's geometry alone. Each entry here is a different answer, and they are
written to be compared on the same footing: every model is handed the pair of
objects, is allowed exactly one number fitted to the curve's SIZE, and must then
predict its SHAPE.

The models, in the order they were tried:

``midline``
    the curve halfway between the two objects in every direction. The obvious
    guess, and wrong: it follows both objects, while the data follow one.

``uniform``
    the median shape of a curve drawn uniformly from all smooth closed curves
    that fit in the gap. Also wrong, and instructively so -- see
    ``sample_annulus_curves`` for why no measure of this kind can work.

``dilation``
    the interior's boundary pushed outward by a constant distance. Right shape
    of dependence, systematically too elongated.

``harmonic``
    the level set of the equilibrium potential between the two objects: the
    solution of Laplace's equation that is 0 on the interior and 1 on the
    feature. No shape parameters at all, and the closest of the four.

Everything is in the plane and assumes both objects are star-shaped about the
interior's centre, which every cell of experiment 5 satisfies.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _cell(name: str):
    from experiments.exp5_plan import derived_plan
    return {c["name"]: c for c in derived_plan()}[name]


def objects(name: str):
    """(interior dict, feature semi-axes, centre of the interior)."""
    rec = _cell(name)
    a_f = 0.5 * float(rec["diameter"])
    aspect = float(rec["params"].get("feature_aspect", 1.0))
    return rec["interior"], (a_f, a_f / aspect), np.asarray(
        rec["interior"].get("offset", (0.0, 0.0)), dtype=float)


def annulus_radial(name: str, n_theta: int = 720):
    """r_in(theta), r_out(theta) about the interior's centre, plus the angles."""
    from analysis.geometry_sweep import _radial_function, interior_boundary_points

    interior, (a_f, b_f), centre = objects(name)
    angles = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    t = np.linspace(0, 2 * math.pi, 2001)[:-1]
    feature = np.stack([a_f * np.cos(t), b_f * np.sin(t)], axis=1)
    r_out = _radial_function(feature, centre, angles)
    r_in = _radial_function(interior_boundary_points(interior, 1441), centre, angles)
    return angles, r_in, r_out, centre


def aspect_of_radial(r, angles, centre):
    """Aspect ratio of a curve given as r(theta) about ``centre``."""
    import torch
    from analysis.geometry_sweep import a2_to_aspect
    from analysis.shape_descriptors import fourier_descriptor

    pts = centre + np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    desc = fourier_descriptor(torch.tensor(pts, dtype=torch.float32))
    return float(a2_to_aspect(np.array([desc["amplitude"][2]]))[0])


# ----------------------------------------------------------------------
# the space of admissible curves
# ----------------------------------------------------------------------

def sample_annulus_curves(name: str, n: int = 400, max_k: int = 6, burn: int = 300,
                          thin: int = 15, seed: int = 0, n_theta: int = 720):
    """Curves drawn uniformly from those that fit inside the gap.

    A curve is admissible when ``r_in(theta) <= r(theta) <= r_out(theta)``
    everywhere; smoothness is imposed by keeping only the first ``max_k``
    harmonics, so the admissible set is a convex body in R^(2k+1) and
    hit-and-run samples it uniformly.

    Worth knowing before running it: the answer is fixed in advance by a
    symmetry. Write ``m = (r_in + r_out)/2`` for the midline. Any measure
    invariant under reflection through the midline has ``E[r] = m``, so the
    AVERAGE CURVE is the midline exactly. And the shape of the average is a
    lower bound on the average shape: ``a2 = |c2| / c0`` is jointly convex in
    the Fourier coefficients (it is the perspective of a norm), so Jensen gives
    ``E[a2] >= a2(m)``. **No such model can produce a curve rounder than the
    midline** -- the noise can only add apparent elongation. Since the measured
    curves are far rounder than the midline wherever the objects are elongated,
    the whole family is excluded, and the exclusion is a theorem rather than a
    fit.
    """
    angles, r_in, r_out, centre = annulus_radial(name, n_theta)
    basis = np.column_stack(
        [np.ones(n_theta)] + [f(k * angles) for k in range(1, max_k + 1)
                              for f in (np.cos, np.sin)])
    rng = np.random.default_rng(seed)
    lo, hi = r_in + 1e-9, r_out - 1e-9

    # A feasible starting point is found by maximising the slack, not guessed:
    # in the displaced cells of block D neither the midline's projection nor any
    # circle about the interior's centre fits, while an admissible curve still
    # exists. If the LP says the slack is negative, the set is genuinely empty
    # for this many harmonics and the cell is reported as such.
    from scipy.optimize import linprog

    n_coef = basis.shape[1]
    A_ub = np.vstack([np.column_stack([basis, np.ones(n_theta)]),
                      np.column_stack([-basis, np.ones(n_theta)])])
    b_ub = np.concatenate([hi, -lo])
    res = linprog(c=np.concatenate([np.zeros(n_coef), [-1.0]]), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * n_coef + [(None, None)], method="highs")
    if not res.success or res.x[-1] <= 0:
        return angles, np.zeros((0, n_theta)), centre
    x = res.x[:n_coef]
    r = basis @ x

    out = []
    for i in range(burn + n * thin):
        d = rng.standard_normal(basis.shape[1])
        d /= np.linalg.norm(d)
        w = basis @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = min(np.nanmin(np.where(w > 0, (hi - r) / w, np.inf)),
                       np.nanmin(np.where(w < 0, (lo - r) / w, np.inf)))
            t_lo = max(np.nanmax(np.where(w < 0, (hi - r) / w, -np.inf)),
                       np.nanmax(np.where(w > 0, (lo - r) / w, -np.inf)))
        if not (np.isfinite(t_hi) and np.isfinite(t_lo)) or t_hi <= t_lo:
            continue
        x = x + rng.uniform(t_lo, t_hi) * d
        r = basis @ x
        if i >= burn and (i - burn) % thin == 0:
            out.append(r.copy())
    return angles, np.asarray(out), centre


# ----------------------------------------------------------------------
# the equilibrium potential between the two objects
# ----------------------------------------------------------------------

def harmonic_potential(name: str, per_minor: int = 14, margin: float = 1.12):
    """Solve Laplace's equation in the gap: 0 on the interior, 1 on the feature.

    This is the capacitor formed by the two objects, and its level sets are the
    canonical interpolation between two shapes: near the inner conductor they
    copy the interior, near the outer one the feature, and in between they round
    off -- because the multipole terms of a conductor's field decay like
    ``r^-k``, so the quadrupole that carries elongation dies as ``1/r^2`` while
    the monopole that carries size dies as ``log r``. Rounding is thus not put in
    by hand; it is what potential theory does.

    The grid step is tied to the SMALLER minor axis of the two objects, because
    that is the feature the discretisation has to resolve.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    interior, (a_f, b_f), centre = objects(name)
    step = min(interior["b"], b_f) / per_minor
    xs = np.arange(-a_f * margin, a_f * margin + step, step)
    ys = np.arange(-b_f * margin, b_f * margin + step, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    in_feature = (X / a_f) ** 2 + (Y / b_f) ** 2 <= 1.0
    dx, dy = X - centre[0], Y - centre[1]
    if interior.get("rotation"):
        c_, s_ = math.cos(-interior["rotation"]), math.sin(-interior["rotation"])
        dx, dy = c_ * dx - s_ * dy, s_ * dx + c_ * dy
    in_interior = (dx / interior["a"]) ** 2 + (dy / interior["b"]) ** 2 <= 1.0

    free = in_feature & ~in_interior
    index = -np.ones(free.shape, dtype=np.int64)
    index[free] = np.arange(int(free.sum()))
    rows, cols, vals = [], [], []
    rhs = np.zeros(int(free.sum()))
    for k, (i, j) in enumerate(zip(*np.nonzero(free))):
        rows.append(k); cols.append(k); vals.append(-4.0)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            inside = (0 <= ii < free.shape[0] and 0 <= jj < free.shape[1])
            if inside and free[ii, jj]:
                rows.append(k); cols.append(index[ii, jj]); vals.append(1.0)
            elif not inside or not in_feature[ii, jj]:
                rhs[k] -= 1.0                          # Dirichlet 1 on the feature
    matrix = sp.coo_matrix((vals, (rows, cols)),
                           shape=(len(rhs), len(rhs))).tocsr()
    field = np.full(free.shape, np.nan)
    field[free] = spla.spsolve(matrix, rhs)
    field[~in_feature] = 1.0
    field[in_interior] = 0.0
    return xs, ys, field, step


def enclosing_level_curve(xs, ys, field, level: float, centre):
    """The closed contour of ``field == level`` that winds around ``centre``."""
    from skimage import measure

    best = None
    for seg in measure.find_contours(field, level):
        pts = np.stack([xs[0] + seg[:, 0] * (xs[1] - xs[0]),
                        ys[0] + seg[:, 1] * (ys[1] - ys[0])], axis=1)
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            continue
        rel = pts - centre
        turn = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        if abs(turn[-1] - turn[0]) < 6.0:
            continue
        if best is None or (np.linalg.norm(pts - centre, axis=1).mean()
                            < np.linalg.norm(best - centre, axis=1).mean()):
            best = pts
    return best


def level_at_size(xs, ys, field, centre, target_r0: float, lo: float = 1e-3,
                  hi: float = 1 - 1e-3, n_iter: int = 26):
    """The level whose curve has mean radius ``target_r0`` -- the one fitted number.

    Fitting on size and predicting shape is the whole protocol: ``r0`` is the
    mean of the radial profile and the aspect comes from the second harmonic of
    the NORMALISED profile, so knowing one says nothing about the other.
    """
    import torch
    from analysis.shape_descriptors import fourier_descriptor

    desc = None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        curve = enclosing_level_curve(xs, ys, field, mid, centre)
        if curve is None:
            hi = mid
            continue
        desc = fourier_descriptor(torch.tensor(curve, dtype=torch.float32))
        if desc["r0"] < target_r0:
            lo = mid
        else:
            hi = mid
    level = 0.5 * (lo + hi)
    curve = enclosing_level_curve(xs, ys, field, level, centre)
    if curve is None:
        return level, None, None
    desc = fourier_descriptor(torch.tensor(curve, dtype=torch.float32))
    return level, curve, desc


def harmonic_prediction(name: str, target_r0: float, per_minor: int = 14):
    """Aspect of the potential's level curve whose size matches the data."""
    from analysis.geometry_sweep import a2_to_aspect

    interior, _, centre = objects(name)
    xs, ys, field, step = harmonic_potential(name, per_minor)
    level, curve, desc = level_at_size(xs, ys, field, centre, target_r0)
    if desc is None:
        return {"aspect": float("nan"), "level": float("nan"),
                "r0": float("nan"), "step": step}
    return {"aspect": float(a2_to_aspect(np.array([desc["amplitude"][2]]))[0]),
            "level": float(level), "r0": float(desc["r0"]), "step": step}


def equidistant_curve(name: str, per_minor: int = 14, margin: float = 1.12):
    """The max-margin curve: equally far from both objects.

    The first thing a classifier person proposes, and worth testing for exactly
    that reason. It is NOT the radial midline: distance is Euclidean, so the
    curve bulges away from whichever boundary is locally closer.
    """
    from scipy.ndimage import distance_transform_edt

    interior, (a_f, b_f), centre = objects(name)
    step = min(interior["b"], b_f) / per_minor
    xs = np.arange(-a_f * margin, a_f * margin + step, step)
    ys = np.arange(-b_f * margin, b_f * margin + step, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    in_feature = (X / a_f) ** 2 + (Y / b_f) ** 2 <= 1.0
    dx, dy = X - centre[0], Y - centre[1]
    if interior.get("rotation"):
        c_, s_ = math.cos(-interior["rotation"]), math.sin(-interior["rotation"])
        dx, dy = c_ * dx - s_ * dy, s_ * dx + c_ * dy
    in_interior = (dx / interior["a"]) ** 2 + (dy / interior["b"]) ** 2 <= 1.0
    d_in = distance_transform_edt(~in_interior) * step
    d_out = distance_transform_edt(in_feature) * step
    return xs, ys, d_in - d_out, centre


def dilation_from_gap(name: str, k: float = 0.45):
    """The dilated interior with delta taken from the gap, not from the data.

    The honest version of the dilation model: ``delta = k * min gap``, one
    constant for the whole grid, so the prediction uses nothing but the two
    objects. Reported instead of the size-calibrated variant because that one
    reads the answer's scale off the measurement it is meant to predict.
    """
    import torch
    from analysis.geometry_sweep import a2_to_aspect, dilated_interior
    from analysis.shape_descriptors import fourier_descriptor

    angles, r_in, r_out, _ = annulus_radial(name)
    delta = k * float((r_out - r_in).min())
    curve = dilated_interior(_cell(name)["interior"], delta)
    desc = fourier_descriptor(torch.tensor(curve, dtype=torch.float32))
    return {"aspect": float(a2_to_aspect(np.array([desc["amplitude"][2]]))[0]),
            "delta": delta, "curve": curve}


def freedom_measures(name: str, n_theta: int = 720) -> dict:
    """Dimensionless ways to say "how much room the curve has".

    All are invariant under scaling the whole configuration, which is the
    minimum a shape law may assume. They exist to test the intuition that a
    small interior inside a large feature leaves the curve free to be round,
    while a tight fit forces it to copy the boundaries.
    """
    angles, r_in, r_out, _ = annulus_radial(name, n_theta)
    area_in = 0.5 * np.trapezoid(r_in ** 2, angles)
    area_out = 0.5 * np.trapezoid(r_out ** 2, angles)
    gap = r_out - r_in
    return {"size_ratio": float(r_out.mean() / r_in.mean()),
            "gap_min_rel": float(gap.min() / r_in.mean()),
            "gap_mean_rel": float(gap.mean() / r_in.mean()),
            "area_ratio": float((area_out - area_in) / area_in),
            "interior_share": float(area_in / area_out)}
