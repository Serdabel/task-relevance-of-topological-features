"""Criterion 3: extremum-anchored structural criterion.

Fixes the main methodological flaw of the structural criterion (PROJECT.md
§Criterion 3): the free ``min`` over all levels c lets incidental witnesses
arise when some level set happens to pass through the feature during the
sweep. Here candidate levels are restricted to critical values
c* = f(p*) from ``analysis.extremum_sampling``.

PLAN.md caveats built into the implementation:
  - the scored object is the WHOLE connected component at level c*, not a
    spatial neighborhood of p* — cutting to a ball around the anchor can chop
    an arc out of a ridge that runs along the entire feature and produce a
    false negative;
  - this criterion is a FILTER, not a decision: several legitimate extrema at
    different scales, or tanh plateaus, can leave a continuum of candidate
    levels where coincidence is still possible. The decisive anti-incidental
    tool remains the null baseline (analysis.controls).

Integration (no changes to the evaluation loop):

    evaluate_neurons_on_feature(
        ...,
        extra_criteria={"extremum": extremum_anchored_criterion},
    )
    collect_compact_neuron_statistics(..., extra_criteria=...)
    # -> columns extremum_score, extremum_score_normalized_by_diameter,
    #    extremum_is_witness, extremum_n_extrema, ... (auto-detected)
"""

import torch

from analysis.fields import FieldSample, evaluate_field_on_grid
from analysis.geometry import feature_diameter, nearest_distances_to_set
from analysis.extremum_sampling import find_local_extrema, critical_values
from analysis.criterions import (
    level_set_component_points,
    best_component_chamfer_score,
    choose_local_adaptive_level_eps,
)


def _resolve_level_eps(
    field: FieldSample,
    feature_points: torch.Tensor,
    candidate_levels: torch.Tensor,
    level_eps,
    distance_batch_size=None,
) -> float:
    """Adaptive level_eps needs a grid field; explicit floats always work."""
    if not isinstance(level_eps, str):
        assert level_eps > 0, "level_eps must be positive"
        return float(level_eps)

    assert level_eps == "adaptive", "level_eps must be positive float or 'adaptive'"
    assert field.is_grid, (
        "level_eps='adaptive' needs a grid field; pass an explicit level_eps "
        "for gridless samples"
    )

    info = choose_local_adaptive_level_eps(
        Z=field.values_nd,
        grid_points_nd=field.points_nd,
        feature_points=feature_points,
        grid_step=max(float(field.grid_step), 1e-12),
        # The sparse-level guard (min_level_step_factor) is meant for the
        # STRUCTURAL criterion, which sweeps n_levels=101 uniformly spaced
        # levels: there, flooring level_eps at half the median level step
        # stops consecutive bands leaving gaps. Criterion 3's levels are not
        # a sweep — they are the handful of critical values from extremum
        # sampling, and their spacing carries no information about how wide
        # a band should be. Feeding them to the guard is a category error,
        # and on a monotone field it is catastrophic: two anchors that span
        # the whole range make the "median step" the range itself, so the
        # floor becomes 0.5*range(f) and the "level set" scored against A is
        # half the domain. Measured on halfplane_base model2 (all 35
        # neurons, resolution 200): level_eps/range was 0.5000 for 35/35 and
        # the scored band covered 47-58% of the grid, so the reported score
        # was the distance from half the box to the circle, not from a level
        # set to the circle. See AUDIT/08_independent_review.md section 4.10.
        levels=None,
        min_level_step_factor=None,
        distance_batch_size=distance_batch_size,
    )
    return float(info["level_eps"])


def extremum_anchored_criterion(
    function,
    feature_points: torch.Tensor,
    bounds,
    resolution: int | list[int] | tuple[int, ...] = 100,
    device: str = "cpu",
    field: FieldSample | None = None,

    # extremum sampling
    n_restarts: int = 64,
    ascent_steps: int = 300,
    ascent_lr: float = 5e-2,
    grad_tol: float = 1e-5,
    anchor_grad_norm_max: float | None = None,
    critical_value_merge_tol: float = 1e-3,
    seed: int | None = 0,

    # level scoring (same knobs as structural_criterion)
    level_eps: float | str = "adaptive",
    level_window: float = 0.0,
    connectivity: str = "full",
    min_component_size: int = 5,
    threshold: float = 0.1,
    distance_batch_size: int | None = None,
    **kwargs,
) -> dict:
    """
    Structural (Chamfer) score restricted to extremum-anchored levels.

    1. finds local maxima AND minima of ``function`` in ``bounds``,
       optionally dropping anchors whose gradient norm exceeds
       ``anchor_grad_norm_max`` (boundary-clamped anchors);
    2. candidate levels = merged critical values c* (plus, if
       level_window > 0, levels c* +- level_window);
    3. reuses the shared ``field`` and the structural-criterion machinery
       (``level_set_component_points`` + ``best_component_chamfer_score``)
       on ONLY those levels;
    4. normalizes by diam(A); witness iff normalized score <= threshold.

    Returns dict (int/float/bool values become CSV columns automatically):
        score, score_normalized_by_diameter, is_witness,
        n_extrema, n_candidate_levels, best_level, best_anchor_grad_norm
    """
    feature_points = feature_points.to(device)
    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    assert feature_points.ndim == 2 and len(feature_points) > 0
    assert feature_points.shape[1] == len(bounds)
    assert threshold > 0

    if field is None:
        field = evaluate_field_on_grid(
            function,
            bounds=bounds,
            resolution=resolution,
            device=device,
            dtype=feature_points.dtype,
        )

    extrema_kwargs = dict(
        bounds=bounds,
        n_restarts=n_restarts,
        n_steps=ascent_steps,
        lr=ascent_lr,
        grad_tol=grad_tol,
        record_trajectories=False,
        device=device,
        seed=seed,
    )

    maxima = find_local_extrema(function, mode="max", **extrema_kwargs)
    minima = find_local_extrema(function, mode="min", **extrema_kwargs)

    anchor_values = torch.cat([maxima.values, minima.values])
    anchor_grad_norms = torch.cat([maxima.grad_norms, minima.grad_norms])
    anchor_points = torch.cat([maxima.points, minima.points])

    # Is each anchor a genuine interior stationary point, or just where the
    # ascent ran into the wall of the evaluation box?
    #
    # This distinction is the whole validity question for criterion 3. A tanh
    # network that solves a linearly separable task learns near-monotone ridge
    # functions with NO interior critical point, so ascent and descent
    # terminate in opposite corners and the two "critical values" are simply
    # min f and max f over the box. The criterion then reports a verdict
    # computed from levels that are artifacts of the domain, not of the
    # function. Reported rather than enforced (the researcher's choice): a
    # boundary-anchored verdict is not wrong, it is INAPPLICABLE, and the
    # honest reading of it is "no test was administered here".
    #
    # `best_anchor_grad_norm` does not capture this — measured, a gradient
    # threshold of 0.1 rejects 1.4 % of boundary anchors while this test
    # identifies 100 % of them.
    if len(anchor_points) > 0 and bounds is not None:
        lo = torch.tensor([float(b[0]) for b in bounds],
                          device=anchor_points.device, dtype=anchor_points.dtype)
        hi = torch.tensor([float(b[1]) for b in bounds],
                          device=anchor_points.device, dtype=anchor_points.dtype)
        tol = 1e-3 * (hi - lo).clamp(min=1e-12)
        on_boundary = (
            (anchor_points <= lo + tol) | (anchor_points >= hi - tol)
        ).any(dim=1)
    else:
        on_boundary = torch.zeros(len(anchor_points), dtype=torch.bool)

    if anchor_grad_norm_max is not None and len(anchor_values) > 0:
        keep = anchor_grad_norms <= anchor_grad_norm_max
        anchor_values = anchor_values[keep]
        anchor_grad_norms = anchor_grad_norms[keep]
        anchor_points = anchor_points[keep]
        on_boundary = on_boundary[keep]

    n_extrema = int(len(anchor_values))
    frac_boundary = (
        float(on_boundary.float().mean()) if len(on_boundary) else float("nan")
    )

    empty_result = {
        "score": float("inf"),
        "score_normalized_by_diameter": float("inf"),
        "is_witness": False,
        "n_extrema": n_extrema,
        "n_candidate_levels": 0,
        "best_level": float("nan"),
        "best_anchor_grad_norm": float("nan"),
        "level_eps": float("nan"),
        "frac_anchors_on_boundary": frac_boundary,
        "best_anchor_on_boundary": float("nan"),
    }

    if n_extrema == 0:
        return empty_result

    from analysis.extremum_sampling import ExtremaResult

    merged = critical_values(
        ExtremaResult(
            points=torch.empty(0, len(bounds)),
            values=anchor_values,
            grad_norms=anchor_grad_norms,
            mode="max",
        ),
        merge_tol=critical_value_merge_tol,
    )

    levels = [float(v.item()) for v in merged]

    if level_window > 0:
        widened = []
        for c in levels:
            widened.extend([c - level_window, c, c + level_window])
        levels = sorted(set(widened))

    candidate_levels = torch.tensor(
        levels, device=field.values.device, dtype=field.values.dtype
    )

    level_eps_value = _resolve_level_eps(
        field,
        feature_points,
        candidate_levels,
        level_eps,
        distance_batch_size=distance_batch_size,
    )

    values_flat = field.values
    best_score = float("inf")
    best_level = float("nan")

    for level in candidate_levels:
        level_mask_flat = torch.abs(values_flat - level) <= level_eps_value

        component_point_sets = level_set_component_points(
            field,
            level_mask_flat,
            connectivity=connectivity,
            min_component_size=min_component_size,
        )

        level_best_score, _ = best_component_chamfer_score(
            component_point_sets,
            feature_points,
            distance_batch_size=distance_batch_size,
        )

        if level_best_score < best_score:
            best_score = level_best_score
            best_level = float(level.item())

    diameter = max(float(feature_diameter(feature_points, method="exact").item()), 1e-12)
    normalized = best_score / diameter

    # Anchor closest in value to the winning level (diagnostics).
    if best_level == best_level and len(anchor_values) > 0:  # not NaN
        nearest_anchor = int(torch.argmin((anchor_values - best_level).abs()).item())
        best_anchor_grad_norm = float(anchor_grad_norms[nearest_anchor].item())
        best_anchor_on_boundary = float(on_boundary[nearest_anchor].item())
    else:
        best_anchor_grad_norm = float("nan")
        best_anchor_on_boundary = float("nan")

    return {
        "score": best_score,
        "score_normalized_by_diameter": normalized,
        "is_witness": normalized <= threshold,
        "n_extrema": n_extrema,
        "n_candidate_levels": int(len(candidate_levels)),
        "best_level": best_level,
        "best_anchor_grad_norm": best_anchor_grad_norm,
        # 1.0 => the level that produced this score came from a box-clamp
        # extreme, so the verdict is about the domain, not the function.
        "frac_anchors_on_boundary": frac_boundary,
        "best_anchor_on_boundary": best_anchor_on_boundary,
        # Reported so that a pathological band width cannot hide again: this
        # is the quantity that silently became 0.5*range(f) (see
        # _resolve_level_eps). Surfaces as the extremum_level_eps CSV column
        # and is worth reading against the field's range.
        "level_eps": level_eps_value,
    }


def classify_witness_basin_or_ridge(
    function,
    feature_points: torch.Tensor,
    bounds,
    device: str = "cpu",
    field: FieldSample | None = None,
    resolution: int | list[int] | tuple[int, ...] = 100,
    n_radial_bins: int = 40,
    ridge_location_tol: float = 0.3,
    prominence_factor: float = 0.1,
    n_restarts: int = 64,
    seed: int | None = 0,
    **kwargs,
) -> dict:
    """
    Experiment 6 (PLAN.md): direct basin-vs-ridge typing of a witness neuron.

    Uses the radial profile of the neuron around the feature's centroid:
    p(r) = mean of f over field points at distance ~r from the centroid,
    for r in [0, 1.5 * r_A] where r_A = mean feature radius.

      - "ridge": p has an interior local extremum whose radius is within
        ``ridge_location_tol * r_A`` of the ring radius and whose prominence
        over the profile ends exceeds ``prominence_factor * range(p)`` —
        a crest/trough OVER the ring (level sets hug the ring on both sides);
      - "basin": p is essentially monotone across the ring — the governing
        extremum sits over the INTERIOR (level sets sweep the interior
        boundary as c moves);
      - "ambiguous": neither pattern is clear.

    Also reports the extremum-sampling anchor nearest to the feature center
    (its position/value distinguish an interior basin anchor from a ring
    crest anchor) — feeds exp-6's density/relevance prediction test and the
    per-seed bimodality analysis (A.7).
    """
    feature_points = feature_points.to(device)
    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    if field is None:
        field = evaluate_field_on_grid(
            function,
            bounds=bounds,
            resolution=resolution,
            device=device,
            dtype=feature_points.dtype,
        )

    center = feature_points.mean(dim=0)
    feature_radii = (feature_points - center).norm(dim=1)
    r_feature = float(feature_radii.mean().item())
    r_max = 1.5 * r_feature

    point_radii = (field.points - center).norm(dim=1)
    values = field.values

    edges = torch.linspace(0.0, r_max, n_radial_bins + 1)
    profile = []
    radii_mid = []

    for i in range(n_radial_bins):
        mask = (point_radii >= edges[i]) & (point_radii < edges[i + 1])
        if int(mask.sum().item()) == 0:
            continue
        profile.append(float(values[mask].mean().item()))
        radii_mid.append(float((edges[i] + edges[i + 1]).item() / 2.0))

    result = {
        "witness_type": "ambiguous",
        "r_feature": r_feature,
        "radial_profile_monotonicity": float("nan"),
        "ridge_location": float("nan"),
        "ridge_prominence": float("nan"),
        "anchor_point": None,
        "anchor_value": float("nan"),
        "anchor_dist_to_center": float("nan"),
        "anchor_dist_to_feature": float("nan"),
        "profile_r": radii_mid,
        "profile_values": profile,
    }

    if len(profile) < 5:
        return result

    p = torch.tensor(profile)
    r = torch.tensor(radii_mid)
    p_range = float((p.max() - p.min()).item())

    if p_range <= 1e-9:
        return result

    diffs = torch.diff(p)
    overall_sign = 1.0 if float(p[-1] - p[0]) >= 0 else -1.0
    monotonicity = float((torch.sign(diffs) == overall_sign).float().mean().item())
    result["radial_profile_monotonicity"] = monotonicity

    # Interior extremum of the profile (crest or trough), excluding the ends.
    interior = slice(1, len(p) - 1)
    idx_max = 1 + int(torch.argmax(p[interior]).item())
    idx_min = 1 + int(torch.argmin(p[interior]).item())

    def _prominence(idx: int, is_max: bool) -> float:
        v = float(p[idx].item())
        end_ref = max(float(p[0].item()), float(p[-1].item())) if is_max else min(
            float(p[0].item()), float(p[-1].item())
        )
        return (v - end_ref) if is_max else (end_ref - v)

    candidates = [
        (idx_max, _prominence(idx_max, True)),
        (idx_min, _prominence(idx_min, False)),
    ]
    best_idx, best_prominence = max(candidates, key=lambda t: t[1])
    ridge_r = float(r[best_idx].item())

    result["ridge_location"] = ridge_r
    result["ridge_prominence"] = best_prominence

    is_ridge = (
        best_prominence >= prominence_factor * p_range
        and abs(ridge_r - r_feature) <= ridge_location_tol * r_feature
    )

    if is_ridge:
        result["witness_type"] = "ridge"
    elif monotonicity >= 0.8:
        result["witness_type"] = "basin"

    # Nearest anchor from extremum sampling (diagnostics for exp. 6).
    extrema_kwargs = dict(
        bounds=bounds,
        n_restarts=n_restarts,
        record_trajectories=False,
        device=device,
        seed=seed,
    )
    anchors = []
    for mode in ("max", "min"):
        res = find_local_extrema(function, mode=mode, **extrema_kwargs)
        for i in range(len(res.points)):
            anchors.append((res.points[i], float(res.values[i].item())))

    if anchors:
        center_cpu = center.detach().cpu()
        dists = [float((pt - center_cpu).norm().item()) for pt, _ in anchors]
        best = int(torch.argmin(torch.tensor(dists)).item())
        anchor_point, anchor_value = anchors[best]

        result["anchor_point"] = anchor_point.tolist()
        result["anchor_value"] = anchor_value
        result["anchor_dist_to_center"] = dists[best]
        result["anchor_dist_to_feature"] = float(
            nearest_distances_to_set(
                anchor_point[None, :], feature_points.detach().cpu()
            ).item()
        )

    return result
