"""PCA-slice sampling: the high-dimensional route (mandatory at d >= 10).

PLAN.md §Sampling, "PCA-slice sampling". Pipeline per (neuron, anchor):

    1. anchor cloud in AMBIENT space:
         - main case: the gradient-ascent trajectory to a local extremum
           (analysis.extremum_sampling, Wang-Ponce), or
         - alternative: the feature points A themselves, when A is known to
           lie in its own low-dimensional affine plane;
    2. PCA -> k-dim affine subspace P
       (analysis.geometry.fit_affine_subspace);
    3. restrict the neuron: f|_P via models.NN_analysis.SubspaceRestrictedFunction;
    4. f|_P is a scalar field on R^k (k = 2 or 3) -> the ENTIRE existing
       toolkit (grid criteria, adaptive eps, heatmaps) applies unchanged in
       PC coordinates.

The order is fixed: subspace FIRST, level sets SECOND — projecting
already-extracted level sets is invalid (the operations do not commute).

Certification asymmetry (PLAN.md): one slice can CONFIRM structure but cannot
certify its absence — use several slices per neuron and/or cross-check with
the homological criterion before reporting a negative.
"""

import torch

from models.NN_analysis import SubspaceRestrictedFunction
from analysis.extremum_sampling import find_local_extrema


def pca_slice_from_trajectory(
    function,
    trajectory_points: torch.Tensor,
    k: int = 2,
    bounds_margin: float = 0.5,
) -> dict:
    """
    Builds one slice from an ambient-space anchor cloud.

    fit_affine_subspace(trajectory_points, k) -> SubspaceRestrictedFunction;
    slice bounds = bounding box of the projected anchor cloud + margin;
    explained variance is recorded — a slice explaining little trajectory
    variance is a weak anchor, flag it downstream.

    Returns dict:
        restricted: SubspaceRestrictedFunction
        bounds: [(lo, hi)] * k  (in PC coordinates)
        explained_variance_ratio: float (top-k share)
        anchor_kind: "trajectory"
    """
    assert trajectory_points.ndim == 2 and len(trajectory_points) >= 2

    restricted = SubspaceRestrictedFunction.from_points(
        function, trajectory_points, k=k
    )

    bounds = restricted.bounds_for_points(
        trajectory_points.to(restricted.origin.dtype), margin=bounds_margin
    )

    explained = float(restricted.explained_variance_ratio[:k].sum().item())

    return {
        "restricted": restricted,
        "bounds": bounds,
        "explained_variance_ratio": explained,
        "anchor_kind": "trajectory",
    }


def evaluate_neuron_on_slice(
    slice_info: dict,
    feature_points: torch.Tensor,
    resolution: int = 100,
    device: str = "cpu",
    structural_kwargs: dict | None = None,
    out_of_plane_factor: float = 0.25,
    min_feature_points_in_plane: int = 10,
    **kwargs,
) -> dict:
    """
    Runs the 2D/3D grid criteria on a restricted neuron, in PC coordinates.

    The feature is projected into the slice (restricted.project); points of A
    lying far FROM the slice plane are excluded — scoring the projection of a
    feature that lives nowhere near P is meaningless. "Far" means an
    out-of-plane distance above ``out_of_plane_factor`` x the feature
    diameter; the out-of-plane distribution is reported either way. When too
    few feature points survive the cut the slice is returned as invalid
    (valid=False, no witness claim) rather than scored.

    Returns the structural-criterion dict + slice diagnostics
    (valid, n_feature_points_in_plane, out_of_plane_q50/q90,
    explained_variance_ratio).
    """
    from analysis.criterions import structural_criterion
    from analysis.geometry import feature_diameter

    restricted: SubspaceRestrictedFunction = slice_info["restricted"]
    bounds = slice_info["bounds"]

    feature_points = feature_points.detach().float()
    assert feature_points.shape[1] == restricted.ambient_dim

    u = restricted.project(feature_points)              # in-plane coords
    back = restricted.embed(u)                          # closest points on P
    out_of_plane = (feature_points - back).norm(dim=1)

    diameter = float(feature_diameter(feature_points, method="exact").item())
    cutoff = out_of_plane_factor * max(diameter, 1e-12)

    keep = out_of_plane <= cutoff
    u_kept = u[keep]

    q50 = float(out_of_plane.median().item())
    q90 = float(torch.quantile(out_of_plane, 0.9).item())

    diagnostics = {
        "valid": True,
        "n_feature_points_in_plane": int(keep.sum().item()),
        "n_feature_points_total": int(len(feature_points)),
        "out_of_plane_q50": q50,
        "out_of_plane_q90": q90,
        "out_of_plane_cutoff": cutoff,
        "explained_variance_ratio": slice_info["explained_variance_ratio"],
    }

    if int(keep.sum().item()) < min_feature_points_in_plane:
        diagnostics["valid"] = False
        diagnostics["is_structural_witness"] = False
        diagnostics["score_normalized_by_diameter"] = float("nan")
        return diagnostics

    structural_kwargs = {} if structural_kwargs is None else dict(structural_kwargs)

    result = structural_criterion(
        function=restricted,
        feature_points=u_kept,
        bounds=bounds,
        resolution=resolution,
        device=device,
        **structural_kwargs,
    )

    result.update(diagnostics)
    return result


def multi_slice_witness_report(
    function,
    feature_points: torch.Tensor,
    bounds,
    n_slices: int = 5,
    k: int = 2,
    device: str = "cpu",
    n_restarts: int = 64,
    resolution: int = 100,
    structural_kwargs: dict | None = None,
    include_feature_plane_slice: bool = True,
    seed: int | None = 0,
    **kwargs,
) -> dict:
    """
    The multi-slice policy: several anchors (gradient-ascent trajectories to
    different extrema, plus optionally the feature's own PCA plane) ->
    several slices -> per-slice scores -> a single verdict.

    Verdict semantics (PLAN.md certification asymmetry):
        "witness"      — ANY valid slice confirms (confirmation is monotone);
        "no_evidence"  — no valid slice confirms; this is NOT "no witness" —
                         pair with the homological criterion for negatives.

    Returns dict:
        verdict: "witness" | "no_evidence"
        is_witness: bool
        best_score_normalized: float
        slices: list of per-slice result dicts (with anchor metadata)
    """
    extrema_kwargs = dict(
        bounds=bounds,
        n_restarts=n_restarts,
        record_trajectories=True,
        device=device,
        seed=seed,
    )

    anchors = []

    for mode in ("max", "min"):
        result = find_local_extrema(function, mode=mode, **extrema_kwargs)
        for i, trajectory in enumerate(result.trajectories):
            anchors.append(
                {
                    "kind": f"trajectory_{mode}",
                    "points": trajectory,
                    "value": float(result.values[i].item()),
                }
            )

    # Strongest anchors first (most extreme critical values).
    anchors.sort(key=lambda a: -abs(a["value"]))
    anchors = anchors[:n_slices]

    if include_feature_plane_slice:
        anchors.append(
            {"kind": "feature_plane", "points": feature_points.float(), "value": None}
        )

    slices = []
    best_score = float("inf")
    any_confirms = False

    for anchor in anchors:
        if len(anchor["points"]) < 2:
            continue

        slice_info = pca_slice_from_trajectory(
            function, anchor["points"], k=k
        )

        result = evaluate_neuron_on_slice(
            slice_info,
            feature_points,
            resolution=resolution,
            device=device,
            structural_kwargs=structural_kwargs,
        )

        result["anchor_kind"] = anchor["kind"]
        result["anchor_value"] = anchor["value"]
        slices.append(result)

        if result["valid"]:
            score = result.get("score_normalized_by_diameter", float("nan"))
            if score == score and score < best_score:
                best_score = score
            if result.get("is_structural_witness", False):
                any_confirms = True

    return {
        "verdict": "witness" if any_confirms else "no_evidence",
        "is_witness": any_confirms,
        "best_score_normalized": best_score,
        "slices": slices,
    }
