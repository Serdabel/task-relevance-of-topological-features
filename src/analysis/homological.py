"""Criterion 4: homological (dimension-invariant) witness criterion.

Persistent homology of the superlevel filtration {f >= c} of a neuron,
computed on a finite sampled point cloud (PROJECT.md §Criterion 4; the main
criterion for d >= 3, same language as Naitzat-Lim). A neuron witnesses a
feature with known homology class [A] when its filtration produces a
persistent bar of the right dimension at the right scale.

Key implementation points (PLAN.md):
  - the complex is built on a finite point cloud, NOT on a grid over R^d —
    ambient dimension enters only through pairwise distances (a neighborhood
    graph), which is why this criterion survives high dimension;
  - superlevel of f = sublevel (lower-star) of -f on the neighborhood
    complex: vertices born at -f(x), an edge born when both endpoints exist;
  - BOTH orientations are checked: a ring encoded as a crest (annulus in
    {f >= c}) fires on f, a ring encoded as a basin boundary fires on -f
    (the superlevel sets of -f around a basin are annuli in the box) — the
    better-matching orientation is reported;
  - locality: by default the filtration runs on the sample restricted to a
    neighborhood of the feature (``neighborhood_factor`` x feature radius
    around its centroid), so a persistent cycle elsewhere in the domain does
    not count for THIS feature. The decisive specificity tool remains the
    null baseline (analysis.controls).

Backends: ``ripser`` (primary, PLAN.md) and ``gudhi`` (fallback, already in
the env). Both are imported INSIDE the functions — the core pipeline must
import without TDA packages installed.

Integration: ``extra_criteria={"homological": homological_criterion}`` — same
plug-in route as criterion 3, columns auto-detected downstream.
"""

import torch

from analysis.fields import FieldSample, evaluate_field_on_grid
from analysis.geometry import (
    feature_nearest_neighbor_spacing,
    evaluate_scalar_function_on_points,
)


def _neighborhood_radius(points: torch.Tensor, factor: float = 2.0) -> float:
    spacing = float(feature_nearest_neighbor_spacing(points).item())
    return max(factor * spacing, 1e-8)


def _superlevel_diagrams_ripser(points, neg_values, radius, max_homology_dim):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree
    from ripser import ripser

    points_np = points.detach().cpu().numpy()
    v = neg_values.detach().cpu().numpy().astype("float64")
    n = len(points_np)

    tree = cKDTree(points_np)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")

    if len(pairs) == 0:
        rows = np.arange(n)
        cols = np.arange(n)
        vals = v
    else:
        i, j = pairs[:, 0], pairs[:, 1]
        edge_vals = np.maximum(v[i], v[j])
        rows = np.concatenate([np.arange(n), i, j])
        cols = np.concatenate([np.arange(n), j, i])
        vals = np.concatenate([v, edge_vals, edge_vals])

    D = coo_matrix((vals, (rows, cols)), shape=(n, n))

    result = ripser(D, distance_matrix=True, maxdim=max_homology_dim)
    return result["dgms"]


def _superlevel_diagrams_delaunay(points, neg_values, max_homology_dim):
    """Lower-star filtration on the Delaunay (alpha) complex.

    The right complex for RANDOM volumetric samples (the d >= 3 field
    source): a radius graph on a random cloud has persistent geometric
    voids — spurious H_k bars spanning most of the filtration that have
    nothing to do with f — while the Delaunay complex triangulates the
    convex hull without sample voids, so only the function's topology
    persists. Simplices come from gudhi.AlphaComplex (geometry only); their
    alpha filtration values are REPLACED by the lower-star rule
    (max over vertices of -f).
    """
    import numpy as np
    import gudhi

    points_np = points.detach().cpu().numpy().astype("float64")
    v = neg_values.detach().cpu().numpy().astype("float64")
    n = len(points_np)

    alpha = gudhi.AlphaComplex(points=points_np)
    st = alpha.create_simplex_tree()

    assert st.num_vertices() == n, (
        "AlphaComplex dropped points (duplicates?); vertex ids would not "
        "match the value array"
    )

    for simplex, _ in st.get_simplices():
        st.assign_filtration(simplex, float(v[simplex].max()))

    st.make_filtration_non_decreasing()
    st.compute_persistence()

    diagrams = []
    for dim in range(max_homology_dim + 1):
        bars = st.persistence_intervals_in_dimension(dim)
        diagrams.append(np.asarray(bars).reshape(-1, 2))

    return diagrams


def _superlevel_diagrams_gudhi(points, neg_values, radius, max_homology_dim):
    import numpy as np
    from scipy.spatial import cKDTree
    import gudhi

    points_np = points.detach().cpu().numpy()
    v = neg_values.detach().cpu().numpy().astype("float64")
    n = len(points_np)

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=float(v[i]))

    tree = cKDTree(points_np)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")

    for i, j in pairs:
        st.insert([int(i), int(j)], filtration=float(max(v[i], v[j])))

    st.expansion(max_homology_dim + 1)
    st.compute_persistence()

    diagrams = []
    for dim in range(max_homology_dim + 1):
        bars = st.persistence_intervals_in_dimension(dim)
        diagrams.append(np.asarray(bars).reshape(-1, 2))

    return diagrams


def superlevel_persistence(
    points: torch.Tensor,
    values: torch.Tensor,
    max_homology_dim: int = 1,
    backend: str = "ripser",
    graph_radius: float | None = None,
    graph_radius_factor: float = 2.0,
    clip_essential: bool = True,
    **backend_kwargs,
) -> dict:
    """
    Persistent homology of the superlevel filtration {f >= c} on a sampled
    cloud: the lower-star filtration of -f on a complex over ``points`` —
    a radius graph (backends "ripser"/"gudhi"; radius = ``graph_radius``,
    default ``graph_radius_factor`` x median nearest-neighbor spacing) or
    the Delaunay/alpha complex (backend "delaunay"; no radius parameter).
    Prefer "delaunay" for random volumetric samples: radius graphs on
    random clouds carry spurious sample-void bars (see
    _superlevel_diagrams_delaunay), grids do not.

    Returns dict:
        diagrams: {dim: Tensor [n_bars, 2]} — (birth_level, death_level) in
            units of f, birth >= death (the filtration sweeps c downward);
            essential classes are clipped to the lowest swept level (the
            standard finite-range convention — infinite persistence would
            poison persistence arithmetic downstream); pass
            clip_essential=False for raw -inf deaths.
        backend: str
        n_points: int
        graph_radius: float
    """
    assert points.ndim == 2, "points must have shape [N, dim]"
    assert values.ndim == 1 and len(values) == len(points)
    assert backend in {"ripser", "gudhi", "delaunay"}, (
        "backend must be 'ripser', 'gudhi' or 'delaunay'"
    )

    neg_values = -values

    if backend == "delaunay":
        graph_radius = float("nan")
        raw = _superlevel_diagrams_delaunay(
            points, neg_values, max_homology_dim
        )
    else:
        if graph_radius is None:
            graph_radius = _neighborhood_radius(
                points, factor=graph_radius_factor
            )

        if backend == "ripser":
            try:
                raw = _superlevel_diagrams_ripser(
                    points, neg_values, graph_radius, max_homology_dim
                )
            except ImportError:
                backend = "gudhi"
                raw = _superlevel_diagrams_gudhi(
                    points, neg_values, graph_radius, max_homology_dim
                )
        else:
            raw = _superlevel_diagrams_gudhi(
                points, neg_values, graph_radius, max_homology_dim
            )

    min_level = float(values.min().item())

    diagrams = {}
    for dim in range(max_homology_dim + 1):
        bars = torch.as_tensor(raw[dim], dtype=torch.float32).reshape(-1, 2)
        # (b, d) in -f units -> (birth_level, death_level) = (-b, -d) in f units.
        bars = torch.stack([-bars[:, 0], -bars[:, 1]], dim=1)
        if clip_essential and len(bars) > 0:
            bars[:, 1] = torch.clamp(bars[:, 1], min=min_level)
        diagrams[dim] = bars

    return {
        "diagrams": diagrams,
        "backend": backend,
        "n_points": int(len(points)),
        "graph_radius": float(graph_radius),
    }


def persistence_of_point_cloud(
    points: torch.Tensor,
    max_homology_dim: int = 1,
    backend: str = "ripser",
    max_edge_length: float | None = None,
    **backend_kwargs,
) -> dict:
    """
    Plain geometric persistence (Rips on pairwise distances) of a point set.

    Needed by experiment 11's control: compare persistence diagrams of the
    data BEFORE vs AFTER dimensionality reduction — if the reduction destroyed
    the relevant homology, "task-relevance after PCA" measures a projection
    artifact, not the network (PLAN.md exp. 11 caveat). Also useful to verify
    sampled level-set clouds against the feature's known H_*.

    Returns dict: diagrams {dim: Tensor [n_bars, 2]} (birth, death in
    distance units), backend, n_points.
    """
    assert points.ndim == 2, "points must have shape [N, dim]"
    assert backend in {"ripser", "gudhi"}

    points_np = points.detach().cpu().numpy()

    if backend == "ripser":
        try:
            from ripser import ripser

            kwargs = {}
            if max_edge_length is not None:
                kwargs["thresh"] = float(max_edge_length)

            raw = ripser(points_np, maxdim=max_homology_dim, **kwargs)["dgms"]
        except ImportError:
            backend = "gudhi"

    if backend == "gudhi":
        import numpy as np
        import gudhi

        rips = gudhi.RipsComplex(
            points=points_np,
            max_edge_length=(
                float(max_edge_length) if max_edge_length is not None else float("inf")
            ),
        )
        st = rips.create_simplex_tree(max_dimension=max_homology_dim + 1)
        st.compute_persistence()
        raw = [
            np.asarray(st.persistence_intervals_in_dimension(dim)).reshape(-1, 2)
            for dim in range(max_homology_dim + 1)
        ]

    diagrams = {
        dim: torch.as_tensor(raw[dim], dtype=torch.float32).reshape(-1, 2)
        for dim in range(max_homology_dim + 1)
    }

    return {
        "diagrams": diagrams,
        "backend": backend,
        "n_points": int(len(points)),
    }


def match_bars_to_feature(
    diagrams: dict,
    feature_homology: dict,
    min_persistence: float | None = None,
    min_persistence_floor: float = 0.0,
) -> dict:
    """
    Decides whether ``diagrams`` contain persistent bars matching the
    feature's known homology signature.

    feature_homology:
        e.g. {1: 1} for a circle (one H_1 class), {2: 1} for S^2,
        {1: 2} for two independent loops.

    A bar matches when its homological dimension is right and its persistence
    exceeds ``min_persistence``. With min_persistence=None the threshold is
    adaptive: the k-th bar must dominate the (k+1)-th ("noise") bar by a
    factor of 2 AND exceed ``min_persistence_floor`` (an absolute floor;
    dominance alone declares two tiny noise bars a match). The MARGIN is
    reported, not just the boolean: with an explicit threshold
    margin = p_k - min_persistence, adaptively
    margin = p_k - max(p_{k+1}, floor).

    Returns dict:
        is_match: bool
        matched_bars: {dim: Tensor [k, 2]}
        persistence_margin: float (min over required dimensions)
        per_dim: {dim: {"p_k": ..., "p_next": ..., "matched": ...}}
    """
    assert len(feature_homology) > 0, "feature_homology must be non-empty"

    matched_bars = {}
    per_dim = {}
    margins = []
    all_matched = True

    for dim, k in feature_homology.items():
        assert k >= 1, "required bar counts must be >= 1"

        bars = diagrams.get(dim, torch.empty(0, 2))
        if len(bars) == 0:
            # NaN, not 0.0: an EMPTY diagram and a bar that existed but lost are
            # different facts and must not collapse to the same number. A real
            # near-miss returns a NEGATIVE margin (p_k - max(p_next, floor)), so
            # 0.0 here used to sit above every genuine failure while meaning
            # "nothing was computed at all". On halfplane_base every one of the
            # 35 neurons yields an empty H_1 diagram, which made the column a
            # constant 0.0 that then read as three criteria agreeing — counting
            # a constant as a vote (AUDIT/09 section B.3). NaN propagates
            # visibly through the aggregates instead.
            per_dim[dim] = {"p_k": 0.0, "p_next": 0.0, "matched": False}
            matched_bars[dim] = torch.empty(0, 2)
            margins.append(float("nan"))
            all_matched = False
            continue

        persistence = (bars[:, 0] - bars[:, 1]).abs()
        order = torch.argsort(persistence, descending=True)
        sorted_pers = persistence[order]

        p_k = float(sorted_pers[k - 1].item()) if len(sorted_pers) >= k else 0.0
        p_next = float(sorted_pers[k].item()) if len(sorted_pers) > k else 0.0

        if min_persistence is not None:
            matched = p_k >= float(min_persistence)
            margin = p_k - float(min_persistence)
        else:
            floor = float(min_persistence_floor)
            matched = (
                p_k > 0
                and (p_next == 0.0 or p_k >= 2.0 * p_next)
                and p_k >= floor
            )
            margin = p_k - max(p_next, floor)

        per_dim[dim] = {"p_k": p_k, "p_next": p_next, "matched": matched}
        matched_bars[dim] = bars[order[:k]]
        margins.append(margin)
        all_matched &= matched

    # NaN wins over any number: if ANY required dimension had no diagram, the
    # overall margin is undefined, not "the smallest of the ones we did get".
    # Python's min() is order-dependent in the presence of NaN
    # (min([nan, 1.0]) is nan but min([1.0, nan]) is 1.0), so decide explicitly.
    overall = (
        float("nan") if any(m != m for m in margins)
        else float(min(margins))
    )

    return {
        "is_match": bool(all_matched),
        "matched_bars": matched_bars,
        "persistence_margin": overall,
        "per_dim": per_dim,
    }


def homological_criterion(
    function,
    feature_points: torch.Tensor,
    bounds,
    resolution=None,
    device: str = "cpu",
    field: FieldSample | None = None,

    # sampling: which point cloud carries the filtration
    sample_source: str = "field",  # "field" | "random"
    max_sample_points: int = 4000,
    neighborhood_factor: float | None = 1.5,
    seed: int = 0,

    # persistence
    feature_homology: dict | str | None = None,
    max_homology_dim: int = 1,
    backend: str = "ripser",
    min_persistence: float | None = None,
    relative_floor: float = 0.1,
    graph_radius_factor: float = 2.0,
    **kwargs,
) -> dict:
    """
    Criterion-4 entry point, extra_criteria-compatible.

    1. obtains the sample: the shared ``field`` (its points+values,
       subsampled to ``max_sample_points``), or a fresh uniform-random cloud
       inside ``bounds`` (sample_source="random" — the grid-free route);
    2. optionally restricts it to the feature neighborhood
       (``neighborhood_factor`` x max feature radius around the feature
       centroid) so distant cycles do not count;
    3. superlevel persistence for BOTH orientations (f and -f);
    4. match_bars_to_feature against ``feature_homology``
       (default {1: 1}: one circle-like loop; the string "codim1" resolves to
       {ambient_dim - 1: 1} at call time — the S^{d-1} shell signature, which
       adapts per dataset in mixed-dimension experiments like exp. 9); with
       adaptive matching the absolute floor is ``relative_floor`` x the
       sample's value range;
    5. returns flat numerics for the CSV.

    Combinatorial cost caveat: the clique expansion of the radius graph grows
    steeply with the homology dimension — for H_3 (S^3 in R^4) keep
    ``max_sample_points`` small (hundreds) and tune ``graph_radius_factor``.

    Returns dict:
        is_witness, persistence_margin, n_matched_bars,
        top_bar_birth, top_bar_death, orientation (1.0 = f, -1.0 = -f),
        n_sample_points
    """
    feature_points = feature_points.to("cpu")
    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    if feature_homology is None:
        feature_homology = {1: 1}
    elif isinstance(feature_homology, str):
        assert feature_homology == "codim1", (
            f"Unknown feature_homology sentinel: {feature_homology!r} "
            "(the only supported string is 'codim1')"
        )
        feature_homology = {feature_points.shape[1] - 1: 1}

    max_homology_dim = max(max_homology_dim, max(feature_homology.keys()))

    generator = torch.Generator()
    generator.manual_seed(seed)

    if sample_source == "field" and field is None and resolution is not None:
        field = evaluate_field_on_grid(
            function, bounds=bounds, resolution=resolution, device=device
        )

    if sample_source == "field":
        assert field is not None, (
            "sample_source='field' needs a FieldSample (or resolution to build one)"
        )
        points = field.points.detach().cpu()
        values = field.values.detach().cpu()
    elif sample_source in {"random", "sampler"}:
        lows = torch.tensor([b[0] for b in bounds])
        highs = torch.tensor([b[1] for b in bounds])
        points = lows + (highs - lows) * torch.rand(
            max_sample_points, len(bounds), generator=generator
        )
        values = evaluate_scalar_function_on_points(function, points.to(device)).cpu()
        points = points.cpu()
    else:
        raise ValueError(
            f"Unknown sample_source: {sample_source!r} (use 'field' or 'random')"
        )

    if neighborhood_factor is not None:
        center = feature_points.mean(dim=0)
        r_feature = float((feature_points - center).norm(dim=1).max().item())
        window = neighborhood_factor * max(r_feature, 1e-8)
        keep = (points - center).norm(dim=1) <= window
        points = points[keep]
        values = values[keep]

    if len(points) > max_sample_points:
        idx = torch.randperm(len(points), generator=generator)[:max_sample_points]
        points = points[idx]
        values = values[idx]

    empty = {
        "is_witness": False,
        "persistence_margin": float("nan"),
        "n_matched_bars": 0,
        "n_candidate_bars": 0,
        "top_bar_birth": float("nan"),
        "top_bar_death": float("nan"),
        "orientation": float("nan"),
        "n_sample_points": int(len(points)),
    }

    if len(points) < 8:
        return empty

    value_range = float((values.max() - values.min()).item())
    persistence_floor = relative_floor * max(value_range, 1e-12)

    best = None

    for orientation in (1.0, -1.0):
        persistence = superlevel_persistence(
            points,
            orientation * values,
            max_homology_dim=max_homology_dim,
            backend=backend,
            graph_radius_factor=graph_radius_factor,
        )

        match = match_bars_to_feature(
            persistence["diagrams"],
            feature_homology=feature_homology,
            min_persistence=min_persistence,
            min_persistence_floor=persistence_floor,
        )

        candidate = {
            "orientation": orientation,
            "match": match,
            "diagrams": persistence["diagrams"],
        }

        # Pick the better orientation, treating "no diagram at all" (NaN) as
        # strictly worse than any computed margin, however negative. Two bugs
        # are avoided here. With the old 0.0-for-empty convention an EMPTY
        # diagram outranked a genuine sub-threshold bar, whose margin is
        # negative — so the reported orientation could be the one that found
        # nothing (AUDIT/09 section B.3). And a bare `>` against NaN is always
        # False, which would now make the FIRST orientation win by default.
        if best is None:
            best = candidate
        else:
            new_m = match["persistence_margin"]
            old_m = best["match"]["persistence_margin"]
            new_ok, old_ok = new_m == new_m, old_m == old_m  # not NaN
            if (new_ok and not old_ok) or (new_ok and old_ok and new_m > old_m):
                best = candidate

    match = best["match"]

    top_dim = max(feature_homology.keys())
    top_bars = match["matched_bars"].get(top_dim, torch.empty(0, 2))

    if len(top_bars) > 0:
        top_bar_birth = float(top_bars[0, 0].item())
        top_bar_death = float(top_bars[0, 1].item())
    else:
        top_bar_birth = float("nan")
        top_bar_death = float("nan")

    return {
        "is_witness": bool(match["is_match"]),
        "persistence_margin": float(match["persistence_margin"]),
        # Bars that actually MATCHED. matched_bars is populated for every
        # dimension that had any bars at all, whether or not the dominance test
        # passed, so summing it unconditionally reported "1 matched bar" beside
        # a False verdict — the confusing pair AUDIT/05 section A.1 tabulated.
        # Count only the dimensions whose per-dim verdict is True; the raw
        # count stays available as n_candidate_bars.
        "n_matched_bars": int(sum(
            len(bars) for dim, bars in match["matched_bars"].items()
            if match["per_dim"].get(dim, {}).get("matched", False)
        )),
        "n_candidate_bars": int(sum(
            len(b) for b in match["matched_bars"].values()
        )),
        "top_bar_birth": top_bar_birth,
        "top_bar_death": top_bar_death,
        "orientation": float(best["orientation"]),
        "n_sample_points": int(len(points)),
    }
