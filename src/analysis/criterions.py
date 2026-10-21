"""Neuron-level feature-detection criteria.

Implemented criteria (see PROJECT.md §Feature Detection Criteria):

    - Structural (Chamfer) criterion — ``structural_criterion``:
      sweeps levels c, extracts connected components of {x : |f(x) - c| <= eps},
      and scores the best component's max-symmetrized Chamfer distance to the
      feature point set A, normalized by diam(A). Witness threshold: 0.1.

    - Tubular (Wasserstein) criterion — ``tubular_criterion``:
      statistical baseline comparing the distribution of f-values on a tube
      around A against a surrounding shell.

``evaluate_neurons_on_feature`` runs the criteria over all selected neurons of
a model. Future criteria (extremum-anchored, homological) should be added as
new modules in ``analysis/`` and hooked in via its ``extra_criteria`` argument.

Criteria consume an evaluated ``FieldSample`` (see ``analysis.fields``): the
grid is the default producer, but any point sample works — component extraction
dispatches between grid image-labeling and point-cloud clustering. The
sampler-backed variant ``structural_criterion_sampled`` connects the gradient
samplers from ``analysis.level_set_sampling`` for d >= 3, where a grid is
infeasible.

Generic geometry/grid utilities live in ``analysis.geometry`` (re-exported here
for backward compatibility), MLP introspection in ``models.NN_analysis``.
"""

import math

import torch
from scipy.stats import wasserstein_distance

from models.NN_analysis import (
    TruncatedMLPFunction,
    ScalarProjection,
    validate_mlp_layer_id,
    infer_activation_name_from_model,
    get_hidden_dim,
    infer_level_activation_for_neuron,
)
from analysis.geometry import (
    mean_set_to_set_distance,
    max_chamfer_distance,
    nearest_distances_to_set,
    feature_diameter,
    feature_nearest_neighbor_spacing,
    make_connectivity_structure,
    connected_components,
    connected_components_pointcloud,
    make_grid,
    grid_step_from_bounds_resolution,
    evaluate_scalar_function_on_points,
)
from analysis.fields import (
    FieldSample,
    evaluate_field_on_grid,
    field_from_points,
)

__all__ = [
    # criteria
    "structural_criterion",
    "structural_criterion_sampled",
    "sampled_structural_criterion",
    "tubular_criterion",
    "evaluate_neurons_on_feature",
    "best_component_chamfer_score",
    # level construction
    "relu_exponential_levels",
    "make_candidate_levels",
    "choose_local_adaptive_level_eps",
    "collect_local_neighbor_value_jumps",
    "median_level_step",
    # tubular helpers
    "resolve_tubular_radii",
    "normalized_wasserstein_1_distance",
    "auc_from_scores",
    # re-exported geometry (canonical home: analysis.geometry)
    "mean_set_to_set_distance",
    "max_chamfer_distance",
    "nearest_distances_to_set",
    "feature_diameter",
    "feature_nearest_neighbor_spacing",
    "make_connectivity_structure",
    "connected_components",
    "make_grid",
    "grid_step_from_bounds_resolution",
    "evaluate_scalar_function_on_points",
    # re-exported model introspection (canonical home: models.NN_analysis)
    "validate_mlp_layer_id",
    "infer_activation_name_from_model",
    "get_hidden_dim",
    "infer_level_activation_for_neuron",
]


# -------------------------
# Candidate levels
# -------------------------

def relu_exponential_levels(
    low: float,
    high: float,
    n_levels: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    min_positive: float = 1e-4,
):
    """
    Exponential level grid for ReLU-like nonnegative activations.

    Includes 0 if low <= 0.
    """
    assert n_levels >= 1, "n_levels must be positive"

    low = float(low)
    high = float(high)

    if high <= 0:
        return torch.zeros(n_levels, device=device, dtype=dtype)

    if n_levels == 1:
        return torch.tensor([0.0], device=device, dtype=dtype)

    if low <= 0:
        positive_low = max(min_positive, high * 1e-4)
        positive_levels = torch.logspace(
            math.log10(positive_low),
            math.log10(high),
            steps=n_levels - 1,
            device=device,
            dtype=dtype,
        )
        levels = torch.cat(
            [torch.zeros(1, device=device, dtype=dtype), positive_levels],
            dim=0,
        )
        return levels

    positive_low = max(low, min_positive)

    return torch.logspace(
        math.log10(positive_low),
        math.log10(high),
        steps=n_levels,
        device=device,
        dtype=dtype,
    )


def make_candidate_levels(
    n_levels: int,
    activation: str | None = None,
    level_bounds: tuple[float, float] | None = None,
    observed_values: torch.Tensor | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    activation:
        "tanh"    -> linear levels in [-1, 1]
        "sigmoid" -> linear levels in [0, 1]
        "relu"    -> exponential levels in [0, max_value]
        None      -> use level_bounds or observed_values range
    """
    assert n_levels >= 1, "n_levels must be positive"

    if activation is not None:
        activation = activation.lower()

    if level_bounds is not None:
        low, high = level_bounds

        if activation == "relu":
            return relu_exponential_levels(
                low=low,
                high=high,
                n_levels=n_levels,
                device=device,
                dtype=dtype,
            )

        return torch.linspace(
            low,
            high,
            n_levels,
            device=device,
            dtype=dtype,
        )

    if activation == "tanh":
        return torch.linspace(-1.0, 1.0, n_levels, device=device, dtype=dtype)

    if activation == "sigmoid":
        return torch.linspace(0.0, 1.0, n_levels, device=device, dtype=dtype)

    if activation == "relu":
        assert observed_values is not None, (
            "For activation='relu' without level_bounds, observed_values must be provided"
        )
        high = float(observed_values.max().item())
        return relu_exponential_levels(
            low=0.0,
            high=high,
            n_levels=n_levels,
            device=device,
            dtype=dtype,
        )

    if activation is not None:
        raise ValueError("activation must be one of: None, 'tanh', 'sigmoid', 'relu'")

    assert observed_values is not None, (
        "Either activation, level_bounds, or observed_values must be provided"
    )

    low = float(observed_values.min().item())
    high = float(observed_values.max().item())

    return torch.linspace(low, high, n_levels, device=device, dtype=dtype)


# -------------------------
# Adaptive level_eps
# -------------------------

def collect_local_neighbor_value_jumps(
    Z: torch.Tensor,
    local_mask: torch.Tensor,
    edge_mode: str = "either",
) -> torch.Tensor:
    """
    Collects |f(x_next) - f(x)| over neighboring grid points near the feature.

    Z:
        Tensor of shape [N1, ..., Nd], values of f on grid.

    local_mask:
        Boolean tensor of the same shape. True means that the grid point is
        considered to be near the feature.

    edge_mode:
        "both":
            use only edges where both endpoints are inside local_mask.

        "either":
            use edges where at least one endpoint is inside local_mask.

    Returns:
        Tensor of local neighboring value jumps.
    """
    assert Z.shape == local_mask.shape, "Z and local_mask must have the same shape"
    assert edge_mode in {"both", "either"}, "edge_mode must be 'both' or 'either'"

    local_mask = local_mask.bool()
    jumps = []

    for axis in range(Z.ndim):
        z0 = Z.narrow(axis, 0, Z.shape[axis] - 1)
        z1 = Z.narrow(axis, 1, Z.shape[axis] - 1)

        m0 = local_mask.narrow(axis, 0, local_mask.shape[axis] - 1)
        m1 = local_mask.narrow(axis, 1, local_mask.shape[axis] - 1)

        if edge_mode == "both":
            edge_mask = m0 & m1
        else:
            edge_mask = m0 | m1

        diff = (z1 - z0).abs()

        if edge_mask.any():
            jumps.append(diff[edge_mask])

    if len(jumps) == 0:
        return torch.empty(0, device=Z.device, dtype=Z.dtype)

    return torch.cat(jumps, dim=0)


def median_level_step(levels: torch.Tensor) -> torch.Tensor:
    """
    Median distance between consecutive scanned levels.
    """
    assert levels.ndim == 1, "levels must be a 1D tensor"

    if len(levels) <= 1:
        return torch.tensor(0.0, device=levels.device, dtype=levels.dtype)

    sorted_levels = torch.sort(levels).values
    steps = torch.diff(sorted_levels).abs()

    return torch.median(steps)


def choose_local_adaptive_level_eps(
    Z: torch.Tensor,
    grid_points_nd: torch.Tensor,
    feature_points: torch.Tensor,
    grid_step: float,
    levels: torch.Tensor | None = None,
    local_radius: float | None = None,
    initial_radius_factor: float = 4.0,
    max_radius_factor: float = 20.0,
    radius_growth: float = 1.5,
    min_jumps: int = 100,
    quantile: float = 1.0,
    multiplier: float = 1.0,
    edge_mode: str = "either",
    min_level_step_factor: float | None = 0.5,
    distance_batch_size: int | None = None,
    eps_min: float = 1e-8,
) -> dict:
    """
    Chooses level_eps using local variation of f near the target feature.

    Main idea:
        level_eps = multiplier * quantile(
            |f(x_neighbor) - f(x)| near feature
        )

    The local neighborhood is defined by distance to feature_points.
    If too few neighboring jumps are found, the neighborhood radius is
    increased automatically.
    """
    assert Z.ndim + 1 == grid_points_nd.ndim, (
        "grid_points_nd must have shape [N1, ..., Nd, dim]"
    )
    assert grid_points_nd.shape[:-1] == Z.shape, (
        "grid_points_nd spatial shape must match Z"
    )
    assert feature_points.ndim == 2, "feature_points must have shape [n_A, dim]"
    assert feature_points.shape[1] == grid_points_nd.shape[-1], (
        "feature_points dimension must match grid dimension"
    )
    assert grid_step > 0, "grid_step must be positive"
    assert 0.0 <= quantile <= 1.0, "quantile must be between 0 and 1"
    assert multiplier > 0, "multiplier must be positive"

    flat_grid = grid_points_nd.reshape(-1, grid_points_nd.shape[-1])

    nearest_to_feature = nearest_distances_to_set(
        flat_grid,
        feature_points,
        batch_size=distance_batch_size,
    )

    if local_radius is None:
        radius = initial_radius_factor * grid_step
    else:
        radius = float(local_radius)

    max_radius = max_radius_factor * grid_step

    best_jumps = None
    used_radius = radius

    while radius <= max_radius:
        local_mask = (nearest_to_feature <= radius).reshape(Z.shape)

        jumps = collect_local_neighbor_value_jumps(
            Z=Z,
            local_mask=local_mask,
            edge_mode=edge_mode,
        )

        if len(jumps) >= min_jumps:
            best_jumps = jumps
            used_radius = radius
            break

        best_jumps = jumps
        used_radius = radius
        radius *= radius_growth

    if best_jumps is None or len(best_jumps) == 0:
        raw_eps = torch.tensor(eps_min, device=Z.device, dtype=Z.dtype)
    else:
        positive_jumps = best_jumps[best_jumps > eps_min]

        if len(positive_jumps) == 0:
            raw_eps = torch.tensor(eps_min, device=Z.device, dtype=Z.dtype)
        else:
            raw_eps = multiplier * torch.quantile(positive_jumps, quantile)

    if levels is not None and min_level_step_factor is not None and len(levels) > 1:
        step = median_level_step(levels).to(device=Z.device, dtype=Z.dtype)
        min_eps_from_levels = min_level_step_factor * step
        raw_eps = torch.maximum(raw_eps, min_eps_from_levels)

    return {
        "level_eps": float(raw_eps.item()),
        "local_radius": float(used_radius),
        "n_local_jumps": 0 if best_jumps is None else int(len(best_jumps)),
        "quantile": quantile,
        "multiplier": multiplier,
        "edge_mode": edge_mode,
        "min_level_step_factor": min_level_step_factor,
    }


# -------------------------
# Criterion 2: structural (Chamfer) criterion
# -------------------------

def level_set_component_points(
    field: FieldSample,
    level_mask_flat: torch.Tensor,
    connectivity: str = "full",
    min_component_size: int = 5,
    component_radius: float | None = None,
):
    """
    Splits {x in field.points : level_mask_flat} into connected components and
    returns the list of their point sets.

    Grid samples use image labeling (``connected_components``), gridless
    samples use radius-graph clustering (``connected_components_pointcloud``)
    with ``component_radius`` (default: 2x median nearest-neighbor spacing of
    the masked points).
    """
    assert level_mask_flat.ndim == 1, "level_mask_flat must be flat [N]"
    assert len(level_mask_flat) == len(field.points), (
        "level_mask_flat must match field.points"
    )

    if field.is_grid:
        level_mask = level_mask_flat.reshape(field.grid_shape)

        components = connected_components(
            level_mask,
            connectivity=connectivity,
            min_size=min_component_size,
        )

        labeled_mask = components["labeled_mask"]
        points_nd = field.points_nd

        component_point_sets = []

        for component_id in components["component_ids"]:
            component_mask = torch.from_numpy(labeled_mask == component_id).to(
                device=field.points.device
            )
            component_point_sets.append(points_nd[component_mask])

        return component_point_sets

    masked_points = field.points[level_mask_flat]

    if len(masked_points) == 0:
        return []

    if component_radius is None:
        spacing = float(feature_nearest_neighbor_spacing(masked_points).item())
        component_radius = 2.0 * spacing

    component_radius = max(float(component_radius), 1e-8)

    components = connected_components_pointcloud(
        masked_points,
        radius=component_radius,
        min_size=min_component_size,
    )

    labels = components["labels"].to(device=masked_points.device)

    return [
        masked_points[labels == component_id]
        for component_id in components["component_ids"]
    ]


def best_component_chamfer_score(
    component_point_sets,
    feature_points: torch.Tensor,
    distance_batch_size: int | None = None,
):
    """
    Max-symmetrized Chamfer score of the best component against the feature.

    Returns (best_score, best_component_points); (inf, None) when empty.
    """
    best_score = float("inf")
    best_points = None

    for K_points in component_point_sets:
        if len(K_points) == 0:
            continue

        distances = max_chamfer_distance(
            K_points,
            feature_points,
            batch_size=distance_batch_size,
        )

        score = float(distances["score"].item())

        if score < best_score:
            best_score = score
            best_points = K_points

    return best_score, best_points


def structural_criterion(
    function,
    feature_points: torch.Tensor,
    bounds,
    resolution: int | list[int] | tuple[int, ...] = 100,
    n_levels: int = 101,
    levels: torch.Tensor | list[float] | None = None,
    activation: str | None = None,
    level_bounds: tuple[float, float] | None = None,
    level_eps: float | str = 0.03,
    connectivity: str = "full",
    min_component_size: int = 5,
    eval_batch_size: int = 8192,
    distance_batch_size: int | None = None,
    device: str = "cpu",
    threshold: float = 0.1,
    return_best_component_points: bool = False,
    diameter_method: str = "exact",  # or 'bbox'

    # pre-evaluated field (shared across criteria) / gridless samples
    field: FieldSample | None = None,
    component_radius: float | None = None,

    # adaptive level_eps parameters
    adaptive_eps_quantile: float | None = None,
    adaptive_eps_multiplier: float = 1.0,
    adaptive_eps_local_radius: float | None = None,
    adaptive_eps_initial_radius_factor: float = 4.0,
    adaptive_eps_max_radius_factor: float = 20.0,
    adaptive_eps_radius_growth: float = 1.5,
    adaptive_eps_min_jumps: int = 100,
    adaptive_eps_edge_mode: str = "either",
    adaptive_eps_min_level_step_factor: float | None = 0.5,
):
    """
    Computes simplified structural level-set criterion.

    Scans levels c, finds connected components of

        {x : |f(x) - c| <= level_eps},

    and returns the best geometric matching score with feature_points.

    The function is evaluated on a regular grid over ``bounds`` by default.
    Passing a pre-computed ``field`` (see ``analysis.fields.FieldSample``)
    skips the evaluation — this is how ``evaluate_neurons_on_feature`` shares
    one field across all criteria, and how gridless samples (gradient-sampled
    clouds, d >= 3) are scored: for a non-grid field, connected components are
    extracted by radius-graph clustering with ``component_radius`` (default:
    2x median nearest-neighbor spacing) and adaptive level_eps is unavailable.

    If level_eps == "adaptive", the epsilon is derived from the local spread of
    f-values near the feature (see choose_local_adaptive_level_eps):

        adaptive_eps_quantile=None (default) -> tries quantiles 1.0 and 0.95,
        returns the best result among both variants.

        adaptive_eps_quantile=q -> uses only quantile q.

    Output:
        score
        score_normalized_by_diameter
        score_normalized_by_grid_step
        is_structural_witness

    Additionally:
        best_level_eps
        best_adaptive_eps_quantile, if level_eps == "adaptive"

    Optionally:
        best_component_points
    """
    assert len(bounds) > 0, "bounds must be non-empty"
    assert min_component_size >= 1, "min_component_size must be >= 1"
    assert threshold > 0, "threshold must be positive"

    if isinstance(level_eps, str):
        assert level_eps == "adaptive", "level_eps must be positive float or 'adaptive'"
    else:
        assert level_eps > 0, "level_eps must be positive"

    feature_points = feature_points.to(device)
    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    assert feature_points.ndim == 2, "feature_points must have shape [n_A, dim]"
    assert len(feature_points) > 0, "feature_points must be non-empty"
    assert feature_points.shape[1] == len(bounds), (
        "feature_points dimension must match len(bounds)"
    )

    dtype = feature_points.dtype

    if field is None:
        field = evaluate_field_on_grid(
            function,
            bounds=bounds,
            resolution=resolution,
            batch_size=eval_batch_size,
            device=device,
            dtype=dtype,
        )

    assert field.dim == len(bounds), "field dimension must match len(bounds)"

    values_flat = field.values

    if level_eps == "adaptive":
        assert field.is_grid, (
            "level_eps='adaptive' needs a grid field (local neighbor jumps); "
            "pass an explicit level_eps for gridless samples"
        )

    if levels is None:
        candidate_levels = make_candidate_levels(
            n_levels=n_levels,
            activation=activation,
            level_bounds=level_bounds,
            observed_values=values_flat,
            device=device,
            dtype=dtype,
        )
    else:
        candidate_levels = torch.as_tensor(
            levels,
            device=device,
            dtype=dtype,
        )

    if field.is_grid:
        grid_step = field.grid_step
        if grid_step is None:
            grid_step = grid_step_from_bounds_resolution(bounds, resolution)
        grid_step = max(float(grid_step), 1e-12)
    else:
        grid_step = None

    best_score = float("inf")
    best_component_points = None
    best_level_eps = None
    best_adaptive_eps_quantile = None

    if level_eps == "adaptive":
        if adaptive_eps_quantile is None:
            quantiles_to_try = [1.0, 0.95]
        else:
            quantiles_to_try = [float(adaptive_eps_quantile)]

        level_eps_variants = []

        for quantile in quantiles_to_try:
            adaptive_eps_info = choose_local_adaptive_level_eps(
                Z=field.values_nd,
                grid_points_nd=field.points_nd,
                feature_points=feature_points,
                grid_step=grid_step,
                levels=candidate_levels,
                local_radius=adaptive_eps_local_radius,
                initial_radius_factor=adaptive_eps_initial_radius_factor,
                max_radius_factor=adaptive_eps_max_radius_factor,
                radius_growth=adaptive_eps_radius_growth,
                min_jumps=adaptive_eps_min_jumps,
                quantile=quantile,
                multiplier=adaptive_eps_multiplier,
                edge_mode=adaptive_eps_edge_mode,
                min_level_step_factor=adaptive_eps_min_level_step_factor,
                distance_batch_size=distance_batch_size,
            )

            level_eps_variants.append(
                {
                    "level_eps_value": float(adaptive_eps_info["level_eps"]),
                    "adaptive_eps_quantile": quantile,
                }
            )

    else:
        level_eps_variants = [
            {
                "level_eps_value": float(level_eps),
                "adaptive_eps_quantile": None,
            }
        ]

    for eps_variant in level_eps_variants:
        level_eps_value = eps_variant["level_eps_value"]
        cur_adaptive_eps_quantile = eps_variant["adaptive_eps_quantile"]

        for level in candidate_levels:
            level_mask_flat = torch.abs(values_flat - level) <= level_eps_value

            component_point_sets = level_set_component_points(
                field,
                level_mask_flat,
                connectivity=connectivity,
                min_component_size=min_component_size,
                component_radius=component_radius,
            )

            level_best_score, level_best_points = best_component_chamfer_score(
                component_point_sets,
                feature_points,
                distance_batch_size=distance_batch_size,
            )

            if level_best_score < best_score:
                best_score = level_best_score
                best_level_eps = level_eps_value
                best_adaptive_eps_quantile = cur_adaptive_eps_quantile

                if return_best_component_points:
                    best_component_points = level_best_points.detach().cpu()

    diameter = feature_diameter(feature_points, method=diameter_method)
    diameter = max(float(diameter.item()), 1e-12)

    score_normalized_by_diameter = best_score / diameter

    if grid_step is None:
        score_normalized_by_grid_step = float("nan")
    else:
        score_normalized_by_grid_step = best_score / grid_step

    result = {
        "score": best_score,
        "score_normalized_by_diameter": score_normalized_by_diameter,
        "score_normalized_by_grid_step": score_normalized_by_grid_step,
        "is_structural_witness": score_normalized_by_diameter <= threshold,
        "best_level_eps": best_level_eps,
    }

    if level_eps == "adaptive":
        result["best_adaptive_eps_quantile"] = best_adaptive_eps_quantile

    if return_best_component_points:
        result["best_component_points"] = best_component_points

    return result


def structural_criterion_sampled(
    function,
    feature_points: torch.Tensor,
    bounds,
    n_levels: int = 21,
    levels: torch.Tensor | list[float] | None = None,
    activation: str | None = None,
    level_bounds: tuple[float, float] | None = None,
    sampler_method: str = "optimization",
    sampler_kwargs: dict | None = None,
    n_points_per_level: int = 1024,
    min_component_size: int = 5,
    component_radius: float | None = None,
    distance_batch_size: int | None = None,
    n_range_probe_points: int = 4096,
    device: str = "cpu",
    threshold: float = 0.1,
    return_best_component_points: bool = False,
    diameter_method: str = "exact",
):
    """
    Sampler-backed structural criterion (experimental; the d >= 3 route).

    Instead of a dense grid, each candidate level c is approximated by a
    gradient-based sampler from ``analysis.level_set_sampling`` (minimization
    of |f(x) - c|^2 from many random starts). Connected components of the
    sampled cloud come from radius-graph clustering, and are scored with the
    same max-symmetrized Chamfer distance as the grid criterion.

    Coverage caveat (PLAN.md): the sampler is pulled to the nearest sheet of
    the level surface, so under-sampling masquerades as "witness not found".
    ``coverage`` in the output reports the number of accepted sample points per
    level — treat low-coverage levels as unmeasured, not as negative evidence.

    When ``levels``/``activation``/``level_bounds`` are all None, the level
    range is estimated by probing f on uniform random points inside bounds.
    """
    from analysis.level_set_sampling import LevelSetSampler

    assert len(bounds) > 0, "bounds must be non-empty"
    assert min_component_size >= 1, "min_component_size must be >= 1"
    assert threshold > 0, "threshold must be positive"

    feature_points = feature_points.to(device)
    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    assert feature_points.ndim == 2, "feature_points must have shape [n_A, dim]"
    assert len(feature_points) > 0, "feature_points must be non-empty"
    assert feature_points.shape[1] == len(bounds), (
        "feature_points dimension must match len(bounds)"
    )

    dtype = feature_points.dtype
    sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)

    observed_values = None

    if levels is None and activation is None and level_bounds is None:
        lows = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
        highs = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)

        probe_points = lows + (highs - lows) * torch.rand(
            n_range_probe_points, len(bounds), device=device, dtype=dtype
        )

        observed_values = evaluate_scalar_function_on_points(
            function=function,
            points=probe_points,
        )

    if levels is None:
        candidate_levels = make_candidate_levels(
            n_levels=n_levels,
            activation=activation,
            level_bounds=level_bounds,
            observed_values=observed_values,
            device=device,
            dtype=dtype,
        )
    else:
        candidate_levels = torch.as_tensor(levels, device=device, dtype=dtype)

    sampler = LevelSetSampler(
        function=function,
        method=sampler_method,
        device=device,
    )

    best_score = float("inf")
    best_level = None
    best_component_points = None
    coverage = {}

    for level in candidate_levels:
        level_value = float(level.item())

        level_points = sampler.sample(
            level=level_value,
            n_points=n_points_per_level,
            bounds=bounds,
            **sampler_kwargs,
        )

        coverage[level_value] = int(len(level_points))

        if len(level_points) < min_component_size:
            continue

        cloud_field = FieldSample(
            points=level_points.to(device=device, dtype=dtype),
            values=torch.full(
                (len(level_points),), level_value, device=device, dtype=dtype
            ),
        )

        component_point_sets = level_set_component_points(
            cloud_field,
            torch.ones(len(level_points), dtype=torch.bool, device=device),
            min_component_size=min_component_size,
            component_radius=component_radius,
        )

        level_best_score, level_best_points = best_component_chamfer_score(
            component_point_sets,
            feature_points,
            distance_batch_size=distance_batch_size,
        )

        if level_best_score < best_score:
            best_score = level_best_score
            best_level = level_value

            if return_best_component_points:
                best_component_points = level_best_points.detach().cpu()

    diameter = feature_diameter(feature_points, method=diameter_method)
    diameter = max(float(diameter.item()), 1e-12)

    score_normalized_by_diameter = best_score / diameter

    result = {
        "score": best_score,
        "score_normalized_by_diameter": score_normalized_by_diameter,
        "is_structural_witness": score_normalized_by_diameter <= threshold,
        "best_level": best_level,
        "n_levels_scanned": int(len(candidate_levels)),
        "n_levels_covered": int(
            sum(1 for n in coverage.values() if n >= min_component_size)
        ),
        "min_level_coverage": min(coverage.values()) if coverage else 0,
        "coverage": coverage,
    }

    if return_best_component_points:
        result["best_component_points"] = best_component_points

    return result


def sampled_structural_criterion(
    function,
    feature_points: torch.Tensor,
    bounds,
    resolution=None,
    device: str = "cpu",
    field: FieldSample | None = None,

    n_levels: int = 11,
    levels: torch.Tensor | list[float] | None = None,
    activation: str | None = None,
    level_bounds: tuple[float, float] | None = None,
    sampler_method: str = "optimization",
    sampler_kwargs: dict | None = None,
    n_points_per_level: int = 2048,
    min_component_size: int = 5,
    component_radius: float | None = None,
    component_radius_feature_factor: float = 3.0,
    threshold: float = 0.1,
    **kwargs,
):
    """
    ``extra_criteria``-protocol adapter over ``structural_criterion_sampled``
    — the structural criterion for d >= 3, where the dense grid is dead
    (PLAN.md exp. 9: gradient level-set sampling instead of a grid).

    ``resolution`` and ``field`` are accepted for protocol compatibility and
    ignored: each candidate level is sampled afresh by the gradient sampler.
    When ``levels``/``activation``/``level_bounds`` are all None the level
    range comes from random probing of ``function`` inside ``bounds``.

    ``component_radius`` default differs from the raw
    ``structural_criterion_sampled`` rule on purpose: gradient-sampled clouds
    are CLUMPED (many restarts converge to nearby points), which collapses
    the cloud's median nearest-neighbor spacing and shatters level-set sheets
    into fragments. The default here clusters at the FEATURE's resolution
    instead: ``component_radius_feature_factor`` x median nearest-neighbor
    spacing of ``feature_points``.

    Returns flat numerics for the CSV (columns "sampled_structural_*"):
        score, score_normalized_by_diameter, is_witness, best_level,
        n_levels_scanned, n_levels_covered, min_level_coverage.

    Coverage caveat (PLAN.md): treat low-coverage levels as unmeasured, not
    as negative evidence — check n_levels_covered before trusting a negative.
    """
    if component_radius is None:
        feature_spacing = feature_nearest_neighbor_spacing(
            feature_points.to(device).float()
        )
        component_radius = component_radius_feature_factor * float(
            feature_spacing.item()
        )

    result = structural_criterion_sampled(
        function=function,
        feature_points=feature_points,
        bounds=bounds,
        n_levels=n_levels,
        levels=levels,
        activation=activation,
        level_bounds=level_bounds,
        sampler_method=sampler_method,
        sampler_kwargs=sampler_kwargs,
        n_points_per_level=n_points_per_level,
        min_component_size=min_component_size,
        component_radius=component_radius,
        device=device,
        threshold=threshold,
    )

    return {
        "score": result["score"],
        "score_normalized_by_diameter": result["score_normalized_by_diameter"],
        "is_witness": result["is_structural_witness"],
        "best_level": result["best_level"],
        "n_levels_scanned": result["n_levels_scanned"],
        "n_levels_covered": result["n_levels_covered"],
        "min_level_coverage": result["min_level_coverage"],
    }


# -------------------------
# Criterion 1: tubular (Wasserstein) criterion
# -------------------------

def auc_from_scores(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Computes AUC for separating pos_scores from neg_scores.
    AUC close to 1 means pos_scores are usually larger than neg_scores.
    AUC close to 0.5 means no separation.

    Note: ranks are not tie-averaged, so heavily tied inputs (e.g. saturated
    tanh values) get an order-dependent AUC. Fine as a rough diagnostic; use a
    tie-aware implementation before relying on it quantitatively.
    """
    assert pos_scores.ndim == 1
    assert neg_scores.ndim == 1

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([
        torch.ones(n_pos, device=scores.device),
        torch.zeros(n_neg, device=scores.device),
    ])

    order = torch.argsort(scores)
    sorted_labels = labels[order]

    ranks = torch.arange(
        1,
        len(scores) + 1,
        device=scores.device,
        dtype=scores.dtype,
    )

    sum_pos_ranks = ranks[sorted_labels == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return float(auc.item())


def resolve_tubular_radii(
    feature_points: torch.Tensor,
    bounds,
    resolution,
    delta: float | str = "auto",
    Delta: float | str = "auto",
    delta_factor: float = 2.0,
    Delta_factor: float = 2.0,
    spacing_batch_size: int | None = None,
) -> dict:
    """
    Resolves delta and Delta for tubular neighborhoods.

    If delta="auto":
        delta = delta_factor * max(grid_step, feature_spacing)

    If Delta="auto":
        Delta = Delta_factor * delta

    resolution=None means "no grid" (gridless field sample): grid_step is
    treated as 0, so auto radii fall back to the feature spacing scale.
    """
    assert delta == "auto" or isinstance(delta, (int, float)), (
        "delta must be a positive float or 'auto'"
    )
    assert Delta == "auto" or isinstance(Delta, (int, float)), (
        "Delta must be a positive float or 'auto'"
    )

    assert delta_factor > 0, "delta_factor must be positive"
    assert Delta_factor > 1, "Delta_factor must be > 1"

    if resolution is None:
        grid_step_value = 0.0
    else:
        grid_step_value = float(grid_step_from_bounds_resolution(bounds, resolution))

    feature_spacing = feature_nearest_neighbor_spacing(
        feature_points,
        batch_size=spacing_batch_size,
    )
    feature_spacing_value = float(feature_spacing.item())

    base_scale = max(grid_step_value, feature_spacing_value)

    if delta == "auto":
        delta_value = delta_factor * base_scale
    else:
        delta_value = float(delta)

    if Delta == "auto":
        Delta_value = Delta_factor * delta_value
    else:
        Delta_value = float(Delta)

    assert delta_value > 0, "resolved delta must be positive"
    assert Delta_value > delta_value, "resolved Delta must be larger than delta"

    return {
        "delta": delta_value,
        "Delta": Delta_value,
        "grid_step": grid_step_value,
        "feature_spacing": feature_spacing_value,
        "base_scale": base_scale,
        "delta_was_auto": delta == "auto",
        "Delta_was_auto": Delta == "auto",
        "delta_factor": delta_factor,
        "Delta_factor": Delta_factor,
    }


def normalized_wasserstein_1_distance(
    values_1: torch.Tensor,
    values_2: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Computes Wasserstein-1 distance between two 1D empirical distributions,
    normalized by robust local value range q90 - q10 of the pooled values.
    """
    assert values_1.ndim == 1, "values_1 must be 1D"
    assert values_2.ndim == 1, "values_2 must be 1D"

    if len(values_1) == 0 or len(values_2) == 0:
        return float("nan")

    values_1_np = values_1.detach().cpu().numpy()
    values_2_np = values_2.detach().cpu().numpy()

    w1 = wasserstein_distance(values_1_np, values_2_np)

    pooled = torch.cat([values_1, values_2], dim=0)
    local_range = torch.quantile(pooled, 0.90) - torch.quantile(pooled, 0.10)
    local_range = max(float(local_range.item()), eps)

    return float(w1 / local_range)


def tubular_criterion(
    function,
    feature_points: torch.Tensor,
    bounds,
    delta: float | str = "auto",
    Delta: float | str = "auto",
    resolution: int | list[int] | tuple[int, ...] = 100,
    delta_factor: float = 2.0,
    Delta_factor: float = 2.0,
    eval_batch_size: int = 8192,
    distance_batch_size: int | None = None,
    spacing_batch_size: int | None = None,
    device: str = "cpu",
    return_points: bool = False,
    return_metadata: bool = False,
    compute_auc: bool = False,
    field: FieldSample | None = None,
):
    """
    Computes simplified tubular value criterion.

    Builds two regions around feature A:

        T = {x : d(x, A) < delta}
        S = {x : delta < d(x, A) < Delta}

    Then compares distributions f|_T and f|_S.

    The function is evaluated on a regular grid over ``bounds`` by default;
    passing a pre-computed ``field`` (grid or gridless) skips the evaluation.

    Main output:
        tubular_score == normalized_wasserstein_1

    Optional output if compute_auc=True:
        auc_best_direction
    """
    assert len(bounds) > 0, "bounds must be non-empty"

    feature_points = feature_points.to(device)

    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    assert feature_points.ndim == 2, "feature_points must have shape [n_A, dim]"
    assert len(feature_points) > 0, "feature_points must be non-empty"
    assert feature_points.shape[1] == len(bounds), (
        "feature_points dimension must match len(bounds)"
    )

    dtype = feature_points.dtype

    if field is None:
        field = evaluate_field_on_grid(
            function,
            bounds=bounds,
            resolution=resolution,
            batch_size=eval_batch_size,
            device=device,
            dtype=dtype,
        )

    assert field.dim == len(bounds), "field dimension must match len(bounds)"

    radii_info = resolve_tubular_radii(
        feature_points=feature_points,
        bounds=bounds,
        resolution=resolution if field.is_grid else None,
        delta=delta,
        Delta=Delta,
        delta_factor=delta_factor,
        Delta_factor=Delta_factor,
        spacing_batch_size=spacing_batch_size,
    )

    delta_value = radii_info["delta"]
    Delta_value = radii_info["Delta"]

    grid_points = field.points
    grid_shape = field.grid_shape
    values = field.values

    distances_to_feature = nearest_distances_to_set(
        grid_points,
        feature_points,
        batch_size=distance_batch_size,
    )

    tube_mask = distances_to_feature < delta_value
    shell_mask = (
        (distances_to_feature > delta_value)
        & (distances_to_feature < Delta_value)
    )

    tube_values = values[tube_mask]
    shell_values = values[shell_mask]

    result = {}

    if len(tube_values) == 0 or len(shell_values) == 0:
        result["tubular_score"] = float("nan")

        if compute_auc:
            result["auc_best_direction"] = float("nan")
    else:
        result["tubular_score"] = normalized_wasserstein_1_distance(
            tube_values,
            shell_values,
        )

        if compute_auc:
            auc_tube_high = auc_from_scores(tube_values, shell_values)
            result["auc_best_direction"] = float(
                max(auc_tube_high, 1.0 - auc_tube_high)
            )

    if return_points:
        result["tube_points"] = grid_points[tube_mask].detach().cpu()
        result["shell_points"] = grid_points[shell_mask].detach().cpu()

    if return_metadata:
        result["delta"] = delta_value
        result["Delta"] = Delta_value
        result["radii_info"] = radii_info
        result["grid_shape"] = grid_shape
        result["n_tube_points"] = int(tube_mask.sum().item())
        result["n_shell_points"] = int(shell_mask.sum().item())

    return result


# -------------------------
# Per-neuron evaluation loop
# -------------------------

def field_regime_over_feature(function, feature_points, device: str = "cpu") -> dict:
    """How active this neuron's field is ON the feature: mean |grad f|, std f.

    Not a criterion, and NOT a covariate to match a control against. It is an
    ADMISSIBILITY CHECK on a single feature: is this a place where the criterion
    could fire at all? A feature over which f is constant to numerical precision
    admits no level set whatever the network encodes, so it contributes a
    guaranteed non-detection. In a set of A.1 nulls such a draw silently lowers
    the floor and makes the target look more distinguished than it is — the same
    failure as a null sitting half outside the evaluation box, which
    ``null_placement.placement_report`` catches with ``in_bounds``.

    Read it as a per-draw admissibility flag, then, and not as a quantity the
    nulls should be matched to the target on. Matching a control to its target
    on covariate after covariate terminates in the target itself, at which point
    there is no control left; what a null set owes the target is its SHAPE and
    a representative coverage of the problem, nothing more.

    A caveat on causal reading. Across A.1 placements this quantity is
    collinear with position, so a correlation between it and the score says
    nothing about which of the two acts. The one controlled measurement is in
    ``analysis/calibration.py``: a strictly monotone rescaling of a neuron
    preserves every level set exactly yet moves the structural score by ~0.07
    and flips the verdict in saturation. That is a non-invariance of the
    CRITERION (see RESEARCH_PROGRAMME section 8), not a property of where a
    null was placed.

    Cheap: one backward pass over the feature's points per neuron-feature,
    against ~1 s for the criteria themselves.
    """
    P = feature_points.detach().to(device)
    if not torch.is_floating_point(P):
        P = P.float()

    P = P.clone().requires_grad_(True)

    try:
        values = function(P)
        grad = torch.autograd.grad(values.sum(), P)[0]
        grad_norm = float(grad.norm(dim=1).mean().item())
    except RuntimeError:
        # a non-differentiable or detached function: report the value spread
        # only rather than failing the whole evaluation
        with torch.no_grad():
            values = function(P.detach())
        grad_norm = float("nan")

    v = values.detach()

    return {
        "field_grad_norm_on_feature": grad_norm,
        "field_value_std_on_feature": float(v.std().item()),
        "field_value_abs_mean_on_feature": float(v.abs().mean().item()),
    }


def evaluate_neurons_on_feature(
    model,
    feature_points: torch.Tensor,
    bounds,
    resolution: int | list[int] | tuple[int, ...] = 100,
    layer_ids: list[int] | None = None,
    coord_ids_by_layer: dict[int, list[int]] | None = None,
    kinds: tuple[str, ...] = ("post",),
    device: str = "cpu",

    structural_kwargs: dict | None = None,
    value_kwargs: dict | None = None,
    extra_criteria: dict | None = None,

    run_structural: bool = True,
    run_tubular: bool = True,
    eval_batch_size: int = 8192,

    field_source: str = "grid",
    n_field_points: int = 20000,
    field_seed: int = 0,

    return_full_result_structural: bool = False,
    return_full_result_tubular: bool = False,
    verbose: bool = True,
):
    """
    Computes structural and tubular criteria for selected hidden neurons.

    For each selected neuron:
        1. builds ScalarProjection of one hidden coordinate;
        2. evaluates the neuron once on the shared point sample (a shared
           FieldSample);
        3. computes structural_criterion (unless run_structural=False);
        4. computes tubular_criterion (unless run_tubular=False);
        5. computes each criterion in extra_criteria, if provided.

    field_source:
        Where the shared FieldSample is evaluated — the d >= 3 switch:

        "grid" (default) — the regular grid over ``bounds`` with
            ``resolution`` points per axis; exponentially dead for d >= 4.
        "random" — ``n_field_points`` uniform points inside ``bounds``
            (seeded by ``field_seed``; the SAME cloud is shared by every
            neuron, like a grid is). Gridless criteria (tubular, homological,
            sampled_structural) work unchanged; ``run_structural`` requires
            an explicit ``level_eps`` (the adaptive rule needs a grid).

    extra_criteria:
        Optional dict {criterion_name: callable}. Each callable is invoked as

            fn(function=neuron, feature_points=..., bounds=...,
               resolution=..., device=..., field=<shared FieldSample>)

        and must return a dict; its int/float/bool entries are merged into the
        row as "{criterion_name}_{key}". This is the extension point for
        Criterion 3 (extremum-anchored) and Criterion 4 (homological).
        Implementations should accept **kwargs so that future keyword
        additions do not break them; ``field`` holds the already-evaluated
        neuron values and may be used instead of re-evaluating the function.

    Main row fields:
        layer_id
        kind
        coord_id

        structural_score
        structural_score_normalized_by_diameter
        structural_score_normalized_by_grid_step
        is_structural_witness

        tubular_score

    Optional row fields:
        auc_best_direction, if tubular_criterion was called with compute_auc=True.

        delta, Delta, n_tube_points, n_shell_points, if tubular_criterion was
        called with return_metadata=True.

        structural_result, if return_full_result_structural=True.
        value_result, if return_full_result_tubular=True.

        "{name}_{key}" for each extra criterion output.
    """
    model.to(device)
    model.eval()

    feature_points = feature_points.to(device)

    if layer_ids is None:
        assert hasattr(model, "n_layers"), (
            "model must have attribute n_layers if layer_ids is None"
        )
        layer_ids = list(range(model.n_layers))

    assert all(kind in {"pre", "post"} for kind in kinds), (
        "all kinds must be either 'pre' or 'post'"
    )

    structural_kwargs = {} if structural_kwargs is None else dict(structural_kwargs)
    value_kwargs = {} if value_kwargs is None else dict(value_kwargs)
    extra_criteria = {} if extra_criteria is None else dict(extra_criteria)

    assert field_source in {"grid", "random"}, (
        "field_source must be 'grid' or 'random'"
    )

    shared_field_points = None

    if field_source == "random":
        generator = torch.Generator()
        generator.manual_seed(field_seed)

        lows = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        highs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)

        uniform = torch.rand(n_field_points, len(bounds), generator=generator)
        shared_field_points = (lows + (highs - lows) * uniform).to(device)

    results = []

    total = 0
    layer_to_coord_ids = {}

    for layer_id in layer_ids:
        hidden_dim = get_hidden_dim(model, layer_id)

        if coord_ids_by_layer is None or layer_id not in coord_ids_by_layer:
            coord_ids = list(range(hidden_dim))
        else:
            coord_ids = list(coord_ids_by_layer[layer_id])

        for coord_id in coord_ids:
            assert 0 <= coord_id < hidden_dim, (
                f"Invalid coord_id={coord_id} for layer_id={layer_id}. "
                f"Expected 0 <= coord_id < {hidden_dim}"
            )

        layer_to_coord_ids[layer_id] = coord_ids
        total += len(coord_ids) * len(kinds)

    counter = 0

    for layer_id in layer_ids:
        coord_ids = layer_to_coord_ids[layer_id]

        for kind in kinds:
            vector_function = TruncatedMLPFunction(
                model=model,
                layer_id=layer_id,
                kind=kind,
            )

            inferred_activation = infer_level_activation_for_neuron(
                model=model,
                layer_id=layer_id,
                kind=kind,
            )

            for coord_id in coord_ids:
                counter += 1

                if verbose:
                    print(
                        f"[{counter}/{total}] "
                        f"layer={layer_id}, kind={kind}, coord={coord_id}"
                    )

                neuron = ScalarProjection(
                    vector_function=vector_function,
                    mode="coord",
                    coord_id=coord_id,
                )

                # One evaluation per neuron, shared by all criteria.
                field_dtype = (
                    feature_points.dtype
                    if torch.is_floating_point(feature_points)
                    else torch.float32
                )

                if field_source == "grid":
                    field = evaluate_field_on_grid(
                        neuron,
                        bounds=bounds,
                        resolution=resolution,
                        batch_size=eval_batch_size,
                        device=device,
                        dtype=field_dtype,
                    )
                else:
                    field = field_from_points(
                        neuron,
                        shared_field_points.to(dtype=field_dtype),
                        batch_size=eval_batch_size,
                        bounds=bounds,
                    )

                row = {
                    "layer_id": layer_id,
                    "kind": kind,
                    "coord_id": coord_id,
                    **field_regime_over_feature(
                        neuron, feature_points, device=device
                    ),
                }

                structural_result = None
                value_result = None

                if run_structural:
                    cur_structural_kwargs = dict(structural_kwargs)

                    if (
                        "activation" not in cur_structural_kwargs
                        and "levels" not in cur_structural_kwargs
                        and "level_bounds" not in cur_structural_kwargs
                    ):
                        cur_structural_kwargs["activation"] = inferred_activation

                    structural_result = structural_criterion(
                        function=neuron,
                        feature_points=feature_points,
                        bounds=bounds,
                        resolution=resolution,
                        device=device,
                        field=field,
                        **cur_structural_kwargs,
                    )

                    row.update(
                        {
                            "structural_score": structural_result["score"],
                            "structural_score_normalized_by_diameter": structural_result[
                                "score_normalized_by_diameter"
                            ],
                            "structural_score_normalized_by_grid_step": structural_result[
                                "score_normalized_by_grid_step"
                            ],
                            "is_structural_witness": structural_result[
                                "is_structural_witness"
                            ],
                        }
                    )

                if run_tubular:
                    value_result = tubular_criterion(
                        function=neuron,
                        feature_points=feature_points,
                        bounds=bounds,
                        resolution=resolution,
                        device=device,
                        field=field,
                        **value_kwargs,
                    )

                    row["tubular_score"] = value_result["tubular_score"]

                    for optional_key in (
                        "auc_best_direction",
                        "delta",
                        "Delta",
                        "n_tube_points",
                        "n_shell_points",
                    ):
                        if optional_key in value_result:
                            row[optional_key] = value_result[optional_key]

                for criterion_name, criterion_fn in extra_criteria.items():
                    criterion_result = criterion_fn(
                        function=neuron,
                        feature_points=feature_points,
                        bounds=bounds,
                        resolution=resolution,
                        device=device,
                        field=field,
                    )

                    assert isinstance(criterion_result, dict), (
                        f"extra criterion '{criterion_name}' must return a dict"
                    )

                    for key, value in criterion_result.items():
                        if isinstance(value, (int, float, bool)):
                            row[f"{criterion_name}_{key}"] = value

                if return_full_result_structural and structural_result is not None:
                    row["structural_result"] = structural_result

                if return_full_result_tubular and value_result is not None:
                    row["value_result"] = value_result

                results.append(row)

    return results
