"""Generic geometric utilities shared by all feature-detection criteria.

Everything here is criterion-agnostic: point-cloud distances, grid
construction, scalar-field evaluation, and connected components of binary
masks. New criteria (extremum-anchored, homological) should import from this
module rather than from ``analysis.criterions``.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree


# -------------------------
# Distances between point clouds
# -------------------------

def mean_set_to_set_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    Computes mean_{x in X} d(x, Y), where d(x, Y) = min_{y in Y} ||x-y||_2.

    X: [n_X, dim]
    Y: [n_Y, dim]
    """
    return nearest_distances_to_set(X, Y, batch_size=batch_size).mean()


def max_chamfer_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int | None = None,
) -> dict:
    """
    Computes max(
        mean_{x in X} d(x, Y),
        mean_{y in Y} d(y, X)
    ).

    Lower is better.
    """
    if batch_size is None:
        # One cdist serves both directions.
        _validate_point_clouds(X, Y)
        distances = torch.cdist(X, Y)
        X_to_Y = distances.min(dim=1).values.mean()
        Y_to_X = distances.min(dim=0).values.mean()
    else:
        X_to_Y = mean_set_to_set_distance(X, Y, batch_size=batch_size)
        Y_to_X = mean_set_to_set_distance(Y, X, batch_size=batch_size)

    score = torch.maximum(X_to_Y, Y_to_X)

    return {
        "score": score,
        "X_to_Y": X_to_Y,
        "Y_to_X": Y_to_X,
    }


def _validate_point_clouds(X: torch.Tensor, Y: torch.Tensor):
    assert X.ndim == 2, "X must have shape [n_X, dim]"
    assert Y.ndim == 2, "Y must have shape [n_Y, dim]"
    assert X.shape[1] == Y.shape[1], "X and Y must have the same ambient dimension"
    assert len(X) > 0, "X must be non-empty"
    assert len(Y) > 0, "Y must be non-empty"


def nearest_distances_to_set(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    For every x in X computes d(x, Y) = min_{y in Y} ||x - y||_2.

    X: [n_X, dim]
    Y: [n_Y, dim]

    Returns:
        distances: [n_X]
    """
    _validate_point_clouds(X, Y)

    if batch_size is None:
        distances = torch.cdist(X, Y)
        return distances.min(dim=1).values

    chunks = []

    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        distances = torch.cdist(X_batch, Y)
        chunks.append(distances.min(dim=1).values)

    return torch.cat(chunks, dim=0)


def feature_diameter(
    feature_points: torch.Tensor,
    method: str = "bbox",
) -> torch.Tensor:
    """
    Estimates diameter / scale of the feature.

    method="bbox":
        diagonal of bounding box, cheap and stable.

    method="exact":
        max pairwise distance, more expensive.
    """
    assert feature_points.ndim == 2, "feature_points must have shape [n_points, dim]"
    assert len(feature_points) > 0, "feature_points must be non-empty"

    if method == "bbox":
        mins = feature_points.min(dim=0).values
        maxs = feature_points.max(dim=0).values
        return torch.linalg.norm(maxs - mins)

    if method == "exact":
        distances = torch.cdist(feature_points, feature_points)
        return distances.max()

    raise ValueError("method must be 'bbox' or 'exact'")


def feature_nearest_neighbor_spacing(
    feature_points: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    Estimates feature discretization scale as median nearest-neighbor distance
    inside feature_points.

    feature_points: [n_A, dim]
    """
    assert feature_points.ndim == 2, "feature_points must have shape [n_A, dim]"
    assert len(feature_points) > 0, "feature_points must be non-empty"

    n = len(feature_points)

    if n == 1:
        return torch.tensor(
            0.0,
            device=feature_points.device,
            dtype=feature_points.dtype,
        )

    if batch_size is None:
        distances = torch.cdist(feature_points, feature_points)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        return nearest.median()

    nearest_chunks = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        X_batch = feature_points[start:end]
        distances = torch.cdist(X_batch, feature_points)

        row_ids = torch.arange(end - start, device=feature_points.device)
        col_ids = torch.arange(start, end, device=feature_points.device)
        distances[row_ids, col_ids] = float("inf")

        nearest_chunks.append(distances.min(dim=1).values)

    nearest = torch.cat(nearest_chunks, dim=0)
    return nearest.median()


# -------------------------
# Connected components
# -------------------------

def _to_numpy_bool_mask(binary_mask):
    if torch.is_tensor(binary_mask):
        binary_mask = binary_mask.detach().cpu().numpy()

    binary_mask = np.asarray(binary_mask)
    assert binary_mask.ndim >= 1, "binary_mask must have at least one dimension"

    return binary_mask.astype(bool)


def make_connectivity_structure(ndim: int, connectivity: str = "full"):
    """
    connectivity="full":
        2D -> 8-connectivity
        3D -> 26-connectivity
        nD -> full 3^n neighborhood

    connectivity="face":
        2D -> 4-connectivity
        3D -> 6-connectivity
        nD -> axis-neighbors only
    """
    assert connectivity in {"full", "face"}, "connectivity must be 'full' or 'face'"

    if connectivity == "full":
        return np.ones((3,) * ndim, dtype=bool)

    return ndimage.generate_binary_structure(ndim, 1)


def connected_components(
    binary_mask,
    connectivity: str = "full",
    min_size: int = 1,
):
    mask = _to_numpy_bool_mask(binary_mask)

    structure = make_connectivity_structure(
        ndim=mask.ndim,
        connectivity=connectivity,
    )

    labeled_mask, raw_n_components = ndimage.label(mask, structure=structure)

    new_labeled_mask = np.zeros_like(labeled_mask, dtype=np.int32)
    component_ids = []
    component_sizes = {}

    new_id = 1

    for old_id in range(1, raw_n_components + 1):
        component_mask = labeled_mask == old_id
        size = int(component_mask.sum())

        if size < min_size:
            continue

        new_labeled_mask[component_mask] = new_id
        component_ids.append(new_id)
        component_sizes[new_id] = size
        new_id += 1

    return {
        "labeled_mask": new_labeled_mask,
        "n_components": len(component_ids),
        "component_ids": component_ids,
        "component_sizes": component_sizes,
    }


def connected_components_pointcloud(
    points: torch.Tensor,
    radius: float,
    min_size: int = 1,
):
    """
    Connected components of a gridless point cloud.

    Two points are neighbors when their Euclidean distance is <= radius;
    components are the connected components of that radius graph
    (cKDTree pairs + union-find). This is the cloud counterpart of
    ``connected_components`` (which labels regular-grid boolean masks) and is
    what lets criteria consume gradient-sampled level sets in d >= 3.

    Returns dict:
        labels: LongTensor [N], component id per point (0 = filtered out)
        n_components: int
        component_ids: list[int] (1-based, size-filtered, relabeled)
        component_sizes: dict[int, int]
    """
    assert points.ndim == 2, "points must have shape [N, dim]"
    assert radius > 0, "radius must be positive"
    assert min_size >= 1, "min_size must be >= 1"

    n = len(points)
    labels = torch.zeros(n, dtype=torch.long)

    if n == 0:
        return {
            "labels": labels,
            "n_components": 0,
            "component_ids": [],
            "component_sizes": {},
        }

    points_np = points.detach().cpu().numpy()

    tree = cKDTree(points_np)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")

    parent = np.arange(n)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri

    roots = np.array([find(int(i)) for i in range(n)])

    labels_np = np.zeros(n, dtype=np.int64)
    component_ids = []
    component_sizes = {}
    new_id = 1

    for root in np.unique(roots):
        member_mask = roots == root
        size = int(member_mask.sum())

        if size < min_size:
            continue

        labels_np[member_mask] = new_id
        component_ids.append(new_id)
        component_sizes[new_id] = size
        new_id += 1

    return {
        "labels": torch.from_numpy(labels_np),
        "n_components": len(component_ids),
        "component_ids": component_ids,
        "component_sizes": component_sizes,
    }


# -------------------------
# Affine subspaces (PCA)
# -------------------------

def fit_affine_subspace(
    points: torch.Tensor,
    k: int,
) -> dict:
    """
    Fits a k-dimensional affine subspace to ambient-space points via PCA (SVD).

    Intended anchor clouds (see PLAN.md §Sampling): the trajectory of gradient
    ascent to a neuron's local extremum (Wang-Ponce style), or the feature
    point set A itself. The subspace is fixed FIRST; level sets of the
    restricted function are computed SECOND (the order does not commute).

    Returns dict:
        origin: Tensor [dim] — mean of points
        basis: Tensor [k, dim] — orthonormal rows, top principal directions
        singular_values: Tensor [min(N, dim)]
        explained_variance_ratio: Tensor [min(N, dim)]
    """
    assert points.ndim == 2, "points must have shape [N, dim]"
    assert len(points) >= 2, "need at least 2 points to fit a subspace"
    assert 1 <= k <= points.shape[1], "k must be in [1, dim]"

    points = points.detach()
    if not torch.is_floating_point(points):
        points = points.float()

    origin = points.mean(dim=0)
    centered = points - origin

    _, singular_values, Vh = torch.linalg.svd(centered, full_matrices=False)

    variance = singular_values**2
    explained_variance_ratio = variance / variance.sum().clamp_min(1e-12)

    return {
        "origin": origin,
        "basis": Vh[:k],
        "singular_values": singular_values,
        "explained_variance_ratio": explained_variance_ratio,
    }


# -------------------------
# Grids and scalar-field evaluation
# -------------------------

def make_grid(
    bounds,
    resolution: int | list[int] | tuple[int, ...],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    bounds: [(a1, b1), ..., (ad, bd)]
    resolution:
        int -> same resolution for each axis
        list/tuple -> separate resolution for each axis

    Returns:
        grid_points: [N, dim]
        grid_shape: tuple
    """
    dim = len(bounds)

    if isinstance(resolution, int):
        resolutions = [resolution] * dim
    else:
        resolutions = list(resolution)
        assert len(resolutions) == dim, "resolution must have length equal to len(bounds)"

    axes = [
        torch.linspace(a, b, res, device=device, dtype=dtype)
        for (a, b), res in zip(bounds, resolutions)
    ]

    mesh = torch.meshgrid(*axes, indexing="ij")

    grid_points = torch.stack(
        [m.reshape(-1) for m in mesh],
        dim=1,
    )

    grid_shape = tuple(resolutions)

    return grid_points, grid_shape


def grid_step_from_bounds_resolution(bounds, resolution) -> float:
    """
    Returns the maximal grid step over all axes.
    """
    dim = len(bounds)

    if isinstance(resolution, int):
        resolutions = [resolution] * dim
    else:
        resolutions = list(resolution)
        assert len(resolutions) == dim

    steps = []

    for (a, b), res in zip(bounds, resolutions):
        assert res > 1, "resolution must be > 1"
        steps.append((b - a) / (res - 1))

    return max(steps)


def evaluate_scalar_function_on_points(
    function,
    points: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    """
    Evaluates scalar function on points.

    Returns:
        values: [N]
    """
    values = []

    with torch.no_grad():
        for batch in points.split(batch_size):
            batch_values = function(batch)

            if batch_values.ndim == 0:
                batch_values = batch_values[None]
            elif batch_values.ndim > 1:
                assert batch_values.shape[-1] == 1, (
                    f"Expected scalar output, got shape {batch_values.shape}"
                )
                batch_values = batch_values.reshape(-1)

            values.append(batch_values.detach())

    return torch.cat(values, dim=0)
