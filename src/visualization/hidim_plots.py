"""Visualization gallery for d >= 3 experiments (exp. 9-10).

A deliberately WIDE menu of views for the higher-dimensional setting, built
for the exp9 trial notebook: the caller picks a few that earn their keep and
the rest can be deleted later. Grouped by what they look at:

  whole space (3D only):
    plot_level_set_isosurface   — marching-cubes surface of {f = c}
    plot_decision_isosurface    — the model's P=0.5 surface
    plot_sampled_level_set_3d   — gradient-sampled level set, colored by
                                  connected component (the d>=3 workhorse:
                                  works in R^4 too, viewed after projection)

  2D restrictions (any d):
    plot_field_slices           — axis-plane cross-section heatmaps
    plot_pca_slice              — f restricted to the feature's PCA plane
                                  (subspace FIRST, level sets second)

  scalar summaries (any d):
    plot_value_vs_radius        — f vs distance-to-center scatter density
    plot_tube_shell_histograms  — the tubular criterion's two distributions
    plot_superlevel_barcode     — barcode view of superlevel persistence
    plot_level_coverage         — sampler points per level (coverage caveat)

matplotlib figures are returned after plt.show(); plotly figures call
fig.show() and are returned.
"""

import torch

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# whole space (3D)
# ----------------------------------------------------------------------

def _grid_values_3d(function, bounds, resolution, device="cpu", batch_size=8192):
    from analysis.geometry import make_grid, evaluate_scalar_function_on_points

    grid, shape = make_grid(bounds=bounds, resolution=resolution, device=device)
    values = evaluate_scalar_function_on_points(
        function=function, points=grid, batch_size=batch_size
    )
    return values.reshape(shape).cpu().numpy(), shape


def plot_level_set_isosurface(
    function,
    bounds,
    level: float,
    resolution: int = 60,
    feature_points: torch.Tensor | None = None,
    X: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    opacity: float = 0.45,
    title: str | None = None,
    device: str = "cpu",
):
    """Marching-cubes isosurface of {f = level} as an interactive plotly mesh,
    with the feature (and optionally the data) overlaid. 3D only — the
    clearest possible view of one level set, at grid cost (resolution^3)."""
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    assert len(bounds) == 3, "isosurfaces are 3D-only"
    Z, shape = _grid_values_3d(function, bounds, resolution, device=device)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    spacing = (hi - lo) / (np.array(shape) - 1)

    fig = go.Figure()

    try:
        verts, faces, _, _ = marching_cubes(Z, level=level, spacing=tuple(spacing))
        verts = verts + lo  # grid is x-major (indexing="ij"), so axes align
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=opacity, color="crimson", name=f"f = {level:.3f}",
        ))
    except ValueError:
        print(f"level {level:.3f} not crossed inside bounds — nothing to draw")

    if feature_points is not None:
        p = feature_points.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=2, color="black"), name="feature",
        ))

    if X is not None:
        Xn = X.detach().cpu().numpy()
        yn = None if y is None else y.detach().cpu().numpy()
        for cls in ([None] if yn is None else np.unique(yn)):
            m = slice(None) if cls is None else (yn == cls)
            fig.add_trace(go.Scatter3d(
                x=Xn[m, 0], y=Xn[m, 1], z=Xn[m, 2], mode="markers",
                marker=dict(size=1.5, opacity=0.35),
                name="data" if cls is None else f"class {int(cls)}",
            ))

    fig.update_layout(
        title=title or f"level set f = {level:.3f}",
        width=750, height=700,
        scene=dict(aspectmode="data"),
    )
    fig.show()


def plot_decision_isosurface(
    model,
    bounds,
    resolution: int = 60,
    X: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    device: str = "cpu",
):
    """The whole model's decision boundary: the P(class 1) = 0.5 isosurface.
    3D only."""

    def prob(points):
        with torch.no_grad():
            return torch.sigmoid(model(points).view(-1))

    return plot_level_set_isosurface(
        prob, bounds, level=0.5, resolution=resolution,
        X=X, y=y, opacity=0.35, device=device,
        title="decision boundary  P(class 1) = 0.5",
    )


def plot_sampled_level_set_3d(
    points: torch.Tensor,
    feature_points: torch.Tensor | None = None,
    component_radius: float | None = None,
    min_component_size: int = 5,
    title: str = "sampled level set (colored by component)",
):
    """Gradient-sampled level-set cloud as an interactive 3D scatter, colored
    by connected component (radius-graph clustering — same machinery the
    sampled-structural criterion scores). THE grid-free view: in R^4 the same
    cloud is drawn after projecting out one coordinate or through PCA."""
    import plotly.graph_objects as go
    from analysis.geometry import connected_components_pointcloud
    from analysis.geometry import feature_nearest_neighbor_spacing

    assert points.shape[1] == 3, "3D scatter; project higher-dim clouds first"

    if component_radius is None and feature_points is not None:
        component_radius = 3.0 * float(
            feature_nearest_neighbor_spacing(feature_points.float()).item()
        )

    fig = go.Figure()

    if len(points) >= min_component_size and component_radius is not None:
        comp = connected_components_pointcloud(
            points, radius=component_radius, min_size=min_component_size
        )
        labels = comp["labels"].cpu().numpy()
        p = points.detach().cpu().numpy()
        for cid in comp["component_ids"]:
            m = labels == cid
            fig.add_trace(go.Scatter3d(
                x=p[m, 0], y=p[m, 1], z=p[m, 2], mode="markers",
                marker=dict(size=2.5), name=f"component {cid} ({m.sum()} pts)",
            ))
    else:
        p = points.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=2.5), name=f"level set ({len(p)} pts)",
        ))

    if feature_points is not None:
        f = feature_points.detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(
            x=f[:, 0], y=f[:, 1], z=f[:, 2], mode="markers",
            marker=dict(size=1.5, color="black", opacity=0.5), name="feature",
        ))

    fig.update_layout(title=title, width=750, height=700,
                      scene=dict(aspectmode="data"))
    fig.show()


# ----------------------------------------------------------------------
# 2D restrictions (any d)
# ----------------------------------------------------------------------

def plot_field_slices(
    function,
    bounds,
    center: torch.Tensor,
    resolution: int = 120,
    feature_points: torch.Tensor | None = None,
    slab_eps: float = 0.15,
    cmap: str = "viridis",
    device: str = "cpu",
):
    """Axis-plane cross-sections through ``center``: one 2D heatmap per pair
    of axes, the remaining coordinates frozen at the center's values. Feature
    points within ``slab_eps`` of each plane are overlaid (white). Works in
    any dimension — in R^4 you get 6 panels."""
    from itertools import combinations

    from analysis.geometry import make_grid, evaluate_scalar_function_on_points

    dim = len(bounds)
    center = torch.as_tensor(center, dtype=torch.float32).reshape(dim)
    pairs = list(combinations(range(dim), 2))

    n = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)

    feat = None if feature_points is None else feature_points.detach().cpu()

    for idx, (ax_i, ax_j) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]

        plane_bounds = [bounds[ax_i], bounds[ax_j]]
        grid2, shape2 = make_grid(bounds=plane_bounds, resolution=resolution,
                                  device=device)
        full = center.repeat(len(grid2), 1)
        full[:, ax_i] = grid2[:, 0]
        full[:, ax_j] = grid2[:, 1]

        values = evaluate_scalar_function_on_points(function, full)
        Z = values.reshape(shape2).T.cpu().numpy()   # x-major -> imshow

        (x0, x1), (y0, y1) = plane_bounds
        im = ax.imshow(Z, extent=[x0, x1, y0, y1], origin="lower",
                       cmap=cmap, aspect="equal")

        if feat is not None:
            frozen = [a for a in range(dim) if a not in (ax_i, ax_j)]
            m = torch.ones(len(feat), dtype=torch.bool)
            for a in frozen:
                m &= (feat[:, a] - center[a]).abs() <= slab_eps
            if m.any():
                ax.scatter(feat[m, ax_i], feat[m, ax_j], s=4, c="white",
                           alpha=0.8, linewidths=0)

        frozen_txt = ", ".join(
            f"x{a}={center[a]:.2f}" for a in range(dim) if a not in (ax_i, ax_j)
        )
        ax.set_title(f"axes (x{ax_i}, x{ax_j})  |  {frozen_txt}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("axis-plane cross-sections (white = feature within the slab)",
                 fontsize=11)
    fig.tight_layout()
    plt.show()


def plot_pca_slice(
    function,
    anchor_points: torch.Tensor,
    feature_points: torch.Tensor | None = None,
    k: int = 2,
    resolution: int = 150,
    bounds_margin: float = 0.5,
    cmap: str = "viridis",
):
    """f restricted to the top-k PCA plane of ``anchor_points`` (typically the
    feature itself, or a gradient-ascent trajectory) — the project's canonical
    d >= 3 route: SUBSPACE FIRST, level sets second. Returns the slice dict
    from analysis.subspace_slices (restricted fn, bounds, explained
    variance)."""
    from analysis.fields import evaluate_field_on_grid
    from analysis.subspace_slices import pca_slice_from_trajectory

    assert k == 2, "the heatmap view is k=2 (use evaluate_neuron_on_slice for k=3)"

    slice_ = pca_slice_from_trajectory(
        function, anchor_points, k=k, bounds_margin=bounds_margin
    )
    restricted = slice_["restricted"]

    field = evaluate_field_on_grid(
        restricted, bounds=slice_["bounds"], resolution=resolution
    )
    Z = field.values_nd.T.cpu().numpy()
    (x0, x1), (y0, y1) = slice_["bounds"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(Z, extent=[x0, x1, y0, y1], origin="lower", cmap=cmap,
                   aspect="equal")

    if feature_points is not None:
        proj = (feature_points.float() - restricted.origin) @ restricted.basis.T
        ax.scatter(proj[:, 0], proj[:, 1], s=4, c="white", alpha=0.7,
                   linewidths=0, label="feature (projected)")
        ax.legend(fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, label="activation")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"f restricted to the anchor PCA plane  "
                 f"(explained var: {slice_['explained_variance_ratio']:.2f})",
                 fontsize=10)
    fig.tight_layout()
    plt.show()
    return slice_


# ----------------------------------------------------------------------
# scalar summaries (any d)
# ----------------------------------------------------------------------

def plot_value_vs_radius(
    field_points: torch.Tensor,
    field_values: torch.Tensor,
    center: torch.Tensor,
    r_feature: float | None = None,
    gridsize: int = 60,
):
    """Density scatter (hexbin) of f vs distance to ``center`` over a field
    sample — the FULL joint picture behind the radial profile: a
    shell-encoding neuron shows a narrow band at the feature radius, a
    half-space neuron a wide wedge. Any dimension."""
    r = (field_points - torch.as_tensor(center, dtype=field_points.dtype)) \
        .norm(dim=1).cpu().numpy()
    v = field_values.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    hb = ax.hexbin(r, v, gridsize=gridsize, cmap="magma", bins="log")
    if r_feature is not None:
        ax.axvline(r_feature, color="cyan", ls="--", lw=1.2,
                   label=f"feature radius ≈ {r_feature:.2f}")
        ax.legend(fontsize=8)
    fig.colorbar(hb, ax=ax, label="log count")
    ax.set_xlabel("distance to feature center")
    ax.set_ylabel("activation f(x)")
    ax.set_title("value vs radius over the field sample", fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_tube_shell_histograms(
    field,
    feature_points: torch.Tensor,
    delta: float | None = None,
    Delta: float | None = None,
):
    """What the tubular criterion actually compares: the distribution of f on
    the tube {d(x, A) < delta} vs the shell {delta < d(x, A) < Delta}. The
    reported score is the normalized Wasserstein-1 distance between the two
    histograms. Any dimension."""
    from analysis.criterions import (
        resolve_tubular_radii, normalized_wasserstein_1_distance,
    )
    from analysis.geometry import nearest_distances_to_set

    radii = resolve_tubular_radii(
        feature_points=feature_points.float(),
        bounds=[(0, 1)],          # unused when resolution=None
        resolution=None,
        delta="auto" if delta is None else delta,
        Delta="auto" if Delta is None else Delta,
    )
    d, D = radii["delta"], radii["Delta"]

    dist = nearest_distances_to_set(field.points, feature_points.float())
    tube = field.values[dist < d]
    shell = field.values[(dist > d) & (dist < D)]

    score = normalized_wasserstein_1_distance(tube, shell)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(tube.cpu().numpy(), bins=40, alpha=0.6, density=True,
            label=f"tube d<{d:.2f}  ({len(tube)} pts)")
    ax.hist(shell.cpu().numpy(), bins=40, alpha=0.6, density=True,
            label=f"shell {d:.2f}<d<{D:.2f}  ({len(shell)} pts)")
    ax.set_xlabel("activation f(x)"); ax.set_ylabel("density")
    ax.set_title(f"tubular criterion input  |  W1-normalized = {score:.3f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_superlevel_barcode(
    diagrams: dict,
    title: str = "superlevel persistence barcode",
):
    """Barcode view of ``superlevel_persistence`` diagrams ({dim: [n, 2]
    (birth_level, death_level)}, filtration sweeping DOWN): one horizontal bar
    per homology class from its birth level down to its death level, grouped
    and colored by dimension. The witness signature = one long H_{d-1} bar."""
    colors = {0: "#999999", 1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}

    rows = []
    for dim in sorted(diagrams):
        bars = np.asarray(diagrams[dim], dtype=float).reshape(-1, 2)
        bars = bars[np.isfinite(bars).all(axis=1)]
        order = np.argsort(-(np.abs(bars[:, 0] - bars[:, 1]))) if len(bars) else []
        for b in bars[order]:
            rows.append((dim, b[0], b[1]))

    fig, ax = plt.subplots(figsize=(6.5, max(2.0, 0.16 * len(rows) + 1.2)))
    for i, (dim, birth, death) in enumerate(rows):
        ax.hlines(y=i, xmin=death, xmax=birth,
                  color=colors.get(dim, "black"), lw=2.5)

    handles = [plt.Line2D([0], [0], color=colors.get(d, "black"), lw=2.5,
                          label=f"H{d} ({len(np.asarray(diagrams[d]).reshape(-1, 2))})")
               for d in sorted(diagrams)]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xlabel("level c (filtration sweeps right → left)")
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_level_coverage(coverage: dict, min_component_size: int = 5):
    """The sampled-structural coverage caveat, drawn: accepted sampler points
    per candidate level. Levels below ``min_component_size`` (red) were
    UNMEASURED — a negative verdict there is missing data, not evidence."""
    levels = sorted(coverage)
    counts = [coverage[l] for l in levels]
    colors = ["crimson" if c < min_component_size else "steelblue"
              for c in counts]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(levels)), counts, color=colors)
    ax.axhline(min_component_size, color="grey", ls="--", lw=1,
               label=f"min component size = {min_component_size}")
    ax.set_xticks(range(len(levels)),
                  [f"{l:.2f}" for l in levels], rotation=45, fontsize=8)
    ax.set_xlabel("candidate level c"); ax.set_ylabel("accepted sample points")
    ax.set_title("level-set sampler coverage (red = unmeasured levels)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
