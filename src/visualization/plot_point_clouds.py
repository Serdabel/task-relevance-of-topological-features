import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Хорошо различимые цвета (colorblind-friendly)
COLOR_PALETTE = [
    "#1f77b4",  # синий
    "#d62728",  # красный
    "#2ca02c",  # зелёный
    "#ff7f0e",  # оранжевый
    "#9467bd",  # фиолетовый
    "#17becf",  # бирюзовый
    "#8c564b",  # коричневый
    "#e377c2",  # розовый
]


def get_color(cls):
    return COLOR_PALETTE[int(cls) % len(COLOR_PALETTE)]


def set_equal_axes_(ax, X):
    X = to_numpy(X)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = X[:, 2].min(), X[:, 2].max()

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def plot_point_cloud_3d(
    X,
    y=None,
    level_points=None,
    title=None,
    size=4,
    elev=20,
    azim=0,
):
    X = to_numpy(X)
    if y is not None:
        y = to_numpy(y)
    if level_points is not None:
        level_points = to_numpy(level_points)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- основной датасет ---
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=size, color="#1f77b4")
    else:
        classes = np.unique(y)

        for cls in classes:
            mask = y == cls
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                X[mask, 2],
                s=size,
                color=get_color(cls),
                label=f"class {cls}",
            )

        ax.legend()

    # --- level set точки ---
    if level_points is not None and len(level_points) > 0:
        ax.scatter(
            level_points[:, 0],
            level_points[:, 1],
            level_points[:, 2],
            s=size * 0.5,
            color="black",
            alpha=0.25,
            label=None,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)
    set_equal_axes_(ax, X)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_point_cloud_3d_interactive(
    X,
    y=None,
    level_points=None,
    size=4,
    level_size=2,
    title="Point cloud",
):
    X = to_numpy(X)

    if y is not None:
        y = to_numpy(y)

    fig = go.Figure()

    if y is None:
        fig.add_trace(go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(size=size),
            name="data",
        ))
    else:
        for cls in np.unique(y):
            mask = y == cls
            fig.add_trace(go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2],
                mode="markers",
                marker=dict(size=size),
                name=f"class {cls}",
            ))

    if level_points is not None:
        P = to_numpy(level_points)

        fig.add_trace(go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers",
            marker=dict(
                size=level_size,
                color="black",
                opacity=0.25,
            ),
            name="level set",
        ))

    fig.update_layout(
        title=title,
        width=800,
        height=700,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
    )

    fig.show()


def plot_point_cloud_2d_interactive(
    X,
    y=None,
    level_points=None,
    size=5,
    level_size=3,
    title="Point cloud",
):
    X = to_numpy(X)

    if y is not None:
        y = to_numpy(y)

    fig = go.Figure()

    # --- основной датасет ---
    if y is None:
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                size=size,
                color="#1f77b4",
                opacity=0.85,
            ),
            name="data",
        ))
    else:
        for cls in np.unique(y):
            mask = y == cls

            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(
                    size=size,
                    color=get_color(cls),
                    opacity=0.85,
                ),
                name=f"class {cls}",
            ))

    # --- level set points ---
    if level_points is not None:
        P = to_numpy(level_points)

        if len(P) > 0:
            fig.add_trace(go.Scatter(
                x=P[:, 0],
                y=P[:, 1],
                mode="markers",
                marker=dict(
                    size=level_size,
                    color="black",
                    opacity=0.25,
                ),
                name="level set",
            ))

    fig.update_layout(
        title=title,
        width=750,
        height=700,
        xaxis=dict(
            title="x",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="y",
        ),
        legend=dict(
            itemsizing="constant",
        ),
    )

    fig.show()


def plot_scalar_field_2d_interactive(
    function,
    bounds,
    resolution=200,
    X=None,
    y=None,
    level_points=None,
    levels=None,
    batch_size=8192,
    device="cpu",
    title="Scalar field",
    data_size=5,
    level_size=5,
    opacity_data=0.95,
    opacity_level=0.75,
):
    """
    Interactive heatmap for scalar function f: R^2 -> R.

    Draw order:
        1. heatmap
        2. contour lines
        3. original data points
        4. sampled level-set points
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    xs = torch.linspace(x_min, x_max, resolution, device=device)
    ys = torch.linspace(y_min, y_max, resolution, device=device)

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
        ],
        dim=1,
    )

    values = []

    with torch.no_grad():
        for batch in grid.split(batch_size):
            v = function(batch).reshape(-1)
            values.append(v.detach().cpu())

    values = torch.cat(values).numpy()
    Z = values.reshape(resolution, resolution)

    xs_np = xs.detach().cpu().numpy()
    ys_np = ys.detach().cpu().numpy()

    fig = go.Figure()

    # --- 1. heatmap ---
    fig.add_trace(go.Heatmap(
        x=xs_np,
        y=ys_np,
        z=Z,
        colorscale="Viridis",
        colorbar=dict(title="f(x)"),
        name="activation",
    ))

    # --- 2. contour lines ---
    if levels is not None:
        for c in levels:
            fig.add_trace(go.Contour(
                x=xs_np,
                y=ys_np,
                z=Z,
                contours=dict(
                    type="constraint",
                    operation="=",
                    value=float(c),
                    coloring="lines",
                    showlabels=True,
                ),
                line=dict(
                    color="white",
                    width=3,
                ),
                showscale=False,
                name=f"contour f={float(c):.3f}",
            ))

    # --- 3. original data points ---
    if X is not None:
        X_np = to_numpy(X)

        if y is None:
            fig.add_trace(go.Scatter(
                x=X_np[:, 0],
                y=X_np[:, 1],
                mode="markers",
                marker=dict(
                    size=data_size,
                    color="white",
                    opacity=opacity_data,
                    line=dict(width=0),
                ),
                name="data",
            ))
        else:
            y_np = to_numpy(y)

            for cls in np.unique(y_np):
                mask = y_np == cls
                color = get_color(cls)

                fig.add_trace(go.Scatter(
                    x=X_np[mask, 0],
                    y=X_np[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=data_size,
                        color=color,
                        opacity=opacity_data,
                        line=dict(width=0),
                    ),
                    name=f"class {cls}",
                ))

    # --- 4. sampled level-set points ---
    if level_points is not None:
        P = to_numpy(level_points)

        if len(P) > 0:
            fig.add_trace(go.Scatter(
                x=P[:, 0],
                y=P[:, 1],
                mode="markers",
                marker=dict(
                    size=level_size,
                    color="black",
                    opacity=opacity_level,
                    symbol="circle",
                    line=dict(width=0),
                ),
                name="sampled level set",
            ))

    fig.update_layout(
        title=title,
        width=800,
        height=750,
        xaxis=dict(
            title="x",
            scaleanchor="y",
            scaleratio=1,
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title="y",
            range=[y_min, y_max],
        ),
    )

    fig.show()

    return Z


def plot_all_hidden_neuron_fields_2d(
    model,
    bounds,
    X=None,
    y=None,
    kind="post",
    resolution=150,
    batch_size=8192,
    device="cpu",
    cmap="viridis",              # стандартная и яркая
    vmin=-1.0,                   # фиксированный минимум
    vmax=1.0,                    # фиксированный максимум
    data_size=3,
    data_alpha=0.8,
    figsize_per_cell=(3.0, 3.0),
    show_colorbar=True,
    title=None,
    interpolation="bilinear",
):
    """
    Очень простая визуализация активаций нейронов.
    - Без автоматической нормализации: vmin/vmax фиксированы (по умолчанию -1 и 1).
    - Без усечения выбросов, без симметризации.
    """
    model.eval()
    model.to(device)

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # сетка
    xs = torch.linspace(x_min, x_max, resolution, device=device)
    ys = torch.linspace(y_min, y_max, resolution, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    if not hasattr(model, "n_layers"):
        raise ValueError("Модель должна иметь атрибут `n_layers`")
    n_layers = model.n_layers

    # вычисляем активации для всех слоёв
    layer_fields = []
    layer_widths = []
    with torch.no_grad():
        for layer_id in range(n_layers):
            outputs = []
            for batch in grid.split(batch_size):
                h = model.forward_until(batch, layer_id=layer_id, kind=kind)
                outputs.append(h.detach().cpu())
            H = torch.cat(outputs, dim=0)
            width_l = H.shape[1]
            H = H.numpy()
            Z_l = H.reshape(resolution, resolution, width_l)
            layer_fields.append(Z_l)
            layer_widths.append(width_l)

    max_width = max(layer_widths)

    # подготовка данных для оверлея
    X_np = to_numpy(X) if X is not None else None
    y_np = to_numpy(y) if y is not None else None

    # построение фигуры
    colorbar_extra_width = 0.35 if show_colorbar else 0.0
    fig_width = figsize_per_cell[0] * n_layers + colorbar_extra_width
    fig_height = figsize_per_cell[1] * max_width
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    if show_colorbar:
        gs = GridSpec(max_width, n_layers + 1, figure=fig,
                      width_ratios=[1.0] * n_layers + [0.08],
                      wspace=0.06, hspace=0.12)
    else:
        gs = GridSpec(max_width, n_layers, figure=fig,
                      wspace=0.06, hspace=0.12)

    im_for_colorbar = None

    for layer_id in range(n_layers):
        Z_l = layer_fields[layer_id]
        width_l = layer_widths[layer_id]
        for coord_id in range(max_width):
            ax = fig.add_subplot(gs[coord_id, layer_id])
            if coord_id >= width_l:
                ax.axis("off")
                continue

            Z = Z_l[:, :, coord_id]
            im = ax.imshow(
                Z,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                cmap=cmap,
                vmin=vmin,          # жёстко заданы
                vmax=vmax,
                aspect="equal",
                interpolation=interpolation,
            )
            im_for_colorbar = im

            # оверлей исходных точек
            if X_np is not None:
                if y_np is None:
                    ax.scatter(X_np[:, 0], X_np[:, 1], s=data_size,
                               c="white", alpha=data_alpha, linewidths=0)
                else:
                    for cls in np.unique(y_np):
                        mask = y_np == cls
                        ax.scatter(X_np[mask, 0], X_np[mask, 1],
                                   s=data_size, color=get_color(cls),
                                   alpha=data_alpha, linewidths=0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            if coord_id == 0:
                ax.set_title(f"layer {layer_id}", fontsize=10, color="black")
            if layer_id == 0:
                ax.set_ylabel(f"coord {coord_id}", fontsize=10, color="black")

    if show_colorbar and im_for_colorbar is not None:
        cax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im_for_colorbar, cax=cax)
        cbar.set_label("activation", color="black")
        cbar.ax.yaxis.set_tick_params(color="black")
        plt.setp(cbar.ax.get_yticklabels(), color="black")

    if title:
        fig.suptitle(title, fontsize=16, color="black", y=0.995)

    plt.show()
    return layer_fields


def plot_all_hidden_neuron_fields_2d_with_criteria(
    model,
    results,
    bounds,
    X=None,
    y=None,
    kind="post",
    resolution=150,
    batch_size=8192,
    device="cpu",
    cmap="viridis",
    vmin=-1.0,
    vmax=1.0,
    data_size=3,
    data_alpha=0.8,
    component_size=6,
    component_alpha=1.0,
    component_color="red",
    feature_points=None,
    feature_color="white",
    figsize_per_cell=None,
    show_colorbar=True,
    title=None,
    interpolation="bilinear",
    text_fontsize=7,
):
    """
    Visualizes hidden neuron scalar fields and overlays:

        1. data points X, y;
        2. best structural level-set component;
        3. main structural/tubular criterion values;
        4. optionally the target feature itself (``feature_points``).

    Pass ``feature_points`` whenever the feature is NOT part of the data — an
    A.1 null is a rigid copy of the target placed elsewhere and appears nowhere
    in X, so without it the panel shows a red component with nothing to compare
    it against.

    Requires results from evaluate_neurons_on_feature(...):

        structural_kwargs["return_best_component_points"] = True
        return_full_result_structural = True
    """
    model.eval()
    model.to(device)

    if not hasattr(model, "n_layers"):
        raise ValueError("model must have attribute `n_layers`")

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x_range = x_max - x_min
    y_range = y_max - y_min

    if figsize_per_cell is None:
        cell_width = 3.2
        cell_height = cell_width * y_range / x_range
        figsize_per_cell = (cell_width, cell_height)

    # map: (layer_id, coord_id) -> result row
    result_by_neuron = {
        (row["layer_id"], row["coord_id"]): row
        for row in results
        if row["kind"] == kind
    }

    # grid
    xs = torch.linspace(x_min, x_max, resolution, device=device)
    ys = torch.linspace(y_min, y_max, resolution, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # evaluate all layer fields
    n_layers = model.n_layers
    layer_fields = []
    layer_widths = []

    with torch.no_grad():
        for layer_id in range(n_layers):
            outputs = []

            for batch in grid.split(batch_size):
                h = model.forward_until(
                    batch,
                    layer_id=layer_id,
                    kind=kind,
                )
                outputs.append(h.detach().cpu())

            H = torch.cat(outputs, dim=0)
            width_l = H.shape[1]
            Z_l = H.numpy().reshape(resolution, resolution, width_l)

            layer_fields.append(Z_l)
            layer_widths.append(width_l)

    max_width = max(layer_widths)

    X_np = to_numpy(X) if X is not None else None
    y_np = to_numpy(y) if y is not None else None
    A_np = to_numpy(feature_points) if feature_points is not None else None

    # figure
    colorbar_extra_width = 0.35 if show_colorbar else 0.0
    title_extra_height = 0.45 if title else 0.05

    fig_width = figsize_per_cell[0] * n_layers + colorbar_extra_width
    fig_height = figsize_per_cell[1] * max_width + title_extra_height

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # Layout fractions.
    # top is computed so title takes a fixed absolute height, not a huge fraction.
    grid_top = 1.0 - title_extra_height / fig_height
    grid_top = min(grid_top, 0.965)
    grid_bottom = 0.035
    grid_left = 0.055
    grid_right = 0.965 if show_colorbar else 0.985

    if show_colorbar:
        gs = GridSpec(
            max_width,
            n_layers + 1,
            figure=fig,
            width_ratios=[1.0] * n_layers + [0.08],
            wspace=0.06,
            hspace=0.14,
            left=grid_left,
            right=grid_right,
            bottom=grid_bottom,
            top=grid_top,
        )
    else:
        gs = GridSpec(
            max_width,
            n_layers,
            figure=fig,
            wspace=0.06,
            hspace=0.14,
            left=grid_left,
            right=grid_right,
            bottom=grid_bottom,
            top=grid_top,
        )

    im_for_colorbar = None

    for layer_id in range(n_layers):
        Z_l = layer_fields[layer_id]
        width_l = layer_widths[layer_id]

        for coord_id in range(max_width):
            ax = fig.add_subplot(gs[coord_id, layer_id])

            if coord_id >= width_l:
                ax.axis("off")
                continue

            Z = Z_l[:, :, coord_id]

            im = ax.imshow(
                Z,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation=interpolation,
            )
            im_for_colorbar = im

            # data overlay
            if X_np is not None:
                if y_np is None:
                    ax.scatter(
                        X_np[:, 0],
                        X_np[:, 1],
                        s=data_size,
                        c="white",
                        alpha=data_alpha,
                        linewidths=0,
                    )
                else:
                    for cls in np.unique(y_np):
                        mask = y_np == cls
                        ax.scatter(
                            X_np[mask, 0],
                            X_np[mask, 1],
                            s=data_size,
                            color=get_color(cls),
                            alpha=data_alpha,
                            linewidths=0,
                        )

            # the feature the criterion was scored against, when it is not
            # itself part of the data (nulls never are)
            if A_np is not None:
                ax.scatter(
                    A_np[:, 0],
                    A_np[:, 1],
                    s=data_size + 2,
                    facecolors="none",
                    edgecolors=feature_color,
                    linewidths=0.5,
                )

            row = result_by_neuron.get((layer_id, coord_id), None)

            # best structural component overlay
            if row is not None and "structural_result" in row:
                structural_result = row["structural_result"]
                best_component_points = structural_result.get(
                    "best_component_points",
                    None,
                )

                if best_component_points is not None:
                    if torch.is_tensor(best_component_points):
                        best_component_points = (
                            best_component_points.detach().cpu().numpy()
                        )

                    ax.scatter(
                        best_component_points[:, 0],
                        best_component_points[:, 1],
                        s=component_size,
                        color=component_color,
                        alpha=component_alpha,
                        linewidths=0,
                    )

            # criteria text: one line per criterion the evaluation actually
            # ran. Which keys a row carries depends on the EvalSpec — tubular
            # is off by default and the extra criteria are opt-in — so every
            # line is conditional instead of indexed.
            if row is not None:
                witness = bool(row.get("is_structural_witness"))

                lines = []

                for label, key, fmt in (
                    ("s/diam", "structural_score_normalized_by_diameter", ".3f"),
                    ("s/grid", "structural_score_normalized_by_grid_step", ".2f"),
                    ("tube", "tubular_score", ".3f"),
                    ("extr", "extremum_score_normalized_by_diameter", ".3f"),
                    ("homol", "homological_persistence_margin", ".2f"),
                ):
                    value = row.get(key)

                    if value is None:
                        continue

                    lines.append(f"{label:<6} = {value:{fmt}}")

                lines.append(f"witness: {witness}")

                text = "\n".join(lines)

                bbox_edgecolor = "limegreen" if witness else "black"

                ax.text(
                    0.02,
                    0.98,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=text_fontsize,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.78,
                        edgecolor=bbox_edgecolor,
                        boxstyle="round,pad=0.25",
                    ),
                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            if coord_id == 0:
                ax.set_title(
                    f"layer {layer_id}",
                    fontsize=10,
                    color="black",
                    pad=4,
                )

            if layer_id == 0:
                ax.set_ylabel(
                    f"coord {coord_id}",
                    fontsize=10,
                    color="black",
                )

    if show_colorbar and im_for_colorbar is not None:
        cax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im_for_colorbar, cax=cax)
        cbar.set_label("activation", color="black")
        cbar.ax.yaxis.set_tick_params(color="black")
        plt.setp(cbar.ax.get_yticklabels(), color="black")

    if title:
        fig.suptitle(
            title,
            fontsize=16,
            color="black",
            y=0.99,
        )

    plt.show()

    return layer_fields

def plot_decision_boundary_2d(
    model,
    bounds,
    X=None,
    y=None,
    resolution=300,
    device="cpu",
    title=None,
    show_data=True,
):
    """The head: P(class 1) over the plane, with the 0.5 contour and the data.

    The counterpart of the hidden-neuron grids for the network's OUTPUT — what
    the model actually computes, as opposed to what any one neuron contributes.
    Static; ``ExperimentLab.show_decision_boundary`` is the same picture on the
    lab's own box.
    """
    import matplotlib.pyplot as plt
    from analysis.geometry import make_grid

    model.eval()
    model.to(device)

    grid, shape = make_grid(bounds=bounds, resolution=resolution, device=device)
    with torch.no_grad():
        logits = model(grid).view(-1)
        # make_grid is x-major (Z[ix, iy]); imshow wants Z[iy, ix] -> .T
        prob = torch.sigmoid(logits).reshape(shape).T.cpu().numpy()

    (x0, x1), (y0, y1) = bounds[0], bounds[1]
    fig, ax = plt.subplots(figsize=(9.0, 9.0 * (y1 - y0) / (x1 - x0) + 1.0))
    im = ax.imshow(prob, extent=[x0, x1, y0, y1], origin="lower", cmap="RdBu_r",
                   vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
    ax.contour(prob, levels=[0.5], extent=[x0, x1, y0, y1], colors="k",
               linewidths=1.2)
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, label="P(class 1)")

    if show_data and X is not None:
        X_np = to_numpy(X)
        y_np = to_numpy(y) if y is not None else None
        if y_np is None:
            ax.scatter(X_np[:, 0], X_np[:, 1], s=4, alpha=0.6, linewidths=0)
        else:
            for cls in sorted(set(y_np.reshape(-1).tolist())):
                m = y_np.reshape(-1) == cls
                ax.scatter(X_np[m, 0], X_np[m, 1], s=4, alpha=0.7, linewidths=0,
                           color=get_color(int(cls)), label=f"class {int(cls)}")
            ax.legend(fontsize=8, loc="lower right", markerscale=2.5)

    ax.set_title(title or "decision boundary (head)", fontsize=11)
    fig.tight_layout()
    plt.show()
    return fig
