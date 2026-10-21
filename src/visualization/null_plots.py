"""
Pictures for manually placed A.1 null control features.

Companion to ``analysis.null_placement``. The whole point of placing nulls by
hand is to be able to SEE where they landed relative to the data before
spending compute on them, so this draws the training points, the target
feature, and every null, in one frame with the evaluation box.
"""

from __future__ import annotations

import torch


def _np(x):
    return torch.as_tensor(x).detach().cpu().numpy()


def _bounds_of(bounds, dim):
    return None if bounds is None else [bounds[k] for k in range(dim)]


def plot_null_placements(
    X,
    y,
    target_points,
    nulls: dict,
    bounds=None,
    title: str = "A.1 null placements",
    figsize=(11, 6),
    point_size: float = 6.0,
    show: bool = True,
):
    """Draw data + target + nulls. 2D matplotlib, 3D plotly, d>=4 skipped."""
    Xt = torch.as_tensor(X).detach().cpu()
    dim = Xt.shape[1]

    if dim == 2:
        return _plot_2d(
            Xt, y, target_points, nulls, bounds, title, figsize, point_size, show
        )
    if dim == 3:
        return _plot_3d(Xt, y, target_points, nulls, bounds, title, point_size, show)

    print(
        f"[null_plots] dim={dim}: no picture (nothing faithful to draw). "
        "Read the placement_report table instead, or project with "
        "analysis.subspace_slices first."
    )
    return None


def _colors(n):
    # distinct, colour-blind-safe-ish, and deliberately NOT the class colours
    base = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    ]
    return [base[i % len(base)] for i in range(n)]


def _plot_2d(Xt, y, target_points, nulls, bounds, title, figsize, s, show):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    if y is not None:
        yt = torch.as_tensor(y).flatten()
        for lab, col in ((0, "#9ecae1"), (1, "#fdae6b")):
            m = yt == lab
            if bool(m.any()):
                ax.scatter(
                    *_np(Xt[m]).T, s=s, c=col, alpha=0.45,
                    label=f"class {lab}", linewidths=0, zorder=1,
                )
    else:
        ax.scatter(*_np(Xt).T, s=s, c="#bbbbbb", alpha=0.45, linewidths=0, zorder=1)

    A = torch.as_tensor(target_points).detach().cpu()
    ax.scatter(
        *_np(A).T, s=s * 2.2, c="#111111", label="TARGET", linewidths=0, zorder=3,
    )

    for (name, pts), col in zip(nulls.items(), _colors(len(nulls))):
        P = torch.as_tensor(pts).detach().cpu()
        ax.scatter(
            *_np(P).T, s=s * 1.6, c=col, label=name, linewidths=0,
            alpha=0.9, zorder=2,
        )

    if bounds is not None:
        (x0, x1), (y0, y1) = bounds[0], bounds[1]
        ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            "--", c="#555555", lw=1.0, zorder=4, label="eval bounds",
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def _plot_3d(Xt, y, target_points, nulls, bounds, title, s, show):
    import plotly.graph_objects as go

    traces = []

    if y is not None:
        yt = torch.as_tensor(y).flatten()
        for lab, col in ((0, "#9ecae1"), (1, "#fdae6b")):
            m = yt == lab
            if bool(m.any()):
                P = _np(Xt[m])
                traces.append(go.Scatter3d(
                    x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                    marker=dict(size=s * 0.35, color=col, opacity=0.4),
                    name=f"class {lab}",
                ))
    else:
        P = _np(Xt)
        traces.append(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=s * 0.35, color="#bbbbbb", opacity=0.4), name="data",
        ))

    A = _np(torch.as_tensor(target_points).detach().cpu())
    traces.append(go.Scatter3d(
        x=A[:, 0], y=A[:, 1], z=A[:, 2], mode="markers",
        marker=dict(size=s * 0.55, color="#111111"), name="TARGET",
    ))

    for (name, pts), col in zip(nulls.items(), _colors(len(nulls))):
        P = _np(torch.as_tensor(pts).detach().cpu())
        traces.append(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=s * 0.45, color=col), name=name,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=32, b=0),
        legend=dict(itemsizing="constant"),
    )

    if show:
        fig.show()
    return fig
