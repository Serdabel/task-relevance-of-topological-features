"""Extremum sampling: gradient ascent/descent to critical points (Wang-Ponce).

Serves three consumers (PLAN.md §Sampling Methods, "Extremum sampling"):

  1. Criterion 3 (``analysis.extremum_anchored``): candidate levels are the
     critical values c* = f(p*) instead of a free sweep over all c.
  2. Experiment 6 basin/ridge typing: the critical-point type of a witness.
  3. PCA-slice sampling (``analysis.subspace_slices``): the *trajectory* of
     the ascent is the ambient-space anchor cloud that the PCA subspace is
     fitted on — finding the extremum alone is NOT enough in high dimension
     (the level set near it is still (d-1)-dimensional).

All gradient operations are exact autograd through the neuron function
(``ScalarProjection`` / ``TruncatedMLPFunction``), no finite differences.

tanh caveat (PLAN.md): away from the decision boundary saturated activations
create near-zero-gradient plateaus where "criticality" is diffuse — anchors
found there may be uninformative. ``ExtremaResult.grad_norms`` exists so
downstream code can filter/flag such anchors instead of trusting them blindly.
Boundary extrema (iterates stuck on the clamp) are kept but keep a LARGE
gradient norm — the same field flags them.
"""

from dataclasses import dataclass, field

import torch

from analysis.geometry import connected_components_pointcloud


@dataclass
class ExtremaResult:
    """Output of ``find_local_extrema``.

    points:        Tensor [n_found, dim] — deduplicated critical points p*.
    values:        Tensor [n_found] — critical values f(p*).
    grad_norms:    Tensor [n_found] — |grad f(p*)| at the anchor (plateau flag).
    mode:          "max" or "min".
    trajectories:  list of Tensors [n_steps_i, dim] — full ascent/descent path
                   per restart that converged to each point (PCA-slice input).
    n_restarts:    int — how many restarts were run.
    converged:     Tensor [n_restarts] bool — which restarts converged.
    """

    points: torch.Tensor
    values: torch.Tensor
    grad_norms: torch.Tensor
    mode: str
    trajectories: list = field(default_factory=list)
    n_restarts: int = 0
    converged: torch.Tensor | None = None


def _bounds_tensors(bounds, device, dtype):
    lows = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    highs = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)
    return lows, highs


def find_local_extrema(
    function,
    bounds,
    mode: str = "max",
    n_restarts: int = 64,
    n_steps: int = 300,
    lr: float = 5e-2,
    grad_tol: float = 1e-5,
    value_tol: float = 1e-7,
    stall_patience: int = 5,
    dedupe_radius: float | None = None,
    init_points: torch.Tensor | None = None,
    record_trajectories: bool = True,
    device: str = "cpu",
    seed: int | None = None,
) -> ExtremaResult:
    """
    Finds local maxima (mode="max") or minima (mode="min") of a scalar field
    by gradient ascent/descent from many random restarts inside ``bounds``.

    Batched: all restarts are one [n_restarts, dim] tensor driven by Adam
    (a fixed-lr vanilla ascent converges too slowly on smooth caps and never
    terminates on saturated-tanh plateaus). Iterates are clamped to
    ``bounds`` each step. A restart is converged when either |grad| <=
    grad_tol, or the objective improves by less than ``value_tol`` for
    ``stall_patience`` consecutive steps — which covers interior extrema,
    boundary extrema (kept, flagged by a large grad norm) and plateaus
    (kept, flagged by a small grad norm) alike. Restarts that never satisfy
    either within n_steps are dropped (``converged``).

    ``init_points`` overrides random initialization (e.g. start the ascent
    from data points — often the right choice for data-adjacent basins).
    Deterministic in ``seed`` when given. Multiple restarts are ESSENTIAL:
    basins can be small (PROJECT.md).
    """
    assert mode in {"max", "min"}, "mode must be 'max' or 'min'"
    assert len(bounds) > 0, "bounds must be non-empty"
    assert n_steps >= 1 and lr > 0

    dtype = torch.float32
    lows, highs = _bounds_tensors(bounds, device, dtype)
    dim = len(bounds)
    diagonal = float(torch.linalg.norm(highs - lows).item())

    if dedupe_radius is None:
        dedupe_radius = 0.02 * diagonal

    if init_points is not None:
        assert init_points.ndim == 2 and init_points.shape[1] == dim, (
            f"init_points must have shape [n, {dim}]"
        )
        x = init_points.detach().to(device=device, dtype=dtype).clone()
        n_restarts = len(x)
    else:
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        u = torch.rand(n_restarts, dim, device=device, dtype=dtype, generator=generator)
        x = lows + (highs - lows) * u

    sign = 1.0 if mode == "max" else -1.0

    x = torch.clamp(x, lows, highs).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    converged = torch.zeros(n_restarts, dtype=torch.bool, device=device)
    stall_counter = torch.zeros(n_restarts, dtype=torch.long, device=device)
    grad_norms = torch.zeros(n_restarts, device=device, dtype=dtype)
    prev_values = None

    trajectories = (
        [x.detach().cpu().clone()] if record_trajectories else None
    )

    for _ in range(n_steps):
        values = function(x)
        assert values.ndim == 1, "function must map [B, dim] -> [B]"

        loss = -sign * values.sum()

        optimizer.zero_grad()
        loss.backward()

        grad_norms = x.grad.norm(dim=1).detach()

        if prev_values is not None:
            improvement = sign * (values.detach() - prev_values)
            stall_counter = torch.where(
                improvement <= value_tol,
                stall_counter + 1,
                torch.zeros_like(stall_counter),
            )
        prev_values = values.detach()

        converged |= (grad_norms <= grad_tol) | (stall_counter >= stall_patience)

        optimizer.step()

        with torch.no_grad():
            x.clamp_(lows, highs)

        if record_trajectories:
            trajectories.append(x.detach().cpu().clone())

        if bool(converged.all()):
            break

    with torch.no_grad():
        final_values = function(x)
    final_grad_norms = grad_norms

    keep = converged.nonzero(as_tuple=True)[0]

    if len(keep) == 0:
        return ExtremaResult(
            points=torch.empty(0, dim, dtype=dtype),
            values=torch.empty(0, dtype=dtype),
            grad_norms=torch.empty(0, dtype=dtype),
            mode=mode,
            trajectories=[],
            n_restarts=n_restarts,
            converged=converged.cpu(),
        )

    kept_points = x[keep].detach().cpu()
    kept_values = final_values[keep].detach().cpu()
    kept_grad_norms = final_grad_norms[keep].detach().cpu()

    # Merge near-duplicates; "best" = extreme in the search direction.
    dedupe_values = kept_values if mode == "max" else -kept_values
    kept_idx = dedupe_extrema(kept_points, dedupe_values, radius=dedupe_radius)

    result_trajectories = []
    if record_trajectories:
        stacked = torch.stack(trajectories, dim=0)  # [n_steps+1, n_restarts, dim]
        for i in kept_idx.tolist():
            restart_id = int(keep[i].item())
            result_trajectories.append(stacked[:, restart_id, :].clone())

    return ExtremaResult(
        points=kept_points[kept_idx],
        values=kept_values[kept_idx],
        grad_norms=kept_grad_norms[kept_idx],
        mode=mode,
        trajectories=result_trajectories,
        n_restarts=n_restarts,
        converged=converged.cpu(),
    )


def dedupe_extrema(
    points: torch.Tensor,
    values: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """
    Merges near-duplicate critical points: clusters ``points`` with the
    radius graph (``analysis.geometry.connected_components_pointcloud``) and
    keeps one representative per cluster — the LARGEST ``values`` entry
    (pass ``-values`` when hunting minima).

    Returns LongTensor of kept indices into ``points``.
    """
    assert points.ndim == 2, "points must have shape [N, dim]"
    assert len(points) == len(values), "points and values must match"

    if len(points) == 0:
        return torch.empty(0, dtype=torch.long)

    components = connected_components_pointcloud(
        points, radius=max(float(radius), 1e-12), min_size=1
    )
    labels = components["labels"]

    kept = []
    for component_id in components["component_ids"]:
        member_idx = (labels == component_id).nonzero(as_tuple=True)[0]
        best_local = torch.argmax(values[member_idx])
        kept.append(int(member_idx[best_local].item()))

    return torch.tensor(sorted(kept), dtype=torch.long)


def critical_values(
    extrema: ExtremaResult,
    merge_tol: float = 1e-3,
) -> torch.Tensor:
    """
    Unique candidate levels c* = f(p*) from an ExtremaResult, merging values
    closer than ``merge_tol`` (a saddle ridge along a whole feature yields a
    continuum of near-equal critical values — treat them as one level).

    Returns 1D Tensor of levels, sorted ascending.
    """
    values = extrema.values

    if len(values) == 0:
        return torch.empty(0, dtype=torch.float32)

    sorted_values = torch.sort(values.flatten()).values

    merged = [float(sorted_values[0].item())]
    group = [float(sorted_values[0].item())]

    for v in sorted_values[1:]:
        v = float(v.item())
        if v - group[-1] <= merge_tol:
            group.append(v)
            merged[-1] = sum(group) / len(group)
        else:
            group = [v]
            merged.append(v)

    return torch.tensor(merged, dtype=values.dtype)
