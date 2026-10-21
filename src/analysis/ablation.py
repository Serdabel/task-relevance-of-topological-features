"""Ablation test: does removing a candidate witness hurt the task locally?

A.3 (PLAN.md): after a criterion nominates a witness, zero the neuron out
(forward hook) and measure the SELECTIVE accuracy drop — globally vs on points
near/inside the feature. A genuine witness should hurt locally; an excessive
one should not. Intersects experiment 7 (multitasking): ablating a
multi-feature witness should hurt near ALL features it witnesses.

Cheap enough to attach to many experiments, not only a dedicated one.

Note: the hooks fire on ``model.blocks[layer_id]`` module calls, i.e. on the
full ``model(X)`` forward used for classification. ``forward_until`` bypasses
the block modules (it calls .linear/.activation directly), so criteria
evaluations are NOT affected by an active ablation — by design: ablation
measures the classifier, not the field.
"""

from contextlib import contextmanager

import torch

from analysis.geometry import nearest_distances_to_set


@contextmanager
def ablate_neuron(
    model,
    layer_id: int,
    coord_id: int,
    mode: str = "zero",
):
    """
    Context manager that suppresses one hidden neuron for the duration of the
    block via a forward hook on ``model.blocks[layer_id]``.

    mode:
        "zero" — post-activation output of the neuron forced to 0;
        "mean" — forced to its mean over the current batch (softer ablation,
                 controls for shift effects).

    Hook registered on entry, ALWAYS removed on exit (try/finally); model
    weights untouched; nesting different neurons works (independent hooks).

    Usage:
        with ablate_neuron(model, 2, 0):
            acc = evaluate(...)
    """
    assert mode in {"zero", "mean"}, "mode must be 'zero' or 'mean'"
    assert 0 <= layer_id < model.n_layers, f"invalid layer_id={layer_id}"

    block = model.blocks[layer_id]
    assert 0 <= coord_id < block.linear.out_features, (
        f"invalid coord_id={coord_id}"
    )

    def hook(module, args, output):
        output = output.clone()
        if mode == "zero":
            output[:, coord_id] = 0.0
        else:
            output[:, coord_id] = output[:, coord_id].mean()
        return output

    handle = block.register_forward_hook(hook)

    try:
        yield model
    finally:
        handle.remove()


@torch.no_grad()
def region_accuracy(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_points: torch.Tensor,
    delta: float,
    device: str = "cpu",
) -> dict:
    """
    Accuracy split by distance to the feature: inside the delta-tube around A
    and outside it.

    Returns dict: acc_global, acc_near_feature, acc_far, n_near, n_far.
    """
    assert delta > 0, "delta must be positive"

    model = model.to(device)
    model.eval()

    X = X.to(device).float()
    y = y.to(device).long()
    feature_points = feature_points.to(device).float()

    logits = model(X).view(-1)
    pred = (torch.sigmoid(logits) > 0.5).long()
    correct = (pred == y).float()

    distances = nearest_distances_to_set(X, feature_points)
    near = distances <= delta
    far = ~near

    def _acc(mask):
        n = int(mask.sum().item())
        if n == 0:
            return float("nan"), 0
        return float(correct[mask].mean().item()), n

    acc_near, n_near = _acc(near)
    acc_far, n_far = _acc(far)

    return {
        "acc_global": float(correct.mean().item()),
        "acc_near_feature": acc_near,
        "acc_far": acc_far,
        "n_near": n_near,
        "n_far": n_far,
    }


def ablation_impact(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_points: torch.Tensor,
    layer_id: int,
    coord_id: int,
    delta: float,
    mode: str = "zero",
    device: str = "cpu",
) -> dict:
    """
    The A.3 measurement: region_accuracy with vs without the neuron.

    Returns dict (flat numerics, CSV-friendly):
        drop_global, drop_near_feature, drop_far,
        acc_global_before/after, acc_near_feature_before/after,
        acc_far_before/after, n_near, n_far,
        selectivity = drop_near_feature - drop_far
    """
    before = region_accuracy(
        model, X, y, feature_points, delta=delta, device=device
    )

    with ablate_neuron(model, layer_id, coord_id, mode=mode):
        after = region_accuracy(
            model, X, y, feature_points, delta=delta, device=device
        )

    drop_global = before["acc_global"] - after["acc_global"]
    drop_near = before["acc_near_feature"] - after["acc_near_feature"]
    drop_far = before["acc_far"] - after["acc_far"]

    return {
        "drop_global": drop_global,
        "drop_near_feature": drop_near,
        "drop_far": drop_far,
        "selectivity": drop_near - drop_far,
        "acc_global_before": before["acc_global"],
        "acc_global_after": after["acc_global"],
        "acc_near_feature_before": before["acc_near_feature"],
        "acc_near_feature_after": after["acc_near_feature"],
        "acc_far_before": before["acc_far"],
        "acc_far_after": after["acc_far"],
        "n_near": before["n_near"],
        "n_far": before["n_far"],
        "layer_id": layer_id,
        "coord_id": coord_id,
        "mode": mode,
        "delta": float(delta),
    }
