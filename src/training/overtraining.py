"""Task E of experiment 2: one run per model, several checkpoints of that run.

WHY THIS EXISTS AT ALL. The training gate stops a network the moment it first
holds 100 % test accuracy for ``patience`` epochs, and on this separable task
that moment arrives the sooner the wider the network is. Measured on the
experiment 2 pool, ``simple_relevant`` at depth 5: median 124 epochs at width 3
against 13 at width 20. The width axis of tasks A and B is therefore ALSO a
training-duration axis, running the other way and spanning a factor of nine, so
every width curve there can be read as a duration curve instead.

WHY IT CANNOT BE DONE BY LOOKING. Within a cell the Spearman correlation
between stop epoch and best score changes sign across cells (+0.70 at w3 d5,
-0.50 at w5 d2). A model that needed 1861 epochs is a model whose
initialisation made the problem hard, so the correlation reads equally well as
"trained longer, ended worse" and as "worse runs take longer". Observation
cannot separate the two.

WHY THE CHECKPOINTS HAVE TO COME FROM ONE RUN. ``train_many_mlps_on_dataset_instance``
sets no seed. Retraining the same architecture gives a different trajectory,
not a longer one, so an early ensemble and a late ensemble would differ by
initialisation as well as by duration. Taking both states from ONE run makes
the comparison within-model: same initialisation, same trajectory, seed
variance gone from the difference entirely. What is left is duration.

HOW IT IS STORED. Every rung is written as an ordinary model instance under its
own tag ``{act}_w{W}_d{D}_ot{rung}``, so the evaluation path needs no changes
whatsoever -- a rung is just another ensemble.

MODEL IDS DO NOT LINE UP BETWEEN RUNGS, and the pairing must not assume they
do. ``store_model_instance`` takes the first free id inside each rung's OWN name
space, while an upper rung is missing for any run whose gate came too late for
it to fit in the epoch budget -- so one such run shifts every later id of that
rung by one. Measured on the finished data: at ``w3 d5`` on simple_relevant only
2 of 38 indices happened to match between ``x1`` and ``x10``. ``x1`` and ``end``
DO line up, because every successful run produces both.

Each rung therefore records ``run_index``, and ``analysis.arch_sweep`` keys the
pairing on it. For data written before that field existed, run order is
recoverable exactly: within a rung the models are stored in run order, so the
gate-epoch sequence of an upper rung is a SUBSEQUENCE of the ``x1`` one and a
two-pointer walk reconstructs the mapping (verified on all ten cells).
"""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.mlp import MLP
from management.experiment_management import (
    load_dataset_instance,
    store_model_instance,
)
from training.mass_training import (
    activation_to_short_name,
    make_optimizer,
    resolve_activation_cls,
    resolve_device,
    train_one_mlp_until_accuracy,
)

DEFAULT_MULTIPLES = (1.0, 3.0, 10.0)


def rung_tag(model_name_base: str, rung: str) -> str:
    """Ensemble tag of one rung: ``tanh_w5_d5`` + ``x10`` -> ``tanh_w5_d5_otx10``."""
    return f"{model_name_base}_ot{rung}"


def saturation_report(model, X, device=None, threshold: float = 0.99) -> dict:
    """How saturated the hidden units are on the data, and how big the weights got.

    Recorded at every rung because the criterion scans 101 levels spread over the
    field's OBSERVED range, and overtraining moves that range: weights keep
    growing on a separable task and tanh units drift towards +-1. A score that
    moves under overtraining therefore has to be read together with how much the
    field's own scale moved, rather than on its own.
    """
    device = resolve_device(device)
    model = model.to(device)
    X = X.to(device)

    with torch.no_grad():
        saturated, sizes = [], []
        for layer_id in range(model.n_layers):
            post = model.forward_until(X, layer_id=layer_id, kind="post")
            saturated.append(float((post.abs() >= threshold).float().mean()))
            sizes.append(float(post.abs().mean()))
        weight_norm = float(torch.sqrt(sum(
            (p.detach() ** 2).sum() for p in model.parameters())))
        logit = model(X).view(-1)

    return {
        "weight_norm": weight_norm,
        "saturated_fraction": float(sum(saturated) / max(len(saturated), 1)),
        "saturated_fraction_per_layer": saturated,
        "post_abs_mean_per_layer": sizes,
        "logit_abs_mean": float(logit.abs().mean()),
        "logit_abs_max": float(logit.abs().max()),
        "saturation_threshold": threshold,
    }


def train_overtrained_ensemble(
    dataset_instance_path,
    n_models: int = 50,

    # architecture
    activation="tanh",
    hidden_dim: int = 5,
    n_layers: int = 5,
    out_dim: int = 1,
    include_output_layer: bool = True,

    # the gate, verbatim from the spec -- it decides the FIRST rung, not the end
    target_test_acc: float = 1.0,
    min_epochs: int = 0,
    patience: int = 5,

    # how far past the gate to go, and where to look on the way
    n_epochs: int = 3000,
    checkpoint_multiples=DEFAULT_MULTIPLES,

    # optimisation
    batch_size: int = 64,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    optimizer_kwargs: dict | None = None,

    device=None,
    model_name: str | None = None,
    stop_after_successful: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Train ``n_models`` runs to ``n_epochs``, storing every rung of each run.

    ``model_name`` is the BASE tag; each rung is stored under ``rung_tag(base,
    rung)``. Runs that never reach the gate are stored nowhere -- without a gate
    epoch there is no ladder to hang. Every stored rung carries ``run_index``,
    which is the only safe key for pairing rungs (see the module docstring).

    Returns one summary per attempt.
    """
    assert n_models > 0, "n_models must be positive"
    assert n_epochs > 0, "n_epochs must be positive"

    device = resolve_device(device)
    activation_cls = resolve_activation_cls(activation)
    activation_name = activation_to_short_name(activation)
    if model_name is None:
        model_name = f"{activation_name}_w{hidden_dim}_d{n_layers}"

    loaded = load_dataset_instance(
        dataset_instance_path, map_location="cpu", reconstruct_data=False,
    )
    train_dataset, test_dataset, X = (
        loaded["train_dataset"], loaded["test_dataset"], loaded["X"])
    in_dim = X.shape[1]

    train_config = {
        "target_test_acc": target_test_acc, "n_epochs": n_epochs,
        "min_epochs": min_epochs, "patience": patience,
        "batch_size": batch_size, "lr": lr,
        "optimizer_name": optimizer_name, "weight_decay": weight_decay,
        "optimizer_kwargs": optimizer_kwargs,
        "activation": activation_name, "hidden_dim": hidden_dim,
        "n_layers": n_layers, "out_dim": out_dim,
        "include_output_layer": include_output_layer,
    }

    summaries = []

    for attempt in tqdm(range(n_models), desc="runs", disable=not verbose,
                        dynamic_ncols=True):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MLP(
            in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            out_dim=out_dim, activation_cls=activation_cls,
            include_output_layer=include_output_layer,
        ).to(device)
        optimizer = make_optimizer(
            model=model, optimizer_name=optimizer_name, lr=lr,
            weight_decay=weight_decay, optimizer_kwargs=optimizer_kwargs,
        )

        snapshots = []

        def _snapshot(rung, epoch, state, _model=model):
            # Copy now: training continues on this very object.
            snapshots.append({
                "rung": rung, "epoch": epoch,
                "state_dict": copy.deepcopy(_model.state_dict()),
                "saturation": saturation_report(_model, X, device=device),
                **state,
            })

        metrics = train_one_mlp_until_accuracy(
            model=model, train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer, n_epochs=n_epochs,
            target_test_acc=target_test_acc, patience=patience,
            min_epochs=min_epochs, device=device, verbose=False,
            stop_at_gate=False, checkpoint_multiples=checkpoint_multiples,
            on_checkpoint=_snapshot,
        )

        if not metrics["reached_target"]:
            summaries.append({
                "attempt": attempt, "reached_target": False, "saved_rungs": [],
                "gate_epoch": None, "model_name": model_name,
            })
            continue

        saved = []
        for snapshot in snapshots:
            model.load_state_dict(snapshot["state_dict"])
            rung_metrics = {
                k: v for k, v in metrics.items() if k != "checkpoints"
            }
            rung_metrics.update({
                "saved": True,
                # The run this rung came from. Recorded explicitly because model
                # ids do NOT line up between rungs: store_model_instance takes
                # the first free id inside each rung's own name space, so a run
                # whose upper rung fell outside the epoch budget shifts every
                # later id in that rung by one. Without this the pairing has to
                # be reconstructed from gate_epoch afterwards.
                "run_index": attempt,
                "rung": snapshot["rung"],
                "rung_epoch": snapshot["epoch"],
                "gate_epoch": metrics["stopped_epoch"],
                # The pipeline reads `train_stopped_epoch` from here into the
                # per-neuron CSV; for a rung it must mean the age of THIS state,
                # not the gate, or every rung would report the same duration.
                "stopped_epoch": snapshot["epoch"],
                "final_train_loss": snapshot["train_loss"],
                "final_train_acc": snapshot["train_acc"],
                "final_test_loss": snapshot["test_loss"],
                "final_test_acc": snapshot["test_acc"],
                "saturation": snapshot["saturation"],
                "checkpoint_multiples": list(checkpoint_multiples or ()),
            })
            path = store_model_instance(
                dataset_instance_path=dataset_instance_path,
                model=model, model_id=None,
                model_name=rung_tag(model_name, snapshot["rung"]),
                train_config=train_config, metrics=rung_metrics,
                extra={"dataset_metadata": loaded.get("metadata", {})},
            )
            saved.append({"rung": snapshot["rung"], "epoch": snapshot["epoch"],
                          "path": path})

        # Leave the object in its final state; the loop rebuilds it anyway.
        model.load_state_dict(snapshots[-1]["state_dict"])

        summaries.append({
            "attempt": attempt, "reached_target": True,
            "gate_epoch": metrics["stopped_epoch"],
            "rungs": [s["rung"] for s in snapshots],
            "rung_epochs": [s["epoch"] for s in snapshots],
            "saved_rungs": saved, "model_name": model_name,
            "final_test_acc": metrics["final_test_acc"],
        })

        # The attempt budget is a ceiling, not a plan: stop as soon as the
        # ensemble is full (see scripts/train_cell.py for the reasoning).
        n_done = sum(s["reached_target"] for s in summaries)
        if stop_after_successful is not None and n_done >= stop_after_successful:
            if verbose:
                print(f"[переобучение] цель достигнута: {n_done} прогонов "
                      f"за {attempt + 1} попыток")
            break

    return summaries
