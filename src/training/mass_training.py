import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from models.mlp import MLP
from management.experiment_management import (
    load_dataset_instance,
    store_model_instance,
    name_to_activation,
)




def resolve_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        return torch.device(device)

    return device


def resolve_activation_cls(activation):
    """
    activation can be:
        "tanh", "relu", "sigmoid", "leaky_relu"
        or nn.Module class, e.g. nn.Tanh
    """
    if isinstance(activation, str):
        return name_to_activation(activation)

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation

    raise ValueError(
        "activation must be a string or nn.Module class, "
        "e.g. 'tanh' or nn.Tanh"
    )


def activation_to_short_name(activation):
    if isinstance(activation, str):
        return activation

    name = activation.__name__.lower()

    if name == "tanh":
        return "tanh"
    if name == "relu":
        return "relu"
    if name == "sigmoid":
        return "sigmoid"
    if name == "leakyrelu":
        return "leaky_relu"

    return name


def make_optimizer(
    model,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_kwargs: dict | None = None,
):
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        )

    raise ValueError(f"Unknown optimizer_name: {optimizer_name}")


@torch.no_grad()
def evaluate_binary_classifier_weighted(
    model,
    data_loader,
    device,
):
    """
    Computes loss and accuracy weighted by sample count, not by batch count.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        y_float = y.float()
        logits = model(X).view(-1)

        loss_sum = F.binary_cross_entropy_with_logits(
            logits,
            y_float,
            reduction="sum",
        )

        pred_labels = (torch.sigmoid(logits) > 0.5).long()

        total_loss += float(loss_sum.item())
        total_correct += int((pred_labels == y.long()).sum().item())
        total_n += int(y.numel())

    if total_n == 0:
        return float("nan"), float("nan")

    return total_loss / total_n, total_correct / total_n


def train_one_mlp_until_accuracy(
    model,
    train_loader,
    test_loader,
    optimizer,
    n_epochs: int = 300,
    target_test_acc: float = 0.99,
    patience: int = 5,
    min_epochs: int = 50,
    device=None,
    verbose: bool = True,
    progress_position: int = 1,
    progress_leave: bool = False,
    progress_desc: str = "epochs",

    # Overtraining hooks (experiment 2, task E). The defaults reproduce the
    # gate loop exactly, so every existing caller is unaffected.
    stop_at_gate: bool = True,
    checkpoint_multiples=None,
    on_checkpoint=None,
):
    """
    Trains one MLP until either:

        test accuracy >= target_test_acc for `patience` consecutive epochs
        after `min_epochs`

    or until n_epochs is exhausted.

    ``stop_at_gate=False`` keeps training past the gate instead of breaking.
    The gate epoch is still recorded in ``stopped_epoch``, so such a run stays
    comparable with a gate-stopped one; what changes is only that the
    trajectory continues. That is what task E needs: the early and the late
    state have to come from ONE run, because training sets no seed, so a second
    run of the same architecture is a different trajectory rather than a longer
    one.

    ``checkpoint_multiples`` are rungs RELATIVE to this run's own gate epoch
    (``(1.0, 3.0, 10.0)`` -> gate, 3x gate, 10x gate). Relative and not
    absolute because gate epochs differ by an order of magnitude between cells,
    so one absolute epoch lands before the gate in one network and far past it
    in another. A rung at or beyond the last epoch is dropped rather than
    quietly merged into it; the last epoch is always offered as ``end``.

    ``on_checkpoint(rung, epoch, snapshot)`` is called with the model in its
    state AT THAT EPOCH, and ``snapshot`` carrying that epoch's losses and
    accuracies. The callee must copy whatever it needs out of the model before
    returning, because training continues on the same object.

    Returns:
        metrics dict.
    """
    assert n_epochs > 0, "n_epochs must be positive"
    assert 0.0 <= target_test_acc <= 1.0, "target_test_acc must be in [0, 1]"
    assert patience >= 1, "patience must be >= 1"
    assert min_epochs >= 0, "min_epochs must be >= 0"

    device = resolve_device(device)
    model.to(device)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_test_acc = -float("inf")
    best_test_loss = float("inf")
    best_epoch = None

    good_epochs = 0
    reached_target = False
    stopped_epoch = None

    checkpoints = []      # [{"rung", "epoch"}] actually taken, in order
    scheduled = {}        # epoch -> rung; filled in the moment the gate fires
    last_epoch = n_epochs - 1

    def _take(rung, epoch):
        checkpoints.append({"rung": rung, "epoch": epoch})
        if on_checkpoint is not None:
            on_checkpoint(rung, epoch, {
                "train_loss": train_losses[-1], "train_acc": train_accs[-1],
                "test_loss": test_losses[-1], "test_acc": test_accs[-1],
            })

    with tqdm(
        total=n_epochs,
        desc=progress_desc,
        leave=progress_leave,
        position=progress_position,
        dynamic_ncols=True,
        disable=not verbose,
    ) as epoch_bar:

        for epoch in range(n_epochs):
            model.train()

            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device)

                y_float = y.float()
                logits = model(X).view(-1)

                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    y_float,
                    reduction="mean",
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss, train_acc = evaluate_binary_classifier_weighted(
                model,
                train_loader,
                device,
            )

            test_loss, test_acc = evaluate_binary_classifier_weighted(
                model,
                test_loader,
                device,
            )

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_loss = test_loss
                best_epoch = epoch

            if epoch >= min_epochs and test_acc >= target_test_acc:
                good_epochs += 1
            else:
                good_epochs = 0

            if verbose:
                epoch_bar.set_postfix(
                    {
                        "train_acc": f"{train_acc:.4f}",
                        "test_acc": f"{test_acc:.4f}",
                        "best": f"{best_test_acc:.4f}",
                        "good": good_epochs,
                    }
                )

            epoch_bar.update(1)

            if good_epochs >= patience and not reached_target:
                reached_target = True
                stopped_epoch = epoch

                if stop_at_gate:
                    break

                # The ladder is knowable only now: it hangs off this run's gate.
                for multiple in (checkpoint_multiples or ()):
                    target = int(round(multiple * (epoch + 1))) - 1
                    if epoch <= target < last_epoch:
                        scheduled.setdefault(target, f"x{multiple:g}")

            if reached_target and not stop_at_gate:
                if epoch in scheduled:
                    _take(scheduled[epoch], epoch)
                if epoch == last_epoch:
                    _take("end", epoch)

    if stopped_epoch is None:
        stopped_epoch = n_epochs - 1

    metrics = {
        "reached_target": reached_target,
        "target_test_acc": target_test_acc,
        "patience": patience,
        "min_epochs": min_epochs,
        "stopped_epoch": stopped_epoch,
        "n_epochs_requested": n_epochs,
        "stop_at_gate": stop_at_gate,
        "checkpoints": checkpoints,

        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_test_loss": test_losses[-1],
        "final_test_acc": test_accs[-1],

        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,

        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
    }

    return metrics


def train_many_mlps_on_dataset_instance(
    dataset_instance_path,
    n_models: int = 10,

    # architecture
    activation="tanh",
    hidden_dim: int = 8,
    n_layers: int = 5,
    out_dim: int = 1,
    include_output_layer: bool = True,

    # training
    target_test_acc: float = 0.99,
    n_epochs: int = 300,
    min_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    optimizer_kwargs: dict | None = None,

    # dataloader/device
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    device=None,

    # saving
    save_successful_only: bool = True,
    model_name: str | None = None,
    stop_after_successful: int | None = None,

    # logging
    verbose: bool = True,
    log_each_model: bool = False,
):
    """
    Trains many MLPs on one stored dataset instance.

    Model weights are initialized by the default PyTorch random initialization.
    No manual seed is set inside this function.

    Success means:

        test_acc >= target_test_acc for `patience` consecutive epochs
        after `min_epochs`.

    ``n_models`` is the ATTEMPT budget, and every successful attempt is written
    to disk the moment it succeeds -- so an interrupted run keeps everything it
    had already earned, and re-running tops the ensemble up instead of starting
    over.

    ``stop_after_successful`` turns the attempt budget into a ceiling rather
    than a plan: the loop stops as soon as that many models have been saved in
    this call. This is what makes the server path robust without guessing a
    success rate in advance -- ask for a hundred attempts, stop at fifty
    successes, and a cell that exhausts the budget is telling you something
    about the task rather than about the schedule.

    Returns:
        run_summaries: list[dict]
    """
    assert n_models > 0, "n_models must be positive"
    assert n_epochs > 0, "n_epochs must be positive"
    assert stop_after_successful is None or stop_after_successful > 0, (
        "stop_after_successful must be positive"
    )

    device = resolve_device(device)

    if num_workers is None:
        num_workers = 4 if os.cpu_count() is not None and os.cpu_count() > 4 else 1

    if pin_memory is None:
        pin_memory = device.type == "cuda"

    loaded = load_dataset_instance(
        dataset_instance_path,
        map_location="cpu",
        reconstruct_data=False,
    )

    train_dataset = loaded["train_dataset"]
    test_dataset = loaded["test_dataset"]
    X = loaded["X"]

    assert X.ndim == 2, "X must have shape [n_samples, in_dim]"
    in_dim = X.shape[1]

    activation_cls = resolve_activation_cls(activation)
    activation_name = activation_to_short_name(activation)

    if model_name is None:
        model_name = f"{activation_name}_w{hidden_dim}_d{n_layers}"

    train_config = {
        "target_test_acc": target_test_acc,
        "n_epochs": n_epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "optimizer_kwargs": optimizer_kwargs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,

        "activation": activation_name,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "out_dim": out_dim,
        "include_output_layer": include_output_layer,
    }

    run_summaries = []

    with tqdm(
        total=n_models,
        desc="models",
        leave=True,
        position=0,
        dynamic_ncols=True,
        disable=not verbose,
    ) as model_bar:

        for attempt_idx in range(n_models):
            n_saved_before = sum(s["saved"] for s in run_summaries)

            if verbose:
                model_bar.set_postfix(
                    {
                        "saved": n_saved_before,
                    }
                )
                model_bar.refresh()

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            model = MLP(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                out_dim=out_dim,
                activation_cls=activation_cls,
                include_output_layer=include_output_layer,
            ).to(device)

            optimizer = make_optimizer(
                model=model,
                optimizer_name=optimizer_name,
                lr=lr,
                weight_decay=weight_decay,
                optimizer_kwargs=optimizer_kwargs,
            )

            metrics = train_one_mlp_until_accuracy(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                n_epochs=n_epochs,
                target_test_acc=target_test_acc,
                patience=patience,
                min_epochs=min_epochs,
                device=device,
                verbose=verbose,
                progress_position=1,
                progress_leave=False,
                progress_desc=f"model {attempt_idx + 1}/{n_models}",
            )

            should_save = metrics["reached_target"] or not save_successful_only

            saved_path = None

            if should_save:
                cur_metrics = dict(metrics)
                cur_metrics["saved"] = True

                saved_path = store_model_instance(
                    dataset_instance_path=dataset_instance_path,
                    model=model,
                    model_id=None,
                    model_name=model_name,
                    train_config=train_config,
                    metrics=cur_metrics,
                    extra={
                        "dataset_metadata": loaded.get("metadata", {}),
                    },
                )

            summary = {
                "dataset_instance_path": str(dataset_instance_path),

                "saved": saved_path is not None,
                "model_path": saved_path,
                "model_name": model_name,

                "reached_target": metrics["reached_target"],
                "target_test_acc": target_test_acc,
                "stopped_epoch": metrics["stopped_epoch"],
                "best_epoch": metrics["best_epoch"],

                "final_train_acc": metrics["final_train_acc"],
                "final_test_acc": metrics["final_test_acc"],
                "best_test_acc": metrics["best_test_acc"],

                "final_train_loss": metrics["final_train_loss"],
                "final_test_loss": metrics["final_test_loss"],
                "best_test_loss": metrics["best_test_loss"],

                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "activation": activation_name,
                "optimizer_name": optimizer_name,
                "lr": lr,
                "batch_size": batch_size,
            }

            run_summaries.append(summary)

            n_saved_after = sum(s["saved"] for s in run_summaries)

            if verbose:
                model_bar.set_postfix(
                    {
                        "saved": n_saved_after,
                        "last_acc": f"{metrics['final_test_acc']:.4f}",
                        "best": f"{metrics['best_test_acc']:.4f}",
                    }
                )

            if stop_after_successful is not None and n_saved_after >= stop_after_successful:
                model_bar.update(1)
                if verbose:
                    print(f"[обучение] цель достигнута: {n_saved_after} моделей "
                          f"за {attempt_idx + 1} попыток")
                break

            if verbose and log_each_model:
                status = "saved" if saved_path is not None else "not saved"
                model_bar.write(
                    f"attempt {attempt_idx + 1}/{n_models} | "
                    f"{status} | "
                    f"final_test_acc={metrics['final_test_acc']:.4f} | "
                    f"best_test_acc={metrics['best_test_acc']:.4f} | "
                    f"stopped_epoch={metrics['stopped_epoch']}"
                )
                model_bar.refresh()

            model_bar.update(1)

    return run_summaries