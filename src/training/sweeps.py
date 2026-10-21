"""Architecture sweeps: ensembles over a width x depth x activation grid.

Experiment 2 (excessive architecture) and A.5 (robustness grid) both need
many ensembles that differ only in architecture. Each grid cell delegates to
``train_many_mlps_on_dataset_instance``; runs of different cells coexist in
the same dataset-instance directory because ``model_name`` encodes the
architecture ("{activation}_w{width}_d{depth}"), which is how downstream
stages separate them (``model_name`` CSV column / ``model_name_filter``).
"""

from itertools import product
from pathlib import Path

import pandas as pd

from training.mass_training import train_many_mlps_on_dataset_instance


def architecture_model_name(activation: str, width: int, depth: int) -> str:
    """The canonical architecture tag used in run directory names."""
    return f"{activation}_w{width}_d{depth}"


def train_architecture_grid(
    dataset_instance_path,
    widths: list[int],
    depths: list[int],
    activations: list[str] = ("tanh",),
    n_models: int = 50,
    summary_csv_path: str | Path | None = None,
    verbose: bool = True,
    **train_kwargs,
):
    """
    Trains an ensemble for every (width, depth, activation) grid cell.

    train_kwargs are forwarded to ``train_many_mlps_on_dataset_instance``
    (target_test_acc, n_epochs, lr, batch_size, device, ...).

    Note for experiment 2: the minimal architecture is width=1, depth=1 —
    a single hidden neuron suffices for the linearly separable task, so the
    grid should normally include (1, 1) as the lower calibration cell.

    Returns pd.DataFrame with one row per trained model (the union of the
    per-cell run summaries, plus width/depth/activation columns), also written
    to ``summary_csv_path`` if given.
    """
    all_summaries = []

    for activation, width, depth in product(activations, widths, depths):
        model_name = architecture_model_name(activation, width, depth)

        if verbose:
            print(f"=== grid cell: {model_name}, n_models={n_models} ===")

        summaries = train_many_mlps_on_dataset_instance(
            dataset_instance_path,
            n_models=n_models,
            activation=activation,
            hidden_dim=width,
            n_layers=depth,
            model_name=model_name,
            verbose=verbose,
            **train_kwargs,
        )

        for summary in summaries:
            summary = dict(summary)
            summary["grid_width"] = width
            summary["grid_depth"] = depth
            summary["grid_activation"] = activation
            all_summaries.append(summary)

    df = pd.DataFrame(all_summaries)

    if summary_csv_path is not None:
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_csv_path, index=False)

    return df
