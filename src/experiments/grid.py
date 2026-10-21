"""Grid runs: sweep several knobs at once inside one experiment.

An ``ExperimentSpec`` is a single point in configuration space along two axes
(datasets x architectures) — its ``TrainSpec`` is scalar, so "the same
architecture trained at two learning rates" cannot be expressed as one spec.
This module builds the cartesian product over ALL knobs and lays it out so
the existing four stages still run unchanged:

  * every cell that differs in a TRAINING knob gets its own spec (one
    ``TrainSpec`` each) but shares the results id, so everything lands in one
    results directory;
  * every cell is tagged through ``ArchSpec.tag``, which is what separates
    model pools inside a dataset instance and what names the eval CSV — so
    cells never mix and stay distinguishable downstream;
  * ``grid_manifest.csv`` maps each tag back to the full configuration that
    produced it, which is how stage-4 analyses recover the swept knobs (the
    per-neuron CSV carries architecture, not learning rate).

The knob convention is deliberate and uniform:

    a LIST means "sweep these values"; anything else is a single value.

That distinction matters because several knobs are naturally tuple-valued —
``circle_axes=(1.6, 1.0)`` is one ellipse, not two numbers to sweep. To sweep
tuples, nest them in a list: ``[(1.6, 1.0), (1.5, 1.5)]``.

Only knobs that require RETRAINING are swept here (dataset geometry,
architecture, training hyperparameters). Evaluation knobs (resolution,
criteria, thresholds) are one configuration per grid run: they need no
retraining, so comparing them is a separate, much cheaper run.

Typical use (see notebooks/experiments/*_mass.ipynb)::

    plan = build_grid_plan(
        EXPERIMENTS["exp1_sanity_check"],
        run_tag="sweep1",
        knobs={"width": [7, 15], "lr": [1e-3, 3e-4], "dataset_overrides": [
            {"label": "canonical"},
            {"label": "ellipse", "circle_axes": (1.6, 1.0)},
        ]},
    )
    save_grid(plan)
    run_grid(plan)
"""

import re
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

from experiments.configs import (
    ArchSpec,
    DatasetSpec,
    ExperimentSpec,
    save_spec,
)

# knob name -> TrainSpec field (the notebook-facing names follow
# ExperimentLab's, which are shorter than the dataclass field names)
TRAIN_KNOBS = {
    "n_models": "n_models",
    "target_acc": "target_test_acc",
    "n_epochs": "n_epochs",
    "min_epochs": "min_epochs",
    "patience": "patience",
    "batch_size": "batch_size",
    "lr": "lr",
    "optimizer": "optimizer_name",
    "weight_decay": "weight_decay",
}

ARCH_KNOBS = ("width", "depth", "activation")
DATA_KNOBS = ("dataset_overrides",)

# compact, readable abbreviations for the run tag
_SHORT = {
    "lr": "lr", "batch_size": "bs", "n_epochs": "ep", "min_epochs": "minep",
    "patience": "pat", "target_acc": "acc", "optimizer": "opt",
    "weight_decay": "wd", "n_models": "n",
}

KNOWN_KNOBS = tuple(TRAIN_KNOBS) + ARCH_KNOBS + DATA_KNOBS


def _values(value) -> list:
    """A list means "sweep these"; anything else is one value.

    Tuples are values on purpose — circle_axes=(1.6, 1.0) is one ellipse.
    """
    return list(value) if isinstance(value, list) else [value]


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "-".join(_fmt(v) for v in value)
    return str(value)


def _sanitize(text: str) -> str:
    """Filesystem- and CSV-safe fragment for directory and column names."""
    return re.sub(r"[^0-9A-Za-z.+-]+", "-", str(text)).strip("-")


def expand_grid(knobs: dict) -> list[dict]:
    """Cartesian product over the list-valued knobs, in declaration order."""
    keys = list(knobs)
    combos = product(*(_values(knobs[k]) for k in keys))
    return [dict(zip(keys, values)) for values in combos]


def swept_knobs(knobs: dict) -> list[str]:
    """The knobs that actually vary (a one-element list does not)."""
    return [k for k, v in knobs.items() if len(_values(v)) > 1]


def _dataset_label(overrides: dict) -> str:
    """Explicit "label" wins; otherwise derive one from the override values."""
    overrides = dict(overrides)
    label = overrides.pop("label", None)

    if label is not None:
        return _sanitize(label)
    if not overrides:
        return ""

    return _sanitize("_".join(f"{k}{_fmt(v)}" for k, v in sorted(overrides.items())))


def _dataset_spec(base: DatasetSpec, overrides, n_null_features: int | None):
    """A DatasetSpec for one cell.

    Recipe overrides always produce their own instance name, because
    prepare_datasets keys on the NAME and skips a dataset whose instance
    already exists — reusing a name with new params would silently train on
    the previous variant's data. An entry with no recipe overrides (``{}`` or
    just a ``label``) is the canonical instance, reused as is.
    """
    overrides = dict(overrides or {})
    label = _dataset_label(overrides)
    overrides.pop("label", None)

    spec = base
    if n_null_features is not None:
        spec = replace(spec, n_null_features=n_null_features)

    if not overrides:
        return spec

    return replace(
        spec,
        name=f"{base.name}__{label}",
        params={**base.params, **overrides},
    )


@dataclass(frozen=True)
class GridPlan:
    """One grid run: the cells, the specs that realise them, the manifest."""

    run_tag: str
    results_id: str
    knobs: dict
    swept: tuple[str, ...]
    cells: tuple[dict, ...]
    train_specs: tuple[ExperimentSpec, ...]
    eval_spec: ExperimentSpec

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_models_total(self) -> int:
        return sum(c["n_models"] for c in self.cells)

    @property
    def n_neuron_evals(self) -> int:
        """Neuron-feature evaluations, the quantity the eval stage costs."""
        return sum(
            c["width"] * c["depth"] * c["n_models"] * c["n_features"]
            for c in self.cells
        )

    @property
    def results_dir(self) -> Path:
        return Path(self.eval_spec.results_dir)

    def manifest(self):
        """One row per cell: its tag, its dataset, and every knob value.

        This is the join key between the per-neuron CSVs (which carry
        dataset_type and model_name) and the knobs that are not stored in
        them — learning rate, batch size, epoch budget.
        """
        import pandas as pd

        return pd.DataFrame(list(self.cells))


def build_grid_plan(
    base_spec: ExperimentSpec,
    run_tag: str,
    knobs: dict,
    n_null_features: int | None = None,
    eval_overrides: dict | None = None,
    analyses: tuple[str, ...] | None = None,
) -> GridPlan:
    """Expands ``knobs`` into a grid over ``base_spec``."""
    unknown = set(knobs) - set(KNOWN_KNOBS)
    assert not unknown, (
        f"unknown knobs {sorted(unknown)}; sweepable: {sorted(KNOWN_KNOBS)}. "
        "Evaluation knobs are fixed per grid run — see the module docstring."
    )

    has_arch_knobs = any(k in knobs for k in ARCH_KNOBS)
    base_arch_default = base_spec.archs[0]

    defaults = {
        "dataset_overrides": {},
        **{k: getattr(base_spec.train, field) for k, field in TRAIN_KNOBS.items()},
    }
    if has_arch_knobs:
        defaults.update({
            "width": base_arch_default.width,
            "depth": base_arch_default.depth,
            "activation": base_arch_default.activation,
        })
    resolved = {**defaults, **knobs}

    swept = tuple(swept_knobs(resolved))
    swept_train = [k for k in swept if k in TRAIN_KNOBS]

    results_id = f"{base_spec.id}__{_sanitize(run_tag)}"

    eval_spec_fields = base_spec.eval
    if eval_overrides:
        eval_spec_fields = replace(eval_spec_fields, **eval_overrides)

    cells = []
    by_train_tag: dict[str, dict] = {}

    for base_dataset in base_spec.datasets:
        for combo in expand_grid(resolved):
            arch_list = (
                [ArchSpec(width=combo["width"], depth=combo["depth"], activation=combo["activation"])]
                if has_arch_knobs
                else base_spec.archs
            )
            for base_arch in arch_list:
                train_bits = [
                    f"{_SHORT.get(k, k)}{_fmt(combo[k])}" for k in swept_train
                ]
                tag = "_".join([_sanitize(run_tag)] + [_sanitize(b) for b in train_bits])

                arch = ArchSpec(
                    width=base_arch.width,
                    depth=base_arch.depth,
                    activation=base_arch.activation,
                    tag=tag,
                )
                dataset = _dataset_spec(
                    base_dataset, combo["dataset_overrides"], n_null_features
                )
                train = replace(
                    base_spec.train,
                    **{field: combo[k] for k, field in TRAIN_KNOBS.items()},
                )

                group = by_train_tag.setdefault(
                    tag, {"train": train, "datasets": {}, "archs": {}}
                )
                group["datasets"][dataset.name] = dataset
                group["archs"][arch.model_name] = arch

                cells.append(
                    {
                        "cell": len(cells),
                        "dataset_type": dataset.name,
                        "model_name": arch.model_name,
                        "run_tag": tag,
                        "width": base_arch.width,
                        "depth": base_arch.depth,
                        "activation": base_arch.activation,
                        "n_features": 1 + (
                            n_null_features
                            if n_null_features is not None
                            else dataset.n_null_features
                        ),
                        **{k: combo[k] for k in TRAIN_KNOBS if k in combo},
                        "dataset_overrides": str(combo["dataset_overrides"]),
                    }
                )

    train_specs = tuple(
        replace(
            base_spec,
            id=results_id,
            datasets=tuple(group["datasets"].values()),
            archs=tuple(group["archs"].values()),
            train=group["train"],
            eval=eval_spec_fields,
            notes=f"[grid {run_tag} cell-group {tag}] {base_spec.notes}",
        )
        for tag, group in by_train_tag.items()
    )

    all_datasets, all_archs = {}, {}
    for group in by_train_tag.values():
        all_datasets.update(group["datasets"])
        all_archs.update(group["archs"])

    eval_spec = replace(
        base_spec,
        id=results_id,
        datasets=tuple(all_datasets.values()),
        archs=tuple(all_archs.values()),
        eval=eval_spec_fields,
        analyses=analyses if analyses is not None else base_spec.analyses,
        notes=f"[grid {run_tag}: {len(cells)} cells] {base_spec.notes}",
    )

    return GridPlan(
        run_tag=run_tag,
        results_id=results_id,
        knobs=resolved,
        swept=swept,
        cells=tuple(cells),
        train_specs=train_specs,
        eval_spec=eval_spec,
    )


def save_grid(plan: GridPlan) -> dict:
    """Persists the manifest and one spec JSON per cell-group.

    The per-group specs are what ``run_experiment.py --spec-json`` consumes,
    so a grid can be trained on a server without the registry knowing it.
    """
    results_dir = plan.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / "grid_manifest.csv"
    plan.manifest().to_csv(manifest_path, index=False)

    spec_paths = []
    for spec, group_tag in zip(plan.train_specs, _group_tags(plan)):
        spec_paths.append(
            save_spec(spec, results_dir / f"spec_{_sanitize(group_tag)}.json")
        )

    eval_path = save_spec(plan.eval_spec, results_dir / "spec.json")

    return {
        "manifest": manifest_path,
        "train_specs": spec_paths,
        "eval_spec": eval_path,
    }


def _group_tags(plan: GridPlan) -> list[str]:
    seen = []
    for cell in plan.cells:
        if cell["run_tag"] not in seen:
            seen.append(cell["run_tag"])
    return seen


def grid_commands(plan: GridPlan) -> list[str]:
    """The CLI equivalent of ``run_grid`` — for a server that only runs shell."""
    results_dir = plan.results_dir
    lines = []

    for group_tag in _group_tags(plan):
        spec_json = results_dir / f"spec_{_sanitize(group_tag)}.json"
        lines.append(
            f"python scripts/run_experiment.py --stage data train "
            f"--spec-json {spec_json}"
        )

    lines.append(
        f"python scripts/run_experiment.py --stage eval aggregate "
        f"--spec-json {results_dir / 'spec.json'}"
    )
    return lines


def run_grid(
    plan: GridPlan,
    stages=("data", "train", "eval", "aggregate"),
    device=None,
    verbose: bool = True,
) -> dict:
    """Runs the grid stage by stage.

    data/train run per cell-group (each has its own TrainSpec); eval and
    aggregate run once over the union, so every cell lands in one results
    directory and stage-4 sees the whole grid at once.
    """
    from experiments import runner

    outputs = {}

    if "data" in stages:
        # one prepare over the union: instances are shared between groups
        outputs["data"] = runner.prepare_datasets(plan.eval_spec)

    if "train" in stages:
        summaries = []
        for i, spec in enumerate(plan.train_specs, start=1):
            if verbose:
                print(f"[grid] training cell-group {i}/{len(plan.train_specs)}: "
                      f"{spec.archs[0].tag}")
            summaries.extend(runner.train(spec, device=device, verbose=verbose))
        outputs["train"] = summaries

    if "eval" in stages:
        outputs["eval"] = runner.evaluate(
            plan.eval_spec, device=device or "cpu", verbose=verbose
        )

    if "aggregate" in stages:
        outputs["aggregate"] = runner.aggregate(plan.eval_spec)

    return outputs
