from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from management.experiment_management import (
    load_dataset_instance,
    load_model_instance,
    get_feature_points,
)

from analysis.criterions import (
    evaluate_neurons_on_feature,
    structural_criterion,
    tubular_criterion,
)
from analysis.eval_config import (
    build_eval_config, save_eval_config, resolve_kwargs,
)
from analysis.null_placement import bounds_covering

def dataset_instance_path(
    dataset_root: str,
    dataset_type: str,
    instance_id: int = 0,
):
    dataset_root = dataset_root.rstrip("/")
    return f"{dataset_root}/{dataset_type}/{dataset_type}_{instance_id}/"


def to_project_path(path):
    """
    Converts '/models/...' -> 'models/...'
    Converts '/data/...' -> 'data/...'

    This is needed because your save/load functions often return paths
    with leading '/', while files are project-root-relative.
    """
    path = Path(path)

    if path.is_absolute():
        return Path(str(path).lstrip("/"))

    return path


def filter_models_by_train_gate(model_paths, require_train_gate: dict):
    """Keeps only models accepted under a gate at least as strict as required.

    Experimental-integrity guard (AUDIT/03_triage.md B5). A stored ensemble is
    a directory, so models trained under an older, weaker acceptance rule sit
    beside current ones and a later eval globs them all. The stored
    ``halfplane_base`` models were accepted at ``stopped_epoch=1`` under
    ``min_epochs=0, patience=1`` — networks essentially at initialization —
    while the spec now requires ``min_epochs=50, patience=5``.

    A model is admissible iff every gate field it recorded is at least as
    strict as the requirement. Models that recorded no gate at all are
    REJECTED, because "unknown provenance" must not read as "fine".

    Returns (kept_paths, [(rejected_path, reason), ...]).
    """
    import torch

    kept, rejected = [], []

    for path in model_paths:
        metrics_path = Path(path) / "metrics.pt"

        if not metrics_path.exists():
            rejected.append((path, "no metrics.pt — training gate unknown"))
            continue

        metrics = torch.load(metrics_path, map_location="cpu", weights_only=False) or {}
        reasons = []

        for field, wanted in require_train_gate.items():
            if wanted is None:
                continue
            got = metrics.get(field)
            if got is None:
                reasons.append(f"{field} not recorded")
            elif got < wanted:
                reasons.append(f"{field}={got} < required {wanted}")

        if metrics.get("reached_target") is False:
            reasons.append("reached_target=False")

        if reasons:
            rejected.append((path, "; ".join(reasons)))
        else:
            kept.append(path)

    return kept, rejected


def find_model_paths_for_dataset(
    dataset_type: str,
    instance_id: int = 0,
    runs_root: str = "models/runs",
    model_name: str | None = None,
):
    """
    Finds all saved model instance directories for one dataset instance.

    Expected structure:

        models/runs/{dataset_type}/{dataset_type}_{instance_id}/{model_dir}/checkpoint.pt

    model_name:
        If given, keeps only runs of that architecture tag (the model_name
        used at training time, e.g. "tanh_w7_d5"); different architectures
        trained on the same instance coexist in one directory.
    """

    runs_root = to_project_path(runs_root)
    instance_name = f"{dataset_type}_{instance_id}"

    model_root = runs_root / dataset_type / instance_name

    if not model_root.exists():
        return []

    model_paths = []

    for p in sorted(model_root.iterdir()):
        if not (p.is_dir() and (p / "checkpoint.pt").exists()):
            continue

        if model_name is not None and f"_{model_name}_model" not in p.name:
            continue

        model_paths.append(str(p))

    return model_paths


def bounds_from_X(
    X: torch.Tensor,
    margin: float = 0.5,
):
    X_cpu = X.detach().cpu()

    mins = X_cpu.min(dim=0).values
    maxs = X_cpu.max(dim=0).values

    bounds = []

    for lo, hi in zip(mins, maxs):
        bounds.append((float(lo.item() - margin), float(hi.item() + margin)))

    return bounds


def to_python_scalar(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()

    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    return x


def feature_role_from_metadata(loaded_dataset, feature_name: str):
    """
    Resolves the analysis role of a feature, in priority order:

      1. the feature's own metadata ("feature_role", e.g. "null" set by
         experiments.runner.prepare_datasets for A.1 null features);
      2. dataset-level metadata: target_feature / diagnostic_features /
         null_features;
      3. default "feature".
    """
    features = loaded_dataset.get("features", {})
    feature_meta = features.get(feature_name, {}).get("metadata", {}) or {}

    if feature_meta.get("feature_role"):
        return str(feature_meta["feature_role"])

    metadata = loaded_dataset.get("metadata", {})

    if feature_name == metadata.get("target_feature", None):
        return "target"

    if feature_name in metadata.get("diagnostic_features", []):
        return "diagnostic"

    if feature_name in metadata.get("null_features", []):
        return "null"

    return "feature"

def compact_rows_from_evaluate_result(
    rows,
    *,
    dataset_type: str,
    dataset_instance_path: str,
    feature_name: str,
    feature_role: str,
    model_path: str,
    checkpoint: dict,
    extra_columns: list[str] | None = None,
    extra_criteria_names: list[str] | None = None,
    eval_columns: dict | None = None,
):
    """
    Converts evaluate_neurons_on_feature output into compact numeric rows.

    Keeps only:

        structural_score
        structural_score_normalized_by_diameter
        structural_score_normalized_by_grid_step
        is_structural_witness
        tubular_score

    plus identifying metadata, plus any row keys listed in extra_columns
    (e.g. columns produced by extra criteria plugged into
    evaluate_neurons_on_feature).

    extra_columns=None with extra_criteria_names given auto-includes every
    "{name}_*" row key produced by those criteria, so plugging in a new
    criterion needs no manual column list.

    eval_columns:
        Optional {column: value} of evaluation parameters inlined into every
        row (e.g. eval_resolution, structural_threshold) so rows stay
        self-describing after CSV merges; the full parameter record lives in
        the eval-config JSON sidecar (see analysis.eval_config).

    Does not store model_seed or model_index.
    Use model_id and model_path as model identifiers.
    """

    model_config = checkpoint.get("model_config", {}) or {}
    train_metrics = checkpoint.get("metrics", {}) or {}

    if extra_columns is None and extra_criteria_names:
        prefixes = tuple(f"{name}_" for name in extra_criteria_names)

        auto_columns = []
        for row in rows:
            for key in row.keys():
                if key.startswith(prefixes) and key not in auto_columns:
                    auto_columns.append(key)

        extra_columns = auto_columns

    compact_rows = []

    for row in rows:
        structural_result = row.get("structural_result", {}) or {}
        value_result = row.get("value_result", {}) or {}

        compact_row = {
            # dataset / feature
            "dataset_type": dataset_type,
            "dataset_instance_path": dataset_instance_path,
            "feature_name": feature_name,
            "feature_role": feature_role,

            # model
            "model_path": model_path,
            "model_id": checkpoint.get("model_id", None),
            "model_name": checkpoint.get("model_name", None),
            "model_instance_name": checkpoint.get("model_instance_name", None),

            # architecture
            "hidden_dim": model_config.get("hidden_dim", None),
            "n_layers": model_config.get("n_layers", None),
            "activation": model_config.get("activation", None),

            # training result
            "train_final_test_acc": train_metrics.get("final_test_acc", None),
            "train_best_test_acc": train_metrics.get("best_test_acc", None),
            "train_reached_target": train_metrics.get("reached_target", None),
            "train_stopped_epoch": train_metrics.get("stopped_epoch", None),

            # the GATE that accepted this model, inlined so a row is
            # self-describing after merges (AUDIT/04_repair_log.md R4): a
            # stopped_epoch of 1 means something very different under
            # min_epochs=0 than under min_epochs=50.
            "train_target_test_acc": train_metrics.get("target_test_acc", None),
            "train_min_epochs": train_metrics.get("min_epochs", None),
            "train_patience": train_metrics.get("patience", None),

            # neuron identity
            "layer_id": row.get("layer_id", None),
            "kind": row.get("kind", None),
            "coord_id": row.get("coord_id", None),

            # field regime ON this feature — the ADMISSIBILITY check: did the
            # neuron have any dynamic range where the criteria were asked to
            # look? A feature over which f is numerically constant admits no
            # level set at all and contributes a guaranteed non-detection.
            # Kept as a first-class column rather than an extra_column because
            # it qualifies EVERY criterion's verdict in the row, not one of
            # them. Not a covariate to match nulls against — see
            # criterions.field_regime_over_feature.
            "field_grad_norm_on_feature": row.get(
                "field_grad_norm_on_feature", None),
            "field_value_std_on_feature": row.get(
                "field_value_std_on_feature", None),
            "field_value_abs_mean_on_feature": row.get(
                "field_value_abs_mean_on_feature", None),

            # structural criterion
            "structural_score": row.get(
                "structural_score",
                structural_result.get("score", None),
            ),
            "structural_score_normalized_by_diameter": row.get(
                "structural_score_normalized_by_diameter",
                structural_result.get("score_normalized_by_diameter", None),
            ),
            "structural_score_normalized_by_grid_step": row.get(
                "structural_score_normalized_by_grid_step",
                structural_result.get("score_normalized_by_grid_step", None),
            ),
            "is_structural_witness": row.get(
                "is_structural_witness",
                structural_result.get("is_structural_witness", None),
            ),

            # tubular criterion
            "tubular_score": row.get(
                "tubular_score",
                value_result.get("tubular_score", None),
            ),
        }

        if extra_columns is not None:
            for column in extra_columns:
                compact_row[column] = row.get(column, None)

        if eval_columns is not None:
            compact_row.update(eval_columns)

        compact_row = {
            key: to_python_scalar(value)
            for key, value in compact_row.items()
        }

        compact_rows.append(compact_row)

    return compact_rows

def collect_compact_neuron_statistics(
    dataset_types: list[str],
    dataset_root: str = "/data/dataset_instances",
    instance_id: int = 0,
    runs_root: str = "models/runs",

    # features
    feature_names: list[str] | None = None,

    # Evaluation-time features, built per dataset instead of read from disk.
    # fn(loaded_dataset, dataset_type) -> {name: (points, role)}.
    #
    # This is how manually placed A.1 nulls reach the evaluation without being
    # persisted into the dataset instance (analysis.null_placement). Nulls are
    # a CONTROL, not data: keeping them out of structure.pt means the stored
    # instance stays a pure record of what the network was trained on, and the
    # control can be re-chosen per run without a provenance refresh. The price
    # is that the placement list must be recorded next to the results, or the
    # run is not reconstructible — collect_compact_neuron_statistics writes it
    # into the eval-config sidecar for exactly that reason.
    extra_features_fn=None,

    # evaluation
    resolution: int | list[int] | tuple[int, ...] = 150,
    bounds_margin: float = 0.5,

    # Explicit evaluation box, shared by every dataset of the call. None keeps
    # the derived box (data + margin, grown to cover the A.1 placements).
    #
    # The derived box is a function of the dataset AND of the null set, so two
    # datasets meant to be compared can end up on different grids at the same
    # `resolution` -- see EvalSpec.bounds for the case that motivated this.
    # Pass a box when scores from different datasets have to be comparable.
    bounds_override=None,
    layer_ids: list[int] | None = None,
    coord_ids_by_layer: dict[int, list[int]] | None = None,
    kinds: tuple[str, ...] = ("post",),
    device: str | torch.device = "cpu",

    structural_kwargs: dict | None = None,
    tubular_kwargs: dict | None = None,
    run_structural: bool = True,
    run_tubular: bool = True,

    # shared-field source: "grid" (default) or "random" (the d >= 3 route),
    # see evaluate_neurons_on_feature.
    field_source: str = "grid",
    n_field_points: int = 20000,
    field_seed: int = 0,

    # extra criteria: {name: callable}, see evaluate_neurons_on_feature.
    # Their numeric outputs appear as columns "{name}_{key}";
    # extra_columns=None auto-detects them.
    extra_criteria: dict | None = None,
    extra_columns: list[str] | None = None,

    # model selection
    model_name_filter: str | None = None,

    # Explicit model list, replacing discovery. This is the ONLY hook the
    # parallel evaluation path needs: a shard is "these models of this
    # architecture on this dataset", and it must reach the loop without
    # duplicating the discovery/gate/bounds logic that surrounds it
    # (analysis.parallel_eval, scripts/eval_shard.py). Nothing downstream of
    # the selection changes, so a sharded run and a sequential run compute the
    # same rows.
    model_paths_by_dataset: dict[str, list[str]] | None = None,

    # saving
    output_csv_path: str | Path | None = None,
    save_partial: bool = True,

    # debug / limits
    max_models_per_dataset: int | None = None,
    require_train_gate: dict | None = None,
    verbose: bool = True,
):
    """
    Main compact batch evaluation function.

    For every:

        dataset_type
        saved model for this dataset
        feature in this dataset
        neuron in the model

    computes:

        structural_score
        structural_score_normalized_by_diameter
        structural_score_normalized_by_grid_step
        is_structural_witness
        tubular_score

    model_name_filter:
        Restricts to runs of one architecture tag (e.g. "tanh_w7_d5") when a
        dataset instance holds an architecture sweep.

    When output_csv_path is given, the full evaluation-parameter record is
    written to the sidecar "<csv>.eval_config.json" (see analysis.eval_config),
    and key parameters (eval_resolution, structural_threshold) are inlined as
    CSV columns.

    Returns:
        pd.DataFrame with one row per neuron-feature-model.
    """

    if structural_kwargs is None:
        structural_kwargs = {}

    if tubular_kwargs is None:
        tubular_kwargs = {}

    structural_kwargs = dict(structural_kwargs)
    tubular_kwargs = dict(tubular_kwargs)

    # Force compact mode.
    structural_kwargs["return_best_component_points"] = False
    tubular_kwargs["return_points"] = False
    tubular_kwargs["return_metadata"] = False

    structural_kwargs.setdefault("threshold", 0.1)
    tubular_kwargs.setdefault("compute_auc", False)

    eval_columns = {
        "eval_resolution": resolution,
        "eval_field_source": field_source,
        "structural_threshold": (
            structural_kwargs["threshold"] if run_structural else None
        ),
    }

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        eval_config = build_eval_config(
            dataset_types=dataset_types,
            dataset_root=dataset_root,
            instance_id=instance_id,
            runs_root=runs_root,
            feature_names=feature_names,
            resolution=resolution,
            bounds_margin=bounds_margin,
            bounds_override=(
                None if bounds_override is None
                else [list(axis) for axis in bounds_override]
            ),
            layer_ids=layer_ids,
            coord_ids_by_layer=coord_ids_by_layer,
            kinds=list(kinds),
            device=str(device),
            # Defaults resolved, not just what the caller passed — n_levels
            # (101) and the component/eps parameters define the search the
            # score minimises over, so two CSVs are only comparable if they
            # are recorded (AUDIT/09 section D4).
            structural_kwargs=resolve_kwargs(structural_criterion, structural_kwargs),
            tubular_kwargs=resolve_kwargs(tubular_criterion, tubular_kwargs),
            run_structural=run_structural,
            run_tubular=run_tubular,
            field_source=field_source,
            n_field_points=n_field_points,
            field_seed=field_seed,
            extra_criteria=extra_criteria,
            extra_columns=extra_columns,
            model_name_filter=model_name_filter,
            max_models_per_dataset=max_models_per_dataset,
            explicit_model_paths=(
                None if model_paths_by_dataset is None
                else {k: list(v) for k, v in model_paths_by_dataset.items()}
            ),
            output_csv_path=str(output_csv_path),
            # Run-local features are not on disk anywhere, so without this the
            # result could never be reconstructed. Recorded as the placement
            # list itself, which is enough to rebuild the exact point sets.
            extra_features=getattr(
                extra_features_fn, "placement_records", None
            ),
        )

        save_eval_config(eval_config, output_csv_path)

    all_rows = []

    dataset_bar = tqdm(
        dataset_types,
        desc="datasets",
        disable=not verbose,
        position=0,
        leave=True,
    )

    for dataset_type in dataset_bar:
        cur_dataset_path = dataset_instance_path(
            dataset_root=dataset_root,
            dataset_type=dataset_type,
            instance_id=instance_id,
        )

        loaded_dataset = load_dataset_instance(
            cur_dataset_path,
            map_location="cpu",
            reconstruct_data=False,
        )

        X = loaded_dataset["X"]
        bounds = (
            [tuple(float(v) for v in axis) for axis in bounds_override]
            if bounds_override is not None
            else bounds_from_X(X, margin=bounds_margin)
        )

        available_features = list(loaded_dataset.get("features", {}).keys())

        # Features constructed for this run only (manually placed nulls, A.1).
        # They shadow nothing: a name collision with a stored feature is an
        # error, because silently overriding a persisted feature with a
        # run-local one is precisely the provenance failure this design is
        # trying to avoid.
        extra_features = {}
        if extra_features_fn is not None:
            extra_features = dict(extra_features_fn(loaded_dataset, dataset_type) or {})
            clashes = sorted(set(extra_features) & set(available_features))
            assert not clashes, (
                f"run-local features {clashes} collide with features stored in "
                f"{dataset_type}. Rename the placements, or drop the stored "
                "copies — do not let a control silently replace persisted data."
            )
            available_features = available_features + list(extra_features)

            # Expand-to-fit: a placement may legitimately sit outside the data
            # box (the "far from every cluster" case is one of the regimes
            # worth sampling), and a feature partly outside `bounds` would be
            # scored against a truncated level set. Grow only as far as the
            # placements require — the box also sets the level-sweep range and
            # the adaptive level_eps statistics FOR THE TARGET, so enlarging it
            # is not free and must not be done by a fixed factor.
            grown = (
                bounds if bounds_override is not None
                else bounds_covering(
                    bounds, [p for p, _ in extra_features.values()],
                    margin=bounds_margin,
                )
            )
            if grown != bounds:
                print(
                    f"  [bounds] {dataset_type}: expanded to fit null placements "
                    f"{[tuple(round(v, 2) for v in b) for b in bounds]} -> "
                    f"{[tuple(round(v, 2) for v in b) for b in grown]}"
                )
                bounds = grown

        if bounds_override is not None:
            # An explicit box that clips the data or a feature scores it against
            # a truncated level set, and does so silently -- exactly what
            # bounds_covering exists to prevent. Fail loudly instead.
            def _check_inside(points, what):
                pts = points.detach().cpu()
                for axis, (lo, hi) in enumerate(bounds):
                    lo_gap = float(lo - pts[:, axis].min())
                    hi_gap = float(pts[:, axis].max() - hi)
                    assert lo_gap <= 0 and hi_gap <= 0, (
                        f"{dataset_type}: явный bounds={bounds} не покрывает {what} "
                        f"по оси {axis}: вылет {max(lo_gap, hi_gap):.3f}. "
                        "Расширьте бокс — усечённая фича даёт молча неверный скор."
                    )

            _check_inside(X, "данные")
            for _name, (_pts, _role) in extra_features.items():
                _check_inside(_pts, f"нуль-фичу {_name!r}")
            for _name in loaded_dataset.get("features", {}):
                _check_inside(get_feature_points(loaded_dataset, _name),
                              f"фичу {_name!r}")

        if feature_names is None:
            cur_feature_names = available_features
        else:
            cur_feature_names = [
                name for name in feature_names
                if name in available_features
            ]

        if model_paths_by_dataset is not None:
            assert dataset_type in model_paths_by_dataset, (
                f"model_paths_by_dataset was given but has no entry for "
                f"{dataset_type!r}; keys: {sorted(model_paths_by_dataset)}"
            )
            model_paths = list(model_paths_by_dataset[dataset_type])
        else:
            model_paths = find_model_paths_for_dataset(
                dataset_type=dataset_type,
                instance_id=instance_id,
                runs_root=runs_root,
                model_name=model_name_filter,
            )

        # The gate stays on either way: it is an integrity check on what the
        # models were trained under, not part of selecting them.
        if require_train_gate is not None:
            model_paths, rejected = filter_models_by_train_gate(
                model_paths, require_train_gate
            )
            for path, why in rejected:
                print(f"  [gate] EXCLUDED {Path(path).name}: {why}")
            if rejected:
                print(f"  [gate] {len(rejected)} model(s) excluded from "
                      f"{dataset_type}: trained under a weaker gate than this "
                      f"spec requires (AUDIT/04_repair_log.md R7)")

        if max_models_per_dataset is not None:
            model_paths = model_paths[:max_models_per_dataset]

        dataset_bar.set_postfix(
            {
                "dataset": dataset_type,
                "models": len(model_paths),
                "features": len(cur_feature_names),
            }
        )

        if len(model_paths) == 0:
            # Raise rather than warn-and-continue. The realistic way to reach
            # this is not "the directory is empty" but "every model was trained
            # under a weaker gate and the admission filter rejected all of
            # them" — e.g. an N=50 run driven through scripts/train.py's old
            # defaults. Warning and continuing produced an exit-0 run with
            # empty tables, which reads exactly like a clean null result
            # (AUDIT/08 section 5.9).
            raise AssertionError(
                f"{dataset_type}: no admissible models. Either none are stored "
                f"under {runs_root}/{dataset_type}/ with model_name_filter="
                f"{model_name_filter!r}, or every stored model was excluded by "
                "the training gate (the [gate] lines above name each one and "
                "why). Train under this spec's gate, or relax require_train_gate "
                "deliberately — do not read an empty result as a null result."
            )

        model_bar = tqdm(
            model_paths,
            desc=f"{dataset_type}: models",
            disable=not verbose,
            position=1,
            leave=False,
        )

        for model_path in model_bar:
            model, checkpoint = load_model_instance(
                model_path,
                device=device,
                eval_mode=True,
            )

            feature_bar = tqdm(
                cur_feature_names,
                desc="features",
                disable=not verbose,
                position=2,
                leave=False,
            )

            for feature_name in feature_bar:
                if feature_name in extra_features:
                    feature_points, feature_role = extra_features[feature_name]
                else:
                    feature_points = get_feature_points(
                        loaded_dataset,
                        feature_name,
                    )

                    feature_role = feature_role_from_metadata(
                        loaded_dataset,
                        feature_name,
                    )

                eval_rows = evaluate_neurons_on_feature(
                    model=model,
                    feature_points=feature_points,
                    bounds=bounds,
                    resolution=resolution,
                    layer_ids=layer_ids,
                    coord_ids_by_layer=coord_ids_by_layer,
                    kinds=kinds,
                    device=device,
                    structural_kwargs=structural_kwargs,
                    value_kwargs=tubular_kwargs,
                    extra_criteria=extra_criteria,
                    run_structural=run_structural,
                    run_tubular=run_tubular,
                    field_source=field_source,
                    n_field_points=n_field_points,
                    field_seed=field_seed,
                    return_full_result_structural=False,
                    return_full_result_tubular=False,
                    verbose=False,
                )

                compact_rows = compact_rows_from_evaluate_result(
                    eval_rows,
                    dataset_type=dataset_type,
                    dataset_instance_path=cur_dataset_path,
                    feature_name=feature_name,
                    feature_role=feature_role,
                    model_path=model_path,
                    checkpoint=checkpoint,
                    extra_columns=extra_columns,
                    extra_criteria_names=(
                        list(extra_criteria.keys()) if extra_criteria else None
                    ),
                    eval_columns=eval_columns,
                )

                all_rows.extend(compact_rows)

                if output_csv_path is not None and save_partial:
                    pd.DataFrame(all_rows).to_csv(
                        output_csv_path,
                        index=False,
                    )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)

    return df