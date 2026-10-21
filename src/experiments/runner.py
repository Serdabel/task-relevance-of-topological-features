"""Stage runner for experiment specs.

Thin composition over the existing pipeline machinery — each stage persists
its artifacts and can run independently (locally or on DataSphere):

    prepare_datasets(spec)  recipes -> store_dataset_instance (+ null features)
    train(spec)             -> training.mass_training / training.sweeps
    evaluate(spec)          -> analysis.stat_aggregation (+ eval-config sidecar)
    aggregate(spec)         -> analysis.witness_stats tables/figures  [scaffold]

Must run with the repo root as CWD (persistence-layer convention); the CLI
wrappers in scripts/ take care of that.
"""

from pathlib import Path

import data.dataset_recipes as dataset_recipes
from experiments.configs import ExperimentSpec, save_spec
from analysis.stat_aggregation import find_model_paths_for_dataset
from management.experiment_management import store_dataset_instance
from training.mass_training import train_many_mlps_on_dataset_instance


def _criteria_registry() -> dict:
    """Extra-criterion names -> callables (extra_criteria protocol).

    Lazy imports: TDA backends may be missing; resolving a name whose
    module cannot import raises there, not at experiments import time.
    """
    from analysis.criterions import sampled_structural_criterion
    from analysis.extremum_anchored import extremum_anchored_criterion
    from analysis.homological import homological_criterion
    from analysis.shape_descriptors import harmonic_criterion

    return {
        "extremum": extremum_anchored_criterion,
        "harmonic": harmonic_criterion,
        "homological": homological_criterion,
        "sampled_structural": sampled_structural_criterion,
    }


def _resolve_null_features_fn(eval_spec):
    """A.1 provider for eval_spec.null_placements, or None if there are none.

    Returning None (rather than a provider that yields nothing) keeps the
    "this experiment has no negative control" case visible: the evaluation
    simply produces no null rows, and the aggregate stage then fails loudly
    naming the cause instead of reporting a gap computed from an empty set.
    """
    if not getattr(eval_spec, "null_placements", ()):
        return None

    from analysis.null_placement import null_features_provider

    return null_features_provider(eval_spec.null_placements)


def _resolve_extra_criteria(eval_spec) -> dict:
    """{name: callable} for eval_spec.extra_criteria, with any
    eval_spec.extra_criteria_kwargs bound via functools.partial."""
    import functools

    if not eval_spec.extra_criteria:
        return {}

    registry = _criteria_registry()
    resolved = {}

    for name in eval_spec.extra_criteria:
        fn = registry[name]
        bound_kwargs = eval_spec.extra_criteria_kwargs.get(name)

        if bound_kwargs:
            bound = functools.partial(fn, **bound_kwargs)
            # eval-config sidecars record criteria by name; partials lack one.
            bound.__name__ = f"{fn.__name__}[{bound_kwargs}]"
            resolved[name] = bound
        else:
            resolved[name] = fn

    return resolved


def dataset_instance_path(name: str, instance_id: int = 0) -> str:
    return f"data/dataset_instances/{name}/{name}_{instance_id}/"


class StaleInstanceError(RuntimeError):
    """A stored instance does not correspond to the requested DatasetSpec."""


def _assert_stored_params_match(ds, instance_dir, allow_unverified=False) -> bool:
    """Guards the "same name, different spec" trap.

    prepare_datasets keys on the dataset NAME, so a spec that changes how the
    instance is built — the recipe params, or which control features are
    attached — would otherwise silently reuse the previous variant's data.
    That is invisible afterwards, because the stored instance is the only
    record of what the models saw and what the criteria were scored against.

    Compares the full ``dataset_signature`` and refuses on any mismatch, with a
    remedy chosen for the blast radius of the field that changed:

      * ``construction`` differs (recipe / params) -> the TRAINING data would
        differ, so every model in models/runs/<name>/ is invalid for this spec.
        The instance must get a new name.
      * ``controls`` differs (target feature, null count/kind/version) -> only
        evaluation-time features differ; X/y/split and therefore every trained
        model are still valid, so the instance can be repaired in place with
        scripts/refresh_null_features.py.

    Instances persisted before signatures were recorded cannot be verified.
    Those are REFUSED rather than waved through (the previous behaviour, which
    is how stale null features survived a change of ``null_kind`` — see
    AUDIT/02_plan_traceability.md X7). Pass allow_unverified=True to proceed
    anyway; the decision is then recorded in the run output.
    """
    from management.experiment_management import load_dataset_instance
    from experiments.configs import (
        dataset_signature, _plain_json, SIGNATURE_VERSION,
    )

    try:
        stored = load_dataset_instance(
            str(instance_dir), map_location="cpu", reconstruct_data=False
        )
    except Exception as error:
        raise StaleInstanceError(
            f"{ds.name}: an instance directory exists at {instance_dir} but "
            f"could not be read ({error}). Refusing to continue: a run must "
            f"not proceed against an instance whose contents are unknown."
        ) from error

    metadata = stored.get("metadata", {}) or {}
    wanted = dataset_signature(ds)

    stored_signature = metadata.get("dataset_signature")

    if stored_signature is None:
        # Legacy instance. Verify whatever was recorded, then refuse on the rest.
        legacy_params = metadata.get("recipe_params")
        detail = ""

        if legacy_params is not None:
            if _plain_json(legacy_params) != wanted["construction"]["params"]:
                raise StaleInstanceError(
                    f"{ds.name}: stored instance was built with different recipe "
                    f"params.\n  stored: {_plain_json(legacy_params)}"
                    f"\n  spec:   {wanted['construction']['params']}\n"
                    f"Give the new variant its own DatasetSpec.name — the instance "
                    f"is the ground truth for every model trained on it."
                )
            detail = " (recipe params match; control config was never recorded)"

        if allow_unverified:
            print(
                f"[prepare_datasets] {ds.name}: WARNING — reusing an instance "
                f"with no dataset_signature{detail}; provenance unverified "
                f"(allow_unverified=True)."
            )
            return False

        raise StaleInstanceError(
            f"{ds.name}: stored instance predates provenance signatures"
            f"{detail}, so it cannot be shown to match this spec.\n"
            f"  spec wants: {wanted['controls']}\n"
            f"This is the path by which stale null features survived a change of "
            f"null_kind (AUDIT X7). Choose one:\n"
            f"  1. refresh only the control features in place, keeping X/y/split "
            f"and every trained model:\n"
            f"       python scripts/refresh_null_features.py --experiment <id> "
            f"--dataset {ds.name}\n"
            f"  2. give the variant its own DatasetSpec.name (a fresh instance);\n"
            f"  3. re-run with --allow-unverified-instances if you are certain "
            f"the stored instance matches."
        )

    # An instance stamped under an older signature FORMAT cannot be compared
    # field-by-field: version 1 recorded no resolved_params, so the comparison
    # below would report a construction mismatch that is really a format
    # difference. Say so precisely and offer the migration, instead of sending
    # the researcher off to rename a dataset that is in fact fine.
    stored_version = int(stored_signature.get("signature_version", 1))

    if stored_version < SIGNATURE_VERSION:
        stored_c = dict(stored_signature.get("construction") or {})
        wanted_c = dict(wanted["construction"])
        comparable = (
            stored_c.get("recipe") == wanted_c.get("recipe")
            and stored_c.get("params") == wanted_c.get("params")
        )

        if not comparable:
            raise StaleInstanceError(
                f"{ds.name}: stored instance was BUILT differently (and under "
                f"signature format v{stored_version}).\n"
                f"  stored: {stored_c}\n  spec:   {wanted_c}\n"
                f"Give the new variant its own DatasetSpec.name."
            )

        if not allow_unverified:
            raise StaleInstanceError(
                f"{ds.name}: stored instance carries signature format "
                f"v{stored_version}, this code writes v{SIGNATURE_VERSION}. "
                f"The recipe and the explicitly-passed params match, but v1 "
                f"did not record the recipe's DEFAULTS, so an edit to one of "
                f"them cannot be ruled out.\n"
                f"Re-stamp it after verifying the stored data still matches a "
                f"fresh build of the recipe:\n"
                f"    python scripts/refresh_null_features.py --experiment "
                f"<id> --dataset {ds.name} --restamp --apply\n"
                f"or re-run with --allow-unverified-instances."
            )

        print(
            f"  {ds.name}: signature format v{stored_version} < "
            f"v{SIGNATURE_VERSION}; recipe and params match, defaults "
            f"UNVERIFIED, reusing"
        )
        return False

    if stored_signature.get("construction") != wanted["construction"]:
        raise StaleInstanceError(
            f"{ds.name}: stored instance was BUILT differently — the training "
            f"data would differ, so every model in models/runs/{ds.name}/ is "
            f"invalid for this spec.\n"
            f"  stored: {stored_signature.get('construction')}\n"
            f"  spec:   {wanted['construction']}\n"
            f"Give the new variant its own DatasetSpec.name."
        )

    if stored_signature.get("controls") != wanted["controls"]:
        raise StaleInstanceError(
            f"{ds.name}: stored instance has different CONTROL features.\n"
            f"  stored: {stored_signature.get('controls')}\n"
            f"  spec:   {wanted['controls']}\n"
            f"X/y/split are unaffected, so models trained on this instance stay "
            f"valid. Refresh only the features in place:\n"
            f"    python scripts/refresh_null_features.py --experiment <id> "
            f"--dataset {ds.name}"
        )

    return True


def prepare_datasets(
    spec: ExperimentSpec,
    train_ratio: float = 0.8,
    split_seed: int = 0,
    allow_unverified_instances: bool = False,
) -> dict:
    """
    Builds and persists one instance per DatasetSpec. Reuses a dataset type
    that already has an instance 0 on disk ONLY if its stored provenance
    signature matches this spec; otherwise raises StaleInstanceError with the
    appropriate remedy (see _assert_stored_params_match).

    Null features (spec.n_null_features > 0) are generated matched to the
    target feature and registered with feature_role="null"
    (analysis.controls.make_null_features).

    allow_unverified_instances:
        proceed against instances persisted before provenance signatures
        existed. Off by default: silently reusing such an instance is how
        stale null features survived a change of null_kind (AUDIT X7).

    Returns {dataset_name: stored_instance_path}.
    """
    stored = {}
    wrote_anything = False

    for ds in spec.datasets:
        existing = Path(dataset_instance_path(ds.name))

        if existing.exists():
            verified = _assert_stored_params_match(
                ds, existing, allow_unverified=allow_unverified_instances
            )
            status = "provenance verified" if verified else "provenance UNVERIFIED"
            print(f"[prepare_datasets] {ds.name}: instance 0 exists, "
                  f"{status}, reusing")
            stored[ds.name] = dataset_instance_path(ds.name)
            continue

        recipe = getattr(dataset_recipes, ds.recipe, None)
        assert recipe is not None, f"Unknown recipe: {ds.recipe}"

        data = recipe(**ds.params)

        from experiments.configs import dataset_signature

        extra_metadata = {
            "experiment_id": spec.id,
            "recipe": ds.recipe,
            "recipe_params": dict(ds.params),
            # Full provenance record; _assert_stored_params_match compares it
            # on every later run so a stale instance cannot masquerade as a
            # current one (AUDIT X7).
            "dataset_signature": dataset_signature(ds),
        }

        target_name = ds.target_feature or "C_mid"

        if target_name not in data.features:
            assert ds.target_feature is None, (
                f"{ds.name}: target_feature {ds.target_feature!r} not among "
                f"registered features {list(data.features)}"
            )
            target_name = next(iter(data.features))

        extra_metadata["target_feature"] = target_name

        if ds.n_null_features > 0:
            from analysis.controls import make_null_features

            reference = data.get_feature_points(target_name)

            lows = data.get_data().min(dim=0).values
            highs = data.get_data().max(dim=0).values
            bounds = [
                (float(lo), float(hi)) for lo, hi in zip(lows, highs)
            ]

            nulls = make_null_features(
                reference_feature_points=reference,
                bounds=bounds,
                n_features=ds.n_null_features,
                kind=ds.null_kind,
                seed=ds.params.get("seed", 0),
            )

            for null_name, null_points in nulls.items():
                data.add_feature(
                    name=null_name,
                    points=null_points,
                    metadata={"feature_role": "null"},
                )

            extra_metadata["null_features"] = list(nulls.keys())

        path = store_dataset_instance(
            name=ds.name,
            data=data,
            train_ratio=train_ratio,
            split_seed=split_seed,
            extra_metadata=extra_metadata,
            stratify_by=ds.stratify_split,
        )

        stored[ds.name] = path
        wrote_anything = True
        print(f"[prepare_datasets] {ds.name}: stored at {path}")

    # Only record the spec if this stage actually built something. save_spec
    # used to run unconditionally on entry, so merely CHECKING a dataset
    # rewrote spec.json in the results directory — which is a provenance
    # record of a run that did not happen, and is how two historical spec
    # files were silently overwritten during a verification pass
    # (AUDIT/09 section D6).
    if wrote_anything:
        save_spec(spec)

    return stored


def train(
    spec: ExperimentSpec,
    instance_id: int = 0,
    device=None,
    verbose: bool = True,
) -> list:
    """
    Fills every (dataset x architecture) cell to spec.train.n_models models.
    Delegates to train_many_mlps_on_dataset_instance; the architecture tag
    (ArchSpec.model_name) separates grid cells inside one instance directory.

    Resumable and idempotent, which is what makes it safe to re-run on a server
    without supervision. Two levels of it:

    * Between calls — the models already stored under the cell's tag are
      counted, and only the shortfall is trained. A cell already at its target
      is skipped without loading anything.
    * Inside a call — every successful model is written the moment it succeeds,
      so an interrupted run keeps what it had already earned.

    The attempt budget is a CEILING, not a plan. ``train_many_mlps_on_dataset_
    instance`` takes ATTEMPTS and saves the ones that clear the gate, so asking
    it for exactly ``n_models`` silently returns a short ensemble whenever the
    success rate is below 1 — which is what happened to experiment 4's
    task-relevant cells. Here it is given ``spec.train.max_attempts`` attempts
    and told to stop at the first ``need`` successes: an easy cell costs
    exactly ``need``, a hard one is allowed the whole ceiling, and a cell that
    exhausts the ceiling is reported loudly and left short. That case is a
    statement about the task and wants a person, not a bigger budget.

    Returns the union of run summaries.
    """
    train_kwargs = {
        "target_test_acc": spec.train.target_test_acc,
        "n_epochs": spec.train.n_epochs,
        "min_epochs": spec.train.min_epochs,
        "patience": spec.train.patience,
        "batch_size": spec.train.batch_size,
        "lr": spec.train.lr,
        "optimizer_name": spec.train.optimizer_name,
        "weight_decay": spec.train.weight_decay,
    }

    all_summaries = []
    short_cells = []

    for ds in spec.datasets:
        path = dataset_instance_path(ds.name, instance_id)

        for arch in spec.archs:
            have = len(find_model_paths_for_dataset(
                dataset_type=ds.name, instance_id=instance_id,
                model_name=arch.model_name,
            ))
            need = spec.train.n_models - have

            if need <= 0:
                if verbose:
                    print(f"[train] {ds.name} x {arch.model_name}: "
                          f"уже есть {have}/{spec.train.n_models}, пропуск")
                continue

            attempts = max(need, spec.train.max_attempts)
            if verbose:
                print(f"[train] {ds.name} x {arch.model_name}: есть {have}, "
                      f"добрать {need}, потолок попыток {attempts}")

            summaries = train_many_mlps_on_dataset_instance(
                path,
                n_models=attempts,
                stop_after_successful=need,
                activation=arch.activation,
                hidden_dim=arch.width,
                n_layers=arch.depth,
                model_name=arch.model_name,
                device=device,
                verbose=verbose,
                **train_kwargs,
            )

            all_summaries.extend(summaries)

            now = have + sum(s["saved"] for s in summaries)
            if now < spec.train.n_models:
                short_cells.append((ds.name, arch.model_name, now, len(summaries)))
                print(f"[train] ВНИМАНИЕ {ds.name} x {arch.model_name}: недобор "
                      f"{now}/{spec.train.n_models} за {len(summaries)} попыток. "
                      f"Потолок исчерпан — разберитесь, почему эта конфигурация "
                      f"не обучается, прежде чем поднимать бюджет.")

    if short_cells and verbose:
        print(f"\n[train] недобрано ячеек: {len(short_cells)} — "
              f"это результат, а не сбой; отчитывайтесь по фактическому N:")
        for name, tag, now, tried in short_cells:
            print(f"    {name} x {tag}: {now}/{spec.train.n_models} за {tried} попыток")

    return all_summaries


def evaluate(
    spec: ExperimentSpec,
    instance_id: int = 0,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Stage 3 for every architecture of the spec: one CSV (+ eval-config
    sidecar) per architecture under spec.results_dir — or one per
    (architecture x dataset) when any DatasetSpec carries eval_overrides
    (per-dataset EvalSpec adjustments, e.g. different sample budgets for
    R^3 vs R^4 in exp. 9). runner.aggregate globs neuron_stats_*.csv, so
    both layouts aggregate identically.

    Returns {model_name: DataFrame} (per-dataset frames concatenated).
    """
    import pandas as pd
    from dataclasses import replace

    from analysis.stat_aggregation import collect_compact_neuron_statistics

    save_spec(spec)

    # (dataset_names, effective EvalSpec, csv suffix) groups.
    if any(ds.eval_overrides for ds in spec.datasets):
        eval_groups = [
            ([ds.name], replace(spec.eval, **ds.eval_overrides), f"_{ds.name}")
            for ds in spec.datasets
        ]
    else:
        eval_groups = [([ds.name for ds in spec.datasets], spec.eval, "")]

    results = {}

    for arch in spec.archs:
        frames = []

        for dataset_names, eval_spec, suffix in eval_groups:
            output_csv = (
                spec.results_dir / f"neuron_stats_{arch.model_name}{suffix}.csv"
            )

            df = collect_compact_neuron_statistics(
                dataset_types=dataset_names,
                instance_id=instance_id,
                resolution=eval_spec.resolution,
                bounds_margin=eval_spec.bounds_margin,
                bounds_override=eval_spec.bounds,
                layer_ids=(
                    None
                    if eval_spec.layer_ids is None
                    else list(eval_spec.layer_ids)
                ),
                kinds=eval_spec.kinds,
                device=device,
                structural_kwargs=dict(eval_spec.structural_kwargs),
                tubular_kwargs=dict(eval_spec.tubular_kwargs),
                run_structural=eval_spec.run_structural,
                run_tubular=eval_spec.run_tubular,
                field_source=eval_spec.field_source,
                n_field_points=eval_spec.n_field_points,
                field_seed=eval_spec.field_seed,
                extra_criteria=_resolve_extra_criteria(eval_spec) or None,
                # A.1 controls, built per run from the spec's placements rather
                # than read from the instance (EvalSpec.null_placements). The
                # provider carries its own record, which the eval-config sidecar
                # writes — these features exist nowhere on disk, so that record
                # is what keeps the run reconstructible.
                extra_features_fn=_resolve_null_features_fn(eval_spec),
                model_name_filter=arch.model_name,
                # Experimental integrity (AUDIT/03_triage.md B5): admit only
                # models accepted under a gate at least as strict as this
                # spec's, so models trained under an older/weaker rule cannot
                # silently join the ensemble that a later eval globs.
                require_train_gate={
                    "target_test_acc": spec.train.target_test_acc,
                    "min_epochs": spec.train.min_epochs,
                    "patience": spec.train.patience,
                },
                output_csv_path=output_csv,
                verbose=verbose,
            )

            frames.append(df)

        results[arch.model_name] = (
            frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        )

    return results


def aggregate(spec: ExperimentSpec, instance_id: int = 0):
    """
    Stage 4: runs the analyses named in spec.analyses over the evaluation
    CSVs in spec.results_dir (all architectures concatenated — rows carry
    model_name/hidden_dim/n_layers, so per-architecture analyses regroup).

    Analysis names -> implementations (experiments.analyses.ANALYSES):
        "witness_tables"       -> witness_stats.build_witness_aggregate_tables
        "feature_overlap"      -> witness_stats.build_feature_overlap_tables
        "witness_rate_gap"     -> controls.witness_rate_gap (exp. 1)
        "rate_gap_by_criterion"-> the same gap for EVERY witness criterion
        "consistency_table"    -> witness_stats.consistency_table
        "rate_vs_width"        -> exp-2 curve: rate[A]-rate[null] vs width
        "basin_ridge_typing"   -> extremum_anchored.classify_... (exp. 6)
        "basin_ridge_vs_ratio" -> exp-4 curve: witness type vs density
        "seed_bimodality"      -> A.7 per-seed distributions
        "sparse_probing"       -> probing (A.4, exp. 3)
        "layer_persistence"    -> exp-3 witness-persists-on-deeper-layers
        "ablation"             -> ablation (A.3, exp. 7)

    Tables land in <results>/aggregates/, figures in <results>/figures/.
    Returns {analysis_name: outputs}.
    """
    import pandas as pd

    from experiments.analyses import ANALYSES, AggregationContext

    csv_paths = sorted(Path(spec.results_dir).glob("neuron_stats_*.csv"))
    assert csv_paths, (
        f"no neuron_stats_*.csv under {spec.results_dir} — run the eval "
        "stage first"
    )

    df = pd.concat(
        [
            # feature_role="null" must survive the round-trip: pandas parses
            # a bare "null" cell as NaN unless the column is read verbatim.
            pd.read_csv(p, converters={"feature_role": lambda v: v})
            for p in csv_paths
        ],
        ignore_index=True,
    )
    ctx = AggregationContext(spec, instance_id=instance_id)

    outputs = {}

    for analysis in spec.analyses:
        fn = ANALYSES.get(analysis)
        assert fn is not None, (
            f"unknown analysis {analysis!r}; known: {sorted(ANALYSES)}"
        )

        print(f"[aggregate] {spec.id}: {analysis}")
        outputs[analysis] = fn(df, ctx)

    return outputs
