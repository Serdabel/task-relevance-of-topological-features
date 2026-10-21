"""Declarative experiment specifications.

An ``ExperimentSpec`` is data, not code: which dataset recipes with which
parameters (possibly a sweep), which architectures (possibly a grid), the
training config, the evaluation config, and which aggregations to run.
``experiments.registry`` holds the named specs for PLAN.md's experiments;
``experiments.runner`` executes any stage of any spec; the spec itself is
persisted as JSON next to its results, so every number in
``analysis/results/experiments/<id>/`` is traceable to an exact spec.

Specs are frozen dataclasses versioned in git (not YAML): fewer moving parts,
and JSON persistence comes for free via ``spec_to_dict``.
"""

import json
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

PROFILES = ("smoke", "local", "server")

# Bumped whenever the SEMANTICS of null-feature construction change, so that
# instances persisted under an older law are detected as stale rather than
# silently reused. History:
#   1  original: centre uniform in the data bbox shrunk by the null radius.
#      Collapsed to a point/segment whenever diam(A) >= a bbox extent, and
#      allowed nulls to land on top of the target (AUDIT/01_math.md C1).
#   2  rejection-sampled placement with a minimum separation from the target
#      and from previously accepted nulls (AUDIT/04_repair_log.md R3).
NULL_SPEC_VERSION = 2


# Format version of dataset_signature itself. Version 2 records RESOLVED
# recipe parameters (defaults merged under the spec's overrides); version 1
# recorded only what the spec explicitly passed. Instances stamped under an
# earlier format are recognised and reported as such rather than silently
# compared against a signature they cannot match.
SIGNATURE_VERSION = 2


def _resolved_recipe_params(ds) -> dict:
    """The recipe's defaults with the spec's params applied over them.

    Returns {} if the recipe cannot be resolved (a scaffold, a renamed
    function, a spec built by hand in a notebook). An empty dict compares
    equal only to another empty dict, so an unresolvable recipe degrades to
    the old behaviour instead of raising at import time.
    """
    try:
        import inspect
        from data import dataset_recipes

        fn = getattr(dataset_recipes, ds.recipe, None)
        if fn is None:
            return {}

        resolved = {
            name: p.default
            for name, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty
            and (
                isinstance(p.default, (str, int, float, bool, tuple, list))
                or p.default is None
            )
        }
        resolved.update(dict(ds.params))
        return _plain_json(resolved)
    except Exception:
        return {}


def dataset_signature(ds, null_spec_version: int = NULL_SPEC_VERSION) -> dict:
    """Everything about a DatasetSpec that determines what lands on disk.

    Split in two because the two halves have different blast radii when they
    change:

    ``construction``
        what the TRAINING data is. A change here invalidates every model
        trained on the instance — the instance must get a new name.

    ``controls``
        which evaluation-time features (target + A.1 nulls) are registered.
        A change here leaves X/y/split — and therefore every trained model —
        untouched, so the instance can be repaired in place by regenerating
        only its features (``scripts/refresh_null_features.py``).

    Recorded at persist time and compared by
    ``experiments.runner._assert_stored_params_match``.
    """
    return {
        "signature_version": SIGNATURE_VERSION,
        "construction": {
            "recipe": ds.recipe,
            "params": _plain_json(dict(ds.params)),
            # RESOLVED params: the recipe's own defaults merged under the
            # spec's overrides. Recording only `params` hid everything the
            # spec did not mention — exp1's params is {"seed": 0} while
            # make_halfplane_disk_circle_dataset carries eight geometry
            # defaults (centres, radii, ellipse axes, counts). Editing one of
            # those silently invalidated every stored instance with nothing to
            # detect it, and made halfplane_base and halfplane_base_smoke
            # byte-identical in signature (AUDIT/09 section P1).
            "resolved_params": _resolved_recipe_params(ds),
            # Only present when set, so specs that do not use it keep producing
            # byte-identical signatures and no stored instance is invalidated
            # by the field's existence.
            **({"stratify_split": ds.stratify_split}
               if getattr(ds, "stratify_split", None) else {}),
        },
        "controls": {
            "target_feature": ds.target_feature,
            "n_null_features": ds.n_null_features,
            "null_kind": ds.null_kind,
            "null_spec_version": null_spec_version,
        },
    }


def _plain_json(value):
    """Normalizes tuples->lists so torch/JSON round-trips compare equal."""
    if isinstance(value, (list, tuple)):
        return [_plain_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain_json(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class ArchSpec:
    """One MLP architecture. model_name is the run-directory tag.

    tag:
        optional suffix appended to model_name. The tag exists because the
        run directory is the ONLY thing separating model pools inside one
        dataset instance, and it otherwise encodes just activation/width/
        depth — so two runs of the same architecture with different training
        hyperparameters (lr, batch size, epoch budget) would land in one
        directory and be silently averaged together by the eval stage. Give
        such a run its own tag and the pools stay disjoint: the model-path
        filter matches "_{model_name}_model", so "tanh_w7_d5" no longer
        matches "..._tanh_w7_d5_lr3e4_model0".
    """

    width: int = 7
    depth: int = 5
    activation: str = "tanh"
    tag: str = ""

    @property
    def model_name(self) -> str:
        base = f"{self.activation}_w{self.width}_d{self.depth}"
        return f"{base}_{self.tag}" if self.tag else base


@dataclass(frozen=True)
class TrainSpec:
    """Passed through to train_many_mlps_on_dataset_instance.

    Gate semantics (``training.mass_training.train_one_mlp_until_accuracy``):
    training stops at the first epoch where ``epoch >= min_epochs`` AND
    ``test_acc >= target_test_acc`` has held for ``patience`` CONSECUTIVE
    epochs. Note that ``patience`` counts epochs *meeting* the gate — it is not
    a no-improvement / plateau counter.

    Defaults (raised 2026-08-16, AUDIT/04_repair_log.md R4). They were
    ``n_epochs=100, min_epochs=0, patience=1``, which stops at the FIRST epoch
    clearing the accuracy target. On a task a single hyperplane solves that
    fires on epoch 1 with the loss still at ~ln 2: the stored halfplane_base
    model0 had ``stopped_epoch=1``, i.e. weights essentially at initialization
    that happen to separate two well-spaced clusters. Witness statistics
    measured there describe a random network, not a trained one, so the
    accuracy floor is now backed by an epoch floor and a persistence
    requirement. Experiments that need something different still override the
    whole TrainSpec (exp1/exp2 do, with these same values).

    ``n_models`` is the SIZE OF THE ENSEMBLE, ``max_attempts`` the ceiling on
    the attempts allowed to fill it. Training stops at the first of the two:
    a cell whose success rate is 1 costs exactly ``n_models`` attempts, and a
    hard one may spend up to ``max_attempts``. A cell that burns the whole
    ceiling without filling up is reported loudly and left short — that is a
    statement about the task, not a budget to be raised silently.
    """

    n_models: int = 50
    max_attempts: int = 100
    target_test_acc: float = 0.99
    n_epochs: int = 300
    min_epochs: int = 50
    patience: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    optimizer_name: str = "adam"
    weight_decay: float = 0.0


@dataclass(frozen=True)
class EvalSpec:
    """Passed through to collect_compact_neuron_statistics.

    extra_criteria names are resolved to callables by
    experiments.runner.CRITERIA_REGISTRY ("extremum", "homological",
    "sampled_structural", ...); extra_criteria_kwargs optionally binds
    per-criterion keyword arguments to those callables
    ({criterion_name: {kwarg: value}}), e.g. the homological criterion's
    feature_homology.

    field_source selects where each neuron's shared FieldSample is
    evaluated: "grid" (default; resolution^d points — 2D/3D only) or
    "random" (n_field_points uniform points in bounds, seeded by
    field_seed — the d >= 3 route; run_structural then needs an explicit
    level_eps, see evaluate_neurons_on_feature).

    primary_witness_column names the CSV column stage-4 analyses treat as
    THE witness verdict (None -> "is_structural_witness"); experiments that
    disable the grid-structural criterion point it at e.g.
    "homological_is_witness". Every criterion listed above is written to the
    CSV regardless, so changing the verdict later re-runs the AGGREGATE
    stage only — no retraining, no re-evaluation.

    run_tubular defaults to False: the tubular (Wasserstein) criterion
    produces a bare score with no threshold and no boolean column, so no
    stage-4 analysis can consume it (they all need a witness column), while
    it costs ~40% of the evaluation time. It is kept in the codebase as the
    rejected value-baseline of PROJECT.md criterion 1 and can be switched
    back on per experiment.

    null_placements holds the A.1 negative control: a tuple of
    analysis.null_placement.Placement objects (rigid motions of the target
    feature, built with `at(...)` / `shift(...)`). The evaluation stage
    constructs the null features from them per run — they are NOT persisted
    into the dataset instance, because a control is not data and the stored
    instance should remain a faithful record of what the network was trained
    on.

    They live HERE, in the spec, rather than only in a notebook, for one
    reason: the spec is serialised to spec.json beside the results, so a CLI
    run records the exact control it used and stays reconstructible.
    analysis.lab.ExperimentLab.set_null_placements() overrides them freely for
    exploration without touching the committed definition.

    Placement is deliberate rather than random for COVERAGE. The null set is a
    sample from "features of this shape in this problem", and the floor it
    estimates is only as good as that sample: a uniform draw over the data
    bbox put all five nulls in the same empty corridor (AUDIT/08 section 5.1),
    which is one control reported five times. Chosen placements span the
    problem instead.

    What a null owes the target is its SHAPE — rigid motions give congruence
    for free — and representative coverage. Deliberately NOT more: matching a
    control to its target on one covariate after another terminates at the
    target itself, where nothing is being controlled. Consequently the spread
    of the score across placements is not a confound to be removed but the
    width of the floor, and the target is read against that width rather than
    against a pooled mean.

    Empty () means no A.1 control: correct for experiments that report no
    rate (exp6), and a defect for any experiment that does — the aggregate
    stage fails loudly in that case rather than reporting a gap of zero.
    For d >= 4, where placing by hand is not feasible, generate the tuple
    with analysis.null_placement.random_placements() and paste it in, so the
    draw is still recorded rather than regenerated per run.
    """

    resolution: int = 200
    bounds_margin: float = 0.5
    # Explicit evaluation box, ((x_lo, x_hi), (y_lo, y_hi), ...), shared by every
    # dataset of the spec. None (default) keeps the old behaviour: the box comes
    # from each dataset's own data plus bounds_margin, then grows to cover the
    # A.1 placements.
    #
    # It exists because that default makes the box a function of the DATASET and
    # of the null set, so two datasets meant to be compared can end up measured
    # by different instruments. Experiment 2 hit exactly that: dropping two null
    # placements shrank the box on simple_relevant (whose data sits at the
    # origin) from side 13.5 to 7.5 while leaving simple_irrelevant at 13.0,
    # which at a fixed `resolution` is a 1.7x difference in grid step, i.e. in
    # N_A = diam(A)/max(h) -- 79.6 against 45.9. Scores are not comparable
    # across such a difference.
    #
    # When set, bounds_margin is unused and the box is asserted to contain the
    # data and every feature: an explicit box that clips a feature would score it
    # against a truncated level set, silently.
    bounds: tuple[tuple[float, float], ...] | None = None
    kinds: tuple[str, ...] = ("post",)
    layer_ids: tuple[int, ...] | None = None
    run_structural: bool = True
    run_tubular: bool = False
    extra_criteria: tuple[str, ...] = ()
    structural_kwargs: dict = field(
        # activation=None pins the LEVEL GRID to the neuron's observed range,
        # linspace(min f, max f, n_levels). The key must be present even though
        # its value is None: evaluate_neurons_on_feature injects the model's own
        # activation (-> linspace(-1, 1, n_levels)) whenever "activation" is
        # absent from these kwargs, and that grid is not equivariant under
        # f -> lambda*f, so two neurons with identical level-set families but
        # different amplitudes would score differently. See docs/explanation.md.
        default_factory=lambda: {
            "level_eps": "adaptive", "threshold": 0.1, "activation": None
        }
    )
    tubular_kwargs: dict = field(default_factory=dict)
    extra_criteria_kwargs: dict = field(default_factory=dict)
    field_source: str = "grid"
    n_field_points: int = 20000
    field_seed: int = 0
    primary_witness_column: str | None = None
    null_placements: tuple = ()


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset instance family used by an experiment.

    name:
        dataset_type on disk: data/dataset_instances/<name>/<name>_<k>/.
    recipe:
        function name in data.dataset_recipes.
    params:
        recipe kwargs (seeded, so the instance is reproducible).
    target_feature:
        the feature the experiment is about (nulls are diameter-matched to
        it). None -> "C_mid" if the recipe registers it, else the first
        registered feature.
    n_null_features:
        how many matched random null features (A.1) the runner attaches
        at persist time via analysis.controls.make_null_features.
    null_kind:
        make_null_features kind. "shuffled" (default) reuses the target
        feature's OWN shape, randomly rotated and translated, so the null
        differs from the target only in placement — which is what A.1 asks
        for. "circle" plants a fresh circle of matched diameter instead (in
        dim > 2, inside a random 2-plane); that is equivalent to "shuffled"
        when the target IS a circle and wrong otherwise. An ellipse with
        axes (1.6, 1.0) has diameter 3.2, so its circle nulls have radius
        1.6 — bigger than the target across the minor axis, which makes
        rate[A] - rate[null] confound shape and size with relevance.
    eval_overrides:
        optional per-dataset overrides of EvalSpec fields
        ({field_name: value}, applied via dataclasses.replace; dict-valued
        fields are replaced wholesale, not merged). When any dataset of an
        experiment uses overrides, the eval stage runs per dataset and
        writes one CSV per (architecture x dataset). Needed when datasets
        of one experiment live in different ambient dimensions (exp. 9:
        R^3 and R^4 want different sample budgets).
    """

    name: str
    recipe: str
    params: dict = field(default_factory=dict)
    target_feature: str | None = None
    n_null_features: int = 0
    null_kind: str = "shuffled"
    eval_overrides: dict = field(default_factory=dict)
    # "component" -> stratify the train/test split by component instead of
    # drawing it uniformly. None (default) is the historical behaviour and
    # keeps every stored instance's signature unchanged.
    #
    # It exists because the uniform draw makes the accuracy gate vacuous once
    # a component is small: experiment 4's thinning sweep reaches 7 points on
    # the target circle, where an 80/20 draw leaves 1-2 of them in the test
    # set, and "100 % test accuracy" then certifies nothing about the circle.
    stratify_split: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    """One PLAN.md experiment: datasets x architectures x train x eval."""

    id: str
    title: str
    plan_ref: str  # e.g. "PLAN.md exp. 4"
    datasets: tuple[DatasetSpec, ...]
    archs: tuple[ArchSpec, ...] = (ArchSpec(),)
    train: TrainSpec = TrainSpec()
    eval: EvalSpec = EvalSpec()
    analyses: tuple[str, ...] = ("witness_tables",)
    notes: str = ""

    @property
    def results_dir(self) -> Path:
        return Path("analysis") / "results" / "experiments" / self.id


def apply_profile(spec: ExperimentSpec, profile: str | None) -> ExperimentSpec:
    """
    Rewrites a spec for one of the three launch profiles:

      "server" (or None) — the spec as registered: the real run
        (N=50 ensembles; train on DataSphere, eval/aggregate anywhere).

      "local" — a 1-model hypothesis check on the local machine. Shares the
        dataset instances (and hence the models/runs/ directory) with the
        server profile — a locally trained model is a legitimate member of
        the future ensemble — but writes results under <id>_local so a later
        full aggregate does not overwrite it.

      "smoke" — a sanity pass of the whole pipeline: 1 model, 2 epochs, no
        accuracy gate (target_test_acc=0 saves whatever came out), coarse
        eval grid. Uses SEPARATE dataset instances (<name>_smoke) so that
        throwaway checkpoints never sit next to real ones; everything it
        produces can be deleted by removing the *_smoke directories.

    The returned spec is persisted as usual, so results stay traceable to
    the exact profile that produced them.
    """
    if profile is None or profile == "server":
        return spec

    if profile == "local":
        return replace(
            spec,
            id=f"{spec.id}_local",
            train=replace(spec.train, n_models=1),
            notes=f"[profile=local] {spec.notes}",
        )

    if profile == "smoke":
        def _smoke_dataset(d: DatasetSpec) -> DatasetSpec:
            overrides = dict(d.eval_overrides)
            if "n_field_points" in overrides:
                overrides["n_field_points"] = min(
                    4000, overrides["n_field_points"]
                )
            return replace(d, name=f"{d.name}_smoke", eval_overrides=overrides)

        return replace(
            spec,
            id=f"{spec.id}_smoke",
            datasets=tuple(_smoke_dataset(d) for d in spec.datasets),
            train=replace(
                spec.train,
                n_models=1,
                n_epochs=2,
                min_epochs=1,
                patience=1,
                target_test_acc=0.0,
            ),
            eval=replace(
                spec.eval,
                resolution=60,
                n_field_points=min(4000, spec.eval.n_field_points),
            ),
            notes=f"[profile=smoke] {spec.notes}",
        )

    raise ValueError(f"unknown profile {profile!r}; known: {PROFILES}")


def spec_to_dict(spec: ExperimentSpec) -> dict:
    return asdict(spec)


def spec_from_dict(raw: dict) -> ExperimentSpec:
    """Inverse of spec_to_dict.

    JSON has no tuples, so every tuple-typed field is restored here; the
    result compares equal to the spec that was saved. One asymmetry survives
    on purpose: tuples nested inside DatasetSpec.params (e.g. circle_axes)
    come back as lists, which every recipe accepts.
    """
    eval_raw = dict(raw.get("eval", {}))

    for key in ("kinds", "extra_criteria"):
        if eval_raw.get(key) is not None:
            eval_raw[key] = tuple(eval_raw[key])

    if eval_raw.get("layer_ids") is not None:
        eval_raw["layer_ids"] = tuple(eval_raw["layer_ids"])

    if eval_raw.get("bounds") is not None:
        eval_raw["bounds"] = tuple(tuple(float(v) for v in axis)
                                   for axis in eval_raw["bounds"])

    # A.1 placements round-trip through their record form. asdict() flattened
    # each Placement to a dict on the way out and JSON turned its `vector`
    # tuple into a list; placements_from_records restores both, so a spec
    # reloaded from disk rebuilds byte-identical null features.
    # JSON object keys are always strings, so feature_homology={1: 1} — the
    # homology signature exp10 uses — reloads as {"1": 1}. The criterion then
    # looks up diagrams.get("1") against an int-keyed dict, finds nothing, and
    # reports "no bars" for every neuron: a SILENT wrong answer on the
    # --spec-json route, not an error. Restore the integer keys.
    kwargs_raw = eval_raw.get("extra_criteria_kwargs") or {}
    if "homological" in kwargs_raw:
        fh = kwargs_raw["homological"].get("feature_homology")
        if isinstance(fh, dict):
            kwargs_raw["homological"] = {
                **kwargs_raw["homological"],
                "feature_homology": {
                    (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): v
                    for k, v in fh.items()
                },
            }
            eval_raw["extra_criteria_kwargs"] = kwargs_raw

    if eval_raw.get("null_placements") is not None:
        from analysis.null_placement import placements_from_records

        # Always a tuple, including the empty case: JSON has no tuples, so an
        # absent control round-tripped to [] and the reloaded spec compared
        # unequal to the one that was saved.
        eval_raw["null_placements"] = tuple(
            placements_from_records(eval_raw["null_placements"])
        )

    # The same restoration for PER-DATASET overrides. ``eval_overrides`` is a
    # plain dict carried verbatim into DatasetSpec, so none of the work above
    # reaches it, and a spec that keeps its placements THERE — experiment 5
    # gives every cell its own — reloaded them as bare dicts. The failure then
    # surfaced only inside the shard subprocess, which is the one place that
    # reloads a spec from JSON, and only after training had finished.
    for _d in raw.get("datasets", ()):
        _ov = _d.get("eval_overrides")
        if not _ov:
            continue
        _ov = dict(_ov)
        if _ov.get("null_placements") is not None:
            from analysis.null_placement import placements_from_records

            _ov["null_placements"] = tuple(
                placements_from_records(_ov["null_placements"])
            )
        if _ov.get("bounds") is not None:
            _ov["bounds"] = [tuple(float(v) for v in axis)
                             for axis in _ov["bounds"]]
        _d["eval_overrides"] = _ov

    archs = tuple(ArchSpec(**a) for a in raw.get("archs", ()))

    return ExperimentSpec(
        id=raw["id"],
        title=raw["title"],
        plan_ref=raw["plan_ref"],
        datasets=tuple(DatasetSpec(**d) for d in raw["datasets"]),
        archs=archs or (ArchSpec(),),
        train=TrainSpec(**raw.get("train", {})),
        eval=EvalSpec(**eval_raw),
        analyses=tuple(raw.get("analyses", ("witness_tables",))),
        notes=raw.get("notes", ""),
    )


def save_spec(spec: ExperimentSpec, path: str | Path | None = None) -> Path:
    """Persists the spec as JSON (default: <results_dir>/spec.json)."""
    path = Path(path) if path is not None else spec.results_dir / "spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(spec_to_dict(spec), f, indent=2, ensure_ascii=False)

    return path


def load_spec(path: str | Path) -> ExperimentSpec:
    """Reads a spec JSON back into an ExperimentSpec.

    This is what makes an ad-hoc run reproducible without editing the
    registry: build a spec however you like (typically
    ``dataclasses.replace`` over a registered one), ``save_spec`` it, then
    drive it anywhere with ``run_experiment.py --spec-json <path>``. Profiles
    are NOT re-applied on load — a saved spec is already the resolved one.
    """
    path = Path(path)
    assert path.exists(), f"no spec JSON at {path}"

    with open(path) as f:
        return spec_from_dict(json.load(f))
