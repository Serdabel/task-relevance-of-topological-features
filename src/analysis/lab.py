"""Single-model experiment lab: the interactive, local research path.

``ExperimentLab`` is the thin engine behind the per-experiment notebooks
(``notebooks/experiments/<id>.ipynb``). It wraps the existing pipeline
(recipes -> training -> criteria -> plots) into one object whose whole
configuration is a handful of plain attributes, so a notebook cell can tune
every hyperparameter and then call named methods instead of re-deriving the
wiring each time.

It is the LOCAL, single-model counterpart to the batch harness
(``experiments.runner`` + ``scripts/run_experiment.py``, which trains ~50-model
ensembles and writes CSVs): same underlying functions, but the model is kept
in memory and the criteria feed straight into heatmaps, for hyperparameter
tuning and eyeballing results. Models are still saved to
``models/runs/<dataset>/`` (honest runs, reusable later as ensemble members).

Typical use (see the notebooks)::

    lab = ExperimentLab("exp1_sanity_check", dataset="simple_relevant")
    lab.width, lab.depth, lab.n_epochs = 7, 5, 100   # any knob
    lab.prepare()          # build + persist the dataset instance (idempotent)
    lab.fit()              # train one model, save it, show training curves
    lab.show_neuron_fields()   # the per-neuron field heatmap grid
    lab.criterion_table()      # per-neuron criterion numbers, witnesses marked

Escape hatches: ``lab.model``, ``lab.checkpoint``, ``lab.rows`` (the raw
``evaluate_neurons_on_feature`` output), ``lab.loaded`` (the dataset dict) and
``lab.bounds()`` are all public, so any raw library call remains one line away.

The 2D result views (``show_neuron_fields`` / ``show_neuron`` /
``show_decision_boundary``) require a 2D input space; higher-dimensional
experiments (exp. 9/10) use the homological/sampled criteria and PCA slices
instead — those get their own template.
"""

from dataclasses import replace
from pathlib import Path

import torch

from experiments.registry import EXPERIMENTS
from experiments import runner
from experiments.configs import ArchSpec


class ExperimentLab:
    """One (experiment, dataset, architecture) research session.

    All configuration is plain attributes, seeded from the experiment's
    registered spec and freely overridable before ``prepare``/``fit``:

    Architecture:  ``width``, ``depth``, ``activation``
    Training:      ``n_models``, ``n_epochs``, ``min_epochs``, ``patience``,
                   ``batch_size``, ``lr``, ``optimizer``, ``weight_decay``,
                   ``target_acc``
    Evaluation:    ``resolution`` (criterion grid), ``bounds_margin``
    Data:          ``dataset_overrides`` (recipe-kwarg overrides -> a separate
                   labelled instance so the canonical data is never clobbered),
                   ``dataset_label``

    The eval criteria themselves (structural/tubular/extremum/homological,
    ``field_source``, null features, the primary witness column) come from the
    experiment's ``EvalSpec`` and ``DatasetSpec`` and are applied for you.
    """

    def __init__(
        self,
        experiment_id: str,
        dataset: str | None = None,
        instance_id: int = 0,
        device: str = "cpu",
    ):
        assert experiment_id in EXPERIMENTS, (
            f"unknown experiment {experiment_id!r}; known: {sorted(EXPERIMENTS)}"
        )
        self.experiment_id = experiment_id
        self.spec = EXPERIMENTS[experiment_id]
        self.instance_id = instance_id
        self.device = device

        # Which dataset of the experiment (several for sweeps/abc series).
        names = [d.name for d in self.spec.datasets]
        if dataset is None:
            dataset = names[0]
        assert dataset in names, (
            f"{experiment_id} has no dataset {dataset!r}; choose from {names}"
        )
        self._dataset_spec = next(d for d in self.spec.datasets if d.name == dataset)

        # Architecture defaults from the spec's first arch.
        arch0 = self.spec.archs[0]
        self.width = arch0.width
        self.depth = arch0.depth
        self.activation = arch0.activation

        # Training defaults from the spec's TrainSpec (n_models=1: local research).
        t = self.spec.train
        self.n_models = 1
        self.n_epochs = t.n_epochs
        self.min_epochs = t.min_epochs
        self.patience = t.patience
        self.batch_size = t.batch_size
        self.lr = t.lr
        self.optimizer = t.optimizer_name
        self.weight_decay = t.weight_decay
        self.target_acc = t.target_test_acc

        # Evaluation defaults from the spec's EvalSpec.
        e = self.spec.eval
        eval_res = e.resolution
        self.resolution = eval_res if isinstance(eval_res, int) else max(eval_res)
        self.bounds_margin = e.bounds_margin

        # Data overrides (empty -> the canonical instance).
        self.dataset_overrides: dict = {}
        self.dataset_label: str | None = None

        # State filled by prepare/fit/load/evaluate.
        self.model = None
        self.checkpoint = None
        self.loaded = None
        self._rows_cache: dict = {}

        # A.1 controls chosen for this session via set_null_placements().
        # Deliberately in memory only — a null is a control, not data.
        self._local_features: dict = {}
        self._local_placements: list = []

    # ------------------------------------------------------------------
    # Resolved names / paths
    # ------------------------------------------------------------------

    @property
    def arch(self) -> ArchSpec:
        return ArchSpec(width=self.width, depth=self.depth, activation=self.activation)

    @property
    def arch_tag(self) -> str:
        """model_name tag, e.g. 'tanh_w7_d5' (separates archs in one dir)."""
        return self.arch.model_name

    @property
    def dataset_name(self) -> str:
        """Instance type on disk — canonical, or a labelled variant when
        dataset_overrides and/or dataset_label are set (so custom data never
        clobbers canonical). A label WITHOUT overrides makes a fresh instance
        of the same recipe — e.g. exp-8's re-persist-with-nulls variants next
        to null-less canonical instances."""
        base = self._dataset_spec.name
        if not self.dataset_overrides and self.dataset_label is None:
            return base
        label = self.dataset_label or "custom"
        return f"{base}__{label}"

    @property
    def instance_path(self) -> str:
        return runner.dataset_instance_path(self.dataset_name, self.instance_id)

    @property
    def target_feature(self) -> str:
        """The feature the experiment is about (nulls are matched to it)."""
        tf = self._dataset_spec.target_feature
        if tf is not None:
            return tf
        if self.loaded is not None:
            meta_tf = self.loaded.get("metadata", {}).get("target_feature")
            if meta_tf:
                return meta_tf
            return next(iter(self.loaded["features"]))
        return "C_mid"

    @property
    def primary_witness_column(self) -> str:
        return self.spec.eval.primary_witness_column or "is_structural_witness"

    # ------------------------------------------------------------------
    # Stage 1: data
    # ------------------------------------------------------------------

    def prepare(self, force: bool = False):
        """Build and persist the dataset instance (idempotent; nulls attached).

        With ``dataset_overrides`` set, a separate labelled instance is built
        (canonical data is never touched). ``force=True`` regenerates an
        existing instance (deletes it first).
        """
        import shutil

        if force and Path(self.instance_path).exists():
            shutil.rmtree(Path(self.instance_path).parent)

        ds = self._dataset_spec
        # Rename whenever the resolved name differs from the spec's — which is
        # true for overrides AND for a bare dataset_label. Keying this on
        # `dataset_overrides` alone made a label-only lab persist under the
        # CANONICAL name while `instance_path` pointed at the labelled one: the
        # canonical instance was silently reused, nothing was created, and the
        # next call failed with "no dataset instance at ..." naming a path
        # prepare() had just claimed to build.
        if self.dataset_name != ds.name:
            ds = replace(
                ds,
                name=self.dataset_name,
                params={**ds.params, **self.dataset_overrides},
            )

        # A one-dataset spec so runner.prepare_datasets reuses its null-attach
        # logic without touching the experiment's other datasets; the __lab id
        # keeps its spec.json out of the experiment's canonical results dir.
        one = replace(self.spec, id=f"{self.spec.id}__lab", datasets=(ds,))
        runner.prepare_datasets(one)
        self.loaded = None  # force reload
        return self.instance_path

    # ------------------------------------------------------------------
    # Stage 2: training
    # ------------------------------------------------------------------

    def fit(self, plot: bool = True, verbose: bool = True):
        """Train ``n_models`` model(s) on the instance, save them, and load the
        last one into ``self.model``. Returns the run summaries.

        Models are always saved (``save_successful_only=False``) — a run that
        misses the accuracy target is still worth inspecting, and honest runs
        that hit it become ensemble members.
        """
        from training.mass_training import train_many_mlps_on_dataset_instance

        assert Path(self.instance_path).exists(), (
            f"no dataset instance at {self.instance_path} — call prepare() first"
        )

        summaries = train_many_mlps_on_dataset_instance(
            self.instance_path,
            n_models=self.n_models,
            activation=self.activation,
            hidden_dim=self.width,
            n_layers=self.depth,
            model_name=self.arch_tag,
            target_test_acc=self.target_acc,
            n_epochs=self.n_epochs,
            min_epochs=self.min_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            lr=self.lr,
            optimizer_name=self.optimizer,
            weight_decay=self.weight_decay,
            save_successful_only=False,
            device=self.device,
            verbose=verbose,
        )

        last_path = summaries[-1]["model_path"]
        self.load_path(last_path)
        self._rows_cache.clear()

        if plot:
            self.show_training()

        return summaries

    def load(self, index: int = -1):
        """Load a previously saved model of this architecture (no training).

        ``index`` selects among saved runs of ``arch_tag`` (default -1: latest).
        """
        from analysis.stat_aggregation import find_model_paths_for_dataset

        paths = find_model_paths_for_dataset(
            self.dataset_name,
            instance_id=self.instance_id,
            model_name=self.arch_tag,
        )
        assert paths, (
            f"no saved {self.arch_tag} models for {self.dataset_name}; fit() first"
        )
        return self.load_path(paths[index])

    def load_path(self, model_path: str):
        """Load a specific saved model directory into ``self.model``."""
        from management.experiment_management import load_model_instance

        self.model, self.checkpoint = load_model_instance(
            model_path, device=self.device, eval_mode=True
        )
        self.model_path = model_path
        self._rows_cache.clear()
        return self.model

    # ------------------------------------------------------------------
    # Dataset accessors
    # ------------------------------------------------------------------

    def dataset(self) -> dict:
        """The loaded instance dict (X, y, features, metadata, ...), cached."""
        if self.loaded is None:
            from management.experiment_management import load_dataset_instance

            assert Path(self.instance_path).exists(), (
                f"no dataset instance at {self.instance_path} — call prepare()"
            )
            self.loaded = load_dataset_instance(
                self.instance_path, map_location="cpu", reconstruct_data=False
            )
        return self.loaded

    def feature_points(self, name: str | None = None) -> torch.Tensor:
        from management.experiment_management import get_feature_points

        name = name or self.target_feature
        if name in self._local_features:
            return self._local_features[name]

        return get_feature_points(self.dataset(), name)

    def set_null_placements(self, placements, prefix: str = "null"):
        """Choose the A.1 controls for THIS session, by hand (2D/3D).

        ``placements`` is a list of ``analysis.null_placement.at`` / ``shift``
        objects: a rotation about the shape's own centroid plus a translation,
        applied to the target feature. Because rigid motions are isometries the
        controls are congruent to the target by construction — same shape, same
        diameter, same point count — so they differ from it only in placement,
        which is the whole requirement on an A.1 null.

        These features are NOT written to the dataset instance. A null is a
        control, not data: keeping it out of ``structure.pt`` means the stored
        instance stays a faithful record of what the network was trained on,
        and the control can be re-chosen freely without a provenance refresh.
        The placement list is what makes the run reproducible, so print or
        commit it alongside any number you keep.

        Placing them by hand buys COVERAGE. The null set is a sample from
        "features of this shape in this problem", so it has to span the
        problem — on each cluster, in the gap between them, beside the target,
        outside the data — for the floor it estimates to mean anything. A
        uniform draw put all of them in one empty corridor
        (``AUDIT/08_independent_review.md`` section 5.1): one control reported
        five times.

        What a null owes the target is its shape and that coverage, and
        nothing further — chasing agreement on covariate after covariate ends
        at the target itself, where no control is left. So read the spread
        across placements as the width of the floor, not as a confound.

        Returns the ``placement_report`` table.
        """
        from analysis.null_placement import build_null_features, placement_report

        target = self.feature_points(self.target_feature)
        self._local_features = build_null_features(
            target, placements, prefix=prefix
        )
        self._local_placements = list(placements)
        self._rows_cache.clear()   # the bounds may have moved; nothing survives

        loaded = self.dataset()
        return placement_report(
            target,
            self._local_features,
            X=loaded["X"],
            y=loaded.get("y"),
            bounds=self.bounds(),
        )

    @property
    def null_features(self) -> dict:
        """The run-local A.1 controls, {name: points} (empty until placed)."""
        return dict(self._local_features)

    def bounds(self):
        from analysis.stat_aggregation import bounds_from_X
        from analysis.null_placement import bounds_covering

        b = bounds_from_X(self.dataset()["X"], margin=self.bounds_margin)

        # Expand-to-fit, minimally: a placement may legitimately sit outside
        # the data box ("far from every cluster" is a regime worth sampling),
        # and a feature partly outside the box would be scored against a
        # truncated level set. Only as far as the placements require — the box
        # also fixes the level-sweep range and the adaptive level_eps
        # statistics for the TARGET, so growing it is never free.
        if self._local_features:
            b = bounds_covering(
                b, list(self._local_features.values()), margin=self.bounds_margin
            )
        return b

    @property
    def in_dim(self) -> int:
        return int(self.dataset()["X"].shape[1])

    # ------------------------------------------------------------------
    # Stage 3: per-neuron evaluation (in memory)
    # ------------------------------------------------------------------

    def evaluate(self, feature: str | None = None, full: bool = True):
        """Run the experiment's criteria over every neuron for one feature and
        return the raw ``evaluate_neurons_on_feature`` rows (list of dicts),
        cached per (feature, full).

        The criteria set / field source / extra criteria all come from the
        experiment's EvalSpec. ``full=True`` additionally returns the best
        structural level-set component per neuron (what the heatmap grid draws);
        ``full=False`` is the lean version used for the target-vs-null tallies.
        """
        from analysis.criterions import evaluate_neurons_on_feature

        feature = feature or self.target_feature
        key = (feature, full)
        if key in self._rows_cache:
            return self._rows_cache[key]

        assert self.model is not None, "no model — call fit() or load() first"

        e = self.spec.eval
        structural_kwargs = dict(e.structural_kwargs)
        if full and e.run_structural:
            structural_kwargs["return_best_component_points"] = True

        rows = evaluate_neurons_on_feature(
            model=self.model,
            feature_points=self.feature_points(feature),
            bounds=self.bounds(),
            resolution=self.resolution,
            kinds=e.kinds,
            device=self.device,
            structural_kwargs=structural_kwargs,
            value_kwargs=dict(e.tubular_kwargs),
            run_structural=e.run_structural,
            run_tubular=e.run_tubular,
            field_source=e.field_source,
            n_field_points=e.n_field_points,
            field_seed=e.field_seed,
            extra_criteria=runner._resolve_extra_criteria(e) or None,
            return_full_result_structural=full and e.run_structural,
            verbose=False,
        )
        self._rows_cache[key] = rows
        return rows

    @property
    def rows(self):
        """The evaluation rows for the target feature (full), evaluated lazily."""
        return self.evaluate()

    def criterion_table(self, feature: str | None = None, style: bool = True):
        """Per-neuron criterion values as a DataFrame, witnesses highlighted.

        One row per (layer, kind, neuron): the witness verdict column, the
        scores of whichever criteria the EvalSpec ran, and any
        extra-criterion numeric columns.
        Also stored raw on ``self.criterion_df``. With ``style=True`` returns a
        pandas Styler (green = witness) for display; ``style=False`` the plain
        DataFrame.
        """
        import pandas as pd

        rows = self.evaluate(feature=feature, full=False)

        keep_first = ["layer_id", "kind", "coord_id"]
        # Numeric criterion columns present in the rows, in a stable order.
        preferred = [
            self.primary_witness_column,
            "is_structural_witness",
            "structural_score_normalized_by_diameter",
            "tubular_score",
        ]
        extra = sorted(
            k
            for r in rows
            for k, v in r.items()
            if k not in keep_first
            and k not in preferred
            and isinstance(v, (int, float, bool))
        )
        cols = keep_first + [c for c in preferred if any(c in r for r in rows)]
        cols += [c for c in extra if c not in cols]

        df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
        self.criterion_df = df

        if not style:
            return df

        wcol = self.primary_witness_column
        if wcol not in df.columns:
            return df

        def _hl(row):
            is_w = bool(row.get(wcol))
            return ["background-color: #d6f5d6" if is_w else "" for _ in row]

        return df.style.apply(_hl, axis=1).format(precision=4)

    def witness_summary(self, style: bool = True):
        """Target-vs-null witness tally for THIS model (the experiment's core
        comparison, at single-model scale).

        For the target feature and every null feature: how many neurons witness
        it (by the primary witness column) and the fraction. A task-irrelevant
        target (exp. 1-2) should not beat the nulls; a task-relevant one
        (exp. 3+) should. The ensemble version with bootstrap CIs is the
        server run (analysis.controls.witness_rate_gap).

        Returned as a DataFrame (also on ``self.witness_df``); ``style=True``
        highlights the target row.
        """
        import pandas as pd

        from analysis.stat_aggregation import feature_role_from_metadata

        loaded = self.dataset()
        wcol = self.primary_witness_column

        # Stored features, plus whatever set_null_placements() built for this
        # session. Run-local nulls are listed with role "null" directly: they
        # have no metadata on disk to infer a role from, by design.
        names = [(n, None) for n in loaded["features"]]
        names += [(n, "null") for n in self._local_features]

        rows = []
        for name, forced_role in names:
            role = forced_role or feature_role_from_metadata(loaded, name)
            evals = self.evaluate(feature=name, full=False)
            witness = [bool(r.get(wcol)) for r in evals if wcol in r]
            n = len(witness)
            n_w = int(sum(witness))
            rows.append(
                {
                    "feature": name,
                    "role": role,
                    "n_neurons": n,
                    "n_witnesses": n_w,
                    "witness_rate": (n_w / n) if n else float("nan"),
                }
            )

        df = pd.DataFrame(rows).sort_values(
            ["role", "feature"]
        ).reset_index(drop=True)
        self.witness_df = df

        if not style:
            return df

        def _hl(row):
            is_target = row["role"] == "target"
            return ["font-weight: bold" if is_target else "" for _ in row]

        return df.style.apply(_hl, axis=1).format(precision=4)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show_dataset(self, size=8, show_nulls: bool = False):
        """Scatter the point cloud (coloured by class), the target feature
        outlined. 2D static; 3D falls back to the interactive plotly view.

        ``show_nulls=True`` also draws the A.1 null features — the random
        same-diameter shapes the target's witness rate is measured against.
        Worth a look once per dataset: a null that landed on top of the real
        feature makes the gap it anchors meaningless.
        """
        import matplotlib.pyplot as plt

        if self.in_dim == 3:
            from visualization.plot_point_clouds import (
                plot_point_cloud_3d_interactive,
            )

            loaded = self.dataset()
            return plot_point_cloud_3d_interactive(
                loaded["X"], loaded["y"],
                level_points=self.feature_points(),
                title=f"{self.dataset_name} (small dots: target feature)",
            )

        self._assert_2d("show_dataset")
        loaded = self.dataset()
        X = loaded["X"].cpu().numpy()
        y = loaded["y"].cpu().numpy()
        feat = self.feature_points().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        for cls in sorted(set(y.tolist())):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=size, label=f"class {cls}", alpha=0.7,
                       linewidths=0)

        if show_nulls:
            from analysis.stat_aggregation import feature_role_from_metadata

            n_nulls = 0
            for feature_name in loaded.get("features", {}):
                if feature_role_from_metadata(loaded, feature_name) != "null":
                    continue
                n_nulls += 1
                null_points = self.feature_points(feature_name).cpu().numpy()
                ax.scatter(
                    null_points[:, 0], null_points[:, 1], s=size,
                    facecolors="none", edgecolors="#777777", linewidths=0.6,
                    alpha=0.7,
                    label="null features (A.1)" if n_nulls == 1 else None,
                )

        ax.scatter(feat[:, 0], feat[:, 1], s=size + 10, facecolors="none",
                   edgecolors="crimson", linewidths=1.0,
                   label=f"target: {self.target_feature}")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{self.dataset_name}  (target feature outlined)")
        plt.show()

    def show_training(self):
        """Loss and accuracy curves of the loaded model (from its checkpoint)."""
        import matplotlib.pyplot as plt

        m = (self.checkpoint or {}).get("metrics", {})
        assert m.get("train_losses"), (
            "no training history on the loaded checkpoint — fit() records it"
        )
        epochs = range(1, len(m["train_losses"]) + 1)

        fig, (axl, axa) = plt.subplots(1, 2, figsize=(11, 4))
        axl.plot(epochs, m["train_losses"], label="train")
        axl.plot(epochs, m["test_losses"], label="test")
        axl.set_xlabel("epoch"); axl.set_ylabel("BCE loss"); axl.legend()
        axl.set_title("loss")

        axa.plot(epochs, m["train_accs"], label="train")
        axa.plot(epochs, m["test_accs"], label="test")
        axa.axhline(self.target_acc, ls="--", c="grey", lw=1,
                    label=f"target {self.target_acc:g}")
        axa.set_xlabel("epoch"); axa.set_ylabel("accuracy"); axa.legend()
        axa.set_title(
            f"accuracy  (best {m.get('best_test_acc', float('nan')):.4f}, "
            f"reached target: {m.get('reached_target')})"
        )
        fig.tight_layout()
        plt.show()

    def show_decision_boundary(self, resolution: int | None = None):
        """Heatmap of the model's class-1 probability over the plane, data
        overlaid — what the whole network learned. 2D only."""
        import matplotlib.pyplot as plt
        from analysis.geometry import make_grid

        self._assert_2d("show_decision_boundary")
        assert self.model is not None, "no model — fit() or load() first"
        res = resolution or self.resolution
        bounds = self.bounds()

        grid, shape = make_grid(bounds=bounds, resolution=res, device=self.device)
        with torch.no_grad():
            logits = self.model(grid).view(-1)
            # make_grid is x-major (Z[ix, iy]); imshow wants Z[iy, ix] -> .T
            prob = torch.sigmoid(logits).reshape(shape).T.cpu().numpy()

        loaded = self.dataset()
        X = loaded["X"].cpu().numpy(); y = loaded["y"].cpu().numpy()
        (x0, x1), (y0, y1) = bounds[0], bounds[1]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(prob, extent=[x0, x1, y0, y1], origin="lower",
                       cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
        ax.contour(prob, levels=[0.5], extent=[x0, x1, y0, y1], colors="k",
                   linewidths=1.0)
        for cls in sorted(set(y.tolist())):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=6, alpha=0.6, linewidths=0,
                       edgecolors="none")
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(class 1)")
        ax.set_title("learned decision boundary")
        plt.show()

    def show_neuron_fields(self, feature: str | None = None):
        """The centrepiece: every hidden neuron's field as a heatmap, with the
        data, the best structural level-set component, and the criterion scores
        overlaid. 2D + structural criterion only (the d>=3 route is different)."""
        from visualization.plot_point_clouds import (
            plot_all_hidden_neuron_fields_2d_with_criteria,
        )

        self._assert_2d("show_neuron_fields")
        assert self.spec.eval.run_structural, (
            "show_neuron_fields needs the structural criterion; this experiment "
            "runs run_structural=False (hi-dim) — use the homological/sampled "
            "criteria and PCA slices instead"
        )
        rows = self.evaluate(feature=feature, full=True)
        loaded = self.dataset()
        plot_all_hidden_neuron_fields_2d_with_criteria(
            self.model,
            rows,
            self.bounds(),
            X=loaded["X"],
            y=loaded["y"],
            kind=self.spec.eval.kinds[0],
            resolution=self.resolution,
            device=self.device,
            title=f"{self.dataset_name} — {self.arch_tag} — "
                  f"neuron fields vs {feature or self.target_feature}",
        )

    def _neuron_function(self, layer: int, coord: int, kind: str | None = None):
        from models.NN_analysis import TruncatedMLPFunction, ScalarProjection

        kind = kind or self.spec.eval.kinds[0]
        return ScalarProjection(
            TruncatedMLPFunction(self.model, layer_id=layer, kind=kind),
            mode="coord",
            coord_id=coord,
        )

    def random_field(self, layer: int, coord: int,
                     n_points: int | None = None, seed: int | None = None):
        """One neuron's FieldSample on a seeded uniform random cloud inside
        the bounds — the same kind of shared sample the d >= 3 eval path uses
        (defaults mirror the spec's n_field_points / field_seed). Any
        dimension; the input to the scalar-summary gallery views."""
        from analysis.fields import field_from_points

        e = self.spec.eval
        n_points = n_points or e.n_field_points
        seed = e.field_seed if seed is None else seed

        generator = torch.Generator()
        generator.manual_seed(seed)

        bounds = self.bounds()
        lows = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        highs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        points = lows + (highs - lows) * torch.rand(
            n_points, len(bounds), generator=generator
        )

        return field_from_points(
            self._neuron_function(layer, coord), points.to(self.device),
            bounds=bounds,
        )

    def show_neuron(self, layer: int, coord: int, feature: str | None = None,
                    resolution: int | None = None):
        """Static drill-down on one neuron: its field heatmap, the target
        feature, and the best structural level-set component overlaid.
        See ``show_neuron_interactive`` for a zoomable plotly version."""
        import matplotlib.pyplot as plt

        self._assert_2d("show_neuron")
        res = resolution or self.resolution
        bounds = self.bounds()
        f = self._neuron_function(layer, coord)

        from analysis.geometry import make_grid
        grid, shape = make_grid(bounds=bounds, resolution=res, device=self.device)
        with torch.no_grad():
            # make_grid is x-major (Z[ix, iy]); imshow wants Z[iy, ix] -> .T
            Z = f(grid).reshape(shape).T.cpu().numpy()

        feat = self.feature_points(feature).cpu().numpy()
        (x0, x1), (y0, y1) = bounds[0], bounds[1]

        # Best structural component for this neuron, if available.
        best = None
        for r in self.evaluate(feature=feature, full=True):
            if r["layer_id"] == layer and r["coord_id"] == coord:
                sr = r.get("structural_result", {})
                bp = sr.get("best_component_points")
                if bp is not None:
                    best = bp.cpu().numpy()
                break

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(Z, extent=[x0, x1, y0, y1], origin="lower",
                       cmap="viridis", aspect="equal")
        ax.scatter(feat[:, 0], feat[:, 1], s=6, c="white", alpha=0.7,
                   linewidths=0, label="target feature")
        if best is not None and len(best):
            ax.scatter(best[:, 0], best[:, 1], s=8, c="crimson", linewidths=0,
                       label="best level-set component")
        fig.colorbar(im, ax=ax, fraction=0.046, label="activation")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"neuron L{layer} · N{coord}")
        plt.show()

    def show_neuron_interactive(self, layer: int, coord: int,
                                feature: str | None = None,
                                resolution: int | None = None):
        """Interactive (plotly) field heatmap + contours for one neuron, with
        data and target feature — zoom/hover in a live notebook."""
        from visualization.plot_point_clouds import plot_scalar_field_2d_interactive

        self._assert_2d("show_neuron_interactive")
        loaded = self.dataset()
        plot_scalar_field_2d_interactive(
            self._neuron_function(layer, coord),
            self.bounds(),
            resolution=resolution or self.resolution,
            X=loaded["X"],
            y=loaded["y"],
            level_points=self.feature_points(feature),
            device=self.device,
            title=f"neuron L{layer} · N{coord}",
        )

    def show_witness_summary(self):
        """Bar chart of witness rate per feature (target vs nulls) for this
        model — the visual form of ``witness_summary``."""
        import matplotlib.pyplot as plt

        df = self.witness_summary(style=False)
        colors = {"target": "crimson", "null": "grey"}
        bar_colors = [colors.get(r, "steelblue") for r in df["role"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["feature"], df["witness_rate"], color=bar_colors)
        ax.set_ylabel(f"witness rate ({self.primary_witness_column})")
        ax.set_title(f"{self.dataset_name} — {self.arch_tag}: target vs nulls "
                     "(one model)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        plt.show()

    def layer_witness_profile(self, feature: str | None = None, plot: bool = True):
        """Per-layer witness tally for THIS model (exp-3's layer question).

        Returns a DataFrame [layer_id, n_witnesses, n_neurons, fraction] by
        the primary witness column, and checks the PLAN.md exp-3 hypothesis
        "a witness appearing at layer l persists at every layer > l":
        ``df.attrs["first_witness_layer"]`` (None if no witness) and
        ``df.attrs["persists"]`` (True/False/None). ``plot=True`` adds a bar
        chart of the per-layer fractions.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        wcol = self.primary_witness_column
        df = self.criterion_table(feature=feature, style=False)
        assert wcol in df.columns, f"no {wcol!r} column in the criterion table"

        w = df[wcol].fillna(False).astype(bool)
        prof = (
            pd.DataFrame({"layer_id": df["layer_id"], "witness": w})
            .groupby("layer_id")["witness"]
            .agg(n_witnesses="sum", n_neurons="count")
            .reset_index()
        )
        prof["fraction"] = prof["n_witnesses"] / prof["n_neurons"]

        has = prof["n_witnesses"].to_numpy() > 0
        if has.any():
            first = int(prof["layer_id"][has.argmax()])
            tail = has[list(has).index(True) + 1:]
            prof.attrs["first_witness_layer"] = first
            prof.attrs["persists"] = bool(tail.all()) if len(tail) else True
        else:
            prof.attrs["first_witness_layer"] = None
            prof.attrs["persists"] = None

        if plot:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(prof["layer_id"], prof["fraction"], color="steelblue")
            ax.set_xlabel("layer"); ax.set_ylabel("witness fraction")
            ax.set_xticks(prof["layer_id"])
            title = f"witnesses per layer ({feature or self.target_feature})"
            if prof.attrs["first_witness_layer"] is not None:
                title += (f"  |  first at L{prof.attrs['first_witness_layer']}, "
                          f"persists: {prof.attrs['persists']}")
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            plt.show()

        return prof

    def sparse_probe_table(self, ks=(1, 2), delta: float | None = None,
                           feature: str | None = None):
        """A.4 sparse-probing baseline vs the witness criteria, per layer.

        Fits cross-validated k-sparse logistic probes on each hidden layer's
        activations for the target "is this point within delta of A?"
        (delta default: 0.1 x diam(A)) and sets the per-layer structural
        witnesses alongside — the localized-vs-distributed reference of
        exp. 3: a feature with no single-neuron witness but a good small-k
        probe is encoded distributively.

        Returns a DataFrame [layer_id, k, selected_neurons, cv_auc,
        cv_accuracy, witnesses (list by the primary criterion), method].
        """
        import pandas as pd

        from analysis.geometry import feature_diameter
        from analysis.probing import (
            collect_activation_dataset,
            feature_membership_targets,
            fit_sparse_probe,
        )

        assert self.model is not None, "no model — fit() or load() first"

        loaded = self.dataset()
        X = loaded["X"]
        feat = self.feature_points(feature)

        if delta is None:
            delta = 0.1 * float(feature_diameter(feat, method="exact").item())

        targets = feature_membership_targets(X, feat, delta=delta)
        kind = self.spec.eval.kinds[0]
        acts = collect_activation_dataset(
            self.model, X, kinds=(kind,), device=self.device
        )

        # Per-layer witness lists by the primary criterion, for comparison.
        cdf = self.criterion_table(feature=feature, style=False)
        wcol = self.primary_witness_column
        witnesses_by_layer = {
            int(l): sorted(
                int(c) for c, w in zip(g["coord_id"], g[wcol].fillna(False))
                if bool(w)
            )
            for l, g in cdf.groupby("layer_id")
        }

        rows = []
        for (layer_id, _), H in sorted(acts.items()):
            for k in ks:
                if k > H.shape[1]:
                    continue
                probe = fit_sparse_probe(H, targets, k=k)
                rows.append(
                    {
                        "layer_id": layer_id,
                        "k": k,
                        "selected_neurons": probe["selected_neurons"],
                        "cv_auc": probe["cv_auc"],
                        "cv_accuracy": probe["cv_accuracy"],
                        "witnesses": witnesses_by_layer.get(layer_id, []),
                        "method": probe["method"],
                    }
                )

        df = pd.DataFrame(rows)
        df.attrs["delta"] = delta
        self.probe_df = df
        return df

    # ------------------------------------------------------------------
    # Multi-feature machinery (exp. 7 / A.3)
    # ------------------------------------------------------------------

    def witness_matrix(self, features: list[str] | None = None,
                       roles: tuple[str, ...] = ("target", "feature",
                                                 "diagnostic")):
        """Exp-7's core object: which neuron witnesses WHICH feature.

        Rows = (layer, kind, coord), columns = features (by default every
        non-null feature), values = the primary witness verdict. The final
        column ``n_features_witnessed`` counts per neuron — any value >= 2 is
        a MULTITASKING neuron (one level-set family serving several
        features). Also stored on ``self.witness_matrix_df``.
        """
        import pandas as pd

        from analysis.stat_aggregation import feature_role_from_metadata

        loaded = self.dataset()
        if features is None:
            features = [
                name for name in loaded["features"]
                if feature_role_from_metadata(loaded, name) in roles
            ]
        assert features, "no features to tabulate"

        wcol = self.primary_witness_column
        matrix = None
        for name in features:
            evals = self.evaluate(feature=name, full=False)
            col = pd.DataFrame(
                [{
                    "layer_id": r["layer_id"],
                    "kind": r["kind"],
                    "coord_id": r["coord_id"],
                    name: bool(r.get(wcol)),
                } for r in evals]
            )
            matrix = col if matrix is None else matrix.merge(
                col, on=["layer_id", "kind", "coord_id"]
            )

        matrix["n_features_witnessed"] = matrix[features].sum(axis=1)
        self.witness_matrix_df = matrix
        return matrix

    def show_witness_matrix(self, features: list[str] | None = None):
        """Heatmap of the witness matrix (neurons x features); multitasking
        neurons are the rows with more than one filled cell."""
        import matplotlib.pyplot as plt
        import numpy as np

        m = self.witness_matrix(features=features)
        feats = [c for c in m.columns
                 if c not in ("layer_id", "kind", "coord_id",
                              "n_features_witnessed")]
        M = m[feats].to_numpy(dtype=float)
        labels = [f"L{l}·N{c}" for l, c in zip(m["layer_id"], m["coord_id"])]

        fig, ax = plt.subplots(
            figsize=(1.1 * len(feats) + 3, 0.28 * len(labels) + 1.5)
        )
        ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(feats)), feats, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=6)
        n_multi = int((m["n_features_witnessed"] >= 2).sum())
        ax.set_title(f"witness matrix — {n_multi} multitasking neuron(s)",
                     fontsize=10)
        fig.tight_layout()
        plt.show()

    def ablation_report(self, layer: int, coord: int,
                        feature: str | None = None,
                        delta: float | None = None, mode: str = "zero"):
        """A.3: is this neuron FUNCTIONALLY responsible for the feature?

        Zeroes the neuron (forward hook) and measures the accuracy drop
        globally vs near/inside the feature (within ``delta``, default
        0.1 x diam(A)). ``selectivity = drop_near - drop_far`` > 0 means the
        damage is feature-selective — the witness is causally involved, not
        excessive. Returns the flat impact dict.
        """
        from analysis.ablation import ablation_impact
        from analysis.geometry import feature_diameter

        assert self.model is not None, "no model — fit() or load() first"
        loaded = self.dataset()
        feat = self.feature_points(feature)

        if delta is None:
            delta = 0.1 * float(feature_diameter(feat, method="exact").item())

        impact = ablation_impact(
            self.model, loaded["X"], loaded["y"], feat,
            layer_id=layer, coord_id=coord, delta=delta, mode=mode,
            device=self.device,
        )
        impact["layer_id"], impact["coord_id"], impact["delta"] = layer, coord, delta
        return impact

    # ------------------------------------------------------------------
    # Basin/ridge machinery (exp. 4 / exp. 6 / A.7)
    # ------------------------------------------------------------------

    def basin_ridge_table(self, feature: str | None = None,
                          n_restarts: int = 32, seed: int = 0):
        """Exp-6's direct measurement: basin-vs-ridge type of every witness
        of the target feature (by the primary witness column), via
        ``analysis.extremum_anchored.classify_witness_basin_or_ridge``.

        Returns a DataFrame [layer_id, coord_id, witness_type,
        radial_profile_monotonicity, ridge_location, ridge_prominence,
        r_feature, anchor_value, anchor_dist_to_center,
        anchor_dist_to_feature]; empty if this model has no witnesses.
        """
        import pandas as pd

        from analysis.extremum_anchored import classify_witness_basin_or_ridge

        self._assert_2d("basin_ridge_table")
        cdf = self.criterion_table(feature=feature, style=False)
        wcol = self.primary_witness_column
        witnesses = cdf[cdf[wcol].fillna(False).astype(bool)]

        rows = []
        for _, r in witnesses.iterrows():
            layer, coord = int(r["layer_id"]), int(r["coord_id"])
            typed = classify_witness_basin_or_ridge(
                self._neuron_function(layer, coord),
                self.feature_points(feature),
                self.bounds(),
                resolution=min(self.resolution, 100),
                n_restarts=n_restarts,
                seed=seed,
                device=self.device,
            )
            rows.append({"layer_id": layer, "coord_id": coord, **{
                k: typed[k] for k in (
                    "witness_type", "radial_profile_monotonicity",
                    "ridge_location", "ridge_prominence", "r_feature",
                    "anchor_value", "anchor_dist_to_center",
                    "anchor_dist_to_feature",
                )
            }})

        df = pd.DataFrame(rows)
        self.typing_df = df
        return df

    def show_radial_profile(self, layer: int, coord: int,
                            feature: str | None = None,
                            n_bins: int = 40, r_factor: float = 1.5):
        """THE basin/ridge picture for one neuron: its radial profile
        p(r) = mean field value at distance r from the feature's centroid.
        A crest/trough AT the ring radius = ridge encoding; an essentially
        monotone profile peaking over the interior = basin encoding.

        Any dimension: 2D uses the grid field, d >= 3 the random cloud."""
        import matplotlib.pyplot as plt
        from analysis.fields import evaluate_field_on_grid

        feat = self.feature_points(feature).float()
        center = feat.mean(dim=0)
        r_feature = float((feat - center).norm(dim=1).mean().item())
        r_max = r_factor * r_feature

        if self.in_dim == 2:
            field = evaluate_field_on_grid(
                self._neuron_function(layer, coord),
                bounds=self.bounds(),
                resolution=min(self.resolution, 150),
                device=self.device,
            )
        else:
            field = self.random_field(layer, coord)
        radii = (field.points - center).norm(dim=1)
        edges = torch.linspace(0.0, r_max, n_bins + 1)

        rs, ps = [], []
        for i in range(n_bins):
            m = (radii >= edges[i]) & (radii < edges[i + 1])
            if m.any():
                rs.append(float((edges[i] + edges[i + 1]) / 2))
                ps.append(float(field.values[m].mean().item()))

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(rs, ps, marker=".", color="steelblue")
        ax.axvline(r_feature, color="crimson", ls="--", lw=1.2,
                   label=f"ring radius ≈ {r_feature:.2f}")
        ax.set_xlabel("distance r from feature centroid")
        ax.set_ylabel("mean activation p(r)")
        ax.set_title(f"radial profile — neuron L{layer} · N{coord}\n"
                     "(crest/trough AT the dashed line = ridge; "
                     "monotone across it = basin)", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()

    def show_neuron_extrema(self, layer: int, coord: int,
                            n_restarts: int = 64, seed: int = 0,
                            resolution: int | None = None):
        """One neuron's field heatmap with its local maxima (▲) and minima
        (▼) from extremum sampling — where the gradient flow terminates.
        The anchor positions are what distinguish a basin (extremum over the
        interior) from a ridge (crest along the ring)."""
        import matplotlib.pyplot as plt
        from analysis.extremum_sampling import find_local_extrema
        from analysis.geometry import make_grid

        self._assert_2d("show_neuron_extrema")
        res = resolution or min(self.resolution, 150)
        bounds = self.bounds()
        f = self._neuron_function(layer, coord)

        grid, shape = make_grid(bounds=bounds, resolution=res, device=self.device)
        with torch.no_grad():
            Z = f(grid).reshape(shape).T.cpu().numpy()

        maxima = find_local_extrema(f, bounds, mode="max",
                                    n_restarts=n_restarts, seed=seed,
                                    device=self.device)
        minima = find_local_extrema(f, bounds, mode="min",
                                    n_restarts=n_restarts, seed=seed,
                                    device=self.device)

        feat = self.feature_points().cpu().numpy()
        (x0, x1), (y0, y1) = bounds[0], bounds[1]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(Z, extent=[x0, x1, y0, y1], origin="lower",
                       cmap="viridis", aspect="equal")
        ax.scatter(feat[:, 0], feat[:, 1], s=5, c="white", alpha=0.6,
                   linewidths=0, label="target feature")
        for res_, marker, color, name in ((maxima, "^", "red", "maxima"),
                                          (minima, "v", "deepskyblue", "minima")):
            pts = res_.points.detach().cpu().numpy()
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=90, marker=marker,
                           c=color, edgecolors="black", linewidths=0.7,
                           label=f"{name} ({len(pts)})")
        fig.colorbar(im, ax=ax, fraction=0.046, label="activation")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"extrema of neuron L{layer} · N{coord}")
        plt.show()

    def scan_saved_models(self, max_models: int | None = 10,
                          type_witnesses: bool = True, verbose: bool = True):
        """A.7 preview at notebook scale: walk the SAVED models of this
        (dataset, architecture), count structural witnesses of the target per
        model, and (optionally) type them basin/ridge — the per-seed
        distribution whose bimodality is the A.7 question.

        Lean on purpose: structural criterion only, resolution capped at 150.
        Returns a DataFrame [model_path, n_witnesses, witness_neurons,
        basin_share]; leaves ``self.model`` pointing at the LAST scanned
        model.
        """
        import pandas as pd

        from analysis.criterions import evaluate_neurons_on_feature
        from analysis.extremum_anchored import classify_witness_basin_or_ridge
        from analysis.stat_aggregation import find_model_paths_for_dataset

        self._assert_2d("scan_saved_models")
        paths = find_model_paths_for_dataset(
            self.dataset_name, instance_id=self.instance_id,
            model_name=self.arch_tag,
        )
        assert paths, f"no saved {self.arch_tag} models for {self.dataset_name}"
        if max_models is not None:
            paths = paths[:max_models]

        feat = self.feature_points()
        bounds = self.bounds()
        res = min(self.resolution, 150)
        e = self.spec.eval

        rows = []
        for i, p in enumerate(paths):
            self.load_path(p)
            evals = evaluate_neurons_on_feature(
                model=self.model, feature_points=feat, bounds=bounds,
                resolution=res, kinds=e.kinds, device=self.device,
                structural_kwargs=dict(e.structural_kwargs),
                run_tubular=False, verbose=False,
            )
            witnesses = [(r["layer_id"], r["coord_id"]) for r in evals
                         if r.get("is_structural_witness")]

            basin_share = float("nan")
            if type_witnesses and witnesses:
                types = []
                for layer, coord in witnesses:
                    typed = classify_witness_basin_or_ridge(
                        self._neuron_function(layer, coord), feat, bounds,
                        resolution=min(res, 100), n_restarts=32, seed=0,
                        device=self.device,
                    )
                    types.append(typed["witness_type"])
                basin_share = types.count("basin") / len(types)

            rows.append({
                "model_path": p,
                "n_witnesses": len(witnesses),
                "witness_neurons": witnesses,
                "basin_share": basin_share,
            })
            if verbose:
                print(f"[scan {i + 1}/{len(paths)}] "
                      f"witnesses={len(witnesses)} basin_share={basin_share}")

        df = pd.DataFrame(rows)
        self.seed_scan_df = df
        return df

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def _assert_2d(self, what: str):
        if self.in_dim != 2:
            raise NotImplementedError(
                f"{what} is 2D-only; {self.experiment_id}'s input space is "
                f"{self.in_dim}D. Higher-dimensional experiments use the "
                "homological/sampled criteria and PCA slices — see the hi-dim "
                "notebook template."
            )

    def config(self) -> dict:
        """The current knob values (for printing at the top of a run)."""
        return {
            "experiment": self.experiment_id,
            "dataset": self.dataset_name,
            "arch": self.arch_tag,
            "n_models": self.n_models,
            "n_epochs": self.n_epochs,
            "target_acc": self.target_acc,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "resolution": self.resolution,
            "target_feature": self._dataset_spec.target_feature,
            "primary_witness_column": self.primary_witness_column,
        }

    def __repr__(self):
        loaded = "model loaded" if self.model is not None else "no model"
        return (f"ExperimentLab({self.experiment_id!r}, dataset={self.dataset_name!r}, "
                f"arch={self.arch_tag!r}, {loaded})")


# ----------------------------------------------------------------------
# Architecture sweeps (exp. 2 / A.5) at single-model-per-cell scale
# ----------------------------------------------------------------------

def sweep_architectures(
    experiment_id: str,
    archs,
    dataset: str | None = None,
    device: str = "cpu",
    verbose: bool = False,
    **knobs,
):
    """Train ONE model per architecture and tally target-vs-null witness rates.

    The local, eyeball-scale version of exp-2's rate-vs-width curve (the
    ensemble version with bootstrap CIs is the server run). Every model is
    saved as usual and joins the future ensemble.

    archs:
        iterable of ``(width, depth)`` / ``(width, depth, activation)``
        tuples or ``ArchSpec``s.
    knobs:
        any ExperimentLab attribute to set on every lab (n_epochs=..., lr=...,
        dataset_overrides=..., dataset_label=..., ...).

    Returns ``(df, labs)``: a tidy DataFrame [width, depth, activation,
    arch, final_test_acc, reached_target, rate_target, rate_null, gap,
    n_neurons] and ``labs[arch_tag]`` -> the fitted ExperimentLab (for
    drill-down: ``labs["tanh_w31_d1"].show_neuron_fields()``).
    """
    import pandas as pd

    df_rows = []
    labs = {}

    for arch in archs:
        if isinstance(arch, ArchSpec):
            w, d, act = arch.width, arch.depth, arch.activation
        else:
            w, d = arch[0], arch[1]
            act = arch[2] if len(arch) > 2 else "tanh"

        lab = ExperimentLab(experiment_id, dataset=dataset, device=device)
        lab.width, lab.depth, lab.activation = w, d, act
        for k, v in knobs.items():
            assert hasattr(lab, k), f"unknown ExperimentLab knob {k!r}"
            setattr(lab, k, v)

        lab.prepare()
        summary = lab.fit(plot=False, verbose=verbose)[-1]

        wdf = lab.witness_summary(style=False)
        target_rate = float(
            wdf.loc[wdf["role"] == "target", "witness_rate"].mean()
        )
        null_rate = float(wdf.loc[wdf["role"] == "null", "witness_rate"].mean())

        df_rows.append(
            {
                "width": w,
                "depth": d,
                "activation": act,
                "arch": lab.arch_tag,
                "final_test_acc": summary["final_test_acc"],
                "reached_target": summary["reached_target"],
                "rate_target": target_rate,
                "rate_null": null_rate,
                "gap": target_rate - null_rate,
                "n_neurons": w * d,
            }
        )
        labs[lab.arch_tag] = lab
        print(f"[sweep] {lab.arch_tag}: acc={summary['final_test_acc']:.3f} "
              f"rate_target={target_rate:.3f} rate_null={null_rate:.3f}")

    return pd.DataFrame(df_rows), labs


def sweep_datasets(
    experiment_id: str,
    datasets=None,
    device: str = "cpu",
    verbose: bool = False,
    **knobs,
):
    """Train ONE model per dataset variant of an experiment (exp. 4's density
    ratios, exp. 8's a/b/c series) and tally target-vs-null witness rates.

    Full retraining per variant is the project rule (checkpoints are never
    reused across dataset instances); every model is saved as usual.

    datasets: names from the spec (default: all of them, in spec order).
    knobs:    any ExperimentLab attribute, applied to every lab.

    Returns ``(df, labs)``: DataFrame [dataset, final_test_acc,
    reached_target, rate_target, rate_null, gap] and ``labs[dataset]`` for
    drill-down.
    """
    import pandas as pd

    spec = EXPERIMENTS[experiment_id]
    if datasets is None:
        datasets = [d.name for d in spec.datasets]

    df_rows = []
    labs = {}

    for name in datasets:
        lab = ExperimentLab(experiment_id, dataset=name, device=device)
        for k, v in knobs.items():
            assert hasattr(lab, k), f"unknown ExperimentLab knob {k!r}"
            setattr(lab, k, v)

        lab.prepare()
        summary = lab.fit(plot=False, verbose=verbose)[-1]

        wdf = lab.witness_summary(style=False)
        target_rate = float(wdf.loc[wdf["role"] == "target", "witness_rate"].mean())
        null_rates = wdf.loc[wdf["role"] == "null", "witness_rate"]
        null_rate = float(null_rates.mean()) if len(null_rates) else float("nan")

        df_rows.append({
            "dataset": lab.dataset_name,
            "final_test_acc": summary["final_test_acc"],
            "reached_target": summary["reached_target"],
            "rate_target": target_rate,
            "rate_null": null_rate,
            "gap": target_rate - null_rate,
        })
        labs[name] = lab
        print(f"[sweep] {lab.dataset_name}: acc={summary['final_test_acc']:.3f} "
              f"rate_target={target_rate:.3f} rate_null={null_rate:.3f}")

    return pd.DataFrame(df_rows), labs


def show_dataset_gaps(df, title: str = "witness-rate gap per dataset variant"):
    """Bar chart of ``sweep_datasets`` gaps: rate[target] − rate[null] per
    variant (exp-8's a/b/c contrast; exp-4's per-ratio view). NaN gaps
    (variants without nulls) are drawn as hatched zero bars."""
    import matplotlib.pyplot as plt
    import numpy as np

    gaps = df["gap"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(df) + 2, 4))
    for i, (name, g) in enumerate(zip(df["dataset"], gaps)):
        if np.isnan(g):
            ax.bar(i, 0, color="none", edgecolor="grey", hatch="//")
        else:
            ax.bar(i, g, color="steelblue")
    ax.axhline(0.0, c="grey", lw=1)
    ax.set_xticks(range(len(df)), df["dataset"], rotation=45, ha="right",
                  fontsize=8)
    ax.set_ylabel("rate[target] − rate[null]")
    ax.set_title(title + "  (1 model per variant)", fontsize=10)
    fig.tight_layout()
    plt.show()


def basin_ridge_vs_dataset(labs, plot: bool = True):
    """Exp-4's readout over a ``sweep_datasets`` result: witness count and
    basin/ridge composition per dataset variant, with the density ratio
    parsed from ``ring_den{n_ring}x{n_interior}`` names when present.

    Returns a DataFrame [dataset, density_ratio, n_witnesses, n_basin,
    n_ridge, n_ambiguous, basin_share]; ``plot=True`` draws basin share and
    witness count vs the ratio (log-x).
    """
    import re

    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for name, lab in labs.items():
        typing = lab.basin_ridge_table()
        m = re.search(r"den(\d+)x(\d+)", str(name))
        ratio = float(m.group(1)) / float(m.group(2)) if m else float("nan")
        counts = typing["witness_type"].value_counts() if len(typing) else {}
        n = int(len(typing))
        rows.append({
            "dataset": name,
            "density_ratio": ratio,
            "n_witnesses": n,
            "n_basin": int(counts.get("basin", 0)),
            "n_ridge": int(counts.get("ridge", 0)),
            "n_ambiguous": int(counts.get("ambiguous", 0)),
            "basin_share": (counts.get("basin", 0) / n) if n else float("nan"),
        })

    df = pd.DataFrame(rows).sort_values("density_ratio").reset_index(drop=True)

    if plot and df["density_ratio"].notna().any():
        fig, (axs, axn) = plt.subplots(1, 2, figsize=(11, 4))
        axs.plot(df["density_ratio"], df["basin_share"], marker="o",
                 color="steelblue")
        axs.set_xscale("log")
        axs.set_xlabel("density ratio  n_ring : n_interior")
        axs.set_ylabel("basin share among witnesses")
        axs.set_ylim(-0.05, 1.05)
        axs.set_title("encoding strategy vs density (1 model per ratio)")

        axn.plot(df["density_ratio"], df["n_witnesses"], marker="o",
                 color="darkorange")
        axn.set_xscale("log")
        axn.set_xlabel("density ratio  n_ring : n_interior")
        axn.set_ylabel("n witnesses (target)")
        axn.set_title("witness count vs density")
        fig.tight_layout()
        plt.show()

    return df


def show_arch_sweep(df, x: str = "width"):
    """Two-panel view of a ``sweep_architectures`` table: the gap curve and
    (the exp-2 confound, PLAN.md) the raw target/null rates side by side —
    the null rate growing with width is exactly why the GAP is the measured
    quantity, not the raw count. Lines are grouped by the other dimension
    (depth when x="width")."""
    import matplotlib.pyplot as plt

    other = "depth" if x == "width" else "width"
    fig, (axg, axr) = plt.subplots(1, 2, figsize=(11, 4))

    for val, g in df.groupby(other):
        g = g.sort_values(x)
        if len(g) < 2:
            axg.scatter(g[x], g["gap"], label=f"{other}={val}")
            axr.scatter(g[x], g["rate_target"], marker="o")
            axr.scatter(g[x], g["rate_null"], marker="x")
            continue
        axg.plot(g[x], g["gap"], marker="o", label=f"{other}={val}")
        axr.plot(g[x], g["rate_target"], marker="o", label=f"target ({other}={val})")
        axr.plot(g[x], g["rate_null"], marker="x", ls="--",
                 label=f"null ({other}={val})")

    axg.axhline(0.0, c="grey", lw=1)
    axg.set_xlabel(x); axg.set_ylabel("rate[target] − rate[null]")
    axg.set_title("witness-rate gap (1 model per cell)")
    axg.legend(fontsize=8)

    axr.set_xlabel(x); axr.set_ylabel("witness rate")
    axr.set_title("raw rates: the null confound")
    axr.legend(fontsize=7)

    fig.tight_layout()
    plt.show()
