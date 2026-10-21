"""Stage-4 analyses: one function per name in ``ExperimentSpec.analyses``.

Each analysis takes the combined per-neuron DataFrame (all architectures of
the experiment, concatenated from ``neuron_stats_*.csv``) plus an
``AggregationContext``, writes its tables under ``<results>/aggregates/`` and
its figures under ``<results>/figures/``, and returns its main outputs.
``experiments.runner.aggregate`` is the dispatcher; the mapping lives in
``ANALYSES`` at the bottom.

Model-loading analyses (basin_ridge_typing, sparse_probing, ablation) cache
loaded models/instances on the context, and share the typing table within one
aggregate() call.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.witness_stats import (
    _coerce_bool_series,
    _bootstrap_ci_mean,
    build_witness_aggregate_tables,
    build_feature_overlap_tables,
    consistency_table,
)


class AggregationContext:
    """Shared state of one aggregate() run: dirs, caches, memoized tables."""

    def __init__(self, spec, instance_id: int = 0):
        self.spec = spec
        self.instance_id = instance_id
        self.aggregates_dir = Path(spec.results_dir) / "aggregates"
        self.figures_dir = Path(spec.results_dir) / "figures"
        self.aggregates_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self._instances = {}
        self._models = {}
        self._typing_table = None

    def load_instance(self, dataset_instance_path: str):
        from management.experiment_management import load_dataset_instance

        key = str(dataset_instance_path)
        if key not in self._instances:
            self._instances[key] = load_dataset_instance(
                key, map_location="cpu", reconstruct_data=False
            )
        return self._instances[key]

    def load_model(self, model_path: str):
        from management.experiment_management import load_model_instance

        key = str(model_path)
        if key not in self._models:
            self._models[key] = load_model_instance(
                key, device="cpu", eval_mode=True
            )
        return self._models[key]


# -------------------------
# Small shared helpers
# -------------------------

def _primary_witness_column(ctx: AggregationContext) -> str:
    """The witness column the experiment treats as THE verdict.

    EvalSpec.primary_witness_column when set (hi-dim experiments point it at
    e.g. "homological_is_witness"), else the grid-structural default.
    """
    return ctx.spec.eval.primary_witness_column or "is_structural_witness"


def _dataset_features_by_role(df: pd.DataFrame, dataset_type: str):
    """(target_feature, null_features, other_features) for one dataset."""
    g = df[df["dataset_type"] == dataset_type]

    roles = g[["feature_name", "feature_role"]].drop_duplicates()

    targets = roles.loc[roles["feature_role"] == "target", "feature_name"].tolist()
    nulls = roles.loc[roles["feature_role"] == "null", "feature_name"].tolist()
    others = roles.loc[
        ~roles["feature_role"].isin(["target", "null"]), "feature_name"
    ].tolist()

    target = targets[0] if targets else (others[0] if others else None)
    return target, nulls, others


def _witness_rows(
    df: pd.DataFrame,
    dataset_type: str,
    feature_name: str,
    witness_column: str = "is_structural_witness",
    kind: str = "post",
) -> pd.DataFrame:
    g = df[
        (df["dataset_type"] == dataset_type)
        & (df["feature_name"] == feature_name)
        & (df["kind"] == kind)
    ].copy()
    g["_witness"] = _coerce_bool_series(g[witness_column])
    return g[g["_witness"]]


def _feature_scale_delta(feature_points: torch.Tensor) -> float:
    """Region half-width for near-feature accuracy/probing: 0.1 x diam(A)."""
    from analysis.geometry import feature_diameter

    return 0.1 * float(feature_diameter(feature_points, method="exact").item())


# -------------------------
# Analyses over the CSV alone
# -------------------------

def analysis_witness_tables(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp-agnostic witness count/fraction tables (analysis.witness_stats)."""
    tables = build_witness_aggregate_tables(
        df,
        save_dir=ctx.aggregates_dir,
        save_prefix="witness",
        witness_column=_primary_witness_column(ctx),
    )
    return {"tables": tables}


def analysis_witness_rate_gap(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp 1: rate[A] - rate[null] per dataset, CSV + bar figure."""
    from analysis.controls import witness_rate_gap
    from visualization.experiment_plots import plot_rate_gap_bars

    per_model_parts = []
    summaries = {}

    for dataset_type in df["dataset_type"].unique():
        target, nulls, _ = _dataset_features_by_role(df, dataset_type)
        assert target is not None, f"{dataset_type}: no target feature in CSV"
        assert nulls, (
            f"{dataset_type}: no null features in the CSV — the instance was "
            "persisted without nulls (DatasetSpec.n_null_features == 0?)"
        )

        result = witness_rate_gap(
            df[df["dataset_type"] == dataset_type],
            feature_name=target,
            null_feature_names=nulls,
            witness_column=_primary_witness_column(ctx),
        )

        per_model_parts.append(result["per_model"])
        summaries[dataset_type] = result["summary"]

        plot_rate_gap_bars(
            result["per_model"],
            result["summary"],
            title=f"{dataset_type}: rate[{target}] vs rate[null]",
            save_path=ctx.figures_dir / f"rate_gap_{dataset_type}.png",
        )

    per_model = pd.concat(per_model_parts, ignore_index=True)
    per_model.to_csv(ctx.aggregates_dir / "rate_gap_per_model.csv", index=False)

    summary_df = pd.DataFrame(
        [{"dataset_type": k, **v} for k, v in summaries.items()]
    )
    summary_df.to_csv(ctx.aggregates_dir / "rate_gap_summary.csv", index=False)

    return {"per_model": per_model, "summary": summary_df}


def _witness_columns(df: pd.DataFrame) -> list[str]:
    """Every boolean witness verdict present in the CSV, structural first.

    Extra criteria land as "<name>_is_witness" (the extra_criteria protocol
    prefixes each returned key); the core structural one is the odd name out.
    All-NaN columns are dropped: _coerce_bool_series turns NaN into True on a
    numeric column, so a criterion that silently failed would otherwise be
    counted as witnessing everything.
    """
    cols = []

    if "is_structural_witness" in df.columns:
        cols.append("is_structural_witness")

    cols += sorted(c for c in df.columns if c.endswith("_is_witness"))

    return [c for c in cols if not df[c].isna().all()]


def _criterion_label(witness_column: str) -> str:
    if witness_column == "is_structural_witness":
        return "structural"
    return witness_column[: -len("_is_witness")]


def analysis_rate_gap_by_criterion(
    df: pd.DataFrame, ctx: AggregationContext
) -> dict:
    """rate[A] - rate[null] for EVERY witness criterion in the CSV.

    analysis_witness_rate_gap answers the question for the ONE column the
    spec calls the verdict; this answers it for all of them side by side, so
    the verdict can be chosen by looking at the evidence instead of being
    committed to up front. Criteria are all evaluated during stage 3
    regardless, so switching EvalSpec.primary_witness_column afterwards costs
    a re-run of the aggregate stage only.

    Writes <aggregates>/rate_gap_by_criterion.csv with one row per
    (dataset, criterion) and an is_primary flag marking the current verdict.
    """
    from analysis.controls import witness_rate_gap

    primary = _primary_witness_column(ctx)
    witness_columns = _witness_columns(df)
    assert witness_columns, "no witness columns in the evaluation CSVs"

    rows = []

    for dataset_type in df["dataset_type"].unique():
        sub = df[df["dataset_type"] == dataset_type]
        target, nulls, _ = _dataset_features_by_role(sub, dataset_type)

        assert target is not None, f"{dataset_type}: no target feature in CSV"
        assert nulls, (
            f"{dataset_type}: no null features in the CSV — the instance was "
            "persisted without nulls (DatasetSpec.n_null_features == 0?)"
        )

        for column in witness_columns:
            summary = witness_rate_gap(
                sub,
                feature_name=target,
                null_feature_names=nulls,
                witness_column=column,
            )["summary"]

            rows.append(
                {
                    "dataset_type": dataset_type,
                    "criterion": _criterion_label(column),
                    "witness_column": column,
                    "is_primary": column == primary,
                    "target_feature": target,
                    "rate_target": summary["mean_rate_feature"],
                    "rate_null": summary["mean_rate_null"],
                    "gap": summary["mean_gap"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "n_models": summary["n_groups"],
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(ctx.aggregates_dir / "rate_gap_by_criterion.csv", index=False)

    return {"table": table}


def analysis_grid_witness_rates(
    df: pd.DataFrame, ctx: AggregationContext
) -> dict:
    """One row per GRID CELL x criterion: rate[A], rate[null], gap, CI.

    The unit here is (dataset, run tag) rather than (dataset, width, depth):
    a grid may sweep learning rate or epoch budget, which the per-neuron CSV
    does not record — only the run tag in model_name does. The tag is joined
    back to its full configuration through <results>/grid_manifest.csv
    (written by experiments.grid.save_grid), so the swept knobs come back as
    columns and the table can be pivoted on any of them.

    Without a manifest the analysis still runs; the knob columns are simply
    absent.
    """
    from analysis.controls import witness_rate_gap

    witness_columns = _witness_columns(df)
    assert witness_columns, "no witness columns in the evaluation CSVs"

    primary = _primary_witness_column(ctx)
    rows = []

    for (dataset_type, model_name), cell in df.groupby(
        ["dataset_type", "model_name"], dropna=False
    ):
        target, nulls, _ = _dataset_features_by_role(cell, dataset_type)

        if target is None or not nulls:
            print(f"[grid_witness_rates] {dataset_type} x {model_name}: "
                  "no target/null pair, skipped")
            continue

        for column in witness_columns:
            summary = witness_rate_gap(
                cell,
                feature_name=target,
                null_feature_names=nulls,
                witness_column=column,
            )["summary"]

            rows.append(
                {
                    "dataset_type": dataset_type,
                    "model_name": model_name,
                    "criterion": _criterion_label(column),
                    "is_primary": column == primary,
                    "rate_target": summary["mean_rate_feature"],
                    "rate_null": summary["mean_rate_null"],
                    "gap": summary["mean_gap"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "n_models": summary["n_groups"],
                }
            )

    table = pd.DataFrame(rows)
    assert len(table) > 0, "no grid cells with a target/null pair"

    manifest_path = Path(ctx.spec.results_dir) / "grid_manifest.csv"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        knob_cols = [
            c for c in manifest.columns
            if c not in ("cell", "dataset_type", "model_name", "n_features")
        ]

        # n_models means two different things on the two sides: the ensemble
        # size the grid ASKED for, and the number of saved models the CI was
        # actually computed over (training drops runs that miss the accuracy
        # gate). Keep both, named for what they are, instead of letting the
        # merge produce n_models_x / n_models_y.
        collisions = sorted(set(knob_cols) & set(table.columns))
        if collisions:
            manifest = manifest.rename(
                columns={c: f"{c}_requested" for c in collisions}
            )
            knob_cols = [
                f"{c}_requested" if c in collisions else c for c in knob_cols
            ]

        table = table.merge(
            manifest[["dataset_type", "model_name", *knob_cols]].drop_duplicates(
                ["dataset_type", "model_name"]
            ),
            on=["dataset_type", "model_name"],
            how="left",
        )
    else:
        print(f"[grid_witness_rates] no manifest at {manifest_path}; "
              "swept knobs will not appear as columns")

    table = table.sort_values(["criterion", "dataset_type", "model_name"])
    table.to_csv(ctx.aggregates_dir / "grid_witness_rates.csv", index=False)

    return {"table": table}


def analysis_rate_vs_width(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp 2: the gap curve over architecture width (and depth), with the
    null-rate confound shown alongside (PLAN.md exp-2 caveat)."""
    from analysis.controls import witness_rate_gap
    from visualization.experiment_plots import plot_rate_vs_width

    rows = []

    for (dataset_type, width, depth), g in df.groupby(
        ["dataset_type", "hidden_dim", "n_layers"]
    ):
        target, nulls, _ = _dataset_features_by_role(g, dataset_type)
        if target is None or not nulls:
            continue

        result = witness_rate_gap(
            g,
            feature_name=target,
            null_feature_names=nulls,
            witness_column=_primary_witness_column(ctx),
        )

        # PLAN's mechanism is "more neurons -> more random directions -> more
        # incidental near-radial levels", i.e. each neuron is roughly an
        # independent Bernoulli(p) trial. Under that model the per-neuron
        # FRACTION is constant at p, so plotting only the fraction is the one
        # functional form in which the hypothesised mechanism predicts a FLAT
        # line (AUDIT/02_plan_traceability.md, exp2). The count (~ p x width)
        # and P(>=1 witness per model) (= 1-(1-p)^width) do carry the signal,
        # so they are recorded alongside.
        witness_col = _primary_witness_column(ctx)
        target_rows = g[(g["feature_name"] == target) & (g["kind"] == "post")].copy()
        target_rows["_w"] = _coerce_bool_series(target_rows[witness_col])
        per_model = target_rows.groupby("model_path")["_w"].agg(["sum", "any", "size"])

        rows.append(
            {
                "dataset_type": dataset_type,
                "width": int(width),
                "depth": int(depth),
                **result["summary"],
                "mean_witness_count": float(per_model["sum"].mean()),
                "p_has_witness": float(per_model["any"].mean()),
                "n_neurons_per_model": int(per_model["size"].median()),
                "n_models": int(len(per_model)),
            }
        )

    curve = pd.DataFrame(rows)
    assert len(curve) > 0, "no (width, depth) cells with target+null rows"

    curve.to_csv(ctx.aggregates_dir / "rate_vs_width.csv", index=False)

    # PLAN exp-2's estimand is the SLOPE of the gap against width, which no
    # analysis computed. An OLS slope with a bootstrap CI over cells is the
    # minimum honest version; with <3 cells it is reported as NaN rather than
    # fabricated.
    trend_rows = []
    for (dataset_type, depth), g in curve.groupby(["dataset_type", "depth"]):
        for y_col in ("mean_gap", "mean_witness_count", "p_has_witness"):
            if y_col not in g or len(g) < 3:
                trend_rows.append({
                    "dataset_type": dataset_type, "depth": int(depth),
                    "quantity": y_col, "slope_per_neuron": float("nan"),
                    "n_cells": int(len(g)),
                    "note": "need >= 3 width cells for a slope",
                })
                continue

            x = g["width"].to_numpy(dtype=float)
            y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
            ok = ~np.isnan(y)
            slope = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else float("nan")

            trend_rows.append({
                "dataset_type": dataset_type, "depth": int(depth),
                "quantity": y_col, "slope_per_neuron": slope,
                "n_cells": int(ok.sum()),
                "note": ("the mechanism predicts ~0 here even when real"
                         if y_col == "mean_gap" else
                         "the mechanism predicts a POSITIVE slope here"),
            })

    trend = pd.DataFrame(trend_rows)
    trend.to_csv(ctx.aggregates_dir / "rate_vs_width_trend.csv", index=False)

    for (dataset_type, depth), g in curve.groupby(["dataset_type", "depth"]):
        if len(g) < 2:
            continue
        plot_rate_vs_width(
            g.rename(columns={"width": "width"}),
            x_col="width",
            title=f"{dataset_type} (depth {depth}): gap vs width",
            save_path=ctx.figures_dir
            / f"rate_vs_width_{dataset_type}_d{depth}.png",
        )

    for (dataset_type, width), g in curve.groupby(["dataset_type", "width"]):
        if len(g) < 2:
            continue
        plot_rate_vs_width(
            g.rename(columns={"depth": "depth"}),
            x_col="depth",
            title=f"{dataset_type} (width {width}): gap vs depth",
            save_path=ctx.figures_dir
            / f"rate_vs_depth_{dataset_type}_w{width}.png",
        )

    return {"curve": curve, "trend": trend}


def analysis_feature_overlap(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp 7: witness-set overlap for every pair of non-null features."""
    tables = build_witness_aggregate_tables(
        df, save_dir=None, save_prefix="witness"
    )
    neuron_table = tables["neuron_table"]

    outputs = {}

    for dataset_type in df["dataset_type"].unique():
        _, nulls, _ = _dataset_features_by_role(df, dataset_type)
        features = [
            f
            for f in df.loc[
                df["dataset_type"] == dataset_type, "feature_name"
            ].unique()
            if f not in nulls
        ]

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                f1, f2 = features[i], features[j]
                key = f"{dataset_type}:{f1}_vs_{f2}"
                outputs[key] = build_feature_overlap_tables(
                    neuron_table,
                    f1,
                    f2,
                    dataset_types=[dataset_type],
                    save_dir=ctx.aggregates_dir,
                    save_prefix=f"overlap_{dataset_type}_{f1}_vs_{f2}",
                )

    return outputs


def analysis_consistency_table(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """PLAN.md §C: structural x extremum x homological agreement matrix."""
    from visualization.experiment_plots import plot_consistency_heatmap

    table = consistency_table(df)
    table.to_csv(ctx.aggregates_dir / "consistency_table.csv", index=False)

    plot_consistency_heatmap(
        table,
        save_path=ctx.figures_dir / "consistency_heatmap.png",
    )

    return {"table": table}


def analysis_layer_persistence(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp 3: witness fraction per layer + the 'witness at layer l persists
    for layers > l' hypothesis check (violation rate over models)."""
    from visualization.experiment_plots import plot_layer_persistence

    outputs = {}

    for dataset_type in df["dataset_type"].unique():
        target, _, _ = _dataset_features_by_role(df, dataset_type)
        g = df[
            (df["dataset_type"] == dataset_type)
            & (df["feature_name"] == target)
            & (df["kind"] == "post")
        ].copy()

        if len(g) == 0:
            continue

        g["_witness"] = _coerce_bool_series(g["is_structural_witness"])

        per_model_layer = (
            g.groupby(["model_path", "layer_id"])["_witness"]
            .agg(["mean", "sum"])
            .rename(columns={"mean": "fraction", "sum": "count"})
            .reset_index()
        )

        # Monotonicity: after the first witnessing layer, every later layer
        # should witness too. Count violating (model, layer) transitions.
        n_after_first = 0
        n_violations = 0

        for model_path, gm in per_model_layer.groupby("model_path"):
            gm = gm.sort_values("layer_id")
            has = gm["count"].to_numpy() > 0
            if not has.any():
                continue
            first = int(np.argmax(has))
            tail = has[first + 1:]
            n_after_first += len(tail)
            n_violations += int((~tail).sum())

        violation_rate = (
            n_violations / n_after_first if n_after_first > 0 else float("nan")
        )

        rows = []
        for layer_id, gl in per_model_layer.groupby("layer_id"):
            fractions = gl["fraction"].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci_mean(fractions)
            rows.append(
                {
                    "dataset_type": dataset_type,
                    "layer_id": int(layer_id),
                    "mean_fraction": float(fractions.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_models": int(len(gl)),
                }
            )

        layer_table = pd.DataFrame(rows)
        layer_table.attrs["violation_rate"] = violation_rate

        layer_table.to_csv(
            ctx.aggregates_dir / f"layer_persistence_{dataset_type}.csv",
            index=False,
        )

        # violation_rate is the actual VERDICT on PLAN.md exp-3's hypothesis
        # "a witness at layer l persists at all layers > l". It used to live
        # only in .attrs, which DataFrame.to_csv drops, and run_experiment.py
        # discards aggregate()'s return value — so the answer was computed and
        # then thrown away (AUDIT/02_plan_traceability.md X8). Persist it.
        pd.DataFrame([{
            "dataset_type": dataset_type,
            "target_feature": target,
            "violation_rate": violation_rate,
            "n_witness_layer_pairs_after_first": int(n_after_first),
            "n_violations": int(n_violations),
            "hypothesis": "witness at layer l persists at all layers > l",
            "reading": "violation_rate near 0 supports the hypothesis; near 1 refutes it",
        }]).to_csv(
            ctx.aggregates_dir / f"layer_persistence_verdict_{dataset_type}.csv",
            index=False,
        )

        plot_layer_persistence(
            layer_table,
            title=f"{dataset_type}: witness fraction by layer ({target})",
            save_path=ctx.figures_dir / f"layer_persistence_{dataset_type}.png",
        )

        outputs[dataset_type] = {
            "layer_table": layer_table,
            "violation_rate": violation_rate,
        }

    return outputs


# -------------------------
# Analyses that load models
# -------------------------

def _typing_table(df: pd.DataFrame, ctx: AggregationContext) -> pd.DataFrame:
    """Basin/ridge type for every structural witness of every target feature
    (memoized: basin_ridge_typing / seed_bimodality / basin_ridge_vs_ratio
    share it)."""
    if ctx._typing_table is not None:
        return ctx._typing_table

    from models.NN_analysis import TruncatedMLPFunction, ScalarProjection
    from analysis.fields import evaluate_field_on_grid
    from analysis.stat_aggregation import bounds_from_X
    from analysis.extremum_anchored import classify_witness_basin_or_ridge
    from management.experiment_management import get_feature_points

    resolution = min(
        ctx.spec.eval.resolution
        if isinstance(ctx.spec.eval.resolution, int)
        else max(ctx.spec.eval.resolution),
        100,
    )

    rows = []

    for dataset_type in df["dataset_type"].unique():
        target, _, _ = _dataset_features_by_role(df, dataset_type)
        witnesses = _witness_rows(df, dataset_type, target)

        if len(witnesses) == 0:
            continue

        instance_path = witnesses["dataset_instance_path"].iloc[0]
        loaded = ctx.load_instance(instance_path)
        feature_points = get_feature_points(loaded, target)
        bounds = bounds_from_X(loaded["X"], margin=ctx.spec.eval.bounds_margin)

        for (model_path, layer_id), g in witnesses.groupby(
            ["model_path", "layer_id"]
        ):
            model, _ = ctx.load_model(model_path)
            vector_fn = TruncatedMLPFunction(model, int(layer_id), kind="post")

            for coord_id in g["coord_id"].astype(int):
                neuron = ScalarProjection(vector_fn, coord_id=int(coord_id))
                field = evaluate_field_on_grid(
                    neuron, bounds=bounds, resolution=resolution
                )

                typed = classify_witness_basin_or_ridge(
                    neuron,
                    feature_points,
                    bounds,
                    field=field,
                    n_restarts=32,
                    seed=0,
                )

                rows.append(
                    {
                        "dataset_type": dataset_type,
                        "feature_name": target,
                        "model_path": model_path,
                        "layer_id": int(layer_id),
                        "coord_id": int(coord_id),
                        "witness_type": typed["witness_type"],
                        "radial_profile_monotonicity": typed[
                            "radial_profile_monotonicity"
                        ],
                        "ridge_location": typed["ridge_location"],
                        "ridge_prominence": typed["ridge_prominence"],
                        "r_feature": typed["r_feature"],
                        "anchor_value": typed["anchor_value"],
                        "anchor_dist_to_center": typed["anchor_dist_to_center"],
                        "anchor_dist_to_feature": typed["anchor_dist_to_feature"],
                    }
                )

    ctx._typing_table = pd.DataFrame(rows)
    return ctx._typing_table


def analysis_basin_ridge_typing(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp 6: direct basin/ridge classification of every witness."""
    from visualization.experiment_plots import plot_basin_ridge_bars

    typing = _typing_table(df, ctx)
    typing.to_csv(ctx.aggregates_dir / "basin_ridge_typing.csv", index=False)

    if len(typing) > 0:
        plot_basin_ridge_bars(
            typing,
            save_path=ctx.figures_dir / "basin_ridge_typing.png",
        )

    return {"typing": typing}


def analysis_seed_bimodality(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """A.7: per-seed distributions — witness count and basin share per model,
    not just the mean."""
    from visualization.experiment_plots import plot_seed_bimodality

    rows = []

    for dataset_type in df["dataset_type"].unique():
        target, _, _ = _dataset_features_by_role(df, dataset_type)
        g = df[
            (df["dataset_type"] == dataset_type)
            & (df["feature_name"] == target)
            & (df["kind"] == "post")
        ].copy()
        g["_witness"] = _coerce_bool_series(g["is_structural_witness"])

        for model_path, gm in g.groupby("model_path"):
            rows.append(
                {
                    "dataset_type": dataset_type,
                    "model_path": model_path,
                    "n_witnesses": int(gm["_witness"].sum()),
                    "has_witness": bool(gm["_witness"].any()),
                }
            )

    per_model = pd.DataFrame(rows)

    typing = _typing_table(df, ctx)
    if len(typing) > 0:
        shares = (
            typing.groupby(["dataset_type", "model_path"])["witness_type"]
            .apply(lambda s: float((s == "basin").mean()))
            .rename("basin_share")
            .reset_index()
        )
        per_model = per_model.merge(
            shares, on=["dataset_type", "model_path"], how="left"
        )

    per_model.to_csv(
        ctx.aggregates_dir / "seed_bimodality_per_model.csv", index=False
    )

    if len(per_model) > 0:
        plot_seed_bimodality(
            per_model,
            value_col="n_witnesses",
            save_path=ctx.figures_dir / "seed_witness_counts.png",
        )
        if "basin_share" in per_model.columns and per_model["basin_share"].notna().any():
            plot_seed_bimodality(
                per_model[per_model["basin_share"].notna()],
                value_col="basin_share",
                title="Per-seed basin share (A.7 bimodality check)",
                save_path=ctx.figures_dir / "seed_basin_share.png",
            )

    return {"per_model": per_model}


def analysis_basin_ridge_vs_ratio(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """Exp 4: witness type composition as a function of the density ratio
    (dataset names ring_den{n_ring}x{n_interior})."""
    import re

    from visualization.experiment_plots import plot_basin_ridge_bars

    typing = _typing_table(df, ctx)

    if len(typing) == 0:
        print("[aggregate] basin_ridge_vs_ratio: no witnesses to type")
        return {"typing": typing, "by_ratio": pd.DataFrame()}

    def _ratio(name: str):
        m = re.search(r"den(\d+)x(\d+)", str(name))
        return float(m.group(1)) / float(m.group(2)) if m else float("nan")

    typing = typing.copy()
    typing["density_ratio"] = typing["dataset_type"].map(_ratio)

    by_ratio = (
        typing.groupby(["dataset_type", "density_ratio", "witness_type"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = by_ratio.groupby("dataset_type")["n"].transform("sum")
    by_ratio["fraction"] = by_ratio["n"] / totals

    by_ratio.sort_values("density_ratio", inplace=True)
    by_ratio.to_csv(
        ctx.aggregates_dir / "basin_ridge_vs_ratio.csv", index=False
    )

    order = (
        typing.drop_duplicates("dataset_type")
        .sort_values("density_ratio")["dataset_type"]
        .tolist()
    )
    typing["dataset_type"] = pd.Categorical(
        typing["dataset_type"], categories=order, ordered=True
    )

    plot_basin_ridge_bars(
        typing,
        title="Witness type vs density ratio (ring : interior)",
        save_path=ctx.figures_dir / "basin_ridge_vs_ratio.png",
    )

    return {"typing": typing, "by_ratio": by_ratio}


def analysis_sparse_probing(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """A.4 / exp 3: k-sparse probes per (dataset, model, layer) + agreement
    with the structural witnesses."""
    from management.experiment_management import get_feature_points
    from analysis.probing import (
        collect_activation_dataset,
        feature_membership_targets,
        fit_sparse_probe,
        probe_vs_witness_table,
    )

    probe_rows = []
    agreements = {}

    for dataset_type in df["dataset_type"].unique():
        target, _, _ = _dataset_features_by_role(df, dataset_type)
        g = df[df["dataset_type"] == dataset_type]

        instance_path = g["dataset_instance_path"].iloc[0]
        loaded = ctx.load_instance(instance_path)
        X = loaded["X"]
        feature_points = get_feature_points(loaded, target)
        delta = _feature_scale_delta(feature_points)

        targets = feature_membership_targets(X, feature_points, delta=delta)

        probe_results = {}

        for model_path in g["model_path"].unique():
            model, _ = ctx.load_model(model_path)
            layer_ids = sorted(
                g.loc[g["model_path"] == model_path, "layer_id"]
                .astype(int)
                .unique()
            )
            activations = collect_activation_dataset(
                model, X, layer_ids=layer_ids
            )

            for layer_id in layer_ids:
                H = activations[(layer_id, "post")]

                for k in (1, 2):
                    if k > H.shape[1]:
                        continue
                    probe = fit_sparse_probe(H, targets, k=k)

                    if k == 1:
                        probe_results[(model_path, layer_id)] = probe

                    probe_rows.append(
                        {
                            "dataset_type": dataset_type,
                            "feature_name": target,
                            "model_path": model_path,
                            "layer_id": layer_id,
                            "k": k,
                            "selected_neurons": probe["selected_neurons"],
                            "cv_auc": probe["cv_auc"],
                            "cv_accuracy": probe["cv_accuracy"],
                            "method": probe["method"],
                            "delta": delta,
                        }
                    )

        if probe_results:
            agreements[dataset_type] = probe_vs_witness_table(
                probe_results, df, dataset_type, target
            )
            agreements[dataset_type]["agreement"].to_csv(
                ctx.aggregates_dir / f"probe_vs_witness_{dataset_type}.csv",
                index=False,
            )

    probe_table = pd.DataFrame(probe_rows)
    probe_table.to_csv(ctx.aggregates_dir / "sparse_probes.csv", index=False)

    return {"probes": probe_table, "agreement": agreements}


def analysis_ablation(df: pd.DataFrame, ctx: AggregationContext) -> dict:
    """A.3 / exp 7: region-selective accuracy drop for every witness."""
    from management.experiment_management import get_feature_points
    from analysis.ablation import ablation_impact

    rows = []

    for dataset_type in df["dataset_type"].unique():
        g = df[df["dataset_type"] == dataset_type]
        instance_path = g["dataset_instance_path"].iloc[0]
        loaded = ctx.load_instance(instance_path)
        X, y = loaded["X"], loaded["y"]

        _, nulls, _ = _dataset_features_by_role(df, dataset_type)
        features = [
            f for f in g["feature_name"].unique() if f not in nulls
        ]

        for feature_name in features:
            feature_points = get_feature_points(loaded, feature_name)
            delta = _feature_scale_delta(feature_points)
            witnesses = _witness_rows(df, dataset_type, feature_name)

            for _, w in witnesses.iterrows():
                model, _ = ctx.load_model(w["model_path"])

                impact = ablation_impact(
                    model,
                    X,
                    y,
                    feature_points,
                    layer_id=int(w["layer_id"]),
                    coord_id=int(w["coord_id"]),
                    delta=delta,
                )

                rows.append(
                    {
                        "dataset_type": dataset_type,
                        "feature_name": feature_name,
                        "model_path": w["model_path"],
                        **impact,
                    }
                )

    ablation_table = pd.DataFrame(rows)
    ablation_table.to_csv(
        ctx.aggregates_dir / "ablation_impacts.csv", index=False
    )

    return {"ablation": ablation_table}


def analysis_criterion_diagnostics(
    df: pd.DataFrame, ctx: AggregationContext
) -> dict:
    """Surface the per-criterion validity diagnostics into a durable table.

    Every criterion reports, alongside its boolean verdict, the numbers needed
    to decide whether that verdict is trustworthy — the size of the level
    search, the gradient norm at the anchor, how many levels the sampler
    actually covered. They all reached the CSV and were read by NOTHING, so a
    run whose anchors were all saturated plateaus, or whose sampler covered 2
    of 9 levels, produced tables indistinguishable from a clean one
    (AUDIT/raw/plan-lead-items.md S1). Three of them are quantities the
    project's own documents instruct the reader to check.

    Writes ``aggregates/criterion_diagnostics.csv``: one row per
    (dataset, model_name, criterion, diagnostic) with min/median/max, and a
    ``concern`` column flagging the known failure directions.
    """
    # column -> (human reading, direction that indicates trouble)
    WATCHED = {
        "extremum_n_candidate_levels":
            ("size of the level search (criterion 3)", "low"),
        "extremum_best_anchor_grad_norm":
            ("|grad f| at the anchor — high means a saturated tanh plateau, "
             "so 'criticality' is diffuse (PLAN.md:242)", "high"),
        "extremum_n_extrema": ("extrema found", "low"),
        "extremum_frac_anchors_on_boundary":
            ("fraction of criterion-3 anchors that are box-clamp extremes "
             "rather than interior critical points — 1.0 means the criterion "
             "had nothing to anchor to and its verdict is about the domain, "
             "not the network (AUDIT/09 B.2)", "high"),
        "homological_persistence_margin":
            ("margin over the noise bar / floor", "low"),
        "homological_n_sample_points":
            ("points carrying the filtration", "low"),
        "field_grad_norm_on_feature":
            ("mean |grad f| ON this feature — LOW means the neuron is "
             "saturated here, so no level set can trace the feature "
             "regardless of encoding, and a negative verdict is geometry "
             "rather than evidence (AUDIT/10)", "low"),
        "field_value_std_on_feature":
            ("spread of f across the feature — the dynamic range available "
             "to a level set here", "low"),
        "sampled_structural_n_levels_covered":
            ("levels the sampler actually covered — low means under-sampling "
             "masquerading as 'witness not found' (RUNBOOK.md:264)", "low"),
        "sampled_structural_min_level_coverage": ("worst per-level coverage", "low"),
    }

    diagnostic_cols = [c for c in WATCHED if c in df.columns]

    if not diagnostic_cols:
        print("[aggregate] criterion_diagnostics: no diagnostic columns in the "
              "CSV (no extra criteria were run)")
        table = pd.DataFrame()
        table.to_csv(ctx.aggregates_dir / "criterion_diagnostics.csv", index=False)
        return {"diagnostics": table}

    group_cols = [c for c in ("dataset_type", "model_name") if c in df.columns]
    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for col in diagnostic_cols:
            values = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(values) == 0:
                continue

            reading, bad_direction = WATCHED[col]
            lo, med, hi = (float(values.min()), float(values.median()),
                           float(values.max()))

            # Flag only the direction that actually indicates trouble.
            concern = ""
            if col == "extremum_best_anchor_grad_norm" and hi > 1e-2:
                concern = (f"max |grad| {hi:.3g} — some anchors are not critical "
                           f"points; criterion-3 levels there are unreliable")
            elif col == "sampled_structural_n_levels_covered":
                scanned = pd.to_numeric(
                    g.get("sampled_structural_n_levels_scanned", pd.Series(dtype=float)),
                    errors="coerce").dropna()
                if len(scanned) and lo < 0.5 * float(scanned.median()):
                    concern = (f"min coverage {lo:.0f} of ~{scanned.median():.0f} "
                               f"levels — negatives may be under-sampling")
            elif bad_direction == "low" and lo <= 0:
                concern = f"min {lo:.3g} — degenerate for some rows"

            rows.append({
                **dict(zip(group_cols, keys)),
                "criterion": col.split("_")[0],
                "diagnostic": col,
                "reading": reading,
                "min": lo, "median": med, "max": hi,
                "n_rows": int(len(values)),
                "concern": concern,
            })

    table = pd.DataFrame(rows)
    table.to_csv(ctx.aggregates_dir / "criterion_diagnostics.csv", index=False)

    flagged = table[table["concern"] != ""] if len(table) else table
    for _, r in flagged.iterrows():
        print(f"[aggregate] diagnostic concern: {r['diagnostic']} — {r['concern']}")
    if len(table) and not len(flagged):
        print(f"[aggregate] criterion_diagnostics: {len(table)} diagnostic(s) "
              f"summarized, none flagged")

    return {"diagnostics": table}


ANALYSES = {
    "witness_tables": analysis_witness_tables,
    "criterion_diagnostics": analysis_criterion_diagnostics,
    "witness_rate_gap": analysis_witness_rate_gap,
    "rate_gap_by_criterion": analysis_rate_gap_by_criterion,
    "grid_witness_rates": analysis_grid_witness_rates,
    "rate_vs_width": analysis_rate_vs_width,
    "feature_overlap": analysis_feature_overlap,
    "consistency_table": analysis_consistency_table,
    "layer_persistence": analysis_layer_persistence,
    "basin_ridge_typing": analysis_basin_ridge_typing,
    "seed_bimodality": analysis_seed_bimodality,
    "basin_ridge_vs_ratio": analysis_basin_ridge_vs_ratio,
    "sparse_probing": analysis_sparse_probing,
    "ablation": analysis_ablation,
}
