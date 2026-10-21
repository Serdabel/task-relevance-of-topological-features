"""Aggregation of per-neuron criterion statistics into witness tables.

Extracted from mass_training.ipynb so the paper's aggregate tables
(analysis/results/aggregates/*.csv) are reproducible from library code.

Input is the flat per-neuron CSV/DataFrame produced by
``analysis.stat_aggregation.collect_compact_neuron_statistics`` (one row per
neuron-feature-model). Outputs are the witness count / overlap tables used in
PROJECT.md §Implemented Experiments and the corresponding figures
(see ``visualization.witness_plots``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET_ORDER = [
    "a_empty_irrelevant",
    "b_filled_same_irrelevant",
    "c_filled_opposite_relevant",
    "a1_empty_irrelevant",
    "b1_filled_same_irrelevant",
    "c1_filled_opposite_relevant",
]

DEFAULT_DATASET_LABELS = {
    "a_empty_irrelevant": "a",
    "b_filled_same_irrelevant": "b",
    "c_filled_opposite_relevant": "c",
    "a1_empty_irrelevant": "a.1",
    "b1_filled_same_irrelevant": "b.1",
    "c1_filled_opposite_relevant": "c.1",
}


def _load_stats_source(stats_source):
    """
    stats_source can be:
        - pd.DataFrame
        - path to csv

    feature_role is read verbatim: pandas would otherwise parse the role
    value "null" (A.1 null features) as NaN.
    """
    if isinstance(stats_source, pd.DataFrame):
        return stats_source.copy()

    stats_source = Path(stats_source)
    assert stats_source.exists(), f"File does not exist: {stats_source}"

    return pd.read_csv(
        stats_source, converters={"feature_role": lambda v: v}
    )


def _coerce_bool_series(s):
    """
    Robust conversion of bool-like column loaded from csv.
    """
    if s.dtype == bool:
        return s

    if np.issubdtype(s.dtype, np.number):
        return s.astype(bool)

    return (
        s.astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def _filter_model_name(df, model_name=None):
    """
    Filters by stored model_name.

    If exact df['model_name'] match exists, uses it.
    Otherwise tries substring match in model_instance_name.
    """
    if model_name is None:
        return df

    if "model_name" in df.columns:
        exact_mask = df["model_name"].astype(str) == str(model_name)
        if exact_mask.any():
            return df[exact_mask].copy()

    if "model_instance_name" in df.columns:
        contains_mask = df["model_instance_name"].astype(str).str.contains(
            str(model_name),
            regex=False,
            na=False,
        )
        if contains_mask.any():
            return df[contains_mask].copy()

    raise ValueError(
        f"No models found for model_name={model_name!r}. "
        "Check df['model_name'] and df['model_instance_name']."
    )


def _bootstrap_ci_mean(
    values,
    n_bootstrap=5000,
    ci=0.95,
    random_state=0,
):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(random_state)

    # Vectorized resampling: identical distribution to a per-iteration loop,
    # but exact CI endpoints differ in trailing decimals from the historical
    # (loop-based) notebook run for the same random_state.
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    boot_means = samples.mean(axis=1)

    alpha = 1.0 - ci
    low, high = np.quantile(
        boot_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return float(low), float(high)


def _summarize_numeric_group(
    group,
    value_col,
    n_bootstrap=5000,
    ci=0.95,
    random_state=0,
):
    values = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return pd.Series({
            "n_models": 0,
            f"mean_{value_col}": np.nan,
            f"std_{value_col}": np.nan,
            f"sem_{value_col}": np.nan,
            f"ci_low_{value_col}": np.nan,
            f"ci_high_{value_col}": np.nan,
        })

    ci_low, ci_high = _bootstrap_ci_mean(
        values,
        n_bootstrap=n_bootstrap,
        ci=ci,
        random_state=random_state,
    )

    return pd.Series({
        "n_models": len(values),
        f"mean_{value_col}": float(np.mean(values)),
        f"std_{value_col}": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"sem_{value_col}": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        f"ci_low_{value_col}": ci_low,
        f"ci_high_{value_col}": ci_high,
    })


def build_witness_aggregate_tables(
    stats_source,
    dataset_types=None,
    feature_names=None,
    model_name=None,
    n_bootstrap=5000,
    ci=0.95,
    random_state=0,
    save_dir="analysis/results/aggregates",
    save_prefix="witness",
    witness_column="is_structural_witness",
):
    """
    Builds aggregate witness tables from compact_neuron_feature_stats.csv.

    Designed for the output of collect_compact_neuron_statistics.

    Parameters
    ----------
    stats_source:
        pd.DataFrame or path to csv.

    dataset_types:
        list of dataset_type names. If None, uses all available.

    feature_names:
        list of feature names. If None, uses all available.

    model_name:
        optional filter by stored model_name, e.g. "tanh_w7_d5".

    witness_column:
        which boolean column carries the witness verdict (default the
        grid-structural criterion; hi-dim experiments pass e.g.
        "homological_is_witness"). The OUTPUT tables keep the canonical
        column name "is_structural_witness" regardless, so downstream
        plots work unchanged; the source column is recorded in
        result["witness_column"].

    Returns
    -------
    dict with:
        neuron_table
        model_feature_layer
        layer_summary
        model_feature_total
        pair_summary
        witness_column
    """

    df = _load_stats_source(stats_source)

    required_cols = [
        "dataset_type",
        "feature_name",
        "model_path",
        "layer_id",
        "kind",
        "coord_id",
        witness_column,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    assert len(missing) == 0, f"Missing required columns: {missing}"

    if witness_column != "is_structural_witness":
        df = df.copy()
        df["is_structural_witness"] = df[witness_column]
    else:
        assert df[witness_column].notna().any(), (
            "is_structural_witness has no values — the evaluation ran with "
            "run_structural=False; pass witness_column explicitly (e.g. "
            "'homological_is_witness')"
        )

    df = _filter_model_name(df, model_name=model_name)

    if dataset_types is not None:
        df = df[df["dataset_type"].isin(dataset_types)].copy()

    if feature_names is not None:
        df = df[df["feature_name"].isin(feature_names)].copy()

    assert len(df) > 0, "No rows left after filtering."

    df["is_structural_witness"] = _coerce_bool_series(
        df["is_structural_witness"]
    )

    # Use model_path as the main model identifier.
    # model_id is only unique inside one dataset/model_name family.
    optional_model_cols = [
        "model_id",
        "model_name",
        "model_instance_name",
        "hidden_dim",
        "n_layers",
        "activation",
    ]

    optional_model_cols = [
        col for col in optional_model_cols
        if col in df.columns
    ]

    # One row per neuron-feature-model.
    keep_cols = [
        "dataset_type",
        "feature_name",
        "model_path",
        "layer_id",
        "kind",
        "coord_id",
        "is_structural_witness",
    ]

    for col in optional_model_cols:
        if col not in keep_cols:
            keep_cols.append(col)

    for col in [
        "structural_score",
        "structural_score_normalized_by_diameter",
        "structural_score_normalized_by_grid_step",
        "tubular_score",
    ]:
        if col in df.columns:
            keep_cols.append(col)

    neuron_table = df[keep_cols].copy()

    neuron_table["neuron_key"] = (
        neuron_table["kind"].astype(str)
        + ":L"
        + neuron_table["layer_id"].astype(str)
        + ":N"
        + neuron_table["coord_id"].astype(str)
    )

    # One row per dataset-feature-model-layer.
    group_cols_layer = [
        "dataset_type",
        "feature_name",
        "model_path",
        "layer_id",
    ]

    model_feature_layer = (
        neuron_table
        .groupby(group_cols_layer, dropna=False)
        .agg(
            witness_count=("is_structural_witness", "sum"),
            n_neurons=("is_structural_witness", "size"),
        )
        .reset_index()
    )

    model_feature_layer["witness_fraction"] = (
        model_feature_layer["witness_count"]
        / model_feature_layer["n_neurons"]
    )

    # Add optional model metadata to model_feature_layer.
    model_meta_cols = ["dataset_type", "model_path"] + optional_model_cols
    model_meta_cols = list(dict.fromkeys(model_meta_cols))

    model_meta = (
        neuron_table[model_meta_cols]
        .drop_duplicates(subset=["dataset_type", "model_path"])
    )

    model_feature_layer = model_feature_layer.merge(
        model_meta,
        on=["dataset_type", "model_path"],
        how="left",
    )

    # Layer summary: one row per dataset-feature-layer.
    layer_summary = (
        model_feature_layer
        .groupby(["dataset_type", "feature_name", "layer_id"], dropna=False)
        .apply(
            lambda g: _summarize_numeric_group(
                g,
                value_col="witness_count",
                n_bootstrap=n_bootstrap,
                ci=ci,
                random_state=random_state,
            )
        )
        .reset_index()
    )

    # Also summarize layer-wise witness fraction.
    layer_fraction_summary = (
        model_feature_layer
        .groupby(["dataset_type", "feature_name", "layer_id"], dropna=False)
        .apply(
            lambda g: _summarize_numeric_group(
                g,
                value_col="witness_fraction",
                n_bootstrap=n_bootstrap,
                ci=ci,
                random_state=random_state,
            )
        )
        .reset_index()
    )

    layer_summary = layer_summary.merge(
        layer_fraction_summary,
        on=["dataset_type", "feature_name", "layer_id"],
        how="left",
        suffixes=("", "_fraction"),
    )

    # One row per dataset-feature-model.
    group_cols_total = [
        "dataset_type",
        "feature_name",
        "model_path",
    ]

    model_feature_total = (
        neuron_table
        .groupby(group_cols_total, dropna=False)
        .agg(
            total_witness_count=("is_structural_witness", "sum"),
            total_neurons=("is_structural_witness", "size"),
            has_witness=("is_structural_witness", "any"),
            best_structural_score_normalized_by_diameter=(
                "structural_score_normalized_by_diameter",
                "min",
            ) if "structural_score_normalized_by_diameter" in neuron_table.columns else (
                "is_structural_witness",
                "size",
            ),
        )
        .reset_index()
    )

    model_feature_total["total_witness_fraction"] = (
        model_feature_total["total_witness_count"]
        / model_feature_total["total_neurons"]
    )

    model_feature_total = model_feature_total.merge(
        model_meta,
        on=["dataset_type", "model_path"],
        how="left",
    )

    # Pair summary: one row per dataset-feature.
    total_count_summary = (
        model_feature_total
        .groupby(["dataset_type", "feature_name"], dropna=False)
        .apply(
            lambda g: _summarize_numeric_group(
                g,
                value_col="total_witness_count",
                n_bootstrap=n_bootstrap,
                ci=ci,
                random_state=random_state,
            )
        )
        .reset_index()
    )

    total_fraction_summary = (
        model_feature_total
        .groupby(["dataset_type", "feature_name"], dropna=False)
        .apply(
            lambda g: _summarize_numeric_group(
                g,
                value_col="total_witness_fraction",
                n_bootstrap=n_bootstrap,
                ci=ci,
                random_state=random_state,
            )
        )
        .reset_index()
    )

    best_score_summary = (
        model_feature_total
        .groupby(["dataset_type", "feature_name"], dropna=False)
        .apply(
            lambda g: _summarize_numeric_group(
                g,
                value_col="best_structural_score_normalized_by_diameter",
                n_bootstrap=n_bootstrap,
                ci=ci,
                random_state=random_state,
            )
        )
        .reset_index()
    )

    has_witness_summary = (
        model_feature_total
        .groupby(["dataset_type", "feature_name"], dropna=False)
        .agg(
            n_models=("model_path", "nunique"),
            p_has_witness=("has_witness", "mean"),
        )
        .reset_index()
    )

    pair_summary = total_count_summary.merge(
        total_fraction_summary,
        on=["dataset_type", "feature_name"],
        how="left",
        suffixes=("", "_fraction"),
    )

    pair_summary = pair_summary.merge(
        best_score_summary,
        on=["dataset_type", "feature_name"],
        how="left",
        suffixes=("", "_best_score"),
    )

    pair_summary = pair_summary.merge(
        has_witness_summary,
        on=["dataset_type", "feature_name"],
        how="left",
        suffixes=("", "_has_witness"),
    )

    # Sort tables.
    if dataset_types is None:
        dataset_types = list(pair_summary["dataset_type"].unique())

    dataset_order = {name: i for i, name in enumerate(dataset_types)}

    for table in [
        neuron_table,
        model_feature_layer,
        layer_summary,
        model_feature_total,
        pair_summary,
    ]:
        table["_dataset_order"] = table["dataset_type"].map(dataset_order)
        table.sort_values(
            [col for col in ["_dataset_order", "feature_name", "layer_id", "model_path"] if col in table.columns],
            inplace=True,
        )
        table.drop(columns=["_dataset_order"], inplace=True)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        neuron_table.to_csv(save_dir / f"{save_prefix}_neuron_table.csv", index=False)
        model_feature_layer.to_csv(save_dir / f"{save_prefix}_model_feature_layer.csv", index=False)
        layer_summary.to_csv(save_dir / f"{save_prefix}_layer_summary.csv", index=False)
        model_feature_total.to_csv(save_dir / f"{save_prefix}_model_feature_total.csv", index=False)
        pair_summary.to_csv(save_dir / f"{save_prefix}_pair_summary.csv", index=False)

    return {
        "neuron_table": neuron_table,
        "model_feature_layer": model_feature_layer,
        "layer_summary": layer_summary,
        "model_feature_total": model_feature_total,
        "pair_summary": pair_summary,
        "witness_column": witness_column,
    }


def build_feature_overlap_tables(
    neuron_table,
    feature1,
    feature2,
    dataset_types=None,
    model_name=None,
    n_bootstrap=5000,
    ci=0.95,
    random_state=0,
    save_dir="analysis/results/aggregates",
    save_prefix=None,
):
    """
    Builds overlap tables for witnesses of two features.

    Categories:
        feature1_only = W1 \\ W2
        feature2_only = W2 \\ W1
        both          = W1 ∩ W2

    A neuron is identified by:
        (kind, layer_id, coord_id)

    Parameters
    ----------
    neuron_table:
        tables["neuron_table"] from build_witness_aggregate_tables.

    feature1, feature2:
        feature names, e.g. "C_big", "C_mid".

    Returns
    -------
    dict with:
        overlap_by_model
        overlap_membership
        overlap_summary
    """

    df = neuron_table.copy()

    if model_name is not None:
        df = _filter_model_name(df, model_name=model_name)

    if dataset_types is not None:
        df = df[df["dataset_type"].isin(dataset_types)].copy()

    df = df[df["feature_name"].isin([feature1, feature2])].copy()

    assert len(df) > 0, "No rows left after filtering."

    df["is_structural_witness"] = _coerce_bool_series(
        df["is_structural_witness"]
    )

    if "neuron_key" not in df.columns:
        df["neuron_key"] = (
            df["kind"].astype(str)
            + ":L"
            + df["layer_id"].astype(str)
            + ":N"
            + df["coord_id"].astype(str)
        )

    # We work model-by-model inside each dataset.
    by_model_rows = []
    membership_rows = []

    group_cols = ["dataset_type", "model_path"]

    for (dataset_type, model_path), g in df.groupby(group_cols, dropna=False):
        available_features = set(g["feature_name"].unique())

        if feature1 not in available_features or feature2 not in available_features:
            continue

        g1 = g[
            (g["feature_name"] == feature1)
            & (g["is_structural_witness"])
        ]

        g2 = g[
            (g["feature_name"] == feature2)
            & (g["is_structural_witness"])
        ]

        W1 = set(g1["neuron_key"])
        W2 = set(g2["neuron_key"])

        feature1_only = W1 - W2
        feature2_only = W2 - W1
        both = W1 & W2
        union = W1 | W2

        n1 = len(W1)
        n2 = len(W2)
        n_both = len(both)
        n_union = len(union)

        row = {
            "dataset_type": dataset_type,
            "model_path": model_path,
            "feature1": feature1,
            "feature2": feature2,

            "n_feature1": n1,
            "n_feature2": n2,
            "n_feature1_only": len(feature1_only),
            "n_feature2_only": len(feature2_only),
            "n_both": n_both,
            "n_union": n_union,

            "feature1_shared_fraction": n_both / n1 if n1 > 0 else np.nan,
            "feature2_shared_fraction": n_both / n2 if n2 > 0 else np.nan,
            "jaccard": n_both / n_union if n_union > 0 else np.nan,
        }

        # Add available model metadata.
        for col in [
            "model_id",
            "model_name",
            "model_instance_name",
            "hidden_dim",
            "n_layers",
            "activation",
        ]:
            if col in g.columns:
                values = g[col].dropna().unique()
                row[col] = values[0] if len(values) > 0 else None

        by_model_rows.append(row)

        for neuron_key in sorted(feature1_only):
            membership_rows.append({
                "dataset_type": dataset_type,
                "model_path": model_path,
                "feature1": feature1,
                "feature2": feature2,
                "neuron_key": neuron_key,
                "category": "feature1_only",
            })

        for neuron_key in sorted(feature2_only):
            membership_rows.append({
                "dataset_type": dataset_type,
                "model_path": model_path,
                "feature1": feature1,
                "feature2": feature2,
                "neuron_key": neuron_key,
                "category": "feature2_only",
            })

        for neuron_key in sorted(both):
            membership_rows.append({
                "dataset_type": dataset_type,
                "model_path": model_path,
                "feature1": feature1,
                "feature2": feature2,
                "neuron_key": neuron_key,
                "category": "both",
            })

    overlap_by_model = pd.DataFrame(by_model_rows)
    overlap_membership = pd.DataFrame(membership_rows)

    assert len(overlap_by_model) > 0, (
        "No model has both requested features. "
        f"Check feature names: {feature1}, {feature2}."
    )

    metrics_to_summarize = [
        "n_feature1",
        "n_feature2",
        "n_feature1_only",
        "n_feature2_only",
        "n_both",
        "n_union",
        "feature1_shared_fraction",
        "feature2_shared_fraction",
        "jaccard",
    ]

    summary_parts = []

    for metric in metrics_to_summarize:
        cur_summary = (
            overlap_by_model
            .groupby(["dataset_type", "feature1", "feature2"], dropna=False)
            .apply(
                lambda g: _summarize_numeric_group(
                    g,
                    value_col=metric,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    random_state=random_state,
                )
            )
            .reset_index()
        )

        summary_parts.append(cur_summary)

    overlap_summary = summary_parts[0]

    for part in summary_parts[1:]:
        # Every per-metric summary carries an identical n_models column; keep
        # only the first to avoid duplicate columns on merge (pandas >= 2
        # raises MergeError instead of silently suffixing them).
        part = part.drop(columns=["n_models"], errors="ignore")
        overlap_summary = overlap_summary.merge(
            part,
            on=["dataset_type", "feature1", "feature2"],
            how="left",
        )

    # Sort by dataset order if provided.
    if dataset_types is not None:
        order = {name: i for i, name in enumerate(dataset_types)}
        for table in [overlap_by_model, overlap_membership, overlap_summary]:
            table["_dataset_order"] = table["dataset_type"].map(order)
            table.sort_values(
                [col for col in ["_dataset_order", "model_path", "category", "neuron_key"] if col in table.columns],
                inplace=True,
            )
            table.drop(columns=["_dataset_order"], inplace=True)

    if save_prefix is None:
        save_prefix = f"overlap_{feature1}_vs_{feature2}"

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        overlap_by_model.to_csv(
            save_dir / f"{save_prefix}_by_model.csv",
            index=False,
        )

        overlap_membership.to_csv(
            save_dir / f"{save_prefix}_membership.csv",
            index=False,
        )

        overlap_summary.to_csv(
            save_dir / f"{save_prefix}_summary.csv",
            index=False,
        )

    return {
        "overlap_by_model": overlap_by_model,
        "overlap_membership": overlap_membership,
        "overlap_summary": overlap_summary,
    }


def get_pair_value(pair_summary, dataset_type, feature_name, value_col):
    """Extracts a single dataset-feature cell from pair_summary."""
    row = pair_summary[
        (pair_summary["dataset_type"] == dataset_type)
        & (pair_summary["feature_name"] == feature_name)
    ]
    assert len(row) == 1, (dataset_type, feature_name, len(row))
    return float(row[value_col].iloc[0])


def build_comparison_table(
    pair_summary,
    comparison_specs,
    value_col="mean_total_witness_count",
):
    """
    Builds the compact comparison table for the paper's key contrasts.

    comparison_specs:
        list of tuples (name, meaning, dataset_pos, dataset_neg, feature_name),
        e.g. ("c - b", "task relevance, centered",
              "c_filled_opposite_relevant", "b_filled_same_irrelevant", "C_mid").
    """
    comparisons = []

    for name, meaning, dataset_pos, dataset_neg, feature_name in comparison_specs:
        mean_pos = get_pair_value(pair_summary, dataset_pos, feature_name, value_col)
        mean_neg = get_pair_value(pair_summary, dataset_neg, feature_name, value_col)

        comparisons.append({
            "comparison": name,
            "meaning": meaning,
            "feature": feature_name,
            "mean_positive_dataset": mean_pos,
            "mean_negative_dataset": mean_neg,
            "mean_difference": mean_pos - mean_neg,
        })

    return pd.DataFrame(comparisons)


def witness_counts_by_model_layer(df):
    """
    Pivot table: rows = model_id, columns = layer_id, values = witness counts.

    Input: the flat per-neuron stats DataFrame (or a filtered subset).
    """
    all_layers = sorted(df["layer_id"].unique())

    return (
        df[_coerce_bool_series(df["is_structural_witness"])]
        .groupby(["model_id", "layer_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=all_layers, fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def consistency_table(
    stats_source,
    witness_columns: dict | None = None,
    group_by: tuple = ("dataset_type", "feature_name"),
    model_name: str | None = None,
):
    """
    Criteria agreement matrix (PLAN.md §C, "Consistency table").

    Cross-tabulates the witness verdicts of the implemented criteria —
    structural x extremum-anchored x homological — per dataset x feature.
    Disagreements are the diagnostic output, not noise:

        geometric-yes / extremum-no  -> likely incidental witness;
        homology-yes / geometric-no  -> topologically present but deformed.

    witness_columns:
        {criterion_label: column_name} in the per-neuron table, default
        {"structural": "is_structural_witness",
         "sampled_structural": "sampled_structural_is_witness",
         "extremum": "extremum_is_witness",
         "homological": "homological_is_witness"} — only columns actually
        present in the table AND carrying at least one non-null value are
        used (a criterion that was switched off leaves an all-null column).

    Returns pd.DataFrame: one row per group x verdict-combination, with one
    boolean column per criterion, n_neurons, fraction (within the group),
    and per-criterion group-level witness rates (rate_{label}) repeated on
    every row of the group for context.
    """
    if witness_columns is None:
        witness_columns = {
            "structural": "is_structural_witness",
            "sampled_structural": "sampled_structural_is_witness",
            "extremum": "extremum_is_witness",
            "homological": "homological_is_witness",
        }

    df = _load_stats_source(stats_source)
    df = _filter_model_name(df, model_name=model_name)

    present = {
        label: col
        for label, col in witness_columns.items()
        if col in df.columns and df[col].notna().any()
    }

    assert len(present) >= 2, (
        f"Need at least two witness columns to cross-tabulate; found "
        f"{list(present)} of {witness_columns} in the table. Run the "
        f"evaluation with extra_criteria=('extremum', 'homological') first."
    )

    labels = list(present.keys())

    for label, col in present.items():
        df[f"_w_{label}"] = _coerce_bool_series(df[col])

    rows = []

    for group_key, g in df.groupby(list(group_by), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_id = dict(zip(group_by, group_key))
        n_total = len(g)
        rates = {
            f"rate_{label}": float(g[f"_w_{label}"].mean()) for label in labels
        }

        combo_counts = (
            g.groupby([f"_w_{label}" for label in labels], dropna=False)
            .size()
            .reset_index(name="n_neurons")
        )

        for _, combo in combo_counts.iterrows():
            row = dict(group_id)
            for label in labels:
                row[label] = bool(combo[f"_w_{label}"])
            row["n_neurons"] = int(combo["n_neurons"])
            row["fraction"] = float(combo["n_neurons"] / n_total)
            row["n_group_neurons"] = n_total
            row.update(rates)
            rows.append(row)

    table = pd.DataFrame(rows)

    sort_cols = list(group_by) + labels
    table.sort_values(sort_cols, inplace=True, ignore_index=True)

    return table
