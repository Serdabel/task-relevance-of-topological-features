"""Experiment 2 read-out: the architecture sweep as tables, trends and fits.

The counterpart of ``analysis/calibration.py`` for experiment 2. That module
measures the instrument; this one reads the finished per-neuron table of the
architecture pool and turns it into the quantities tasks A and B are decided
on. It computes nothing about networks that ``collect_compact_neuron_statistics``
has not already computed — every number here is an aggregate of
``<results>/neuron_stats_*.csv``.

TWO CLASSES OF NEURON CANNOT WITNESS A CIRCLE, BY THEOREM, AND BOTH ARE IN THE
POOL ON PURPOSE.

A neuron at ``layer_id == 0`` with ``kind == "post"`` is ``tanh(w.x + b)``. Its
level sets are the preimages of points under an affine map composed with a
strictly monotone scalar function, i.e. straight lines; no training makes one
into a circle. Independently, a tanh stack of width <= 2 on R^2 is a
diffeomorphism onto its image, so a level set of ANY of its neurons is the
preimage of a line and never a closed loop (Johnson, ICLR 2019, "Deep, Skinny
Neural Networks are not Universal Approximators"; the bounded-decision-region
threshold is width n+1 = 3 here).

Both are confirmed on the pool: 0 witnesses of 31 950 rows at layer 0, and 0 of
2 250 at width <= 2. They are kept because they are the experiment's only
theorem-backed validity check — a criterion that fired there would be broken —
but two consequences must survive into the write-up:

* A depth-1 network IS layer 0. Its zero is an identity, not a measurement, and
  ``w24 d1`` (24 neurons, zero witnesses) is the demonstration.
* The witness FRACTION has a depth-dependent denominator: at depth 2 half the
  neurons are incapable, at depth 8 an eighth. The primary fraction here is
  over ALL neurons, as in experiment 1, and ``witness_fraction_capable``
  carries the same number over capable neurons only — a diagnostic that shows
  whether a depth trend survives the change of denominator, not a second
  headline.

THE FLOOR IS PER CELL. Experiment 1's third transferable rule: a rate at width
W is read only against a floor measured at width W, because the floor itself
falls with the number of tries as about ``N^(-0.28)``. Every statistic here is
therefore paired with the same statistic on the null placements of the same
cell, and ``multiplicity_check`` makes that law explicit.

THE UNIT OF RESAMPLING IS THE MODEL, never the neuron — a frozen convention of
``CALIBRATION.md``. All confidence intervals come from
``calibration.bootstrap_over_models`` over per-model records built once by
``_cell_records``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = "analysis/results/experiments/exp2_arch_sweep"

#: The verdict column and the score behind it. Both frozen by experiment 1.
WITNESS_COLUMN = "is_structural_witness"
SCORE_COLUMN = "structural_score_normalized_by_diameter"
TAU = 0.1

#: The second criterion, carried alongside on purpose. It asks a DIFFERENT
#: question — "does this field have a loop at all" — with no reference to where
#: A sits, so its agreement with the structural verdict is evidence rather than
#: bookkeeping. Reported as a rate: its score is a persistence margin that is
#: NaN wherever no bar of the right dimension exists, which is most neurons.
HOMOLOGICAL_WITNESS_COLUMN = "homological_is_witness"

#: THERE IS NO IMPORTED SCALING LAW HERE, AND THAT IS DELIBERATE.
#:
#: An earlier version of this module carried experiment 1's ``N^(-0.28)`` as
#: the yardstick for "did width buy a better representation, or just more
#: draws". Reading experiment 1 again, that number is about something else
#: entirely: its section B fits it to a sweep over ``n_levels`` (x7.9 LEVELS at
#: a fixed architecture), on the MINIMUM over 10 000 null neurons — and the
#: same section then says the minimum is "величина инструмента, а не сети" and
#: that only the medians transfer. Wrong variable, wrong statistic, and
#: explicitly marked as non-transferable.
#:
#: The right yardstick is measured in this pool instead: the null placements of
#: the SAME cell have the same neuron count and the same multiplicity, and by
#: construction no signal. Their decline with width IS the multiplicity curve,
#: in the same units, on the same statistic. ``floor_scaled_check`` uses it.

#: The two theorem-backed exclusions (see the module docstring).
FIRST_WITNESSING_LAYER = 1
MIN_WITNESSING_WIDTH = 3

#: pandas parses a bare ``null`` cell as NaN and would silently drop exactly
#: the rows the floor is made of.
_CONVERTERS = {"feature_role": lambda v: v}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_neuron_stats(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Every ``neuron_stats_*.csv`` of the pool, with the derived columns.

    Added columns:
      ``score``      the normalised structural score, under a stable name
      ``witness``    the verdict as a real bool
      ``capable``    can this neuron carry a closed level set at all
      ``n_neurons``  width x depth, the model's total
      ``n_capable``  width x (depth - 1), zero at depth 1
      ``arch``       "w5 d5", for labelling rows and selecting populations
      ``tag``        the TRAINING tag of the ensemble the row came from
      ``rung``       the overtraining rung, or "" for an ordinary ensemble

    ``tag`` and ``rung`` exist because ``arch`` alone is not an identity. A
    relu ensemble, a sigmoid ensemble and three overtraining rungs of one run
    are all "w5 d5"; pooling them would be silent and wrong. Selecting a
    population by ``arch`` is only safe inside one ``tag``.
    """
    from analysis.witness_stats import _coerce_bool_series

    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("neuron_stats_*.csv"))
    assert paths, f"нет таблиц neuron_stats_*.csv в {results_dir}"

    df = pd.concat([pd.read_csv(p, converters=_CONVERTERS) for p in paths],
                   ignore_index=True)

    df["score"] = df[SCORE_COLUMN]
    df["witness"] = _coerce_bool_series(df[WITNESS_COLUMN])
    if HOMOLOGICAL_WITNESS_COLUMN in df.columns:
        df["hom_witness"] = _coerce_bool_series(df[HOMOLOGICAL_WITNESS_COLUMN])
    df["capable"] = ((df["layer_id"] >= FIRST_WITNESSING_LAYER)
                     & (df["hidden_dim"] >= MIN_WITNESSING_WIDTH))
    df["n_neurons"] = df["hidden_dim"] * df["n_layers"]
    # Width <= 2 makes EVERY layer incapable, not just the first, so the count
    # is zero there rather than width x (depth - 1).
    df["n_capable"] = np.where(
        df["hidden_dim"] >= MIN_WITNESSING_WIDTH,
        df["hidden_dim"] * (df["n_layers"] - 1),
        0,
    )
    df["arch"] = ("w" + df["hidden_dim"].astype(str)
                  + " d" + df["n_layers"].astype(str))
    df["tag"] = df["model_name"].astype(str)
    # "tanh_w5_d5_otx10" -> "x10"; anything without the marker gets "".
    df["rung"] = df["tag"].str.extract(r"_ot([^_]+)$", expand=False).fillna("")

    assert df["score"].notna().all(), "структурный скор не должен содержать NaN"
    return df


def to_scores(df: pd.DataFrame) -> pd.DataFrame:
    """The long schema ``visualization/calibration_plots.py`` already speaks.

    ``plot_unit_strips`` and ``plot_populations`` were written for experiment
    1's ``ensemble_scores`` output — ``criterion/score/model_path/feature/role``
    plus whatever columns a selector wants. Producing that view of the
    per-neuron CSV is the whole adapter: the figures then work on experiment 2
    without a line of new drawing code.

    Both criteria are emitted. Their directions differ (``calibration.DIRECTIONS``
    knows: structural lower-is-better, homological margin higher-is-better), so
    never pool them.
    """
    common = ["model_path", "dataset_type", "hidden_dim", "n_layers",
              "n_neurons", "n_capable", "arch", "tag", "rung", "layer_id",
              "coord_id", "capable"]
    rename = {"dataset_type": "dataset", "hidden_dim": "width",
              "n_layers": "depth", "feature_name": "feature",
              "feature_role": "role"}

    frames = []
    for criterion, column in (("structural", SCORE_COLUMN),
                              ("homological", "homological_persistence_margin")):
        if column not in df.columns:
            continue
        block = df[[*common, "feature_name", "feature_role", column]].copy()
        block = block.rename(columns={**rename, column: "score"})
        block["criterion"] = criterion
        frames.append(block)

    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Per-model records: the unit every interval is resampled over
# ----------------------------------------------------------------------

def _best_margin(block) -> float:
    """Largest homological persistence margin in a block, NaN if no bar."""
    if "homological_persistence_margin" not in block:
        return float("nan")
    values = block["homological_persistence_margin"].dropna()
    return float(values.max()) if len(values) else float("nan")


def _cell_records(df: pd.DataFrame) -> dict:
    """(dataset, width, depth) -> list of one record per model.

    Built once and reused by every statistic, so a bootstrap draw is a resample
    of this list rather than a re-scan of 109 050 rows. Each record holds
    everything a cell-level number needs from one model:

      ``best``            best neuron's score on the target
      ``null_bests``      one best per null PLACEMENT (never pooled into one:
                          collapsing placements would hand the floor a
                          multiplicity the target never had)
      ``frac``/``frac_capable``   witness fraction on the target
      ``null_frac``       witness fraction pooled over the null placements
      ``count``           number of target witnesses in this model
    """
    out: dict = {}

    for (dataset, width, depth), cell in df.groupby(
        ["dataset_type", "hidden_dim", "n_layers"], sort=True
    ):
        target = cell[cell["feature_role"] == "target"]
        nulls = cell[cell["feature_role"] == "null"]
        assert len(target) and len(nulls), (
            f"{dataset} w{width} d{depth}: нет пары цель/нуль"
        )

        records = []
        for model_path, m in target.groupby("model_path", sort=True):
            n = nulls[nulls["model_path"] == model_path]
            mc, nc = m[m["capable"]], n[n["capable"]]
            records.append({
                "model_path": model_path,
                "best": float(m["score"].min()),
                "null_bests": (n.groupby("feature_name")["score"].min()
                               .to_numpy(dtype=float)),
                "frac": float(m["witness"].mean()),
                "null_frac": float(n["witness"].mean()),
                "frac_capable": float(mc["witness"].mean()) if len(mc) else np.nan,
                "null_frac_capable": float(nc["witness"].mean()) if len(nc) else np.nan,
                "count": int(m["witness"].sum()),
                "null_count": int(n["witness"].sum()),
                # The homological SCORE, not just its verdict: a persistence
                # margin, where larger is better and NaN means "no bar of the
                # right dimension existed", which is most neurons. Best per
                # unit, same convention as the structural side.
                "hom_best": _best_margin(m),
                "hom_null_bests": np.array(
                    [_best_margin(b) for _, b in n.groupby("feature_name")],
                    dtype=float),
                "hom_frac": (float(m["hom_witness"].mean())
                             if "hom_witness" in m else float("nan")),
                "hom_null_frac": (float(n["hom_witness"].mean())
                                  if "hom_witness" in n else float("nan")),
                "hom_count": (int(m["hom_witness"].sum())
                              if "hom_witness" in m else 0),
                "both_count": (int((m["witness"] & m["hom_witness"]).sum())
                               if "hom_witness" in m else 0),
            })

        out[(dataset, int(width), int(depth))] = records

    return out


# ----------------------------------------------------------------------
# Statistics over a list of records
# ----------------------------------------------------------------------

def _nanmedian(values) -> float:
    values = np.asarray(list(values), dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _p_superior(records) -> float:
    from analysis.calibration import superiority_probability

    target = np.array([r["best"] for r in records], dtype=float)
    floor = np.concatenate([r["null_bests"] for r in records])
    return superiority_probability(target, floor)


def _mean_capable(records, key: str) -> float:
    """Mean over models, NaN when the cell has no capable neuron at all.

    Depth 1 and width <= 2 are that case by theorem, and ``np.nanmean`` of an
    all-NaN list is a warning plus a NaN — say NaN directly instead.
    """
    values = np.array([r[key] for r in records], dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


#: name -> statistic mapping a list of per-model records to one number. Every
#: entry is bootstrapped over models with the SAME resample, so differences
#: between them (target minus floor) keep their pairing.
STATISTICS = {
    "witness_fraction": lambda rs: float(np.mean([r["frac"] for r in rs])),
    "witness_fraction_capable": lambda rs: _mean_capable(rs, "frac_capable"),
    "null_fraction": lambda rs: float(np.mean([r["null_frac"] for r in rs])),
    "rate_gap": lambda rs: float(np.mean([r["frac"] - r["null_frac"] for r in rs])),
    "witness_count": lambda rs: float(np.mean([r["count"] for r in rs])),
    "p_has_witness": lambda rs: float(np.mean([r["count"] > 0 for r in rs])),
    "median_best": lambda rs: float(np.median([r["best"] for r in rs])),
    "median_best_null":
        lambda rs: float(np.median(np.concatenate([r["null_bests"] for r in rs]))),
    # Positive = the target is CLOSER to its feature than the floor is to its
    # own, i.e. detection. Negative = the target scores worse than a congruent
    # circle placed somewhere the task never asked about.
    "advantage_over_floor":
        lambda rs: (float(np.median(np.concatenate([r["null_bests"] for r in rs])))
                    - float(np.median([r["best"] for r in rs]))),
    "p_superior": _p_superior,
    # Второй критерий: та же ячейка, тот же пол, другой вопрос.
    "hom_fraction": lambda rs: float(np.mean([r["hom_frac"] for r in rs])),
    "hom_null_fraction": lambda rs: float(np.mean([r["hom_null_frac"] for r in rs])),
    "hom_p_has_witness": lambda rs: float(np.mean([r["hom_count"] > 0 for r in rs])),
    # The margin is the informative homological quantity on this pool: the
    # RATES sit at 0.003 against 0.003 on irrelevant, which is four decimals of
    # nothing, while the best margins are +1.74 against -0.02.
    "hom_median_best": lambda rs: _nanmedian([r["hom_best"] for r in rs]),
    "hom_median_best_null": lambda rs: _nanmedian(
        np.concatenate([r["hom_null_bests"] for r in rs])),
    # Нейроны, которые прошли ОБА критерия — это и есть строка матрицы согласия
    # PLAN §C, и она информативна ровно потому, что критерии независимы.
    "both_fraction": lambda rs: float(np.mean(
        [r["both_count"] / max(r["hom_count"], 1) if r["hom_count"] else 0.0
         for r in rs])),
}


def cell_table(
    df: pd.DataFrame,
    statistics=None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per (dataset, width, depth) with every statistic and its CI.

    Columns come in threes — ``x``, ``x_lo``, ``x_hi`` — the central 90 %
    interval from resampling MODELS. ``n_capable = 0`` (depth 1) leaves the
    capable-only columns NaN rather than pretending to a value.
    """
    from analysis.calibration import bootstrap_over_models

    statistics = statistics or STATISTICS
    records = _cell_records(df)
    rows = []

    for (dataset, width, depth), recs in sorted(records.items()):
        row = {
            "dataset_type": dataset, "width": width, "depth": depth,
            "arch": f"w{width} d{depth}",
            "n_neurons": width * depth,
            # То же правило, что в load_neuron_stats: ширина <= 2 делает
            # неспособным КАЖДЫЙ слой, а не только нулевой.
            "n_capable": (width * (depth - 1)
                          if width >= MIN_WITNESSING_WIDTH else 0),
            "n_models": len(recs),
            "n_null_placements": int(len(recs[0]["null_bests"])),
        }
        for name, statistic in statistics.items():
            estimate, lo, hi = bootstrap_over_models(
                recs, statistic, n_boot=n_boot, seed=seed, alpha=alpha,
            )
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = estimate, lo, hi
        rows.append(row)

    table = pd.DataFrame(rows)
    return table.sort_values(["dataset_type", "depth", "width"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Trends along an architecture axis
# ----------------------------------------------------------------------

def axis_trend(
    df: pd.DataFrame,
    cells: list[tuple[int, int]],
    statistic: str,
    dataset_type: str,
    x: str = "n_neurons",
    log_x: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """OLS slope of one statistic along one architecture axis, with a CI.

    ``cells`` is the ordered list of (width, depth) that makes the axis — the
    width row at fixed depth, or the depth row at fixed width. The interval
    comes from resampling models INDEPENDENTLY WITHIN EACH CELL and refitting,
    which is the only resample that respects the design: cells are separate
    ensembles, not repeated measures of one.

    ``log_x`` because the pool spaces widths geometrically (3, 5, 10, 20) — a
    slope per neuron would be dominated by the largest cell.
    """
    records = _cell_records(df)
    fn = STATISTICS[statistic]

    keys = [(dataset_type, w, d) for w, d in cells]
    missing = [k for k in keys if k not in records]
    assert not missing, f"нет таких ячеек: {missing}"

    xs = np.array([{"n_neurons": w * d, "n_capable": w * (d - 1),
                    "width": w, "depth": d}[x] for w, d in cells], dtype=float)
    assert (xs > 0).all(), f"ось {x} содержит ноль — при глубине 1 нет способных нейронов"
    xv = np.log10(xs) if log_x else xs

    def fit(record_lists):
        ys = np.array([fn(rs) for rs in record_lists], dtype=float)
        ok = np.isfinite(ys)
        if ok.sum() < 2:
            return float("nan")
        return float(np.polyfit(xv[ok], ys[ok], 1)[0])

    per_cell = [records[k] for k in keys]
    estimate = fit(per_cell)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        resampled = [
            [rs[i] for i in rng.integers(0, len(rs), size=len(rs))]
            for rs in per_cell
        ]
        value = fit(resampled)
        if value == value:
            draws.append(value)

    lo, hi = (np.quantile(draws, [alpha, 1 - alpha]) if draws
              else (float("nan"), float("nan")))

    return {
        "dataset_type": dataset_type, "statistic": statistic,
        "axis": x, "log_x": log_x, "cells": list(cells),
        "slope": estimate, "slope_lo": float(lo), "slope_hi": float(hi),
        "includes_zero": bool(lo <= 0 <= hi),
        "values": {f"w{w} d{d}": fn(records[(dataset_type, w, d)])
                   for w, d in cells},
    }


def trend_table(df: pd.DataFrame, axes: list[dict], **kwargs) -> pd.DataFrame:
    """``axis_trend`` over several (dataset, cells, statistic) requests."""
    rows = []
    for request in axes:
        result = axis_trend(df, **{**kwargs, **request})
        rows.append({k: v for k, v in result.items() if k != "values"}
                    | {"cells": " ".join(f"w{w}d{d}" for w, d in result["cells"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The two model comparisons task B is decided on
# ----------------------------------------------------------------------

def floor_scaled_check(
    cells: pd.DataFrame,
    dataset_type: str,
    reference: tuple[int, int] = (5, 5),
    x: str = "n_neurons",
) -> pd.DataFrame:
    """Did the best neuron improve MORE than pure multiplicity would give?

    The best neuron of a model improves with width whether or not anything
    about the representation improved: it is a minimum over more draws. The
    question is how much of the improvement is that.

    The null placements answer it without an imported constant. They live in
    the same cell, are drawn from the same number of neurons, and carry no
    signal by construction — so their own decline with width is the
    multiplicity effect, measured here, on the same statistic and in the same
    units. Scale the reference cell's target by how much the FLOOR moved:

        predicted(cell) = target(ref) x floor(cell) / floor(ref)

    A cell sitting ON that prediction means width bought only draws. A cell
    BELOW it means the neurons themselves got closer to A.

    ``within_ci`` compares the prediction against the measured interval of the
    observed median — the prediction is a point, the measurement has spread.
    """
    block = cells[cells["dataset_type"] == dataset_type].copy()
    ref = block[(block["width"] == reference[0]) & (block["depth"] == reference[1])]
    assert len(ref) == 1, f"референсная ячейка {reference} не найдена"
    ref = ref.iloc[0]

    block["predicted_median_best"] = (
        ref["median_best"] * block["median_best_null"] / ref["median_best_null"]
    )
    block["observed_over_predicted"] = (
        block["median_best"] / block["predicted_median_best"]
    )
    block["within_ci"] = (
        (block["predicted_median_best"] >= block["median_best_lo"])
        & (block["predicted_median_best"] <= block["median_best_hi"])
    )
    keep = ["dataset_type", "arch", "width", "depth", x, "median_best",
            "median_best_lo", "median_best_hi", "median_best_null",
            "predicted_median_best", "observed_over_predicted", "within_ci"]
    return block[keep].sort_values(x).reset_index(drop=True)


def loglog_exponent(
    df: pd.DataFrame,
    cells_axis: list[tuple[int, int]],
    dataset_type: str,
    statistic: str = "median_best",
    x: str = "n_neurons",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Power-law exponent of a statistic against model size, with a CI.

    ``axis_trend`` fits the RAW statistic against ``log10(N)``, which answers
    "how many points does it move per decade". That is the wrong comparison
    between two curves at different levels: the target sits near 0.03 and the
    floor near 0.17, so the same RELATIVE rate shows up as a twice-larger
    absolute slope for whichever curve is higher. Comparing decline rates
    needs both axes in logs, and this is that fit.

    ``local_slopes`` are the segment-by-segment exponents in order. They are
    what says whether a single power law is even the right description: on the
    irrelevant target they steepen (-0.09, -0.20, -0.58) while the floor's
    flatten, and no single exponent shows that.
    """
    records = _cell_records(df)
    fn = STATISTICS[statistic]
    keys = [(dataset_type, w, d) for w, d in cells_axis]
    missing = [k for k in keys if k not in records]
    assert not missing, f"нет таких ячеек: {missing}"

    xs = np.array([{"n_neurons": w * d, "n_capable": w * (d - 1),
                    "width": w, "depth": d}[x] for w, d in cells_axis], dtype=float)
    lx = np.log(xs)

    def fit(record_lists):
        ys = np.array([fn(rs) for rs in record_lists], dtype=float)
        ok = np.isfinite(ys) & (ys > 0)
        if ok.sum() < 2:
            return float("nan")
        return float(np.polyfit(lx[ok], np.log(ys[ok]), 1)[0])

    per_cell = [records[k] for k in keys]
    estimate = fit(per_cell)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        resampled = [[rs[i] for i in rng.integers(0, len(rs), size=len(rs))]
                     for rs in per_cell]
        value = fit(resampled)
        if value == value:
            draws.append(value)
    lo, hi = (np.quantile(draws, [alpha, 1 - alpha]) if draws
              else (float("nan"), float("nan")))

    ys = np.array([fn(rs) for rs in per_cell], dtype=float)
    local = [float((np.log(ys[i + 1]) - np.log(ys[i])) / (lx[i + 1] - lx[i]))
             for i in range(len(ys) - 1)]

    return {"dataset_type": dataset_type, "statistic": statistic, "axis": x,
            "cells": list(cells_axis), "exponent": estimate,
            "exponent_lo": float(lo), "exponent_hi": float(hi),
            "local_slopes": local,
            "values": {f"w{w} d{d}": float(v) for (w, d), v in zip(cells_axis, ys)}}


# ----------------------------------------------------------------------
# Task E: overtraining rungs of one run
# ----------------------------------------------------------------------

def load_overtraining(results_dir, runs_root: str = "models/runs") -> pd.DataFrame:
    """The overtraining rungs, with the RUN each row belongs to.

    A rung is an ordinary ensemble on disk, so ``load_neuron_stats`` already
    reads it. What it cannot know is which rows come from the same training
    run, and that is the whole point of task E: the comparison is within-run,
    because only then is the difference between two rungs a difference in
    duration rather than in initialisation.

    Model ids are NOT that key. ``store_model_instance`` takes the first free id
    inside each rung's own name space, and an upper rung is missing for runs
    whose gate came too late to fit the epoch budget, so one such run shifts
    every later id of that rung. Measured here: at ``w3 d5`` on simple_relevant
    only 2 of 38 indices match between ``x1`` and ``x10``.

    Runs written by the current trainer carry ``run_index`` in their metrics and
    are keyed on it directly. For earlier data the mapping is still exact, not
    approximate: models are stored in run order within a rung, so the gate-epoch
    sequence of any rung is a SUBSEQUENCE of the ``x1`` sequence, and a
    two-pointer walk recovers it. The function asserts that property rather than
    assuming it.

    Adds: ``base`` (the tag without the rung), ``gate_epoch``, ``run_index`` and
    ``run`` (a key unique across datasets and cells).
    """
    import torch

    df = load_neuron_stats(results_dir)
    ot = df[df["rung"] != ""].copy()
    assert len(ot), f"в {results_dir} нет ступеней переобучения"
    ot["base"] = ot["tag"].str.replace(r"_ot[^_]+$", "", regex=True)

    meta = {}
    for path in ot["model_path"].unique():
        m = torch.load(Path(str(path).lstrip("/")) / "metrics.pt", weights_only=False)
        meta[path] = (m.get("gate_epoch"), m.get("run_index"))
    ot["gate_epoch"] = ot["model_path"].map(lambda p: meta[p][0])
    ot["model_index"] = (ot["model_path"].astype(str)
                         .str.rsplit("model", n=1).str[-1].astype(int))

    stored = ot["model_path"].map(lambda p: meta[p][1])
    if stored.notna().all():
        ot["run_index"] = stored.astype(int)
    else:
        ot["run_index"] = _recover_run_index(ot)

    ot["run"] = (ot["dataset_type"] + "|" + ot["base"] + "|"
                 + ot["run_index"].astype(str))
    return ot


def _recover_run_index(ot: pd.DataFrame) -> pd.Series:
    """Run order from the gate-epoch subsequence, for data without ``run_index``."""
    out = pd.Series(index=ot.index, dtype="int64")

    for (dataset, base), block in ot.groupby(["dataset_type", "base"]):
        by_rung = {}
        for rung, part in block.groupby("rung"):
            models = (part[["model_path", "model_index", "gate_epoch"]]
                      .drop_duplicates("model_path").sort_values("model_index"))
            by_rung[rung] = models
        assert "x1" in by_rung, f"{dataset} {base}: нет ступени x1, порядок не восстановить"
        reference = list(by_rung["x1"]["gate_epoch"])

        for rung, models in by_rung.items():
            runs, cursor = [], 0
            for gate in models["gate_epoch"]:
                while cursor < len(reference) and reference[cursor] != gate:
                    cursor += 1
                assert cursor < len(reference), (
                    f"{dataset} {base} {rung}: гейты не ложатся подпоследовательностью "
                    "в x1 — порядок прогонов восстановить нельзя")
                runs.append(cursor)
                cursor += 1
            mapping = dict(zip(models["model_path"], runs))
            mask = (ot["dataset_type"] == dataset) & (ot["base"] == base) & (ot["rung"] == rung)
            out.loc[mask] = ot.loc[mask, "model_path"].map(mapping).to_numpy()

    assert out.notna().all(), "не для всех строк восстановлен номер прогона"
    return out


def paired_rungs(ot: pd.DataFrame, dataset_type: str, base: str,
                 rungs=("x1", "x10", "end"), n_boot: int = 2000) -> pd.DataFrame:
    """Each rung of one cell, on the runs present at ALL of them.

    Restricting to the common runs is what makes the comparison paired: an upper
    rung is missing exactly for the slowest runs, so comparing full rungs would
    compare different populations and call the difference an effect of duration.

    Per run and rung: the best target score, how many of that model's neurons
    witness, and the best homological margin. Returned per rung with intervals
    bootstrapped over RUNS, plus the paired difference against the first rung.
    """
    from analysis.calibration import bootstrap_over_models

    block = ot[(ot["dataset_type"] == dataset_type) & (ot["base"] == base)
               & (ot["feature_role"] == "target") & (ot["rung"].isin(rungs))]
    assert len(block), f"нет строк для {dataset_type} {base}"

    per = (block.groupby(["run", "rung"])
           .agg(best=("score", "min"), witnesses=("witness", "sum"),
                neurons=("witness", "size"),
                hom_witnesses=("hom_witness", "sum"),
                hom_best=("homological_persistence_margin", "max"),
                epoch=("train_stopped_epoch", "first"))
           .reset_index())

    common = set.intersection(*(set(per.loc[per["rung"] == r, "run"]) for r in rungs))
    per = per[per["run"].isin(common)]

    rows = []
    reference = per[per["rung"] == rungs[0]].set_index("run")
    for rung in rungs:
        part = per[per["rung"] == rung].set_index("run").loc[sorted(common)]
        units = [(b, w, n, d, h) for b, w, n, d, h in zip(
            part["best"], part["witnesses"], part["neurons"],
            part["best"] - reference.loc[sorted(common), "best"],
            part["hom_witnesses"])]
        row = {"dataset_type": dataset_type, "base": base, "rung": rung,
               "n_runs": len(common),
               "median_epoch": float(part["epoch"].median()),
               "median_hom_best": float(part["hom_best"].median())}
        for name, stat in (
            ("median_best", lambda u: float(np.median([x[0] for x in u]))),
            ("fraction", lambda u: float(sum(x[1] for x in u) / sum(x[2] for x in u))),
            ("p_has_witness", lambda u: float(np.mean([x[1] > 0 for x in u]))),
            ("paired_delta", lambda u: float(np.median([x[3] for x in u]))),
            ("hom_fraction", lambda u: float(sum(x[4] for x in u) / sum(x[2] for x in u))),
        ):
            est, lo, hi = bootstrap_over_models(units, stat, n_boot=n_boot)
            row |= {name: est, f"{name}_lo": lo, f"{name}_hi": hi}
        rows.append(row)

    return pd.DataFrame(rows)


def saturation_table(ot: pd.DataFrame, n_boot: int = 0) -> pd.DataFrame:
    """Weight norm and saturated-unit share at each rung, from the stored metrics.

    The instrument control for task E. The criterion scans 101 levels spread
    over the field's OBSERVED range, so a score that moves under overtraining
    could in principle be reporting a moved range rather than a moved level set.
    These numbers say how far the range actually travelled, and the ``w24 d1``
    cell -- where a witness is impossible by theorem -- says how much the score
    moves when only the range does.
    """
    import torch

    rows = []
    for path in ot["model_path"].unique():
        m = torch.load(Path(str(path).lstrip("/")) / "metrics.pt", weights_only=False)
        sat = m.get("saturation") or {}
        rows.append({"model_path": path,
                     "weight_norm": sat.get("weight_norm"),
                     "saturated_fraction": sat.get("saturated_fraction"),
                     "logit_abs_mean": sat.get("logit_abs_mean")})

    key = ot[["model_path", "dataset_type", "base", "rung"]].drop_duplicates()
    merged = key.merge(pd.DataFrame(rows), on="model_path")
    return (merged.groupby(["dataset_type", "base", "rung"])
            .agg(weight_norm=("weight_norm", "median"),
                 saturated_fraction=("saturated_fraction", "median"),
                 logit_abs_mean=("logit_abs_mean", "median"),
                 n_models=("model_path", "size"))
            .reset_index())


def duration_table(df: pd.DataFrame, dataset_type: str, cells) -> pd.DataFrame:
    """How long the models of each cell actually trained.

    The gate stops a network when it first holds the target accuracy, and on a
    separable task that happens the sooner the wider the network is. So every
    architecture axis is also a training-duration axis, and any curve along it
    can be read as a duration curve instead. This table is what makes that
    readable rather than invisible: it is the confound, printed.

    The numbers come from ``train_stopped_epoch``, which the evaluation writes
    into every per-neuron row, so this costs nothing beyond a groupby. For an
    overtraining rung that column carries the age of THAT state, not the gate.
    """
    rows = []
    for width, depth in cells:
        block = df[(df["dataset_type"] == dataset_type)
                   & (df["hidden_dim"] == width) & (df["n_layers"] == depth)]
        if not len(block):
            continue
        epochs = block.groupby("model_path")["train_stopped_epoch"].first()
        rows.append({
            "arch": f"w{width} d{depth}", "width": width, "depth": depth,
            "n_models": int(len(epochs)),
            "median_epochs": float(epochs.median()),
            "min_epochs": int(epochs.min()), "max_epochs": int(epochs.max()),
            "q1_epochs": float(epochs.quantile(0.25)),
            "q3_epochs": float(epochs.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def duration_matched(df: pd.DataFrame, dataset_type: str, cells,
                     max_epochs: int, n_boot: int = 2000) -> pd.DataFrame:
    """Each cell twice: all its models, and only those trained no longer than
    ``max_epochs``.

    The point is one specific confound. Along any architecture axis the gate
    epoch moves too, so a cell that looks worse may simply be a cell that
    trained longer. Restricting every cell to a COMMON window of gate epochs
    removes the difference in duration between cells; if the ordering survives
    that, it is not a duration effect.

    This is a matched SUBSAMPLE, not an intervention, and the difference
    matters: selecting models by their gate epoch also selects on how lucky
    their initialisation was, and a slow run is a run whose initialisation made
    the problem hard. So a shift here bounds the confound, and identifies it
    only up to that selection. Task E settles it by varying duration inside one
    run instead.

    Intervals are bootstrapped over models, as everywhere else.
    """
    from analysis.calibration import bootstrap_over_models

    rows = []
    for width, depth in cells:
        block = df[(df["dataset_type"] == dataset_type)
                   & (df["hidden_dim"] == width) & (df["n_layers"] == depth)]
        target = block[block["feature_role"] == "target"]
        if not len(target):
            continue

        per_model = target.groupby("model_path").agg(
            best=("score", "min"), epochs=("train_stopped_epoch", "first"),
            witnesses=("witness", "sum"), neurons=("witness", "size"))
        window = per_model[per_model["epochs"] <= max_epochs]

        # dataset_type / n_neurons / n_capable are carried so the frame can go
        # straight into visualization.arch_sweep_plots.plot_axis_panels.
        row = {"dataset_type": dataset_type, "arch": f"w{width} d{depth}",
               "width": width, "depth": depth,
               "n_neurons": width * depth,
               "n_capable": (width * (depth - 1)
                             if width >= MIN_WITNESSING_WIDTH else 0),
               "n_all": int(len(per_model)), "n_window": int(len(window))}
        for name, part in (("all", per_model), ("window", window)):
            if not len(part):
                row |= {f"median_best_{name}": float("nan"),
                        f"median_best_{name}_lo": float("nan"),
                        f"median_best_{name}_hi": float("nan"),
                        f"fraction_{name}": float("nan")}
                continue
            units = list(part.itertuples())
            est, lo, hi = bootstrap_over_models(
                units, lambda u: float(np.median([x.best for x in u])), n_boot=n_boot)
            row |= {
                f"median_best_{name}": est,
                f"median_best_{name}_lo": lo, f"median_best_{name}_hi": hi,
                f"fraction_{name}": float(part["witnesses"].sum()
                                          / part["neurons"].sum()),
                f"median_epochs_{name}": float(part["epochs"].median()),
            }
        rows.append(row)

    return pd.DataFrame(rows)


def placement_diagnostics(df: pd.DataFrame, dataset_type: str,
                          width: int, depth: int) -> pd.DataFrame:
    """What the field looks like WHERE each feature sits, target and nulls.

    Experiment 1 keeps ``field_grad_norm_on_feature`` as a per-draw
    admissibility check — "is firing possible here at all?" — and states that
    the absolute floor value is a property of the PLACE, not something that
    transfers. This surfaces the place.

    It matters for reading task A: the target scoring worse than the floor is
    only evidence about the network if the floor's places are like the
    target's, and on ``simple_irrelevant`` they are not — ``null_far_right``
    sits where the median gradient is seven times smaller and the field is
    nearly saturated.
    """
    cell = df[(df["dataset_type"] == dataset_type)
              & (df["hidden_dim"] == width) & (df["n_layers"] == depth)]
    assert len(cell), f"нет ячейки {dataset_type} w{width} d{depth}"

    table = cell.groupby("feature_name").agg(
        role=("feature_role", "first"),
        grad_norm=("field_grad_norm_on_feature", "median"),
        value_std=("field_value_std_on_feature", "median"),
        value_abs_mean=("field_value_abs_mean_on_feature", "median"),
        median_score=("score", "median"),
        n_rows=("score", "size"),
    ).reset_index()
    return table.sort_values("median_score").reset_index(drop=True)


def count_fit(
    df: pd.DataFrame,
    dataset_type: str,
    cells: list[tuple[int, int]],
    x: str = "n_neurons",
) -> dict:
    """Constant FRACTION against constant COUNT, fitted on the same points.

    The two readings of "does the feature survive excess capacity" make
    opposite predictions and cannot both be right:

      constant fraction   count = a * N   -> the feature is re-used across the
                                             whole network, every added neuron
                                             carries it with the same chance
      constant count      count = c       -> a fixed group does the work and
                                             the rest of the width idles

    Fitted per model (not per cell) so both models see the same spread, and
    compared on residual sum of squares. ``a`` is the pooled witness fraction
    by construction, which is why the flat-fraction plot and this fit are two
    views of one claim.
    """
    records = _cell_records(df)
    xs, ys = [], []
    for width, depth in cells:
        n = {"n_neurons": width * depth, "n_capable": width * (depth - 1)}[x]
        for r in records[(dataset_type, width, depth)]:
            xs.append(n)
            ys.append(r["count"])
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    a = float(xs @ ys / (xs @ xs))          # least squares through the origin
    c = float(ys.mean())
    rss_proportional = float(((ys - a * xs) ** 2).sum())
    rss_constant = float(((ys - c) ** 2).sum())
    total = float(((ys - ys.mean()) ** 2).sum())

    return {
        "dataset_type": dataset_type, "axis": x,
        "cells": list(cells), "n_models": int(len(ys)),
        "fraction": a, "constant": c,
        "rss_proportional": rss_proportional, "rss_constant": rss_constant,
        "r2_proportional": 1.0 - rss_proportional / total if total else float("nan"),
        "rss_ratio": rss_constant / rss_proportional if rss_proportional else np.inf,
        "winner": "proportional" if rss_proportional < rss_constant else "constant",
    }


def witness_counts(df: pd.DataFrame, dataset_type: str,
                   cells: list[tuple[int, int]]) -> pd.DataFrame:
    """Per-model witness counts, long — the input to the B4 histograms.

    A mean answers "how many on average"; only the distribution answers "is it
    ONE neuron", which is what task B asks at the minimal architecture.
    """
    records = _cell_records(df)
    rows = []
    for width, depth in cells:
        for r in records[(dataset_type, width, depth)]:
            rows.append({
                "dataset_type": dataset_type, "arch": f"w{width} d{depth}",
                "width": width, "depth": depth,
                "n_neurons": width * depth,
            # То же правило, что в load_neuron_stats: ширина <= 2 делает
            # неспособным КАЖДЫЙ слой, а не только нулевой.
            "n_capable": (width * (depth - 1)
                          if width >= MIN_WITNESSING_WIDTH else 0),
                "count": r["count"], "fraction": r["frac"],
                "model_path": r["model_path"],
            })
    return pd.DataFrame(rows)


def witness_rows(df: pd.DataFrame, dataset_type: str, width: int, depth: int,
                 role: str = "target") -> pd.DataFrame:
    """Every neuron that crossed tau in one cell — who fired, and where.

    Used by task A on ``w20 d5``, where four models of fifty produce a witness
    for a circle the task does not need: the count alone cannot say whether
    that is a real closed level set or an artefact, so the rows are needed to
    go and look at the fields.
    """
    cell = df[(df["dataset_type"] == dataset_type)
              & (df["hidden_dim"] == width)
              & (df["n_layers"] == depth)
              & (df["feature_role"] == role)
              & df["witness"]]
    keep = ["model_path", "feature_name", "layer_id", "coord_id", "score",
            "homological_is_witness", "homological_persistence_margin"]
    keep = [c for c in keep if c in cell.columns]
    return cell[keep].sort_values("score").reset_index(drop=True)


# ----------------------------------------------------------------------
# Looking at one neuron: the only way to tell a witness from an artefact
# ----------------------------------------------------------------------

def neuron_field_and_component(
    model_path: str,
    layer_id: int,
    coord_id: int,
    feature_points,
    eval_spec,
    kind: str = "post",
    device: str = "cpu",
) -> dict:
    """One neuron's field on the evaluation grid, plus the component scored.

    ``structural_criterion`` returns a NUMBER; a number cannot say whether the
    thing it measured is a closed curve tracing the feature or a fragment that
    happened to land near it. ``return_best_component_points=True`` hands back
    the actual point set the winning score was computed from, so the picture
    shows what the criterion saw rather than an illustration of it.

    Everything is taken from ``eval_spec`` — the pinned box, the resolution,
    the structural kwargs — so this reproduces the stored row instead of
    recomputing it under a nearby configuration.
    """
    from analysis.calibration import criteria, neuron_function
    from analysis.fields import evaluate_field_on_grid
    from management.experiment_management import load_model_instance

    model, _ = load_model_instance(model_path, device=device)
    function = neuron_function(model, layer_id, coord_id, kind)
    bounds = [tuple(b) for b in eval_spec.bounds]

    field = evaluate_field_on_grid(function, bounds, eval_spec.resolution,
                                   device=device)
    spec = criteria(eval_spec)["structural"]
    result = spec.run(function, feature_points, bounds, eval_spec.resolution,
                      device=device, field=field,
                      return_best_component_points=True)

    from analysis.geometry import feature_diameter

    component = result.get("best_component_points")
    diameter = float(feature_diameter(feature_points, method="exact"))
    loop = component_loop_persistence(component, diameter) if component is not None else {}

    return {
        "model_path": str(model_path), "layer_id": layer_id,
        "coord_id": coord_id, "bounds": bounds,
        "field": field, "feature_points": feature_points,
        "component_points": component,
        "score": float(result["score_normalized_by_diameter"]),
        "level": result.get("best_level"),
        "feature_diameter": diameter,
        "loop": loop,
        "label": f"L{layer_id}n{coord_id}  score {result['score_normalized_by_diameter']:.4f}",
    }


def feature_points_for(dataset_type: str, feature_name: str, eval_spec=None,
                       instance_id: int = 0, dataset_root: str = "data/dataset_instances"):
    """Points of one feature: a stored target, or a null placement rebuilt.

    The target features live in the dataset instance; the A.1 nulls live
    NOWHERE on disk — they are generated at evaluation time from the target by
    ``null_placement.null_features_provider``. Rebuilding them from the same
    ``eval_spec`` is what makes a picture of a null neuron show the set the
    stored score was actually computed against.
    """
    from analysis.stat_aggregation import dataset_instance_path
    from management.experiment_management import (
        load_dataset_instance, get_feature_points,
    )

    loaded = load_dataset_instance(dataset_instance_path(
        dataset_root=dataset_root, dataset_type=dataset_type,
        instance_id=instance_id,
    ))

    if feature_name in loaded.get("features", {}):
        return get_feature_points(loaded, feature_name)

    assert eval_spec is not None, (
        f"{feature_name!r} нет в инстансе; для нулевого размещения нужен eval_spec"
    )
    from experiments.runner import _resolve_null_features_fn

    provider = _resolve_null_features_fn(eval_spec)
    assert provider is not None, "в спеке нет нулевых размещений"
    built = provider(loaded, dataset_type)
    assert feature_name in built, (
        f"{feature_name!r} нет ни среди фич инстанса, ни среди нулей "
        f"{sorted(built)}"
    )
    return built[feature_name][0]


def dataset_points(dataset_type: str, instance_id: int = 0,
                   dataset_root: str = "data/dataset_instances"):
    """All points of the dataset, for drawing faintly under a field."""
    import torch
    from analysis.stat_aggregation import dataset_instance_path
    from management.experiment_management import load_dataset_instance

    loaded = load_dataset_instance(dataset_instance_path(
        dataset_root=dataset_root, dataset_type=dataset_type,
        instance_id=instance_id,
    ))
    if "X" in loaded:
        return loaded["X"]
    return torch.cat([c["points"] for c in loaded["components"].values()])


def component_loop_persistence(points, feature_diameter_value: float | None = None,
                               max_points: int = 900, seed: int = 0) -> dict:
    """Is the matched component a LOOP or an ARC? H_1 of the component itself.

    The structural criterion scores a component by
    ``geometry.max_chamfer_distance``, which is the max over the two directions
    of the MEAN nearest-neighbour distance — a mean, not a supremum. That is
    what makes it tolerant of grid holes, and it is also why an arc hugging
    part of A can pass tau: on ``simple_irrelevant w20 d5`` the component of
    ``model40`` has a mean distance of 0.27 while the farthest point of A is
    0.83 away.

    So "loop or arc" is a separate question from "close to A", and this answers
    it separately: Vietoris-Rips H_1 of the component point cloud. A thickened
    closed curve has a bar; a thickened arc has none. Reported normalised by
    diam(A), so it is on the same scale as the score.

    Deliberately NOT the homological criterion: that one runs on the whole
    field and asks whether the FIELD has a loop anywhere. This asks whether the
    matched SET is one — a diagnostic on the structural verdict, not a third
    criterion, and not something the pipeline's stored columns contain.

    WHAT THE NUMBER LOOKS LIKE, measured on this pool. A component that really
    traces the circle gives ``loop_persistence_normalized`` around **0.78** —
    the hole is the circle, so the bar lives about as long as diam(A) allows.
    A component that closes around a small pinch somewhere else gives about
    **0.12**, six times less. On ``simple_irrelevant w20 d5`` the four neurons
    that crossed tau split exactly along that line: 0.65 and 0.53 against 0.13
    and 0.11. So "does it have a hole" is the wrong question — every one of
    them does — and "is the hole the size of A" is the right one.
    """
    import numpy as np
    from ripser import ripser

    P = np.asarray(points, dtype=float)
    if len(P) == 0:
        return {"n_points": 0, "loop_persistence": 0.0, "loop_birth": float("nan"),
                "loop_death": float("nan"), "loop_persistence_normalized": 0.0,
                "has_hole": False}

    if len(P) > max_points:
        rng = np.random.default_rng(seed)
        P = P[rng.choice(len(P), size=max_points, replace=False)]

    bars = ripser(P, maxdim=1)["dgms"][1]
    if len(bars) == 0:
        best = (float("nan"), float("nan"), 0.0)
    else:
        lengths = bars[:, 1] - bars[:, 0]
        i = int(np.argmax(lengths))
        best = (float(bars[i, 0]), float(bars[i, 1]), float(lengths[i]))

    persistence = best[2]
    scale = feature_diameter_value
    return {
        "n_points": int(len(P)),
        "loop_birth": best[0], "loop_death": best[1],
        "loop_persistence": persistence,
        "loop_persistence_normalized": (persistence / scale if scale else float("nan")),
        # A bar shorter than a couple of grid steps is sampling noise, not a
        # hole; 5 % of diam(A) is two grid steps at N_A = 44. Note this says
        # "there is a hole", NOT "the hole is A" — see the docstring.
        "has_hole": bool(scale and persistence / scale > 0.05),
    }


# ----------------------------------------------------------------------
# Task C: the layer axis
# ----------------------------------------------------------------------
#
# What is C's and what is not. The pool carries ONE target feature per dataset
# and no intervention, which draws the boundary against the neighbouring
# experiments without needing a convention: multitasking (one neuron, several
# features) needs a second target feature and has none here; ablation needs to
# perturb the network and C only observes. So "several neurons carry the
# feature" means, throughout C, the OBSERVED multiplicity — never necessity,
# which no observational statistic can establish.
#
# What is left is the layer index, and the questions only it can answer.

def layer_profile(
    df: pd.DataFrame,
    dataset_type: str,
    cells: list[tuple[int, int]],
    role: str = "target",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Witness fraction per layer, per cell, with a bootstrap CI over models.

    One row per (cell, layer). ``fraction`` is the mean over models of that
    model's witness fraction in that layer, so the unit of resampling stays the
    model — the frozen convention — even though the quantity is indexed by
    layer.

    Three coordinates are attached to every row because C2 needs to draw the
    same profile against each of them:

      ``layer_id``     absolute index, 0 = first hidden layer
      ``from_output``  depth - 1 - layer_id, 0 = last hidden layer
      ``rel_pos``      layer_id / (depth - 1), in [0, 1]; NaN at depth 1

    For ``role="null"`` a model's layer fraction pools its null placements,
    which is the floor's own profile — and the profile HAS structure on the
    nulls too (experiment 1 measured it), so a target profile drawn without it
    is unreadable.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for width, depth in cells:
        cell = df[(df["dataset_type"] == dataset_type)
                  & (df["hidden_dim"] == width) & (df["n_layers"] == depth)
                  & (df["feature_role"] == role)]
        assert len(cell), f"нет ячейки {dataset_type} w{width} d{depth} ({role})"

        # Одна и та же величина в четырёх валютах. Внутри модели и слоя —
        # агрегатор из первого поля, между моделями — из второго: доли
        # усредняются, скоры (экстремальная статистика) берутся по медиане.
        QUANTITIES = {
            "fraction": ("witness", "mean", "mean"),
            "median_score": ("score", "median", "median"),
            "best_score": ("score", "min", "median"),
            "hom_fraction": ("hom_witness", "mean", "mean"),
        }
        grouped = cell.groupby(["model_path", "layer_id"])
        matrices, layers = {}, None
        for name, (column, within, _) in QUANTITIES.items():
            if column not in cell.columns:
                continue
            table = grouped[column].agg(within).unstack("layer_id").sort_index(axis=1)
            matrices[name] = table.to_numpy(dtype=float)
            layers = list(table.columns)

        estimates, intervals = {}, {}
        for name, values in matrices.items():
            across = getattr(np, QUANTITIES[name][2])
            estimates[name] = across(values, axis=0)
            draws = np.empty((n_boot, values.shape[1]), dtype=float)
            for b in range(n_boot):
                idx = rng.integers(0, len(values), size=len(values))
                draws[b] = across(values[idx], axis=0)
            intervals[name] = np.quantile(draws, [alpha, 1 - alpha], axis=0)

        for j, layer in enumerate(layers):
            row = {
                "dataset_type": dataset_type, "role": role,
                "arch": f"w{width} d{depth}", "width": width, "depth": depth,
                "layer_id": int(layer),
                "from_output": int(depth - 1 - layer),
                "rel_pos": (float(layer) / (depth - 1)) if depth > 1 else float("nan"),
                # Позиция с учётом ГОЛОВЫ как последнего слоя: голова стоит в
                # 1.0, первый скрытый слой в 0.0. Только в этой координате все
                # кривые имеют общую область определения и общие концы, а
                # значит их вообще законно сравнивать между собой.
                "rel_pos_head": float(layer) / depth,
                "n_models": int(len(next(iter(matrices.values())))),
            }
            for name in matrices:
                row[name] = float(estimates[name][j])
                row[f"{name}_lo"] = float(intervals[name][0][j])
                row[f"{name}_hi"] = float(intervals[name][1][j])
            rows.append(row)

    return pd.DataFrame(rows)


def alignment_spread(
    profile: pd.DataFrame,
    coordinates=("layer_id", "from_output", "rel_pos"),
    min_depths: int = 3,
    grid: int = 21,
) -> pd.DataFrame:
    """Which coordinate makes the profiles of different depths coincide.

    The question task E was asking, and it cannot be skipped: layer 3 of a
    depth-5 network sits against the output while layer 3 of a depth-8 network
    is in the middle, so until it is known which coordinate the profile is a
    function of, a curve describes only the depth it was measured at.

    For each coordinate, profiles of different depths are compared at matched
    positions and the statistic is the mean over positions of
    ``max - min`` across depths — the disagreement between depths. Smaller is
    better alignment. Integer coordinates use the positions present in at least
    ``min_depths`` curves; ``rel_pos`` is interpolated onto a common grid, and
    the support size is reported so the comparison can be read honestly.
    """
    rows = []
    depths = sorted(profile["depth"].unique())

    for coordinate in coordinates:
        curves = {}
        for depth in depths:
            block = profile[profile["depth"] == depth].sort_values(coordinate)
            x = block[coordinate].to_numpy(dtype=float)
            y = block["fraction"].to_numpy(dtype=float)
            if np.isnan(x).any() or len(x) < 2:
                continue
            curves[depth] = (x, y)
        if len(curves) < min_depths:
            continue

        if coordinate == "rel_pos":
            positions = np.linspace(0.0, 1.0, grid)
            stack = np.array([np.interp(positions, x, y) for x, y in curves.values()])
        else:
            common = sorted(set.intersection(*(set(x.astype(int)) for x, _ in curves.values())))
            counts = {p: sum(p in set(x.astype(int)) for x, _ in curves.values())
                      for p in set().union(*(set(x.astype(int)) for x, _ in curves.values()))}
            common = sorted(p for p, c in counts.items() if c >= min_depths)
            positions = np.array(common, dtype=float)
            stack = np.array([
                [y[list(x).index(p)] if p in list(x) else np.nan for p in positions]
                for x, y in curves.values()])

        with np.errstate(invalid="ignore"):
            spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        rows.append({
            "coordinate": coordinate, "n_depths": len(curves),
            "n_positions": int(len(positions)),
            "mean_spread": float(np.nanmean(spread)),
            "max_spread": float(np.nanmax(spread)),
        })

    return pd.DataFrame(rows).sort_values("mean_spread").reset_index(drop=True)


def profile_monotonicity(df: pd.DataFrame, dataset_type: str,
                         cells: list[tuple[int, int]]) -> pd.DataFrame:
    """Does the witness count ever FALL as you go deeper, within one model?

    A feature that is built up layer by layer gives a monotone profile; one
    that appears at an intermediate layer and is dropped gives a dip. The
    statistic is per model — a dip averaged away across an ensemble would be
    invisible in the mean profile.
    """
    rows = []
    for width, depth in cells:
        cell = df[(df["dataset_type"] == dataset_type)
                  & (df["hidden_dim"] == width) & (df["n_layers"] == depth)
                  & (df["feature_role"] == "target")]
        counts = (cell.groupby(["model_path", "layer_id"])["witness"].sum()
                  .unstack("layer_id").sort_index(axis=1).to_numpy(dtype=float))
        steps = np.diff(counts, axis=1)
        rows.append({
            "arch": f"w{width} d{depth}", "width": width, "depth": depth,
            "n_models": int(len(counts)),
            "monotone_share": float((steps >= 0).all(axis=1).mean()),
            "worst_drop": int(-steps.min()) if steps.size else 0,
        })
    return pd.DataFrame(rows)


def layer_counts(df: pd.DataFrame, dataset_type: str,
                 cells: list[tuple[int, int]], capable_only: bool = True) -> pd.DataFrame:
    """Witnesses per (model, LAYER) — the coherent version of "is it one neuron".

    "One witness in the whole network" is not a coherent hypothesis: if the
    only witness sits at layer l then no deeper layer carries the feature and
    the head, which reads the last layer, does not use it. Per LAYER the
    question is well posed — a neuron can be the only one of its layer and
    still pass the feature on — and this is the table it is answered from.

    ``capable_only`` drops layer 0 and width <= 2, where a closed level set is
    impossible by theorem and a zero is an identity rather than a measurement.
    """
    block = df[(df["dataset_type"] == dataset_type) & (df["feature_role"] == "target")]
    if capable_only:
        block = block[block["capable"]]
    want = {f"w{w} d{d}" for w, d in cells}
    block = block[block["arch"].isin(want)]

    out = (block.groupby(["arch", "hidden_dim", "n_layers", "model_path", "layer_id"])
           ["witness"].agg(["sum", "size"]).reset_index()
           .rename(columns={"sum": "count", "size": "n_neurons",
                            "hidden_dim": "width", "n_layers": "depth"}))
    out["count"] = out["count"].astype(int)
    # Ширина слоя — сколько нейронов в нём вообще есть; для гистограмм это
    # верхняя граница оси и она же подпись панели.
    out["n_capable"] = out["n_neurons"]
    return out


def continuity_filter(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Every witness, tagged with whether the SAME MODEL witnesses deeper.

    The reasoning: the network's output is assembled from the last hidden
    layer. If a feature is picked out at layer l and no deeper layer picks it
    out, the head cannot be reading it there — so such a witness is a candidate
    false positive, on grounds independent of tau.

    ``flagged`` = no deeper witness AND not in the last layer. The second
    conjunct matters: a witness in the last layer trivially has nothing deeper
    and must not be flagged for it.

    THIS IS A NECESSARY CONDITION, NOT A SUFFICIENT ONE, and two limits belong
    with every use of it. A feature can be linearly decodable from the last
    layer without any single neuron's level set tracing it, so an unflagged
    witness is not thereby "used". And where a dataset produces no last-layer
    witnesses at all, the filter degenerates into "does the model witness
    deeper", which is nearly the question it was meant to answer.
    """
    block = df[(df["dataset_type"] == dataset_type) & (df["feature_role"] == "target")]
    witnesses = block[block["witness"]].copy()
    if not len(witnesses):
        return witnesses

    deepest = block[block["witness"]].groupby("model_path")["layer_id"].max()
    witnesses["deepest_witness_layer"] = witnesses["model_path"].map(deepest)
    witnesses["has_deeper_witness"] = (
        witnesses["layer_id"] < witnesses["deepest_witness_layer"])
    witnesses["is_last_layer"] = witnesses["layer_id"] == witnesses["n_layers"] - 1

    witnesses["flagged"] = ~witnesses["has_deeper_witness"] & ~witnesses["is_last_layer"]

    witnesses = witnesses.rename(columns={"hidden_dim": "width", "n_layers": "depth"})
    keep = ["dataset_type", "arch", "width", "depth", "model_path", "layer_id",
            "coord_id", "score", "deepest_witness_layer", "has_deeper_witness",
            "is_last_layer", "flagged"]
    return witnesses[keep].reset_index(drop=True)


def head_scores(
    df: pd.DataFrame,
    dataset_type: str,
    cells: list[tuple[int, int]],
    eval_spec,
    feature_points=None,
    cache: str | Path | None = None,
    device: str = "cpu",
    output: str = "prob",
    verbose: bool = True,
) -> pd.DataFrame:
    """Score the network's OUTPUT the same way its hidden neurons are scored.

    The output layer of these models is a single linear unit, so the network is
    itself a scalar field on R^2 and the criteria apply to it unchanged. That
    makes the head one more position on the layer axis — the position where the
    verdict actually lives, because the classification is read off exactly this
    function.

    WHICH scalar field, though, is a real choice. Hidden neurons are scored at
    ``kind="post"``, i.e. AFTER the squashing nonlinearity; the head's own
    nonlinearity is the sigmoid the loss applies (``mass_training`` predicts
    ``sigmoid(logit) > 0.5``). So ``output="prob"`` — the network as the
    function into (0, 1) that the classifier actually is — is the consistent
    analogue, and it is the default. ``output="logit"`` scores the raw linear
    output instead.

    The two differ less than they look and not in the way that matters most:
    sigmoid is strictly monotone, so the level SETS of the two fields coincide
    exactly, and any difference is a reparametrisation of WHICH levels the scan
    visits. The scan takes 101 levels linearly spaced over the observed range,
    so under ``"prob"`` those levels are uniform in probability — dense near
    the decision boundary, and on a scale that means the same thing in every
    model, where the logit range varies with how confident training left the
    network.

    Two things depend on it. The layer profile becomes a curve with both ends
    pinned: it starts at 0 on the first hidden layer, where a closed level set
    is impossible by theorem, and ends at the head. And the continuity filter
    stops having to exempt the last hidden layer by construction: "is the
    feature still there deeper" becomes well defined all the way through.

    Whether the head witnesses is a MEASUREMENT here, not an assumption — on
    ``simple_relevant`` the decision boundary is the circle and one expects it
    to, on ``simple_irrelevant`` it is not and one expects it not to, and both
    predictions are worth checking rather than asserting.

    Costs a couple of seconds per model, so the result is cached to ``cache``.
    """
    import torch

    from analysis.calibration import criteria
    from management.experiment_management import load_model_instance

    assert output in {"prob", "logit"}, "output must be 'prob' or 'logit'"

    cache = Path(cache) if cache else None
    if cache is not None and cache.exists():
        stored = pd.read_csv(cache)
        if "output" not in stored:            # cache written before the choice existed
            stored["output"] = "logit"
        stored = stored[stored["output"] == output]
        want = {f"w{w} d{d}" for w, d in cells}
        if want <= set(stored.loc[stored["dataset_type"] == dataset_type, "arch"]):
            return stored[(stored["dataset_type"] == dataset_type)
                          & stored["arch"].isin(want)].reset_index(drop=True)

    if feature_points is None:
        target = df.loc[(df["dataset_type"] == dataset_type)
                        & (df["feature_role"] == "target"), "feature_name"].iat[0]
        feature_points = feature_points_for(dataset_type, target)

    bounds = [tuple(b) for b in eval_spec.bounds]
    spec = criteria(eval_spec)["structural"]
    rows = []

    for width, depth in cells:
        paths = sorted(df.loc[(df["dataset_type"] == dataset_type)
                              & (df["hidden_dim"] == width)
                              & (df["n_layers"] == depth), "model_path"].unique())
        for i, path in enumerate(paths):
            model, _ = load_model_instance(path, device=device)
            if output == "prob":
                def field(x, model=model):
                    return torch.sigmoid(model(x)).squeeze(-1)
            else:
                def field(x, model=model):
                    return model(x).squeeze(-1)
            result = spec.run(field, feature_points,
                              bounds, eval_spec.resolution, device=device)
            score = float(result["score_normalized_by_diameter"])
            rows.append({
                "dataset_type": dataset_type, "arch": f"w{width} d{depth}",
                "width": width, "depth": depth, "model_path": str(path),
                # The head sits one position past the last hidden layer.
                "layer_id": depth, "output": output,
                "score": score, "witness": score <= TAU,
            })
        if verbose:
            block = [r for r in rows if r["arch"] == f"w{width} d{depth}"]
            print(f"  {dataset_type} w{width} d{depth}: {len(block)} моделей, "
                  f"доля свидетелей у головы "
                  f"{np.mean([r['witness'] for r in block]):.3f}", flush=True)

    out = pd.DataFrame(rows)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            previous = pd.read_csv(cache)
            if "output" not in previous:
                previous["output"] = "logit"
            keep = ~((previous["dataset_type"] == dataset_type)
                     & previous["arch"].isin(out["arch"])
                     & (previous["output"] == output))
            out = pd.concat([previous[keep], out], ignore_index=True)
        out.to_csv(cache, index=False)
    return out
