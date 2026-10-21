"""Detector calibration — the executable half of ``CALIBRATION.md``.

That document argues that a criterion's output is not a result until three
questions are answered, in order: is the instrument connected (probes with a
KNOWN answer), is the number stable (a region of discretisation parameters
inside which it does not move), and how does a number become a verdict (a
decision rule that is derived rather than typed in). This module supplies the
machinery for all three, and ``notebooks/experiments/exp1_sanity_check.ipynb``
drives it on the matched pair ``simple_irrelevant`` / ``simple_relevant``
(write-up: ``reports/exp1_sanity_check.md``). The earlier driver,
``exp1_calibration.ipynb`` on ``halfplane_base``, is retired to
``notebooks/legacy/``.

The organising idea is the **probe**: a scalar field whose correct verdict is
known before the criterion runs. Four roles, following ``CALIBRATION.md`` 2.5:

    ceiling   a planted radial ridge sitting exactly on A          -> witness
    partial   the same ridge, degraded by a measured amount        -> graded
    working   a real trained neuron                                -> unknown
    floor     a real neuron scored against an A.1 null feature     -> non-witness

"Degraded by a measured amount" is what makes the middle rung usable. The
degradations here are rigid geometric ones — displace the ridge's centre,
mis-size its radius, hide part of A — so the intended level set stays a curve
we can write down. Its normalised Chamfer distance to A is therefore computable
independently of the criterion, and that number is carried alongside every
measurement as ``ideal_score``. The gap between measured and ideal is the
instrument's own overhead; the value of the degradation at which the measured
curve crosses tau is the tolerance the threshold ACTUALLY buys, as opposed to
the tolerance it was advertised to buy.

One probe is not a degradation but an invariance. ``carrier_gain`` rescales the
final linear row feeding the measured neuron: post-activation is then
``tanh(G * pre)`` instead of ``tanh(pre)``, a strictly monotone reparameterisation
that maps every level set to itself. The correct answer is that no criterion
moves at all. Anything that does move is an artefact of how levels are placed
and thickened, which is exactly what part II of CALIBRATION.md is about — and it
answers 1.4's complaint that the planted control is only ever read in the linear
regime while real neurons over the target are saturated.
"""

import copy
import itertools
import math
import time
from dataclasses import dataclass, field as _dc_field

import torch


# ----------------------------------------------------------------------
# Step 1 (CALIBRATION.md 2.1, 2.5): conventions, frozen without computing
# ----------------------------------------------------------------------
#
# A convention changes WHAT the number means, not how precisely it is
# computed, so it is chosen once and never swept: sweeping it produces a
# family of different measurements rather than a stability region for one.
# Recorded here so that a scan can assert it did not drift.

CONVENTIONS = {
    "diameter_method": {
        "value": "exact",
        "read_from": "structural_criterion / extremum_anchored_criterion",
        "why": (
            "the score is normalised by diam(A); 'bbox' and 'exact' differ by "
            "sqrt(2) for a circle, so tau means two different tolerances "
            "depending on which is used"
        ),
    },
    "kind": {
        "value": "post",
        "read_from": "EvalSpec.kinds",
        "why": (
            "post-activation is the neuron as the next layer sees it; pre- and "
            "post- have the same level sets only while the activation is "
            "strictly monotone, which is true for tanh but not a property to "
            "rely on silently"
        ),
    },
    "homology_backend": {
        "value": "delaunay",
        "read_from": "EvalSpec.extra_criteria_kwargs['homological']['backend']",
        "why": (
            "a radius graph is correct on a full regular grid only; on any "
            "subsampled or random cloud it carries bars coming from holes of "
            "the SAMPLE (docs/RUNBOOK.md)"
        ),
    },
    "homology_orientations": {
        "value": "both (f and -f)",
        "read_from": "homological_criterion (fixed in code)",
        "why": (
            "a circle can appear as a ridge or as a basin rim; checking one "
            "orientation makes the criterion sensitive to a sign that carries "
            "no meaning"
        ),
    },
    "level_grid": {
        "value": "observed range",
        "read_from": "EvalSpec.structural_kwargs['activation'] (None = observed)",
        "why": (
            "the level sweep must sample the neuron's OWN family of level sets: "
            "with linspace(min f, max f, n) the whole criterion is equivariant "
            "under f -> lambda*f (levels, adaptive eps and its clamp all scale, "
            "the slab mask does not move), so identical geometry gets an "
            "identical score. The activation-anchored grid linspace(-1, 1, n) "
            "is not: a neuron using half the tanh range gets half the usable "
            "levels, and observed range varies systematically with layer, which "
            "would put a depth-correlated bias into exp2's own question"
        ),
    },
    "bootstrap_unit": {
        "value": "model",
        "read_from": "analysis.witness_stats / calibration.bootstrap_over_models",
        "why": (
            "neurons inside one model are not independent draws — they share "
            "weights, data and seed; resampling neurons produces confidence "
            "intervals for a population that does not exist"
        ),
    },
}


def conventions_table():
    """The frozen conventions as a DataFrame (Step 1 of CALIBRATION.md VI)."""
    import pandas as pd

    return pd.DataFrame(
        [{"convention": k, **v} for k, v in CONVENTIONS.items()]
    )[["convention", "value", "read_from", "why"]]


def check_conventions(eval_spec):
    """Compare a live ``EvalSpec`` against the frozen conventions.

    Returns a DataFrame with one row per checkable convention and an ``agrees``
    column. A mismatch is not automatically an error — it means the spec is
    measuring something other than what this calibration calibrated, and the
    calibration does not transfer to it.
    """
    import pandas as pd

    homological_kwargs = dict(
        eval_spec.extra_criteria_kwargs.get("homological", {})
    )
    structural_kwargs = dict(eval_spec.structural_kwargs)

    activation_kwarg = structural_kwargs.get("activation", "__absent__")
    if activation_kwarg == "__absent__":
        # evaluate_neurons_on_feature will inject the model's activation.
        level_grid = "activation range (auto-injected)"
    elif activation_kwarg is None:
        level_grid = "observed range"
    else:
        level_grid = f"activation range ({activation_kwarg})"

    observed = {
        "level_grid": level_grid,
        "diameter_method": structural_kwargs.get("diameter_method", "exact"),
        "kind": ",".join(eval_spec.kinds),
        "homology_backend": homological_kwargs.get("backend", "ripser"),
        "homology_orientations": "both (f and -f)",
        "bootstrap_unit": "model",
    }

    rows = []
    for name, spec in CONVENTIONS.items():
        rows.append(
            {
                "convention": name,
                "frozen": spec["value"],
                "spec": observed.get(name, "(not in EvalSpec)"),
                "agrees": observed.get(name) == spec["value"],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The criteria, wrapped uniformly
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionSpec:
    """One criterion behind a uniform (score, verdict) interface.

    ``score_key`` names the continuous quantity a scan tracks and
    ``lower_is_better`` says which direction is "closer to A" — the homological
    criterion reports a persistence MARGIN, where larger is better, and
    averaging it with a Chamfer distance would be meaningless.
    """

    name: str
    fn: object
    score_key: str
    witness_key: str
    lower_is_better: bool
    kwargs: dict = _dc_field(default_factory=dict)

    def run(self, function, feature_points, bounds, resolution, device="cpu",
            **overrides) -> dict:
        kwargs = {**self.kwargs, **overrides}
        return self.fn(
            function=function,
            feature_points=feature_points,
            bounds=bounds,
            resolution=resolution,
            device=device,
            **kwargs,
        )


def criteria(eval_spec=None) -> dict:
    """The three live criteria, configured exactly as the pipeline runs them.

    Passing an ``EvalSpec`` binds its kwargs, so a calibration is of the same
    instrument the experiments use rather than of a look-alike assembled here.
    """
    from analysis.criterions import structural_criterion
    from analysis.extremum_anchored import extremum_anchored_criterion
    from analysis.homological import homological_criterion

    structural_kwargs = {"level_eps": "adaptive", "threshold": 0.1,
                         "activation": None}
    extremum_kwargs = {}
    homological_kwargs = {"backend": CONVENTIONS["homology_backend"]["value"]}

    if eval_spec is not None:
        structural_kwargs = dict(eval_spec.structural_kwargs)
        extra = eval_spec.extra_criteria_kwargs
        extremum_kwargs = dict(extra.get("extremum", {}))
        homological_kwargs = dict(extra.get("homological", homological_kwargs))

    return {
        "structural": CriterionSpec(
            name="structural",
            fn=structural_criterion,
            score_key="score_normalized_by_diameter",
            witness_key="is_structural_witness",
            lower_is_better=True,
            kwargs=structural_kwargs,
        ),
        "extremum": CriterionSpec(
            name="extremum",
            fn=extremum_anchored_criterion,
            score_key="score_normalized_by_diameter",
            witness_key="is_witness",
            lower_is_better=True,
            kwargs=extremum_kwargs,
        ),
        "homological": CriterionSpec(
            name="homological",
            fn=homological_criterion,
            score_key="persistence_margin",
            witness_key="is_witness",
            lower_is_better=False,
            kwargs=homological_kwargs,
        ),
    }


# ----------------------------------------------------------------------
# Step 2 (CALIBRATION.md 1.4, 2.5): probes with a known answer
# ----------------------------------------------------------------------


@dataclass
class Probe:
    """A scalar field plus the answer we already know about it.

    ``ideal_score`` is the normalised max-symmetrised Chamfer distance between
    the level set the construction INTENDS and the feature it is scored
    against, computed from the two point sets directly with the project's own
    ``max_chamfer_distance``. It is a property of the geometry, not of any
    criterion, so ``measured - ideal`` isolates what the instrument adds.
    ``None`` where no such curve exists (a trained neuron).
    """

    name: str
    role: str            # ceiling | partial | working | floor
    function: object
    feature_points: torch.Tensor
    expected: str        # witness | graded | non-witness | unknown
    ideal_score: float | None = None
    meta: dict = _dc_field(default_factory=dict)


def _chamfer(X: torch.Tensor, Y: torch.Tensor) -> float:
    from analysis.geometry import max_chamfer_distance

    return float(max_chamfer_distance(X, Y)["score"])


def _diam(P: torch.Tensor) -> float:
    """diam(A) under the frozen convention — never the cheap bbox default."""
    from analysis.geometry import feature_diameter

    return float(
        feature_diameter(P, method=CONVENTIONS["diameter_method"]["value"])
    )


def normalized_chamfer(K: torch.Tensor, A: torch.Tensor) -> float:
    """The quantity every geometric criterion reports, computed on two clouds."""
    return _chamfer(K, A) / _diam(A)


def circle_points(center, radius: float, n: int = 800) -> torch.Tensor:
    """A dense circle — the curve a planted radial ridge intends to be."""
    center = torch.as_tensor(center, dtype=torch.float32).reshape(2)
    theta = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    return center + radius * torch.stack([theta.cos(), theta.sin()], dim=1)


def feature_circle(feature_points: torch.Tensor) -> tuple[torch.Tensor, float]:
    """(centre, radius) of a roughly circular feature, as the planting sees it."""
    center = feature_points.mean(dim=0)
    radius = float((feature_points - center).norm(dim=1).mean())
    return center, radius


def arc_feature(feature_points: torch.Tensor, fraction: float,
                start_angle: float = 0.0) -> torch.Tensor:
    """The sub-arc of a circular feature covering ``fraction`` of its angle.

    Degrading the FEATURE rather than the field is how partial coverage gets a
    known answer: the planted ridge still traces the whole circle, so the
    level set now over-covers A. That is the second direction of the
    max-symmetrisation — the one that punishes "the component wanders off A" —
    and it is the only one the centre/radius degradations do not exercise.
    """
    assert 0.0 < fraction <= 1.0, "fraction must be in (0, 1]"
    if fraction == 1.0:
        return feature_points

    center, _ = feature_circle(feature_points)
    delta = feature_points - center
    angle = torch.atan2(delta[:, 1], delta[:, 0])
    angle = (angle - start_angle) % (2.0 * math.pi)
    keep = angle <= 2.0 * math.pi * fraction
    kept = feature_points[keep]
    assert len(kept) >= 3, "arc kept fewer than 3 points — raise the fraction"
    return kept


def neuron_function(model, layer_id: int, coord_id: int, kind: str | None = None):
    from models.NN_analysis import TruncatedMLPFunction, ScalarProjection

    kind = kind or CONVENTIONS["kind"]["value"]
    return ScalarProjection(
        TruncatedMLPFunction(model, layer_id, kind=kind), coord_id=coord_id
    )


def planted_probe(
    model,
    feature_points: torch.Tensor,
    *,
    layer_id: int = 3,
    coord_id: int = 0,
    center_offset=(0.0, 0.0),
    radius_scale: float = 1.0,
    coverage: float = 1.0,
    carrier_gain: float = 1.0,
    sharpness: float = 2.0,
    name: str | None = None,
) -> Probe:
    """A planted radial ridge, optionally degraded by a measured amount.

    The undegraded probe is the top of the scale: ``plant_radial_neuron``
    overwrites every weight on the path feeding (``layer_id``, ``coord_id``)
    so that the neuron's maximum is attained exactly on the circle through A.
    Because the criteria read only that neuron, the verdict is a deterministic
    function of (A, bounds, field source and resolution, criterion config) and
    of nothing else — in particular it does not depend on the trained weights,
    which is why measuring it across several trained models adds no information.

    Degradations, each with a computable intended curve:

    ``center_offset``  displace the ridge, in DATA units (see
                       ``offset_fractions_to_vectors`` for fractions of diam A)
    ``radius_scale``   plant a ridge of radius ``radius_scale * r_A``
    ``coverage``       keep only this fraction of A; the ridge is unchanged and
                       therefore over-covers what remains
    ``carrier_gain``   rescale the final linear row into the neuron: a strictly
                       monotone reparameterisation, so every level set is
                       preserved and the correct answer is "nothing changes"

    The planted model is a copy; the model passed in is untouched. It is a
    detector calibration object, not a classifier — planting drops accuracy on
    the task (1.0 -> 0.165 on halfplane_base), which is why this probe controls
    the INSTRUMENT and not the claim (CALIBRATION.md 1.3).
    """
    from analysis.controls import plant_radial_neuron

    center, radius = feature_circle(feature_points)
    offset = torch.as_tensor(center_offset, dtype=torch.float32).reshape(2)
    planted_center = center + offset
    planted_radius = radius * radius_scale

    planted = copy.deepcopy(model)
    plant_radial_neuron(
        planted,
        layer_id=layer_id,
        coord_id=coord_id,
        center=planted_center,
        radius=planted_radius,
        sharpness=sharpness,
        kind="ridge",
    )

    if carrier_gain != 1.0:
        # tanh(G * pre) instead of tanh(pre): strictly monotone in the
        # pre-activation, so {x : f(x) = c} is merely relabelled, never moved.
        with torch.no_grad():
            row = planted.blocks[layer_id].linear
            row.weight[coord_id] *= carrier_gain
            row.bias[coord_id] *= carrier_gain

    scored_against = arc_feature(feature_points, coverage)
    intended = circle_points(planted_center, planted_radius)

    degraded = (
        float(offset.norm()) > 0.0 or radius_scale != 1.0 or coverage != 1.0
    )
    if name is None:
        name = "planted" if not degraded else (
            f"planted[d={float(offset.norm()):.3f},"
            f"g={radius_scale:.2f},cov={coverage:.2f}]"
        )

    return Probe(
        name=name,
        role="partial" if degraded else "ceiling",
        function=neuron_function(planted, layer_id, coord_id),
        feature_points=scored_against,
        expected="graded" if degraded else "witness",
        ideal_score=normalized_chamfer(intended, scored_against),
        meta={
            "layer_id": layer_id,
            "coord_id": coord_id,
            "center_offset_norm": float(offset.norm()),
            "center_offset_frac_diam": float(offset.norm()) / _diam(feature_points),
            "radius_scale": radius_scale,
            "coverage": coverage,
            "carrier_gain": carrier_gain,
            "model": planted,
        },
    )


def offset_vector(feature_points: torch.Tensor, fraction: float,
                  direction=(0.0, 1.0)) -> torch.Tensor:
    """A displacement of ``fraction`` x diam(A), for a readable curve axis.

    The default direction is +y: on ``halfplane_base`` displacing along +x walks
    the ridge towards the decision boundary, where the trained field has strong
    structure of its own, so the geometric degradation would be read together
    with whatever that structure contributes.
    """
    d = torch.as_tensor(direction, dtype=torch.float32).reshape(2)
    d = d / d.norm()
    return d * (fraction * _diam(feature_points))


def trained_probe(model, layer_id: int, coord_id: int,
                  feature_points: torch.Tensor, role: str = "working",
                  name: str | None = None) -> Probe:
    """A real neuron of a trained network — the working point, answer unknown.

    ``role="floor"`` when ``feature_points`` is an A.1 null: same neuron, a
    congruent copy of A placed elsewhere, so the expected answer IS known.
    """
    return Probe(
        name=name or f"L{layer_id}n{coord_id}",
        role=role,
        function=neuron_function(model, layer_id, coord_id),
        feature_points=feature_points,
        expected="non-witness" if role == "floor" else "unknown",
        ideal_score=None,
        meta={"layer_id": layer_id, "coord_id": coord_id, "model": model},
    )


# ----------------------------------------------------------------------
# Step 3 (CALIBRATION.md 2.2, 2.3, 2.5): scans and stability regions
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """A named bundle of kwargs that must move together.

    The parameters that matter most are not independent: "adaptive epsilon at
    quantile 0.95" is two keywords, and sweeping either alone measures
    something else. A ``Setting`` is one level of such a compound parameter —
    the scan records ``label`` and passes ``kwargs``.
    """

    label: str
    kwargs: dict = _dc_field(default_factory=dict)


def _expand(overrides: dict) -> tuple[dict, dict]:
    """(what the row records, what the criterion is called with)."""
    record, call = {}, {}
    for key, value in overrides.items():
        if isinstance(value, Setting):
            record[key] = value.label
            call.update(value.kwargs)
        else:
            record[key] = value
            call[key] = value
    return record, call


def measure(probe: Probe, criterion: CriterionSpec, bounds,
            resolution: int, device: str = "cpu", **overrides) -> dict:
    """Run one criterion on one probe and flatten the answer to a row.

    A criterion that raises is recorded as NaN with its message rather than
    aborting the scan: a parameter value that breaks a backend is a fact about
    the stability region, and losing the rest of the sweep to it hides that.
    """
    record, overrides = _expand(overrides)
    resolution = int(overrides.pop("resolution", resolution))
    record.pop("resolution", None)

    row = {
        "probe": probe.name,
        "role": probe.role,
        "expected": probe.expected,
        "criterion": criterion.name,
        "resolution": resolution,
        "ideal_score": probe.ideal_score,
    }
    row.update(record)

    start = time.perf_counter()
    try:
        result = criterion.run(
            probe.function, probe.feature_points, bounds, resolution,
            device=device, **overrides,
        )
        score = result.get(criterion.score_key, float("nan"))
        row["score"] = float(score) if score is not None else float("nan")
        row["is_witness"] = bool(result.get(criterion.witness_key, False))
        row["error"] = ""
        # Carry the criterion's own diagnostics through: the boundary-anchor
        # fraction, the winning level, the number of matched bars. A scan that
        # kept only the score could not tell a criterion that answered from one
        # that had nothing to answer with.
        for key, value in result.items():
            if key not in row and isinstance(value, (int, float, bool)):
                row[key] = value
    except Exception as exc:                       # noqa: BLE001 — recorded, see above
        row["score"] = float("nan")
        row["is_witness"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"

    row["seconds"] = time.perf_counter() - start
    if probe.ideal_score is not None and criterion.lower_is_better:
        row["overhead"] = row["score"] - probe.ideal_score
    else:
        row["overhead"] = float("nan")
    return row


def scan(probes, criteria_specs, bounds, resolution: int,
         grid: dict | None = None, device: str = "cpu", verbose: bool = True):
    """Every (probe x criterion x grid point), as a long DataFrame.

    ``grid`` maps a criterion keyword to the values it should take; a single
    key sweeps one parameter (CALIBRATION.md 2.5 step 1), several keys take the
    full Cartesian product (step 2 — the factorial needed for coupled pairs,
    where the effect of one parameter depends on the level of the other and no
    one-at-a-time sweep can see it).
    """
    import pandas as pd

    if isinstance(criteria_specs, CriterionSpec):
        criteria_specs = {criteria_specs.name: criteria_specs}
    grid = grid or {}
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) or [()]

    rows = []
    total = len(probes) * len(criteria_specs) * len(combos)
    done = 0
    for probe in probes:
        for criterion in criteria_specs.values():
            for combo in combos:
                overrides = dict(zip(keys, combo))
                # resolution is both a positional argument and a sweepable
                # parameter; take it out of the kwargs so a grid over it does
                # not collide with the default.
                call = dict(overrides)
                res = call.pop("resolution", resolution)
                rows.append(
                    measure(probe, criterion, bounds, res,
                            device=device, **call)
                )
                done += 1
                if verbose and done % 10 == 0:
                    print(f"  {done}/{total}", end="\r")
    if verbose:
        print(f"  {total}/{total} evaluations, "
              f"{sum(r['seconds'] for r in rows):.1f}s total")
    return pd.DataFrame(rows)


def _plateau(values, scores, reference_index: int, tol: float):
    """Widest contiguous window around the reference within ``tol``.

    The window has to CONTAIN the reference: a stability region is a
    guarantee about the point you intend to operate at, and the widest flat
    stretch elsewhere in the sweep says nothing about it.
    """
    n = len(values)
    lo = hi = reference_index
    while True:
        grew = False
        for cand_lo, cand_hi in ((lo - 1, hi), (lo, hi + 1)):
            if cand_lo < 0 or cand_hi >= n:
                continue
            window = [s for s in scores[cand_lo:cand_hi + 1] if s == s]
            if window and max(window) - min(window) <= tol:
                lo, hi = cand_lo, cand_hi
                grew = True
        if not grew:
            return lo, hi


def stability_table(scan_df, param: str, tol: float = 0.01,
                    reference=None, group=("probe", "role", "criterion")):
    """"parameter -> stability region -> delta", the deliverable of part II.

    ``tol`` is the width you are willing to call "the same number". The default
    0.01 is a tenth of tau = 0.1: a threshold decision that could be flipped by
    a tenth of its own distance from the score is not a decision about the
    network. The region is reported, not assumed — a parameter whose plateau is
    a single value has no stability region, and that is the finding.
    """
    import pandas as pd

    rows = []
    for keys, sub in scan_df.groupby(list(group), sort=False):
        assert sub[param].duplicated().sum() == 0, (
            f"more than one row per {param!r} value in group {keys!r} — this is "
            "a factorial scan, and collapsing it here would report the "
            "stability of an arbitrary slice; use interaction_table, or "
            "restrict the frame to one level of the other parameter first"
        )
        order = {v: i for i, v in enumerate(dict.fromkeys(sub[param]))}
        sub = sub.sort_values(param, key=lambda s: s.map(order))
        values = list(sub[param])
        scores = list(sub["score"])
        verdicts = list(sub["is_witness"])

        ref = values[-1] if reference is None else reference
        ref_index = values.index(ref) if ref in values else len(values) - 1
        lo, hi = _plateau(values, scores, ref_index, tol)

        finite = [s for s in scores if s == s]
        window = [s for s in scores[lo:hi + 1] if s == s]

        rows.append(
            {
                **dict(zip(group, keys if isinstance(keys, tuple) else (keys,))),
                "param": param,
                "n_values": len(values),
                "reference": ref,
                "reference_score": scores[ref_index],
                "plateau_from": values[lo],
                "plateau_to": values[hi],
                "plateau_n": hi - lo + 1,
                "delta_plateau": (max(window) - min(window)) if window else float("nan"),
                "delta_full_range": (max(finite) - min(finite)) if finite else float("nan"),
                "verdict_constant_plateau": len(set(verdicts[lo:hi + 1])) == 1,
                "verdict_constant_full": len(set(verdicts)) == 1,
            }
        )
    return pd.DataFrame(rows)


def interaction_table(scan_df, param: str, by: str,
                      group=("probe", "criterion")):
    """Does the effect of ``param`` depend on the level of ``by``?

    CALIBRATION.md 2.2: one-at-a-time sweeps are adequate for monotone
    parameters and blind to coupled pairs. The coupling shows up here as
    ``interaction`` — the spread of "range of the score across ``param``"
    across the levels of ``by``. Zero means the two can be tuned separately;
    anything comparable to the ranges themselves means they cannot.
    """
    import pandas as pd

    rows = []
    for keys, sub in scan_df.groupby(list(group), sort=False):
        ranges = {}
        for level, block in sub.groupby(by, sort=False):
            finite = [s for s in block["score"] if s == s]
            ranges[level] = (max(finite) - min(finite)) if finite else float("nan")
        finite_ranges = [r for r in ranges.values() if r == r]
        rows.append(
            {
                **dict(zip(group, keys if isinstance(keys, tuple) else (keys,))),
                "param": param,
                "by": by,
                **{f"range@{level}": r for level, r in ranges.items()},
                "interaction": (max(finite_ranges) - min(finite_ranges))
                if finite_ranges else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def contrast_stability(scan_df, param: str, target_role: str = "working",
                       null_role: str = "floor", criterion: str | None = None):
    """Is the target-minus-null contrast steadier than either score alone?

    Part IV of CALIBRATION.md claims it should be: both are computed on the
    same grid with the same epsilon, so part of the discretisation error is
    common mode and cancels in the difference — which is the second reason a
    single calibration can serve many experiments. Cancellation needs the
    parameter to act on target and null alike, which the adaptive epsilon does
    not: it is derived from LOCAL field variation, and that differs between any
    two placements. This function tests the claim rather than assuming it —
    measured on halfplane_base the contrast is steadier than the target but not
    than the null, so the cancellation is partial.

    The null series is the per-parameter-value MINIMUM over null probes: the
    model-level statistic is best-neuron-versus-best-null (CALIBRATION.md 3.3),
    not an average over placements.
    """
    import pandas as pd

    df = scan_df if criterion is None else scan_df[scan_df["criterion"] == criterion]
    target = df[df["role"] == target_role].groupby(param)["score"].min()
    null = df[df["role"] == null_role].groupby(param)["score"].min()
    common = [v for v in target.index if v in null.index]

    series = {
        "target": [float(target[v]) for v in common],
        "null": [float(null[v]) for v in common],
    }
    series["contrast"] = [t - n for t, n in zip(series["target"], series["null"])]

    rows = []
    for name, values in series.items():
        finite = [v for v in values if v == v]
        rows.append(
            {
                "series": name,
                "param": param,
                "n_values": len(values),
                "min": min(finite) if finite else float("nan"),
                "max": max(finite) if finite else float("nan"),
                "delta_full_range": (max(finite) - min(finite))
                if finite else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The detection curve (CALIBRATION.md 1.4): what happens between the rungs
# ----------------------------------------------------------------------

DEGRADATION_AXES = {
    # axis name -> (planted_probe keyword, human-readable unit)
    "center_offset_frac": ("center_offset", "fraction of diam(A)"),
    "radius_scale": ("radius_scale", "planted radius / r_A"),
    "coverage": ("coverage", "fraction of A that remains"),
    "carrier_gain": ("carrier_gain", "monotone rescaling of the neuron"),
}


def detection_curve(model, feature_points, bounds, criteria_specs, axis: str,
                    values, resolution: int, layer_id: int = 3,
                    coord_id: int = 0, device: str = "cpu",
                    direction=(0.0, 1.0), verbose: bool = True):
    """Score versus a measured amount of degradation, with the ideal alongside.

    This is the object CALIBRATION.md 1.4 says is missing: two points were
    known — a perfect planted ridge at 0.0342 and the best trained neuron at
    0.165 — with nothing calibrated between them, so "how partial can an
    encoding be and still cross tau" had no answer. Sweeping a degradation whose
    geometry is written down supplies the whole curve, and ``crossing`` reads
    the answer off it.
    """
    assert axis in DEGRADATION_AXES, f"unknown axis {axis!r}"
    keyword, _unit = DEGRADATION_AXES[axis]

    probes = []
    for value in values:
        if keyword == "center_offset":
            kwargs = {keyword: offset_vector(feature_points, value, direction)}
        else:
            kwargs = {keyword: value}
        probe = planted_probe(
            model, feature_points, layer_id=layer_id, coord_id=coord_id,
            name=f"{axis}={value:g}", **kwargs,
        )
        probe.meta[axis] = value
        probes.append(probe)

    df = scan(probes, criteria_specs, bounds, resolution,
              device=device, verbose=verbose)
    df[axis] = df["probe"].map({p.name: p.meta[axis] for p in probes})
    return df


def crossing(df, threshold: float, x: str, y: str = "score"):
    """Where a monotone curve first crosses ``threshold``, linearly interpolated.

    Returns ``None`` when the curve never crosses inside the swept range — an
    honest answer that a numeric extrapolation would hide.
    """
    points = sorted(
        ((float(a), float(b)) for a, b in zip(df[x], df[y]) if b == b),
        key=lambda p: p[0],
    )
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if (y0 - threshold) * (y1 - threshold) <= 0 and y0 != y1:
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return None


# ----------------------------------------------------------------------
# Step 5 (CALIBRATION.md 3.3, 3.4): from a number to a verdict
# ----------------------------------------------------------------------


def null_percentile_threshold(null_scores, false_positive_rate: float = 0.05,
                              lower_is_better: bool = True) -> float:
    """Approach A: tau is read off the nulls, not typed in.

    The false-positive rate is then fixed by construction instead of inherited
    from a hand-picked 0.1, and the comparison stays meaningful even where no
    score comes near 0.1 — which is the regime the halfplane dataset is
    actually in.

    Note the tail. CALIBRATION.md 3.3 says "the 95th percentile of the null
    scores", which is right for an index where larger means more evidence. The
    structural and extremum scores are DISTANCES: smaller means more evidence,
    so a 5% false-positive rate is the 5th percentile, and taking the 95th
    would declare almost every neuron a witness. ``lower_is_better`` picks the
    tail so the rule cannot be applied to the wrong direction by accident;
    the homological margin needs ``lower_is_better=False``.

    What the number is: the nulls here are deliberately placed across different
    field regimes, so their distribution is a MIXTURE and this is its
    percentile (CALIBRATION.md V) — a legitimate quantity, but one to name
    accurately rather than to call "the null distribution".
    """
    import numpy as np

    values = np.asarray([s for s in null_scores if s == s], dtype=float)
    assert len(values) > 0, "no finite null scores"
    q = false_positive_rate if lower_is_better else 1.0 - false_positive_rate
    return float(np.quantile(values, q))


def superiority_probability(target_scores, null_scores) -> float:
    """P(target score < null score) over all pairs; ties count a half.

    The common-language effect size (equivalently the AUC): 0.5 means the
    target is indistinguishable from a congruent copy of itself placed
    elsewhere, 1.0 that it is systematically closer. Non-parametric on purpose
    — the score is a minimum over ~200 (level, epsilon) combinations, i.e. an
    extreme-value statistic with a skewed distribution, and a mean-based test
    on it is not valid (CALIBRATION.md V).
    """
    import numpy as np

    a = np.asarray([s for s in target_scores if s == s], dtype=float)
    b = np.asarray([s for s in null_scores if s == s], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((diff < 0).mean() + 0.5 * (diff == 0).mean())


def bootstrap_over_models(units, statistic, n_boot: int = 2000,
                          seed: int = 0, alpha: float = 0.05):
    """Resample MODELS with replacement — never neurons (a frozen convention).

    ``units`` is one entry per model (any object the statistic understands);
    ``statistic`` maps a list of units to a number. Returns
    (estimate, lo, hi) for the central (1 - 2*alpha) interval, so the default
    is the 90% interval that a two-one-sided-test at 5% consumes.
    """
    import numpy as np

    units = list(units)
    assert units, "no units to resample"
    rng = np.random.default_rng(seed)
    estimate = statistic(units)

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), size=len(units))
        value = statistic([units[i] for i in idx])
        if value == value:
            draws.append(value)

    if not draws:
        return estimate, float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(estimate), float(lo), float(hi)


def equivalence(estimate: float, lo: float, hi: float, delta: float,
                null_value: float = 0.5) -> dict:
    """Approach B's second half: did we SHOW there is no effect?

    An ordinary test cannot confirm a null hypothesis — "not rejected" is not
    "shown absent" — and experiment 1 predicts absence, so the claim needs a
    two-one-sided-test. Equivalence holds when the whole confidence interval
    lies inside the pre-declared indifference band ``null_value +- delta``.
    """
    return {
        "estimate": estimate,
        "ci_lo": lo,
        "ci_hi": hi,
        "null_value": null_value,
        "delta": delta,
        "band_lo": null_value - delta,
        "band_hi": null_value + delta,
        "equivalent": bool(lo >= null_value - delta and hi <= null_value + delta),
        "distinguishable": bool(lo > null_value or hi < null_value),
    }


def best_per_model(rows, score_col: str, model_col: str = "model_path",
                   role_col: str = "role"):
    """Collapse 35 neurons to one number per (model, role).

    CALIBRATION.md 3.1 counts the cost of not doing this: calling a model a
    witness carrier when ANY of its neurons crosses a 5%-correct threshold
    gives a 0.83 chance of a false positive at 35 neurons and 0.96 at 63, so a
    per-neuron count grows with width by construction — and the slope of
    exactly that dependence is what experiment 2 sets out to measure. Comparing
    the model's best neuron against the model's best NULL neuron collapses the
    35 comparisons into one and removes the multiplicity instead of correcting
    for it.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    return (
        df.groupby([model_col, role_col])[score_col]
        .min()
        .reset_index()
        .rename(columns={score_col: "best_score"})
    )


def rank_neurons(model, feature_points, bounds, resolution: int,
                 criterion: CriterionSpec, layer_ids=None, coord_ids=None,
                 device: str = "cpu", verbose: bool = False):
    """Every neuron of a trained model scored against one feature, ranked.

    Used to pick the working-point and floor probes: the model-level statistic
    is the BEST neuron (CALIBRATION.md 3.3), so the neuron a stability scan
    should follow is the one that carries the verdict at the reference
    configuration.
    """
    import pandas as pd
    from models.NN_analysis import get_hidden_dim

    if layer_ids is None:
        layer_ids = list(range(model.n_layers))

    rows = []
    for layer_id in layer_ids:
        hidden = get_hidden_dim(model, layer_id)
        for coord_id in (coord_ids if coord_ids is not None else range(hidden)):
            probe = trained_probe(model, layer_id, coord_id, feature_points)
            row = measure(probe, criterion, bounds, resolution, device=device)
            row["layer_id"] = layer_id
            row["coord_id"] = coord_id
            rows.append(row)
            if verbose:
                print(f"  L{layer_id}n{coord_id}: {row['score']:.4f}")
    return pd.DataFrame(rows).sort_values(
        "score", ascending=criterion.lower_is_better
    ).reset_index(drop=True)


def ensemble_scores(model_paths, features: dict, bounds, resolution: int,
                    criterion: CriterionSpec, target_name: str,
                    layer_ids=None, device: str = "cpu", verbose: bool = True):
    """Score every neuron of every stored model against target and nulls.

    The long table the decision rules of part III consume: one row per
    (model, feature, neuron). ``features`` maps feature name to points and must
    contain ``target_name``; every other entry is treated as an A.1 null, so
    the ``role`` column comes out as target/null without a second convention.

    Deliberately not ``collect_compact_neuron_statistics``: that writes a CSV
    per experiment under the spec's own configuration, whereas a calibration
    needs the same quantity recomputed under configurations the spec does not
    have. The numbers agree where the configurations do.
    """
    import pandas as pd
    from management.experiment_management import load_model_instance

    rows = []
    for path in model_paths:
        model, _checkpoint = load_model_instance(path, device=device)
        for name, points in features.items():
            ranked = rank_neurons(model, points, bounds, resolution,
                                  criterion, layer_ids=layer_ids, device=device)
            ranked["model_path"] = str(path)
            ranked["feature"] = name
            ranked["role"] = "target" if name == target_name else "null"
            rows.append(ranked)
        if verbose:
            print(f"  {path}: done")
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------
# Step 6 (CALIBRATION.md II): reading a stored sweep back
# ----------------------------------------------------------------------

#: Which direction is "closer to A" for each criterion, by name. The
#: homological criterion reports a persistence MARGIN, so its best neuron is
#: the largest and not the smallest; getting this backwards silently reports
#: the worst neuron of every model.
DIRECTIONS = {"structural": True, "extremum": True, "homological": False}


def read_scores_csv(path):
    """Read a stored score table back with ``role='null'`` intact.

    pandas parses a bare ``null`` cell as NaN, so a table written by
    ``ensemble_scores`` loses exactly the rows the floor is made of unless the
    column is read verbatim — the same trap ``runner.aggregate`` and
    ``witness_stats._load_stats_source`` already guard against.
    """
    import pandas as pd

    return pd.read_csv(
        path, converters={"role": lambda v: v, "feature_role": lambda v: v}
    )


def best_per_unit(scores, lower_is_better: bool,
                  unit=("model_path", "feature"), score_col: str = "score",
                  role_col: str = "role"):
    """One number per (model, feature): that model's best neuron for it.

    The unit is the PAIR, not the model. The target is one feature and the
    floor is several placements, so collapsing the floor to a single
    best-of-all-nulls would hand it a multiplicity the target never had; per
    placement both sides are a best-of-W*D, and the spread across placements is
    then the width of the floor (`AUDIT/08` 5.1) rather than noise to average
    away. NaN scores are dropped, not treated as bad values — for the
    homological criterion a NaN means "no bar of the right dimension was
    matched at all", which is a detection failure and is counted separately by
    ``sweep_summary``.
    """
    import pandas as pd

    df = pd.DataFrame(scores)
    keys = [c for c in (*unit, role_col) if c in df.columns]
    finite = df[df[score_col].notna()]
    agg = "min" if lower_is_better else "max"
    out = (
        finite.groupby(keys, sort=False, dropna=False)[score_col]
        .agg(agg)
        .reset_index()
        .rename(columns={score_col: "best_score"})
    )
    return out


def _quantile(q):
    def f(s):
        return s.quantile(q)
    f.__name__ = f"q{int(q * 100):02d}"
    return f


def sweep_summary(scores, directions=None, by=("criterion", "resolution", "dataset"),
                  unit=("model_path", "feature")):
    """Per (cell, role): how many neurons fired at all, and how good the best was.

    Two populations live in one table on purpose. ``neuron_*`` describes every
    neuron — the picture a histogram shows — while ``best_*`` describes the
    per-unit best, which is the statistic a verdict is allowed to use
    (CALIBRATION.md 3.1: a per-neuron count inherits a multiplicity that grows
    with width by construction). ``detected_*`` is the censoring: for the
    homological criterion most neurons return no matched bar, and a median over
    the survivors alone would read as a score when it is a score conditional on
    detection.
    """
    import pandas as pd

    directions = dict(DIRECTIONS if directions is None else directions)
    df = pd.DataFrame(scores)
    by = [c for c in by if c in df.columns]
    keys = by + ["role"]

    frames = []
    for name, block in df.groupby("criterion", sort=False):
        lower = directions.get(name, True)
        neuron = (
            block.groupby(keys, sort=False, dropna=False)["score"]
            .agg(n_neurons="size",
                 detected_neurons=lambda s: float(s.notna().mean()),
                 neuron_median="median",
                 neuron_q10=_quantile(0.10),
                 neuron_q90=_quantile(0.90))
            .reset_index()
        )
        units = df.loc[block.index].groupby(
            [*unit, *keys], sort=False, dropna=False)["score"]
        n_units = (units.size().reset_index(name="n")
                   .groupby(keys, sort=False, dropna=False).size()
                   .reset_index(name="n_units"))
        detected = (units.apply(lambda s: bool(s.notna().any()))
                    .reset_index(name="hit")
                    .groupby(keys, sort=False, dropna=False)["hit"].mean()
                    .reset_index(name="detected_units"))
        best = (
            best_per_unit(block, lower_is_better=lower, unit=(*unit, *by))
            .groupby(keys, sort=False, dropna=False)["best_score"]
            .agg(best_median="median",
                 best_q10=_quantile(0.10),
                 best_q90=_quantile(0.90),
                 best_extreme="min" if lower else "max")
            .reset_index()
        )
        out = neuron.merge(n_units, on=keys).merge(detected, on=keys).merge(
            best, on=keys, how="left")
        out["lower_is_better"] = lower
        frames.append(out)

    return pd.concat(frames, ignore_index=True).sort_values(keys).reset_index(drop=True)


def sweep_contrast(scores, directions=None,
                   by=("criterion", "resolution", "dataset"),
                   unit=("model_path", "feature"), threshold=None):
    """Per cell: how far the target sits above its own floor.

    ``gap`` is signed so that positive always means "target better", whichever
    direction the criterion runs in, and ``p_superior`` is the same statement
    without a scale — the probability that a randomly picked target unit beats
    a randomly picked null unit (0.5 = indistinguishable). ``null_softest`` is
    the single placement whose median comes closest to the target: the floor is
    a set of placements and a verdict has to clear the highest of them, not
    their average.
    """
    import numpy as np
    import pandas as pd

    directions = dict(DIRECTIONS if directions is None else directions)
    df = pd.DataFrame(scores)
    by = [c for c in by if c in df.columns]

    rows = []
    for keys, block in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        name = dict(zip(by, keys))["criterion"]
        lower = directions.get(name, True)
        sign = 1.0 if lower else -1.0

        best = best_per_unit(block, lower_is_better=lower, unit=unit)
        tgt = best.loc[best["role"] == "target", "best_score"].to_numpy(float)
        nul = best.loc[best["role"] == "null", "best_score"].to_numpy(float)
        if len(tgt) == 0 or len(nul) == 0:
            continue

        per_place = (best[best["role"] == "null"]
                     .groupby("feature")["best_score"].median())
        softest = per_place.idxmin() if lower else per_place.idxmax()

        row = dict(zip(by, keys))
        row.update(
            target_median=float(np.median(tgt)),
            null_median=float(np.median(nul)),
            null_softest=str(softest),
            null_softest_median=float(per_place.loc[softest]),
            gap=float(sign * (np.median(nul) - np.median(tgt))),
            gap_vs_softest=float(sign * (per_place.loc[softest] - np.median(tgt))),
            p_superior=superiority_probability(sign * tgt, sign * nul),
        )
        if threshold is not None:
            crossed = (tgt <= threshold) if lower else (tgt >= threshold)
            crossed_null = (nul <= threshold) if lower else (nul >= threshold)
            row["target_rate"] = float(crossed.mean())
            row["null_rate"] = float(crossed_null.mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def selectivity(scores, lower_is_better: bool = True, by=(),
                datasets=("relevant", "irrelevant"), threshold=None,
                unit=("model_path", "feature")):
    """The E comparison: one dataset's target against the other's.

    The two targets are the SAME feature — bitwise the same points in the same
    place — carried by networks trained on different tasks, so this is the only
    comparison in the calibration where nothing but task relevance differs. It
    is unpaired by construction (different models), hence a rank statistic:
    ``p_superior`` is P(a unit of the first dataset beats a unit of the second),
    and ``margin`` is the signed distance between the two populations' facing
    ends — positive means they do not overlap at all.
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(scores)
    by = [c for c in by if c in df.columns]
    sign = 1.0 if lower_is_better else -1.0
    a_name, b_name = datasets

    rows = []
    for keys, block in (df.groupby(by, sort=False) if by else [((), df)]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        # "dataset" must survive best_per_unit's projection — it is what the
        # two sides are told apart by, and it is not always one of ``by``.
        unit_keys = [*unit, *by]
        if "dataset" not in unit_keys:
            unit_keys.append("dataset")
        best = best_per_unit(block, lower_is_better=lower_is_better,
                             unit=tuple(unit_keys))
        best = best[best["role"] == "target"]
        a = best.loc[best["dataset"] == a_name, "best_score"].to_numpy(float)
        b = best.loc[best["dataset"] == b_name, "best_score"].to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            continue

        row = dict(zip(by, keys))
        row.update({
            f"median_{a_name}": float(np.median(a)),
            f"median_{b_name}": float(np.median(b)),
            "p_superior": superiority_probability(sign * a, sign * b),
            # worst of a against best of b: > 0 means the populations are
            # disjoint, and by how much
            "margin": float(sign * (b.min() - a.max()) if lower_is_better
                            else (b.max() - a.min())),
            f"n_{a_name}": int(len(a)),
            f"n_{b_name}": int(len(b)),
        })
        if threshold is not None:
            crossed = (lambda v: v <= threshold) if lower_is_better else (lambda v: v >= threshold)
            row[f"rate_{a_name}"] = float(crossed(a).mean())
            row[f"rate_{b_name}"] = float(crossed(b).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True) if by \
        else pd.DataFrame(rows)
