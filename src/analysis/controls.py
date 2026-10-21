"""Cross-cutting validation controls: null baseline and positive control.

PLAN.md §C: no geometric criterion by itself removes the incidental geometric
effect. The decisive quantitative tool is the NULL BASELINE (A.1, negative
control): compare the witness rate on the real feature against random features
of the same diameter/type and report rate[A] - rate[null]. Its pair is the
POSITIVE CONTROL (A.6): a setting where a single neuron provably MUST witness
(a planted radial neuron), where every criterion is required to fire — without
both calibrations, "no witness found" is indistinguishable from "detector
broken/insensitive".

Null features are attached to dataset instances by the experiment runner at
persist time (they are per-experiment controls, not part of the geometric
recipe); ``feature_role`` metadata already distinguishes target/diagnostic
features, extended here with role "null".
"""

import math

import torch

from analysis.geometry import feature_diameter, max_chamfer_distance


# -------------------------
# A.1: null baseline
# -------------------------

def _random_orthonormal_2frame(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Two random orthonormal directions in R^dim, rows [2, dim]."""
    M = torch.randn(dim, 2, generator=generator)
    Q, _ = torch.linalg.qr(M)
    return Q.T


def _random_rotation(dim: int, generator: torch.Generator) -> torch.Tensor:
    M = torch.randn(dim, dim, generator=generator)
    Q, R = torch.linalg.qr(M)
    # Fix signs so the distribution is uniform (Haar) and det = +1.
    Q = Q * torch.sign(torch.diagonal(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class NullPlacementError(RuntimeError):
    """No admissible placement exists for a diameter-matched null feature."""


def make_null_features(
    reference_feature_points: torch.Tensor,
    bounds,
    n_features: int = 5,
    kind: str = "circle",
    seed: int = 0,
    eval_margin: float = 0.5,
    min_separation_from_target: float = 0.3,
    min_separation_between_nulls: float = 0.1,
    max_attempts_per_feature: int = 20000,
) -> dict:
    """
    Random task-irrelevant features matched to a reference feature (A.1).

      - kind="circle": circles with the same diameter and point count as the
        reference; in dim > 2 the circle lies in a random 2-plane;
      - kind="shuffled": the reference shape, randomly rotated about its
        centroid and translated;
      - matched DIAMETER and SAMPLE SIZE are non-negotiable — the null must
        differ from A only in placement, otherwise the rate difference
        confounds size with relevance;
      - deterministic in ``seed`` (nulls are persisted with the dataset).

    PLACEMENT (rewritten 2026-08-16 — see AUDIT/01_math.md C1 and
    AUDIT/04_repair_log.md R3). The previous law drew the centre uniformly from
    the data bounding box shrunk by the null RADIUS. Because every target in
    this project is nearly as large as its own data, that region collapsed:
    for `halfplane_base` it degenerated to a segment (all five nulls shared
    y == 0, in the empty corridor between the classes), and for dataset (c)
    three of five nulls landed within the witness threshold OF THE TARGET, so a
    perfect witness of A was automatically also a witness of its own control.

    Two changes:

    1. the admissible centre region is computed against the EVALUATION bounds
       (``bounds`` expanded by ``eval_margin``), not the tight data box — the
       null only has to be scoreable, and the field is evaluated on the
       expanded box;
    2. placements are rejection-sampled against a separation contract:
       ``max_chamfer(A, N)/diam(A) > min_separation_from_target`` (default 0.3
       = 3x the witness threshold, so a null cannot inherit the target's
       verdict) and the same quantity between accepted nulls with
       ``min_separation_between_nulls`` (default 0.1 = the witness threshold,
       so the controls are not near-duplicates of each other).

    If no placement satisfies the contract, this raises ``NullPlacementError``
    rather than returning a degenerate control. That is deliberate: a
    diameter-matched null is geometrically infeasible when the feature is large
    relative to its domain, and silently returning a coincident null is exactly
    the failure this repair exists to remove.

    Returns dict {null_name: Tensor [n, dim]}, names "null_{kind}_{i}".
    """
    assert kind in {"circle", "shuffled"}, "kind must be 'circle' or 'shuffled'"
    assert n_features >= 1
    assert min_separation_from_target >= 0
    assert min_separation_between_nulls >= 0

    ref = reference_feature_points.detach().cpu()
    if not torch.is_floating_point(ref):
        ref = ref.float()

    assert ref.ndim == 2 and len(ref) > 0
    dim = ref.shape[1]
    assert len(bounds) == dim, "bounds must match the feature dimension"

    n_points = len(ref)
    diameter = float(feature_diameter(ref, method="exact").item())
    radius = diameter / 2.0
    centroid = ref.mean(dim=0)

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Admissible centres: the null must fit inside the EVALUATION box. The
    # containment radius is the CIRCUMRADIUS about the centroid, not diam/2 —
    # they differ whenever the centroid is not the circumcentre (a sampled
    # S^2 of radius 2 has diam/2 = 2.0 but circumradius 2.029, and shrinking by
    # the smaller number let nulls poke outside the box).
    if kind == "circle":
        circumradius = radius
    else:
        circumradius = float((ref - centroid).norm(dim=1).max().item())

    lows = torch.tensor([b[0] for b in bounds], dtype=torch.float32) - eval_margin
    highs = torch.tensor([b[1] for b in bounds], dtype=torch.float32) + eval_margin

    center_lows = torch.minimum(lows + circumradius, (lows + highs) / 2)
    center_highs = torch.maximum(highs - circumradius, (lows + highs) / 2)

    def _draw_candidate():
        u = torch.rand(dim, generator=generator)
        center = center_lows + (center_highs - center_lows) * u

        if kind == "circle":
            angles = torch.linspace(0, 2 * math.pi, n_points + 1)[:-1]
            unit = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

            if dim == 2:
                return radius * unit + center

            frame = _random_orthonormal_2frame(dim, generator)  # [2, dim]
            return radius * (unit @ frame) + center

        R = _random_rotation(dim, generator)
        return (ref - centroid) @ R.T + center

    def _sep(P, Q) -> float:
        return float(max_chamfer_distance(P, Q)["score"].item()) / max(diameter, 1e-12)

    nulls: dict[str, torch.Tensor] = {}
    best_seen = 0.0

    for i in range(n_features):
        accepted = None

        for _ in range(max_attempts_per_feature):
            candidate = _draw_candidate()

            sep_target = _sep(ref, candidate)
            best_seen = max(best_seen, sep_target)

            if sep_target <= min_separation_from_target:
                continue

            if any(_sep(candidate, P) <= min_separation_between_nulls
                   for P in nulls.values()):
                continue

            accepted = candidate
            break

        if accepted is None:
            span = [float(h - l) for l, h in zip(center_lows, center_highs)]
            raise NullPlacementError(
                f"could not place null {len(nulls)} of {n_features} after "
                f"{max_attempts_per_feature} attempts.\n"
                f"  feature diameter      : {diameter:.4f}\n"
                f"  admissible centre box : {[round(s, 4) for s in span]}\n"
                f"  best separation seen  : {best_seen:.4f} "
                f"(need > {min_separation_from_target})\n"
                "A diameter-matched null cannot be placed far enough from this "
                "feature inside this domain — the feature is too large relative "
                "to its data. This is a property of the DATASET, not a bug: it "
                "means rate[A] - rate[null] cannot be formed here with a "
                "shape-matched control. Options: enlarge the domain, shrink the "
                "feature, use a control matched on something other than "
                "diameter, or lower min_separation_from_target deliberately and "
                "record that the control is weak."
            )

        nulls[f"null_{kind}_{i}"] = accepted

    return nulls


def witness_rate_gap(
    neuron_table,
    feature_name: str,
    null_feature_names: list[str],
    witness_column: str = "is_structural_witness",
    group_by: tuple[str, ...] = ("dataset_type", "model_path"),
) -> dict:
    """
    The decisive quantity rate[A] - rate[null] (PLAN.md §C, exp. 1-2).

    Per group (default per model), witness rate on the real feature minus the
    rate pooled over all null features; the summary aggregates the per-group
    gaps with a bootstrap CI (same machinery as analysis.witness_stats).
    Works for ANY witness column (structural / extremum / homological) —
    pass the column name.

    Returns dict:
        per_model: DataFrame [group..., rate_feature, rate_null, gap,
                   n_feature_rows, n_null_rows]
        summary: {mean_gap, ci_low, ci_high, n_groups,
                  mean_rate_feature, mean_rate_null}
    """
    import numpy as np
    import pandas as pd

    from analysis.witness_stats import (
        _load_stats_source,
        _coerce_bool_series,
        _bootstrap_ci_mean,
    )

    df = _load_stats_source(neuron_table)

    assert witness_column in df.columns, (
        f"Column {witness_column!r} not in the table; "
        f"available: {list(df.columns)}"
    )

    relevant = df[df["feature_name"].isin([feature_name] + list(null_feature_names))]
    assert len(relevant) > 0, "No rows for the requested features."

    relevant = relevant.copy()
    relevant["_witness"] = _coerce_bool_series(relevant[witness_column])

    rows = []

    for group_key, g in relevant.groupby(list(group_by), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        feature_rows = g[g["feature_name"] == feature_name]
        null_rows = g[g["feature_name"].isin(null_feature_names)]

        if len(feature_rows) == 0 or len(null_rows) == 0:
            continue

        rate_feature = float(feature_rows["_witness"].mean())
        rate_null = float(null_rows["_witness"].mean())

        row = dict(zip(group_by, group_key))
        row.update(
            {
                "rate_feature": rate_feature,
                "rate_null": rate_null,
                "gap": rate_feature - rate_null,
                "n_feature_rows": int(len(feature_rows)),
                "n_null_rows": int(len(null_rows)),
            }
        )
        rows.append(row)

    per_model = pd.DataFrame(rows)
    assert len(per_model) > 0, (
        "No group has both feature and null rows — were null features "
        "attached at persist time (DatasetSpec.n_null_features > 0)?"
    )

    gaps = per_model["gap"].to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_ci_mean(gaps)

    summary = {
        "mean_gap": float(np.mean(gaps)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_groups": int(len(per_model)),
        "mean_rate_feature": float(per_model["rate_feature"].mean()),
        "mean_rate_null": float(per_model["rate_null"].mean()),
    }

    return {"per_model": per_model, "summary": summary}


# -------------------------
# A.6: positive control
# -------------------------

def plant_radial_neuron(
    model,
    layer_id: int,
    coord_id: int,
    center: torch.Tensor,
    radius: float,
    sharpness: float = 2.0,
    kind: str = "ridge",
):
    """
    A.6 positive control: overwrites the weights along one path of the
    network so that neuron (layer_id, coord_id, post) is a provable witness
    for the circle of radius ``radius`` around ``center``.

    Shared base (2D input; tanh cannot produce ||x - c||^2 in one layer):
      - layer 0: k directional bump pairs (k = 3 when hidden_dim >= 6,
        else 2), directions u_i evenly spaced by pi/k; per direction
            tanh(s (a_i + delta)) - tanh(s (a_i - delta)),
        an even bump in a_i = u_i . (x - c) of half-width delta = radius,
        slope s = sharpness / delta;
      - layer 1, the radial CAP: z = g * sum of the k bumps with
        g = 1 / (2 k tanh(sharpness)), so z_max = 1 and the downstream tanh's
        stay in their linear regime (saturating them flattens the profile and
        washes the level sets out). For k = 3 evenly spaced directions both
        the quadratic AND quartic terms of the sum are exactly isotropic
        (the cos(2m theta) harmonics cancel for m = 1, 2), so the cap's level
        sets are circles up to 6th-order corrections.

    kind="ridge" (default; needs layer_id >= 3) — the calibration object
    proper, "a radial bump with its RIDGE at ``radius``":
      - layer 2: a shifted pair in the cap value v1(x) (radial, monotone
        in ||x - c||): tanh(sigma (v1 - v_r +- eps)) with v_r = v1 on the
        target circle (measured by a forward probe) — their difference is a
        1D bump in v1 centered at v_r;
      - layer 3: the difference, normalized to max 1 — a field whose MAXIMUM
        is attained exactly on the circle ||x - c|| = radius. The crest value
        is a critical value, so the extremum-anchored criterion must fire on
        it by construction (a basin hides the circle at a non-critical level
        and is the wrong calibration object for criterion 3).

    kind="basin" (needs layer_id >= 1) — the cap itself: a local maximum over
    the interior whose level sets are circles around the center; the ground
    truth "basin" object for experiment 6 typing.

    Layers above the construction pass the signal through a monotone carrier
    tanh(1.5 * previous) — monotone maps preserve level sets exactly, so the
    witness property survives arbitrary planting depth.

    Only the weights feeding this path are touched (layer-0 rows, 1-2 rows
    per layer above); everything else is left intact, though downstream
    layers consuming the overwritten neurons will of course see different
    inputs. This model is a detector calibration object, not a trained
    classifier.

    Returns the modified model (in place).
    """
    assert kind in {"ridge", "basin"}, "kind must be 'ridge' or 'basin'"

    min_layer = 3 if kind == "ridge" else 1
    assert layer_id >= min_layer, (
        f"plant_radial_neuron(kind={kind!r}) needs layer_id >= {min_layer} "
        "(see docstring construction)"
    )
    assert layer_id < model.n_layers
    assert model.in_dim == 2, "the planting construction is 2D-input only"
    assert model.hidden_dim >= 4, "need >= 4 first-layer neurons for the bumps"
    assert 0 <= coord_id < model.hidden_dim
    assert radius > 0 and sharpness > 0

    center = torch.as_tensor(center, dtype=torch.float32).reshape(2)
    delta = float(radius)
    s = sharpness / delta

    k = 3 if model.hidden_dim >= 6 else 2
    g = 1.0 / (2.0 * k * math.tanh(sharpness))

    with torch.no_grad():
        block0 = model.blocks[0]

        for i in range(k):  # direction u_i, evenly spaced by pi/k
            theta = math.pi * i / k
            u = torch.tensor([math.cos(theta), math.sin(theta)])
            offset = float(u @ center)

            for sign_idx, offset_sign in enumerate((+1.0, -1.0)):
                neuron = 2 * i + sign_idx
                block0.linear.weight[neuron] = s * u
                block0.linear.bias[neuron] = s * (offset_sign * delta - offset)

        # Layer 1: the radial cap (into coord_id if a basin is planted here,
        # else into the carrier coord 0).
        target1 = coord_id if (kind == "basin" and layer_id == 1) else 0
        block1 = model.blocks[1]
        w1 = torch.zeros(block1.linear.in_features)
        for i in range(k):
            w1[2 * i] = g
            w1[2 * i + 1] = -g
        block1.linear.weight[target1] = w1
        block1.linear.bias[target1] = 0.0

        if kind == "basin":
            carrier = target1
            next_layer = 2
        else:
            # Probe the cap value on the target circle and at 2x the radius
            # to size the crest bump in v1 units.
            probes = torch.stack(
                [
                    center + torch.tensor([radius, 0.0]),
                    center + torch.tensor([2.0 * radius, 0.0]),
                ]
            )
            v = model.forward_until(probes, layer_id=1, kind="post")[:, target1]
            v_center = model.forward_until(center[None], layer_id=1, kind="post")[
                0, target1
            ]

            v_r = float(v[0].item())
            v_2r = float(v[1].item())
            eps = 0.8 * min(float(v_center.item()) - v_r, v_r - v_2r)
            assert eps > 0, "degenerate cap profile — increase sharpness"

            sigma = sharpness / eps

            block2 = model.blocks[2]
            for sign_idx, offset_sign in enumerate((+1.0, -1.0)):
                w = torch.zeros(block2.linear.in_features)
                w[target1] = sigma
                block2.linear.weight[sign_idx] = w
                block2.linear.bias[sign_idx] = sigma * (
                    offset_sign * eps - v_r
                )

            g2 = 1.0 / (2.0 * math.tanh(sharpness))
            target3 = coord_id if layer_id == 3 else 0
            block3 = model.blocks[3]
            w3 = torch.zeros(block3.linear.in_features)
            w3[0], w3[1] = g2, -g2
            block3.linear.weight[target3] = w3
            block3.linear.bias[target3] = 0.0

            carrier = target3
            next_layer = 4

        # Monotone carrier chain up to the requested depth.
        for lid in range(next_layer, layer_id + 1):
            block = model.blocks[lid]
            target = coord_id if lid == layer_id else 0
            w = torch.zeros(block.linear.in_features)
            w[carrier] = 1.5
            block.linear.weight[target] = w
            block.linear.bias[target] = 0.0
            carrier = target

    return model


def positive_control_report(
    model,
    planted: dict,
    feature_points: torch.Tensor,
    bounds,
    criteria: dict,
    device: str = "cpu",
    resolution: int = 100,
) -> dict:
    """
    Runs every criterion in ``criteria`` ({name: fn}, extra_criteria protocol)
    on the planted neuron and asserts each fires; returns the per-criterion
    scores. A criterion failing HERE means the detector, not the network, is
    broken — run this before trusting any negative result.

    ``planted``: {"layer_id": int, "coord_id": int} (the planting location).

    "Fires" = the criterion's boolean witness output (any key equal to or
    ending in "is_witness", or the "is_{name}_witness" convention) is True.

    Returns dict {criterion_name: {**flat_scores, "fired": bool}}; raises
    AssertionError listing every criterion that failed to fire.
    """
    from models.NN_analysis import TruncatedMLPFunction, ScalarProjection
    from analysis.fields import evaluate_field_on_grid

    layer_id = int(planted["layer_id"])
    coord_id = int(planted["coord_id"])

    model = model.to(device)
    model.eval()

    feature_points = feature_points.to(device)
    if not torch.is_floating_point(feature_points):
        feature_points = feature_points.float()

    neuron = ScalarProjection(
        TruncatedMLPFunction(model, layer_id, kind="post"),
        coord_id=coord_id,
    )

    field = evaluate_field_on_grid(
        neuron, bounds=bounds, resolution=resolution, device=device
    )

    report = {}
    failures = []

    for name, fn in criteria.items():
        result = fn(
            function=neuron,
            feature_points=feature_points,
            bounds=bounds,
            resolution=resolution,
            device=device,
            field=field,
        )

        assert isinstance(result, dict), f"criterion {name!r} must return a dict"

        fired = None
        for key, value in result.items():
            if key == "is_witness" or key.endswith("_witness") or (
                key.startswith("is_") and "witness" in key
            ):
                if isinstance(value, bool):
                    fired = value
                    break

        assert fired is not None, (
            f"criterion {name!r} returned no boolean witness key; "
            f"keys: {list(result.keys())}"
        )

        flat = {
            k: v for k, v in result.items() if isinstance(v, (int, float, bool))
        }
        flat["fired"] = fired
        report[name] = flat

        if not fired:
            failures.append(name)

    if failures:
        error = AssertionError(
            f"POSITIVE CONTROL FAILED — criteria did not fire on the planted "
            f"radial neuron: {failures}. Detector calibration problem; report: "
            f"{ {n: report[n] for n in failures} }"
        )
        # Attach the full per-criterion report so a caller that catches this
        # can still persist WHICH criteria fired and with what scores, instead
        # of recording only "it raised" (AUDIT/04_repair_log.md R5).
        error.report = report
        error.failed_criteria = list(failures)
        raise error

    return report
