"""Named dataset-construction recipes.

Every dataset used in the experiments is built here (not in notebooks), so any
experiment can be reproduced by importing a recipe and persisting the result via
``management.experiment_management.store_dataset_instance``.

Naming convention for the paper's dataset series (see PROJECT.md §Datasets):

    a  / a1  -> ``make_single_motif_dataset(disk_label=None)`` / shifted variant
    b  / b1  -> ``make_single_motif_dataset(disk_label=1)``    / shifted variant
    c  / c1  -> ``make_single_motif_dataset(disk_label=0)``    / shifted variant

The centered series (a, b, c) keeps perfect circular symmetry; the shifted
series (a.1, b.1, c.1) breaks it (vertical ellipse outer, off-center inner) to
isolate symmetry-induced incidental witnesses.
"""

import math

import torch

from data.data_generation import (
    Data,
    PointCloudObject,
    Circle2D,
    Circle3D,
    Disk2D,
    EllipseRegion2D,
    RadialCurve2D,
    WavyCircle2D,
    SphereND,
    BallND,
    embed_in_ambient,
)

# NOTE: the only remaining scaffold recipe is make_highdim_embedded_dataset
# (experiments 11/14) at the end of this file.


def linked_circles_3d(n_points: int) -> Data:
    """Two interlinked circles in R^3 (Naitzat-Lim style toy example).

    Both circles are registered as features ("C_1", "C_2") — each is a
    codim-2 curve, the experiment-10 object of study.
    """
    # Circle 1: xy-plane, center = (0, 0, 0)
    circle_1 = Circle3D(n_points)

    # Circle 2: xz-plane, center = (1, 0, 0).
    # Base Circle3D lies in the xy-plane; (cos t, sin t, 0) -> (cos t, 0, sin t).
    linear_xz = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    circle_2 = Circle3D(n_points, label=1).apply(
        linear=linear_xz,
        bias=torch.tensor([1.0, 0.0, 0.0]),
    )

    linked_circles = Data(dim=3)
    linked_circles.add(circle_1, name="C_1", is_feature=True)
    linked_circles.add(circle_2, name="C_2", is_feature=True)
    return linked_circles


def nested_shifted_circles_2d(n_points: int) -> Data:
    """Three nested, slightly shifted circles ("pickasso"-style sanity dataset)."""
    circle_big = Circle2D(n_points // 4, label=0).apply(3 * torch.eye(2), torch.tensor([0, 0]))
    circle_mid = Circle2D(n_points // 2, label=1).apply(2 * torch.eye(2), torch.tensor([0, -0.5]))
    circle_small = Circle2D(n_points // 4, label=0).apply(1 * torch.eye(2), torch.tensor([-0.25, 0]))

    data = Data(2).add(circle_big).add(circle_mid).add(circle_small)

    return data


def make_single_motif_dataset(
    disk_label: int | None = None,
    seed: int = 0,
    n_circle: int = 400,
    n_disk: int = 800,
) -> Data:
    """Centered concentric-circles datasets (a), (b), (c).

    Builds one of:

        a) C_big^0 + C_mid^1                (disk_label=None)
        b) C_big^0 + C_mid^1 + D_in^1       (disk_label=1)
        c) C_big^0 + C_mid^1 + D_in^0       (disk_label=0)

    The target analysis feature is C_mid; C_big is also registered as a feature.
    """
    torch.manual_seed(seed)

    circle_big = Circle2D(n_circle, label=0).apply(
        3.0 * torch.eye(2),
        torch.tensor([0.0, 0.0]),
    )

    circle_mid = Circle2D(n_circle, label=1).apply(
        2.0 * torch.eye(2),
        torch.tensor([0.0, 0.0]),
    )

    data = (
        Data(2)
        .add(circle_big, name="C_big", is_feature=True)
        .add(circle_mid, name="C_mid", is_feature=True)
    )

    if disk_label is not None:
        disk_in = Disk2D(n_disk, label=disk_label).apply(
            1.0 * torch.eye(2),
            torch.tensor([0.0, 0.0]),
        )

        data.add(disk_in, name="D_in")

    return data


def make_single_shifted_dataset(
    disk_label: int | None = None,
    seed: int = 0,
    n_circle: int = 400,
    n_disk: int = 400,
) -> Data:
    """Symmetry-broken counterparts (a.1), (b.1), (c.1).

    Builds one of:

        a.1) C_V_big^0 + C_mid^1                 (disk_label=None)
        b.1) C_V_big^0 + C_mid^1 + D_H_in^1      (disk_label=1)
        c.1) C_V_big^0 + C_mid^1 + D_H_in^0      (disk_label=0)

    Outer circle becomes a tall vertical ellipse, inner circle is shifted
    off-center, the interior disk becomes a horizontal ellipse.

    Primary target feature: C_mid. Additional diagnostic feature: C_V_big.
    """
    torch.manual_seed(seed)

    circle_big_V = Circle2D(n_circle, label=0).apply(
        4.5 * torch.tensor([
            [1, 0],
            [0, 4]
        ]),
        torch.tensor([0.0, 0.0]),
    )

    circle_mid = Circle2D(n_circle, label=1).apply(
        3 * torch.eye(2),
        torch.tensor([0.0, -5]),
    )

    data = (
        Data(2)
        .add(circle_big_V, name="C_V_big", is_feature=True)
        .add(circle_mid, name="C_mid", is_feature=True)
    )

    if disk_label is not None:
        disk_in_H = Disk2D(n_disk, label=disk_label).apply(
            0.5 * torch.tensor([
                [2, 0],
                [0, 0.5]
            ]),
            torch.tensor([0.0, -5.5]),
        )

        data.add(disk_in_H, name="D_H_in")

    return data


def rotation_matrix(theta: float, dtype=torch.float32) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, -s],
         [s,  c]],
        dtype=dtype,
    )


def ellipse_matrix(axis_x: float, axis_y: float, angle: float, dtype=torch.float32) -> torch.Tensor:
    """Maps the unit circle to a rotated ellipse with semiaxes axis_x, axis_y."""
    R = rotation_matrix(angle, dtype=dtype)
    S = torch.diag(torch.tensor([axis_x, axis_y], dtype=dtype))
    return R @ S


def make_multi_feature_motif_dataset(
    seed: int = 0,
    n_ellipse: int = 200,
    n_disk: int = 400,
    outer_labels: tuple[int, ...] = (0, 0, 1, 0, 1),
) -> Data:
    """Multi-feature dataset with five task-relevant ellipse features.

    Each motif has the form

        E_i^{outer_label_i} + D_i^{1 - outer_label_i},

    where E_i is an ellipse and D_i is a disk inside it. Used for the
    multi-feature witness experiment (network width below feature count).

    The target analysis features are E_1, ..., E_5.
    """
    torch.manual_seed(seed)

    if len(outer_labels) != 5:
        raise ValueError("outer_labels must contain exactly 5 labels.")

    dtype = torch.float32

    # Non-symmetric, non-regular placement, shifted so the mean is ~zero.
    centers = torch.tensor(
        [
            [-5.2, -1.2],
            [-2.3,  3.1],
            [ 2.6,  2.4],
            [ 5.1, -1.5],
            [ 0.4, -4.0],
        ],
        dtype=dtype,
    )
    centers = centers - centers.mean(dim=0, keepdim=True)

    # Moderately different ellipses in general position — different enough to
    # break symmetry, similar enough that all targets are the same feature type.
    ellipse_params = [
        # axis_x, axis_y, angle
        (1.15, 0.78,  0.25),
        (1.00, 0.70,  1.10),
        (1.12, 0.84, -0.55),
        (0.95, 0.74,  0.80),
        (1.08, 0.76, -1.20),
    ]

    # Circular disks safely inside the ellipses.
    disk_radii = [0.38, 0.35, 0.42, 0.36, 0.40]

    data = Data(2)

    for i in range(5):
        outer_label = int(outer_labels[i])
        inner_label = 1 - outer_label

        center = centers[i]

        axis_x, axis_y, angle = ellipse_params[i]
        A_ellipse = ellipse_matrix(axis_x, axis_y, angle, dtype=dtype)

        ellipse = Circle2D(n_ellipse, label=outer_label).apply(
            A_ellipse,
            center,
        )

        disk = Disk2D(n_disk, label=inner_label).apply(
            disk_radii[i] * torch.eye(2, dtype=dtype),
            center,
        )

        data.add(
            ellipse,
            name=f"E_{i + 1}",
            is_feature=True,
        )

        data.add(
            disk,
            name=f"D_{i + 1}",
        )

    return data


# ---------------------------------------------------------------------------
# Recipes for PLAN.md experiments. Recipes below the [SCAFFOLD] marker are
# not implemented yet. Convention: every recipe is deterministic given
# `seed`, registers its target features with is_feature=True / add_feature,
# and leaves persistence to store_dataset_instance at the call site
# (experiments/runner.py).
# ---------------------------------------------------------------------------

def make_halfplane_disk_circle_dataset(
    seed: int = 0,
    n_disk: int = 800,
    n_circle: int = 400,
    disk_center: tuple[float, float] = (-2.5, 0.0),
    disk_radius: float = 1.5,
    circle_center: tuple[float, float] = (2.5, 0.0),
    circle_axes: tuple[float, float] = (1.5, 1.5),
    circle_angle: float = 0.0,
) -> Data:
    """Experiments 1-2: linearly separable, topology task-irrelevant.

    Blue disk entirely in {x < 0} (label 0, no topological structure), red
    circle entirely in {x > 0} (label 1, an S^1 the task does NOT need). The
    hyperplane x=0 solves the task, so the prediction is ~0 witnesses for the
    circle beyond the null-baseline rate (analysis.controls.make_null_features
    supplies the nulls; the runner registers them as features).

    Knobs per PLAN.md exp. 1: per-class counts (density), circle shape
    (ellipse axes/angle), center and size.

    Target feature: the circle ("C_right"). Diagnostic: none by default.
    """
    torch.manual_seed(seed)

    disk_center_t = torch.tensor(disk_center, dtype=torch.float32)
    circle_center_t = torch.tensor(circle_center, dtype=torch.float32)

    # Class separation is the point of the dataset — enforce it.
    assert disk_center[0] + disk_radius < 0, (
        "the disk must lie entirely in {x < 0}"
    )
    assert circle_center[0] - max(circle_axes) > 0, (
        "the circle must lie entirely in {x > 0}"
    )

    disk = Disk2D(n_disk, label=0).apply(
        disk_radius * torch.eye(2),
        disk_center_t,
    )

    circle = Circle2D(n_circle, label=1).apply(
        ellipse_matrix(circle_axes[0], circle_axes[1], circle_angle),
        circle_center_t,
    )

    data = (
        Data(2)
        .add(disk, name="D_left")
        .add(circle, name="C_right", is_feature=True)
    )

    return data


def _make_ring_object(shape: str, params: dict, n_points: int, label: int):
    """Ring builders shared by make_ring_interior_dataset.

    shape="circle":  params {"radius": 2.0}
    shape="ellipse": params {"axis_x": 2.4, "axis_y": 1.6, "angle": 0.0}
    shape="wavy":    params {"radius": 2.0, "amplitude": 0.15, "frequency": 8}
    """
    if shape == "circle":
        radius = float(params.get("radius", 2.0))
        return Circle2D(n_points, label=label).apply(
            radius * torch.eye(2), torch.zeros(2)
        )

    if shape == "ellipse":
        axis_x = float(params.get("axis_x", 2.4))
        axis_y = float(params.get("axis_y", 1.6))
        angle = float(params.get("angle", 0.0))
        return Circle2D(n_points, label=label).apply(
            ellipse_matrix(axis_x, axis_y, angle), torch.zeros(2)
        )

    if shape == "wavy":
        radius = float(params.get("radius", 2.0))
        return WavyCircle2D(
            n_points,
            amplitude=float(params.get("amplitude", 0.15)),
            frequency=int(params.get("frequency", 8)),
            phase=float(params.get("phase", 0.0)),
            label=label,
        ).apply(radius * torch.eye(2), torch.zeros(2))

    raise ValueError(f"Unknown ring shape: {shape!r}")


def make_ring_interior_dataset(
    seed: int = 0,
    ring_shape: str = "circle",  # "circle" | "ellipse" | "wavy"
    ring_params: dict | None = None,
    interior_shape: str = "disk",  # "disk" | "ellipse"
    interior_params: dict | None = None,
    n_ring: int = 400,
    n_interior: int = 800,
    n_outer: int = 400,
    outer_scale: float = 3.0,
    n_ballast: int = 0,
    ballast_center: tuple = (7.0, 0.0),
    ballast_radius: float = 1.0,
    reference_n: int | None = None,
) -> Data:
    """Experiments 3-6 workhorse: feature ring + opposite-class interior.

    Generalizes dataset (c): outer circle (label 0) + ring feature (label 1)
    + interior of the OPPOSITE class (label 0), with independently
    controllable shapes and per-object point counts. n_ring/n_interior is the
    density-ratio knob of experiment 4; ring/interior shape mismatch is the
    incidental-geometry knob of experiment 5.

    Defaults reproduce dataset (c)'s geometry: outer circle r=3, ring r=2,
    interior disk r=1 (interior_params {"radius"} / {"axis_x", "axis_y",
    "angle"} scale it).

    Target feature: the ring ("C_mid" naming kept for downstream
    compatibility). Diagnostic: the outer circle ("C_big").
    """
    torch.manual_seed(seed)

    ring_params = {} if ring_params is None else dict(ring_params)
    interior_params = {} if interior_params is None else dict(interior_params)

    outer = Circle2D(n_outer, label=0).apply(
        outer_scale * torch.eye(2), torch.zeros(2)
    )

    ring = _make_ring_object(ring_shape, ring_params, n_ring, label=1)

    if interior_shape == "disk":
        radius = float(interior_params.get("radius", 1.0))
        interior_linear = radius * torch.eye(2)
    elif interior_shape == "ellipse":
        interior_linear = ellipse_matrix(
            float(interior_params.get("axis_x", 1.2)),
            float(interior_params.get("axis_y", 0.8)),
            float(interior_params.get("angle", 0.0)),
        )
    else:
        raise ValueError(f"Unknown interior shape: {interior_shape!r}")

    interior = Disk2D(n_interior, label=0).apply(
        interior_linear,
        torch.tensor(interior_params.get("center", (0.0, 0.0)),
                     dtype=torch.float32),
    )

    data = (
        Data(2)
        .add(outer, name="C_big", is_feature=True)
        .add(ring, name="C_mid", is_feature=reference_n is None)
        .add(interior, name="D_in")
    )

    # Experiment 4 task F: the interior's density is swept with n_interior
    # while class 0 (= outer + interior + ballast) keeps a FIXED total, so the
    # sweep no longer drags class balance along with it. The ballast is inert
    # by placement: far from every feature, contractible, its own class.
    if n_ballast > 0:
        ballast = Disk2D(n_ballast, label=0).apply(
            ballast_radius * torch.eye(2),
            torch.tensor(ballast_center, dtype=torch.float32),
        )
        data.add(ballast, name="ballast", is_feature=False)

    if reference_n is not None:
        assert ring_shape == "circle", (
            "reference_n builds a uniform CIRCLE; for an ellipse/wavy ring the "
            "reference would not be the feature"
        )
        ring_radius = float(ring_params.get("radius", 2.0))
        data.add_feature(
            name="C_mid",
            points=_uniform_reference_circle(reference_n, ring_radius),
            metadata={
                "feature_role": "target",
                "source": "uniform_reference",
                "reference_n": reference_n,
            },
        )

    return data


def make_sphere_shell_dataset(
    dim: int = 3,
    seed: int = 0,
    n_sphere: int = 1000,
    n_ball: int = 2000,
    n_outer: int = 1000,
    outer_scale: float = 3.0,
) -> Data:
    """Experiment 9: S^{dim-1} + opposite-class ball interior, dim in {3, 4}.

    The codim-1 analog of dataset (c): outer sphere r=outer_scale (label 0)
    + feature sphere r=2 (label 1) + interior ball r=1 (label 0), matching
    dataset (c)'s radii 3/2/1. Uses SphereND/BallND from
    data.data_generation. Grid criteria are dead here — evaluation goes
    through the random-cloud field (EvalSpec.field_source="random"), gradient
    sampling (sampled_structural) and the homological criterion.

    Target feature: the middle sphere ("S_mid"). Diagnostic: "S_big".
    """
    assert dim in (3, 4), "experiment 9 covers dim 3 and 4"

    torch.manual_seed(seed)

    sphere_big = SphereND(n_outer, dim=dim, label=0).apply(
        outer_scale * torch.eye(dim), torch.zeros(dim)
    )

    sphere_mid = SphereND(n_sphere, dim=dim, label=1).apply(
        2.0 * torch.eye(dim), torch.zeros(dim)
    )

    ball_in = BallND(n_ball, dim=dim, label=0).apply(
        1.0 * torch.eye(dim), torch.zeros(dim)
    )

    data = (
        Data(dim)
        .add(sphere_big, name="S_big", is_feature=True)
        .add(sphere_mid, name="S_mid", is_feature=True)
        .add(ball_in, name="B_in")
    )

    return data


def linked_circles_4d(n_points: int, seed: int = 0) -> Data:
    """Experiment 10: the linked_circles_3d configuration embedded in R^4.

    In R^4 the two circles can be unlinked by an ambient isotopy — the
    containment/linking obstruction present in R^3 disappears. Comparing
    witness structure across the two ambient dimensions isolates the
    ambient-dependence of codim >= 2 features (a circle in R^3/R^4 cannot be
    a level-set component of a single smooth neuron: codim(level set) = 1).

    The embedding is a seeded random linear isometry (topology exactly
    preserved), so the R^4 instance is the SAME configuration as the R^3 one,
    only the ambient dimension differs. Features "C_1"/"C_2" carry over.
    """
    return embed_in_ambient(
        linked_circles_3d(n_points),
        ambient_dim=4,
        mode="linear_isometry",
        seed=seed,
    )


def make_highdim_embedded_dataset(
    base_recipe,
    ambient_dim: int = 10,
    noise_std: float = 0.0,
    embed_mode: str = "linear_isometry",
    seed: int = 0,
    **base_recipe_kwargs,
) -> Data:
    """Experiments 11/14 composition: recipe -> embed -> (optional) noise.

    Builds ``base_recipe(**base_recipe_kwargs)``, embeds it into
    R^{ambient_dim} (data.data_generation.embed_in_ambient), then adds
    Gaussian jitter (with_gaussian_noise) when noise_std > 0. Records the
    whole construction in metadata so the instance is self-describing.

    Before trusting any conclusion drawn AFTER dimensionality reduction,
    verify preserved homology (PLAN.md exp. 11 caveat):
    analysis.homological.persistence_of_point_cloud before vs after.
    """
    raise NotImplementedError("Experiments-11/14 scaffolding.")


def make_simple_dataset(
        relevant=False,
        n_disk: int = 500,
        n_circle: int = 500,
        seed=0,
        reference_n: int | None = None,
) -> Data:
    """
    Creates a simple linearly separable dataset with a disk from the left
    and circle from the right. This dataset is coupled with simple_relevant -
    its objects, i.e. disk and circle, are intentionally the same as in
    simple_relevant experiment.

    :param relevant: whether make the dataset with a task-relevant feature or no
    :param n_disk: number of points sampled from disk
    :param n_circle: number of points sampled from circle
    :param seed: seed for random number generator
    :param reference_n: when set, the registered feature is a UNIFORM REFERENCE
        circle of this many points instead of the data's own circle points.
        Needed by experiment 4's thinning sweep: scoring a 7-point circle
        against itself measures the sampling of A, not the network, so the
        measurement target must not thin with the data. None (default) keeps
        the exp1/exp2 behaviour — the feature IS the data component — so
        stored instances and their results are unaffected.
    :return: Dataset
    """
    torch.manual_seed(seed)

    circle = Circle2D(n_circle, label=0).apply(
        1.5 * torch.eye(2),
        torch.tensor([0.0, 0.0]),
    )

    if relevant:
        shift = 0.0
        name = "relevant"
    else:
        shift = -6.0
        name = "irrelevant"

    disk = Disk2D(n_disk, label=1).apply(
        torch.eye(2),
        torch.tensor([shift, 0.0]),
    )

    if reference_n is None:
        data = (
            Data(2)
            .add(circle, name=name, is_feature=True)
            .add(disk, name="disk", is_feature=False)
        )
    else:
        data = (
            Data(2)
            .add(circle, name="circle", is_feature=False)
            .add(disk, name="disk", is_feature=False)
        )
        data.add_feature(
            name=name,
            points=_uniform_reference_circle(reference_n, 1.5),
            metadata={
                "feature_role": "target",
                "source": "uniform_reference",
                "reference_n": reference_n,
                "n_circle": n_circle,
            },
        )

    return data

# ---------------------------------------------------------------------------
# Experiment 4 -- density versus topology
# ---------------------------------------------------------------------------
#
# The device of the whole experiment is MASS REDISTRIBUTION: take a region of
# one class, hold its point count fixed, and move a fraction `spike_fraction`
# of its points from the area onto a circle lying inside it.
#
# What that leaves untouched: the class, the total point count of the class,
# the class balance, the support (the circle is interior to the region), the
# region's geometry and the evaluation box. What it changes: the density
# profile, and nothing else. The `spike_fraction = 0` cell is therefore an
# exact matched control -- the same feature, in the same place, in the same
# dataset, with the concentration absent -- which the ORIGINAL exp4 sweep
# (n_ring : n_interior on OPPOSITE classes) could never have, because there the
# density knob was collinear with class balance.
#
# Two arrangements:
#
#   mode="probe"     the task-relevant circle of exp1 (r=1.5, class 0) around
#                    its opposite-class interior disk (r=1, class 1), PLUS a
#                    probe region of class 1 sitting far away, with the spike
#                    circle strictly INSIDE it. Everything around the spike is
#                    one class, so the task cannot see it: any witness of that
#                    circle is the density channel alone. The relevant circle
#                    rides along in the same model as a within-model positive
#                    control -- the network demonstrably CAN make closed level
#                    sets, so a null result on the probe is not "no curved
#                    machinery anywhere".
#
#   mode="boundary"  the same interior disk, but class 0 is an ANNULUS whose
#                    inner radius is fixed, with the spike on that inner
#                    radius. The forced boundary's LOCATION does not move with
#                    the fraction; only the density on it does.
#
# Targets are evaluated against UNIFORM REFERENCE circles of a fixed point
# count (`reference_n`), registered via Data.add_feature(points=...) and hence
# NOT part of the training data. Without that the Chamfer score would itself
# depend on how densely A is sampled, and the sweep would measure the
# instrument rather than the network.


def _annulus_points(n_points: int, inner: float, outer: float) -> torch.Tensor:
    """`n_points` uniform in the annulus inner <= r <= outer, by rejection.

    Disk2D is area-uniform, so filtering it keeps area-uniformity. The
    acceptance rate is 1 - (inner/outer)^2; oversample by its inverse with
    slack and truncate.
    """
    if n_points <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    keep_rate = 1.0 - (inner / outer) ** 2
    assert keep_rate > 0.05, "annulus too thin for rejection sampling"

    kept = []
    total = 0
    while total < n_points:
        batch = max(int(n_points / keep_rate * 1.3), 64)
        pts = Disk2D(batch, label=0).get_data() * outer
        radii = pts.norm(dim=1)
        accepted = pts[radii >= inner]
        kept.append(accepted)
        total += len(accepted)

    return torch.cat(kept, dim=0)[:n_points]


def _uniform_reference_circle(
    n_points: int, radius: float, center=(0.0, 0.0)
) -> torch.Tensor:
    """The measurement target: a circle sampled uniformly at a FIXED count.

    Deliberately independent of how densely the data samples the same circle --
    that independence is the whole point (see the module note above).
    """
    circle = Circle2D(n_points, label=0).apply(
        radius * torch.eye(2),
        torch.tensor(center, dtype=torch.float32),
    )
    return circle.get_data()


def make_density_spike_dataset(
    mode: str = "probe",  # "probe" | "boundary"
    spike_fraction: float = 0.0,
    n_circle: int = 500,
    n_interior: int = 500,
    circle_radius: float = 1.5,
    interior_radius: float = 1.0,
    # --- mode="probe"
    n_probe: int = 500,
    probe_center: tuple = (-6.0, 0.0),
    probe_radius: float = 2.5,
    spike_radius: float = 1.5,
    # --- mode="boundary"
    n_ring_region: int = 1000,
    ring_outer: float = 3.0,
    # --- measurement
    reference_n: int = 2000,
    seed: int = 0,
) -> Data:
    """Experiment 4: density held against topology by redistributing mass.

    ``spike_fraction`` is the ONLY manipulated variable; every count, class and
    boundary location is invariant along the sweep. See the module note above
    for why the two modes exist and what each isolates.

    Features registered (all uniform reference point sets, never the data's own
    points):

      mode="probe"     "probe"    -- the task-invisible circle carrying the
                                     spike (feature_role "target")
                       "relevant" -- exp1's task-relevant circle, the
                                     within-model positive control
                                     (feature_role "diagnostic")
      mode="boundary"  "boundary" -- the forced boundary carrying the spike
                                     (feature_role "target")
    """
    assert mode in ("probe", "boundary"), f"Unknown mode: {mode!r}"
    assert 0.0 <= spike_fraction <= 1.0, "spike_fraction must be in [0, 1]"

    torch.manual_seed(seed)

    data = Data(2)

    interior = Disk2D(n_interior, label=1).apply(
        interior_radius * torch.eye(2), torch.zeros(2)
    )
    data.add(interior, name="interior", is_feature=False)

    if mode == "probe":
        # class 0: exp1's circle, bitwise unchanged along the whole sweep
        circle = Circle2D(n_circle, label=0).apply(
            circle_radius * torch.eye(2), torch.zeros(2)
        )
        data.add(circle, name="circle", is_feature=False)

        # class 1: the probe region, mass split between area and spike
        n_spike = int(round(spike_fraction * n_probe))
        n_area = n_probe - n_spike
        center = torch.tensor(probe_center, dtype=torch.float32)

        if n_area > 0:
            probe_area = Disk2D(n_area, label=1).apply(
                probe_radius * torch.eye(2), center
            )
            data.add(probe_area, name="probe_area", is_feature=False)

        if n_spike > 0:
            probe_spike = Circle2D(n_spike, label=1).apply(
                spike_radius * torch.eye(2), center
            )
            data.add(probe_spike, name="probe_spike", is_feature=False)

        data.add_feature(
            name="probe",
            points=_uniform_reference_circle(
                reference_n, spike_radius, probe_center
            ),
            metadata={
                "feature_role": "target",
                "source": "uniform_reference",
                "reference_n": reference_n,
                "spike_fraction": spike_fraction,
                "n_spike": n_spike,
            },
        )
        data.add_feature(
            name="relevant",
            points=_uniform_reference_circle(reference_n, circle_radius),
            metadata={
                "feature_role": "diagnostic",
                "source": "uniform_reference",
                "reference_n": reference_n,
            },
        )

    else:  # mode == "boundary"
        # class 0: an annulus whose INNER radius is fixed, so the location of
        # the forced boundary does not move with spike_fraction.
        n_spike = int(round(spike_fraction * n_ring_region))
        n_area = n_ring_region - n_spike

        if n_area > 0:
            area = PointCloudObject(
                _annulus_points(n_area, circle_radius, ring_outer), label=0
            )
            data.add(area, name="ring_area", is_feature=False)

        if n_spike > 0:
            spike = Circle2D(n_spike, label=0).apply(
                circle_radius * torch.eye(2), torch.zeros(2)
            )
            data.add(spike, name="ring_spike", is_feature=False)

        data.add_feature(
            name="boundary",
            points=_uniform_reference_circle(reference_n, circle_radius),
            metadata={
                "feature_role": "target",
                "source": "uniform_reference",
                "reference_n": reference_n,
                "spike_fraction": spike_fraction,
                "n_spike": n_spike,
            },
        )

    return data


# ---------------------------------------------------------------------------
# Experiment 4 -- density versus topology (final design)
# ---------------------------------------------------------------------------
#
# Density is varied by GEOMETRY at a constant point count: the circle of
# experiment 1 becomes a RING of width w, sampled uniformly by area over
# [r_mid - w/2, r_mid + w/2]; the interior becomes a DISK of radius R. Each
# object always carries the same number of points, so class balance never moves
# with density, and a ring of nonzero width stays two-dimensional, so the
# support never degenerates. w = 0 reproduces experiment 1's circle exactly.
#
# FOUR MEASUREMENT TARGETS, because a ring does not name one. The criterion is
# computed against the ring itself and against three circles inside it (inner
# edge, midline, outer edge); which of them answers our question best is an
# empirical matter settled after the run, not before it. At w = 0 all four
# coincide, so only one is registered -- registering four copies of the same
# curve would quadruple the rows and say nothing.
#
# Every target carries its OWN A.1 nulls: a null must be a rigid motion of the
# target it is the floor for, so a ring target has ring nulls and a circle
# target has circle nulls of that radius. Nulls are registered into the instance
# rather than built per run, because here they are deterministic rigid motions
# fully described by the recipe's parameters -- the provenance objection that
# keeps random nulls out of instances does not apply.
#
# Reference point counts: circles carry 500 points, as in experiment 1 (step 0.1
# measured the score to be flat from 500 to 4000). The ring carries 1000,
# because a two-dimensional target sampled at 500 has an inter-point spacing
# coarser than the evaluation grid, and sampling A finer than the grid step buys
# nothing while costing time linearly.

_EXP4_PLACEMENTS = {
    "far_right": (4.5, 0.0),      # вдоль оси задачи
    "above_target": (0.0, 4.5),   # поперёк неё
    "in_corridor": (-3.5, -3.0),  # у границы решения; только task-irrelevant
}


def _annulus_uniform(n_points: int, r_in: float, r_out: float, seed: int) -> torch.Tensor:
    """n points uniform BY AREA in the annulus r_in <= r <= r_out."""
    g = torch.Generator().manual_seed(int(seed))
    u = torch.rand(n_points, generator=g)
    r = torch.sqrt(u * (r_out ** 2 - r_in ** 2) + r_in ** 2)
    th = 2 * math.pi * torch.rand(n_points, generator=g)
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1)


def make_density_ring_dataset(
    relevant: bool = True,
    w: float = 0.0,
    disk_radius: float = 1.0,
    n_ring: int = 500,
    n_disk: int = 500,
    r_mid: float = 1.5,
    irrelevant_disk_center: tuple = (-6.0, 0.0),
    circle_reference_n: int = 500,
    ring_reference_n: int = 1000,
    gap_min: float = 0.5,
    seed: int = 0,
) -> Data:
    """Experiment 4's dataset family: ring of width w around a disk of radius R.

    relevant=True  -- ring (class 0) surrounds the disk (class 1) at the origin;
                      the task forces a closed boundary between them, so the
                      ring is task-relevant. Two null placements.
    relevant=False -- same ring at the origin, disk moved far away; the task is
                      linearly separable and the ring is task-irrelevant. Three
                      null placements: the empty corridor between the classes
                      admits a third, which experiment 1 measured to be the
                      HIGHEST floor and therefore its upper estimate.

    Registered features: "t_ring", "t_inner", "t_mid", "t_outer" (only "t_mid"
    when w = 0, where all four coincide), each with its own nulls named
    "n_<target>_<placement>". Roles are "target" and "null".
    """
    assert w >= 0.0, "ширина кольца неотрицательна"
    r_in, r_out = r_mid - w / 2.0, r_mid + w / 2.0
    assert r_in > 0.0, "кольцо шире своего радиуса"

    if relevant:
        clearance = r_in - disk_radius
        assert clearance >= gap_min - 1e-9, (
            f"зазор {clearance:.3f} меньше {gap_min}: при таком зазоре мы мерили бы "
            f"способность сети резко меняться на малом расстоянии, а не плотность"
        )

    torch.manual_seed(int(seed))
    data = Data(2)

    ring = (Circle2D(n_ring, label=0).apply(r_mid * torch.eye(2), torch.zeros(2)).get_data()
            if w == 0 else _annulus_uniform(n_ring, r_in, r_out, seed))
    data.add(PointCloudObject(ring, label=0), name="ring", is_feature=False)

    center = (0.0, 0.0) if relevant else irrelevant_disk_center
    disk = Disk2D(n_disk, label=1).apply(
        disk_radius * torch.eye(2), torch.tensor(center, dtype=torch.float32)
    )
    data.add(disk, name="disk", is_feature=False)

    # --- мишени. При w = 0 все четыре совпадают, поэтому регистрируется одна.
    circle = lambda rho: Circle2D(circle_reference_n).get_data() * rho
    if w == 0:
        targets = {"t_mid": circle(r_mid)}
    else:
        targets = {
            "t_inner": circle(r_in),
            "t_mid": circle(r_mid),
            "t_outer": circle(r_out),
            "t_ring": _annulus_uniform(ring_reference_n, r_in, r_out, seed + 10_000),
        }

    placements = dict(_EXP4_PLACEMENTS)
    if relevant:
        placements.pop("in_corridor")

    for name, points in targets.items():
        data.add_feature(
            name=name, points=points,
            metadata={"feature_role": "target", "target_family": name,
                      "kind": "ring" if name == "t_ring" else "circle",
                      "n_points": int(len(points))},
        )
        for place_name, (dx, dy) in placements.items():
            shift = torch.tensor([dx, dy], dtype=torch.float32)
            data.add_feature(
                name=f"n_{name}_{place_name}", points=points + shift,
                metadata={"feature_role": "null", "target_family": name,
                          "placement": place_name},
            )

    return data


# ---------------------------------------------------------------------------
# Experiment 5 -- the geometric effect
# ---------------------------------------------------------------------------
#
# The object of study is the PAIR (feature curve, interior region), because the
# separating level set lives in the gap between them and its shape is set by
# both edges of that gap. In experiments 1 and 2 both objects are round and
# concentric, so "the level set repeats the feature" and "the level set repeats
# the interior" predict the same thing; every cell here breaks that tie in one
# named way.
#
# Four knobs per object, and they are deliberately independent:
#   shape    -- an ellipse (aspect) or a harmonic deformation (k, eps)
#   measure  -- the angular density profile, which perturbs WHERE points sit
#               along the object without moving the object itself
#   mass     -- the point count
#   pose     -- rotation, and the interior's offset from the feature's centre
#
# The measurement target registered with the dataset is always a UNIFORM
# reference curve of the same shape, never the data's own points (experiment 4,
# step 0.1): a Chamfer score against the data would track how the curve was
# sampled, which is exactly one of the variables under study here.


def ellipse_radial(a: float, b: float):
    """Exact polar form of an ellipse with semiaxes (a, b)."""
    return lambda t: a * b / torch.sqrt((b * torch.cos(t)) ** 2 + (a * torch.sin(t)) ** 2)


def harmonic_radial(radius: float, k: int, eps: float):
    """r(theta) = radius * (1 + eps*cos(k*theta)). Convex iff eps < 1/(1+k^2)."""
    return lambda t: radius * (1.0 + eps * torch.cos(k * t))


def cosine_density(k: int, eps: float):
    """Angular density profile 1 + eps*cos(k*theta); eps in (-1, 1)."""
    assert abs(eps) < 1.0, "density modulation must keep the profile positive"
    return lambda t: 1.0 + eps * torch.cos(k * t)


def convexity_limit(k: int) -> float:
    """Largest eps for which r = 1 + eps*cos(k*theta) is still convex."""
    return 1.0 / (1.0 + k ** 2)


def _feature_semiaxes(aspect: float, hold: str, value: float) -> tuple[float, float]:
    """
    Semiaxes (a, b), a >= b, of an ellipse of the given aspect ratio, under one
    of four ways of pinning its scale. Which one to hold is a design choice, not
    a fact -- they trade off against each other and the trade-off is the point:

        "minor"     b fixed. The gap to a round interior does not shrink, and
                    the interior can be left literally untouched; the feature
                    grows outward instead.
        "major"     a fixed. The feature stays inside the same box, but the gap
                    closes as the aspect grows.
        "perimeter" the curve keeps its length, so a fixed point count keeps the
                    linear density -- at the cost of the gap closing fastest of
                    all (at aspect 2 the minor semiaxis of a perimeter-matched
                    1.5-circle is already 0.97).
        "area"      the enclosed area is kept; a compromise between the two.
    """
    assert aspect >= 1.0, "aspect is the major/minor ratio, so >= 1"
    if hold == "minor":
        b = value
        a = value * aspect
    elif hold == "major":
        a = value
        b = value / aspect
    elif hold == "area":
        # pi*a*b = pi*value^2
        b = value / math.sqrt(aspect)
        a = b * aspect
    elif hold == "perimeter":
        # Ramanujan II, solved for the scale by bisection.
        target = 2 * math.pi * value
        lo, hi = 1e-3, 100.0
        for _ in range(80):
            b = 0.5 * (lo + hi)
            a = b * aspect
            h = ((a - b) / (a + b)) ** 2
            perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
            if perimeter < target:
                lo = b
            else:
                hi = b
        b = 0.5 * (lo + hi)
        a = b * aspect
    else:
        raise ValueError(f"Unknown hold: {hold!r} (minor|major|area|perimeter)")
    return float(a), float(b)


def min_clearance(curve_points: torch.Tensor, region_boundary: torch.Tensor) -> float:
    """Smallest distance between the feature curve and the interior's boundary."""
    return float(torch.cdist(curve_points, region_boundary).min())


def make_geometry_pair_dataset(
    # --- the feature: a closed curve of class 0 -------------------------------
    feature_aspect: float = 1.0,
    feature_scale: float = 1.5,
    feature_hold: str = "minor",
    feature_harmonic: tuple[int, float] | None = None,
    feature_rotation: float = 0.0,
    feature_density: tuple[int, float] | None = None,
    n_feature: int = 500,
    # --- the interior: a filled region of class 1 ----------------------------
    interior_scale: float = 1.0,
    interior_aspect: float = 1.0,
    interior_hold: str = "major",
    interior_rotation: float = 0.0,
    interior_density: tuple[int, float] | None = None,
    interior_offset: tuple[float, float] = (0.0, 0.0),
    n_interior: int = 500,
    # --- bookkeeping ---------------------------------------------------------
    target: str = "cloud",
    reference_n: int = 2000,
    feature_name: str = "ring",
    seed: int = 0,
) -> Data:
    """
    One (feature curve, interior region) pair -- the unit cell of experiment 5.

    ``feature_aspect`` / ``interior_aspect`` are major:minor ratios (1.0 = round).
    ``feature_harmonic = (k, eps)`` replaces the ellipse by
    ``r = scale*(1 + eps*cos(k*theta))``, which is how corners and higher shape
    modes enter; it is convex iff ``eps < 1/(1+k^2)``.
    ``feature_hold`` / ``interior_hold`` say WHICH scale is pinned while the
    aspect ratio changes (see ``_feature_semiaxes``). The feature defaults to
    ``"minor"``, so it grows outward and the gap is not eaten; the interior
    defaults to ``"major"``, so it only ever shrinks and stays inside.

    ``feature_density`` / ``interior_density`` are ``(k, eps)`` angular density
    profiles: they change WHERE the points sit on an object without changing the
    object. A round curve with ``feature_density=(2, 0.6)`` therefore carries an
    elliptic point distribution on a perfectly circular support -- the cell that
    separates the geometric effect from the density effect.

    The curve is always sampled uniformly in ARC LENGTH before the density
    profile is applied, so shape and density stay independent (see
    ``RadialCurve2D``).

    ``target`` says WHICH point set the criteria measure against:

        "cloud"      the curve's own points -- what the network actually saw,
                     and what experiments 1, 2 and 4 used. The default.
        "reference"  a uniform ``reference_n``-point copy of the same curve --
                     the ideal set the data was sampled from, so the ruler does
                     not move when the sampling does.
        "both"       both, as two target features (``<name>`` and
                     ``<name>_ref``). Doubles the evaluation cost of the cell,
                     because the field is re-evaluated per feature.

    Which one to use is an empirical question, and it has been measured on
    experiment 1's stored ensemble (experiment 4, step 0.1): scoring the same
    neurons against uniform references of 125...4000 points instead of the
    data's own 500 moves the best-neuron score by 0.2 %, and by 2 % even at
    125 points. So the default is the cloud, for continuity; the reference is
    there for cells sparse enough to leave that measured range.

    Metadata records the realised geometry -- perimeter, area, point densities
    and, above all, ``min_clearance`` together with ``interior_inside``: geometry
    moves the gap around, so the gap is reported per dataset rather than assumed,
    and the clearance is only meaningful while the interior is still inside (it
    grows again once the interior has escaped through the curve).
    """
    torch.manual_seed(seed)
    assert target in ("cloud", "reference", "both"), (
        f"target must be cloud|reference|both, got {target!r}"
    )

    if feature_harmonic is None:
        a, b = _feature_semiaxes(feature_aspect, feature_hold, feature_scale)
        radial = ellipse_radial(a, b)
        feature_geometry = {"kind": "ellipse", "a": a, "b": b, "aspect": feature_aspect,
                            "hold": feature_hold}
    else:
        k, eps = feature_harmonic
        radial = harmonic_radial(feature_scale, int(k), float(eps))
        feature_geometry = {"kind": "harmonic", "radius": feature_scale, "k": int(k),
                            "eps": float(eps), "convex": bool(abs(eps) < convexity_limit(int(k)))}

    density_fn = None if feature_density is None else cosine_density(*feature_density)

    curve = RadialCurve2D(
        n_points=n_feature, radial=radial, angular_density=density_fn,
        rotation=feature_rotation, label=0,
    )

    ia, ib = _feature_semiaxes(interior_aspect, interior_hold, interior_scale)
    interior = EllipseRegion2D(
        n_points=n_interior, a=ia, b=ib, angle=interior_rotation,
        angular_density=None if interior_density is None else cosine_density(*interior_density),
        label=1, seed=seed,
    )
    offset = torch.tensor(interior_offset, dtype=torch.float32)
    if float(offset.abs().sum()) > 0:
        interior.apply(torch.eye(2), offset)

    # The measurement target: the same curve, evenly sampled, not in the data.
    reference = RadialCurve2D(
        n_points=reference_n, radial=radial, angular_density=None,
        rotation=feature_rotation, label=0,
    ).get_data()

    # The interior's boundary, for the clearance report.
    theta = torch.linspace(0, 2 * math.pi, 2001)[:-1]
    boundary = torch.stack([ia * torch.cos(theta), ib * torch.sin(theta)], dim=1)
    if interior_rotation != 0.0:
        c, s = math.cos(interior_rotation), math.sin(interior_rotation)
        boundary = boundary @ torch.tensor([[c, -s], [s, c]]).T
    boundary = boundary + offset

    clearance = min_clearance(reference, boundary)

    # Is the interior still inside the feature? The curve is star-shaped about
    # the origin, so a radial test against r(theta) settles it. Without this the
    # clearance alone is misleading: it goes back UP once the interior has left
    # the ring entirely.
    b_theta = torch.atan2(boundary[:, 1], boundary[:, 0])
    inside = bool((torch.linalg.norm(boundary, dim=1) < curve.radius(b_theta)).all())

    data = (
        Data(2)
        .add(curve, name="feature_curve", is_feature=False)
        .add(interior, name="interior", is_feature=False)
    )
    # The measured set. "cloud" is the curve's own points, "reference" the
    # uniform copy; "both" registers the reference under a second name so the
    # two can be compared row by row on the same neurons.
    primary_points = curve.get_data() if target in ("cloud", "both") else reference
    primary_source = "curve_points" if target in ("cloud", "both") else "uniform_reference"
    data.add_feature(
        name=feature_name,
        points=primary_points,
        metadata={
            "feature_role": "target",
            "source": primary_source,
            "reference_n": reference_n,
            "geometry": feature_geometry,
            "perimeter": curve.perimeter,
            "n_curve_points": n_feature,
            "linear_density": n_feature / curve.perimeter,
            "feature_density": feature_density,
            "interior": {"a": ia, "b": ib, "aspect": interior_aspect,
                         "hold": interior_hold,
                         "rotation": interior_rotation, "offset": list(interior_offset),
                         "n_points": n_interior,
                         "areal_density": n_interior / (math.pi * ia * ib),
                         "density_profile": interior_density},
            "min_clearance": clearance,
            "interior_inside": inside,
            "class_balance": n_interior / (n_feature + n_interior),
        },
    )
    if target == "both":
        data.add_feature(
            name=f"{feature_name}_ref",
            points=reference,
            metadata={"feature_role": "target", "source": "uniform_reference",
                      "reference_n": reference_n, "geometry": feature_geometry,
                      "min_clearance": clearance, "interior_inside": inside},
        )

    data.metadata["exp5"] = {
        "min_clearance": clearance,
        "interior_inside": inside,
        "perimeter": curve.perimeter,
        "linear_density": n_feature / curve.perimeter,
        "areal_density": n_interior / (math.pi * ia * ib),
        "class_balance": n_interior / (n_feature + n_interior),
        "target": target,
    }
    return data


def make_two_scale_pairs_dataset(
    scale_ratio: float = 2.0,
    big_scale: float = 1.5,
    separation: float = 6.0,
    n_feature: int = 500,
    n_interior: int = 500,
    match_density: bool = True,
    reference_n: int = 2000,
    seed: int = 0,
) -> Data:
    """
    Two (curve, interior) pairs of different size in ONE dataset, side by side
    and far enough apart not to interact -- task F of experiment 5.

    Both pairs are task-relevant and are scored by the SAME model on the SAME
    field, so the comparison between them is within-model: seed variance drops
    out of the difference entirely, which is the strongest design available
    here (it is how task D of experiment 4 is built).

    ``match_density=True`` scales the point counts of the small pair with its
    perimeter and area, so the two pairs differ in SIZE only; with False they
    carry equal counts and the small pair is correspondingly denser -- the two
    readings of "smaller" that this task has to keep apart.
    """
    torch.manual_seed(seed)
    small_scale = big_scale / scale_ratio
    interior_fraction = 1.0 / 1.5  # exp1's 1.0 inside 1.5

    def one_pair(scale, centre, tag, n_curve, n_region):
        curve = RadialCurve2D(n_curve, lambda t, s=scale: torch.full_like(t, s), label=0)
        region = EllipseRegion2D(n_region, a=scale * interior_fraction, label=1, seed=seed)
        shift = torch.tensor(centre, dtype=torch.float32)
        curve.apply(torch.eye(2), shift)
        region.apply(torch.eye(2), shift)
        reference = RadialCurve2D(
            reference_n, lambda t, s=scale: torch.full_like(t, s), label=0
        ).get_data() + shift
        return curve, region, reference, tag

    if match_density:
        n_small_curve = max(8, int(round(n_feature / scale_ratio)))
        n_small_region = max(8, int(round(n_interior / scale_ratio ** 2)))
    else:
        n_small_curve, n_small_region = n_feature, n_interior

    pairs = [
        one_pair(big_scale, (-separation / 2, 0.0), "big", n_feature, n_interior),
        one_pair(small_scale, (separation / 2, 0.0), "small", n_small_curve, n_small_region),
    ]

    data = Data(2)
    for curve, region, _, tag in pairs:
        data.add(curve, name=f"curve_{tag}", is_feature=False)
        data.add(region, name=f"interior_{tag}", is_feature=False)
    for _, _, reference, tag in pairs:
        data.add_feature(
            name=f"ring_{tag}", points=reference,
            metadata={"feature_role": "target", "target_family": tag,
                      "source": "uniform_reference", "scale": float(
                          big_scale if tag == "big" else small_scale)},
        )
    return data
