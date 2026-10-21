"""Named experiment specs for the PLAN.md program.

The single place where "experiment N" is defined. Parameters below are DRAFT
values to review before each run — the structure (which recipes, which
architecture grids, which criteria) is the committed part. Specs whose
recipes/criteria are still scaffolds will fail at the corresponding runner
stage with the scaffold's NotImplementedError, which is the intended signal
of what to implement next.

Usage:
    from experiments.registry import EXPERIMENTS
    from experiments import runner
    spec = EXPERIMENTS["exp4_density_sweep"]
    runner.prepare_datasets(spec); runner.train(spec); runner.evaluate(spec)
"""

from dataclasses import replace

from experiments.configs import (
    ArchSpec,
    TrainSpec,
    EvalSpec,
    DatasetSpec,
    ExperimentSpec,
)
from analysis.null_placement import at

_DEFAULT_ARCH = ArchSpec(width=7, depth=5, activation="tanh")

# [RETIRED 2026-08-30] Density ratios of the ORIGINAL exp. 4 (ring : interior),
# PLAN.md 1:5 ... 5:1. Kept so stored results and provenance still resolve; do
# not start new work from it.
#
# Why it was retired: ring and interior carry OPPOSITE labels, so sweeping the
# ratio sweeps class balance from 0.125 to 0.625, exactly collinear with the
# manipulated variable (RESEARCH_PROGRAMME.md section 7 lists this as the open
# confound). No statistical remedy exists for a collinear confound, and every
# design remedy (weighted loss, balanced gate, compensating component) changes
# what "density effect" means. Experiment 4 now manipulates density by
# REDISTRIBUTING mass inside ONE class instead -- see _EXP4_* below.
DENSITY_RATIOS = [(200, 1000), (300, 900), (400, 800), (600, 600), (800, 400), (900, 300), (1000, 200)]

# ---- Experiment 4: density versus topology -------------------------------
#
# Density is varied by GEOMETRY at a constant point count (see
# data.dataset_recipes.make_density_ring_dataset): the circle becomes a ring of
# width w, the interior a disk of radius R. Class balance therefore never moves
# with density, and the support never degenerates.
#
# The grid: two tasks x three ring widths x three class balances, plus the disk
# radius on the task-relevant side, plus a scale control at five times the
# sample size, plus one anchor cell reproducing experiment 1's geometry.
_EXP4_R_MID = 1.5
_EXP4_W = (0.0, 0.5, 1.0)            # ширина кольца; 0 = окружность эксперимента 1
_EXP4_R_DISK = (0.25, 0.5)           # радиус диска, task-relevant
_EXP4_RHO = (0.25, 1.0, 4.0)         # баланс: точек кольца к точкам диска
_EXP4_T_MAIN, _EXP4_T_BIG = 1000, 5000

# Box and resolution. What transfers from experiment 1 is N_A = diam(A)/max(h)
# ~ 44 for the MIDDLE circle (diam 3.0), not the number 200: the box here has to
# hold the A.1 nulls, whose largest instance is a ring of outer radius 2.0
# placed at distance 4.5. Side 15.0 at resolution 222 gives h = 0.06787 and
# N_A = 44.20, matching experiment 2's 44.222 to within 0.05 %.
_EXP4_BOUNDS = ((-7.5, 7.5), (-7.5, 7.5))
_EXP4_RESOLUTION = 222


def _exp4_gap(w, R):
    return (_EXP4_R_MID - w / 2.0) - R


def _exp4_name(relevant, w, R, rho, T):
    tag = "rel" if relevant else "irr"
    return (f"d4_{tag}_w{int(round(w * 10)):02d}_R{int(round(R * 100)):03d}"
            f"_r{int(round(rho * 100)):03d}_T{T}")


def _exp4_datasets():
    """The 40 cells of experiment 4, in the order of the design notebook."""
    cells = []

    def add(relevant, w, R, rho, T):
        n_ring = int(round(T * rho / (1 + rho)))
        cells.append(DatasetSpec(
            name=_exp4_name(relevant, w, R, rho, T),
            recipe="make_density_ring_dataset",
            params={"relevant": relevant, "w": w, "disk_radius": R,
                    "n_ring": n_ring, "n_disk": T - n_ring, "seed": 0},
            target_feature="t_mid",       # первичная мишень; остальные три идут рядом
            stratify_split="component",
        ))

    # A: task-irrelevant, ширина x баланс, T = 1000 (9)
    for w in _EXP4_W:
        for rho in _EXP4_RHO:
            add(False, w, 1.0, rho, _EXP4_T_MAIN)

    # B: task-relevant, ширина x радиус диска x баланс, T = 1000 (18)
    for w in _EXP4_W:
        for R in _EXP4_R_DISK:
            for rho in _EXP4_RHO:
                add(True, w, R, rho, _EXP4_T_MAIN)

    # якорь: геометрия эксперимента 1 (1)
    add(True, 0.0, 1.0, 1.0, _EXP4_T_MAIN)

    # C.2: объём выборки x5, только ненулевые ширины (6 + 6)
    for w in (w for w in _EXP4_W if w > 0):
        for rho in _EXP4_RHO:
            add(False, w, 1.0, rho, _EXP4_T_BIG)
    for w in (w for w in _EXP4_W if w > 0):
        for rho in _EXP4_RHO:
            add(True, w, 0.5, rho, _EXP4_T_BIG)

    return tuple(cells)


# Training gate for the LINEARLY SEPARABLE half-plane task (exp. 1-2).
#
# The global TrainSpec default (min_epochs=0, patience=1, target_test_acc=
# 0.99) stops training at the first epoch whose test accuracy clears the
# gate. On a task a single hyperplane solves, that fires on epoch 1 with the
# loss still at ~ln 2 — the stored halfplane_base model0 has
# stopped_epoch=1, train_loss 0.696 -> 0.666, accuracy 1.0, i.e. weights
# essentially at initialization that happen to separate two well-spaced
# clusters. Witness statistics measured there describe a random network, not
# a trained one. The accuracy floor is therefore backed by an epoch floor.
_TRAIN_SEPARABLE = TrainSpec(
    n_models=50,
    target_test_acc=0.99,
    n_epochs=300,
    min_epochs=50,
    patience=5,
)

# Criteria for the 2D experiments: structural (the verdict) + extremum-
# anchored + homological. Tubular is off (EvalSpec default) — it emits a
# bare score no analysis can read.
#
# Criterion 3 matters here specifically: PROJECT.md names the free min over
# all levels c as criterion 2's weakness, and that weakness IS the incidental
# geometric effect exp. 2 sets out to measure. Criterion 4 adds the
# orthogonal read: same topology without the same geometry.
#
# backend="delaunay" is REQUIRED, not a tuning choice (AUDIT/05 §A.2). The
# homological criterion windows the field and then subsamples it to
# max_sample_points, which destroys the regularity of the grid it started
# from. On the resulting irregular cloud a radius graph (the "ripser" default)
# carries hundreds of essential classes; clipping their infinite death to
# min(f) hands each one the maximum persistence the diagram can hold, so the
# dominance test cannot pick a winner. Measured on a PLANTED witness:
#   ripser   -> 404 H_1 bars, top four 0.7531/0.7529/0.7526/0.7521, verdict False
#   delaunay ->   1 H_1 bar , persistence 0.4871,                   verdict True
# The Delaunay/alpha complex triangulates the convex hull, so H_k = 0 for
# k >= 1 in the full complex and no essential class can be manufactured.

# A.1 negative controls for the halfplane task, placed deliberately rather
# than drawn at random. Each is a congruent copy of the target circle (rigid
# motions are isometries, so shape, diameter and point count match by
# construction) put in a DIFFERENT REGIME of the trained field:
#
#   data: blue disk centre (-2.5, 0) r=1.5 (class 0); red circle centre
#   (+2.5, 0) r=1.5 (class 1, the target). |x| < 1 contains no data.
#
# Chosen rather than drawn, for one reason: COVERAGE. A uniform draw over the
# data bbox put every null in the same empty corridor between the two classes
# (AUDIT/08 section 5.1) — five draws from one place are one control, not five,
# and a floor estimated from them is not a floor for this problem. These five
# span the problem on purpose: on each class, in the gap, beside the target,
# and outside the data entirely.
#
# What a null owes the target is its SHAPE (rigid motions give that for free)
# and representative coverage. Nothing else: matching a control to its target
# on covariate after covariate ends at the target itself, where no control
# remains. The spread of the score over these five is therefore not a confound
# to remove but the WIDTH OF THE FLOOR, and it is what the target is read
# against.
#
# Edit freely — this is the experiment's control, and it is recorded in
# spec.json with the results. lab.set_null_placements() overrides it for
# exploration without touching this definition.
_NULLS_HALFPLANE = (
    at(-2.5, 0.0, name="on_blue_disk"),    # on data, opposite class
    at(-1.0, 0.0, name="disk_edge"),       # half on the disk, half in the gap
    at(0.0, 0.0, name="corridor"),         # empty gap, straddles the boundary
    at(2.5, 1.6, name="above_target"),     # just off the target, same side
    at(-7.0, 0.0, name="far_outside"),     # outside the data box entirely
)

_EVAL_2D_ALL_CRITERIA = EvalSpec(
    extra_criteria=("extremum", "homological"),
    extra_criteria_kwargs={"homological": {"backend": "delaunay"}},
    null_placements=_NULLS_HALFPLANE,
)

_EVAL_2D_STRUCT_HOMO = EvalSpec(
    extra_criteria=("homological",), # Экстремальный убрали, оставили только гомологический
    # max_sample_points > числа точек сетки в окне (при R=200 их ~6100), чтобы
    # подвыборки не было вовсе: иначе ответ зависит от одного фиксированного
    # розыгрыша seed=0, общего для всех моделей ансамбля. Вердикт от замены не
    # меняется (измерено), цена ~x1.6 по времени. Число привязано к боксу и
    # разрешению ЭТОГО эксперимента, в общий дефолт не выносится.
    extra_criteria_kwargs={"homological": {"backend": "delaunay",
                                           "max_sample_points": 7000}},
    null_placements=_NULLS_HALFPLANE,
)

# A.1 controls for the simple pair (experiment 2). TWO placements, not five.
#
# Experiment 1 measured what a null set owes the target and what it does not.
# Distance is NOT the covariate that orders the floor: far_right (3.5, 0) and
# above_target (0, 3.5) are the same circle at the same distance, related by a
# 90-degree rotation, and their floors differ by 47 %. What orders it is
# LEVEL-SET DIVERSITY — depth buys curvature everywhere except in the corridor
# between the classes, where every neuron computes the same separation.
#
# The guideline that follows (reports/exp1_sanity_check.md section 9): two
# placements, both AWAY from the decision boundary, in two different directions
# from the target — one along the task axis, one across it. The second is not a
# duplicate but the check that the soft direction was found. A third in the
# corridor buys only the WIDTH of the floor and never binds the verdict, so it
# is off here; two floors agreeing to within a few percent would be a coverage
# alarm, not a reassurance.
#
# Cost: three features per neuron instead of five, i.e. 40 % of the evaluation.
_NULLS_SIMPLE = (
    at(3.5, 0.0, name="far_right"),      # вдоль оси задачи; самое мягкое размещение (пол 0.1650)
    at(0.0, 3.5, name="above_target"),   # поперёк оси задачи, то же расстояние
)

# Experiment 2's architecture pool. The frontier of solvability measured in
# step 0 is a VERTICAL LINE at width 3 (= n+1 for input dimension 2, which is
# where Johnson 2019 puts it), not a staircase: depth buys trainability, not
# expressivity, so there is no width/depth trade-off to sample along. Excess
# therefore factorises into an independent width axis and depth axis, and the
# pool is a cross plus one diagonal.
#
# Widths are GEOMETRIC, not arithmetic: exp1's multiplicity law is a power law
# (N^-0.28) and PLAN's mechanism is multiplicative, and a power law is fitted
# from points evenly spaced in the logarithm.
#
# Which cell exists on which dataset is NOT expressible here — width 1 and 2 are
# unsolvable on simple_relevant, so no model is ever stored there. The notebook
# carries the (dataset, width, depth, n_models) pool; archs below is the union.
_ARCHS_EXP2 = (
    # ряд ширины при глубине 5 (глубина, на которой эксперимент 1 померил, что
    # детектор в рабочем диапазоне: 94 % свидетелей на последнем слое)
    ArchSpec(width=1, depth=5),    # мёртвая по теореме: линии уровня неограниченны
    ArchSpec(width=2, depth=5),    # то же
    ArchSpec(width=3, depth=5),    # минимальная ширина: избытка нет
    ArchSpec(width=5, depth=5),    # РЕФЕРЕНС, посчитан в эксперименте 1
    ArchSpec(width=10, depth=5),
    ArchSpec(width=20, depth=5),
    # ряд глубины при ширине 5 (нет отсева, есть избыток — см. отчёт)
    ArchSpec(width=5, depth=1),    # только слой 0: свидетель невозможен геометрически
    ArchSpec(width=5, depth=2),
    ArchSpec(width=5, depth=3),
    ArchSpec(width=5, depth=8),
    # ряд равного бюджета ~24 нейрона: ширина против глубины при постоянном
    # числе попыток
    ArchSpec(width=24, depth=1),
    ArchSpec(width=12, depth=2),
    ArchSpec(width=8, depth=3),
    ArchSpec(width=3, depth=8),
)


# Experiment 4's instrument: experiment 1's frozen configuration, re-expressed
# for this box. The A.1 placements are experiment 1's own two — every target
# here is a circle of radius 1.5, so the same rigid motions are congruent to all
# of them, and the floor stays comparable across the 26 cells and back to
# experiments 1-2. max_sample_points stays 7000 because N_A is unchanged; it is
# a function of the window in grid points, not of the box.
# Experiment 4's instrument: experiment 1's frozen configuration on this box.
#
# null_placements is EMPTY on purpose. Every target of this experiment carries
# its own nulls, because a null must be a rigid motion of the target it is the
# floor for, and the four targets differ in shape and size. Those nulls are
# deterministic translations recorded in the recipe, so they are registered into
# the instance itself and reach the evaluation as ordinary features -- the
# run-local provider, which can only serve one target, would not do.
_EVAL_EXP4 = EvalSpec(
    resolution=_EXP4_RESOLUTION,
    bounds=_EXP4_BOUNDS,
    extra_criteria=("homological",),
    extra_criteria_kwargs={"homological": {"backend": "delaunay",
                                           "max_sample_points": 7000}},
    null_placements=(),
)



# ---------------------------------------------------------------------------
# Эксперимент 5 — геометрический эффект
#
# 56 ячеек, и ни одна не написана здесь руками: они выводятся из четырёх правил
# в experiments/exp5_plan.py и там же нарисованы поштучно
# (notebooks/experiments/exp5_geometry_plan.ipynb). Список такой длины, набранный
# вручную, никто бы не проверил.
#
# Каждая ячейка несёт СВОЙ eval_overrides — бокс, разрешение, размещения нулей и
# геометрию внутренности, — потому что здесь меняется сама мишень. Разрешение
# подобрано так, чтобы диаметр мишени всегда укладывался в одно и то же число
# шагов сетки (N_A эксперимента 1); иначе вытянутые ячейки мерились бы более
# грубым инструментом, и свип частью мерил бы сетку.
#
# n_null_features=0 намеренно: нули не запекаются в инстанс, а ставятся из спеки.
# Случайное размещение здесь невозможно — мишень почти равна своему боксу, и
# конгруэнтной копии некуда лечь, чтобы выполнить контракт разделения.
def _exp5_dataset_specs():
    from experiments.exp5_plan import dataset_specs
    return dataset_specs()


_EXP5_EVAL = EvalSpec(
    kinds=("post",),
    run_structural=True,
    run_tubular=False,
    extra_criteria=("homological", "harmonic"),
    structural_kwargs={"level_eps": "adaptive", "threshold": 0.1, "activation": None},
    # bounds, resolution, null_placements и extra_criteria_kwargs приходят из
    # eval_overrides каждой ячейки — общих значений тут быть не может
)

_EXP5 = ExperimentSpec(
    id="exp5_geometry",
    title="Геометрический эффект: форма пары против формы линии уровня",
    plan_ref="PLAN.md — geometrical effect; дизайн в "
             "notebooks/experiments/exp5_geometry_sandbox.ipynb",
    datasets=tuple(_exp5_dataset_specs()),
    archs=(ArchSpec(width=5, depth=5, activation="tanh"),),
    train=TrainSpec(n_models=50, max_attempts=100, target_test_acc=1.0,
                    n_epochs=3000, patience=5),
    eval=_EXP5_EVAL,
    analyses=(),          # читалка geometry_sweep ещё не написана
    notes="56 ячеек = A 16 + A⊥ 9 + C 8 + D 23; 200 нулей; отклик — harmonic",
)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "exp5_geometry": _EXP5,
    # ---- The new proper first experiment -- consider it to be instead of old exp1 and calibration.
    "exp1_sanity_check": ExperimentSpec(
        id="exp1_sanity_check",
        title="Calibration / sanity check",
        plan_ref="PLAN.md exp. 0 (stage 1); reports/exp1_sanity_check.md",
        datasets=(
            DatasetSpec(
                name="simple_irrelevant",
                recipe="make_simple_dataset",
                params={"relevant": False},
                target_feature="irrelevant", # Взято из вывода ячейки твоего ноутбука
                n_null_features=0
            ),
            DatasetSpec(
                name="simple_relevant",
                recipe="make_simple_dataset",
                params={"relevant": True},
                target_feature="relevant", # Взято из вывода ячейки твоего ноутбука
                n_null_features=0
            ),
        ),
        archs=[ArchSpec(width=5, depth=5, activation='tanh')],
        train=TrainSpec(
            n_models=50,
            target_test_acc=1.0, # Как указано в твоем markdown: "to 100% accuracy"
            patience=5
        ),
        eval=_EVAL_2D_STRUCT_HOMO, # Включает структурный и гомологический критерии
        analyses=(
            "criterion_diagnostics",
            "witness_tables",
            "witness_rate_gap"
        ),
        notes=(
            "EXPERIMENT 1, run 2026-08-24. Write-up: reports/exp1_sanity_check.md; "
            "driver notebooks/experiments/exp1_sanity_check.ipynb (ensembles: "
            "exp1_sanity_check_mass.ipynb). Supersedes exp1_halfplane: the two "
            "datasets carry the BITWISE-SAME circle and the same null "
            "placements, so task relevance is the only difference, and all "
            "three rungs of CALIBRATION.md's scale come out of one run.\n"
            "Frozen here for every later experiment: resolution=200 (transfer "
            "it as N_A = diam(A)/max(h) ~ 40 per axis, not as 200), "
            "n_levels=101, level_eps='adaptive', tau=0.1, "
            "max_sample_points=7000. Unit of observation = the (model, feature) "
            "pair, valued by its BEST neuron; a rate at width W is read only "
            "against a floor measured at width W."
        ),
    ),
    # ---- Stage 1 of the research program: existence of the topological effect
    "exp1_halfplane": ExperimentSpec(
        id="exp1_halfplane",
        title="[RETIRED] Calibration: the detector's floor and chance level (halfplane)",
        plan_ref="CALIBRATION.md; PLAN.md exp. 1 (reclassified 2026-08-18)",
        datasets=(
            DatasetSpec(
                name="halfplane_base",
                recipe="make_halfplane_disk_circle_dataset",
                params={"seed": 0},
                target_feature="C_right",
                # 0 = do not persist A.1 nulls into the instance. They are a
                # CONTROL, not data, and are now placed explicitly per run via
                # analysis.null_placement (lab.set_null_placements, or
                # extra_features_fn on the evaluation stage), then recorded in
                # the eval-config sidecar.
                #
                # Persisted random nulls all landed in the same uninformative
                # place — the empty corridor between the classes (AUDIT/08
                # section 5.1). Five draws from one region are one control, not
                # five. Choosing the placements by hand buys coverage of the
                # problem, which is what a floor estimate needs.
                n_null_features=0,
            ),
            # density / shape variants are added as separate DatasetSpecs
            # once the base variant's results shape the sweep.
        ),
        archs=(_DEFAULT_ARCH,),
        train=_TRAIN_SEPARABLE,
        eval=_EVAL_2D_ALL_CRITERIA,
        analyses=(
            "criterion_diagnostics",
            "witness_tables",
            "witness_rate_gap",
            "rate_gap_by_criterion",
            "consistency_table",
        ),
        notes=(
            "Reclassified from experiment to CALIBRATION (2026-08-18). The id "
            "is unchanged so stored results and provenance still resolve. What "
            "this run produces is two of the three rungs of CALIBRATION.md's "
            "scale: C_right is a feature present in the data that the task does "
            "not need, i.e. the CHANCE LEVEL, and the A.1 nulls are the FLOOR. "
            "Every later experiment is read against those two numbers.\n"
            "Report the continuous statistic, not a count: at tau = 0.1 nothing "
            "crosses on the target or on any null, so the binary scale "
            "degenerates to 0 vs 0 and carries no information (CALIBRATION.md "
            "1.5, 3.1). Where a rate is still wanted, report "
            "rate[circle] - rate[null] with the rule of three attached (0 "
            "witnesses out of N bounds the true rate by 3/N: N=50 -> 0.06).\n"
            "RETIRED 2026-08-25. Experiment 1 is exp1_sanity_check, which runs "
            "the same calibration on a MATCHED PAIR of datasets carrying the "
            "bitwise-same circle (simple_irrelevant / simple_relevant), so task "
            "relevance is the only difference and all three rungs come out of "
            "one run; halfplane_base is only the irrelevant half of that pair. "
            "This spec is kept ONLY so ids in stored results and eval-config "
            "sidecars still resolve, and as the CLI mechanics example in "
            "docs/RUNBOOK.md section 7 -- do not start new work from it. All "
            "three notebooks of this line are in notebooks/legacy/ "
            "(exp1_halfplane_experiment, exp1_calibration, exp1_halfplane_mass)."
        ),
    ),
    "exp2_arch_sweep": ExperimentSpec(
        id="exp2_arch_sweep",
        title="Excessive architecture: witnessing vs width and depth",
        plan_ref="PLAN.md exp. 2; notebooks/experiments/exp2_arch_sweep.ipynb",
        datasets=(
            # The matched pair of experiment 1: the SAME circle in both, the
            # same null placements, only task relevance differs. Keeping them is
            # what lets exp2's numbers be read against a scale (tau, N_A ~ 40,
            # the floor, the chance level) measured on exactly this geometry.
            DatasetSpec(
                name="simple_irrelevant",
                recipe="make_simple_dataset",
                params={"relevant": False},
                target_feature="irrelevant",
                n_null_features=0,   # нули ставятся из спеки, а не хранятся в инстансе
            ),
            DatasetSpec(
                name="simple_relevant",
                recipe="make_simple_dataset",
                params={"relevant": True},
                target_feature="relevant",
                n_null_features=0,
            ),
        ),
        archs=_ARCHS_EXP2,
        # Experiment 1's gate, verbatim. Raising min_epochs would hide the
        # size/training-length confound rather than resolve it: on a linearly
        # separable task a wider net clears 100 % sooner (median stop 12 epochs
        # at w5 d5 against 53 at w1 d1), so width is confounded with how long
        # the net trained. The confound is instead carried into the reporting
        # via stopped_epoch, and RESOLVED by the notebook's task F, which varies
        # training length at fixed architecture.
        train=TrainSpec(
            n_models=50,
            target_test_acc=1.0,
            # 3000, not experiment 1's 300 -- and this does NOT weaken the gate,
            # because n_epochs is the CAP, not the acceptance rule. The rule is
            # unchanged: 100 % test accuracy held for `patience` epochs after
            # `min_epochs`. Every model experiment 1 accepted stopped at or
            # before epoch 219, so raising the cap changes nothing about them.
            #
            # It matters for the depth-1 cells on simple_relevant, which step 0
            # measured converging at a MEDIAN of 1907 epochs (w4 d1) and 2439
            # (w3 d1). At a 300-epoch cap those cells return 20-50 % of their
            # seeds, i.e. an ensemble selected for the atypical seeds that
            # happened to be fast -- which is a confound, not a saving. A cap
            # only costs time on genuine failures.
            n_epochs=3000,
            min_epochs=0,
            patience=5,
        ),
        # bounds is EXPLICIT and shared by both datasets. Left derived, the box
        # comes from each dataset's own data plus the null placements, and the
        # two datasets then end up on different grids at the same `resolution`:
        # dropping two of experiment 1's four placements shrank simple_relevant
        # (data at the origin) from side 13.5 to 7.5 while leaving
        # simple_irrelevant at 13.0, i.e. N_A = 79.6 against 45.9. Scores do not
        # compare across that.
        #
        # The value is experiment 1's own box, not the union of the current two
        # (which would be 13.0). Same cost, and it buys a real reproduction
        # test: the reference cell must return experiment 1's numbers.
        eval=replace(_EVAL_2D_STRUCT_HOMO,
                     null_placements=_NULLS_SIMPLE,
                     bounds=((-8.0, 5.5), (-2.0, 5.5))),
        analyses=(
            "criterion_diagnostics",
            "witness_tables",
            "witness_rate_gap",
            "rate_vs_width",
            "rate_gap_by_criterion",
            "consistency_table",
        ),
        notes=(
            "REDESIGNED 2026-08-27 onto the simple pair; the halfplane version "
            "was never run and its dataset is retired. The full plan lives in "
            "notebooks/experiments/exp2_arch_sweep.ipynb: step 0 (minimal "
            "architectures, measured) then tasks A-F.\n"
            "Minimal architectures from step 0: simple_relevant is width 3 at "
            "ANY depth (practical minimum w3 d2 = 6 neurons; w3 d1 costs 2439 "
            "epochs against 111), simple_irrelevant is w1 d1 = 1 neuron. The "
            "reference w5 d5 is therefore 8.3x excess in neurons for one task "
            "and 25x for the other -- the same cell means different excess, and "
            "that is the content of the axis.\n"
            "Width 1 and 2 exist on simple_irrelevant ONLY: on the relevant "
            "task no net of width <= 2 reaches 100 % at any depth (0 successes "
            "over 180 seeds), because every level set of a network whose widths "
            "are all <= input dimension is unbounded (Johnson, ICLR 2019), so a "
            "bounded decision region -- a disk inside a circle -- is "
            "unreachable. Those cells are the validity check the instrument "
            "must answer with exactly zero witnesses.\n"
            "Stage 2 and stage 3 run through training/parallel_train.py and "
            "analysis/parallel_eval.py: both are resumable and both shard "
            "across processes. Sequential cost of the pool is ~6 h of "
            "evaluation, ~16 min on 32 workers."
        ),
    ),
    "exp3_ring_separation": ExperimentSpec(
        id="exp3_ring_separation",
        title="[RETIRED] Task-relevant separation: single witness or distributed encoding",
        plan_ref="PLAN.md exp. 3",
        datasets=(
            DatasetSpec(
                name="ring_circle_disk",
                recipe="make_ring_interior_dataset",
                params={"seed": 0, "ring_shape": "circle", "interior_shape": "disk"},
                n_null_features=5,
            ),
            DatasetSpec(
                name="ring_ellipse_disk",
                recipe="make_ring_interior_dataset",
                params={"seed": 0, "ring_shape": "ellipse", "interior_shape": "disk"},
                n_null_features=5,
            ),
        ),
        archs=(
            _DEFAULT_ARCH,
            ArchSpec(width=31, depth=2),  # wide-shallow: distributed candidate
        ),
        analyses=("witness_tables", "layer_persistence", "sparse_probing"),
        notes=(
            "RETIRED 2026-08-25, kept only so stored ids resolve. simple_relevant "
            "(exp1_sanity_check) IS a ring-separation dataset -- a circle and its "
            "interior in opposite classes -- so the existence half is answered: "
            "49/50 models carry a witness and 41.9% of ALL neurons score under "
            "tau. The distributed-vs-localised half and the 'witness persists on "
            "layers > l' hypothesis fall out of exp2_arch_sweep's depth axis for "
            "free: at depth 1 a neuron's level sets are straight lines, so a "
            "single-neuron witness is geometrically impossible and the encoding "
            "must be distributed. The ring/ellipse SHAPE variants are not covered "
            "by exp1 and remain open -- they belong to PLAN.md item 5."
        ),
    ),
    # ---- Stage 2: properties of the effect
    "exp4_density_sweep": ExperimentSpec(
        id="exp4_density_sweep",
        title="Density against topology: ring width, disk radius, class balance",
        plan_ref="PLAN.md exp. 4; notebooks/experiments/exp4_design_walkthrough.ipynb",
        datasets=_exp4_datasets(),
        archs=(ArchSpec(width=5, depth=5, activation="tanh"),),
        train=TrainSpec(n_models=50, target_test_acc=1.0, patience=5),
        eval=_EVAL_EXP4,
        analyses=("witness_tables", "witness_rate_gap"),
        notes=(
            "40 cells. Density is varied by GEOMETRY at a constant point count: "
            "ring width w in {0, 0.5, 1} and disk radius R in {0.25, 0.5}, so "
            "class balance never moves with density and the support never "
            "degenerates; w = 0 reproduces experiment 1's circle. Class balance "
            "is a THIRD axis, identical in both tasks, which is what makes the "
            "task-relevant and task-irrelevant grids comparable.\n"
            "Four measurement targets per cell where w > 0 (the ring and the "
            "circles at its inner edge, midline and outer edge), one where w = 0 "
            "since they coincide there. Each target carries its own A.1 nulls, "
            "registered into the instance because they are deterministic rigid "
            "motions fully described by the recipe.\n"
            "The clearance between disk and ring is never below 0.5, and pairs "
            "of cells inside the grid share a clearance while differing in "
            "density -- (w=0.5, R=0.25) with (w=0, R=0.5) at 1.0, and "
            "(w=1, R=0.25) with (w=0.5, R=0.5) at 0.75 -- so the clearance is "
            "controlled without cells of its own. A matched-density pair falls "
            "out the same way: ring density 106.1 at (T=1000, n_ring=500, "
            "w=0.5) and at (T=5000, n_ring=1000, w=1.0).\n"
            "Server driver: notebooks/experiments/exp4_density_server.ipynb."
        ),
    ),
    "exp6_basin_ridge": ExperimentSpec(
        id="exp6_basin_ridge",
        title="Encoding strategy measured directly: basin vs ridge",
        plan_ref="PLAN.md exp. 6",
        datasets=(
            DatasetSpec(
                name="c_filled_opposite_relevant",
                recipe="make_single_motif_dataset",
                params={"disk_label": 0, "seed": 0},
            ),
        ),
        archs=(_DEFAULT_ARCH,),
        eval=EvalSpec(extra_criteria=("extremum",)),
        analyses=("witness_tables", "basin_ridge_typing", "seed_bimodality"),
        notes=(
            "Reuses stored c/c1 ensembles where the eval config matches; "
            "typing via analysis.extremum_anchored.classify_witness_basin_or_ridge."
        ),
    ),
    "exp7_multitask": ExperimentSpec(
        id="exp7_multitask",
        title="Neuron multitasking on multi-motif data (deep narrow net)",
        plan_ref="PLAN.md exp. 7",
        datasets=(
            DatasetSpec(
                name="multi_feature_motifs",
                recipe="make_multi_feature_motif_dataset",
                params={"seed": 0},
            ),
        ),
        archs=(ArchSpec(width=4, depth=8),),  # width below feature count
        analyses=("witness_tables", "feature_overlap", "ablation"),
    ),
    "exp8_abc_redo": ExperimentSpec(
        id="exp8_abc_redo",
        title="a/b/c + a1/b1/c1 redo with effect separation and controls",
        plan_ref="PLAN.md exp. 8",
        datasets=(
            DatasetSpec(name="a_empty_irrelevant", recipe="make_single_motif_dataset",
                        params={"disk_label": None, "seed": 0}, n_null_features=5),
            DatasetSpec(name="b_filled_same_irrelevant", recipe="make_single_motif_dataset",
                        params={"disk_label": 1, "seed": 0}, n_null_features=5),
            DatasetSpec(name="c_filled_opposite_relevant", recipe="make_single_motif_dataset",
                        params={"disk_label": 0, "seed": 0}, n_null_features=5),
            DatasetSpec(name="a1_empty_irrelevant", recipe="make_single_shifted_dataset",
                        params={"disk_label": None, "seed": 0}, n_null_features=5),
            DatasetSpec(name="b1_filled_same_irrelevant", recipe="make_single_shifted_dataset",
                        params={"disk_label": 1, "seed": 0}, n_null_features=5),
            DatasetSpec(name="c1_filled_opposite_relevant", recipe="make_single_shifted_dataset",
                        params={"disk_label": 0, "seed": 0}, n_null_features=5),
        ),
        archs=(_DEFAULT_ARCH,),
        eval=EvalSpec(extra_criteria=("extremum",)),
        analyses=("witness_tables", "witness_rate_gap", "consistency_table"),
        notes=(
            "The existing stored instances/ensembles predate null features; "
            "re-persisting with nulls means new instances and full retraining."
        ),
    ),
    # ---- Stage 3: higher dimensions
    #
    # Shared hi-dim eval design (exp 9-10): the per-neuron field is a shared
    # random cloud (field_source="random" — a grid is exponentially dead in
    # R^4), the grid-structural criterion is off, and the witness verdict
    # comes from the homological criterion on the DELAUNAY complex (a radius
    # graph on a random volumetric sample carries spurious sample-void bars;
    # the Delaunay lower-star filtration does not — verified on synthetic
    # ridge/plane controls in both R^3 and R^4). The sampled-structural
    # criterion (gradient level-set sampling) is the geometric cross-check;
    # nulls are "shuffled" copies of the target so they match its topology,
    # not just its diameter.
    "exp9_spheres": ExperimentSpec(
        id="exp9_spheres",
        title="S^2 in R^3 and S^3 in R^4: 2D results in codim 1",
        plan_ref="PLAN.md exp. 9",
        datasets=(
            DatasetSpec(name="sphere_shell_3d", recipe="make_sphere_shell_dataset",
                        params={"dim": 3, "seed": 0}, target_feature="S_mid",
                        n_null_features=3, null_kind="shuffled"),
            DatasetSpec(name="sphere_shell_4d", recipe="make_sphere_shell_dataset",
                        params={"dim": 4, "seed": 0}, target_feature="S_mid",
                        n_null_features=3, null_kind="shuffled",
                        # R^4 box needs a denser cloud for the same coverage.
                        eval_overrides={"n_field_points": 60000}),
        ),
        archs=(ArchSpec(width=10, depth=5),),
        eval=EvalSpec(
            run_structural=False,  # grid path is dead in R^3+/R^4
            field_source="random",
            n_field_points=30000,
            extra_criteria=("homological", "sampled_structural"),
            extra_criteria_kwargs={
                # "codim1" resolves to {ambient_dim - 1: 1} per dataset:
                # H_2 for S^2 in R^3, H_3 for S^3 in R^4.
                "homological": {
                    "feature_homology": "codim1",
                    "backend": "delaunay",
                    "max_sample_points": 2000,
                },
                "sampled_structural": {
                    "n_levels": 9,
                    "n_points_per_level": 2048,
                    "sampler_kwargs": {"n_steps": 250},
                },
            },
            primary_witness_column="homological_is_witness",
        ),
        analyses=("witness_tables", "witness_rate_gap", "consistency_table"),
        notes=(
            "Codim-1 shells: S^{n-1} matches the codimension of a neuron "
            "level set — the natural entry into d >= 3. Null placement is "
            "tight (centers within +-1 of the target's), so rate[null] is a "
            "conservative baseline."
        ),
    ),
    "exp10_linked": ExperimentSpec(
        id="exp10_linked",
        title="Codim >= 2: linked circles in R^3 vs R^4 (ambient dependence)",
        plan_ref="PLAN.md exp. 10",
        datasets=(
            DatasetSpec(name="linked_circles_r3", recipe="linked_circles_3d",
                        params={"n_points": 600}, target_feature="C_1",
                        n_null_features=3, null_kind="shuffled"),
            DatasetSpec(name="linked_circles_r4", recipe="linked_circles_4d",
                        params={"n_points": 600, "seed": 0},
                        target_feature="C_1",
                        n_null_features=3, null_kind="shuffled",
                        eval_overrides={"n_field_points": 60000}),
        ),
        archs=(ArchSpec(width=10, depth=5),),
        eval=EvalSpec(
            run_structural=False,
            field_source="random",
            n_field_points=30000,
            extra_criteria=("homological", "sampled_structural"),
            extra_criteria_kwargs={
                # A circle keeps H_1 = 1 in any ambient dimension.
                "homological": {
                    "feature_homology": {1: 1},
                    "backend": "delaunay",
                    "max_sample_points": 2000,
                },
                "sampled_structural": {
                    "n_levels": 9,
                    "n_points_per_level": 2048,
                    "sampler_kwargs": {"n_steps": 250},
                },
            },
            primary_witness_column="homological_is_witness",
        ),
        analyses=("witness_tables", "witness_rate_gap", "consistency_table"),
        notes=(
            "The provable core: a level-set component of one smooth neuron "
            "has codim 1, so a codim >= 2 circle can only be witnessed "
            "through a tubular-neighborhood boundary or an intersection of "
            ">= codim level sets — compare witness structure between the "
            "SAME configuration linked in R^3 and unlinkable in R^4 "
            "(linked_circles_r4 is an isometric embedding of _r3). "
            "Multi-neuron intersection analysis is future work."
        ),
    ),
    # Experiments 5, 11-15 get specs when their scaffolds are implemented:
    # exp5 (incidental geometry knobs), exp11 (d>=10 + PCA slices),
    # exp12 (semi-real), exp13 (wavy scale), exp14 (noise),
    # exp15 (chromatic layers).
}


# ----------------------------------------------------------------------
# exp2_extended: the same instrument, extra cells
# ----------------------------------------------------------------------
#
# Task E (overtraining) plus the cells experiment 2 left open: w40 d5, w5 d10,
# and w5 d5 on sigmoid and relu. Everything about HOW a neuron is scored is
# inherited verbatim from exp2_arch_sweep -- the same box ((-8, 5.5), (-2, 5.5))
# giving N_A = 44.222, resolution 200, n_levels 101, level_eps "adaptive",
# structural + homological(delaunay, 7000), the same two null placements, the
# same gate. `replace` rather than a re-typed literal is deliberate: a re-typed
# copy can drift, and a drifted EvalSpec makes the new numbers incomparable
# with the pool without saying so. The notebook asserts the config fingerprints
# of the two specs are equal before anything runs.
#
# It needs its own id, and therefore its own results dir, for one blunt reason:
# `arch_sweep.load_neuron_stats` globs neuron_stats_*.csv, and an overtrained
# rung or a relu ensemble dropped into the pool's directory would silently join
# the "w5 d5" cell of tasks A, B and C.
EXPERIMENTS["exp2_extended"] = replace(
    EXPERIMENTS["exp2_arch_sweep"],
    id="exp2_extended",
    title="Experiment 2, extra cells: overtraining, w40 d5, w5 d10, other activations",
    plan_ref=("PLAN.md exp. 2, tasks D-E; "
              "notebooks/experiments/exp2_extended_server.ipynb"),
    notes=(
        "Server-side companion to exp2_arch_sweep. Eval and train specs are "
        "inherited unchanged, so every number produced here is directly "
        "comparable with the pool's.\n"
        "Three blocks, all on both datasets:\n"
        "(E) overtraining. Five cells -- w3 d2 (the smallest network that "
        "solves the task reliably AND has a neuron capable of witnessing), "
        "w3 d5 (narrowest cell studied; also where the head witnesses only "
        "82 % of the time and where the continuity filter's single relevant "
        "flag lives), w24 d1 (INSTRUMENT CONTROL: no witness is possible by "
        "theorem, so any score movement there is the criterion's baseline "
        "moving, not the representation), w5 d5 (reference, doubles as a "
        "replication) and w3 d8. The last three are the equal-budget row, so E "
        "is the equal-budget comparison unrolled in training time.\n"
        "Each run is trained once to 3000 epochs with rungs at 1x, 3x and 10x "
        "ITS OWN gate epoch plus the end; gates differ tenfold between cells, "
        "so an absolute epoch would land before the gate in one net and far "
        "past it in another. The rungs come from ONE run because training sets "
        "no seed: a second run is a different trajectory, not a longer one, "
        "and only a within-model difference removes seed variance from the "
        "duration effect.\n"
        "(arch) w40 d5 separates the two readings of task A's width trend "
        "(power law against acceleration; four points cannot). w5 d10 tests "
        "two predictions stated before it was run: the saturating-profile "
        "model of C6 gives a witness fraction of ~0.64-0.66 against 0.578 at "
        "d = 8, and the half-rise rule of C4 puts the breakpoint at layer "
        "sqrt(10) ~ 3.2.\n"
        "(activation) sigmoid at w5 d5 is an INSTRUMENT CONTROL, not a "
        "generalisation test: sigmoid(z) = (tanh(z/2)+1)/2 spans the same "
        "function class, so a difference in the answer is evidence about the "
        "criterion. relu is the real generalisation test -- a different, "
        "non-smooth class the project's Morse framing does not cover. Note "
        "that the inherited EvalSpec pins activation=None, i.e. 101 levels "
        "spaced linearly over the OBSERVED range; for relu fields "
        "(non-negative, heavy-tailed) criterions.make_candidate_levels also "
        "offers an exponential grid. The linear grid is kept for "
        "comparability, as a declared choice rather than a default."
    ),
)
