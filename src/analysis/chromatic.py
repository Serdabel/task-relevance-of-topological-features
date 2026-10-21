"""Two-class (chromatic) TDA: six-packs of colored point clouds.

A DATA-SIDE (and representation-side) instrument, separate from witness
detection (PLAN.md §Two-class topology): it works on the colored point cloud,
not on neuron level sets, so it belongs neither to level-set sampling nor to
the criteria consistency table. Two roles:

  1. A.8 — principled ground truth for "task-relevant feature": a
     kernel/relative class of the chromatic complex (a cycle of one class
     filled by points of the other) replaces the hand-drawn feature set A
     wherever applicable.
  2. Experiment 15 — chromatic topology simplification through layers:
     six-packs of h_i(data) per hidden layer; the hypothesis is that the
     network must trivialize exactly the kernel part (separate the classes),
     layer by layer. Complementary to the neuron-level main line.

Backend: ``chromatic-tda`` (PyPI, IST Austria) — imported lazily; alpha
complexes support 2D/3D input only. Layer representations live in R^{width},
so ``layerwise_six_pack`` projects each representation to its top principal
components first (``analysis.geometry.fit_affine_subspace``) and records the
explained variance — the resolution of exp-15's open design point. A slice
that explains little variance is a weak measurement; treat its six-pack as
inconclusive, not as "trivialized".
"""

import torch

from analysis.geometry import fit_affine_subspace

# The backend names the two ends of the map "sub_complex" (the domain, H(K)) and
# "complex" (the codomain, H(L)); it returns no keys called domain/codomain, so
# reading those would silently give empty diagrams.
SIX_PACK_GROUPS = ("kernel", "image", "cokernel", "sub_complex", "complex",
                   "relative")


def _to_backend_inputs(X: torch.Tensor, y: torch.Tensor):
    assert X.ndim == 2, "X must have shape [N, dim]"
    assert X.shape[1] in (2, 3), (
        "chromatic-tda supports 2D/3D clouds; project higher-dimensional "
        "representations first (see layerwise_six_pack)"
    )
    assert y.ndim == 1 and len(y) == len(X), "y must have shape [N]"

    points = X.detach().cpu().double().numpy()
    labels = [int(v) for v in y.detach().cpu().tolist()]
    return points, labels


def _diagrams_to_tensors(six_pack_raw: dict, max_homology_dim: int) -> dict:
    diagrams = {}

    for group, per_dim in six_pack_raw.items():
        group_diagrams = {}
        for dim, bars in per_dim.items():
            dim = int(dim)
            if dim > max_homology_dim:
                continue
            bars_t = torch.tensor(
                [[float(b), float(d)] for b, d in bars], dtype=torch.float32
            ).reshape(-1, 2)
            group_diagrams[dim] = bars_t
        diagrams[group] = group_diagrams

    return diagrams


def chromatic_six_pack(
    X: torch.Tensor,
    y: torch.Tensor,
    sub_complex_class: int = 1,
    max_homology_dim: int = 1,
    **backend_kwargs,
) -> dict:
    """
    Chromatic alpha-complex six-pack of a two-class point cloud: persistence
    diagrams (kernel, image, cokernel, domain, codomain, relative) of the
    inclusion of the ``sub_complex_class`` subcomplex into the full complex.

    The part that matters for us is KERNEL (and relative): a cycle of the
    sub-complex class that dies early in the full complex because points of
    the OTHER class fill it — exactly a task-relevant feature (A.8).

    Returns dict:
        diagrams: {group: {dim: Tensor [n_bars, 2] (birth, death) in alpha
                   units; death=inf for essential classes}}
        meta: {"n_points", "class_sizes", "sub_complex_class"}
    """
    from chromatic_tda import ChromaticAlphaComplex

    points, labels = _to_backend_inputs(X, y)

    assert int(sub_complex_class) in set(labels), (
        f"sub_complex_class={sub_complex_class} not present among labels "
        f"{sorted(set(labels))}"
    )

    alpha = ChromaticAlphaComplex(points, labels, **backend_kwargs)
    scx = alpha.get_simplicial_complex(
        sub_complex=[[int(sub_complex_class)]],
        full_complex="all",
    )

    six_pack_raw = scx.bars_six_pack(return_as="dict")
    diagrams = _diagrams_to_tensors(six_pack_raw, max_homology_dim)

    class_sizes = {}
    for label in sorted(set(labels)):
        class_sizes[int(label)] = int(sum(1 for v in labels if v == label))

    return {
        "diagrams": diagrams,
        "meta": {
            "n_points": len(labels),
            "class_sizes": class_sizes,
            "sub_complex_class": int(sub_complex_class),
        },
    }


def _finite_persistence(bars: torch.Tensor) -> torch.Tensor:
    if len(bars) == 0:
        return torch.empty(0)
    pers = bars[:, 1] - bars[:, 0]
    return pers[torch.isfinite(pers)]


def kernel_task_relevant_features(
    X: torch.Tensor,
    y: torch.Tensor,
    min_persistence: float | None = None,
    sub_complex_class: int = 1,
    homology_dims: tuple[int, ...] = (1,),
    relative_threshold: float = 0.3,
    **kwargs,
) -> dict:
    """
    A.8: extracts the task-relevant features of a colored cloud as its
    persistent kernel classes (cycles of the sub-complex class obstructed by
    the other class). For datasets a/b/c this must reproduce the known
    answer: the filled ring in (c) is kernel-persistent, in (a) it is not —
    that is the calibration test for this function.

    A kernel class counts as task-relevant when its (finite) persistence is
    >= ``min_persistence``; with min_persistence=None the threshold is
    adaptive: ``relative_threshold`` x the largest finite kernel persistence
    over the requested dims (so the output is the set of DOMINANT classes;
    compare ``max_kernel_persistence`` across datasets for absolute claims).

    Representative cycles are not exposed by the backend — converting a
    kernel class back into a feature point set A needs a separate
    localization step (future work); until then use the class's birth scale
    together with the known dataset geometry.

    Returns dict:
        n_task_relevant: int
        classes: list of {dim, birth, death, persistence, group}
        max_kernel_persistence: float
        six_pack: full chromatic_six_pack output (for plots/inspection)
    """
    six = chromatic_six_pack(
        X,
        y,
        sub_complex_class=sub_complex_class,
        max_homology_dim=max(homology_dims),
        **kwargs,
    )

    kernel = six["diagrams"].get("kernel", {})

    all_pers = []
    for dim in homology_dims:
        all_pers.append(_finite_persistence(kernel.get(dim, torch.empty(0, 2))))
    all_pers = torch.cat(all_pers) if all_pers else torch.empty(0)

    max_kernel_persistence = float(all_pers.max().item()) if len(all_pers) else 0.0

    if min_persistence is None:
        threshold = relative_threshold * max_kernel_persistence
    else:
        threshold = float(min_persistence)

    classes = []
    for dim in homology_dims:
        bars = kernel.get(dim, torch.empty(0, 2))
        for b, d in bars.tolist():
            pers = d - b
            if not torch.isfinite(torch.tensor(pers)):
                continue
            if pers >= threshold and pers > 0:
                classes.append(
                    {
                        "dim": int(dim),
                        "birth": float(b),
                        "death": float(d),
                        "persistence": float(pers),
                        "group": "kernel",
                    }
                )

    classes.sort(key=lambda c: -c["persistence"])

    return {
        "n_task_relevant": len(classes),
        "classes": classes,
        "max_kernel_persistence": max_kernel_persistence,
        "threshold": threshold,
        "six_pack": six,
    }


def layerwise_six_pack(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    layer_ids: list[int] | None = None,
    kind: str = "post",
    sub_complex_class: int = 1,
    projection_dim: int = 2,
    max_homology_dim: int = 1,
    include_input: bool = True,
    device: str = "cpu",
    **kwargs,
) -> dict:
    """
    Experiment 15: six-packs of layer representations h_i(X) for each hidden
    layer (via ``MLP.forward_until``), tracking how the kernel part
    simplifies through the network.

    Representations in R^{width} are projected to their top
    ``projection_dim`` principal components (chromatic alpha complexes are
    2D/3D only); ``explained_variance`` is recorded per layer — a
    low-variance projection weakens that layer's measurement.

    Returns dict:
        six_packs: {"input" | layer_id: chromatic_six_pack output}
        summary: list of {layer, max_kernel_persistence,
                 total_kernel_persistence, explained_variance} — the exp-15
                 headline curve (kernel persistence should shrink with depth)
        kind, sub_complex_class
    """
    assert kind in {"pre", "post"}
    assert projection_dim in (2, 3)

    model = model.to(device)
    model.eval()

    X = X.to(device).float()
    y = y.detach().cpu()

    if layer_ids is None:
        layer_ids = list(range(model.n_layers))

    def _project(h: torch.Tensor):
        if h.shape[1] <= 3:
            return h, 1.0
        subspace = fit_affine_subspace(h, k=projection_dim)
        projected = (h - subspace["origin"]) @ subspace["basis"].T
        explained = float(
            subspace["explained_variance_ratio"][:projection_dim].sum().item()
        )
        return projected, explained

    def _kernel_stats(six):
        kernel = six["diagrams"].get("kernel", {})
        pers = _finite_persistence(kernel.get(1, torch.empty(0, 2)))
        return (
            float(pers.max().item()) if len(pers) else 0.0,
            float(pers.sum().item()) if len(pers) else 0.0,
        )

    six_packs = {}
    summary = []

    if include_input:
        input_2d, explained = _project(X.detach().cpu())
        six = chromatic_six_pack(
            input_2d, y, sub_complex_class=sub_complex_class,
            max_homology_dim=max_homology_dim, **kwargs,
        )
        six_packs["input"] = six
        max_p, total_p = _kernel_stats(six)
        summary.append(
            {
                "layer": "input",
                "max_kernel_persistence": max_p,
                "total_kernel_persistence": total_p,
                "explained_variance": explained,
            }
        )

    with torch.no_grad():
        for layer_id in layer_ids:
            h = model.forward_until(X, layer_id=layer_id, kind=kind).detach().cpu()
            h_low, explained = _project(h)

            six = chromatic_six_pack(
                h_low, y, sub_complex_class=sub_complex_class,
                max_homology_dim=max_homology_dim, **kwargs,
            )
            six_packs[layer_id] = six

            max_p, total_p = _kernel_stats(six)
            summary.append(
                {
                    "layer": layer_id,
                    "max_kernel_persistence": max_p,
                    "total_kernel_persistence": total_p,
                    "explained_variance": explained,
                }
            )

    return {
        "six_packs": six_packs,
        "summary": summary,
        "kind": kind,
        "sub_complex_class": int(sub_complex_class),
    }
