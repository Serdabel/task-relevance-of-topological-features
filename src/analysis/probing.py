"""Sparse probing baseline: the non-topological reference detector.

A.4 (PLAN.md) / experiment 3: Gurnee et al.'s sparse probing is the standard
answer to "which neurons encode feature X". A 1-sparse probe is the linear
analog of a single-neuron witness; k-sparse probes give the
localized-vs-distributed reference: a feature with no single-neuron witness
but a good small-k probe is distributively encoded in a way the topological
criteria should agree with.

Backend: scikit-learn (installed in the working env); imported inside
functions so the core pipeline has no sklearn import-time dependency.
"""

from itertools import combinations

import torch

from analysis.geometry import nearest_distances_to_set


def collect_activation_dataset(
    model,
    X: torch.Tensor,
    layer_ids: list[int] | None = None,
    kinds: tuple[str, ...] = ("post",),
    device: str = "cpu",
) -> dict:
    """
    Hidden activations for probing: {(layer_id, kind): Tensor [n, width]}
    via ``MLP.forward_until`` (no grad).
    """
    assert all(kind in {"pre", "post"} for kind in kinds)

    model = model.to(device)
    model.eval()

    X = X.to(device).float()

    if layer_ids is None:
        layer_ids = list(range(model.n_layers))

    activations = {}

    with torch.no_grad():
        for layer_id in layer_ids:
            for kind in kinds:
                activations[(layer_id, kind)] = model.forward_until(
                    X, layer_id=layer_id, kind=kind
                ).detach().cpu()

    return activations


def feature_membership_targets(
    X: torch.Tensor,
    feature_points: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """
    Binary probe targets: 1 for points within ``delta`` of the feature set A,
    0 otherwise.
    """
    assert delta > 0, "delta must be positive"

    distances = nearest_distances_to_set(
        X.float(), feature_points.to(X.device).float()
    )
    return (distances <= delta).long()


def _cv_auc(H_np, t_np, columns, cv_folds: int, seed: int):
    """Cross-validated AUC of a logistic probe on the selected columns."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score

    Hs = H_np[:, list(columns)]

    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    proba = cross_val_predict(clf, Hs, t_np, cv=cv, method="predict_proba")[:, 1]

    auc = float(roc_auc_score(t_np, proba))
    acc = float(accuracy_score(t_np, (proba > 0.5).astype(int)))
    return auc, acc


def fit_sparse_probe(
    H: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
    cv_folds: int = 5,
    seed: int = 0,
    max_exhaustive_subsets: int = 300,
) -> dict:
    """
    k-sparse linear probe on activations H [n, width].

      - k=1: per-neuron logistic probes; reports every neuron's AUC and the
        best one (the direct comparison object for a single-neuron witness);
      - k>1: exhaustive subsets when C(width, k) <= max_exhaustive_subsets
        (exact — feasible at the project's widths), otherwise an
        L1-regularized C-path picks <= k surviving coefficients and the
        selected subset is re-scored;
      - all scores are cross-validated (StratifiedKFold), never train-set.

    Returns dict:
        selected_neurons: list[int]
        cv_auc, cv_accuracy
        per_neuron_auc: list[width] (k=1 diagnostics; [] for k>1)
        k, method
    """
    import numpy as np

    assert H.ndim == 2 and targets.ndim == 1 and len(H) == len(targets)
    assert k >= 1

    H_np = H.detach().cpu().numpy()
    t_np = targets.detach().cpu().numpy().astype(int)
    width = H_np.shape[1]

    assert len(np.unique(t_np)) == 2, (
        "probe targets are single-class; adjust delta so both classes occur"
    )

    if k == 1:
        per_neuron_auc = []
        for j in range(width):
            auc, _ = _cv_auc(H_np, t_np, (j,), cv_folds, seed)
            per_neuron_auc.append(auc)

        # A neuron separating in the negative direction is equally localized.
        effective = [max(a, 1.0 - a) for a in per_neuron_auc]
        best = int(np.argmax(effective))
        best_auc, best_acc = _cv_auc(H_np, t_np, (best,), cv_folds, seed)

        return {
            "selected_neurons": [best],
            "cv_auc": max(best_auc, 1.0 - best_auc),
            "cv_accuracy": best_acc,
            "per_neuron_auc": per_neuron_auc,
            "k": 1,
            "method": "per-neuron",
        }

    n_subsets = 1
    for i in range(k):
        n_subsets = n_subsets * (width - i) // (i + 1)

    if n_subsets <= max_exhaustive_subsets:
        best_subset, best_auc, best_acc = None, -1.0, float("nan")

        for subset in combinations(range(width), k):
            auc, acc = _cv_auc(H_np, t_np, subset, cv_folds, seed)
            score = max(auc, 1.0 - auc)
            if score > best_auc:
                best_subset, best_auc, best_acc = subset, score, acc

        return {
            "selected_neurons": list(best_subset),
            "cv_auc": best_auc,
            "cv_accuracy": best_acc,
            "per_neuron_auc": [],
            "k": k,
            "method": "exhaustive",
        }

    # L1 path: largest C whose solution keeps <= k coefficients.
    from sklearn.linear_model import LogisticRegression

    selected = None
    for C in np.logspace(2, -4, 25):
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=2000
        )
        clf.fit(H_np, t_np)
        nonzero = np.flatnonzero(np.abs(clf.coef_[0]) > 1e-8)
        if 1 <= len(nonzero) <= k:
            selected = nonzero
            break

    if selected is None:
        selected = np.argsort(-np.abs(clf.coef_[0]))[:k]

    auc, acc = _cv_auc(H_np, t_np, tuple(int(j) for j in selected), cv_folds, seed)

    return {
        "selected_neurons": [int(j) for j in selected],
        "cv_auc": max(auc, 1.0 - auc),
        "cv_accuracy": acc,
        "per_neuron_auc": [],
        "k": k,
        "method": "l1-path",
    }


def probe_vs_witness_table(
    probe_results: dict,
    neuron_table,
    dataset_type: str,
    feature_name: str,
    witness_column: str = "is_structural_witness",
    kind: str = "post",
) -> dict:
    """
    Experiment-3 deliverable: agreement between probe-selected neurons and
    criterion witnesses per (model, layer). Disagreement diagnoses
    distributed/nonlinear encoding missed by one of the two languages.

    probe_results:
        {(model_path, layer_id): fit_sparse_probe output} — as produced by
        looping collect_activation_dataset + fit_sparse_probe over stored
        models (see experiments.runner.aggregate, analysis "sparse_probing").

    neuron_table:
        per-neuron stats DataFrame (or CSV path) with the witness column.

    Returns dict:
        agreement: DataFrame [model_path, layer_id, probe_neurons,
                   witness_neurons, n_overlap, jaccard, probe_cv_auc]
        summary: {mean_jaccard, n_rows, mean_probe_auc}
    """
    import numpy as np
    import pandas as pd

    from analysis.witness_stats import _load_stats_source, _coerce_bool_series

    df = _load_stats_source(neuron_table)
    df = df[
        (df["dataset_type"] == dataset_type)
        & (df["feature_name"] == feature_name)
        & (df["kind"] == kind)
    ].copy()

    assert len(df) > 0, "no matching rows in the neuron table"

    df["_witness"] = _coerce_bool_series(df[witness_column])

    rows = []

    for (model_path, layer_id), probe in probe_results.items():
        g = df[(df["model_path"] == model_path) & (df["layer_id"] == layer_id)]
        if len(g) == 0:
            continue

        witness_neurons = sorted(g[g["_witness"]]["coord_id"].astype(int).tolist())
        probe_neurons = sorted(int(j) for j in probe["selected_neurons"])

        overlap = set(witness_neurons) & set(probe_neurons)
        union = set(witness_neurons) | set(probe_neurons)

        rows.append(
            {
                "model_path": model_path,
                "layer_id": layer_id,
                "probe_neurons": probe_neurons,
                "witness_neurons": witness_neurons,
                "n_overlap": len(overlap),
                "jaccard": len(overlap) / len(union) if union else float("nan"),
                "probe_cv_auc": probe["cv_auc"],
            }
        )

    agreement = pd.DataFrame(rows)
    assert len(agreement) > 0, "no (model, layer) pairs matched between inputs"

    return {
        "agreement": agreement,
        "summary": {
            "mean_jaccard": float(np.nanmean(agreement["jaccard"])),
            "n_rows": int(len(agreement)),
            "mean_probe_auc": float(agreement["probe_cv_auc"].mean()),
        },
    }
