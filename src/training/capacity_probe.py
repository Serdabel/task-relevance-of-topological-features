"""Где проходит граница решаемости в плоскости (ширина, глубина).

Шаг 0 эксперимента 2: «избыточная архитектура» определена только относительно
минимальной, а минимальная у двух датасетов разная, поэтому её надо измерить
до всего остального.

Две функции отвечают на два РАЗНЫХ вопроса, и различать их обязательно:

``frontier_scan``   -- лесенкой по ширине: при какой минимальной ширине задача
                       решается устойчиво.  Останавливается на первой ширине,
                       прошедшей порог: архитектуры выше границы решают задачу
                       по построению и не сообщают ничего.

``train_probe``     -- один прогон БЕЗ остановки на гейте, с полной трассой
                       test-accuracy и со снимком весов в первую эпоху, где
                       accuracy достигла цели.  Это то, чем провал «решения не
                       существует» отличается от провала «решение найдено и
                       потеряно»: у второго есть эпоха попадания и есть что
                       нарисовать.

Обучение здесь ничего не сохраняет в ``models/runs/`` -- это пробы, а не члены
ансамблей.  Результаты кладутся по одному CSV на ячейку, так что скан
возобновляем; ключ кэша включает число эпох, иначе прогон с другим бюджетом
молча подхватил бы чужие числа.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.mlp import MLP
from management.experiment_management import load_dataset_instance
from training.mass_training import (
    evaluate_binary_classifier_weighted,
    make_optimizer,
    resolve_activation_cls,
)

# Пара из эксперимента 1: одна и та же окружность, различается только то,
# нужна ли она для решения задачи.
DATASET_PATHS = {
    "irrelevant": "data/dataset_instances/simple_irrelevant/simple_irrelevant_0",
    "relevant": "data/dataset_instances/simple_relevant/simple_relevant_0",
}


def make_loaders(dataset_tag_or_path, batch_size=64):
    """(train_loader, test_loader, in_dim, X, y) для одного датасета."""
    path = DATASET_PATHS.get(dataset_tag_or_path, dataset_tag_or_path)
    loaded = load_dataset_instance(path, map_location="cpu", reconstruct_data=False)
    return (
        DataLoader(loaded["train_dataset"], batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(loaded["test_dataset"], batch_size=batch_size, shuffle=False, num_workers=0),
        loaded["X"].shape[1],
        loaded["X"],
        loaded["y"],
    )


def build_model(width, depth, in_dim=2, activation="tanh", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return MLP(
        in_dim=in_dim, hidden_dim=width, n_layers=depth, out_dim=1,
        activation_cls=resolve_activation_cls(activation), include_output_layer=True,
    )


def train_probe(
    model,
    train_loader,
    test_loader,
    optimizer,
    *,
    n_epochs,
    target_test_acc=1.0,
    patience=5,
    min_epochs=0,
    device="cpu",
    stop_at_gate=True,
    snapshot_first_hit=True,
):
    """Обучает одну сеть и возвращает, ГДЕ сработал бы гейт, а не только сработал ли.

    Гейт -- тот же, что в эксперименте 1: accuracy >= ``target_test_acc``
    ``patience`` эпох подряд после ``min_epochs``.

    ``stop_at_gate=False`` доводит обучение до конца ``n_epochs`` даже после
    срабатывания гейта -- нужно, чтобы увидеть, УДЕРЖИВАЕТСЯ ли решение.

    Возвращает dict; помимо обычных метрик там:
      ``first_hit_epoch``  первая эпоха, где accuracy достигла цели (или None);
      ``gate_epoch``       эпоха, где гейт сработал бы (или None);
      ``snapshot``         копия весов в ``first_hit_epoch`` (или None);
      ``test_accs``        полная трасса по эпохам.
    """
    device = torch.device(device)
    model.to(device)

    test_accs, train_accs = [], []
    first_hit_epoch, gate_epoch, snapshot = None, None, None
    good, best_acc = 0, -float("inf")

    for epoch in range(n_epochs):
        # шаг обучения повторяет train_one_mlp_until_accuracy дословно,
        # иначе числа отсюда несравнимы с числами основного пути
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, train_acc = evaluate_binary_classifier_weighted(model, train_loader, device)
        _, test_acc = evaluate_binary_classifier_weighted(model, test_loader, device)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)

        if test_acc >= target_test_acc and first_hit_epoch is None:
            first_hit_epoch = epoch
            if snapshot_first_hit:
                snapshot = deepcopy(model.state_dict())

        if epoch >= min_epochs and test_acc >= target_test_acc:
            good += 1
        else:
            good = 0

        if good >= patience and gate_epoch is None:
            gate_epoch = epoch
            if stop_at_gate:
                break

    return {
        "reached": gate_epoch is not None,
        "gate_epoch": gate_epoch,
        "first_hit_epoch": first_hit_epoch,
        "stopped_epoch": gate_epoch if gate_epoch is not None else len(test_accs) - 1,
        "best_test_acc": best_acc,
        "final_test_acc": test_accs[-1],
        "final_train_acc": train_accs[-1],
        # доля эпох ПОСЛЕ первого попадания, в которых accuracy держалась на цели:
        # это и есть «устойчиво ли найденное решение»
        "hold_fraction": (
            None if first_hit_epoch is None
            else sum(a >= target_test_acc for a in test_accs[first_hit_epoch:])
            / max(len(test_accs) - first_hit_epoch, 1)
        ),
        "n_epochs_run": len(test_accs),
        "test_accs": test_accs,
        "train_accs": train_accs,
        "snapshot": snapshot,
        "model": model,
    }


def _cell(tag, width, depth, seeds, n_epochs, gate, loaders, parts_dir, lr, activation):
    part = Path(parts_dir) / f"ep{n_epochs}_{tag}_w{width}_d{depth}.csv"
    if part.exists():
        return pd.read_csv(part)

    train_loader, test_loader, in_dim, _, _ = loaders
    rows = []
    for seed in range(seeds):
        model = build_model(width, depth, in_dim, activation, seed=1000 * seed + 7)
        opt = make_optimizer(model=model, optimizer_name="adam", lr=lr, weight_decay=0.0)
        m = train_probe(model, train_loader, test_loader, opt,
                        n_epochs=n_epochs, device="cpu", stop_at_gate=True,
                        snapshot_first_hit=False, **gate)
        rows.append({
            "dataset": tag, "width": width, "depth": depth, "n_neurons": width * depth,
            "seed": seed, "n_epochs_budget": n_epochs,
            "reached": m["reached"], "stopped_epoch": m["stopped_epoch"],
            "first_hit_epoch": m["first_hit_epoch"], "best_test_acc": m["best_test_acc"],
            "final_test_acc": m["final_test_acc"], "final_train_acc": m["final_train_acc"],
        })
    df = pd.DataFrame(rows)
    part.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(part, index=False)
    return df


def frontier_scan(
    datasets=("relevant", "irrelevant"),
    widths=(3, 4, 5, 6, 7, 8),
    depths=(1, 2, 3, 4, 5, 6),
    seeds=10,
    n_epochs=3000,
    threshold=0.8,
    target_test_acc=1.0,
    patience=5,
    min_epochs=0,
    lr=1e-3,
    batch_size=64,
    activation="tanh",
    parts_dir="analysis/results/experiments/exp2_arch_sweep/step0_parts",
    verbose=True,
):
    """Лесенкой по ширине для каждой глубины; стоп на первой ширине, прошедшей порог.

    Возвращает (frontier, per_seed): первая -- по строке на (датасет, глубина) с
    минимальной шириной, вторая -- по строке на сид для всех посчитанных ячеек.
    """
    gate = {"target_test_acc": target_test_acc, "patience": patience, "min_epochs": min_epochs}
    loaders = {t: make_loaders(t, batch_size) for t in datasets}

    per_seed, frontier = [], []
    for tag in datasets:
        for d in depths:
            minimal = None
            for w in widths:
                df = _cell(tag, w, d, seeds, n_epochs, gate, loaders[tag], parts_dir, lr, activation)
                per_seed.append(df)
                ok = int(df.reached.sum())
                if verbose:
                    print(f"{tag:10s} w{w} d{d}: {ok}/{len(df)} дошло, "
                          f"медиана эпохи {int(df.stopped_epoch.median())}", flush=True)
                if ok >= threshold * len(df):
                    minimal = w
                    break
            frontier.append({
                "dataset": tag, "depth": d, "min_width": minimal,
                "min_neurons": None if minimal is None else minimal * d,
                "n_epochs_budget": n_epochs,
                "note": "" if minimal else f"ни одна ширина <= {max(widths)}",
            })
            if verbose:
                print(f"  -> минимальная ширина при d={d}: {minimal}\n", flush=True)

    fr = pd.DataFrame(frontier)
    # явные nullable-типы: иначе колонка из одних None ловит FutureWarning при concat
    fr["min_width"] = fr["min_width"].astype("Int64")
    fr["min_neurons"] = fr["min_neurons"].astype("Int64")
    return fr, pd.concat(per_seed, ignore_index=True)


def frontier_table(per_seed, dataset, seeds=None):
    """Таблица «сколько сидов из N решили задачу» -- ширина по строкам, глубина по столбцам."""
    g = per_seed[per_seed.dataset == dataset]
    n = seeds or int(g.groupby(["width", "depth"]).size().max())
    piv = g.pivot_table(index="width", columns="depth", values="reached", aggfunc="sum")
    return piv.reindex(sorted(g.width.unique())).astype("Int64"), n


def frontier_rates(per_seed, dataset=None, conf=0.95):
    """Доля успеха по ячейкам с доверительным интервалом Уилсона.

    Жёсткий порог ``reached >= threshold * seeds`` в ``frontier_scan`` -- это
    удобство отображения, а НЕ оценка границы: он превращает шумную биномиальную
    долю в бинарный ответ, и при истинной p = 0.9 и десяти сидах ячейка не
    проходит порог 8/10 в 7 % случаев.  На шести глубинах это даёт рваную
    границу с вероятностью 0.35 -- чисто артефакт порога.  Для выводов о границе
    надо смотреть на долю с интервалом, а не на «прошло / не прошло».
    """
    from scipy import stats

    g = per_seed if dataset is None else per_seed[per_seed.dataset == dataset]
    out = []
    for (ds, w, d), block in g.groupby(["dataset", "width", "depth"]):
        k, n = int(block.reached.sum()), len(block)
        lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=conf, method="wilson")
        out.append({"dataset": ds, "width": w, "depth": d, "reached": k, "n": n,
                    "rate": k / n, "ci_lo": float(lo), "ci_hi": float(hi),
                    "median_gate_epoch": block[block.reached].stopped_epoch.median()})
    return pd.DataFrame(out).sort_values(["dataset", "width", "depth"]).reset_index(drop=True)


def homogeneity_test(per_seed, dataset, width, depths=None):
    """Отличаются ли доли успеха между глубинами при фиксированной ширине больше,
    чем позволяет биномиальный шум?  Возвращает (chi2, p, dof, pooled_rate)."""
    from scipy import stats
    import numpy as np

    g = per_seed[(per_seed.dataset == dataset) & (per_seed.width == width)]
    if depths is not None:
        g = g[g.depth.isin(depths)]
    t = g.groupby("depth").reached.agg(["sum", "size"])
    table = np.array([t["sum"].values, (t["size"] - t["sum"]).values])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return chi2, p, dof, t["sum"].sum() / t["size"].sum()
