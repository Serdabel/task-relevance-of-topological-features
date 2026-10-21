"""Train one overtraining cell: N runs to n_epochs, every rung of every run stored.

The counterpart of ``scripts/train_cell.py`` for task E of experiment 2. One
process per (dataset, width, depth):

    python scripts/train_overtrain_cell.py --spec-json <spec.json> \
        --dataset simple_relevant --width 5 --depth 5 --n-models 50 \
        --n-epochs 3000 --multiples 1,3,10

The gate itself comes from the spec and is untouched -- it decides the FIRST
rung rather than the end of training, so the ``x1`` ensemble is directly
comparable with the pool's gate-stopped ensemble of the same architecture and
doubles as an independent replication of it.

Resume is by RUN, not by cell: every successful run writes all of its rungs
before the next run starts, so counting the ``end`` rung counts finished runs.
Re-running tops the cell up and costs only what is missing.

Parallel safety is the same as in train_cell.py: each rung owns its own
``{instance}_{tag}_model{id}`` name space, so the first-free-id scan inside
``store_model_instance`` never races between cells.
"""

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from experiments.configs import load_spec  # noqa: E402
from analysis.stat_aggregation import (  # noqa: E402
    dataset_instance_path,
    find_model_paths_for_dataset,
)
from training.overtraining import (  # noqa: E402
    DEFAULT_MULTIPLES,
    rung_tag,
    train_overtrained_ensemble,
)

# Every successful run produces this rung, so it is the honest run counter.
COUNTER_RUNG = "end"


def count_existing(dataset: str, tag: str, instance_id: int = 0) -> int:
    return len(find_model_paths_for_dataset(
        dataset_type=dataset, instance_id=instance_id, model_name=tag,
    ))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec-json", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--n-models", type=int, required=True,
                   help="сколько успешных ПРОГОНОВ нужно (не попыток)")
    p.add_argument("--n-epochs", type=int, default=3000,
                   help="докуда учить после гейта")
    p.add_argument("--multiples", default=",".join(f"{m:g}" for m in DEFAULT_MULTIPLES),
                   help="ступени лестницы в кратных СОБСТВЕННОГО гейта прогона, "
                        "через запятую; ступень 'end' добавляется всегда")
    p.add_argument("--activation", default=None,
                   help="по умолчанию берётся из спеки")
    p.add_argument("--model-name", default=None,
                   help="базовый тег; по умолчанию {act}_w{W}_d{D}")
    p.add_argument("--instance-id", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=100,
                   help="потолок числа попыток; перебор останавливается раньше, "
                        "как только набрано нужное число прогонов")
    p.add_argument("--success-rate", type=float, default=None,
                   help="устарело и игнорируется, см. --max-attempts")
    p.add_argument("--max-attempt-factor", type=float, default=None,
                   help="устарело и игнорируется, см. --max-attempts")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = load_spec(args.spec_json)
    activation = args.activation or spec.archs[0].activation
    base = args.model_name or f"{activation}_w{args.width}_d{args.depth}"
    multiples = tuple(float(x) for x in args.multiples.split(",") if x.strip())

    counter_tag = rung_tag(base, COUNTER_RUNG)
    have = count_existing(args.dataset, counter_tag, args.instance_id)
    need = args.n_models - have
    if need <= 0:
        print(f"[ot] {args.dataset} {base}: уже есть {have}/{args.n_models}, пропуск")
        return 0

    # A ceiling, not a plan: the loop stops as soon as `need` runs are stored.
    attempts = max(need, args.max_attempts)

    print(f"[ot] {args.dataset} {base}: есть {have}, нужно {args.n_models}, "
          f"добрать {need}, потолок попыток {attempts}, бюджет {args.n_epochs} эпох, "
          f"лестница {multiples} + end", flush=True)

    train_overtrained_ensemble(
        dataset_instance_path=dataset_instance_path(
            dataset_root="data/dataset_instances",
            dataset_type=args.dataset,
            instance_id=args.instance_id,
        ),
        n_models=attempts,
        activation=activation,
        hidden_dim=args.width,
        n_layers=args.depth,
        target_test_acc=spec.train.target_test_acc,
        min_epochs=spec.train.min_epochs,
        patience=spec.train.patience,
        n_epochs=args.n_epochs,
        checkpoint_multiples=multiples,
        batch_size=spec.train.batch_size,
        lr=spec.train.lr,
        optimizer_name=spec.train.optimizer_name,
        weight_decay=spec.train.weight_decay,
        device="cpu",
        model_name=base,
        stop_after_successful=need,
        verbose=False,
    )

    now = count_existing(args.dataset, counter_tag, args.instance_id)
    print(f"[ot] {args.dataset} {base}: стало {now}/{args.n_models}", flush=True)

    # Per-rung census: rungs above the gate collapse for slow runs (a run whose
    # gate is at 1861 has no x10 inside a 3000-epoch budget), so a smaller n on
    # an upper rung is a fact to report, not a failure to hide.
    for multiple in multiples:
        tag = rung_tag(base, f"x{multiple:g}")
        print(f"[ot]   ступень x{multiple:g}: "
              f"{count_existing(args.dataset, tag, args.instance_id)} моделей", flush=True)

    if now < args.n_models:
        print(f"[ot] ВНИМАНИЕ {args.dataset} {base}: недобор {now}/{args.n_models} "
              f"за {attempts} попыток. Потолок исчерпан — разберитесь, почему эта "
              f"конфигурация не обучается, прежде чем поднимать бюджет.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
