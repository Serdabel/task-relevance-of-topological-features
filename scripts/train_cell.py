"""Train one architecture cell on one dataset, resumably.

The unit of parallelism for stage 2, and the counterpart of
``scripts/eval_shard.py``. One process per (dataset, width, depth):

    python scripts/train_cell.py --spec-json <spec.json> \
        --dataset simple_relevant --width 5 --depth 5 --n-models 50

Resume: it counts the successful runs already stored under the cell's own
architecture tag and trains only the shortfall. Running it twice with the same
--n-models is a no-op; raising --n-models tops the ensemble up. Every successful
model is written to disk the moment it succeeds, so an interrupted run keeps
everything it had already earned and a re-run continues from there.

Two things about the underlying trainer that this script exists to handle.

``train_many_mlps_on_dataset_instance`` takes ATTEMPTS, not models: it makes
that many attempts and saves the ones that reach the gate. At a success rate
below 1 that silently returns a short ensemble.

The fix is not to predict the success rate but to stop counting attempts and
start counting successes: the budget is a CEILING (``--max-attempts``, 100 by
default) and training stops the moment the ensemble is full. A cell that
exhausts a hundred attempts without filling fifty slots is not a scheduling
problem to be tuned around -- it is telling you something about the task, and
it is reported loudly so a person looks at it.

Parallel safety is by construction, not by locking: every cell writes into its
own name space ``{instance}_tanh_w{W}_d{D}_model{id}``, so the first-free-id
scan inside ``store_model_instance`` (which does ``mkdir(exist_ok=False)``)
never races between cells. Within a cell training is sequential.

The training gate itself is untouched — it comes from the spec, which for
experiment 2 carries experiment 1's gate verbatim, so the ensembles remain
comparable with the reference cell.
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
from training.mass_training import train_many_mlps_on_dataset_instance  # noqa: E402


def count_existing(dataset: str, model_name: str, instance_id: int = 0) -> int:
    return len(find_model_paths_for_dataset(
        dataset_type=dataset, instance_id=instance_id, model_name=model_name,
    ))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec-json", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--n-models", type=int, required=True,
                   help="сколько успешных моделей нужно в ансамбле (не попыток)")
    p.add_argument("--activation", default=None,
                   help="по умолчанию берётся из спеки (spec.archs[0].activation)")
    p.add_argument("--model-name", default=None,
                   help="тег ансамбля; по умолчанию tanh_w{W}_d{D}. Явный тег "
                        "нужен, чтобы попадать в уже обученный ансамбль "
                        "эксперимента 1 (tanh_w5_d5_mass-train)")
    p.add_argument("--instance-id", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=100,
                   help="потолок числа попыток; перебор останавливается раньше, "
                        "как только ансамбль набран")
    p.add_argument("--success-rate", type=float, default=None,
                   help="устарело и игнорируется: бюджет попыток больше не "
                        "угадывается по доле успеха")
    p.add_argument("--max-attempt-factor", type=float, default=None,
                   help="устарело и игнорируется, см. --max-attempts")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = load_spec(args.spec_json)
    activation = args.activation or spec.archs[0].activation
    model_name = args.model_name or f"{activation}_w{args.width}_d{args.depth}"

    have = count_existing(args.dataset, model_name, args.instance_id)
    need = args.n_models - have
    if need <= 0:
        print(f"[cell] {args.dataset} {model_name}: уже есть {have}/{args.n_models}, пропуск")
        return 0

    # The budget is a ceiling, not a plan: training stops as soon as `need`
    # models have been saved, so a cell with a success rate of 1 costs exactly
    # `need` attempts and a difficult one is allowed up to --max-attempts.
    attempts = max(need, args.max_attempts)

    print(f"[cell] {args.dataset} {model_name}: есть {have}, нужно {args.n_models}, "
          f"добрать {need}, потолок попыток {attempts}", flush=True)

    train_many_mlps_on_dataset_instance(
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
        n_epochs=spec.train.n_epochs,
        min_epochs=spec.train.min_epochs,
        patience=spec.train.patience,
        batch_size=spec.train.batch_size,
        lr=spec.train.lr,
        optimizer_name=spec.train.optimizer_name,
        weight_decay=spec.train.weight_decay,
        device="cpu",
        save_successful_only=True,
        model_name=model_name,
        stop_after_successful=need,
        verbose=False,
        log_each_model=False,
    )

    now = count_existing(args.dataset, model_name, args.instance_id)
    print(f"[cell] {args.dataset} {model_name}: стало {now}/{args.n_models}", flush=True)

    if now < args.n_models:
        # Not an error: at the capacity floor a short ensemble is a RESULT, and
        # silently returning it as if it were full is what must not happen.
        # Exhausting the whole ceiling is a stronger statement still -- it says
        # the configuration itself is hard to train, and that needs a person.
        print(f"[cell] ВНИМАНИЕ {args.dataset} {model_name}: недобор "
              f"{now}/{args.n_models} за {attempts} попыток "
              f"(фактическая доля успеха {now - have}/{attempts} = "
              f"{(now - have) / attempts:.2f}). Потолок попыток исчерпан — это "
              f"утверждение о задаче, а не о расписании: разберитесь, почему "
              f"эта конфигурация не обучается, прежде чем поднимать бюджет.",
              flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
