"""One shard of the criteria evaluation: some models of one architecture.

The unit of parallelism for stage 3. A shard is (dataset, architecture, a slice
of that architecture's model list); ``analysis.parallel_eval`` plans the slices
and runs many of these at once, one OS process each.

    python scripts/eval_shard.py \
        --spec-json analysis/results/experiments/exp2_arch_sweep/spec.json \
        --dataset simple_relevant --model-name tanh_w5_d5 \
        --model-paths-file <file with one model path per line> \
        --out analysis/results/experiments/exp2_arch_sweep/eval_parts/<name>.csv

Why a separate PROCESS rather than a multiprocessing worker: forking after
torch has started its OpenMP pool is a known source of hangs, and "spawn" from a
notebook requires everything crossing the process boundary to be picklable —
including the A.1 null provider, which is a closure. A subprocess is immune to
both, behaves identically in a notebook and in a terminal, and each shard is
independently restartable.

Resume rule: if --out already exists the script exits 0 without computing
anything. That, plus one part file per shard, is the whole resume mechanism.

The evaluation itself is the untouched sequential path
(``collect_compact_neuron_statistics``) with an explicit model list, so a
sharded run and a sequential run compute identical rows.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

# Must precede the torch import: each shard is one core's worth of work, and a
# BLAS pool per shard would oversubscribe the machine 32x over.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from experiments.configs import load_spec  # noqa: E402
from experiments import runner  # noqa: E402
from analysis.stat_aggregation import collect_compact_neuron_statistics  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec-json", required=True,
                   help="spec.json written by experiments.configs.save_spec")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-name", required=True,
                   help="architecture tag, e.g. tanh_w5_d5")
    p.add_argument("--model-paths-file", required=True,
                   help="text file, one stored model directory per line")
    p.add_argument("--out", required=True, help="part CSV to write")
    p.add_argument("--device", default="cpu")
    p.add_argument("--instance-id", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)

    if out.exists():
        print(f"[shard] {out.name}: уже посчитан, пропускаю")
        return 0

    spec = load_spec(args.spec_json)

    model_paths = [
        line.strip()
        for line in Path(args.model_paths_file).read_text().splitlines()
        if line.strip()
    ]
    assert model_paths, f"пустой список моделей: {args.model_paths_file}"

    # Per-dataset EvalSpec overrides, exactly as runner.evaluate resolves them.
    eval_spec = spec.eval
    ds_spec = next((d for d in spec.datasets if d.name == args.dataset), None)
    assert ds_spec is not None, (
        f"спека {spec.id} не содержит датасета {args.dataset!r}; "
        f"есть: {[d.name for d in spec.datasets]}"
    )
    if ds_spec.eval_overrides:
        from dataclasses import replace
        eval_spec = replace(eval_spec, **ds_spec.eval_overrides)

    tmp = out.with_suffix(".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    collect_compact_neuron_statistics(
        dataset_types=[args.dataset],
        instance_id=args.instance_id,
        resolution=eval_spec.resolution,
        bounds_margin=eval_spec.bounds_margin,
        bounds_override=eval_spec.bounds,
        layer_ids=(None if eval_spec.layer_ids is None
                   else list(eval_spec.layer_ids)),
        kinds=eval_spec.kinds,
        device=args.device,
        structural_kwargs=dict(eval_spec.structural_kwargs),
        tubular_kwargs=dict(eval_spec.tubular_kwargs),
        run_structural=eval_spec.run_structural,
        run_tubular=eval_spec.run_tubular,
        field_source=eval_spec.field_source,
        n_field_points=eval_spec.n_field_points,
        field_seed=eval_spec.field_seed,
        extra_criteria=runner._resolve_extra_criteria(eval_spec) or None,
        extra_features_fn=runner._resolve_null_features_fn(eval_spec),
        model_name_filter=args.model_name,
        model_paths_by_dataset={args.dataset: model_paths},
        require_train_gate={
            "target_test_acc": spec.train.target_test_acc,
            "min_epochs": spec.train.min_epochs,
            "patience": spec.train.patience,
        },
        output_csv_path=tmp,
        # The shard is small by construction; rewriting its CSV after every
        # (model, feature) buys nothing that the part file does not already.
        save_partial=False,
        verbose=False,
    )

    # Rename last: a part file exists only when it is complete, so an
    # interrupted shard is retried rather than silently trusted. The sidecar
    # needs no rename — eval_config_path_for_csv keys off the STEM, which
    # ".partial" and ".csv" share, so it is already at its final name.
    tmp.replace(out)

    print(f"[shard] {out.name}: готово ({len(model_paths)} моделей)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
