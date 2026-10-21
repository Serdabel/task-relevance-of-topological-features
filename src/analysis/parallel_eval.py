"""Stage 3 in parallel: plan shards, run them as processes, merge the parts.

The criteria are computed once per (dataset, model, neuron, feature) and nothing
in that grid depends on anything else in it, so the only reason stage 3 took
hours was that it ran in one process. This module parallelises it WITHOUT
touching a line inside the criteria: every shard runs the same
``collect_compact_neuron_statistics`` on a slice of the model list, and the
merged result is compared row-for-row against a sequential run in the
acceptance test.

Three design points worth knowing before editing:

**A shard is a slice of models, not a whole architecture.** Cells differ in cost
by ~20x (w20 d5 is 300 rows per model, w5 d1 is 15), so sharding by cell means
waiting for the slowest one. The slice size is chosen from a target row count,
which makes every shard cost roughly the same wall-clock.

**Shards are OS processes, not multiprocessing workers.** Forking after torch
has started its OpenMP pool hangs; "spawn" from a notebook cannot pickle the A.1
null provider, which is a closure. ``scripts/eval_shard.py`` sidesteps both and
is restartable on its own. The pool here is a *thread* pool, because the threads
only wait on subprocesses.

**Resume is by MODEL, not by shard name.** One part CSV per shard, written under
a .partial name and renamed on completion, so a part exists only when it is
complete. Planning then reads the ``model_path`` column of the parts already
there and builds shards only over the models nobody has covered yet.

It used to be a file test on the shard's name, and that was wrong as soon as an
ensemble was topped up. The name encodes the shard's START INDEX in the model
list, and that list is ``sorted()`` over directory names, i.e. LEXICOGRAPHIC:
model0, model1, model10, ..., model2, ... Adding models 37-49 to a cell that had
0-36 both reorders the list (model37 sorts before model4) and leaves the old
trailing part covering fewer models than its slice now spans. Either way the old
part is skipped by name while standing for a different set of models than it
holds. Keying on the models themselves cannot go wrong that way.

Typical use (see notebooks/experiments/exp2_arch_sweep.ipynb)::

    from experiments.registry import EXPERIMENTS
    from analysis.parallel_eval import evaluate_parallel

    frames = evaluate_parallel(EXPERIMENTS["exp2_arch_sweep"], CELLS, n_jobs=32)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from analysis.eval_config import save_eval_config
from analysis.stat_aggregation import find_model_paths_for_dataset
from experiments.configs import save_spec

# Cost of one (neuron, feature) row at ONE thread, the regime shards run in,
# on the real evaluation path: resolution 200, structural + homological
# (delaunay, max_sample_points=7000).
#
# 3.4 s is MEASURED ON A FULL RUN: experiment 2's pool, 109 050 rows, took
# 5.2 hours on 20 cores. An earlier small probe gave 386 ms and is kept here
# only as a warning -- it understated the real cost about ninefold, because a
# probe of a few neurons misses the expensive tail (level sweeps whose
# components are large). Estimates printed from a probe-sized measurement are
# not to be trusted; this constant comes from the whole thing.
SECONDS_PER_ROW = 3.4

SHARD_SCRIPT = "scripts/eval_shard.py"


@dataclass(frozen=True)
class Cell:
    """One (dataset, architecture) cell of the pool.

    ``model_name`` exists because a stored ensemble is found by its TRAINING
    tag, and experiment 1's reference ensembles were saved as
    ``tanh_w5_d5_mass-train`` rather than ``tanh_w5_d5``. Pinning the tag is
    what lets experiment 2 read the reference cell as the very ensemble
    experiment 1 measured, instead of retraining a lookalike.

    ``n_models`` is the ensemble size: training tops up to it, evaluation caps
    at it. A cell that already has that many models trains nothing.

    ``activation`` enters the default tag, so ``relu_w5_d5`` and ``tanh_w5_d5``
    are different ensembles of the same architecture and never collide -- on
    disk, in the part files, or in the merged CSV.
    """

    dataset: str
    width: int
    depth: int
    n_models: int = 50
    model_name: str | None = None
    activation: str = "tanh"

    @property
    def tag(self) -> str:
        return self.model_name or f"{self.activation}_w{self.width}_d{self.depth}"

    @property
    def n_neurons(self) -> int:
        return self.width * self.depth


def normalize_cells(cells) -> list[Cell]:
    """Accepts Cell, dict, or (dataset, width, depth[, n_models[, tag]])."""
    out = []
    for c in cells:
        if isinstance(c, Cell):
            out.append(c)
        elif isinstance(c, dict):
            out.append(Cell(**c))
        else:
            out.append(Cell(*c))
    return out


@dataclass(frozen=True)
class Shard:
    """One unit of work: some models of one architecture on one dataset."""

    dataset: str
    model_name: str
    model_paths: tuple[str, ...]
    part_csv: Path
    n_features: int
    n_neurons: int

    @property
    def n_rows(self) -> int:
        return len(self.model_paths) * self.n_neurons * self.n_features

    @property
    def est_seconds(self) -> float:
        return self.n_rows * SECONDS_PER_ROW


def config_fingerprint(spec, dataset: str | None = None) -> str:
    """Eight hex characters identifying the evaluation configuration.

    Part files are named per (dataset, architecture, model slice) and carry
    nothing about HOW they were computed, so a changed evaluation parameter
    would leave the planner treating stale parts as done and the merge quietly
    mixing two configurations. Parts therefore live under a directory named by
    this fingerprint: a changed parameter opens a fresh name space, old parts
    are neither reused nor destroyed.

    Everything that changes a number goes in; nothing that does not.
    """
    import hashlib
    import json

    eval_spec = spec.eval
    if dataset is not None:
        ds = next((d for d in spec.datasets if d.name == dataset), None)
        if ds is not None and ds.eval_overrides:
            eval_spec = replace(eval_spec, **ds.eval_overrides)

    payload = {
        "resolution": eval_spec.resolution,
        "bounds": eval_spec.bounds,
        "bounds_margin": eval_spec.bounds_margin,
        "kinds": list(eval_spec.kinds),
        "layer_ids": None if eval_spec.layer_ids is None else list(eval_spec.layer_ids),
        "run_structural": eval_spec.run_structural,
        "run_tubular": eval_spec.run_tubular,
        "structural_kwargs": eval_spec.structural_kwargs,
        "tubular_kwargs": eval_spec.tubular_kwargs,
        "extra_criteria": list(eval_spec.extra_criteria),
        "extra_criteria_kwargs": eval_spec.extra_criteria_kwargs,
        "field_source": eval_spec.field_source,
        "n_field_points": eval_spec.n_field_points,
        "field_seed": eval_spec.field_seed,
        "null_placements": [
            (p.name, tuple(float(v) for v in p.vector))
            for p in getattr(eval_spec, "null_placements", ())
        ],
        "train_gate": (spec.train.target_test_acc, spec.train.min_epochs,
                       spec.train.patience),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def parts_dir_for(spec, parts_dir=None) -> Path:
    """<results>/eval_parts/<fingerprint>/ — see config_fingerprint."""
    root = Path(parts_dir) if parts_dir else Path(spec.results_dir) / "eval_parts"
    return root / config_fingerprint(spec)


def _n_features(spec, dataset: str, instance_id: int = 0) -> int:
    """Rows per neuron: every feature the evaluation will actually score.

    Counted from the STORED INSTANCE when it exists, because that is the ground
    truth: an experiment may register several targets and their nulls into the
    instance itself (experiment 4 registers four targets with their own nulls,
    12-16 features per cell), and the spec's single ``target_feature`` would
    then under-count by an order of magnitude. Under-counting here does not
    corrupt results -- it only makes shards that much larger than
    ``target_rows`` asked for, which costs parallelism and resume granularity.

    Falls back to the spec's own arithmetic when the instance is not on disk
    yet, so planning still works before the data stage has run.
    """
    ds = next(d for d in spec.datasets if d.name == dataset)
    eval_spec = spec.eval
    if ds.eval_overrides:
        eval_spec = replace(eval_spec, **ds.eval_overrides)
    n_run_local = len(getattr(eval_spec, "null_placements", ()))

    try:
        from management.experiment_management import load_dataset_instance
        stored = load_dataset_instance(
            f"data/dataset_instances/{dataset}/{dataset}_{instance_id}",
            map_location="cpu", reconstruct_data=False,
        )
        n_stored = len(stored.get("features", {}) or {})
        if n_stored:
            return n_stored + n_run_local
    except Exception:
        pass

    return (1 if ds.target_feature else 0) + n_run_local


def covered_models(parts_dir: Path, dataset: str, model_name: str) -> set[str]:
    """Model paths already present in the parts of one (dataset, architecture).

    Reads one column, so scanning a whole parts directory is cheap.
    """
    covered: set[str] = set()
    for part in sorted(parts_dir.glob(f"{dataset}__{model_name}__m*.csv")):
        try:
            covered.update(pd.read_csv(part, usecols=["model_path"])["model_path"])
        except Exception as exc:  # a truncated part is not a covered part
            print(f"  [план] ВНИМАНИЕ: часть {part.name} не читается ({exc}); "
                  f"её модели будут посчитаны заново")
    return covered


def _next_part_index(parts_dir: Path, dataset: str, model_name: str) -> int:
    """First shard number not yet used by this (dataset, architecture)."""
    used = set()
    for part in parts_dir.glob(f"{dataset}__{model_name}__m*.csv"):
        try:
            used.add(int(part.stem.rsplit("__m", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(used) + 1 if used else 0


def coverage_report(
    spec,
    cells,
    parts_dir: str | Path | None = None,
    instance_id: int = 0,
    runs_root: str = "models/runs",
) -> pd.DataFrame:
    """What is trained, what is already evaluated, what is left — per cell.

    Read this before re-running an evaluation over topped-up ensembles: it says
    exactly how much is reused and how much is new, without computing anything.
    """
    parts_dir = parts_dir_for(spec, parts_dir)
    rows = []
    for cell in normalize_cells(cells):
        paths = find_model_paths_for_dataset(
            dataset_type=cell.dataset, instance_id=instance_id,
            runs_root=runs_root, model_name=cell.tag,
        )[:cell.n_models]
        covered = covered_models(parts_dir, cell.dataset, cell.tag) if parts_dir.exists() else set()
        todo = [p for p in paths if p not in covered]
        rows.append({
            "dataset": cell.dataset, "model_name": cell.tag,
            "обучено": len(paths), "нужно": cell.n_models,
            "посчитано": len(covered & set(paths)), "к расчёту": len(todo),
            # Parts may hold models the current cell no longer asks for (an
            # ensemble rebuilt, or trimmed by n_models). merge_parts drops them.
            "лишних в частях": len(covered - set(paths)),
        })
    return pd.DataFrame(rows)


def plan_shards(
    spec,
    cells,
    target_rows: int = 800,
    parts_dir: str | Path | None = None,
    instance_id: int = 0,
    runs_root: str = "models/runs",
    verbose: bool = True,
) -> list[Shard]:
    """Enumerate the shards still to compute.

    ``cells`` is the pool as the notebook defines it (see ``normalize_cells``),
    NOT spec.archs, because an architecture may exist on one dataset and not the
    other: width 1-2 is unsolvable on the relevant task, so no model is ever
    stored there.

    Shards whose part CSV already exists are skipped: that is the resume.
    """
    cells = normalize_cells(cells)
    parts_dir = parts_dir_for(spec, parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[план] отпечаток конфигурации: {parts_dir.name}  ({parts_dir})")

    shards: list[Shard] = []
    done_rows = missing_cells = 0

    for cell in cells:
        paths = find_model_paths_for_dataset(
            dataset_type=cell.dataset, instance_id=instance_id,
            runs_root=runs_root, model_name=cell.tag,
        )[:cell.n_models]
        if not paths:
            missing_cells += 1
            if verbose:
                print(f"  [план] {cell.dataset} {cell.tag}: обученных моделей нет, пропуск")
            continue

        n_feat = _n_features(spec, cell.dataset)
        per_model = cell.n_neurons * n_feat
        models_per_shard = max(1, target_rows // max(per_model, 1))

        # Resume: whatever the parts already hold is done, whoever computed it
        # and under whatever slicing (see the module docstring).
        covered = covered_models(parts_dir, cell.dataset, cell.tag)
        todo = [p for p in paths if p not in covered]
        done_rows += (len(paths) - len(todo)) * per_model

        index = _next_part_index(parts_dir, cell.dataset, cell.tag)
        for start in range(0, len(todo), models_per_shard):
            chunk = tuple(todo[start:start + models_per_shard])
            part = parts_dir / f"{cell.dataset}__{cell.tag}__m{index:04d}.csv"
            index += 1
            shards.append(
                Shard(cell.dataset, cell.tag, chunk, part, n_feat, cell.n_neurons))

    if verbose:
        rows = sum(s.n_rows for s in shards)
        seq_h = rows * SECONDS_PER_ROW / 3600
        print(f"\n[план] шардов к расчёту: {len(shards)}, строк {rows}")
        if done_rows:
            print(f"[план] уже посчитано: {done_rows} строк — эти шарды пропущены")
        if missing_cells:
            print(f"[план] ячеек без моделей: {missing_cells}")
        print(f"[план] оценка: {seq_h:.2f} ч последовательно")
        for jobs in (8, 16, 32):
            print(f"        на {jobs:2d} воркерах ~ {seq_h * 60 / jobs:.0f} мин")
    return shards


def run_shards(
    shards,
    n_jobs: int | None = None,
    python: str | None = None,
    spec_json: str | Path | None = None,
    device: str = "cpu",
    instance_id: int = 0,
    verbose: bool = True,
) -> list[Shard]:
    """Run shards as subprocesses; returns the ones that FAILED.

    A failing shard is reported and does not abort the rest — on a run of a
    hundred shards, one bad cell should not cost the other ninety-nine. The
    caller decides what to do with the returned list; ``merge_parts`` will
    refuse to merge while parts are missing.
    """
    if not shards:
        return []

    n_jobs = n_jobs or min(32, (os.cpu_count() or 4))
    python = python or sys.executable
    assert spec_json is not None, "spec_json обязателен: воркер читает спеку с диска"

    total = len(shards)
    failed: list[Shard] = []
    t0 = time.time()

    # Longest shards first: with a fixed number of workers this is what keeps
    # the tail short.
    ordered = sorted(shards, key=lambda s: -s.n_rows)

    def _run(shard: Shard) -> tuple[Shard, int, str]:
        listing = shard.part_csv.with_suffix(".models.txt")
        listing.write_text("\n".join(shard.model_paths) + "\n")
        cmd = [
            python, SHARD_SCRIPT,
            "--spec-json", str(spec_json),
            "--dataset", shard.dataset,
            "--model-name", shard.model_name,
            "--model-paths-file", str(listing),
            "--out", str(shard.part_csv),
            "--device", device,
            "--instance-id", str(instance_id),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return shard, proc.returncode, (proc.stderr or "")[-2000:]

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_run, s) for s in ordered]
        for done, fut in enumerate(as_completed(futures), start=1):
            shard, code, err = fut.result()
            if code != 0:
                failed.append(shard)
                print(f"  [ОШИБКА] {shard.part_csv.name} (код {code})\n{err}")
            elif verbose:
                el = time.time() - t0
                eta = el / done * (total - done)
                print(f"  [{done:3d}/{total}] {shard.part_csv.name} | "
                      f"прошло {el/60:5.1f} мин | осталось ~{eta/60:5.1f} мин", flush=True)

    if verbose:
        print(f"\n[шарды] готово за {(time.time()-t0)/60:.1f} мин, "
              f"успешно {total - len(failed)}/{total}")
    return failed


SORT_KEYS = ["dataset_type", "model_path", "feature_name", "layer_id", "coord_id"]


def merge_parts(
    spec,
    cells,
    parts_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    instance_id: int = 0,
    runs_root: str = "models/runs",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Merge part CSVs into the layout runner.evaluate produces.

    One ``neuron_stats_{model_name}.csv`` per architecture plus its eval-config
    sidecar, which is exactly what ``runner.aggregate`` globs — so stage 4 needs
    no change at all.

    Rows are sorted before writing, so the file does not depend on the order in
    which shards happened to finish.
    """
    parts_dir = parts_dir_for(spec, parts_dir)
    results_dir = Path(results_dir) if results_dir else Path(spec.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cells = normalize_cells(cells)
    by_arch: dict[str, list[pd.DataFrame]] = {}
    expected: dict[str, int] = {}
    wanted: dict[str, set[str]] = {}
    missing: list[str] = []

    for cell in cells:
        dataset, model_name = cell.dataset, cell.tag
        paths = find_model_paths_for_dataset(
            dataset_type=dataset, instance_id=instance_id,
            runs_root=runs_root, model_name=model_name,
        )[:cell.n_models]
        if not paths:
            continue
        expected[model_name] = expected.get(model_name, 0) + (
            len(paths) * cell.n_neurons * _n_features(spec, dataset)
        )
        wanted.setdefault(model_name, set()).update(paths)
        found = sorted(parts_dir.glob(f"{dataset}__{model_name}__m*.csv"))
        if not found:
            missing.append(f"{dataset} {model_name}")
            continue
        for part in found:
            # feature_role round-trips through CSV only when read verbatim:
            # pandas turns a bare `null` cell into NaN (CLAUDE.md gotcha).
            by_arch.setdefault(model_name, []).append(
                pd.read_csv(part, converters={"feature_role": lambda v: v})
            )

    assert not missing, (
        "нет ни одной части для: " + ", ".join(missing) +
        " — сначала досчитайте шарды (run_shards), потом сливайте"
    )

    frames: dict[str, pd.DataFrame] = {}
    for model_name, parts in by_arch.items():
        df = pd.concat(parts, ignore_index=True)

        # Parts accumulate across runs, so they can hold models this pool no
        # longer asks for (an ensemble rebuilt, or trimmed by n_models) and, if
        # a model was covered twice under different slicings, the same row
        # twice. Neither is an error worth failing on; both are dropped here,
        # and the count assert below still guarantees full coverage.
        if "model_path" in df.columns:
            before = len(df)
            df = df[df["model_path"].isin(wanted[model_name])]
            if len(df) < before and verbose:
                print(f"[слияние] {model_name}: отброшено {before - len(df)} строк "
                      f"моделей вне текущего пула")

        keys = [k for k in SORT_KEYS if k in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=keys) if keys else df.drop_duplicates()
        if len(df) < before and verbose:
            print(f"[слияние] {model_name}: отброшено {before - len(df)} строк-дубликатов")

        df = df.sort_values(keys, kind="mergesort").reset_index(drop=True)

        got, want = len(df), expected[model_name]
        assert got == want, (
            f"{model_name}: строк {got}, ожидалось {want} "
            f"(= моделей x нейронов x фич). Части неполны или список моделей "
            f"изменился между расчётом и слиянием."
        )

        out = results_dir / f"neuron_stats_{model_name}.csv"
        df.to_csv(out, index=False)

        # Sidecar: reuse the one a shard already wrote (all shards of an
        # architecture share their evaluation parameters by construction) and
        # record that the file was assembled from parts.
        part_sidecars = sorted(parts_dir.glob(f"*__{model_name}__m*.eval_config.json"))
        assert part_sidecars, (
            f"{model_name}: у частей нет eval-config сайдкара. Результат без "
            "записи параметров оценки непригоден (analysis/eval_config.py), "
            "пересчитайте части."
        )
        import json
        cfg = json.loads(part_sidecars[0].read_text())
        cfg.setdefault("params", {})["merged_from_parts"] = len(parts)
        cfg["params"]["config_fingerprint"] = parts_dir.name
        cfg["params"]["output_csv_path"] = str(out)
        save_eval_config(cfg, out)

        frames[model_name] = df
        if verbose:
            print(f"  [слияние] {out.name}: {got} строк из {len(parts)} частей")

    return frames


def evaluate_parallel(
    spec,
    cells,
    n_jobs: int | None = None,
    target_rows: int = 800,
    device: str = "cpu",
    instance_id: int = 0,
    runs_root: str = "models/runs",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """plan -> run -> merge, resumable at every step.

    Re-running after an interruption recomputes only the shards whose part CSV
    is missing.
    """
    spec_json = save_spec(spec)

    shards = plan_shards(
        spec, cells, target_rows=target_rows, instance_id=instance_id,
        runs_root=runs_root, verbose=verbose,
    )
    failed = run_shards(
        shards, n_jobs=n_jobs, spec_json=spec_json, device=device,
        instance_id=instance_id, verbose=verbose,
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} шард(ов) упало: "
            f"{[s.part_csv.name for s in failed]}. Прогон возобновляем — "
            "исправьте причину и вызовите evaluate_parallel снова, "
            "посчитанные части не пересчитываются."
        )

    return merge_parts(
        spec, cells, instance_id=instance_id, runs_root=runs_root, verbose=verbose,
    )
