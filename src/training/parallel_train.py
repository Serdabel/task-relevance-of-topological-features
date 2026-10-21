"""Stage 2 in parallel: one process per (dataset, architecture) cell.

Thin driver over ``scripts/train_cell.py``, the same shape as
``analysis.parallel_eval`` is over ``scripts/eval_shard.py`` — a thread pool
whose threads only wait on subprocesses.

Training is the cheap half (about half an hour sequentially for experiment 2's
whole pool against six hours of evaluation), so this exists less for speed than
for two other reasons: cells are trained under one resumable command, and a
cell that comes up short says so instead of quietly returning a small ensemble.

Race safety is by construction — see the note in ``scripts/train_cell.py``:
each cell owns its own ``{instance}_tanh_w{W}_d{D}_model{id}`` name space, so
the first-free-id scan in ``store_model_instance`` never collides between cells.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from analysis.parallel_eval import normalize_cells
from analysis.stat_aggregation import find_model_paths_for_dataset
from experiments.configs import save_spec

TRAIN_SCRIPT = "scripts/train_cell.py"
OVERTRAIN_SCRIPT = "scripts/train_overtrain_cell.py"


def ensemble_table(cells, instance_id: int = 0) -> pd.DataFrame:
    """What is on disk right now, per (dataset, architecture) cell.

    ``cells`` is the pool in any form ``analysis.parallel_eval.normalize_cells``
    accepts. A cell with a pinned ``model_name`` (the experiment 1 reference,
    tagged ``tanh_w5_d5_mass-train``) is counted under that tag, so it shows as
    already complete and is never retrained.
    """
    rows = []
    for cell in normalize_cells(cells):
        have = len(find_model_paths_for_dataset(
            dataset_type=cell.dataset, instance_id=instance_id, model_name=cell.tag,
        ))
        rows.append({
            "dataset": cell.dataset, "width": cell.width, "depth": cell.depth,
            "n_neurons": cell.n_neurons, "model_name": cell.tag,
            "have": have, "need": cell.n_models,
            "missing": max(0, cell.n_models - have),
        })
    return pd.DataFrame(rows)


def train_pool(
    spec,
    cells,
    n_jobs: int | None = None,
    python: str | None = None,
    instance_id: int = 0,
    max_attempts: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every cell that is short, in parallel; returns the table after.

    Idempotent at two levels, which is what makes this safe to re-run on a
    server without supervision: a cell already at its target is skipped inside
    ``train_cell.py``, and inside a cell every successful model is written the
    moment it succeeds, so an interrupted run keeps what it earned.

    ``max_attempts`` is a CEILING per cell, not a plan: training stops as soon
    as the ensemble is full, and a cell that burns the whole ceiling without
    filling up says so loudly instead of quietly returning a short ensemble.
    That case wants a person, not a bigger budget.
    """
    spec_json = save_spec(spec)
    n_jobs = n_jobs or min(32, (os.cpu_count() or 4))
    python = python or sys.executable

    cells = normalize_cells(cells)
    before = ensemble_table(cells, instance_id)
    short = {(r.dataset, r.model_name) for r in before.itertuples() if r.missing > 0}
    todo = [c for c in cells if (c.dataset, c.tag) in short]

    if verbose:
        print(before.to_string(index=False))
        print(f"\n[training] cells to train: {len(todo)} of {len(cells)}, "
              f"workers {n_jobs}, attempt ceiling per cell {max_attempts}")

    if not todo:
        return before

    def _run(cell):
        cmd = [
            python, TRAIN_SCRIPT,
            "--spec-json", str(spec_json),
            "--dataset", cell.dataset,
            "--width", str(cell.width), "--depth", str(cell.depth),
            "--n-models", str(cell.n_models),
            "--max-attempts", str(max_attempts),
            "--activation", cell.activation,
            "--model-name", cell.tag,
            "--instance-id", str(instance_id),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return cell, proc.returncode, (proc.stdout or ""), (proc.stderr or "")[-2000:]

    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_run, c) for c in todo]
        for done, fut in enumerate(as_completed(futures), start=1):
            cell, code, out, err = fut.result()
            tag = f"{cell.dataset} {cell.tag}"
            if code != 0:
                failed.append(cell)
                print(f"  [ERROR] {tag} (code {code})\n{err}")
            elif verbose:
                warn = [ln for ln in out.splitlines() if "ВНИМАНИЕ" in ln]
                el = time.time() - t0
                print(f"  [{done:2d}/{len(todo)}] {tag} | elapsed {el/60:.1f} min", flush=True)
                for ln in warn:
                    print(f"      {ln}")

    if failed:
        raise RuntimeError(f"{len(failed)} ячеек упало: {failed}")

    after = ensemble_table(cells, instance_id)
    if verbose:
        print(f"\n[training] done in {(time.time()-t0)/60:.1f} min")
        print(after.to_string(index=False))
        short = after[after.missing > 0]
        if len(short):
            print("\n[training] WARNING: a shortfall is a result, not a failure; "
                  "report the actual N:")
            print(short.to_string(index=False))
    return after


# ----------------------------------------------------------------------
# Overtraining (experiment 2, task E)
# ----------------------------------------------------------------------


def overtrain_table(cells, multiples, instance_id: int = 0) -> pd.DataFrame:
    """What is on disk per (cell, rung) right now.

    One row per cell; the ``end`` column is the run counter, because every
    successful run produces that rung, while an upper rung is missing for runs
    whose gate came too late for it to fit inside the epoch budget. Those gaps
    are a result to report, so they are shown rather than smoothed over.
    """
    from training.overtraining import rung_tag

    rungs = [f"x{m:g}" for m in multiples] + ["end"]
    rows = []
    for cell in normalize_cells(cells):
        row = {"dataset": cell.dataset, "width": cell.width, "depth": cell.depth,
               "база": cell.tag, "need": cell.n_models}
        for rung in rungs:
            row[rung] = len(find_model_paths_for_dataset(
                dataset_type=cell.dataset, instance_id=instance_id,
                model_name=rung_tag(cell.tag, rung),
            ))
        row["missing"] = max(0, cell.n_models - row["end"])
        rows.append(row)
    return pd.DataFrame(rows)


def train_overtrain_pool(
    spec,
    cells,
    n_epochs: int = 3000,
    multiples=(1.0, 3.0, 10.0),
    n_jobs: int | None = None,
    python: str | None = None,
    instance_id: int = 0,
    max_attempts: int = 100,
    success_rates: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """``train_pool`` for task E: one run per model, every rung of it stored.

    Same shape and the same idempotence, over ``scripts/train_overtrain_cell.py``.
    The resume unit is a RUN, not a cell: a run writes all of its rungs before
    the next one begins, so an interrupted cell restarts from the first
    unfinished run instead of from scratch.

    ``max_attempts`` is the per-cell ceiling; the loop stops as soon as the
    ensemble is full. ``success_rates`` is accepted for compatibility with older
    notebooks and ignored -- the budget is no longer guessed from a rate.
    """
    spec_json = save_spec(spec)
    n_jobs = n_jobs or min(32, (os.cpu_count() or 4))
    python = python or sys.executable
    success_rates = success_rates or {}

    cells = normalize_cells(cells)
    before = overtrain_table(cells, multiples, instance_id)
    short = {(r["dataset"], r["база"]) for _, r in before.iterrows() if r["missing"] > 0}
    todo = [c for c in cells if (c.dataset, c.tag) in short]

    if verbose:
        print(before.to_string(index=False))
        print(f"\n[overtraining] cells to train: {len(todo)} of {len(cells)}, "
              f"workers {n_jobs}, budget {n_epochs} epochs")

    if not todo:
        return before

    def _run(cell):
        cmd = [
            python, OVERTRAIN_SCRIPT,
            "--spec-json", str(spec_json),
            "--dataset", cell.dataset,
            "--width", str(cell.width), "--depth", str(cell.depth),
            "--n-models", str(cell.n_models),
            "--max-attempts", str(max_attempts),
            "--n-epochs", str(n_epochs),
            "--multiples", ",".join(f"{m:g}" for m in multiples),
            "--activation", cell.activation,
            "--model-name", cell.tag,
            "--instance-id", str(instance_id),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return cell, proc.returncode, (proc.stdout or ""), (proc.stderr or "")[-2000:]

    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_run, c) for c in todo]
        for done, fut in enumerate(as_completed(futures), start=1):
            cell, code, out, err = fut.result()
            tag = f"{cell.dataset} {cell.tag}"
            if code != 0:
                failed.append(cell)
                print(f"  [ERROR] {tag} (code {code})\n{err}")
            elif verbose:
                el = time.time() - t0
                print(f"  [{done:2d}/{len(todo)}] {tag} | elapsed {el/60:.1f} min",
                      flush=True)
                for ln in out.splitlines():
                    if "ступень" in ln or "ВНИМАНИЕ" in ln:
                        print(f"      {ln.strip()}")

    if failed:
        raise RuntimeError(f"{len(failed)} ячеек упало: {failed}")

    after = overtrain_table(cells, multiples, instance_id)
    if verbose:
        print(f"\n[overtraining] done in {(time.time()-t0)/60:.1f} min")
        print(after.to_string(index=False))
    return after
