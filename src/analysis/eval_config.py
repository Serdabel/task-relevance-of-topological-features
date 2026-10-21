"""Persistence of evaluation parameters (the eval-side analog of train_config).

Every results CSV produced by ``analysis.stat_aggregation`` gets a JSON sidecar
``<csv>.eval_config.json`` recording all parameters the numbers depend on:
grid resolution, bounds margin, layer/kind selection, structural/tubular
kwargs (with the defaults that were actually applied), extra criteria, witness
threshold, device — plus provenance (timestamp, git commit, package versions).

Without this record a CSV is unreproducible: the training side has persisted
``train_config`` since the start, but evaluation parameters used to live only
in notebook call sites (gap found in the July 2026 architecture review).
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EVAL_CONFIG_FORMAT_VERSION = 1


def _jsonable(value):
    """Best-effort conversion to JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if callable(value):
        return getattr(value, "__name__", repr(value))

    return repr(value)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            or None
        )
    except Exception:
        return None


def _package_versions() -> dict:
    versions = {}

    for package in ("torch", "numpy", "scipy", "pandas"):
        try:
            versions[package] = __import__(package).__version__
        except Exception:
            versions[package] = None

    return versions


def resolve_kwargs(fn, passed: dict | None) -> dict:
    """Explicitly-passed kwargs merged over ``fn``'s own defaults.

    The sidecar exists so two result CSVs can be known comparable. Recording
    only what the caller happened to pass defeats that: the structural
    criterion's search space is set by ``n_levels`` (default 101), the
    component filter by ``min_component_size``, the adaptive-eps rule by its
    own quantile parameters — none of which any caller passes, so none of
    which reached the sidecar, while docs/DESIGN.md claimed defaults *were*
    resolved (AUDIT/09 section D4). Two runs differing in n_levels produced
    byte-identical config records.

    Only parameters that HAVE a default and whose value is JSON-scalar are
    recorded; the positional data arguments (function, feature_points,
    bounds, field, ...) are configuration of nothing and are skipped.
    """
    import inspect

    out = {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(passed or {})

    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if isinstance(p.default, (str, int, float, bool)) or p.default is None:
            out[name] = p.default

    out.update(passed or {})
    return out


def build_eval_config(**params) -> dict:
    """
    Wraps resolved evaluation parameters with provenance metadata.

    Pass the parameters exactly as resolved at the call site (after defaults
    were applied), e.g.:

        build_eval_config(
            dataset_types=..., resolution=..., bounds_margin=...,
            layer_ids=..., kinds=..., structural_kwargs=...,
            tubular_kwargs=..., extra_criteria=..., device=...,
        )

    Callables (e.g. extra criteria functions) are recorded by name.
    """
    return {
        "format_version": EVAL_CONFIG_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "params": _jsonable(params),
    }


def eval_config_path_for_csv(csv_path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + ".eval_config.json")


def save_eval_config(config: dict, csv_path) -> Path:
    """Writes the sidecar JSON next to the results CSV; returns its path."""
    path = eval_config_path_for_csv(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return path


def load_eval_config(csv_path) -> dict:
    """Loads the sidecar JSON for a results CSV."""
    path = eval_config_path_for_csv(csv_path)

    assert path.exists(), f"No eval config sidecar found: {path}"

    with open(path) as f:
        return json.load(f)
