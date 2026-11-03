"""Execution layer: shards, resume, aggregation, and the legacy wrapper.

Layout under ``results_root``::

    legacy/outputs_cp6_lock/sweep_{S2,S3,S4,S5A}/runs/<run_dir>
    legacy/outputs_cp7_postlock/sweep_{S9,S5B}/runs/<run_dir>
    legacy/outputs_cp7_s9r/sweep_S9R/runs/<run_dir>
    R<k>/runs/<run_dir>, R<k>/manifest.csv, R<k>/aggregate.csv

Copying the archived ``results_reference/*`` folders into ``results/legacy/``
lets every matching legacy run load as completed instead of being recomputed.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time
from pathlib import Path

import pandas as pd

from .config import SimulationConfig
from .experiments import EXPERIMENTS, build
from .io_utils import config_fingerprint, run_directory_name

LEGACY_LAYOUT = {
    "S2": "outputs_cp6_lock", "S3": "outputs_cp6_lock", "S4": "outputs_cp6_lock",
    "S5A": "outputs_cp6_lock", "S9": "outputs_cp7_postlock", "S5B": "outputs_cp7_postlock",
    "S9R": "outputs_cp7_s9r",
}


def solver_available(name: str) -> bool:
    try:
        import cvxpy as cp
        return name.upper() in {s.upper() for s in cp.installed_solvers()}
    except Exception:  # pragma: no cover
        return False


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def plan_frame(exp_id: str, base: SimulationConfig, opts: dict, runs_root: Path) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(build(exp_id, base, opts)):
        cfg = deepcopy(r.config)
        cfg.output_root = str(runs_root)
        rows.append({"index": idx, "experiment": r.experiment, "label": r.label,
                     "run_dir": run_directory_name(cfg),
                     "config_fingerprint": config_fingerprint(cfg, length=0),
                     **{f"tag_{k}": v for k, v in r.tags.items()}})
    return pd.DataFrame(rows)


def run_experiment(exp_id: str, base: SimulationConfig, results_root: str | Path, *,
                   opts: dict | None = None, shard: tuple[int, int] = (0, 1),
                   make_plots: bool = False, only: list[str] | None = None) -> Path:
    from .ao_solver import run_simulation

    exp_id = exp_id.upper()
    opts = opts or {}
    root = Path(results_root) / exp_id
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    plan = build(exp_id, base, opts)
    _atomic_csv(plan_frame(exp_id, base, opts, runs_root), root / "manifest.csv")
    (root / "EXPERIMENT.json").write_text(json.dumps({
        "experiment": exp_id, "title": EXPERIMENTS[exp_id].title,
        "reviewer_refs": EXPERIMENTS[exp_id].reviewer_refs,
        "notes": EXPERIMENTS[exp_id].notes, "options": opts}, indent=2), encoding="utf-8")
    k, n = shard
    for idx, item in enumerate(plan):
        if idx % n != k or (only and item.label not in only):
            continue
        cfg = deepcopy(item.config)
        cfg.output_root = str(runs_root)
        if not solver_available(cfg.solver.solver):
            print(f"[{exp_id} {idx+1}/{len(plan)}] SKIP {item.label}: solver {cfg.solver.solver} not installed",
                  flush=True)
            continue
        print(f"[{exp_id} {idx+1}/{len(plan)}] RUN {item.label} -> {run_directory_name(cfg)}", flush=True)
        t0 = time.perf_counter()
        result = run_simulation(cfg, make_plots=make_plots)
        wall = time.perf_counter() - t0
        print(f"[{exp_id} {idx+1}/{len(plan)}] DONE {item.label} success={result.success} "
              f"code={result.failure_code or '-'} wall={wall:.0f}s", flush=True)
    return collect(root)


def collect(exp_root: str | Path) -> Path:
    """Rebuild ``aggregate.csv`` from the run directories (shard-safe)."""
    exp_root = Path(exp_root)
    manifest = exp_root / "manifest.csv"
    tags = pd.read_csv(manifest).set_index("run_dir") if manifest.exists() else None
    rows = []
    runs_dir = exp_root / "runs" if (exp_root / "runs").exists() else exp_root
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            continue
        try:
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
        except Exception:
            continue
        row = {"run_dir": run_dir.name,
               "completed": (run_dir / "state" / "COMPLETED").exists(),
               "success": str(summary.get("status", "")).lower() != "failed"}
        if tags is not None and run_dir.name in tags.index:
            row.update(tags.loc[run_dir.name].to_dict())
        row.update(summary)
        rows.append(row)
    out = exp_root / "aggregate.csv"
    _atomic_csv(pd.DataFrame(rows), out)
    return out


def run_legacy(base: SimulationConfig, legacy_root: str | Path, which: list[str], *,
               shard: tuple[int, int] = (0, 1), make_plots: bool = False) -> list[Path]:
    """Re-run (or reload) the submission's frozen studies with this code base."""
    from .ao_solver import run_simulation
    from .legacy.postlock import build_plan
    from .legacy.s9r import build_s9r_plan
    from .legacy.scenarios import run_sweep, sweep_configs

    legacy_root = Path(legacy_root)
    outputs = []
    for name in which:
        name = name.upper()
        root = legacy_root / LEGACY_LAYOUT[name]
        if name in {"S2", "S3", "S4", "S5A"}:
            cfg = deepcopy(base)
            cfg.output_root = str(root)
            k, n = shard
            idx = None
            if n > 1:
                if name in {"S3", "S4"}:
                    raise SystemExit(f"{name} uses sequential warm starts; run it without --shard")
                idx = {i for i in range(len(sweep_configs(cfg, name))) if i % n == k}
            outputs.append(run_sweep(cfg, name, warm_start_adjacent=(name in {"S3", "S4"}),
                                     make_plots=make_plots, index_filter=idx))
            continue
        plan = [p.config for p in (build_s9r_plan(base) if name == "S9R" else build_plan(base, [name]))]
        runs_root = root / f"sweep_{name}" / "runs"
        k, n = shard
        for i, cfg in enumerate(plan):
            if i % n != k:
                continue
            cfg = deepcopy(cfg)
            cfg.output_root = str(runs_root)
            print(f"[LEGACY {name} {i+1}/{len(plan)}] {run_directory_name(cfg)}", flush=True)
            run_simulation(cfg, make_plots=make_plots)
        outputs.append(collect(root / f"sweep_{name}"))
    return outputs
