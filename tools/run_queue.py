"""Run revision experiments as one process per planned run, N in parallel.

    python tools/run_queue.py --workers 16 --exp R11,R2,R5,R8,R7,R3,R4,R6,R1

Same plans, configs, and run directories as ``run_revision.py run --exp X``; the
difference is scheduling: a shared queue instead of static shards, one fresh
process per run, and ``collect`` once per experiment after its last run, so
parallel workers never write ``aggregate.csv`` concurrently. Completed runs are
reloaded by ``run_simulation``, so re-running the same command resumes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _log(path: Path, text: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _worker(config_path: str, results: str, exp_id: str, idx: int, label: str, opts: dict,
            log_dir: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from copy import deepcopy

    from uav_isac.ao_solver import run_simulation
    from uav_isac.config import load_config
    from uav_isac.experiments import build
    from uav_isac.runner import solver_available

    log_path = Path(log_dir) / f"{exp_id}_{label}.log"
    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as fh, redirect_stdout(fh), redirect_stderr(fh):
        try:
            item = build(exp_id, load_config(config_path), opts)[idx]
            assert item.label == label, (item.label, label)
            cfg = deepcopy(item.config)
            cfg.output_root = str(Path(results) / exp_id / "runs")
            if not solver_available(cfg.solver.solver):
                return {"exp": exp_id, "label": label, "status": f"SKIP solver {cfg.solver.solver}",
                        "wall_s": 0.0}
            res = run_simulation(cfg, make_plots=False)
            return {"exp": exp_id, "label": label,
                    "status": "OK" if res.success else f"FAILED {res.failure_code}",
                    "wall_s": time.perf_counter() - t0}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # incl. solver panics (pyo3 PanicException), which cannot be pickled back
            traceback.print_exc()
            return {"exp": exp_id, "label": label, "status": "ERROR (see log)",
                    "wall_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "reference_strict.json"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--exp", required=True, help="comma-separated, in submission order")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="EXP:label entries to restrict the queue")
    ap.add_argument("--skip", nargs="*", default=[], help="EXP:label entries to leave out")
    ap.add_argument("--opts", default="{}", help='JSON {"R1": {...}} of per-experiment options')
    args = ap.parse_args()

    from uav_isac.config import load_config
    from uav_isac.experiments import EXPERIMENTS, build
    from uav_isac.runner import _atomic_csv, collect, plan_frame

    base = load_config(args.config)
    results = Path(args.results)
    log_dir = ROOT / "logs" / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_log = ROOT / "logs" / "queue.log"
    all_opts = json.loads(args.opts)

    jobs, remaining = [], {}
    for exp_id in [x.strip().upper() for x in args.exp.split(",") if x.strip()]:
        opts = all_opts.get(exp_id, {})
        root = results / exp_id
        (root / "runs").mkdir(parents=True, exist_ok=True)
        _atomic_csv(plan_frame(exp_id, base, opts, root / "runs"), root / "manifest.csv")
        (root / "EXPERIMENT.json").write_text(json.dumps({
            "experiment": exp_id, "title": EXPERIMENTS[exp_id].title,
            "reviewer_refs": EXPERIMENTS[exp_id].reviewer_refs,
            "notes": EXPERIMENTS[exp_id].notes, "options": opts}, indent=2), encoding="utf-8")
        for idx, item in enumerate(build(exp_id, base, opts)):
            key = f"{exp_id}:{item.label}"
            if (args.only and key not in args.only) or key in args.skip:
                continue
            jobs.append((exp_id, idx, item.label, opts))
            remaining[exp_id] = remaining.get(exp_id, 0) + 1

    _log(queue_log, f"START {len(jobs)} runs, {args.workers} workers: "
         + ", ".join(f"{k}={v}" for k, v in remaining.items()))
    ctx = mp.get_context("spawn")
    # One fresh process per run needs Python >= 3.11; older versions reuse workers,
    # which is safe because each run is self-contained (config deep-copied, own run dir).
    pool_kwargs = {"max_tasks_per_child": 1} if sys.version_info >= (3, 11) else {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, **pool_kwargs) as pool:
        futures = {pool.submit(_worker, args.config, str(results), e, i, lab, o, str(log_dir)): (e, lab)
                   for e, i, lab, o in jobs}
        done = 0
        for fut in as_completed(futures):
            exp_id, label = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:  # worker process died
                out = {"exp": exp_id, "label": label, "status": f"CRASH {exc!r}", "wall_s": float("nan")}
            done += 1
            _log(queue_log, f"[{done}/{len(jobs)}] {exp_id}:{label} {out['status']} "
                 f"wall={out['wall_s'] / 3600:.2f}h")
            remaining[exp_id] -= 1
            if remaining[exp_id] == 0:
                _log(queue_log, f"COLLECT {exp_id} -> {collect(results / exp_id)}")
    _log(queue_log, "ALL DONE")


if __name__ == "__main__":
    main()
