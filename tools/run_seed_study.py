"""Independent optimization seeds for the level propulsion-first design (R1.9).

Each task reruns the full propulsion-first design with a different random seed:
  propulsion search (30-start Powell, level projection; rng = random_seed + 61000 + k)
  -> robust covariance (nine-state, linear PCRB, CLARABEL) -> independent audit.
Optionally (--ao-reference) the reference design of Table II (Algorithm 1 from the
straight seed, settings of the reference configuration) is rerun with
solver.random_seed = random_seed + k.

Usage (from robustcov_rev/):
  python tools/run_seed_study.py --seeds 0 1 2 3 4 --jobs 4
  python tools/run_seed_study.py --seeds 0 1 2 --missions reference --ao-reference --jobs 3
Finished tasks are skipped on rerun (results_rev2/seeds/*.json).
Expected cost per level task (authors' machine): 25 min (reference) to 60 min
(displaced) of propulsion search plus 2-11 min of covariance solves; tasks are
independent and run in parallel with --jobs.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from rev2_common import (MISSIONS, ROOT, evaluate_path, load_traj,  # noqa: E402
                         mission_config, save_traj, write_json)


def level_task(which: str, k: int, out: str, skip_isac: bool, multistart: int | None = None) -> str:
    from uav_isac.baselines import propulsion_only_trajectory
    from uav_isac.dynamics import trapezoidal_seed
    out_dir = Path(out)
    tag = f"{which}_level_seed{k}"
    if (out_dir / f"{tag}.json").exists():
        return f"{tag}: exists, skipped"
    cfg = mission_config(which)
    if multistart:
        cfg.solver.propulsion_only_multistart = int(multistart)
    seed = trapezoidal_seed(cfg.mission, cfg.physical, altitude_m=cfg.scheme.fixed_altitude_m)
    t0 = time.perf_counter()
    npz = out_dir / f"{tag}.npz"
    if npz.exists():
        # The Powell search is deterministic given rng_seed, so a saved path from an
        # interrupted task is reused and only the covariance/audit stage is redone.
        from uav_isac.dynamics import energy_breakdown
        import numpy as np
        traj = load_traj(npz)
        info = {"propulsion_only_seed_propulsion_j": energy_breakdown(seed, cfg.mission, cfg.physical)["propulsion_j"],
                "propulsion_only_propulsion_j": energy_breakdown(traj, cfg.mission, cfg.physical)["propulsion_j"],
                "propulsion_only_runtime_s": None, "propulsion_only_iterations": None,
                "propulsion_only_max_displacement_m": float(np.max(np.linalg.norm(traj.q - seed.q, axis=1))),
                "propulsion_only_level": bool(np.allclose(traj.v[:, 2], 0.0)),
                "propulsion_only_vertical_j": energy_breakdown(traj, cfg.mission, cfg.physical)["vertical_motion_j"],
                "search_reused_saved_path": True}
    else:
        traj, info = propulsion_only_trajectory(seed, cfg, rng_seed=cfg.solver.random_seed + 61_000 + k)
    rec = {"mission": which, "design": "level_propulsion_first", "seed_index": k,
           "multistart": int(cfg.solver.propulsion_only_multistart),
           "rng_seed": cfg.solver.random_seed + 61_000 + k,
           "search_wall_s": round(time.perf_counter() - t0, 1), **info}
    save_traj(npz, traj)
    if not skip_isac:
        rec["isac"] = evaluate_path(cfg, traj, seed=cfg.solver.random_seed)
    write_json(out_dir / f"{tag}.json", rec)
    return f"{tag}: E_prop={info['propulsion_only_propulsion_j']:.2f} J"


def ao_reference_task(k: int, out: str) -> str:
    from uav_isac.ao_solver import run_simulation
    from uav_isac.config import load_config
    from uav_isac.experiments import REVISION_AO, _proposed, with_overrides
    out_dir = Path(out)
    tag = f"reference_ao_seed{k}"
    if (out_dir / f"{tag}.json").exists():
        return f"{tag}: exists, skipped"
    base = load_config(str(ROOT / "configs" / "reference_strict.json"))
    cfg = with_overrides(_proposed(base), {**REVISION_AO,
                                          "solver.random_seed": base.solver.random_seed + k})
    cfg.output_root = str(out_dir / "runs")
    cfg.scheme.name = f"rev2_reference_ao_seed{k}"
    t0 = time.perf_counter()
    res = run_simulation(cfg, make_plots=False)
    rec = {"mission": "reference", "design": "ao_from_straight_seed", "seed_index": k,
           "random_seed": cfg.solver.random_seed, "success": res.success,
           "reason": res.reason, "wall_s": round(time.perf_counter() - t0, 1),
           "run_dir": str(res.run_dir)}
    if res.audit is not None:
        rec.update({key: res.audit.summary.get(key) for key in (
            "total_j", "propulsion_j", "rf_j", "max_random_channel_outage", "max_pcrb_ucb_95",
            "min_worst_case_sinr_margin", "max_rank_residual", "accepted")})
    if res.trajectory is not None:
        import numpy as np
        from uav_isac.dynamics import trapezoidal_seed
        s0 = trapezoidal_seed(cfg.mission, cfg.physical, altitude_m=cfg.scheme.fixed_altitude_m)
        rec["max_displacement_m"] = float(np.max(np.linalg.norm(res.trajectory.q - s0.q, axis=1)))
    write_json(out_dir / f"{tag}.json", rec)
    return f"{tag}: success={res.success}, total={rec.get('total_j')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", default=list(MISSIONS), choices=MISSIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="k = 0 reproduces the paper's level designs")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--ao-reference", action="store_true",
                    help="also rerun the reference AO design (Table II) for each seed")
    ap.add_argument("--skip-isac", action="store_true", help="propulsion search only")
    ap.add_argument("--multistart", type=int, default=None,
                    help="override the 30 Powell starts (smoke tests only)")
    ap.add_argument("--out", default=str(ROOT / "results_rev2" / "seeds"))
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tasks = [("level", w, k) for w in args.missions for k in args.seeds]
    if args.ao_reference:
        tasks += [("ao", "reference", k) for k in args.seeds]
    print(f"{len(tasks)} tasks on {args.jobs} worker(s)", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {}
        for kind, w, k in tasks:
            f = (pool.submit(level_task, w, k, args.out, args.skip_isac, args.multistart) if kind == "level"
                 else pool.submit(ao_reference_task, k, args.out))
            futs[f] = (kind, w, k)
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as exc:  # keep the other tasks running
                print(f"{futs[f]} failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
