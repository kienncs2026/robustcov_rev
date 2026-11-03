"""Shared helpers for the fair-baseline and seed studies (revision 2).

The covariance problem and the audit are the same as in the paper's
propulsion-first rows: nine-state enforcement, linear PCRB form (Prop. 2),
CLARABEL, followed by the independent Monte Carlo audit.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CLARABEL (Rust, rayon) starts one worker thread per core unless capped; with many
# parallel single-slot solves this oversubscribes the machine (observed 112 threads
# per process, load ~500 on 112 cores, 10-30x slower slots). Cap before import.
import os  # noqa: E402
for _v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from uav_isac.config import load_config  # noqa: E402
from uav_isac.dynamics import Trajectory, energy_breakdown, kinematic_residuals  # noqa: E402

MISSIONS = ("reference", "displaced")


def mission_config(which: str, config_path: str | None = None):
    """Same configuration as the R2 propulsion-only runs (Table III)."""
    from uav_isac.experiments import build
    base = load_config(config_path or str(ROOT / "configs" / "reference_strict.json"))
    item = [it for it in build("R2", base, {}) if it.label == f"{which}_propulsion_only"][0]
    return item.config


def save_traj(path: Path, traj: Trajectory) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q=traj.q, v=traj.v, a=traj.a)


def load_traj(path: Path) -> Trajectory:
    d = np.load(path)
    return Trajectory(q=d["q"], v=d["v"], a=d["a"])


def evaluate_path(cfg, traj: Trajectory, seed: int, solver: str = "CLARABEL") -> dict:
    """Robust covariance on a fixed path, then the independent audit."""
    from uav_isac.audit import audit_solution
    from uav_isac.covariance import solve_covariances
    from uav_isac.sensing import propagate_targets

    cfg.solver.pcrb_formulation = "linear"
    cfg.solver.solver = solver
    users = np.asarray(cfg.mission.users_m, float)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    t0 = time.perf_counter()
    w, s, res = solve_covariances(traj.q, traj.v, users, means, covs, cfg.physical,
                                  cfg.solver, cfg.scheme, seed=seed)
    ok = len(res) == cfg.mission.slots and all(
        r.status in ("optimal", "optimal_inaccurate") for r in res)
    out = {"cov_ok": bool(ok), "cov_statuses": sorted({r.status for r in res}),
           "cov_s": round(time.perf_counter() - t0, 1)}
    energy = energy_breakdown(traj, cfg.mission, cfg.physical)
    out.update({"propulsion_j": energy["propulsion_j"],
                "vertical_motion_j": energy["vertical_motion_j"]})
    kin = kinematic_residuals(traj, cfg.mission, cfg.physical)
    out["kin_min_margin"] = min(kin["altitude_low"], kin["horizontal_speed"],
                                kin["vertical_speed"], kin["acceleration"])
    if ok:
        a = audit_solution(traj, w, s, users, means, covs, cfg.mission, cfg.physical,
                           cfg.solver, cfg.scheme, cfg.evaluation, seed=seed + 999_999,
                           enforcement_pcrb_residual_m2=max(r.pcrb_residual_m2 for r in res))
        for k in ("total_j", "propulsion_j", "rf_j", "max_random_channel_outage",
                  "max_pcrb_ucb_95", "min_worst_case_sinr_margin", "max_rank_residual",
                  "accepted", "rejection_reasons"):
            out[k] = a.summary.get(k)
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist()
                               if isinstance(o, np.ndarray) else str(o)))
