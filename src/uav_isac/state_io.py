from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy

from .audit import AuditResult
from .config import SimulationConfig
from .covariance import CovarianceSlotResult
from .dynamics import Trajectory


def environment_manifest() -> dict[str, Any]:
    installed = []
    try:
        installed = list(cp.installed_solvers())
    except Exception:
        installed = []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "cvxpy": cp.__version__,
        "installed_solvers": installed,
    }


def write_environment_manifest(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "ENVIRONMENT_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment_manifest(), indent=2), encoding="utf-8")
    return path


def _result_to_dict(item: CovarianceSlotResult) -> dict[str, Any]:
    data = asdict(item)
    data.pop("w", None)
    data.pop("s", None)
    return data


def _result_from_dict(data: dict[str, Any], w: np.ndarray, s: np.ndarray) -> CovarianceSlotResult:
    return CovarianceSlotResult(w=w, s=s, **data)


def save_ao_state(
    run_dir: str | Path,
    *,
    next_outer_iteration: int,
    trajectory: Trajectory,
    w_all: np.ndarray,
    s_all: np.ndarray,
    results: list[CovarianceSlotResult],
    iteration_rows: list[dict],
    solver_rows: list[dict],
    previous_energy: float,
    accepted_trajectory_updates: int,
    restoration_summary: dict | None = None,
) -> Path:
    state_dir = Path(run_dir) / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp_npz = state_dir / "latest.tmp.npz"
    np.savez_compressed(
        tmp_npz,
        q=trajectory.q,
        v=trajectory.v,
        a=trajectory.a,
        W=w_all,
        S=s_all,
    )
    tmp_npz.replace(state_dir / "latest.npz")
    metadata = {
        "next_outer_iteration": int(next_outer_iteration),
        "previous_energy": float(previous_energy),
        "accepted_trajectory_updates": int(accepted_trajectory_updates),
        "results": [_result_to_dict(item) for item in results],
        "restoration_summary": restoration_summary or {},
    }
    tmp_json = state_dir / "latest.tmp.json"
    tmp_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_json.replace(state_dir / "latest.json")
    pd.DataFrame(iteration_rows).to_csv(state_dir / "iteration_log.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(state_dir / "solver_log.csv", index=False)
    return state_dir / "latest.json"


def load_ao_state(run_dir: str | Path) -> dict[str, Any] | None:
    state_dir = Path(run_dir) / "state"
    json_path = state_dir / "latest.json"
    npz_path = state_dir / "latest.npz"
    if not json_path.exists() or not npz_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    # Close the npz before returning: on Windows an open handle makes the next
    # atomic checkpoint replace of the same file fail with PermissionError.
    with np.load(npz_path) as arrays:
        trajectory = Trajectory(q=arrays["q"], v=arrays["v"], a=arrays["a"])
        w_all = arrays["W"]
        s_all = arrays["S"]
    results = [
        _result_from_dict(item, w_all[idx], s_all[idx])
        for idx, item in enumerate(metadata.get("results", []))
    ]
    iteration_path = state_dir / "iteration_log.csv"
    solver_path = state_dir / "solver_log.csv"
    iteration_rows = (
        pd.read_csv(iteration_path).to_dict("records")
        if iteration_path.exists() and iteration_path.stat().st_size > 0 else []
    )
    solver_rows = (
        pd.read_csv(solver_path).to_dict("records")
        if solver_path.exists() and solver_path.stat().st_size > 0 else []
    )
    return {
        **metadata,
        "trajectory": trajectory,
        "w_all": w_all,
        "s_all": s_all,
        "results": results,
        "iteration_rows": iteration_rows,
        "solver_rows": solver_rows,
    }


def load_completed_run(run_dir: str | Path) -> tuple[Trajectory, np.ndarray, np.ndarray, AuditResult] | None:
    run_dir = Path(run_dir)
    required = [
        run_dir / "summary.csv", run_dir / "trajectory.csv", run_dir / "covariances.npz",
        run_dir / "constraint_residuals.csv", run_dir / "pcrb_out_of_sample.csv",
        run_dir / "power_covariance.csv",
    ]
    if not all(path.exists() for path in required):
        return None
    summary = pd.read_csv(required[0]).iloc[0].to_dict()
    accepted_value = summary.get("accepted", False)
    if isinstance(accepted_value, str):
        accepted_value = accepted_value.strip().lower() in {"true", "1", "yes"}
    # A design that finished but failed an audit gate (e.g. the PCRB criterion) is
    # still a completed result; the COMPLETED marker distinguishes it from a crash.
    # Archived runs may lack the state/ folder, so a full audit summary (it carries
    # the out-of-sample PCRB bound) also identifies a finished design.
    finished_audit = "max_pcrb_ucb_95" in summary and str(summary.get("status", "")).lower() != "failed"
    if not bool(accepted_value) and not ((run_dir / "state" / "COMPLETED").exists() or finished_audit):
        return None
    traj_df = pd.read_csv(required[1])
    trajectory = Trajectory(
        q=traj_df[["qx_m", "qy_m", "qz_m"]].to_numpy(),
        v=traj_df[["vx_mps", "vy_mps", "vz_mps"]].to_numpy(),
        a=traj_df[["ax_mps2", "ay_mps2", "az_mps2"]].to_numpy(),
    )
    with np.load(required[2]) as arrays:
        w_all, s_all = arrays["W"], arrays["S"]
    audit = AuditResult(
        summary=summary,
        communication_rows=pd.read_csv(required[3]).to_dict("records"),
        pcrb_rows=pd.read_csv(required[4]).to_dict("records"),
        power_rows=pd.read_csv(required[5]).to_dict("records"),
    )
    return trajectory, w_all, s_all, audit


def load_cached_failure(run_dir: str | Path) -> dict[str, Any] | None:
    """Load a completed failure record so resume does not re-solve known failures.

    A missing/partial ``summary.csv`` is not considered a cached failure and will be
    resumed/recomputed from the atomic slot/AO checkpoints instead.
    """
    path = Path(run_dir) / "summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        row = pd.read_csv(path).iloc[0].to_dict()
    except Exception:
        return None
    status = str(row.get("status", "")).strip().lower()
    code = str(row.get("failure_code", "") or "").strip()
    if status == "failed" and code:
        return row
    return None
