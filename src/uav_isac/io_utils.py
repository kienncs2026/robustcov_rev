from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import AuditResult
from .config import SimulationConfig
from .dynamics import Trajectory


def config_fingerprint(config: SimulationConfig, length: int = 12) -> str:
    """Stable scientific-config fingerprint used to isolate sweep artifacts.

    ``output_root`` is intentionally excluded because moving a run must not change
    its scientific identity. Every physical, mission, solver, scheme, evaluation,
    scenario, and topology field remains part of the identity.
    """
    payload = config.to_dict()
    payload.pop("output_root", None)
    _drop_default_revision_fields(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


# Fields added in the revision. When they keep their default value they are left
# out of the fingerprint, so a configuration of the locked submission keeps its
# original run directory and completed legacy runs can be reused as they are.
REVISION_FIELDS: dict[str, tuple[str, ...]] = {
    "solver": (
        "pcrb_formulation", "scenario_beta", "scenario_support", "scenario_max_samples",
        "rank_recovery_mode", "gaussian_randomization_trials", "stopping_rule",
        "joint_bound_check", "joint_bound_slots", "candidate_screening",
        "phase1_enabled", "phase1_iterations", "refinement_enabled", "refinement_rounds",
        "refinement_validation_samples", "refinement_points_per_round",
        "refinement_only_if_failed", "random_search_samples", "propulsion_only_segments",
        "propulsion_only_maxiter", "propulsion_only_multistart", "propulsion_only_method",
        "seed_mode",
    ),
    "scheme": ("trajectory_objective",),
    "evaluation": ("rician_k_db", "target_turn_rate_dps", "mismatch_channel_samples",
                   "lemma_diagnostics"),
}


def _drop_default_revision_fields(payload: dict[str, Any]) -> None:
    from .config import EvaluationConfig, SchemeConfig, SolverConfig
    defaults = {"solver": SolverConfig(), "scheme": SchemeConfig(), "evaluation": EvaluationConfig()}
    for section, names in REVISION_FIELDS.items():
        block = payload.get(section, {})
        for name in names:
            if name in block and block[name] == getattr(defaults[section], name):
                block.pop(name)


def run_directory_name(config: SimulationConfig) -> str:
    fingerprint = config_fingerprint(config)
    return (
        f"{config.scenario}_{config.scheme.name}_seed{config.solver.random_seed}_"
        f"topo{config.topology_seed}__cfg{fingerprint}"
    )


def create_run_directory(config: SimulationConfig) -> Path:
    root = Path(config.output_root)
    run_name = run_directory_name(config)
    path = root / run_name
    path.mkdir(parents=True, exist_ok=True)

    # A hash collision is exceptionally unlikely, but never silently reuse a
    # directory whose recorded scientific identity differs.
    identity_path = path / "RUN_IDENTITY.json"
    identity = {
        "config_fingerprint": config_fingerprint(config, length=0),
        "run_directory_name": run_name,
        "scientific_config": {k: v for k, v in config.to_dict().items() if k != "output_root"},
    }
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != identity["config_fingerprint"]:
            raise RuntimeError(f"Refusing to reuse incompatible run directory: {path}")
    else:
        identity_path.write_text(json.dumps(identity, indent=2, default=_jsonable), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _environment_manifest() -> dict[str, Any]:
    # Lazy import keeps config/run-ID utilities usable for preflight checks even
    # before the heavy solver stack is imported.
    from .state_io import environment_manifest
    return environment_manifest()


def write_outputs(run_dir: Path, config: SimulationConfig, trajectory: Trajectory,
                  w_all: np.ndarray, s_all: np.ndarray, audit: AuditResult,
                  iteration_rows: list[dict[str, Any]], solver_rows: list[dict[str, Any]]) -> None:
    config_data = config.to_dict()
    config_data["software"] = _environment_manifest()
    (run_dir / "config.json").write_text(json.dumps(config_data, indent=2, default=_jsonable), encoding="utf-8")
    pd.DataFrame([audit.summary]).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame({
        "slot": np.arange(trajectory.q.shape[0]),
        "qx_m": trajectory.q[:, 0],
        "qy_m": trajectory.q[:, 1],
        "qz_m": trajectory.q[:, 2],
        "vx_mps": trajectory.v[:, 0],
        "vy_mps": trajectory.v[:, 1],
        "vz_mps": trajectory.v[:, 2],
        "ax_mps2": trajectory.a[:, 0],
        "ay_mps2": trajectory.a[:, 1],
        "az_mps2": trajectory.a[:, 2],
    }).to_csv(run_dir / "trajectory.csv", index=False)
    pd.DataFrame(audit.power_rows).to_csv(run_dir / "power_covariance.csv", index=False)
    pd.DataFrame(audit.communication_rows).to_csv(run_dir / "constraint_residuals.csv", index=False)
    pd.DataFrame(audit.pcrb_rows).to_csv(run_dir / "pcrb_out_of_sample.csv", index=False)
    pd.DataFrame(iteration_rows).to_csv(run_dir / "iteration_log.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(run_dir / "solver_log.csv", index=False)
    np.savez_compressed(run_dir / "covariances.npz", W=w_all, S=s_all)
    for name, rows in getattr(audit, "extra_tables", {}).items():
        pd.DataFrame(rows).to_csv(run_dir / f"{name}.csv", index=False)


def write_failure(run_dir: Path, config: SimulationConfig, reason: str,
                  solver_rows: list[dict[str, Any]], *, failure_code: str = "",
                  details: dict[str, Any] | None = None) -> None:
    config_data = config.to_dict()
    config_data["software"] = _environment_manifest()
    (run_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, default=_jsonable), encoding="utf-8"
    )
    row: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "failure_code": failure_code,
    }
    if details:
        row.update(details)
    pd.DataFrame([row]).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(run_dir / "solver_log.csv", index=False)
    (run_dir / "ENVIRONMENT_MANIFEST.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
