from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd

import numpy as np

from .audit import AuditResult, audit_solution
from .config import SimulationConfig
from .covariance import CovarianceSlotResult, evaluate_covariance_state, solve_covariances
from .dynamics import Trajectory, energy_breakdown, kinematic_residuals, trapezoidal_seed
from .failure import FailureCode, classify_covariance_failure
from .feasibility import restore_feasible_trajectory, write_feasibility_report
from .io_utils import create_run_directory, write_failure, write_outputs
from .plots import plot_run
from .sensing import process_covariance, propagate_targets, transition_matrix
from .state_io import load_ao_state, load_cached_failure, load_completed_run, save_ao_state, write_environment_manifest
from .trajectory_opt import update_trajectory
from . import theory


@dataclass
class RunResult:
    success: bool
    run_dir: Path
    trajectory: Trajectory | None
    w: np.ndarray | None
    s: np.ndarray | None
    audit: AuditResult | None
    reason: str = ""
    failure_code: str = ""


def _solver_rows(results: list[CovarianceSlotResult], outer: int, label: str) -> list[dict]:
    rows = []
    for slot, item in enumerate(results):
        rows.append({
            "outer_iteration": outer,
            "solve_label": label,
            "slot": slot,
            "status": item.status,
            "power_w": item.objective_power_w,
            "runtime_s": item.runtime_s,
            "rank_residual": item.rank_residual,
            "max_comm_slack": item.max_comm_slack,
            "max_sensing_slack": item.max_sensing_slack,
            "solver_iterations": item.solver_iterations,
            "pcrb_residual_m2": item.pcrb_residual_m2,
            "solver_used": item.solver_used,
            "min_w_eigenvalue": item.min_w_eigenvalue,
            "min_s_eigenvalue": item.min_s_eigenvalue,
            "psd_ok": item.psd_ok,
            "total_solver_runtime_s": item.total_solver_runtime_s,
            "solver_call_count": item.solver_call_count,
            "fallback_call_count": item.fallback_call_count,
            "rank_refinement_call_count": item.rank_refinement_call_count,
            "rank_recovery_call_count": item.rank_recovery_call_count,
        })
    return rows


def _covariance_success(
    results: list[CovarianceSlotResult], expected_slots: int, config: SimulationConfig
) -> bool:
    if len(results) != expected_slots:
        return False
    return all(
        np.isfinite(item.objective_power_w)
        and item.status in {"optimal", "optimal_inaccurate"}
        and item.psd_ok
        and np.isfinite(item.pcrb_residual_m2)
        and item.pcrb_residual_m2 <= config.solver.pcrb_residual_tol_m2 + 1e-9
        for item in results
    )


def _perfect_target_realization(mission, physical, target_means, target_covariances,
                                rng: np.random.Generator) -> np.ndarray:
    """Draw one temporally consistent realization of each constant-velocity target."""
    n_slots = mission.slots
    f = transition_matrix(mission.dt_s)
    q_proc = process_covariance(mission.dt_s, physical.process_acceleration_std_mps2)
    states = np.empty_like(target_means)
    zero4 = np.zeros(4)
    for l in range(target_means.shape[0]):
        xi = rng.multivariate_normal(target_means[l, 0], target_covariances[l, 0])
        states[l, 0] = xi
        for n in range(1, n_slots):
            xi = f @ xi + rng.multivariate_normal(zero4, q_proc)
            states[l, n] = xi
    return states


def _kinematic_ok(trajectory: Trajectory, config: SimulationConfig) -> tuple[bool, dict]:
    kin = kinematic_residuals(trajectory, config.mission, config.physical)
    tol = config.solver.trajectory_feasibility_tolerance
    ok = (
        max(kin["dynamics"], kin["terminal_position"], kin["terminal_velocity"]) <= tol
        and min(
            kin["altitude_low"], kin["altitude_high"], kin["horizontal_speed"],
            kin["vertical_speed"], kin["acceleration"],
        ) >= -tol
    )
    return bool(ok), kin


def run_simulation(config: SimulationConfig, *, quick_audit: bool = False,
                   make_plots: bool = True,
                   covariance_warm_start: tuple[np.ndarray, np.ndarray] | None = None,
                   warm_start_label: str = "",
                   feasible_incumbent: tuple[Trajectory, np.ndarray, np.ndarray] | None = None,
                   incumbent_label: str = "") -> RunResult:
    run_dir = create_run_directory(config)
    write_environment_manifest(run_dir)

    if config.solver.resume_enabled and config.solver.skip_completed:
        completed = load_completed_run(run_dir)
        if completed is not None:
            trajectory, w_all, s_all, audit = completed
            return RunResult(True, run_dir, trajectory, w_all, s_all, audit,
                             reason="loaded_completed_run")
        cached_failure = load_cached_failure(run_dir)
        if cached_failure is not None:
            cached_code = str(cached_failure.get("failure_code", ""))
            physical_codes = {
                "INFEASIBLE_FIXED_GEOMETRY_SENSING",
                "INFEASIBLE_FIXED_GEOMETRY_COMMUNICATION",
                "INFEASIBLE_ACTIVE_CONSTRAINTS",
                "KINEMATIC_INFEASIBILITY",
            }
            # Even during a numerical-recovery pass, a previously certified physical
            # infeasibility is not recomputed. Numerical failures are re-run only
            # when rerun_failed=True.
            if (not config.solver.rerun_failed) or cached_code in physical_codes:
                return RunResult(
                    False, run_dir, None, None, None, None,
                    reason="loaded_cached_failure", failure_code=cached_code,
                )

    users = np.asarray(config.mission.users_m, dtype=float)
    target_means, target_covariances = propagate_targets(config.mission, config.physical)
    perfect_channel_errors = None
    perfect_target_states = None
    if config.scheme.perfect_information:
        oracle_rng = np.random.default_rng(config.solver.random_seed + 314159)
        n_slots = config.mission.slots
        k_count = users.shape[0]
        m = config.physical.antennas
        raw = oracle_rng.normal(size=(n_slots, k_count, 2 * m))
        raw /= np.maximum(np.linalg.norm(raw, axis=2, keepdims=True), 1e-12)
        radii = oracle_rng.random((n_slots, k_count, 1)) ** (1.0 / (2.0 * m))
        raw *= radii
        perfect_channel_errors = raw[:, :, :m] + 1j * raw[:, :, m:]
        perfect_target_states = _perfect_target_realization(
            config.mission, config.physical, target_means, target_covariances, oracle_rng
        )

    wall_start = time.perf_counter()
    timing = {"time_restoration_s": 0.0, "time_joint_bound_s": 0.0, "time_covariance_s": 0.0,
              "time_trajectory_s": 0.0, "time_screening_s": 0.0, "time_phase1_s": 0.0,
              "time_refinement_s": 0.0, "time_audit_s": 0.0, "covariance_passes": 0,
              "candidates_screened_out": 0}
    revision_info: dict = {}

    def _timed_cov(*args, **kwargs):
        t = time.perf_counter()
        out = solve_covariances(*args, **kwargs)
        timing["time_covariance_s"] += time.perf_counter() - t
        timing["covariance_passes"] += 1
        return out

    fixed_altitude = config.scheme.fixed_altitude_m
    seed_trajectory = trapezoidal_seed(
        config.mission, config.physical, altitude_m=fixed_altitude
    )
    propulsion_only = config.scheme.trajectory_objective == "propulsion_only"
    trajectory_fixed = bool(config.scheme.fixed_trajectory or propulsion_only)
    if propulsion_only or (config.solver.seed_mode == "propulsion_optimal" and not trajectory_fixed):
        from .baselines import propulsion_only_trajectory
        t = time.perf_counter()
        seed_trajectory, info = propulsion_only_trajectory(
            seed_trajectory, config, rng_seed=config.solver.random_seed + 61_000)
        timing["time_trajectory_s"] += time.perf_counter() - t
        revision_info.update(info)
    initial_seed_trajectory = Trajectory(
        q=seed_trajectory.q.copy(), v=seed_trajectory.v.copy(), a=seed_trajectory.a.copy()
    )
    seed_ok, seed_kin = _kinematic_ok(seed_trajectory, config)
    if not seed_ok:
        code = FailureCode.KINEMATIC_INFEASIBILITY.value
        reason = f"Initial trajectory is dynamically infeasible: {seed_kin}"
        write_failure(run_dir, config, reason, [], failure_code=code)
        return RunResult(False, run_dir, None, None, None, None, reason, code)

    iteration_rows: list[dict] = []
    solver_rows: list[dict] = []
    previous_energy = np.inf
    accepted_w = accepted_s = None
    accepted_results: list[CovarianceSlotResult] = []
    accepted_trajectory_updates = 0
    start_outer = 0
    restoration_summary: dict = {}

    resume_state = load_ao_state(run_dir) if config.solver.resume_enabled else None
    if resume_state is not None and not config.solver.rerun_failed:
        trajectory = resume_state["trajectory"]
        accepted_w = resume_state["w_all"]
        accepted_s = resume_state["s_all"]
        accepted_results = resume_state["results"]
        iteration_rows = resume_state["iteration_rows"]
        solver_rows = resume_state["solver_rows"]
        previous_energy = float(resume_state.get("previous_energy", np.inf))
        accepted_trajectory_updates = int(
            resume_state.get("accepted_trajectory_updates", 0)
        )
        start_outer = int(resume_state.get("next_outer_iteration", 0))
        restoration_summary = dict(resume_state.get("restoration_summary", {}))
    else:
        # Proposition 5 on slot 1. Its UAV state is fixed by the boundary conditions,
        # so the certificate holds for every trajectory and is checked before the
        # restoration phase, which could only change later slots.
        if config.solver.joint_bound_check:
            t_jb = time.perf_counter()
            jb_rows = theory.joint_bound_report(
                seed_trajectory.q, seed_trajectory.v, target_means, target_covariances,
                lambda slot: _enforcement_sets(seed_trajectory, slot, users, target_means,
                                               target_covariances, config),
                config.physical, [0], solver=config.solver.fallback_solver or "CLARABEL")
            pd.DataFrame(jb_rows).to_csv(run_dir / "joint_bound_certificate.csv", index=False)
            timing["time_joint_bound_s"] += time.perf_counter() - t_jb
            first = jb_rows[0]
            jb_summary = {
                "joint_bound_slot1_certified_w": first["joint_bound_certified_w"],
                "joint_bound_slot1_per_point_w": first["per_point_max_pmin_w"],
            }
            revision_info.update(jb_summary)
            if first["certified_infeasible"]:
                code = FailureCode.INFEASIBLE_CERTIFIED_BOUNDARY.value
                reason = (
                    "Proposition 5: joint illumination bound "
                    f"{first['joint_bound_certified_w']:.6g} W exceeds P_max={config.physical.p_max_w} W "
                    "in slot 1, which is fixed by the boundary conditions; the surrogate is "
                    "infeasible for every trajectory."
                )
                write_failure(run_dir, config, reason, [], failure_code=code,
                              details={**jb_summary, **timing})
                return RunResult(False, run_dir, seed_trajectory, None, None, None, reason, code)
        restoration_cfg = config
        if propulsion_only:
            restoration_cfg = deepcopy(config)
            restoration_cfg.scheme.fixed_trajectory = True
        t_rest = time.perf_counter()
        restoration = restore_feasible_trajectory(
            seed_trajectory, users, target_means, target_covariances, restoration_cfg,
            seed=config.solver.random_seed + 51_000,
        )
        timing["time_restoration_s"] += time.perf_counter() - t_rest
        trajectory = restoration.trajectory
        restoration_summary = {
            "restoration_status": restoration.status,
            "restoration_runtime_s": restoration.runtime_s,
            "restoration_candidates_tested": restoration.candidates_tested,
            "restoration_iterations": restoration.optimization_iterations,
            "initial_feasibility_bound_ratio": restoration.initial_worst_ratio,
            "final_feasibility_bound_ratio": restoration.final_worst_ratio,
            "feasibility_bound_pass": restoration.feasible_bound,
            "worst_sensing_bound_ratio": float(
                restoration.report.summary["worst_sensing_ratio"]
            ),
            "worst_communication_bound_ratio": float(
                restoration.report.summary["worst_communication_ratio"]
            ),
        }
        write_feasibility_report(
            restoration.report, run_dir, prefix="fixed_geometry_feasibility_bound"
        )
        iteration_rows.append({
            "outer_iteration": -1,
            "accepted": restoration.feasible_bound,
            "total_energy_j": np.nan,
            "propulsion_energy_j": np.nan,
            "rf_energy_j": np.nan,
            "trajectory_penalty": np.nan,
            "trajectory_status": "F0:" + restoration.status,
            **restoration_summary,
        })
        if not restoration.feasible_bound:
            sensing_ratio = float(restoration.report.summary["worst_sensing_ratio"])
            comm_ratio = float(restoration.report.summary["worst_communication_ratio"])
            if sensing_ratio > 1.0 + config.solver.feasibility_restoration_ratio_tolerance:
                code = FailureCode.INFEASIBLE_FIXED_GEOMETRY_SENSING.value
            elif comm_ratio > 1.0 + config.solver.feasibility_restoration_ratio_tolerance:
                code = FailureCode.INFEASIBLE_FIXED_GEOMETRY_COMMUNICATION.value
            else:
                code = FailureCode.INFEASIBLE_ACTIVE_CONSTRAINTS.value
            reason = (
                "Phase F0 could not remove the fixed-geometry necessary-condition "
                f"violation; worst ratio={restoration.final_worst_ratio:.6g}."
            )
            write_failure(
                run_dir, config, reason, solver_rows, failure_code=code,
                details=restoration_summary,
            )
            return RunResult(False, run_dir, trajectory, None, None, None, reason, code)

    if (config.solver.joint_bound_check and resume_state is None
            and config.solver.joint_bound_slots == "all"):
        t_jb = time.perf_counter()
        jb_rows = theory.joint_bound_report(
            trajectory.q, trajectory.v, target_means, target_covariances,
            lambda slot: _enforcement_sets(trajectory, slot, users, target_means, target_covariances, config),
            config.physical, list(range(config.mission.slots)),
            solver=config.solver.fallback_solver or "CLARABEL")
        pd.DataFrame(jb_rows).to_csv(run_dir / "joint_bound_all_slots.csv", index=False)
        timing["time_joint_bound_s"] += time.perf_counter() - t_jb
        restoration_summary["joint_bound_max_certified_ratio"] = max(r["certified_ratio"] for r in jb_rows)

    state_dir = run_dir / "state" / "covariance_slots"
    warm_w = warm_s = None
    if covariance_warm_start is not None and resume_state is None:
        warm_w, warm_s = covariance_warm_start
    restoration_summary.update({
        "warm_start_used": bool(covariance_warm_start is not None and resume_state is None),
        "warm_start_label": str(warm_start_label),
    })

    # Certified feasible-incumbent preservation for monotone S4 threshold sweeps.
    # The incumbent is evaluated under the CURRENT PCRB threshold and numerical
    # gates before it can be used. It is never accepted merely because it came
    # from a neighboring run.
    preserved = None
    preserved_energy = float("inf")
    preserved_label = str(incumbent_label)
    preserved_used = False
    if feasible_incumbent is not None and resume_state is None:
        inc_traj, inc_w, inc_s = feasible_incumbent
        inc_kin_ok, _ = _kinematic_ok(inc_traj, config)
        shapes_ok = (
            np.shape(inc_w) == (config.mission.slots, len(users), config.physical.antennas, config.physical.antennas)
            and np.shape(inc_s) == (config.mission.slots, config.physical.antennas, config.physical.antennas)
        )
        if inc_kin_ok and shapes_ok:
            inc_results = evaluate_covariance_state(
                inc_traj.q, inc_traj.v, users, target_means, target_covariances,
                inc_w, inc_s, config.physical, config.solver, config.scheme,
                seed=config.solver.random_seed,
            )
            if (_covariance_success(inc_results, config.mission.slots, config)
                    and max((r.rank_residual for r in inc_results), default=np.inf) <= config.solver.rank_tolerance):
                inc_power = np.trace(inc_w.sum(axis=1) + inc_s, axis1=1, axis2=2).real
                inc_energy = energy_breakdown(inc_traj, config.mission, config.physical, inc_power)
                preserved = (inc_traj, inc_w.copy(), inc_s.copy(), inc_results)
                preserved_energy = float(inc_energy["total_j"])
                solver_rows.extend(_solver_rows(inc_results, -1, "certified_feasible_incumbent"))
                restoration_summary.update({
                    "feasible_incumbent_available": True,
                    "feasible_incumbent_label": preserved_label,
                    "feasible_incumbent_energy_j": preserved_energy,
                })
    restoration_summary.setdefault("feasible_incumbent_available", False)
    restoration_summary.setdefault("feasible_incumbent_label", preserved_label)
    restoration_summary.setdefault("feasible_incumbent_energy_j", np.nan)

    legacy_stop = config.solver.stopping_rule == "legacy"
    last_gain_rel = np.inf
    reuse_state = None
    for outer in range(start_outer, config.solver.outer_iterations):
        if (not legacy_stop) and outer > start_outer and last_gain_rel <= config.solver.outer_tolerance:
            # Revision stopping rule (Prop. 6): the last accepted update gained at most
            # outer_tolerance (relative); its covariances are already the accepted state.
            break
        if (not legacy_stop) and reuse_state is not None:
            w_all, s_all, slot_results = reuse_state
            reuse_state = None
        else:
            w_all, s_all, slot_results = _timed_cov(
            trajectory.q, trajectory.v, users, target_means, target_covariances,
            config.physical, config.solver, config.scheme,
            seed=config.solver.random_seed + 100_000 * outer,
            allow_slack=False,
            perfect_channel_errors=perfect_channel_errors,
            perfect_target_states=perfect_target_states,
            checkpoint_dir=state_dir,
            checkpoint_label=f"outer_{outer:03d}_current",
            resume=(config.solver.resume_enabled and not config.solver.rerun_failed),
            initial_w=(warm_w if outer == start_outer else None),
            initial_s=(warm_s if outer == start_outer else None),
            )
        solver_rows.extend(_solver_rows(slot_results, outer, "current"))
        if (not _covariance_success(slot_results, config.mission.slots, config)
                and config.solver.phase1_enabled and not trajectory_fixed and preserved is None):
            t_p1 = time.perf_counter()
            resume_ok = config.solver.resume_enabled and not config.solver.rerun_failed
            restored, p1_rows = _phase1(trajectory, users, target_means, target_covariances,
                                        config, outer, _timed_cov,
                                        checkpoint_dir=state_dir, resume=resume_ok)
            timing["time_phase1_s"] += time.perf_counter() - t_p1
            iteration_rows.extend(p1_rows)
            if restored is not None:
                trajectory = restored
                w_all, s_all, slot_results = _timed_cov(
                    trajectory.q, trajectory.v, users, target_means, target_covariances,
                    config.physical, config.solver, config.scheme,
                    seed=config.solver.random_seed + 100_000 * outer + 31,
                    allow_slack=False,
                    checkpoint_dir=state_dir,
                    checkpoint_label=f"outer_{outer:03d}_after_phase1",
                    resume=resume_ok,
                )
                solver_rows.extend(_solver_rows(slot_results, outer, "current_after_phase1"))
        if not _covariance_success(slot_results, config.mission.slots, config) and preserved is not None:
            # A looser S4 PCRB threshold cannot lose a previously certified
            # stricter-threshold solution because of a weak SCS iterate. Restore
            # the incumbent and continue AO from the known feasible state.
            trajectory, w_all, s_all, slot_results = preserved
            w_all = w_all.copy(); s_all = s_all.copy()
            slot_results = list(slot_results)
            preserved_used = True
            solver_rows.extend(_solver_rows(slot_results, outer, "preserved_incumbent_after_solver_failure"))

        if not _covariance_success(slot_results, config.mission.slots, config):
            _, _, slack_results = _timed_cov(
                trajectory.q, trajectory.v, users, target_means, target_covariances,
                config.physical, config.solver, config.scheme,
                seed=config.solver.random_seed + 100_000 * outer + 77,
                allow_slack=True,
                perfect_channel_errors=perfect_channel_errors,
                perfect_target_states=perfect_target_states,
                checkpoint_dir=state_dir,
                checkpoint_label=f"outer_{outer:03d}_slack",
                resume=(config.solver.resume_enabled and not config.solver.rerun_failed),
            )
            solver_rows.extend(_solver_rows(slack_results, outer, "slack_diagnostic"))
            failure = classify_covariance_failure(
                slot_results,
                worst_sensing_ratio=restoration_summary.get("worst_sensing_bound_ratio"),
                worst_communication_ratio=restoration_summary.get("worst_communication_bound_ratio"),
                psd_tolerance=config.solver.psd_tolerance,
                pcrb_residual_tolerance=config.solver.pcrb_residual_tol_m2,
                ratio_tolerance=config.solver.feasibility_restoration_ratio_tolerance,
            )
            reason = (
                f"Covariance stage failed at outer iteration {outer}; "
                f"classification={failure.value}."
            )
            max_comm_slack = max(
                (item.max_comm_slack for item in slack_results if np.isfinite(item.max_comm_slack)),
                default=np.nan,
            )
            max_sensing_slack = max(
                (item.max_sensing_slack for item in slack_results if np.isfinite(item.max_sensing_slack)),
                default=np.nan,
            )
            write_failure(
                run_dir, config, reason, solver_rows, failure_code=failure.value,
                details={
                    **restoration_summary,
                    "outer_iteration": outer,
                    "max_diagnostic_comm_slack": max_comm_slack,
                    "max_diagnostic_sensing_slack": max_sensing_slack,
                },
            )
            return RunResult(
                False, run_dir, trajectory, None, None, None, reason, failure.value
            )

        if (preserved is not None
                and max((item.rank_residual for item in slot_results), default=np.inf) > config.solver.rank_tolerance):
            trajectory, w_all, s_all, slot_results = preserved
            w_all = w_all.copy(); s_all = s_all.copy()
            slot_results = list(slot_results)
            preserved_used = True
            solver_rows.extend(_solver_rows(slot_results, outer, "preserved_incumbent_rank_guard"))

        tx_power = np.trace(w_all.sum(axis=1) + s_all, axis1=1, axis2=2).real
        energy = energy_breakdown(trajectory, config.mission, config.physical, tx_power)
        if (preserved is not None
                and energy["total_j"] > preserved_energy + config.solver.energy_acceptance_tolerance_j):
            trajectory, w_all, s_all, slot_results = preserved
            w_all = w_all.copy(); s_all = s_all.copy()
            slot_results = list(slot_results)
            tx_power = np.trace(w_all.sum(axis=1) + s_all, axis1=1, axis2=2).real
            energy = energy_breakdown(trajectory, config.mission, config.physical, tx_power)
            preserved_used = True
            solver_rows.extend(_solver_rows(slot_results, outer, "preserved_incumbent_energy_guard"))
        iteration_rows.append({
            "outer_iteration": outer,
            "accepted": True,
            "total_energy_j": energy["total_j"],
            "propulsion_energy_j": energy["propulsion_j"],
            "rf_energy_j": energy["rf_j"],
            "trajectory_penalty": 0.0,
            "trajectory_status": "before_update",
        })
        accepted_w, accepted_s = w_all, s_all
        accepted_results = slot_results

        relative_change = abs(previous_energy - energy["total_j"]) / max(abs(previous_energy), 1.0)
        should_stop = (
            (legacy_stop and outer > 0 and relative_change <= config.solver.outer_tolerance)
            or not config.solver.trajectory_updates
            or trajectory_fixed
        )
        previous_energy = energy["total_j"]

        if not should_stop:
            t_tr = time.perf_counter()
            update = update_trajectory(
                trajectory, w_all, s_all, users, target_means, target_covariances,
                config.mission, config.physical, config.solver, config.scheme,
                seed=config.solver.random_seed + 400_000 + outer,
            )
            timing["time_trajectory_s"] += time.perf_counter() - t_tr
            candidate = update.trajectory
            candidate_displacement_m = float(
                np.max(np.linalg.norm(candidate.q - trajectory.q, axis=1))
            )
            candidate_nontrivial = candidate_displacement_m > 1e-6
            candidate_kinematic_ok, candidate_kin = _kinematic_ok(candidate, config)
            candidate_w = np.empty((0,))
            candidate_s = np.empty((0,))
            candidate_results: list[CovarianceSlotResult] = []
            candidate_lower_bound = np.nan
            screened_out = False
            if candidate_kinematic_ok and candidate_nontrivial and config.solver.candidate_screening:
                t_sc = time.perf_counter()
                candidate_lower_bound = _energy_lower_bound(
                    candidate, users, target_means, target_covariances, config)
                timing["time_screening_s"] += time.perf_counter() - t_sc
                screened_out = candidate_lower_bound > energy["total_j"] + config.solver.energy_acceptance_tolerance_j
                timing["candidates_screened_out"] += int(screened_out)
            if candidate_kinematic_ok and candidate_nontrivial and not screened_out:
                candidate_w, candidate_s, candidate_results = _timed_cov(
                    candidate.q, candidate.v, users, target_means, target_covariances,
                    config.physical, config.solver, config.scheme,
                    seed=config.solver.random_seed + 200_000 * (outer + 1),
                    allow_slack=False,
                    perfect_channel_errors=perfect_channel_errors,
                    perfect_target_states=perfect_target_states,
                    checkpoint_dir=state_dir,
                    checkpoint_label=f"outer_{outer:03d}_candidate",
                    resume=(config.solver.resume_enabled and not config.solver.rerun_failed),
                )
                solver_rows.extend(_solver_rows(candidate_results, outer, "candidate"))
            candidate_covariance_ok = _covariance_success(
                candidate_results, config.mission.slots, config
            )
            candidate_rank_ok = False
            candidate_psd_ok = False
            candidate_enforcement_ok = False
            candidate_energy_ok = False
            candidate_energy = np.inf
            candidate_breakdown = {"propulsion_j": np.nan, "rf_j": np.nan}
            if candidate_kinematic_ok and candidate_covariance_ok:
                candidate_rank = max(
                    (item.rank_residual for item in candidate_results), default=np.inf
                )
                candidate_enforcement = max(
                    (
                        item.pcrb_residual_m2 for item in candidate_results
                        if np.isfinite(item.pcrb_residual_m2)
                    ),
                    default=float("nan"),
                )
                candidate_rank_ok = candidate_rank <= config.solver.rank_tolerance
                candidate_psd_ok = all(item.psd_ok for item in candidate_results)
                candidate_enforcement_ok = (
                    np.isfinite(candidate_enforcement)
                    and candidate_enforcement <= config.solver.pcrb_residual_tol_m2 + 1e-9
                )
                candidate_power = np.trace(
                    candidate_w.sum(axis=1) + candidate_s, axis1=1, axis2=2
                ).real
                candidate_breakdown = energy_breakdown(
                    candidate, config.mission, config.physical, candidate_power
                )
                candidate_energy = candidate_breakdown["total_j"]
                candidate_energy_ok = (
                    candidate_energy
                    <= energy["total_j"] + config.solver.energy_acceptance_tolerance_j
                )
            candidate_accepted = all([
                candidate_kinematic_ok,
                candidate_covariance_ok,
                candidate_rank_ok,
                candidate_psd_ok,
                candidate_enforcement_ok,
                candidate_energy_ok,
            ])
            rejection = []
            if screened_out:
                rejection.append("screened_lower_bound")
            if not candidate_nontrivial:
                rejection.append("trivial_candidate")
            if not candidate_kinematic_ok:
                rejection.append("kinematics")
            if not candidate_covariance_ok:
                rejection.append("covariance")
            if candidate_covariance_ok and not candidate_rank_ok:
                rejection.append("rank")
            if candidate_covariance_ok and not candidate_psd_ok:
                rejection.append("psd")
            if candidate_covariance_ok and not candidate_enforcement_ok:
                rejection.append("pcrb_enforcement")
            if candidate_covariance_ok and not candidate_energy_ok:
                rejection.append("energy")
            iteration_rows.append({
                "outer_iteration": outer,
                "accepted": candidate_accepted,
                "total_energy_j": candidate_energy,
                "propulsion_energy_j": candidate_breakdown["propulsion_j"],
                "rf_energy_j": candidate_breakdown["rf_j"],
                "trajectory_penalty": update.penalty,
                "trajectory_status": update.status,
                "trajectory_backtracking_factor": update.backtracking_factor,
                "candidate_kinematic_ok": candidate_kinematic_ok,
                "candidate_covariance_ok": candidate_covariance_ok,
                "candidate_rank_ok": candidate_rank_ok,
                "candidate_psd_ok": candidate_psd_ok,
                "candidate_enforcement_ok": candidate_enforcement_ok,
                "candidate_energy_ok": candidate_energy_ok,
                "candidate_rejection_reasons": ";".join(rejection),
                "candidate_displacement_m": candidate_displacement_m,
                "candidate_energy_lower_bound_j": candidate_lower_bound,
                "candidate_nontrivial": candidate_nontrivial,
                "candidate_horizontal_speed_margin": candidate_kin["horizontal_speed"],
                "candidate_acceleration_margin": candidate_kin["acceleration"],
            })
            if candidate_accepted and candidate_nontrivial:
                accepted_trajectory_updates += 1
                last_gain_rel = (energy["total_j"] - candidate_energy) / max(abs(energy["total_j"]), 1.0)
                trajectory = candidate
                accepted_w, accepted_s = candidate_w, candidate_s
                accepted_results = candidate_results
                previous_energy = candidate_energy
                reuse_state = (candidate_w, candidate_s, candidate_results)
            elif candidate_accepted:
                # A numerically identical candidate is not a trajectory update.
                should_stop = True
            else:
                should_stop = True

        assert accepted_w is not None and accepted_s is not None
        save_ao_state(
            run_dir,
            next_outer_iteration=outer + 1,
            trajectory=trajectory,
            w_all=accepted_w,
            s_all=accepted_s,
            results=accepted_results,
            iteration_rows=iteration_rows,
            solver_rows=solver_rows,
            previous_energy=previous_energy,
            accepted_trajectory_updates=accepted_trajectory_updates,
            restoration_summary=restoration_summary,
        )
        if should_stop:
            break

    if accepted_w is None or accepted_s is None or not accepted_results:
        code = FailureCode.UNKNOWN_NUMERICAL_FAILURE.value
        reason = "No accepted covariance state was available after AO."
        write_failure(run_dir, config, reason, solver_rows, failure_code=code)
        return RunResult(False, run_dir, trajectory, None, None, None, reason, code)

    refinement_summary: dict = {}
    if config.solver.refinement_enabled and config.scheme.chance_sensing:
        from .refinement import refine_sensing
        ref = refine_sensing(
            trajectory, accepted_w, accepted_s, accepted_results, users, target_means,
            target_covariances, config, seed=config.solver.random_seed + 555_555,
            success_fn=lambda res: (
                _covariance_success(res, config.mission.slots, config)
                and max((r.rank_residual for r in res), default=np.inf) <= config.solver.rank_tolerance),
        )
        timing["time_refinement_s"] += ref.runtime_s
        timing["covariance_passes"] += ref.rounds_used
        accepted_w, accepted_s, accepted_results = ref.w, ref.s, ref.results
        pd.DataFrame(ref.rows).to_csv(run_dir / "refinement_log.csv", index=False)
        refinement_summary = {
            "refinement_validated": bool(ref.validated),
            "refinement_rounds_used": int(ref.rounds_used),
            "refinement_points_added": int(ref.points_added),
        }

    t_audit = time.perf_counter()
    audit_samples = (
        min(500, config.solver.out_of_sample_count)
        if quick_audit else config.solver.out_of_sample_count
    )
    channel_samples = (
        min(100, config.solver.channel_audit_samples)
        if quick_audit else config.solver.channel_audit_samples
    )
    exact_robust = config.mission.slots * len(users) <= 64 or not quick_audit
    enforcement_residual = max(
        (
            item.pcrb_residual_m2 for item in accepted_results
            if np.isfinite(item.pcrb_residual_m2)
        ),
        default=float("nan"),
    )
    audit = audit_solution(
        trajectory, accepted_w, accepted_s, users, target_means, target_covariances,
        config.mission, config.physical, config.solver, config.scheme, config.evaluation,
        seed=config.solver.random_seed + 999_999,
        audit_samples=audit_samples,
        channel_samples=channel_samples,
        exact_robust=exact_robust,
        enforcement_pcrb_residual_m2=enforcement_residual,
    )
    if preserved is not None and not bool(audit.summary.get("accepted", False)):
        inc_traj, inc_w, inc_s, inc_results = preserved
        inc_enforcement = max(
            (r.pcrb_residual_m2 for r in inc_results if np.isfinite(r.pcrb_residual_m2)),
            default=float("nan"),
        )
        inc_audit = audit_solution(
            inc_traj, inc_w, inc_s, users, target_means, target_covariances,
            config.mission, config.physical, config.solver, config.scheme, config.evaluation,
            seed=config.solver.random_seed + 999_999,
            audit_samples=audit_samples, channel_samples=channel_samples,
            exact_robust=exact_robust, enforcement_pcrb_residual_m2=inc_enforcement,
        )
        if bool(inc_audit.summary.get("accepted", False)):
            trajectory, accepted_w, accepted_s, accepted_results, audit = (
                inc_traj, inc_w.copy(), inc_s.copy(), list(inc_results), inc_audit
            )
            enforcement_residual = inc_enforcement
            preserved_used = True
            solver_rows.extend(_solver_rows(inc_results, config.solver.outer_iterations, "preserved_incumbent_final_audit_guard"))

    timing["time_audit_s"] += time.perf_counter() - t_audit
    timing["wall_total_s"] = time.perf_counter() - wall_start
    max_seed_displacement = float(
        np.max(np.linalg.norm(trajectory.q - initial_seed_trajectory.q, axis=1))
    )
    inaccurate_count = sum(item.status == "optimal_inaccurate" for item in accepted_results)
    final_slots = max(len(accepted_results), 1)
    audit.summary.update({
        **restoration_summary,
        "accepted_trajectory_updates": int(accepted_trajectory_updates),
        "warm_start_used": bool(covariance_warm_start is not None),
        "warm_start_label": str(warm_start_label),
        "max_displacement_from_initial_seed_m": max_seed_displacement,
        "trajectory_changed": bool(
            max_seed_displacement
            >= config.solver.trajectory_activation_min_displacement_m
        ),
        "final_solver_slots": int(len(accepted_results)),
        "final_optimal_inaccurate_count": int(inaccurate_count),
        "final_optimal_inaccurate_fraction": float(inaccurate_count / final_slots),
        "final_max_solver_iterations": int(
            max((item.solver_iterations for item in accepted_results), default=0)
        ),
        "solver_numerical_warning": bool(
            inaccurate_count / final_slots
            > config.solver.max_final_inaccurate_fraction
        ),
        "feasible_incumbent_available": bool(preserved is not None),
        "feasible_incumbent_used": bool(preserved_used),
        "feasible_incumbent_label": str(preserved_label),
        "feasible_incumbent_energy_j": float(preserved_energy) if np.isfinite(preserved_energy) else np.nan,
        "failure_code": "",
        **timing,
        **revision_info,
        **refinement_summary,
        "last_accepted_gain_rel": float(last_gain_rel) if np.isfinite(last_gain_rel) else np.nan,
        "stopping_rule": config.solver.stopping_rule,
        "pcrb_formulation": config.solver.pcrb_formulation,
        "pcrb_enforcement": config.solver.pcrb_enforcement,
    })
    write_outputs(
        run_dir, config, trajectory, accepted_w, accepted_s, audit,
        iteration_rows, solver_rows
    )
    (run_dir / "state" / "COMPLETED").write_text("accepted\n", encoding="utf-8")
    if make_plots:
        plot_run(run_dir)
    return RunResult(True, run_dir, trajectory, accepted_w, accepted_s, audit)


def _energy_lower_bound(candidate: Trajectory, users: np.ndarray, target_means: np.ndarray,
                        target_covariances: np.ndarray, config: SimulationConfig) -> float:
    """E_lb of Proposition 4 on a candidate trajectory (no conic solve)."""
    from .covariance import build_enforcement_sets
    plb = np.zeros(config.mission.slots)
    for slot in range(config.mission.slots):
        sets = build_enforcement_sets(
            candidate.q[slot], target_means[:, slot], target_covariances[:, slot],
            config.physical, config.solver, config.scheme,
            np.random.default_rng(config.solver.random_seed + 1009 * slot), k_users=len(users))
        pmins = [theory.pmin_for_points(candidate.q[slot], candidate.v[slot], es.points,
                                        target_covariances[l, slot], config.physical,
                                        es.threshold_m2, es.range_override_m)[0]
                 for l, es in enumerate(sets)]
        plb[slot] = theory.rf_lower_bound_slot(candidate.q[slot], users, pmins, config.physical)
    prop = energy_breakdown(candidate, config.mission, config.physical)["propulsion_j"]
    return theory.energy_lower_bound(prop, plb, config.mission, config.physical)


def _slack_value(results: list[CovarianceSlotResult]) -> float:
    total = 0.0
    for item in results:
        c = item.max_comm_slack if np.isfinite(item.max_comm_slack) else 1e6
        s = item.max_sensing_slack if np.isfinite(item.max_sensing_slack) else 1e6
        total += c + s
    return float(total)


def _phase1(trajectory: Trajectory, users: np.ndarray, target_means: np.ndarray,
            target_covariances: np.ndarray, config: SimulationConfig, outer: int,
            cov_fn, checkpoint_dir=None, resume: bool = False) -> tuple[Trajectory | None, list[dict]]:
    """Slack-based Phase I: move the trajectory to reduce the covariance slack Phi.

    Candidates come from the same trajectory block as the energy phase, with the
    slack-problem covariances as incumbent; a candidate is kept if its re-solved
    slack value is smaller. Returns the restored trajectory if Phi reaches ~0.
    Each slack pass is checkpointed per slot; a checkpoint is reused only when the
    seed and the trajectory hash match, so resuming does not change the result.
    """
    rows: list[dict] = []
    current = trajectory

    def ckpt(label: str) -> dict:
        if checkpoint_dir is None:
            return {}
        return {"checkpoint_dir": checkpoint_dir, "resume": resume,
                "checkpoint_label": f"outer_{outer:03d}_{label}"}

    w, s, res = cov_fn(current.q, current.v, users, target_means, target_covariances,
                       config.physical, config.solver, config.scheme,
                       seed=config.solver.random_seed + 100_000 * outer + 991, allow_slack=True,
                       **ckpt("phase1_start"))
    phi = _slack_value(res)
    rows.append({"outer_iteration": outer, "trajectory_status": "phase1_start", "phase1_phi": phi})
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(s))):
        return None, rows
    for it in range(int(config.solver.phase1_iterations)):
        if phi <= 1e-6:
            break
        upd = update_trajectory(current, w, s, users, target_means, target_covariances,
                                config.mission, config.physical, config.solver, config.scheme,
                                seed=config.solver.random_seed + 700_000 + 10 * outer + it)
        cand = upd.trajectory
        if float(np.max(np.linalg.norm(cand.q - current.q, axis=1))) <= 1e-6:
            rows.append({"outer_iteration": outer, "trajectory_status": f"phase1_{it}_no_move",
                         "phase1_phi": phi})
            break
        cw, cs, cres = cov_fn(cand.q, cand.v, users, target_means, target_covariances,
                              config.physical, config.solver, config.scheme,
                              seed=config.solver.random_seed + 100_000 * outer + 992 + it,
                              allow_slack=True, **ckpt(f"phase1_{it}"))
        cphi = _slack_value(cres)
        accepted = cphi < phi
        rows.append({"outer_iteration": outer, "trajectory_status": f"phase1_{it}",
                     "phase1_phi": cphi, "accepted": accepted})
        if accepted:
            current, w, s, phi = cand, cw, cs, cphi
        else:
            break
    return (current if phi <= 1e-6 else None), rows


def _enforcement_sets(trajectory: Trajectory, slot: int, users: np.ndarray, target_means: np.ndarray,
                      target_covariances: np.ndarray, config: SimulationConfig):
    """Enforcement sets exactly as built by the first covariance pass."""
    from .covariance import build_enforcement_sets
    return build_enforcement_sets(
        trajectory.q[slot], target_means[:, slot], target_covariances[:, slot],
        config.physical, config.solver, config.scheme,
        np.random.default_rng(config.solver.random_seed + 1009 * slot), k_users=len(users))
