from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .channel import nominal_channel
from .config import MissionConfig, PhysicalConfig, SchemeConfig, SimulationConfig, SolverConfig
from .dynamics import (
    Trajectory,
    enforce_terminal_with_last_two,
    kinematic_residuals,
    minimum_jerk_seed,
    reconstruct_from_accelerations,
)
from .geometry import upa_steering
from .sensing import confidence_points, pcrb, propagate_targets, worst_case_range


@dataclass
class FeasibilityReport:
    rows: list[dict]
    summary: dict


@dataclass
class RestorationResult:
    trajectory: Trajectory
    initial_worst_ratio: float
    final_worst_ratio: float
    feasible_bound: bool
    status: str
    runtime_s: float
    candidates_tested: int
    optimization_iterations: int
    report: FeasibilityReport



def _build_enforcement_sets_local(
    q: np.ndarray, target_means: np.ndarray, target_covariances: np.ndarray,
    physical: PhysicalConfig, solver_cfg: SolverConfig, scheme: SchemeConfig,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, float | None, float]]:
    sets: list[tuple[np.ndarray, float | None, float]] = []
    for target in range(target_means.shape[0]):
        mean = target_means[target]
        covariance = target_covariances[target]
        mode = "mean" if not scheme.chance_sensing else solver_cfg.pcrb_enforcement
        points = confidence_points(
            mean, covariance, physical.confidence_radius, mode=mode,
            scenario_count=solver_cfg.scenario_samples, rng=rng,
            extra_boundary_points=solver_cfg.extra_boundary_points,
        )
        range_override = (
            worst_case_range(q, mean, covariance, physical.confidence_radius)
            if scheme.chance_sensing else None
        )
        threshold = physical.pcrb_threshold_m2
        if scheme.chance_sensing:
            threshold *= 1.0 - physical.enforcement_margin
        sets.append((np.asarray(points, dtype=float), range_override, threshold))
    return sets

def _full_power_rx(q: np.ndarray, xi: np.ndarray, physical: PhysicalConfig) -> np.ndarray:
    point = np.array([xi[0], xi[1], 0.0], dtype=float)
    steering = upa_steering(q, point, physical.mx, physical.my, physical.wavelength_m)
    return physical.p_max_w * np.outer(steering, steering.conj())


def evaluate_fixed_geometry_bound(
    trajectory: Trajectory,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    *,
    seed: int,
    slot_stride: int = 1,
) -> FeasibilityReport:
    """Evaluate necessary fixed-geometry feasibility bounds.

    The sensing bound gives each target point the entire transmit-power budget in its
    own steering direction and ignores communication. Therefore a ratio above one is
    a rigorous certificate that the joint covariance problem is infeasible at that
    slot and geometry. The communication bound serves each user alone with matched
    beamforming and ignores interference and CSI uncertainty; a ratio above one is
    likewise a necessary-condition violation.
    """
    rows: list[dict] = []
    worst_sensing = 0.0
    worst_comm = 0.0
    sensing_violations = 0
    comm_violations = 0
    checked_points = 0
    checked_users = 0
    n_slots = trajectory.q.shape[0]
    stride = max(1, int(slot_stride))
    for slot in range(0, n_slots, stride):
        rng = np.random.default_rng(seed + 1009 * slot)
        from .covariance import build_enforcement_sets
        enforcement_sets = [
            (es.points, es.range_override_m, es.threshold_m2)
            for es in build_enforcement_sets(
                trajectory.q[slot], target_means[:, slot], target_covariances[:, slot],
                physical, solver_cfg, scheme, rng, k_users=len(users),
            )
        ]
        for target, (points, range_override, threshold) in enumerate(enforcement_sets):
            for point_id, xi in enumerate(points):
                rx = _full_power_rx(trajectory.q[slot], xi, physical)
                value = pcrb(
                    trajectory.q[slot], trajectory.v[slot], xi,
                    target_covariances[target, slot], rx, physical,
                    range_override_m=range_override,
                )
                ratio = value / max(threshold, 1e-12)
                checked_points += 1
                sensing_violations += int(ratio > 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance)
                worst_sensing = max(worst_sensing, ratio)
                rows.append({
                    "bound_type": "sensing_full_power",
                    "slot": slot,
                    "entity": target,
                    "point_id": point_id,
                    "value": value,
                    "threshold": threshold,
                    "ratio": ratio,
                    "necessary_condition_pass": ratio <= 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance,
                    "qx_m": trajectory.q[slot, 0],
                    "qy_m": trajectory.q[slot, 1],
                    "qz_m": trajectory.q[slot, 2],
                })
        for user_id, user in enumerate(users):
            h = nominal_channel(
                trajectory.q[slot], user,
                mx=physical.mx, my=physical.my,
                wavelength_m=physical.wavelength_m,
                beta0=physical.beta0,
                pathloss_exponent=physical.pathloss_exponent,
            )
            isolated_snr = physical.p_max_w * float(np.vdot(h, h).real) / max(physical.user_noise_w, 1e-30)
            ratio = physical.gamma_min / max(isolated_snr, 1e-30)
            checked_users += 1
            comm_violations += int(ratio > 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance)
            worst_comm = max(worst_comm, ratio)
            rows.append({
                "bound_type": "communication_isolated_mrt",
                "slot": slot,
                "entity": user_id,
                "point_id": -1,
                "value": isolated_snr,
                "threshold": physical.gamma_min,
                "ratio": ratio,
                "necessary_condition_pass": ratio <= 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance,
                "qx_m": trajectory.q[slot, 0],
                "qy_m": trajectory.q[slot, 1],
                "qz_m": trajectory.q[slot, 2],
            })
    worst = max(worst_sensing, worst_comm)
    summary = {
        "checked_slots": int(len(range(0, n_slots, stride))),
        "checked_sensing_points": int(checked_points),
        "checked_users": int(checked_users),
        "sensing_bound_violations": int(sensing_violations),
        "communication_bound_violations": int(comm_violations),
        "worst_sensing_ratio": float(worst_sensing),
        "worst_communication_ratio": float(worst_comm),
        "worst_necessary_condition_ratio": float(worst),
        "fixed_geometry_necessary_conditions_pass": bool(
            worst <= 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance
        ),
        "interpretation": (
            "A ratio above one certifies fixed-geometry infeasibility for the corresponding "
            "necessary condition. A ratio below one does not prove joint feasibility."
        ),
    }
    return FeasibilityReport(rows=rows, summary=summary)


def write_feasibility_report(report: FeasibilityReport, output_dir: str | Path,
                             prefix: str = "feasibility_bound") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = output_dir / f"{prefix}.csv"
    summary = output_dir / f"{prefix}_summary.csv"
    pd.DataFrame(report.rows).to_csv(detail, index=False)
    pd.DataFrame([report.summary]).to_csv(summary, index=False)
    return detail, summary


def _candidate_from_delta(base: Trajectory, x: np.ndarray, segments: int,
                          mission: MissionConfig, fixed_altitude_m: float | None) -> Trajectory:
    controls = np.asarray(x, dtype=float).reshape(segments, 3)
    first_count = mission.slots - 2
    indices = np.minimum(
        (np.arange(first_count) * segments) // max(first_count, 1), segments - 1
    )
    acceleration = np.asarray(base.a, dtype=float).copy()
    acceleration[:first_count] += controls[indices]
    acceleration = enforce_terminal_with_last_two(acceleration, mission)
    q, v = reconstruct_from_accelerations(acceleration, mission)
    if fixed_altitude_m is not None:
        q[:, 2] = fixed_altitude_m
        v[:, 2] = 0.0
        acceleration[:, 2] = 0.0
    return Trajectory(q=q, v=v, a=acceleration)


def _kinematic_vector(traj: Trajectory, physical: PhysicalConfig) -> np.ndarray:
    return np.concatenate([
        traj.q[:, 2] - physical.qz_min_m,
        physical.qz_max_m - traj.q[:, 2],
        physical.vh_max_mps - np.linalg.norm(traj.v[:, :2], axis=1),
        physical.vz_max_mps - np.abs(traj.v[:, 2]),
        physical.a_max_mps2 - np.linalg.norm(traj.a, axis=1),
    ])


def _kinematic_ok(traj: Trajectory, mission: MissionConfig, physical: PhysicalConfig,
                  tolerance: float) -> bool:
    residuals = kinematic_residuals(traj, mission, physical)
    return bool(
        max(residuals["dynamics"], residuals["terminal_position"], residuals["terminal_velocity"])
        <= tolerance
        and min(
            residuals["altitude_low"], residuals["altitude_high"],
            residuals["horizontal_speed"], residuals["vertical_speed"],
            residuals["acceleration"],
        ) >= -tolerance
    )


def restore_feasible_trajectory(
    initial: Trajectory,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    config: SimulationConfig,
    *,
    seed: int,
) -> RestorationResult:
    """Phase F0: reduce fixed-geometry feasibility violations before energy AO.

    This phase does not claim joint feasibility. It only searches for a dynamically
    feasible trajectory whose necessary-condition bound is no longer violated. The
    covariance SDP remains the authoritative feasibility test.
    """
    start_time = time.perf_counter()
    mission, physical, solver_cfg, scheme = (
        config.mission, config.physical, config.solver, config.scheme
    )
    initial_report = evaluate_fixed_geometry_bound(
        initial, users, target_means, target_covariances, physical, solver_cfg, scheme,
        seed=seed,
    )
    initial_ratio = float(initial_report.summary["worst_necessary_condition_ratio"])
    if initial_report.summary["fixed_geometry_necessary_conditions_pass"]:
        return RestorationResult(
            trajectory=initial,
            initial_worst_ratio=initial_ratio,
            final_worst_ratio=initial_ratio,
            feasible_bound=True,
            status="initial_bound_pass",
            runtime_s=time.perf_counter() - start_time,
            candidates_tested=1,
            optimization_iterations=0,
            report=initial_report,
        )
    if not solver_cfg.feasibility_restoration_enabled or scheme.fixed_trajectory:
        return RestorationResult(
            trajectory=initial,
            initial_worst_ratio=initial_ratio,
            final_worst_ratio=initial_ratio,
            feasible_bound=bool(initial_report.summary["fixed_geometry_necessary_conditions_pass"]),
            status="disabled_or_fixed_trajectory",
            runtime_s=time.perf_counter() - start_time,
            candidates_tested=1,
            optimization_iterations=0,
            report=initial_report,
        )

    midpoint = mission.slots // 2
    target_centroid = np.mean(target_means[:, midpoint, :2], axis=0)
    user_centroid = np.mean(users[:, :2], axis=0)
    combined_centroid = 0.5 * (target_centroid + user_centroid)
    attraction_points = [target_centroid, combined_centroid, user_centroid]
    candidates: list[tuple[float, Trajectory, FeasibilityReport, str]] = []

    def add_candidate(traj: Trajectory, label: str) -> None:
        if not _kinematic_ok(
            traj, mission, physical, solver_cfg.trajectory_feasibility_tolerance
        ):
            return
        report = evaluate_fixed_geometry_bound(
            traj, users, target_means, target_covariances, physical, solver_cfg, scheme,
            seed=seed,
        )
        ratio = float(report.summary["worst_necessary_condition_ratio"])
        candidates.append((ratio, traj, report, label))

    add_candidate(initial, "initial")
    for attraction in attraction_points:
        for strength in solver_cfg.feasibility_restoration_attraction_strengths:
            candidate = minimum_jerk_seed(
                mission, physical,
                attraction_xy=attraction,
                attraction_strength=float(strength),
                altitude_m=scheme.fixed_altitude_m,
            )
            add_candidate(candidate, f"attraction_{float(strength):.3f}")

    if not candidates:
        return RestorationResult(
            trajectory=initial,
            initial_worst_ratio=initial_ratio,
            final_worst_ratio=initial_ratio,
            feasible_bound=False,
            status="no_kinematically_feasible_candidate",
            runtime_s=time.perf_counter() - start_time,
            candidates_tested=0,
            optimization_iterations=0,
            report=initial_report,
        )

    candidates.sort(key=lambda item: item[0])
    best_ratio, best_traj, best_report, best_label = candidates[0]
    segments = min(
        max(1, int(solver_cfg.feasibility_restoration_segments)), mission.slots - 2
    )
    dimension = 3 * segments
    trust = max(float(solver_cfg.trajectory_trust_accel_mps2), 0.25)
    bounds = [(-trust, trust)] * dimension
    if scheme.fixed_altitude_m is not None:
        for idx in range(2, dimension, 3):
            bounds[idx] = (0.0, 0.0)

    eval_stride = max(1, mission.slots // 12)

    def objective(x: np.ndarray) -> float:
        candidate = _candidate_from_delta(
            best_traj, x, segments, mission, scheme.fixed_altitude_m
        )
        report = evaluate_fixed_geometry_bound(
            candidate, users, target_means, target_covariances,
            physical, solver_cfg, scheme, seed=seed, slot_stride=eval_stride,
        )
        ratio = float(report.summary["worst_necessary_condition_ratio"])
        smooth = float(np.sum(np.diff(candidate.a, axis=0) ** 2))
        return ratio + solver_cfg.feasibility_restoration_smooth_weight * smooth

    def constraints(x: np.ndarray) -> np.ndarray:
        candidate = _candidate_from_delta(
            best_traj, x, segments, mission, scheme.fixed_altitude_m
        )
        return _kinematic_vector(candidate, physical)

    rng = np.random.default_rng(seed + 8171)
    starts = [np.zeros(dimension)]
    for _ in range(max(0, int(solver_cfg.feasibility_restoration_multistart) - 1)):
        x = rng.normal(size=dimension)
        x *= 0.35 * trust / max(np.linalg.norm(x, ord=np.inf), 1e-12)
        if scheme.fixed_altitude_m is not None:
            x[2::3] = 0.0
        starts.append(x)

    total_iterations = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints}],
            options={
                "maxiter": int(solver_cfg.feasibility_restoration_maxiter),
                "ftol": 1e-4,
                "disp": False,
            },
        )
        total_iterations += int(getattr(result, "nit", 0) or 0)
        x = np.asarray(result.x if np.all(np.isfinite(result.x)) else start)
        candidate = _candidate_from_delta(
            best_traj, x, segments, mission, scheme.fixed_altitude_m
        )
        add_candidate(candidate, f"slsqp:{result.message}")
        candidates.sort(key=lambda item: item[0])
        best_ratio, best_traj, best_report, best_label = candidates[0]
        if best_ratio <= 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance:
            break

    return RestorationResult(
        trajectory=best_traj,
        initial_worst_ratio=initial_ratio,
        final_worst_ratio=float(best_ratio),
        feasible_bound=bool(
            best_ratio <= 1.0 + solver_cfg.feasibility_restoration_ratio_tolerance
        ),
        status=best_label,
        runtime_s=time.perf_counter() - start_time,
        candidates_tested=len(candidates),
        optimization_iterations=total_iterations,
        report=best_report,
    )


def evaluate_config_seed(config: SimulationConfig) -> tuple[Trajectory, FeasibilityReport]:
    users = np.asarray(config.mission.users_m, dtype=float)
    target_means, target_covariances = propagate_targets(config.mission, config.physical)
    seed_traj = minimum_jerk_seed(
        config.mission, config.physical, altitude_m=config.scheme.fixed_altitude_m
    )
    report = evaluate_fixed_geometry_bound(
        seed_traj, users, target_means, target_covariances,
        config.physical, config.solver, config.scheme,
        seed=config.solver.random_seed,
    )
    return seed_traj, report
