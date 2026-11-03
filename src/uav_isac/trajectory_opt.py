from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .channel import robust_quadratic_matrix
from .config import MissionConfig, PhysicalConfig, SchemeConfig, SolverConfig
from .dynamics import (
    Trajectory,
    energy_breakdown,
    enforce_terminal_with_last_two,
    kinematic_residuals,
    reconstruct_from_accelerations,
)
from .geometry import csi_shape, hermitian_sqrt, nominal_channel
from .sensing import confidence_points, pcrb, worst_case_range


@dataclass
class TrajectoryUpdateResult:
    trajectory: Trajectory
    objective: float
    status: str
    runtime_s: float
    iterations: int
    penalty: float
    kinematic_feasible: bool
    backtracking_factor: float


def _cheap_robust_quadratic_lower_bound(q: np.ndarray, user: np.ndarray,
                                        w: np.ndarray, s: np.ndarray,
                                        user_index: int, physical: PhysicalConfig) -> float:
    """Conservative lower bound on h^H A h-Gamma sigma^2 over the ellipsoid."""
    h_hat = nominal_channel(q, user, mx=physical.mx, my=physical.my,
                            wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                            pathloss_exponent=physical.pathloss_exponent)
    c_sqrt = hermitian_sqrt(csi_shape(h_hat, physical.csi_error_radius, physical.csi_model))
    a = robust_quadratic_matrix(w, s, user_index, physical.gamma_min)
    bmat = c_sqrt.conj().T @ a @ c_sqrt
    bvec = c_sqrt.conj().T @ a @ h_hat
    center = float(np.vdot(h_hat, a @ h_hat).real - physical.gamma_min * physical.user_noise_w)
    lambda_min = float(np.min(np.linalg.eigvalsh((bmat + bmat.conj().T) / 2.0)).real)
    return center - 2.0 * float(np.linalg.norm(bvec)) + min(0.0, lambda_min)


def _trajectory_penalty(traj: Trajectory, w_all: np.ndarray, s_all: np.ndarray,
                        users: np.ndarray, target_means: np.ndarray,
                        target_covariances: np.ndarray, mission: MissionConfig,
                        physical: PhysicalConfig, solver_cfg: SolverConfig,
                        scheme: SchemeConfig, rng: np.random.Generator,
                        reference_q: np.ndarray,
                        comm_floors: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    residuals = kinematic_residuals(traj, mission, physical)
    violations = []
    for key in ("altitude_low", "altitude_high", "horizontal_speed", "vertical_speed", "acceleration"):
        violations.append(max(0.0, -residuals[key]))
    violations.extend([residuals["terminal_position"], residuals["terminal_velocity"], residuals["dynamics"]])

    comm_v = 0.0
    if scheme.robust_communication and physical.csi_error_radius > 0:
        # ``_cheap_robust_quadratic_lower_bound`` is a conservative surrogate for the
        # exact S-procedure LMI. At an SDP-certified feasible incumbent it reports
        # every (slot, user) pair as violated (measured: 192/192 pairs negative,
        # worst -0.21, while the exact audit gives a worst-case SINR margin of
        # +0.108). Penalizing that offset makes the merit function 5e5 * 1.056 =
        # 5.3e5 at the incumbent -- 21x the entire energy objective -- so the search
        # chases a phantom violation and cannot see energy at all. Anchoring at the
        # incumbent value penalizes only degradation below where the search started.
        # Genuine feasibility of any returned candidate is unaffected: it is decided
        # downstream by the covariance re-solve and the kinematic/rank/PSD/
        # enforcement/energy gates in ao_solver.
        for n in range(mission.slots):
            for k, user in enumerate(users):
                margin = _cheap_robust_quadratic_lower_bound(
                    traj.q[n], user, w_all[n], s_all[n], k, physical
                )
                normalized = margin / max(physical.gamma_min * physical.user_noise_w, 1e-18)
                floor = 0.0 if comm_floors is None else min(0.0, float(comm_floors[n, k]))
                comm_v += max(0.0, floor - normalized) ** 2
    else:
        for n in range(mission.slots):
            for k, user in enumerate(users):
                h = nominal_channel(traj.q[n], user, mx=physical.mx, my=physical.my,
                                    wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                                    pathloss_exponent=physical.pathloss_exponent)
                a = robust_quadratic_matrix(w_all[n], s_all[n], k, physical.gamma_min)
                margin = float(np.vdot(h, a @ h).real - physical.gamma_min * physical.user_noise_w)
                comm_v += max(0.0, -margin / max(physical.gamma_min * physical.user_noise_w, 1e-18)) ** 2

    pcrb_v = 0.0
    for n in range(mission.slots):
        rx = w_all[n].sum(axis=0) + s_all[n]
        for l in range(target_means.shape[0]):
            mean = target_means[l, n]
            covariance = target_covariances[l, n]
            mode = "mean" if not scheme.chance_sensing else "principal_axes"
            points = confidence_points(mean, covariance, physical.confidence_radius, mode=mode, rng=rng)
            r_wc = worst_case_range(traj.q[n], mean, covariance, physical.confidence_radius) if scheme.chance_sensing else None
            threshold = physical.pcrb_threshold_m2 * (1.0 - physical.enforcement_margin if scheme.chance_sensing else 1.0)
            for xi in points:
                value = pcrb(traj.q[n], traj.v[n], xi, covariance, rx, physical, range_override_m=r_wc)
                pcrb_v += max(0.0, value / threshold - 1.0) ** 2

    kinematic_v = sum(x * x for x in violations)
    displacement = np.linalg.norm(traj.q - reference_q, axis=1)
    trust_v = np.sum(np.maximum(displacement - 35.0, 0.0) ** 2) / 35.0**2
    total_penalty = kinematic_v + comm_v + pcrb_v + trust_v
    return total_penalty, {
        "kinematic": kinematic_v,
        "communication": comm_v,
        "sensing": pcrb_v,
        "trust": trust_v,
    }


def _apply_fixed_altitude(candidate: Trajectory, scheme: SchemeConfig) -> Trajectory:
    if scheme.fixed_altitude_m is not None:
        candidate.q[:, 2] = scheme.fixed_altitude_m
        candidate.v[:, 2] = 0.0
        candidate.a[:, 2] = 0.0
    return candidate


def _candidate_from_deltas(base: Trajectory, x: np.ndarray, segments: int,
                           mission: MissionConfig, scheme: SchemeConfig) -> Trajectory:
    """Apply piecewise-constant acceleration increments to a feasible incumbent.

    Zero increments reproduce the incumbent exactly, so SLSQP starts from a feasible
    point. The final two slot accelerations are recomputed to preserve the terminal
    position and velocity equalities.
    """
    deltas = np.asarray(x, dtype=float).reshape(segments, 3)
    first_count = mission.slots - 2
    indices = np.minimum(
        (np.arange(first_count) * segments) // max(first_count, 1), segments - 1
    )
    acceleration = np.asarray(base.a, dtype=float).copy()
    acceleration[:first_count] += deltas[indices]
    acceleration = enforce_terminal_with_last_two(acceleration, mission)
    q, v = reconstruct_from_accelerations(acceleration, mission)
    candidate = Trajectory(q=q, v=v, a=acceleration)
    return _apply_fixed_altitude(candidate, scheme)


def _kinematic_constraint_vector(candidate: Trajectory, physical: PhysicalConfig) -> np.ndarray:
    """Return all pointwise inequality margins; feasible means every entry >= 0."""
    return np.concatenate([
        candidate.q[:, 2] - physical.qz_min_m,
        physical.qz_max_m - candidate.q[:, 2],
        physical.vh_max_mps - np.linalg.norm(candidate.v[:, :2], axis=1),
        physical.vz_max_mps - np.abs(candidate.v[:, 2]),
        physical.a_max_mps2 - np.linalg.norm(candidate.a, axis=1),
    ])


def _is_kinematically_feasible(candidate: Trajectory, mission: MissionConfig,
                               physical: PhysicalConfig, tolerance: float) -> bool:
    residuals = kinematic_residuals(candidate, mission, physical)
    equalities_ok = max(
        residuals["dynamics"], residuals["terminal_position"], residuals["terminal_velocity"]
    ) <= tolerance
    inequalities_ok = min(
        residuals["altitude_low"], residuals["altitude_high"], residuals["horizontal_speed"],
        residuals["vertical_speed"], residuals["acceleration"],
    ) >= -tolerance
    return bool(equalities_ok and inequalities_ok)


def update_trajectory(traj: Trajectory, w_all: np.ndarray, s_all: np.ndarray,
                      users: np.ndarray, target_means: np.ndarray,
                      target_covariances: np.ndarray, mission: MissionConfig,
                      physical: PhysicalConfig, solver_cfg: SolverConfig,
                      scheme: SchemeConfig, seed: int) -> TrajectoryUpdateResult:
    if scheme.fixed_trajectory:
        return TrajectoryUpdateResult(traj, 0.0, "fixed", 0.0, 0, 0.0, True, 0.0)

    segments = min(solver_cfg.trajectory_control_segments, max(1, mission.slots - 2))
    dimension = segments * 3
    x0 = np.zeros(dimension, dtype=float)
    trust = solver_cfg.trajectory_trust_accel_mps2
    lower = np.full_like(x0, -trust)
    upper = np.full_like(x0, trust)
    if scheme.fixed_altitude_m is not None:
        lower[2::3] = 0.0
        upper[2::3] = 0.0
    bounds = list(zip(lower, upper))
    rng = np.random.default_rng(seed)
    reference_q = traj.q.copy()
    tx_power = np.trace(w_all.sum(axis=1) + s_all, axis1=1, axis2=2).real

    # Anchor the conservative communication surrogate at the incumbent, which the
    # covariance SDP has already certified feasible; see _trajectory_penalty.
    comm_floors = None
    if scheme.robust_communication and physical.csi_error_radius > 0:
        comm_floors = np.zeros((mission.slots, users.shape[0]))
        scale = max(physical.gamma_min * physical.user_noise_w, 1e-18)
        for n in range(mission.slots):
            for k, user in enumerate(users):
                comm_floors[n, k] = _cheap_robust_quadratic_lower_bound(
                    traj.q[n], user, w_all[n], s_all[n], k, physical
                ) / scale

    def merit(candidate: Trajectory) -> tuple[float, float]:
        energy = energy_breakdown(candidate, mission, physical, tx_power)["total_j"]
        penalty, _ = _trajectory_penalty(
            candidate, w_all, s_all, users, target_means, target_covariances,
            mission, physical, solver_cfg, scheme, rng, reference_q,
            comm_floors=comm_floors,
        )
        smooth = float(np.sum(np.diff(candidate.a, axis=0) ** 2))
        return (
            energy + solver_cfg.trajectory_penalty * penalty
            + solver_cfg.trajectory_smooth_weight * smooth
        ), penalty

    def objective(x: np.ndarray) -> float:
        candidate = _candidate_from_deltas(traj, x, segments, mission, scheme)
        return merit(candidate)[0]

    def hard_constraints(x: np.ndarray) -> np.ndarray:
        candidate = _candidate_from_deltas(traj, x, segments, mission, scheme)
        return _kinematic_constraint_vector(candidate, physical)

    # Zero is retained as the reproducible incumbent start. Additional small starts
    # are essential for hover/loiter missions: the derivative at exactly zero speed
    # may vanish even though moderate forward motion lowers rotary-wing power.
    starts: list[np.ndarray] = [x0.copy()]
    probe_scale = min(max(float(solver_cfg.trajectory_probe_scale), 0.0), 1.0)
    requested_starts = max(1, int(solver_cfg.trajectory_multistart))
    if requested_starts >= 2:
        # Deterministic northward closed-excursion probe. It is particularly useful
        # for CP5b and remains reproducible across platforms.
        structured = np.zeros((segments, 3), dtype=float)
        # Four-phase acceleration pattern (+, 0, -, -, 0, +) approximated on
        # ``segments`` controls. It creates a closed excursion after terminal
        # reconstruction and backtracking without imposing a constant bias.
        phase = np.linspace(0.0, 1.0, segments)
        pattern = np.interp(phase, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            [1.0, 0.0, -1.0, -1.0, 0.0, 1.0])
        structured[:, 1] = probe_scale * trust * pattern
        if scheme.fixed_altitude_m is not None:
            structured[:, 2] = 0.0
        starts.append(np.clip(structured.reshape(-1), lower, upper))
    for start_id in range(max(0, requested_starts - len(starts))):
        probe = rng.normal(size=dimension)
        probe *= probe_scale * trust / max(np.linalg.norm(probe, ord=np.inf), 1e-12)
        if scheme.fixed_altitude_m is not None:
            probe[2::3] = 0.0
        # Sign alternation promotes closed excursions instead of a constant bias;
        # terminal equalities are still enforced exactly by reconstruction.
        probe *= np.repeat(np.where(np.arange(segments) % 2 == 0, 1.0, -1.0), 3)
        starts.append(np.clip(probe, lower, upper))

    start_time = time.perf_counter()
    method = solver_cfg.trajectory_method.upper()
    optimize_results = []
    if method == "RANDOM_SEARCH":
        # Derivative-free baseline search (R3.3): the same merit, starts, and
        # backtracking as SLSQP, but candidates are random sign-alternating
        # increments with magnitudes drawn uniformly in [0, trust].
        for _ in range(max(0, int(solver_cfg.random_search_samples))):
            probe = rng.normal(size=dimension)
            probe *= rng.uniform(0.0, 1.0) * trust / max(np.linalg.norm(probe, ord=np.inf), 1e-12)
            if scheme.fixed_altitude_m is not None:
                probe[2::3] = 0.0
            starts.append(np.clip(probe, lower, upper))
    for initial in (starts if method != "RANDOM_SEARCH" else []):
        if method == "SLSQP":
            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": hard_constraints}],
                options={
                    "maxiter": solver_cfg.trajectory_powell_maxiter,
                    "ftol": 2e-4,
                    "disp": False,
                },
            )
        else:
            result = minimize(
                objective,
                initial,
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": solver_cfg.trajectory_powell_maxiter,
                    "xtol": 2e-3,
                    "ftol": 2e-4,
                    "disp": False,
                },
            )
        optimize_results.append(result)
    runtime = time.perf_counter() - start_time

    feasible_trials: list[tuple[float, float, float, Trajectory, object]] = []
    # Include the raw starting probes as well as every optimizer output. This makes
    # the activation test robust to a local solver that returns immediately at x=0.
    trial_vectors: list[tuple[np.ndarray, object]] = [(x, None) for x in starts]
    trial_vectors.extend([
        (np.asarray(r.x if np.all(np.isfinite(r.x)) else x0, dtype=float), r)
        for r in optimize_results
    ])
    for trial_x, source_result in trial_vectors:
        for factor in solver_cfg.trajectory_backtracking_factors:
            x_trial = x0 + float(factor) * (trial_x - x0)
            candidate = _candidate_from_deltas(traj, x_trial, segments, mission, scheme)
            if _is_kinematically_feasible(
                candidate, mission, physical, solver_cfg.trajectory_feasibility_tolerance
            ):
                value, penalty = merit(candidate)
                feasible_trials.append((value, penalty, float(factor), candidate, source_result))

    incumbent_value, incumbent_penalty = merit(traj)
    feasible_trials.append((incumbent_value, incumbent_penalty, 0.0, traj, None))
    best_value, best_penalty, best_factor, best_candidate, best_result = min(
        feasible_trials, key=lambda item: item[0]
    )
    feasible = _is_kinematically_feasible(
        best_candidate, mission, physical, solver_cfg.trajectory_feasibility_tolerance
    )
    if best_result is None:
        status_prefix = "probe_or_incumbent"
        iterations = 0
    else:
        status_prefix = "success" if best_result.success else f"stopped:{best_result.message}"
        iterations = int(getattr(best_result, "nit", 0) or 0)
    if best_factor == 0.0:
        status_prefix += "|incumbent_returned"
    elif best_factor < 1.0:
        status_prefix += "|feasible_backtracking"
    status_prefix += f"|multistart={len(starts)}"

    return TrajectoryUpdateResult(
        trajectory=best_candidate,
        objective=float(best_value),
        status=status_prefix,
        runtime_s=runtime,
        iterations=iterations,
        penalty=float(best_penalty),
        kinematic_feasible=feasible,
        backtracking_factor=float(best_factor),
    )
