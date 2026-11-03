from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import cvxpy as cp
import numpy as np

from .config import PhysicalConfig, SolverConfig, SchemeConfig
from .geometry import csi_shape, hermitian_sqrt, nominal_channel, upa_steering
from .sensing import (
    confidence_points,
    measurement_calibration,
    measurement_jacobian,
    pcrb,
    worst_case_range,
)


@dataclass
class EnforcementSet:
    """Frozen PCRB enforcement data for one target in one slot."""
    points: np.ndarray            # (P, 4) target states at which the LMI is imposed
    range_override_m: float | None
    threshold_m2: float
    # Filled lazily for the linear form (Prop. 2): P_min per point and steering rows.
    pmin_w: np.ndarray | None = None
    steering: np.ndarray | None = None
    # Constraint-generation working set (indices into ``points``), kept across the
    # base solve, rank refinements, fallback, and recovery of the same slot.
    working: np.ndarray | None = None


# Scenario programs (Proposition 3) impose N_s ~ 3e4 rows a_j^H R_x a_j >= P_min,j per
# target. Almost all of them are inactive at the optimum, and a single SCS solve with
# all rows takes hours, so they are generated on demand: solve on a working set, check
# every sample in closed form, add the most violated samples, and repeat. The loop
# stops only when no sample is violated by more than the solver's accuracy on the
# working set, so the result is the optimum of the full scenario program.
SCENARIO_CG_INITIAL = 32
SCENARIO_CG_BATCH = 32
SCENARIO_CG_MAX_ROUNDS = 40
SCENARIO_CG_ABS_TOL_W = 1e-9


@dataclass
class CovarianceSlotResult:
    status: str
    objective_power_w: float
    w: np.ndarray
    s: np.ndarray
    runtime_s: float
    rank_residual: float
    max_comm_slack: float
    max_sensing_slack: float
    solver_iterations: int
    pcrb_residual_m2: float = float("nan")
    solver_used: str = ""
    min_w_eigenvalue: float = float("nan")
    min_s_eigenvalue: float = float("nan")
    psd_ok: bool = False
    # Per-slot accounting across the base solve, fallback, rank refinements, and
    # rank-one recovery. ``runtime_s`` remains the runtime of the selected final
    # solve for backward compatibility; ``total_solver_runtime_s`` is the actual
    # conic-solver time spent on the slot.
    total_solver_runtime_s: float = float("nan")
    solver_call_count: int = 1
    fallback_call_count: int = 0
    rank_refinement_call_count: int = 0
    rank_recovery_call_count: int = 0


def build_enforcement_sets(
    q: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    rng: np.random.Generator,
    perfect_target_states: np.ndarray | None = None,
    k_users: int = 1,
    extra_points: list[np.ndarray] | None = None,
) -> list[EnforcementSet]:
    """Build the finite PCRB enforcement set once per slot.

    ``pcrb_enforcement == "scenario_certified"`` draws the a priori sample size of
    Proposition 3 and enforces the untightened threshold with the sampled range.
    ``extra_points[l]`` appends refinement states (Algorithm 2) for target l.

    Freezing the set here guarantees that the initial SDR solve, every rank-refinement
    re-solve, and the post-solve feasibility residual all use identical points, which
    matters when ``pcrb_enforcement == 'scenario'`` draws random samples.
    """
    sets: list[EnforcementSet] = []
    for l in range(target_means.shape[0]):
        mean = target_means[l]
        covariance = target_covariances[l]
        if scheme.perfect_information and perfect_target_states is not None:
            points = np.asarray([perfect_target_states[l]], dtype=float)
            r_wc = None
        else:
            mode = "mean" if not scheme.chance_sensing else solver_cfg.pcrb_enforcement
            certified = mode == "scenario_certified"
            count = solver_cfg.scenario_samples
            if certified:
                from .theory import scenario_dimension, scenario_sample_size
                d = scenario_dimension(k_users, physical.antennas, solver_cfg.scenario_support)
                count = min(scenario_sample_size(d, physical.pcrb_violation, solver_cfg.scenario_beta),
                            int(solver_cfg.scenario_max_samples))
            points = confidence_points(
                mean,
                covariance,
                physical.confidence_radius,
                mode="scenario" if certified else mode,
                scenario_count=count,
                rng=rng,
                extra_boundary_points=solver_cfg.extra_boundary_points,
            )
            if certified:
                r_wc = None
            else:
                r_wc = worst_case_range(q, mean, covariance, physical.confidence_radius) if scheme.chance_sensing else None
        if extra_points is not None and l < len(extra_points) and len(extra_points[l]):
            points = np.vstack([np.asarray(points, dtype=float), np.asarray(extra_points[l], dtype=float)])
        threshold = physical.pcrb_threshold_m2
        if scheme.chance_sensing and solver_cfg.pcrb_enforcement != "scenario_certified":
            threshold *= 1.0 - physical.enforcement_margin
        sets.append(EnforcementSet(points=np.asarray(points, dtype=float),
                                   range_override_m=r_wc, threshold_m2=threshold))
    return sets


def ensure_linear_data(q: np.ndarray, v: np.ndarray, enforcement: EnforcementSet,
                       prior_cov: np.ndarray, physical: PhysicalConfig) -> EnforcementSet:
    """Attach P_min and steering rows (Proposition 2) to an enforcement set once."""
    if enforcement.pmin_w is None or len(enforcement.pmin_w) != len(enforcement.points):
        from .theory import pmin_for_points
        enforcement.pmin_w, enforcement.steering = pmin_for_points(
            q, v, enforcement.points, prior_cov, physical,
            enforcement.threshold_m2, enforcement.range_override_m,
        )
    return enforcement


def enforcement_pcrb_residual(
    q: np.ndarray,
    v: np.ndarray,
    rx: np.ndarray,
    target_covariances: np.ndarray,
    enforcement_sets: list[EnforcementSet],
    physical: PhysicalConfig,
) -> float:
    """Max violation (m^2) of the PCRB constraints at the enforced points.

    Solver status alone is not trusted: with loose first-order tolerances SCS can
    report ``optimal`` while the returned covariances violate the enforced LMIs by
    several m^2. This residual is recomputed from the numerical solution and gated
    downstream.
    """
    residual = -np.inf
    for l, enforcement in enumerate(enforcement_sets):
        for xi in enforcement.points:
            value = pcrb(q, v, xi, target_covariances[l], rx, physical,
                         range_override_m=enforcement.range_override_m)
            residual = max(residual, value - enforcement.threshold_m2)
    return float(residual)


def _rank_residual(matrix: np.ndarray) -> float:
    matrix = (matrix + matrix.conj().T) / 2.0
    trace = float(np.trace(matrix).real)
    if trace <= 1e-12:
        return 0.0
    largest = float(np.max(np.linalg.eigvalsh(matrix)).real)
    return max(0.0, (trace - largest) / trace)


def _min_eigenvalue(matrix: np.ndarray) -> float:
    matrix = (matrix + matrix.conj().T) / 2.0
    return float(np.min(np.linalg.eigvalsh(matrix)).real)


def _result_violation_score(result: CovarianceSlotResult, solver_cfg: SolverConfig) -> tuple[int, float, int, float]:
    """Lexicographic score used when comparing primary and fallback solves.

    Lower is better. Feasible PSD/PCRB solutions are preferred first, then exact
    solver status, then normalized residual severity, and finally transmit power.
    """
    status_ok = result.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    pcrb_bad = (
        np.isfinite(result.pcrb_residual_m2)
        and result.pcrb_residual_m2 > solver_cfg.pcrb_residual_tol_m2
    )
    psd_bad = (
        not np.isfinite(result.min_w_eigenvalue)
        or not np.isfinite(result.min_s_eigenvalue)
        or min(result.min_w_eigenvalue, result.min_s_eigenvalue) < -solver_cfg.psd_tolerance
    )
    feasible = status_ok and not pcrb_bad and not psd_bad
    severity = 0.0
    if np.isfinite(result.pcrb_residual_m2):
        severity += max(0.0, result.pcrb_residual_m2 - solver_cfg.pcrb_residual_tol_m2) / max(
            solver_cfg.pcrb_residual_tol_m2, 1e-12
        )
    min_eig = min(result.min_w_eigenvalue, result.min_s_eigenvalue)
    if np.isfinite(min_eig):
        severity += max(0.0, -solver_cfg.psd_tolerance - min_eig) / max(solver_cfg.psd_tolerance, 1e-12)
    else:
        severity += 1e9
    inaccurate = int(result.status == cp.OPTIMAL_INACCURATE)
    power = float(result.objective_power_w) if np.isfinite(result.objective_power_w) else float("inf")
    return (0 if feasible else 1, severity, inaccurate, power)


def _principal_vectors(matrices: np.ndarray) -> list[np.ndarray]:
    result = []
    for matrix in matrices:
        values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2.0)
        result.append(vectors[:, int(np.argmax(values))])
    return result


def _normalize_direction(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Beam direction has zero norm")
    return vector / norm


def _phase_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = _normalize_direction(reference)
    candidate = _normalize_direction(candidate)
    inner = np.vdot(reference, candidate)
    if abs(inner) > 1e-12:
        candidate = candidate * np.exp(-1j * np.angle(inner))
    return candidate


def _mrt_vectors(q: np.ndarray, users: np.ndarray, physical: PhysicalConfig) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for user in users:
        h = nominal_channel(
            q, user, mx=physical.mx, my=physical.my,
            wavelength_m=physical.wavelength_m, beta0=physical.beta0,
            pathloss_exponent=physical.pathloss_exponent,
        )
        vectors.append(_normalize_direction(h))
    return vectors


def _sample_psd_direction(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    herm = (matrix + matrix.conj().T) / 2.0
    values, vectors = np.linalg.eigh(herm)
    values = np.maximum(values.real, 0.0)
    sqrt_matrix = (vectors * np.sqrt(values)) @ vectors.conj().T
    z = rng.normal(size=matrix.shape[0]) + 1j * rng.normal(size=matrix.shape[0])
    sample = sqrt_matrix @ z
    if np.linalg.norm(sample) <= 1e-12:
        sample = vectors[:, int(np.argmax(values))]
    return _normalize_direction(sample)


def _rank_recovery_direction_bank(
    matrices: np.ndarray, q: np.ndarray, users: np.ndarray, physical: PhysicalConfig,
    solver_cfg: SolverConfig, seed: int,
) -> list[tuple[str, list[np.ndarray]]]:
    """Return deterministic rank-one beam-direction candidates.

    The bank never relaxes SINR/PCRB/PSD/rank gates.  It only changes the fixed
    communication beam directions before the convex power/sensing-covariance
    re-solve.  This is used to resolve isolated SDR rank failures at CP6 boundary
    points without changing the scientific operating point.
    """
    principal = [_normalize_direction(v) for v in _principal_vectors(matrices)]
    mrt = _mrt_vectors(q, users, physical)
    bank: list[tuple[str, list[np.ndarray]]] = [("principal", principal), ("mrt", mrt)]
    for alpha in (0.25, 0.5, 0.75):
        a = float(alpha)
        blended = []
        for vp, vm in zip(principal, mrt):
            vm_aligned = _phase_align(vp, vm)
            blended.append(_normalize_direction((1.0 - a) * vp + a * vm_aligned))
        bank.append((f"blend_{a:.3f}", blended))

    rng = np.random.default_rng(int(seed))
    recovery_trials = 8
    random_trials = max(0, recovery_trials - len(bank))
    for trial in range(random_trials):
        directions = [_sample_psd_direction(matrix, rng) for matrix in matrices]
        bank.append((f"sdr_random_{trial:02d}", directions))
    return bank[:recovery_trials]


def _gaussian_randomization_bank(matrices: np.ndarray, trials: int,
                                 seed: int) -> list[tuple[str, list[np.ndarray]]]:
    """Classical SDR Gaussian randomization: w_k ~ CN(0, W_k), powers re-optimized."""
    rng = np.random.default_rng(int(seed))
    return [(f"gauss_{t:02d}", [_sample_psd_direction(mat, rng) for mat in matrices])
            for t in range(max(1, int(trials)))]


def solve_covariance_slot(
    q: np.ndarray,
    v: np.ndarray,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    rng: np.random.Generator,
    rank_vectors: list[np.ndarray] | None = None,
    rank_target: float | None = None,
    fixed_directions: list[np.ndarray] | None = None,
    allow_slack: bool = False,
    perfect_channel_errors: np.ndarray | None = None,
    perfect_target_states: np.ndarray | None = None,
    enforcement_sets: list[EnforcementSet] | None = None,
    initial_w: np.ndarray | None = None,
    initial_s: np.ndarray | None = None,
) -> CovarianceSlotResult:
    """Solve the fixed-geometry SDR for one slot.

    The robust communication LMIs are exact for the frozen ellipsoids. The sensing
    LMIs enforce the finite confidence-region set or scenario samples specified by
    the configuration. When ``enforcement_sets`` is provided, those frozen points are
    used verbatim; otherwise a set is built from ``rng``. Scenario-certified sets
    larger than ``SCENARIO_CG_INITIAL`` are solved by constraint generation.
    """
    if enforcement_sets is None:
        enforcement_sets = build_enforcement_sets(
            q, target_means, target_covariances, physical, solver_cfg, scheme, rng,
            perfect_target_states=perfect_target_states,
        )

    def direct(sets: list[EnforcementSet]) -> CovarianceSlotResult:
        return _solve_covariance_slot_direct(
            q, v, users, target_means, target_covariances, physical, solver_cfg, scheme,
            rank_vectors, rank_target, fixed_directions, allow_slack,
            perfect_channel_errors, sets, initial_w, initial_s,
        )

    if (solver_cfg.pcrb_enforcement == "scenario_certified"
            and sum(len(es.points) for es in enforcement_sets) > SCENARIO_CG_INITIAL * len(enforcement_sets)):
        return _solve_by_constraint_generation(q, v, target_covariances, physical, allow_slack,
                                               enforcement_sets, direct)
    return direct(enforcement_sets)


def _solve_by_constraint_generation(q, v, target_covariances, physical: PhysicalConfig,
                                    allow_slack: bool, enforcement_sets: list[EnforcementSet],
                                    direct) -> CovarianceSlotResult:
    """Exact cutting-plane solve of the linear-form scenario program (see SCENARIO_CG_*).

    A working-set subproblem is a relaxation of the full program, so an infeasible
    subproblem certifies infeasibility, and a subproblem optimum that satisfies every
    sample is the full optimum. The PCRB residual is recomputed on all samples.
    """
    full = [ensure_linear_data(q, v, es, target_covariances[l], physical)
            for l, es in enumerate(enforcement_sets)]
    if any(np.any(~np.isfinite(es.pmin_w)) for es in full):
        return direct(full)  # reports linear:unreachable_threshold without solving
    for es in full:
        if es.working is None:
            order = np.argsort(-es.pmin_w)[:SCENARIO_CG_INITIAL]
            es.working = np.sort(order[es.pmin_w[order] > 0.0])
    runtime, iterations, rounds = 0.0, 0, 0
    while True:
        subsets = [EnforcementSet(points=es.points[es.working], range_override_m=es.range_override_m,
                                  threshold_m2=es.threshold_m2, pmin_w=es.pmin_w[es.working],
                                  steering=es.steering[es.working]) for es in full]
        result = direct(subsets)
        rounds += 1
        runtime += float(result.runtime_s) if np.isfinite(result.runtime_s) else 0.0
        iterations += int(result.solver_iterations)
        if result.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break
        rx = result.w.sum(axis=0) + result.s
        added = False
        for es in full:
            gap = es.pmin_w - np.einsum("pi,ij,pj->p", es.steering.conj(), rx, es.steering).real
            outside = np.ones(len(gap), dtype=bool)
            outside[es.working] = False
            # Samples outside the working set must be met as accurately as the solver
            # met the working set; with slack, any outside violation is added.
            tol = SCENARIO_CG_ABS_TOL_W
            if not allow_slack and len(es.working):
                tol += max(0.0, float(gap[es.working].max()))
            candidates = np.flatnonzero(outside & (gap > tol))
            if len(candidates):
                worst = candidates[np.argsort(-gap[candidates])[:SCENARIO_CG_BATCH]]
                es.working = np.union1d(es.working, worst)
                added = True
        if not added or rounds >= SCENARIO_CG_MAX_ROUNDS:
            break
    residual = result.pcrb_residual_m2
    if result.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        residual = enforcement_pcrb_residual(q, v, rx, target_covariances, full, physical)
    sizes = "+".join(str(len(es.working)) for es in full)
    return replace(result, runtime_s=runtime, solver_iterations=iterations,
                   pcrb_residual_m2=float(residual),
                   solver_used=f"{result.solver_used}:cg{rounds}r{sizes}")


def _solve_covariance_slot_direct(
    q: np.ndarray,
    v: np.ndarray,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    rank_vectors: list[np.ndarray] | None,
    rank_target: float | None,
    fixed_directions: list[np.ndarray] | None,
    allow_slack: bool,
    perfect_channel_errors: np.ndarray | None,
    enforcement_sets: list[EnforcementSet],
    initial_w: np.ndarray | None,
    initial_s: np.ndarray | None,
) -> CovarianceSlotResult:
    m = physical.antennas
    k_count = users.shape[0]
    l_count = target_means.shape[0]
    fixed_power_vars: list[cp.Variable] | None = None
    if fixed_directions is None:
        w_matrix_vars = [cp.Variable((m, m), hermitian=True, name=f"W_{k}") for k in range(k_count)]
        w_exprs = list(w_matrix_vars)
        if initial_w is not None and np.shape(initial_w) == (k_count, m, m):
            for k, variable in enumerate(w_matrix_vars):
                value = np.asarray(initial_w[k], dtype=np.complex128)
                variable.value = (value + value.conj().T) / 2.0
    else:
        if len(fixed_directions) != k_count:
            raise ValueError("fixed_directions must contain one vector per communication user")
        fixed_power_vars = [cp.Variable(nonneg=True, name=f"p_{k}") for k in range(k_count)]
        w_matrix_vars = None
        w_exprs = []
        for k, vector in enumerate(fixed_directions):
            vector = np.asarray(vector, dtype=np.complex128).reshape(m)
            norm = np.linalg.norm(vector)
            if norm <= 1e-12:
                raise ValueError("A fixed beam direction has zero norm")
            vector = vector / norm
            direction_matrix = np.outer(vector, vector.conj())
            w_exprs.append(fixed_power_vars[k] * direction_matrix)
    if scheme.sensing_mode == "zero":
        s_var = None
        s_expr = np.zeros((m, m), dtype=np.complex128)
    else:
        s_var = cp.Variable((m, m), hermitian=True, name="S")
        s_expr = s_var
        if initial_s is not None and np.shape(initial_s) == (m, m):
            value = np.asarray(initial_s, dtype=np.complex128)
            s_var.value = (value + value.conj().T) / 2.0
    rx = sum(w_exprs, cp.Constant(np.zeros((m, m), dtype=np.complex128))) + s_expr
    constraints: list[cp.Constraint] = [] if w_matrix_vars is None else [w >> 0 for w in w_matrix_vars]
    if s_var is not None:
        constraints.append(s_var >> 0)
    constraints.append(cp.real(cp.trace(rx)) <= physical.p_max_w)
    if scheme.sensing_mode == "fixed_fraction":
        if s_var is None:
            raise ValueError("fixed_fraction requires a sensing covariance variable")
        constraints.append(cp.real(cp.trace(s_var)) == scheme.sensing_fraction * physical.p_max_w)

    comm_slacks = []
    for k, user in enumerate(users):
        h_hat = nominal_channel(q, user, mx=physical.mx, my=physical.my,
                                wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                                pathloss_exponent=physical.pathloss_exponent)
        if scheme.perfect_information and perfect_channel_errors is not None:
            c_actual = csi_shape(h_hat, physical.csi_error_radius, physical.csi_model)
            h_hat = h_hat + hermitian_sqrt(c_actual) @ perfect_channel_errors[k]
        gamma = physical.gamma_min
        a_expr = w_exprs[k]
        for j in range(k_count):
            if j != k:
                a_expr = a_expr - gamma * w_exprs[j]
        a_expr = a_expr - gamma * s_expr
        if scheme.robust_communication and not scheme.perfect_information and physical.csi_error_radius > 0:
            c = csi_shape(h_hat, physical.csi_error_radius, physical.csi_model)
            c_sqrt = hermitian_sqrt(c)
            noise = physical.user_noise_w
            lam = cp.Variable(nonneg=True, name=f"lambda_{k}")
            top = c_sqrt.conj().T @ a_expr @ c_sqrt / noise + lam * np.eye(m)
            cross = c_sqrt.conj().T @ a_expr @ h_hat / noise
            bottom = cp.real(cp.quad_form(h_hat, a_expr)) / noise - gamma - lam
            if allow_slack:
                slack = cp.Variable(nonneg=True, name=f"comm_slack_{k}")
                comm_slacks.append(slack)
                bottom = bottom + slack
            lmi = cp.bmat([
                [top, cp.reshape(cross, (m, 1), order="F")],
                [cp.reshape(cp.conj(cross), (1, m), order="F"), cp.reshape(bottom, (1, 1), order="F")],
            ])
            constraints.append(lmi >> 0)
        else:
            lhs = cp.real(cp.quad_form(h_hat, a_expr)) / physical.user_noise_w
            if allow_slack:
                slack = cp.Variable(nonneg=True, name=f"comm_slack_{k}")
                comm_slacks.append(slack)
                constraints.append(lhs + slack >= gamma)
            else:
                constraints.append(lhs >= gamma)

    d_inv = np.linalg.inv(measurement_calibration(physical))
    ep = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    sensing_slacks = []
    linear_form = (solver_cfg.pcrb_formulation == "linear"
                   or solver_cfg.pcrb_enforcement == "scenario_certified")
    if solver_cfg.pcrb_enforcement == "scenario_certified" and solver_cfg.pcrb_formulation != "linear":
        raise ValueError("scenario_certified enforcement requires pcrb_formulation='linear'")
    if linear_form:
        from .theory import illumination_rows
        rx_vec = cp.vec(rx, order="F")
        for l in range(l_count):
            enforcement = ensure_linear_data(q, v, enforcement_sets[l], target_covariances[l], physical)
            if np.any(~np.isfinite(enforcement.pmin_w)):
                return CovarianceSlotResult(
                    status="infeasible", objective_power_w=np.inf,
                    w=np.full((k_count, m, m), np.nan + 0j), s=np.full((m, m), np.nan + 0j),
                    runtime_s=0.0, rank_residual=np.inf, max_comm_slack=np.inf,
                    max_sensing_slack=np.inf, solver_iterations=0,
                    solver_used="linear:unreachable_threshold",
                )
            active = enforcement.pmin_w > 0.0
            if not np.any(active):
                continue
            phi = illumination_rows(enforcement.steering[active])
            rhs = enforcement.pmin_w[active]
            lhs = cp.real(phi @ rx_vec)
            if allow_slack:
                slack = cp.Variable(int(active.sum()), nonneg=True, name=f"sense_slack_lin_{l}")
                sensing_slacks.append(slack)
                constraints.append(lhs + slack >= rhs)
            else:
                constraints.append(lhs >= rhs)
    for l in range(l_count if not linear_form else 0):
        covariance = target_covariances[l]
        enforcement = enforcement_sets[l]
        points = enforcement.points
        r_wc = enforcement.range_override_m
        threshold = enforcement.threshold_m2
        j_prior = np.linalg.inv(covariance)
        for point_id, xi in enumerate(points):
            p = np.array([xi[0], xi[1], 0.0])
            steering = upa_steering(q, p, physical.mx, physical.my, physical.wavelength_m)
            actual_range = float(np.linalg.norm(p - q))
            range_for_snr = r_wc if r_wc is not None else actual_range
            alpha = physical.sensing_reference_snr_linear * (
                physical.sensing_reference_range_m / range_for_snr
            ) ** 4 / physical.sensing_reference_power_w
            gamma_expr = alpha * cp.real(cp.quad_form(steering, rx))
            g = measurement_jacobian(q, v, xi, physical.fc_hz)
            h_info = (g.T @ d_inv @ g + (g.T @ d_inv @ g).T) / 2.0
            j_plus = j_prior + gamma_expr * h_info
            u = cp.Variable((2, 2), symmetric=True, name=f"U_{l}_{point_id}")
            constraints.append(u >> 0)
            bound = threshold
            if allow_slack:
                slack = cp.Variable(nonneg=True, name=f"sense_slack_{l}_{point_id}")
                sensing_slacks.append(slack)
                bound = bound + slack
            constraints.append(cp.trace(u) <= bound)
            lmi = cp.bmat([[u, ep], [ep.T, j_plus]])
            constraints.append(lmi >> 0)

    if rank_vectors is not None and rank_target is not None:
        if w_matrix_vars is None:
            raise ValueError("rank refinement and fixed_directions cannot be enabled together")
        for k, vector in enumerate(rank_vectors):
            constraints.append(cp.real(cp.quad_form(vector, w_matrix_vars[k])) >= rank_target * cp.real(cp.trace(w_matrix_vars[k])))

    objective = cp.real(cp.trace(rx))
    if allow_slack:
        if comm_slacks:
            objective += 1e3 * cp.sum(cp.hstack(comm_slacks))
        if sensing_slacks:
            if linear_form:
                objective += 1e3 * cp.sum(cp.hstack(sensing_slacks)) / max(physical.p_max_w, 1e-12)
            else:
                objective += 1e3 * cp.sum(cp.hstack(sensing_slacks)) / max(physical.pcrb_threshold_m2, 1e-12)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    start = time.perf_counter()
    try:
        kwargs = {"verbose": solver_cfg.verbose}
        if solver_cfg.solver.upper() == "SCS":
            kwargs.update({"eps": solver_cfg.solver_eps, "max_iters": solver_cfg.solver_max_iters, "warm_start": True})
        problem.solve(solver=solver_cfg.solver, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # pragma: no cover - solver-specific failure path
        # BaseException: a CLARABEL internal error (e.g. "SVD error" in the PSD cone)
        # surfaces as pyo3 PanicException, which does not derive from Exception.
        runtime = time.perf_counter() - start
        return CovarianceSlotResult(
            status=f"error:{type(exc).__name__}", objective_power_w=np.inf,
            w=np.full((k_count, m, m), np.nan + 0j), s=np.full((m, m), np.nan + 0j),
            runtime_s=runtime, rank_residual=np.inf, max_comm_slack=np.inf,
            max_sensing_slack=np.inf, solver_iterations=0,
        )
    runtime = time.perf_counter() - start
    status = str(problem.status)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return CovarianceSlotResult(
            status=status, objective_power_w=np.inf,
            w=np.full((k_count, m, m), np.nan + 0j), s=np.full((m, m), np.nan + 0j),
            runtime_s=runtime, rank_residual=np.inf, max_comm_slack=np.inf,
            max_sensing_slack=np.inf,
            solver_iterations=int(getattr(problem.solver_stats, "num_iters", 0) or 0),
        )
    if w_matrix_vars is not None:
        w_values = np.asarray([((w.value + w.value.conj().T) / 2.0) for w in w_matrix_vars])
    else:
        w_values_list = []
        assert fixed_power_vars is not None and fixed_directions is not None
        for power_var, vector in zip(fixed_power_vars, fixed_directions):
            vector = np.asarray(vector, dtype=np.complex128).reshape(m)
            vector = vector / max(np.linalg.norm(vector), 1e-12)
            power = max(0.0, float(power_var.value or 0.0))
            w_values_list.append(power * np.outer(vector, vector.conj()))
        w_values = np.asarray(w_values_list)
    if s_var is None:
        s_value = np.zeros((m, m), dtype=np.complex128)
    else:
        s_value = (s_var.value + s_var.value.conj().T) / 2.0
    comm_value = max([float(x.value or 0.0) for x in comm_slacks], default=0.0)
    sensing_value = max([float(np.max(x.value)) if x.value is not None else 0.0 for x in sensing_slacks], default=0.0)
    rx_value = w_values.sum(axis=0) + s_value
    residual = enforcement_pcrb_residual(q, v, rx_value, target_covariances,
                                         enforcement_sets, physical)
    min_w_eigenvalue = min((_min_eigenvalue(x) for x in w_values), default=0.0)
    min_s_eigenvalue = _min_eigenvalue(s_value)
    psd_ok = min(min_w_eigenvalue, min_s_eigenvalue) >= -solver_cfg.psd_tolerance
    return CovarianceSlotResult(
        status=status,
        objective_power_w=float(np.trace(rx_value).real),
        w=w_values,
        s=s_value,
        runtime_s=runtime,
        rank_residual=max((_rank_residual(x) for x in w_values), default=0.0),
        max_comm_slack=comm_value,
        max_sensing_slack=sensing_value,
        solver_iterations=int(getattr(problem.solver_stats, "num_iters", 0) or 0),
        pcrb_residual_m2=residual,
        solver_used=(solver_cfg.solver + (":linear" if linear_form else "")
                     + (":rank1" if fixed_directions is not None else "")),
        min_w_eigenvalue=min_w_eigenvalue,
        min_s_eigenvalue=min_s_eigenvalue,
        psd_ok=bool(psd_ok),
    )


def evaluate_covariance_state(
    trajectory_q: np.ndarray,
    trajectory_v: np.ndarray,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    w_all: np.ndarray,
    s_all: np.ndarray,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    seed: int,
) -> list[CovarianceSlotResult]:
    """Evaluate a previously feasible covariance state under the current gates.

    This function performs no optimization. It is used for S4 monotone-threshold
    incumbent preservation: a solution feasible at a stricter PCRB threshold remains
    a valid candidate after that threshold is relaxed, so a numerically weak new solve
    must not erase the certified feasible state.
    """
    n = trajectory_q.shape[0]
    results: list[CovarianceSlotResult] = []
    for slot in range(n):
        rng = np.random.default_rng(seed + 1009 * slot)
        enforcement = build_enforcement_sets(
            trajectory_q[slot], target_means[:, slot], target_covariances[:, slot],
            physical, solver_cfg, scheme, rng, k_users=users.shape[0],
        )
        rx = w_all[slot].sum(axis=0) + s_all[slot]
        residual = enforcement_pcrb_residual(
            trajectory_q[slot], trajectory_v[slot], rx, target_covariances[:, slot],
            enforcement, physical,
        )
        min_w = min((_min_eigenvalue(x) for x in w_all[slot]), default=0.0)
        min_s = _min_eigenvalue(s_all[slot])
        rank = max((_rank_residual(x) for x in w_all[slot]), default=0.0)
        power = float(np.trace(rx).real)
        psd_ok = min(min_w, min_s) >= -solver_cfg.psd_tolerance
        status = cp.OPTIMAL if (
            np.isfinite(power)
            and power <= physical.p_max_w + 1e-6
            and psd_ok
            and np.isfinite(residual)
            and residual <= solver_cfg.pcrb_residual_tol_m2 + 1e-9
        ) else "incumbent_infeasible"
        results.append(CovarianceSlotResult(
            status=str(status), objective_power_w=power,
            w=np.asarray(w_all[slot]).copy(), s=np.asarray(s_all[slot]).copy(),
            runtime_s=0.0, rank_residual=rank, max_comm_slack=0.0,
            max_sensing_slack=0.0, solver_iterations=0,
            pcrb_residual_m2=float(residual), solver_used="preserved_incumbent",
            min_w_eigenvalue=float(min_w), min_s_eigenvalue=float(min_s),
            psd_ok=bool(psd_ok), total_solver_runtime_s=0.0, solver_call_count=0,
            fallback_call_count=0, rank_refinement_call_count=0,
            rank_recovery_call_count=0,
        ))
    return results


def solve_covariances(
    trajectory_q: np.ndarray,
    trajectory_v: np.ndarray,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    seed: int,
    allow_slack: bool = False,
    perfect_channel_errors: np.ndarray | None = None,
    perfect_target_states: np.ndarray | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_label: str = "covariance",
    resume: bool = False,
    initial_w: np.ndarray | None = None,
    initial_s: np.ndarray | None = None,
    extra_points: dict[int, list[np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[CovarianceSlotResult]]:
    n = trajectory_q.shape[0]
    k_count = users.shape[0]
    m = physical.antennas
    all_w = np.zeros((n, k_count, m, m), dtype=np.complex128)
    all_s = np.zeros((n, m, m), dtype=np.complex128)
    results: list[CovarianceSlotResult] = []
    residual_tol = solver_cfg.pcrb_residual_tol_m2
    fallback = (solver_cfg.fallback_solver or "").upper()
    checkpoint_npz: Path | None = None
    checkpoint_json: Path | None = None
    start_slot = 0
    geometry_hash = hashlib.sha256(
        np.ascontiguousarray(trajectory_q).view(np.uint8).tobytes()
        + np.ascontiguousarray(trajectory_v).view(np.uint8).tobytes()
        + str((physical.p_max_w, physical.pcrb_threshold_m2,
               physical.direction_cosine_std_ref, scheme.name,
               scheme.robust_communication, scheme.chance_sensing,
               scheme.sensing_mode, solver_cfg.pcrb_formulation,
               solver_cfg.pcrb_enforcement, physical.enforcement_margin,
               solver_cfg.rank_recovery_mode,
               None if extra_points is None else sorted(
                   (k, [len(x) for x in val]) for k, val in extra_points.items()))).encode("utf-8")
    ).hexdigest()
    if checkpoint_dir is not None:
        state_dir = Path(checkpoint_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in checkpoint_label)
        checkpoint_npz = state_dir / f"{safe_label}.npz"
        checkpoint_json = state_dir / f"{safe_label}.json"
        if resume and checkpoint_npz.exists() and checkpoint_json.exists():
            metadata = json.loads(checkpoint_json.read_text(encoding="utf-8"))
            if (
                int(metadata.get("seed", -1)) == int(seed)
                and int(metadata.get("slots", -1)) == int(n)
                and str(metadata.get("geometry_hash", "")) == geometry_hash
            ):
                # Closed before the per-slot checkpoint replaces this file (Windows).
                with np.load(checkpoint_npz) as arrays:
                    all_w[:] = arrays["W"]
                    all_s[:] = arrays["S"]
                for idx, item in enumerate(metadata.get("results", [])):
                    results.append(CovarianceSlotResult(
                        w=all_w[idx], s=all_s[idx], **item
                    ))
                start_slot = len(results)
    for slot in range(start_slot, n):
        rng = np.random.default_rng(seed + 1009 * slot)
        slot_states = None if perfect_target_states is None else perfect_target_states[:, slot]
        slot_errors = None if perfect_channel_errors is None else perfect_channel_errors[slot]
        # Freeze the enforcement points once per slot so the base solve, every
        # rank-refinement re-solve, the fallback re-solve, and the feasibility
        # residual all see identical constraint sets (relevant in scenario mode).
        enforcement = build_enforcement_sets(
            trajectory_q[slot], target_means[:, slot], target_covariances[:, slot],
            physical, solver_cfg, scheme, rng, perfect_target_states=slot_states,
            k_users=k_count,
            extra_points=None if extra_points is None else extra_points.get(slot),
        )

        slot_stats = {
            "total_solver_runtime_s": 0.0,
            "solver_call_count": 0,
            "fallback_call_count": 0,
            "rank_refinement_call_count": 0,
            "rank_recovery_call_count": 0,
        }

        def _solve(cfg: SolverConfig, rank_vectors=None, rank_target=None,
                   fixed_directions=None) -> CovarianceSlotResult:
            item = solve_covariance_slot(
                trajectory_q[slot], trajectory_v[slot], users,
                target_means[:, slot], target_covariances[:, slot],
                physical, cfg, scheme, rng,
                rank_vectors=rank_vectors, rank_target=rank_target,
                fixed_directions=fixed_directions,
                allow_slack=allow_slack,
                perfect_channel_errors=slot_errors,
                perfect_target_states=slot_states,
                enforcement_sets=enforcement,
                initial_w=(initial_w[slot] if initial_w is not None and np.shape(initial_w)[0] == n else None),
                initial_s=(initial_s[slot] if initial_s is not None and np.shape(initial_s)[0] == n else None),
            )
            slot_stats["solver_call_count"] += 1
            if np.isfinite(item.runtime_s):
                slot_stats["total_solver_runtime_s"] += float(item.runtime_s)
            return item

        def _attach_stats(item: CovarianceSlotResult) -> CovarianceSlotResult:
            return replace(item, **slot_stats)

        def _solve_with_escalation(rank_vectors=None, rank_target=None,
                                   fixed_directions=None) -> CovarianceSlotResult:
            """Solve with the primary solver and cross-check numerically weak results.

            Escalation is triggered by a PCRB residual, a PSD-cone residual, or an
            ``optimal_inaccurate`` status. The fallback is accepted only when its
            lexicographic feasibility score is better.
            """
            item = _solve(solver_cfg, rank_vectors, rank_target, fixed_directions)
            min_eig = min(item.min_w_eigenvalue, item.min_s_eigenvalue)
            # A solver exception or a non-optimal status must also escalate: the two
            # conic solvers fail on disjoint slot sets (SCS returns slightly
            # non-PSD covariances where CLARABEL raises, and vice versa), so a
            # primary failure that is not cross-checked aborts the whole run on a
            # slot the fallback can solve. The lexicographic score below still
            # decides which result is kept, and an error result always loses.
            status_ok = item.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            # When the fallback equals the primary solver (CLARABEL in R12), a solver
            # exception is retried with the other conic solver instead of aborting the
            # run. Only the exception path changes, so successful solves are unaffected.
            alt = fallback
            if alt == solver_cfg.solver.upper() and item.status.startswith("error:"):
                alt = "SCS" if alt != "SCS" else "CLARABEL"
            needs_escalation = (
                not allow_slack
                and alt
                and alt != solver_cfg.solver.upper()
                and (
                    not status_ok
                    or (np.isfinite(item.pcrb_residual_m2)
                        and item.pcrb_residual_m2 > residual_tol)
                    or (not np.isfinite(min_eig) or min_eig < -solver_cfg.psd_tolerance)
                    or (solver_cfg.fallback_on_inaccurate
                        and item.status == cp.OPTIMAL_INACCURATE)
                )
            )
            if needs_escalation:
                slot_stats["fallback_call_count"] += 1
                retry = _solve(
                    replace(solver_cfg, solver=alt),
                    rank_vectors, rank_target, fixed_directions
                )
                if _result_violation_score(retry, solver_cfg) < _result_violation_score(item, solver_cfg):
                    return retry
            return item

        result = _solve_with_escalation()
        if result.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            results.append(_attach_stats(result))
            return all_w, all_s, results
        best = result
        if solver_cfg.rank_refinement and result.rank_residual > solver_cfg.rank_tolerance:
            vectors = _principal_vectors(result.w)
            for eta in solver_cfg.rank_targets:
                # Once the configured rank gate is met, tighter rank targets only
                # spend solver time without changing acceptance.
                if best.rank_residual <= solver_cfg.rank_tolerance:
                    break
                slot_stats["rank_refinement_call_count"] += 1
                refined = _solve_with_escalation(rank_vectors=vectors, rank_target=eta)
                refined_feasible = (
                    refined.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                    and refined.psd_ok
                    and np.isfinite(refined.pcrb_residual_m2)
                    and refined.pcrb_residual_m2 <= max(residual_tol, best.pcrb_residual_m2)
                )
                if refined_feasible:
                    best = refined
                    vectors = _principal_vectors(refined.w)
                else:
                    break
        if (solver_cfg.rank_one_recovery and not allow_slack
                and best.rank_residual > solver_cfg.rank_tolerance):
            if solver_cfg.rank_recovery_mode == "gaussian_randomization":
                recovery_bank = _gaussian_randomization_bank(
                    best.w, solver_cfg.gaussian_randomization_trials,
                    seed=seed + 7_919 * slot + 1_234_567,
                )
            else:
                recovery_bank = _rank_recovery_direction_bank(
                    best.w, trajectory_q[slot], users, physical, solver_cfg,
                    seed=seed + 7_919 * slot + 1_234_567,
                )
            recovered_best = None
            for recovery_label, directions in recovery_bank:
                slot_stats["rank_recovery_call_count"] += 1
                recovered = _solve_with_escalation(fixed_directions=directions)
                recovery_feasible = (
                    recovered.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                    and recovered.rank_residual <= solver_cfg.rank_tolerance
                    and recovered.psd_ok
                    and np.isfinite(recovered.pcrb_residual_m2)
                    and recovered.pcrb_residual_m2 <= residual_tol
                )
                if not recovery_feasible:
                    continue
                recovered = replace(
                    recovered,
                    solver_used=f"{recovered.solver_used}:{recovery_label}",
                )
                if (recovered_best is None
                        or recovered.objective_power_w < recovered_best.objective_power_w):
                    recovered_best = recovered
            if recovered_best is not None:
                best = recovered_best
        best = _attach_stats(best)
        all_w[slot] = best.w
        all_s[slot] = best.s
        results.append(best)
        if checkpoint_npz is not None and checkpoint_json is not None:
            tmp_npz = checkpoint_npz.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_npz, W=all_w, S=all_s)
            tmp_npz.replace(checkpoint_npz)
            serial = []
            for item in results:
                data = asdict(item)
                data.pop("w", None)
                data.pop("s", None)
                serial.append(data)
            tmp_json = checkpoint_json.with_suffix(".tmp.json")
            tmp_json.write_text(json.dumps({
                "seed": int(seed),
                "slots": int(n),
                "completed_slots": int(len(results)),
                "geometry_hash": geometry_hash,
                "results": serial,
            }, indent=2), encoding="utf-8")
            tmp_json.replace(checkpoint_json)
    return all_w, all_s, results
