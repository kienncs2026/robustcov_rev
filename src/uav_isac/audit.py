from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

from .channel import random_channel_outage, worst_case_sinr_margin
from .config import MissionConfig, PhysicalConfig, SolverConfig, SchemeConfig, EvaluationConfig
from .dynamics import Trajectory, energy_breakdown, kinematic_residuals
from .sensing import (binomial_upper_bound, measurement_information, pcrb,
                      process_covariance, propagate_targets)
from .geometry import nominal_channel, upa_steering
from .channel import sinr
from . import theory


@dataclass
class AuditResult:
    summary: dict[str, float | str | bool]
    communication_rows: list[dict[str, float | int]]
    pcrb_rows: list[dict[str, float | int]]
    power_rows: list[dict[str, float | int]]
    # Additional per-run tables written as <name>.csv (revision diagnostics).
    extra_tables: dict[str, list[dict]] = field(default_factory=dict)


def rank_residual(matrix: np.ndarray) -> float:
    matrix = (matrix + matrix.conj().T) / 2.0
    trace = float(np.trace(matrix).real)
    if trace <= 1e-12:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    return max(0.0, (trace - float(eigenvalues[-1].real)) / trace)


def min_hermitian_eigenvalue(matrix: np.ndarray) -> float:
    matrix = (matrix + matrix.conj().T) / 2.0
    return float(np.min(np.linalg.eigvalsh(matrix)).real)


def audit_solution(
    trajectory: Trajectory,
    w_all: np.ndarray,
    s_all: np.ndarray,
    users: np.ndarray,
    target_means: np.ndarray,
    target_covariances: np.ndarray,
    mission: MissionConfig,
    physical: PhysicalConfig,
    solver_cfg: SolverConfig,
    scheme: SchemeConfig,
    evaluation: EvaluationConfig,
    seed: int,
    audit_samples: int | None = None,
    channel_samples: int | None = None,
    exact_robust: bool = True,
    enforcement_pcrb_residual_m2: float = float("nan"),
) -> AuditResult:
    rng = np.random.default_rng(seed)
    physical_channel = physical
    if evaluation.actual_pathloss_exponent is not None:
        physical_channel = replace(physical, pathloss_exponent=evaluation.actual_pathloss_exponent)
    evaluation_covariances = target_covariances
    if evaluation.actual_process_acceleration_std_mps2 is not None:
        physical_target = replace(physical, process_acceleration_std_mps2=evaluation.actual_process_acceleration_std_mps2)
        _, evaluation_covariances = propagate_targets(mission, physical_target)
    n_test = int(audit_samples or solver_cfg.out_of_sample_count)
    n_channel = int(channel_samples or solver_cfg.channel_audit_samples)
    tx_power = np.trace(w_all.sum(axis=1) + s_all, axis1=1, axis2=2).real
    energy = energy_breakdown(trajectory, mission, physical, tx_power)
    kin = kinematic_residuals(trajectory, mission, physical)

    communication_rows: list[dict[str, float | int]] = []
    min_sinr_margin = np.inf
    min_quadratic_margin = np.inf
    max_random_outage = 0.0
    for slot in range(mission.slots):
        for k, user in enumerate(users):
            if exact_robust:
                sinr_margin, quadratic_margin = worst_case_sinr_margin(
                    trajectory.q[slot], user, w_all[slot], s_all[slot], k, physical_channel
                )
            else:
                sinr_margin, quadratic_margin = np.nan, np.nan
            outage = random_channel_outage(
                trajectory.q[slot], user, w_all[slot], s_all[slot], k,
                physical_channel, n_channel, rng, threshold_tolerance=1e-6
            ) if n_channel > 0 else np.nan
            min_sinr_margin = min(min_sinr_margin, sinr_margin) if np.isfinite(sinr_margin) else min_sinr_margin
            min_quadratic_margin = min(min_quadratic_margin, quadratic_margin) if np.isfinite(quadratic_margin) else min_quadratic_margin
            max_random_outage = max(max_random_outage, outage) if np.isfinite(outage) else max_random_outage
            communication_rows.append({
                "slot": slot,
                "user": k,
                "worst_case_sinr": float(sinr_margin + physical.gamma_min) if np.isfinite(sinr_margin) else np.nan,
                "sinr_threshold": float(physical.gamma_min),
                "sinr_margin": float(sinr_margin),
                "quadratic_margin": float(quadratic_margin),
                "random_error_outage": float(outage),
            })

    pcrb_rows: list[dict[str, float | int]] = []
    lemma2_rows: list[dict] = []
    ct_states = None
    if evaluation.target_turn_rate_dps is not None:
        ct_states = _coordinated_turn_states(mission, physical, target_means, target_covariances,
                                             evaluation.target_turn_rate_dps, n_test, rng)
    max_empirical = 0.0
    max_ucb = 0.0
    max_mean_pcrb = 0.0
    for slot in range(mission.slots):
        rx = w_all[slot].sum(axis=0) + s_all[slot]
        for l in range(target_means.shape[0]):
            mean = target_means[l, slot]
            covariance = target_covariances[l, slot]
            mean_value = pcrb(trajectory.q[slot], trajectory.v[slot], mean, covariance, rx, physical)
            if ct_states is not None:
                samples = ct_states[l, slot]
            else:
                samples = rng.multivariate_normal(mean, evaluation_covariances[l, slot], size=n_test)
            values = np.empty(n_test, dtype=float)
            info_sum = np.zeros((4, 4))
            j_prior = np.linalg.inv(covariance)
            if evaluation.rcs_log_std_db > 0.0:
                rcs_scales = 10.0 ** (rng.normal(0.0, evaluation.rcs_log_std_db, size=n_test) / 10.0)
            else:
                rcs_scales = np.ones(n_test)
            for idx, xi in enumerate(samples):
                jm = measurement_information(trajectory.q[slot], trajectory.v[slot], xi, rx,
                                             physical, snr_scale=float(rcs_scales[idx]))
                info_sum += jm
                jj = j_prior + jm
                values[idx] = float(np.trace(np.linalg.inv((jj + jj.T) / 2.0)[:2, :2]))
            if evaluation.lemma_diagnostics:
                jb = j_prior + info_sum / n_test
                psi_b = float(np.trace(np.linalg.inv((jb + jb.T) / 2.0)[:2, :2]))
                psi0 = float(np.trace(covariance[:2, :2]))
                lemma2_rows.append({
                    "slot": slot, "target": l, "bcrb_m2": psi_b,
                    "mean_conditional_pcrb_m2": float(np.mean(values)),
                    "prior_position_variance_m2": psi0,
                    "lemma2_rhs_m2": (1.0 - physical.pcrb_violation) * physical.pcrb_threshold_m2
                                     + physical.pcrb_violation * psi0,
                })
            violations = int(np.count_nonzero(values > physical.pcrb_threshold_m2))
            empirical = violations / n_test
            ucb = binomial_upper_bound(violations, n_test)
            max_empirical = max(max_empirical, empirical)
            max_ucb = max(max_ucb, ucb)
            max_mean_pcrb = max(max_mean_pcrb, mean_value)
            pcrb_rows.append({
                "slot": slot,
                "target": l,
                "mean_pcrb_m2": float(mean_value),
                "sample_mean_pcrb_m2": float(np.mean(values)),
                "sample_p95_pcrb_m2": float(np.quantile(values, 0.95)),
                "violations": violations,
                "samples": n_test,
                "empirical_violation": float(empirical),
                "upper_confidence_bound_95": float(ucb),
                "allowed_violation": float(physical.pcrb_violation),
            })

    power_rows = []
    max_rank = 0.0
    min_power_margin = np.inf
    min_w_eigenvalue = np.inf
    min_s_eigenvalue = np.inf
    for slot in range(mission.slots):
        powers = [float(np.trace(w_all[slot, k]).real) for k in range(users.shape[0])]
        sensing_power = float(np.trace(s_all[slot]).real)
        total_power = sum(powers) + sensing_power
        min_power_margin = min(min_power_margin, physical.p_max_w - total_power)
        slot_rank = max(rank_residual(w_all[slot, k]) for k in range(users.shape[0]))
        slot_w_min_eig = min(min_hermitian_eigenvalue(w_all[slot, k]) for k in range(users.shape[0]))
        slot_s_min_eig = min_hermitian_eigenvalue(s_all[slot])
        min_w_eigenvalue = min(min_w_eigenvalue, slot_w_min_eig)
        min_s_eigenvalue = min(min_s_eigenvalue, slot_s_min_eig)
        max_rank = max(max_rank, slot_rank)
        row: dict[str, float | int] = {
            "slot": slot,
            "total_power_w": total_power,
            "sensing_power_w": sensing_power,
            "power_margin_w": physical.p_max_w - total_power,
            "rank_residual": slot_rank,
            "min_w_eigenvalue": slot_w_min_eig,
            "min_s_eigenvalue": slot_s_min_eig,
        }
        for k, value in enumerate(powers):
            row[f"user_{k}_power_w"] = value
        comm_cov = w_all[slot].sum(axis=0)
        rx_cov = comm_cov + s_all[slot]
        for l in range(target_means.shape[0]):
            xi = target_means[l, slot]
            point = np.array([xi[0], xi[1], 0.0])
            steering = upa_steering(trajectory.q[slot], point, physical.mx, physical.my, physical.wavelength_m)
            comm_illum = float(np.vdot(steering, comm_cov @ steering).real)
            total_illum = float(np.vdot(steering, rx_cov @ steering).real)
            row[f"target_{l}_comm_illumination_fraction"] = comm_illum / max(total_illum, 1e-15)
        power_rows.append(row)

    communication_ok = (
        (not scheme.robust_communication)
        or (np.isfinite(min_sinr_margin) and min_sinr_margin >= -1e-4)
    )
    sensing_ok = (
        (not scheme.chance_sensing)
        or (np.isfinite(max_ucb) and max_ucb <= physical.pcrb_violation + 1e-12)
    )
    # Solver status is not trusted for the enforced PCRB LMIs: the residual recomputed
    # from the numerical covariances must itself be within tolerance.
    enforcement_ok = (
        np.isfinite(enforcement_pcrb_residual_m2)
        and enforcement_pcrb_residual_m2 <= solver_cfg.pcrb_residual_tol_m2 + 1e-9
    )
    tol_kin = solver_cfg.trajectory_feasibility_tolerance
    kinematic_equalities_ok = (
        max(kin["dynamics"], kin["terminal_position"], kin["terminal_velocity"]) <= tol_kin
    )
    kinematic_inequalities_ok = (
        min(
            kin["altitude_low"], kin["altitude_high"], kin["horizontal_speed"],
            kin["vertical_speed"], kin["acceleration"],
        ) >= -tol_kin
    )
    power_ok = np.isfinite(min_power_margin) and min_power_margin >= -1e-6
    rank_ok = np.isfinite(max_rank) and max_rank <= solver_cfg.rank_tolerance
    min_covariance_eigenvalue = min(min_w_eigenvalue, min_s_eigenvalue)
    psd_ok = (
        np.isfinite(min_covariance_eigenvalue)
        and min_covariance_eigenvalue >= -solver_cfg.psd_tolerance
    )
    gate_flags = {
        "communication_ok": bool(communication_ok),
        "sensing_ok": bool(sensing_ok),
        "enforcement_ok": bool(enforcement_ok),
        "power_ok": bool(power_ok),
        "kinematic_equalities_ok": bool(kinematic_equalities_ok),
        "kinematic_inequalities_ok": bool(kinematic_inequalities_ok),
        "rank_ok": bool(rank_ok),
        "psd_ok": bool(psd_ok),
    }
    accepted = all(gate_flags.values())
    rejection_reasons = ";".join(key for key, value in gate_flags.items() if not value)
    summary: dict[str, float | str | bool] = {
        **energy,
        "min_worst_case_sinr_margin": float(min_sinr_margin),
        "min_robust_quadratic_margin": float(min_quadratic_margin),
        "max_random_channel_outage": float(max_random_outage),
        "min_power_margin_w": float(min_power_margin),
        "max_rank_residual": float(max_rank),
        "min_w_eigenvalue": float(min_w_eigenvalue),
        "min_s_eigenvalue": float(min_s_eigenvalue),
        "min_covariance_eigenvalue": float(min_covariance_eigenvalue),
        "psd_tolerance": float(solver_cfg.psd_tolerance),
        "max_mean_pcrb_m2": float(max_mean_pcrb),
        "max_empirical_pcrb_violation": float(max_empirical),
        "max_pcrb_ucb_95": float(max_ucb),
        "allowed_pcrb_violation": float(physical.pcrb_violation),
        "max_enforcement_pcrb_residual_m2": float(enforcement_pcrb_residual_m2),
        "dynamics_residual": float(kin["dynamics"]),
        "terminal_position_residual": float(kin["terminal_position"]),
        "terminal_velocity_residual": float(kin["terminal_velocity"]),
        "altitude_low_margin": float(kin["altitude_low"]),
        "altitude_high_margin": float(kin["altitude_high"]),
        "horizontal_speed_margin": float(kin["horizontal_speed"]),
        "vertical_speed_margin": float(kin["vertical_speed"]),
        "acceleration_margin": float(kin["acceleration"]),
        **gate_flags,
        "accepted": bool(accepted),
        "rejection_reasons": rejection_reasons,
    }
    extra: dict[str, list[dict]] = {}
    if evaluation.lemma_diagnostics:
        extra["lemma2_bcrb_check"] = lemma2_rows
        rx_all = w_all.sum(axis=1) + s_all
        loop_rows = theory.closed_vs_open_loop(trajectory.q, trajectory.v, rx_all, target_means,
                                               target_covariances, mission, physical)
        extra["lemma3_closed_vs_open"] = loop_rows
        summary["lemma2_max_bcrb_m2"] = max((r["bcrb_m2"] for r in lemma2_rows), default=np.nan)
        summary["lemma2_lower_inequality_ok"] = bool(all(
            r["bcrb_m2"] <= r["mean_conditional_pcrb_m2"] * (1 + 1e-9) + 1e-12 for r in lemma2_rows))
        summary["lemma3_max_closed_over_open"] = max((r["closed_over_open"] for r in loop_rows), default=np.nan)
        summary["lemma3_min_closed_over_open"] = min((r["closed_over_open"] for r in loop_rows), default=np.nan)
    if evaluation.rician_k_db is not None:
        mismatch = _rician_outage(trajectory, w_all, s_all, users, mission, physical,
                                  evaluation.rician_k_db, evaluation.mismatch_channel_samples, rng)
        extra["mismatch_rician_outage"] = mismatch
        summary["rician_k_db"] = float(evaluation.rician_k_db)
        summary["rician_max_outage"] = max(r["outage"] for r in mismatch)
        summary["rician_mean_outage"] = float(np.mean([r["outage"] for r in mismatch]))
    if evaluation.target_turn_rate_dps is not None:
        summary["target_turn_rate_dps"] = float(evaluation.target_turn_rate_dps)
    return AuditResult(summary, communication_rows, pcrb_rows, power_rows, extra)


def _coordinated_turn_states(mission: MissionConfig, physical: PhysicalConfig,
                             target_means: np.ndarray, target_covariances: np.ndarray,
                             turn_rate_dps: float, count: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Truth trajectories from a coordinated-turn model while the design used CV.

    Initial states are drawn from the design prior; each slot applies the CT
    transition with turn rate omega plus the design process noise.
    """
    dt = mission.dt_s
    omega = np.deg2rad(turn_rate_dps)
    if abs(omega) < 1e-12:
        f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    else:
        sw, cw = np.sin(omega * dt), np.cos(omega * dt)
        f = np.array([[1, 0, sw / omega, -(1 - cw) / omega],
                      [0, 1, (1 - cw) / omega, sw / omega],
                      [0, 0, cw, -sw], [0, 0, sw, cw]], dtype=float)
    q_proc = process_covariance(dt, physical.process_acceleration_std_mps2)
    l_count, n_slots = target_means.shape[0], mission.slots
    out = np.empty((l_count, n_slots, count, 4))
    for l in range(l_count):
        x = rng.multivariate_normal(target_means[l, 0], target_covariances[l, 0], size=count)
        out[l, 0] = x
        for n in range(1, n_slots):
            x = x @ f.T + rng.multivariate_normal(np.zeros(4), q_proc, size=count)
            out[l, n] = x
    return out


def _rician_outage(trajectory: Trajectory, w_all: np.ndarray, s_all: np.ndarray,
                   users: np.ndarray, mission: MissionConfig, physical: PhysicalConfig,
                   k_db: float, samples: int, rng: np.random.Generator) -> list[dict]:
    """SINR outage when the true channel is Rician with factor K around the LoS model.

    h = sqrt(K/(K+1)) h_LoS + sqrt(1/(K+1)) ||h_LoS|| / sqrt(M) z, z ~ CN(0, I),
    which keeps the mean received power of the LoS model.
    """
    k_lin = 10.0 ** (k_db / 10.0)
    m = physical.antennas
    rows = []
    for slot in range(mission.slots):
        for k, user in enumerate(users):
            h_los = nominal_channel(trajectory.q[slot], user, mx=physical.mx, my=physical.my,
                                    wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                                    pathloss_exponent=physical.pathloss_exponent)
            scale = np.linalg.norm(h_los) / np.sqrt(m)
            z = (rng.normal(size=(samples, m)) + 1j * rng.normal(size=(samples, m))) / np.sqrt(2.0)
            hs = np.sqrt(k_lin / (k_lin + 1.0)) * h_los[None, :] + np.sqrt(1.0 / (k_lin + 1.0)) * scale * z
            out = sum(sinr(h, w_all[slot], s_all[slot], k, physical.user_noise_w) < physical.gamma_min - 1e-6
                      for h in hs)
            rows.append({"slot": slot, "user": k, "rician_k_db": k_db, "outage": out / samples})
    return rows
