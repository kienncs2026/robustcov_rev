from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import MissionConfig, PhysicalConfig


@dataclass
class Trajectory:
    q: np.ndarray  # (N,3), beginning-of-slot positions
    v: np.ndarray  # (N,3), beginning-of-slot velocities
    a: np.ndarray  # (N,3), slot accelerations

    def terminal_state(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        qf = self.q[-1] + dt * self.v[-1] + 0.5 * dt**2 * self.a[-1]
        vf = self.v[-1] + dt * self.a[-1]
        return qf, vf


def minimum_jerk_seed(mission: MissionConfig, physical: PhysicalConfig,
                      attraction_xy: np.ndarray | None = None,
                      attraction_strength: float = 0.0,
                      altitude_m: float | None = None) -> Trajectory:
    n = mission.slots
    dt = mission.dt_s
    t = np.arange(n, dtype=float) * dt
    s = t / mission.duration_s
    q0 = np.asarray(mission.q_initial_m, dtype=float)
    qf = np.asarray(mission.q_final_m, dtype=float)
    delta = qf - q0
    phi = 10 * s**3 - 15 * s**4 + 6 * s**5
    dphi = (30 * s**2 - 60 * s**3 + 30 * s**4) / mission.duration_s
    ddphi = (60 * s - 180 * s**2 + 120 * s**3) / mission.duration_s**2
    q = q0[None, :] + phi[:, None] * delta[None, :]
    v = dphi[:, None] * delta[None, :]
    a = ddphi[:, None] * delta[None, :]
    if attraction_xy is not None and attraction_strength != 0.0:
        attraction_xy = np.asarray(attraction_xy, dtype=float)
        midpoint = 0.5 * (q0[:2] + qf[:2])
        offset = attraction_xy - midpoint
        bump = 16.0 * s**2 * (1.0 - s)**2
        dbump = 32.0 * s * (1.0 - s) * (1.0 - 2.0 * s) / mission.duration_s
        ddbump = 32.0 * (1.0 - 6.0 * s + 6.0 * s**2) / mission.duration_s**2
        q[:, :2] += attraction_strength * bump[:, None] * offset[None, :]
        v[:, :2] += attraction_strength * dbump[:, None] * offset[None, :]
        a[:, :2] += attraction_strength * ddbump[:, None] * offset[None, :]
    if altitude_m is not None:
        q[:, 2] = altitude_m
        v[:, 2] = 0.0
        a[:, 2] = 0.0
    # Enforce the exact terminal state by replacing the final two accelerations.
    a = enforce_terminal_with_last_two(a, mission)
    q, v = reconstruct_from_accelerations(a, mission)
    return Trajectory(q=q, v=v, a=a)



def trapezoidal_seed(mission: MissionConfig, physical: PhysicalConfig, altitude_m: float | None = None) -> Trajectory:
    """Construct an exact discrete-time accelerate--coast--decelerate seed."""
    n = mission.slots
    dt = mission.dt_s
    q0 = np.asarray(mission.q_initial_m, dtype=float)
    qf = np.asarray(mission.q_final_m, dtype=float)
    delta = qf - q0
    distance = float(np.linalg.norm(delta))
    if distance < 1e-12:
        a_slots = np.zeros((n, 3), dtype=float)
        q, v = reconstruct_from_accelerations(a_slots, mission)
        return Trajectory(q=q, v=v, a=a_slots)
    direction = delta / distance
    selected = None
    # For m acceleration and m deceleration slots, D=a*dt^2*m*(N-m).
    for m in range(n // 2, 0, -1):
        acceleration = distance / (dt**2 * m * (n - m))
        peak_speed = acceleration * dt * m
        horizontal_peak = peak_speed * np.linalg.norm(direction[:2])
        vertical_peak = peak_speed * abs(direction[2])
        if (acceleration <= physical.a_max_mps2 + 1e-12
                and horizontal_peak <= physical.vh_max_mps + 1e-12
                and vertical_peak <= physical.vz_max_mps + 1e-12):
            selected = (m, acceleration)
            break
    if selected is None:
        return minimum_jerk_seed(mission, physical, altitude_m=altitude_m)
    m, acceleration = selected
    a_slots = np.zeros((n, 3), dtype=float)
    a_slots[:m] = acceleration * direction
    a_slots[n-m:] = -acceleration * direction
    q, v = reconstruct_from_accelerations(a_slots, mission)
    if altitude_m is not None and abs(q0[2] - qf[2]) < 1e-9:
        q[:, 2] = altitude_m
        v[:, 2] = 0.0
        a_slots[:, 2] = 0.0
    return Trajectory(q=q, v=v, a=a_slots)

def reconstruct_from_accelerations(a: np.ndarray, mission: MissionConfig) -> tuple[np.ndarray, np.ndarray]:
    n = mission.slots
    dt = mission.dt_s
    a = np.asarray(a, dtype=float)
    if a.shape != (n, 3):
        raise ValueError(f"Expected accelerations with shape {(n,3)}, got {a.shape}")
    q = np.zeros((n, 3), dtype=float)
    v = np.zeros((n, 3), dtype=float)
    q[0] = np.asarray(mission.q_initial_m, dtype=float)
    v[0] = np.asarray(mission.v_initial_mps, dtype=float)
    for i in range(n - 1):
        q[i + 1] = q[i] + dt * v[i] + 0.5 * dt**2 * a[i]
        v[i + 1] = v[i] + dt * a[i]
    return q, v


def enforce_terminal_with_last_two(a: np.ndarray, mission: MissionConfig) -> np.ndarray:
    """Solve the final two slot accelerations to satisfy terminal q and v exactly."""
    a = np.asarray(a, dtype=float).copy()
    n = mission.slots
    dt = mission.dt_s
    if n < 2:
        raise ValueError("At least two slots are required.")
    q0 = np.asarray(mission.q_initial_m, dtype=float)
    v0 = np.asarray(mission.v_initial_mps, dtype=float)
    qf = np.asarray(mission.q_final_m, dtype=float)
    vf = np.asarray(mission.v_final_mps, dtype=float)
    known = a[: n - 2]
    sum_known = known.sum(axis=0) if known.size else np.zeros(3)
    weights = (n - np.arange(n - 2) - 0.5)[:, None] if n > 2 else np.zeros((0, 1))
    weighted_known = (weights * known).sum(axis=0) if known.size else np.zeros(3)
    rhs_v = (vf - v0) / dt - sum_known
    rhs_q = (qf - q0 - n * dt * v0) / dt**2 - weighted_known
    # Last two weights are 1.5 and 0.5.
    # a_{N-2}+a_{N-1}=rhs_v; 1.5a_{N-2}+0.5a_{N-1}=rhs_q.
    a_penultimate = rhs_q - 0.5 * rhs_v
    a_last = rhs_v - a_penultimate
    a[n - 2] = a_penultimate
    a[n - 1] = a_last
    return a


def trajectory_from_segment_controls(controls: np.ndarray, mission: MissionConfig,
                                     segments: int) -> Trajectory:
    n = mission.slots
    controls = np.asarray(controls, dtype=float).reshape(segments, 3)
    first_count = n - 2
    indices = np.minimum((np.arange(first_count) * segments) // max(first_count, 1), segments - 1)
    a = np.zeros((n, 3), dtype=float)
    a[:first_count] = controls[indices]
    a = enforce_terminal_with_last_two(a, mission)
    q, v = reconstruct_from_accelerations(a, mission)
    return Trajectory(q=q, v=v, a=a)


def controls_from_trajectory(traj: Trajectory, segments: int) -> np.ndarray:
    n = traj.a.shape[0]
    first_count = n - 2
    controls = np.zeros((segments, 3), dtype=float)
    for s in range(segments):
        lo = (s * first_count) // segments
        hi = ((s + 1) * first_count) // segments
        if hi <= lo:
            controls[s] = traj.a[min(lo, first_count - 1)]
        else:
            controls[s] = traj.a[lo:hi].mean(axis=0)
    return controls


def horizontal_power(speed: np.ndarray, p: PhysicalConfig) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    induced = np.sqrt(np.sqrt(1.0 + speed**4 / (4.0 * p.v0_mps**4)) - speed**2 / (2.0 * p.v0_mps**2))
    return (
        p.p0_w * (1.0 + 3.0 * speed**2 / p.u_tip_mps**2)
        + 0.5 * p.d0 * p.air_density * p.rotor_solidity * p.rotor_area_m2 * speed**3
        + p.pi_w * induced
    )


def vertical_power(vz: np.ndarray, p: PhysicalConfig) -> np.ndarray:
    vz = np.asarray(vz, dtype=float)
    return p.mass_kg * p.gravity_mps2 * np.maximum(vz, 0.0) + p.vertical_loss_coeff * vz**2


def energy_breakdown(traj: Trajectory, mission: MissionConfig, physical: PhysicalConfig,
                     transmit_power: np.ndarray | None = None) -> dict[str, float]:
    vh = np.linalg.norm(traj.v[:, :2], axis=1)
    eh = mission.dt_s * float(horizontal_power(vh, physical).sum())
    ez = mission.dt_s * float(vertical_power(traj.v[:, 2], physical).sum())
    erf = 0.0
    if transmit_power is not None:
        erf = mission.dt_s / physical.pa_efficiency * float(np.asarray(transmit_power).sum())
    return {
        "horizontal_propulsion_j": eh,
        "vertical_motion_j": ez,
        "propulsion_j": eh + ez,
        "rf_j": erf,
        "total_j": eh + ez + erf,
    }


def kinematic_residuals(traj: Trajectory, mission: MissionConfig, physical: PhysicalConfig) -> dict[str, float]:
    dt = mission.dt_s
    qf, vf = traj.terminal_state(dt)
    q_target = np.asarray(mission.q_final_m)
    v_target = np.asarray(mission.v_final_mps)
    dyn = 0.0
    if mission.slots > 1:
        pred_q = traj.q[:-1] + dt * traj.v[:-1] + 0.5 * dt**2 * traj.a[:-1]
        pred_v = traj.v[:-1] + dt * traj.a[:-1]
        dyn = max(float(np.max(np.linalg.norm(traj.q[1:] - pred_q, axis=1))),
                  float(np.max(np.linalg.norm(traj.v[1:] - pred_v, axis=1))))
    return {
        "dynamics": dyn,
        "terminal_position": float(np.linalg.norm(qf - q_target)),
        "terminal_velocity": float(np.linalg.norm(vf - v_target)),
        "altitude_low": float(np.min(traj.q[:, 2] - physical.qz_min_m)),
        "altitude_high": float(np.min(physical.qz_max_m - traj.q[:, 2])),
        "horizontal_speed": float(np.min(physical.vh_max_mps - np.linalg.norm(traj.v[:, :2], axis=1))),
        "vertical_speed": float(np.min(physical.vz_max_mps - np.abs(traj.v[:, 2]))),
        "acceleration": float(np.min(physical.a_max_mps2 - np.linalg.norm(traj.a, axis=1))),
    }
