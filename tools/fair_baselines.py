"""Fair propulsion baselines and a certified propulsion lower bound.

The paper compares the propulsion-first design with hover. Hover is a weak
reference: flying near the minimum-power speed costs less than hovering. This
module adds references that do not use communication or sensing information.

* ``propulsion_lower_bound``: a certified lower bound on the minimum propulsion
  energy E*_prop over all trajectories that satisfy the boundary conditions and
  kinematic limits (any altitude profile). It is the value of a linear program
  over horizontal speeds s_n = ||v_xy[n]||, using
      |s_{n+1} - s_n| <= a_max * dt          (||a_xy|| <= ||a|| <= a_max),
      0 <= s_n <= v_h,max,  s_1 = s_{N+1} = 0 (zero boundary velocity),
      dt * sum_n (s_n + s_{n+1}) / 2 >= ||q_F,xy - q_I,xy||   (distance),
  and a convex minorant of P_h, so that P_z >= 0 and the minorant never exceed the
  true energy. Hence E*_prop >= LB, and Corollary 2 gives the certified gap
      E_tot(theta) - E* <= E_prop(theta) - LB + E_RF(theta).
* ``straight_line_baseline``: Zeng-style straight flight q_I -> q_F with the
  energy-optimal accelerate-cruise-decelerate profile (exhaustive over the number
  of acceleration slots). Used when q_I != q_F.
* ``circle_baseline``: return-to-base loiter on a circle through the base with a
  tangential accelerate-cruise-decelerate profile; radius and acceleration slots
  minimize propulsion energy under the kinematic limits. Used when q_I == q_F.

Neither baseline uses user or target positions, except the optional heading of the
circle, which only rotates the path (propulsion energy is rotation invariant).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from uav_isac.dynamics import (Trajectory, energy_breakdown, enforce_terminal_with_last_two,
                               horizontal_power, kinematic_residuals,
                               reconstruct_from_accelerations)


# --------------------------------------------------------------------------- #
# Certified lower bound on the minimum propulsion energy
# --------------------------------------------------------------------------- #
def convex_minorant(physical, v_max: float, h: float = 1e-3, max_lines: int = 400):
    """Lines (alpha_k, beta_k) with alpha_k*v + beta_k <= P_h(v) for all v in [0, v_max].

    Lower convex hull of samples shifted down by L*h, where L bounds |P_h'|; this
    margin makes every line valid between samples as well.
    """
    v = np.arange(0.0, v_max + h, h)
    p = horizontal_power(v, physical)
    lip = 1.1 * float(np.max(np.abs(np.diff(p)) / h))
    p = p - lip * h
    hull = [0]
    for i in range(1, len(v)):  # monotone chain, lower hull
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            cross = (v[i2] - v[i1]) * (p[i] - p[i1]) - (p[i2] - p[i1]) * (v[i] - v[i1])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    lines = []
    for a, b in zip(hull[:-1], hull[1:]):
        slope = (p[b] - p[a]) / (v[b] - v[a])
        lines.append((slope, p[a] - slope * v[a]))
    lines = np.array(lines)
    if len(lines) > max_lines:  # dropping cuts keeps the bound valid (only weaker)
        keep = np.unique(np.linspace(0, len(lines) - 1, max_lines).round().astype(int))
        lines = lines[keep]
    return lines, lip * h


def propulsion_lower_bound(mission, physical) -> dict:
    n, dt = mission.slots, mission.dt_s
    q0 = np.asarray(mission.q_initial_m, float)
    qf = np.asarray(mission.q_final_m, float)
    v0 = np.asarray(mission.v_initial_mps, float)
    vf = np.asarray(mission.v_final_mps, float)
    dist = float(np.linalg.norm(qf[:2] - q0[:2]))
    lines, margin = convex_minorant(physical, physical.vh_max_mps)
    # variables: s_1..s_{N+1} (N+1), t_1..t_N (N)
    ns, nt = n + 1, n
    c = np.concatenate([np.zeros(ns), dt * np.ones(nt)])
    A, b = [], []
    for k in range(n):  # |s_{k+1}-s_k| <= a dt
        for sgn in (1, -1):
            row = np.zeros(ns + nt); row[k + 1] = sgn; row[k] = -sgn
            A.append(row); b.append(physical.a_max_mps2 * dt)
    for k in range(n):  # t_k >= alpha s_k + beta
        for al, be in lines:
            row = np.zeros(ns + nt); row[k] = al; row[ns + k] = -1.0
            A.append(row); b.append(-be)
    if dist > 0:  # -dt*sum (s_k+s_{k+1})/2 <= -dist
        row = np.zeros(ns + nt)
        row[:n] -= dt / 2; row[1:n + 1] -= dt / 2
        A.append(row); b.append(-dist)
    bounds = [(0.0, physical.vh_max_mps)] * ns + [(None, None)] * nt
    s1 = float(np.linalg.norm(v0[:2])); sn = float(np.linalg.norm(vf[:2]))
    bounds[0] = (s1, s1); bounds[n] = (sn, sn)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"lower-bound LP failed: {res.message}")
    return {"propulsion_lb_j": float(res.fun), "lb_minorant_margin_w": float(margin),
            "lb_distance_m": dist, "lb_speeds_mps": res.x[:ns].tolist()}


# --------------------------------------------------------------------------- #
# Baseline trajectories (no communication or sensing information)
# --------------------------------------------------------------------------- #
def _finalize(a: np.ndarray, mission, physical, altitude: float) -> Trajectory:
    a = a.copy(); a[:, 2] = 0.0
    a = enforce_terminal_with_last_two(a, mission)
    q, v = reconstruct_from_accelerations(a, mission)
    q[:, 2] = altitude; v[:, 2] = 0.0; a[:, 2] = 0.0
    return Trajectory(q=q, v=v, a=a)


def _feasible(traj: Trajectory, mission, physical, tol: float = 1e-6) -> bool:
    r = kinematic_residuals(traj, mission, physical)
    eq = max(r["dynamics"], r["terminal_position"], r["terminal_velocity"]) <= 1e-6
    ineq = min(r["altitude_low"], r["altitude_high"], r["horizontal_speed"],
               r["vertical_speed"], r["acceleration"]) >= -tol
    return bool(eq and ineq)


def straight_line_baseline(mission, physical) -> tuple[Trajectory, dict]:
    """Energy-optimal accelerate-cruise-decelerate flight along the straight line."""
    n, dt = mission.slots, mission.dt_s
    q0 = np.asarray(mission.q_initial_m, float); qf = np.asarray(mission.q_final_m, float)
    d = qf - q0; dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        raise ValueError("straight-line baseline needs q_I != q_F")
    u = d / dist
    best = None
    for m in range(1, n // 2 + 1):  # D = a dt^2 m (N-m)
        acc = dist / (dt**2 * m * (n - m))
        a = np.zeros((n, 3)); a[:m] = acc * u; a[n - m:] = -acc * u
        traj = _finalize(a, mission, physical, q0[2])
        if not _feasible(traj, mission, physical):
            continue
        e = energy_breakdown(traj, mission, physical)["propulsion_j"]
        if best is None or e < best[0]:
            best = (e, m, acc, traj)
    e, m, acc, traj = best
    return traj, {"baseline": "straight_line", "accel_slots": m, "accel_mps2": acc,
                  "cruise_speed_mps": acc * dt * m, "propulsion_j": e}


def _circle_trajectory(radius, m, heading, mission, physical) -> Trajectory | None:
    n, dt = mission.slots, mission.dt_s
    base = np.asarray(mission.q_initial_m, float)
    length = 2 * np.pi * radius
    acc = length / (dt**2 * m * (n - m))  # tangential acceleration
    t = np.arange(n + 1) * dt
    t_a, t_d = m * dt, (n - m) * dt
    cruise = acc * t_a
    speed = np.where(t <= t_a, acc * t, np.where(t <= t_d, cruise, cruise - acc * (t - t_d)))
    arc = np.where(t <= t_a, 0.5 * acc * t**2,
                   np.where(t <= t_d, 0.5 * acc * t_a**2 + cruise * (t - t_a),
                            length - 0.5 * acc * (n * dt - t)**2))
    center = base[:2] + radius * np.array([np.cos(heading), np.sin(heading)])
    theta = heading + np.pi + arc / radius  # counter-clockwise from the base
    tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    vel = np.zeros((n + 1, 3)); vel[:, :2] = speed[:, None] * tangent
    a = (vel[1:] - vel[:-1]) / dt  # exact boundary velocities, O(dt^2) position error
    traj = _finalize(a, mission, physical, base[2])
    return traj if _feasible(traj, mission, physical) else None


def circle_baseline(mission, physical, heading: float = 0.0,
                    radii=np.arange(10.0, 160.5, 0.5)) -> tuple[Trajectory, dict]:
    """Return-to-base circular loiter with an energy-optimal tangential speed profile."""
    q0 = np.asarray(mission.q_initial_m, float); qf = np.asarray(mission.q_final_m, float)
    if np.linalg.norm(qf - q0) > 1e-9:
        raise ValueError("circle baseline needs q_I == q_F")
    best = None
    for r in radii:
        for m in range(1, mission.slots // 2 + 1):
            traj = _circle_trajectory(r, m, heading, mission, physical)
            if traj is None:
                continue
            e = energy_breakdown(traj, mission, physical)["propulsion_j"]
            if best is None or e < best[0]:
                best = (e, r, m, traj)
    if best is None:
        raise RuntimeError("no kinematically feasible circle found")
    e, r, m, traj = best
    vh = np.linalg.norm(traj.v[:, :2], axis=1)
    return traj, {"baseline": "circle", "radius_m": float(r), "accel_slots": m,
                  "heading_rad": float(heading), "max_speed_mps": float(vh.max()),
                  "propulsion_j": e}


def fair_baseline(mission, physical, heading: float = 0.0):
    q0 = np.asarray(mission.q_initial_m, float); qf = np.asarray(mission.q_final_m, float)
    if np.linalg.norm(qf - q0) < 1e-9:
        return circle_baseline(mission, physical, heading=heading)
    return straight_line_baseline(mission, physical)
