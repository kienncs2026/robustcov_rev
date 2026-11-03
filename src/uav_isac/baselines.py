"""Trajectory baselines requested in the review (R1.7, R2.5).

* ``propulsion_only_trajectory``: minimizes the propulsion energy of Eq. (5)
  under the kinematic limits and boundary conditions only. Communication and
  sensing do not enter the search; the covariance problem is solved once on the
  resulting path. Comparing it with the proposed design separates savings caused
  by the rotor model from gains caused by communication/sensing geometry.
* The straight-line trapezoidal seed and fixed hover are obtained with
  ``SchemeConfig(fixed_trajectory=True)`` (seed of ``ao_solver``).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .config import SimulationConfig
from .dynamics import Trajectory, energy_breakdown
from .trajectory_opt import (
    _candidate_from_deltas,
    _is_kinematically_feasible,
    _kinematic_constraint_vector,
)


def propulsion_only_trajectory(seed: Trajectory, config: SimulationConfig,
                               rng_seed: int) -> tuple[Trajectory, dict]:
    mission, physical, solver_cfg, scheme = (
        config.mission, config.physical, config.solver, config.scheme)
    segments = min(int(solver_cfg.propulsion_only_segments), max(1, mission.slots - 2))
    dim = 3 * segments
    bound = float(physical.a_max_mps2)
    lower = np.full(dim, -bound)
    upper = np.full(dim, bound)
    # Level flight is optimal when the boundary altitudes coincide: P_h depends only on
    # the horizontal speed and the vertical term is nonnegative, zero only at v_z = 0.
    # Restricting the returned path to level flight is therefore exact, and it removes
    # the altitude oscillations Powell used to leave in (1,017 J displaced, 174 J reference).
    level = scheme.fixed_altitude_m is not None or bool(
        np.isclose(mission.q_initial_m[2], mission.q_final_m[2]))
    if level:
        lower[2::3] = 0.0
        upper[2::3] = 0.0
    free = np.flatnonzero(upper > lower)

    def embed(y: np.ndarray) -> np.ndarray:
        x = np.zeros(dim)
        x[free] = y
        return x

    def build(x: np.ndarray) -> Trajectory:
        return _candidate_from_deltas(seed, x, segments, mission, scheme)

    def objective(x: np.ndarray) -> float:
        cand = build(x)
        smooth = float(np.sum(np.diff(cand.a, axis=0) ** 2))
        return energy_breakdown(cand, mission, physical)["propulsion_j"] + 1e-3 * smooth

    def cons(x: np.ndarray) -> np.ndarray:
        return _kinematic_constraint_vector(build(x), physical)

    rng = np.random.default_rng(rng_seed)
    starts = [np.zeros(dim)]
    for _ in range(max(0, int(solver_cfg.propulsion_only_multistart) - 1)):
        x = rng.normal(size=dim)
        x *= 0.25 * bound / max(np.linalg.norm(x, ord=np.inf), 1e-12)
        if level:
            x[2::3] = 0.0
        starts.append(x)

    t0 = time.perf_counter()
    best_x, best_val = np.zeros(dim), objective(np.zeros(dim))
    iterations = 0
    if str(getattr(solver_cfg, "propulsion_only_method", "powell")).lower() == "powell":
        # Hover is a stationary point of P_h (dP_h/dv = 0 at v = 0), so gradient
        # methods started at hover do not move. Derivative-free multi-start Powell
        # with an exterior penalty (1e-3 margin on every kinematic limit) is used.
        def penalized(x: np.ndarray) -> float:
            cand = build(x)
            viol = np.minimum(_kinematic_constraint_vector(cand, physical) - 1e-3, 0.0)
            return objective(x) + 1e6 * float(np.sum(viol ** 2))
        rng_p = np.random.default_rng(rng_seed)
        p_starts = [np.zeros(dim)]
        for _ in range(max(0, int(solver_cfg.propulsion_only_multistart) - 1)):
            x = rng_p.normal(size=dim) * rng_p.uniform(0.3, 2.5)
            if scheme.fixed_altitude_m is not None:
                x[2::3] = 0.0
            p_starts.append(x)
        powell_opts = {"maxiter": 20_000, "xtol": 1e-4, "ftol": 1e-8}
        for x0 in p_starts:
            # The full-space search finds better horizontal paths than a search restricted
            # to the horizontal increments (27,048 vs 26,111 J on the displaced geometry),
            # so it is kept; with level boundaries its vertical part is then dropped and
            # the horizontal increments are polished from the projected point.
            res = minimize(penalized, x0, method="Powell", options=powell_opts)
            iterations += int(getattr(res, "nit", 0) or 0)
            trials = [res.x]
            if level:
                y0 = res.x[free]
                pol = minimize(lambda y: penalized(embed(y)), y0, method="Powell", options=powell_opts)
                iterations += int(getattr(pol, "nit", 0) or 0)
                trials = [embed(y0), embed(pol.x)]
            for x_full in trials:
                if _is_kinematically_feasible(build(x_full), mission, physical,
                                              solver_cfg.trajectory_feasibility_tolerance):
                    val = objective(x_full)
                    if val < best_val:
                        best_x, best_val = x_full.copy(), val
        starts = []
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP",
                       bounds=list(zip(lower, upper)),
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": int(solver_cfg.propulsion_only_maxiter),
                                "ftol": 1e-6, "disp": False})
        iterations += int(getattr(res, "nit", 0) or 0)
        cand = build(res.x)
        if _is_kinematically_feasible(cand, mission, physical,
                                      solver_cfg.trajectory_feasibility_tolerance):
            val = objective(res.x)
            if val < best_val:
                best_x, best_val = res.x.copy(), val
    traj = build(best_x)
    info = {
        "propulsion_only_seed_propulsion_j": energy_breakdown(seed, mission, physical)["propulsion_j"],
        "propulsion_only_propulsion_j": energy_breakdown(traj, mission, physical)["propulsion_j"],
        "propulsion_only_runtime_s": time.perf_counter() - t0,
        "propulsion_only_iterations": iterations,
        "propulsion_only_max_displacement_m": float(np.max(np.linalg.norm(traj.q - seed.q, axis=1))),
        # Marks runs made after the level-flight fix (31/10); older runs lack the column.
        "propulsion_only_level": bool(level),
        "propulsion_only_vertical_j": energy_breakdown(traj, mission, physical)["vertical_motion_j"],
    }
    return traj, info
