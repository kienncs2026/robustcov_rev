"""Algorithm 2: validation-driven sensing refinement (R2.3, R3.2).

The trajectory is frozen. Each round draws fresh validation target states,
evaluates the PCRB of the current covariances, and, for every (slot, target)
whose Clopper--Pearson bound exceeds rho, adds the ``m`` violating states with
the largest illumination deficit P_min(xi) - a^H R_x a to the enforcement set.
The covariance problem is then re-solved with the enlarged sets.

The validation draws use seeds that differ from the final out-of-sample audit,
so the audit remains an independent test of the returned design.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import SimulationConfig
from .covariance import CovarianceSlotResult, build_enforcement_sets, solve_covariances
from .dynamics import Trajectory
from .sensing import binomial_upper_bound, pcrb
from .theory import pmin_point


@dataclass
class RefinementResult:
    w: np.ndarray
    s: np.ndarray
    results: list[CovarianceSlotResult]
    rows: list[dict]
    validated: bool
    rounds_used: int
    points_added: int
    runtime_s: float
    extra_points: dict[int, list[np.ndarray]]


def _validate(trajectory: Trajectory, w: np.ndarray, s: np.ndarray, target_means: np.ndarray,
              target_covariances: np.ndarray, config: SimulationConfig, rng: np.random.Generator,
              samples: int) -> list[dict]:
    phys = config.physical
    out = []
    for slot in range(config.mission.slots):
        rx = w[slot].sum(axis=0) + s[slot]
        for l in range(target_means.shape[0]):
            cov = target_covariances[l, slot]
            xs = rng.multivariate_normal(target_means[l, slot], cov, size=samples)
            vals = np.array([pcrb(trajectory.q[slot], trajectory.v[slot], xi, cov, rx, phys) for xi in xs])
            bad = vals > phys.pcrb_threshold_m2
            out.append({"slot": slot, "target": l, "violations": int(bad.sum()),
                        "ucb": binomial_upper_bound(int(bad.sum()), samples),
                        "violators": xs[bad]})
    return out


def refine_sensing(trajectory: Trajectory, w: np.ndarray, s: np.ndarray,
                   results: list[CovarianceSlotResult], users: np.ndarray,
                   target_means: np.ndarray, target_covariances: np.ndarray,
                   config: SimulationConfig, seed: int,
                   success_fn) -> RefinementResult:
    """Run Algorithm 2 on a fixed trajectory.

    ``success_fn(results)`` is the covariance acceptance test of the caller
    (status, PSD, PCRB residual, rank), so refinement never accepts a weaker design.
    """
    t0 = time.perf_counter()
    cfg = config
    phys, solver_cfg, scheme = cfg.physical, cfg.solver, cfg.scheme
    extra: dict[int, list[np.ndarray]] = {}
    rows: list[dict] = []
    added_total = 0
    validated = False
    rounds_used = 0
    for rnd in range(int(solver_cfg.refinement_rounds) + 1):
        rng = np.random.default_rng(seed + 7_777 * (rnd + 1))
        checks = _validate(trajectory, w, s, target_means, target_covariances, cfg, rng,
                           int(solver_cfg.refinement_validation_samples))
        worst = max(c["ucb"] for c in checks)
        failing = [c for c in checks if c["ucb"] > phys.pcrb_violation]
        rows.append({"round": rnd, "max_validation_ucb": worst,
                     "failing_slot_targets": len(failing), "points_added_total": added_total})
        if not failing:
            validated = True
            break
        if rnd == int(solver_cfg.refinement_rounds):
            break
        # Add the largest-deficit violators under the enforcement convention.
        for c in failing:
            slot, l = c["slot"], c["target"]
            rx = w[slot].sum(axis=0) + s[slot]
            base = build_enforcement_sets(
                trajectory.q[slot], target_means[:, slot], target_covariances[:, slot],
                phys, solver_cfg, scheme, np.random.default_rng(0), k_users=len(users))[l]
            deficits = []
            for xi in c["violators"]:
                pm, a = pmin_point(trajectory.q[slot], trajectory.v[slot], xi,
                                   target_covariances[l, slot], phys, base.threshold_m2,
                                   base.range_override_m)
                deficits.append(pm - float(np.vdot(a, rx @ a).real))
            order = np.argsort(deficits)[::-1][: int(solver_cfg.refinement_points_per_round)]
            chosen = c["violators"][order]
            slot_list = extra.setdefault(slot, [np.zeros((0, 4)) for _ in range(target_means.shape[0])])
            slot_list[l] = np.vstack([slot_list[l], chosen])
            added_total += len(chosen)
        new_w, new_s, new_results = solve_covariances(
            trajectory.q, trajectory.v, users, target_means, target_covariances,
            phys, solver_cfg, scheme, seed=seed, allow_slack=False, extra_points=extra,
            initial_w=w, initial_s=s,
        )
        rounds_used = rnd + 1
        if not success_fn(new_results):
            rows.append({"round": rnd + 1, "max_validation_ucb": np.nan,
                         "failing_slot_targets": -1, "points_added_total": added_total,
                         "note": "covariance re-solve failed; previous design kept"})
            break
        w, s, results = new_w, new_s, new_results
    return RefinementResult(w, s, results, rows, validated, rounds_used, added_total,
                            time.perf_counter() - t0, extra)
