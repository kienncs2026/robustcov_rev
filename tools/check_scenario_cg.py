"""Check scenario constraint generation against the direct full-row solve.

    python tools/check_scenario_cg.py 24 150 compare   # CG vs all rows, same samples
    python tools/check_scenario_cg.py 24 0 cg          # CG only, a priori N_s
"""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uav_isac import covariance  # noqa: E402
from uav_isac.config import load_config  # noqa: E402
from uav_isac.covariance import build_enforcement_sets, ensure_linear_data, solve_covariance_slot  # noqa: E402
from uav_isac.dynamics import trapezoidal_seed  # noqa: E402
from uav_isac.experiments import SCENARIO, _proposed, with_overrides  # noqa: E402
from uav_isac.sensing import propagate_targets  # noqa: E402

slot, n, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
base = _proposed(load_config(Path(__file__).resolve().parents[1] / "configs" / "reference_strict.json"))
over = dict(SCENARIO)
if n:
    over["solver.scenario_max_samples"] = n
cfg = with_overrides(base, over)
traj = trapezoidal_seed(cfg.mission, cfg.physical)
means, covs = propagate_targets(cfg.mission, cfg.physical)
users = np.asarray(cfg.mission.users_m, float)


def run(tag):
    sets = build_enforcement_sets(traj.q[slot], means[:, slot], covs[:, slot], cfg.physical, cfg.solver,
                                  cfg.scheme, np.random.default_rng(1), k_users=len(users))
    t0 = time.perf_counter()
    for l, es in enumerate(sets):
        ensure_linear_data(traj.q[slot], traj.v[slot], es, covs[l, slot], cfg.physical)
    t1 = time.perf_counter()
    res = solve_covariance_slot(traj.q[slot], traj.v[slot], users, means[:, slot], covs[:, slot],
                                cfg.physical, cfg.solver, cfg.scheme, np.random.default_rng(1),
                                enforcement_sets=sets)
    t2 = time.perf_counter()
    rx = res.w.sum(axis=0) + res.s
    worst = max(float((es.pmin_w - np.einsum("pi,ij,pj->p", es.steering.conj(), rx, es.steering).real).max())
                for es in sets)
    print(f"{tag}: N={[len(es.points) for es in sets]} pmin={t1-t0:.1f}s solve={t2-t1:.1f}s "
          f"status={res.status} P={res.objective_power_w:.6f}W residual={res.pcrb_residual_m2:.3e}m2 "
          f"max_gap={worst:.2e}W iters={res.solver_iterations} used={res.solver_used}", flush=True)
    return res


cg = run("cg")
if mode == "compare":
    covariance.SCENARIO_CG_INITIAL = 10**9  # disable CG: all rows in one solve
    full = run("full")
    print(f"relative power difference = {abs(cg.objective_power_w - full.objective_power_w) / full.objective_power_w:.2e}")
