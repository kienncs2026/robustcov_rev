"""Profile one scenario-certified covariance slot as the sample count grows.

    python tools/profile_scenario.py 24 250 1000 4000

Splits the time into P_min evaluation, CVXPY compilation, and solver time, and
reports SCS iterations, so the cost of the full N_s = 29,142 run can be extrapolated.
"""
import sys, time
from pathlib import Path

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uav_isac.config import load_config  # noqa: E402
from uav_isac.covariance import build_enforcement_sets, ensure_linear_data, solve_covariance_slot  # noqa: E402
from uav_isac.dynamics import trapezoidal_seed  # noqa: E402
from uav_isac.experiments import SCENARIO, _proposed, with_overrides  # noqa: E402
from uav_isac.sensing import propagate_targets  # noqa: E402

calls = []
_orig_solve = cp.Problem.solve


def _traced_solve(self, *args, **kwargs):
    t = time.perf_counter()
    out = _orig_solve(self, *args, **kwargs)
    st = self.solver_stats
    calls.append({"wall": time.perf_counter() - t, "compile": getattr(self, "compilation_time", None),
                  "solve": getattr(st, "solve_time", None), "iters": getattr(st, "num_iters", None),
                  "status": self.status, "solver": kwargs.get("solver")})
    return out


cp.Problem.solve = _traced_solve

slot = int(sys.argv[1])
counts = [int(x) for x in sys.argv[2:]] or [250, 1000]
base = _proposed(load_config(Path(__file__).resolve().parents[1] / "configs" / "reference_strict.json"))
for n in counts:
    cfg = with_overrides(base, {**SCENARIO, "solver.scenario_max_samples": n})
    traj = trapezoidal_seed(cfg.mission, cfg.physical)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    users = np.asarray(cfg.mission.users_m, float)
    t0 = time.perf_counter()
    sets = build_enforcement_sets(traj.q[slot], means[:, slot], covs[:, slot], cfg.physical, cfg.solver,
                                  cfg.scheme, np.random.default_rng(1), k_users=len(users))
    for l, es in enumerate(sets):
        ensure_linear_data(traj.q[slot], traj.v[slot], es, covs[l, slot], cfg.physical)
    t1 = time.perf_counter()
    active = [int((es.pmin_w > 0).sum()) for es in sets]
    calls.clear()
    res = solve_covariance_slot(traj.q[slot], traj.v[slot], users, means[:, slot], covs[:, slot],
                                cfg.physical, cfg.solver, cfg.scheme, np.random.default_rng(1),
                                enforcement_sets=sets)
    t2 = time.perf_counter()
    print(f"N={n:6d} active={active} pmin={t1-t0:6.1f}s solve_slot={t2-t1:7.1f}s status={res.status} "
          f"P={res.objective_power_w:.4f}W", flush=True)
    for c in calls:
        print(f"    cvxpy call: wall={c['wall']:7.1f}s compile={c['compile'] or 0:6.1f}s "
              f"solver={c['solve'] or 0:7.1f}s iters={c['iters']} status={c['status']}", flush=True)
