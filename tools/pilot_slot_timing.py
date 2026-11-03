"""Time one fixed-geometry covariance slot for each PCRB enforcement mode (pilot for R6)."""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uav_isac.config import load_config  # noqa: E402
from uav_isac.covariance import build_enforcement_sets, solve_covariance_slot  # noqa: E402
from uav_isac.dynamics import trapezoidal_seed  # noqa: E402
from uav_isac.experiments import SCENARIO, LINEAR, _proposed, with_overrides  # noqa: E402
from uav_isac.sensing import propagate_targets  # noqa: E402

base = _proposed(load_config(Path(__file__).resolve().parents[1] / "configs" / "reference_strict.json"))
slot = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for name, ov in (("schur", {}), ("linear", LINEAR), ("scenario", SCENARIO)):
    cfg = with_overrides(base, ov)
    traj = trapezoidal_seed(cfg.mission, cfg.physical)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    users = np.asarray(cfg.mission.users_m, float)
    t0 = time.perf_counter()
    sets = build_enforcement_sets(traj.q[slot], means[:, slot], covs[:, slot], cfg.physical, cfg.solver,
                                  cfg.scheme, np.random.default_rng(1), k_users=len(users))
    t1 = time.perf_counter()
    res = solve_covariance_slot(traj.q[slot], traj.v[slot], users, means[:, slot], covs[:, slot],
                                cfg.physical, cfg.solver, cfg.scheme, np.random.default_rng(1),
                                enforcement_sets=sets)
    t2 = time.perf_counter()
    print(f"{name:9s} points/target={len(sets[0].points):6d} build={t1-t0:7.1f}s solve={t2-t1:7.1f}s "
          f"status={res.status} P={res.objective_power_w:.4f}W rank_res={res.rank_residual:.2e}", flush=True)
