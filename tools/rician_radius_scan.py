import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np
from dataclasses import replace
from uav_isac.config import load_config
from uav_isac.experiments import LINEAR, _fixed_geometry, with_overrides
from uav_isac.dynamics import trapezoidal_seed
from uav_isac.sensing import propagate_targets
from uav_isac.covariance import solve_covariance_slot
base = with_overrides(_fixed_geometry(load_config("configs/reference_strict.json")), LINEAR)
traj = trapezoidal_seed(base.mission, base.physical, altitude_m=base.scheme.fixed_altitude_m)
users = np.asarray(base.mission.users_m, float)
means, covs = propagate_targets(base.mission, base.physical)
slots = [int(s) for s in sys.argv[2].split(",")]
for r in [float(x) for x in sys.argv[1].split(",")]:
    phys = replace(base.physical, csi_error_radius=r)
    for slot in slots:
        for solver in ("SCS",):
            t = time.perf_counter()
            res = solve_covariance_slot(traj.q[slot], traj.v[slot], users, means[:, slot], covs[:, slot], phys,
                                        replace(base.solver, solver=solver), base.scheme,
                                        np.random.default_rng(base.solver.random_seed + 1009 * slot))
            print(f"r={r:.4f} slot={slot:2d} {solver:8s} status={res.status:24s} P={res.objective_power_w:.4f} W "
                  f"({time.perf_counter()-t:.0f}s)", flush=True)
