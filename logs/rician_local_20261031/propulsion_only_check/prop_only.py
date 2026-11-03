import sys, time
sys.path.insert(0, "src")
import numpy as np, scipy
from uav_isac.config import load_config
from uav_isac.experiments import build
from uav_isac.dynamics import trapezoidal_seed
from uav_isac.baselines import propulsion_only_trajectory
print("scipy", scipy.__version__, "numpy", np.__version__, flush=True)
base = load_config("configs/reference_strict.json")
for it in build("R2", base, {}):
    if "propulsion_only" not in it.label: continue
    cfg = it.config
    seed = trapezoidal_seed(cfg.mission, cfg.physical, altitude_m=cfg.scheme.fixed_altitude_m)
    t = time.perf_counter()
    traj, info = propulsion_only_trajectory(seed, cfg, rng_seed=cfg.solver.random_seed + 61_000)
    print(it.label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()},
          f"{time.perf_counter()-t:.0f}s", flush=True)
