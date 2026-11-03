from __future__ import annotations

from copy import deepcopy
import numpy as np

from .config import SimulationConfig


def randomized_topology(config: SimulationConfig, seed: int) -> SimulationConfig:
    cfg = deepcopy(config)
    rng = np.random.default_rng(seed)
    k_count = len(cfg.mission.users_m)
    l_count = len(cfg.mission.targets_initial)
    users_xy = rng.uniform(50.0, 550.0, size=(k_count, 2))
    cfg.mission.users_m = np.column_stack([users_xy, np.zeros(k_count)]).tolist()

    targets = []
    attempts = 0
    while len(targets) < l_count and attempts < 10_000:
        attempts += 1
        position = rng.uniform(150.0, 450.0, size=2)
        speed = rng.uniform(2.0, 8.0)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        final = position + cfg.mission.duration_s * velocity
        if np.any(final < 50.0) or np.any(final > 550.0):
            continue
        if any(np.linalg.norm(position - np.asarray(t[:2])) < 50.0 for t in targets):
            continue
        targets.append([position[0], position[1], velocity[0], velocity[1]])
    if len(targets) != l_count:
        raise RuntimeError("Could not generate a valid random target topology.")
    cfg.mission.targets_initial = targets
    cfg.topology_seed = seed
    return cfg
