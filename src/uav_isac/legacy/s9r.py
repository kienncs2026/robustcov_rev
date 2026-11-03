from __future__ import annotations

"""Frozen S9R local-topology validation protocol.

S9R is deliberately separated from CP6/CP7 locked data.  It adds exactly two
pre-registered local perturbations around the CP6/CP7 reference topology and
runs the same four mechanisms already used in S9.  No redraw, rejection sampling,
or adaptive seed selection is permitted.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..config import SimulationConfig
from .postlock import S9_METHOD_IDS, scheme_for_method


S9R_TOPOLOGY_SEEDS: tuple[int, ...] = (11_000, 11_001)
S9R_PERTURBATION_BOUND_M: float = 25.0
S9R_EXPECTED_RUNS: int = len(S9R_TOPOLOGY_SEEDS) * len(S9_METHOD_IDS)


@dataclass(frozen=True)
class S9RExperiment:
    experiment: str
    method_id: str
    topology_label: str
    topology_seed: int
    perturbation_bound_m: float
    mobility_mode: str
    config: SimulationConfig


def locally_perturbed_topology(base: SimulationConfig, seed: int) -> SimulationConfig:
    """Perturb only ground-node/target initial XY coordinates around T0.

    The UAV endpoints, target velocities, altitudes, channel/uncertainty model,
    solver tolerances, Monte-Carlo seed, and all service requirements remain
    unchanged.  Perturbations are iid U[-25,25] m in x and y.  Seeds are frozen;
    there is no redraw when a topology is difficult or infeasible.
    """
    if seed not in S9R_TOPOLOGY_SEEDS:
        raise ValueError(f"S9R seed {seed} is not pre-registered: {S9R_TOPOLOGY_SEEDS}")

    cfg = deepcopy(base)
    rng = np.random.default_rng(seed)

    users = np.asarray(cfg.mission.users_m, dtype=float).copy()
    if users.ndim != 2 or users.shape[1] != 3:
        raise ValueError("mission.users_m must have shape (K,3)")
    users[:, :2] += rng.uniform(
        -S9R_PERTURBATION_BOUND_M, S9R_PERTURBATION_BOUND_M,
        size=(len(users), 2),
    )

    targets = np.asarray(cfg.mission.targets_initial, dtype=float).copy()
    if targets.ndim != 2 or targets.shape[1] != 4:
        raise ValueError("mission.targets_initial must have shape (L,4): x,y,vx,vy")
    targets[:, :2] += rng.uniform(
        -S9R_PERTURBATION_BOUND_M, S9R_PERTURBATION_BOUND_M,
        size=(len(targets), 2),
    )

    # The frozen T0 coordinates are far enough from the study-area edge that the
    # +/-25 m perturbation cannot cross the 50--550 m audit region.  Treat any
    # violation as protocol corruption rather than clipping/redrawing.
    if np.any(users[:, :2] < 50.0) or np.any(users[:, :2] > 550.0):
        raise RuntimeError("S9R user perturbation left the frozen 50--550 m region; do not redraw.")
    if np.any(targets[:, :2] < 50.0) or np.any(targets[:, :2] > 550.0):
        raise RuntimeError("S9R target perturbation left the frozen 50--550 m region; do not redraw.")

    cfg.mission.users_m = users.tolist()
    cfg.mission.targets_initial = targets.tolist()
    cfg.topology_seed = int(seed)
    return cfg


def topology_coordinate_rows(base: SimulationConfig) -> list[dict[str, object]]:
    """Return exact T0/T3/T4 coordinates and perturbations for reproducibility."""
    base_users = np.asarray(base.mission.users_m, dtype=float)
    base_targets = np.asarray(base.mission.targets_initial, dtype=float)
    rows: list[dict[str, object]] = []

    for label, seed in (("T0_locked_geometry", 0), ("T3_local_seed11000", 11_000), ("T4_local_seed11001", 11_001)):
        cfg = deepcopy(base) if seed == 0 else locally_perturbed_topology(base, seed)
        users = np.asarray(cfg.mission.users_m, dtype=float)
        targets = np.asarray(cfg.mission.targets_initial, dtype=float)
        for i, xyz in enumerate(users):
            delta = xyz[:2] - base_users[i, :2]
            rows.append({
                "topology_label": label,
                "topology_seed": seed,
                "entity_type": "user",
                "entity_index": i,
                "x_m": float(xyz[0]),
                "y_m": float(xyz[1]),
                "z_or_vx": float(xyz[2]),
                "vy": np.nan,
                "delta_x_m": float(delta[0]),
                "delta_y_m": float(delta[1]),
                "perturbation_bound_m": 0.0 if seed == 0 else S9R_PERTURBATION_BOUND_M,
            })
        for i, state in enumerate(targets):
            delta = state[:2] - base_targets[i, :2]
            rows.append({
                "topology_label": label,
                "topology_seed": seed,
                "entity_type": "target",
                "entity_index": i,
                "x_m": float(state[0]),
                "y_m": float(state[1]),
                "z_or_vx": float(state[2]),
                "vy": float(state[3]),
                "delta_x_m": float(delta[0]),
                "delta_y_m": float(delta[1]),
                "perturbation_bound_m": 0.0 if seed == 0 else S9R_PERTURBATION_BOUND_M,
            })
    return rows


def build_s9r_plan(base: SimulationConfig) -> list[S9RExperiment]:
    plan: list[S9RExperiment] = []
    for idx, seed in enumerate(S9R_TOPOLOGY_SEEDS, start=3):
        topo_cfg = locally_perturbed_topology(base, seed)
        topology_label = f"T{idx}_local_seed{seed}"
        for method_id in S9_METHOD_IDS:
            cfg = deepcopy(topo_cfg)
            cfg.scenario = "S9R"
            cfg.scheme = scheme_for_method(method_id)
            cfg.solver.random_seed = base.solver.random_seed
            plan.append(S9RExperiment(
                experiment="S9R",
                method_id=method_id,
                topology_label=topology_label,
                topology_seed=seed,
                perturbation_bound_m=S9R_PERTURBATION_BOUND_M,
                mobility_mode="fixed" if cfg.scheme.fixed_trajectory else "mobile",
                config=cfg,
            ))
    if len(plan) != S9R_EXPECTED_RUNS:
        raise RuntimeError(f"Protocol corruption: S9R has {len(plan)} runs, expected {S9R_EXPECTED_RUNS}")
    return plan
