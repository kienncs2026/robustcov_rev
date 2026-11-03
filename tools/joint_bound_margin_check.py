"""Proposition 3 joint-illumination bound at slot 1 with and without the 10 % design margin.

    python tools/joint_bound_margin_check.py            # writes results/R11/joint_bound_margin_check.csv

Reproduces the Section V-C statement: with the tightened threshold eps_des = 0.9 eps the
rescaled dual certifies P_J[1] > P_max = 2 W at eps = 4 m^2 (2.023 W); removing the margin
(eps_des = eps) lowers the bound and no longer proves infeasibility. The slot-1 UAV state is
fixed by the boundary conditions, so the bound is trajectory independent.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_isac import theory  # noqa: E402
from uav_isac.ao_solver import _enforcement_sets  # noqa: E402
from uav_isac.config import load_config  # noqa: E402
from uav_isac.dynamics import trapezoidal_seed  # noqa: E402
from uav_isac.experiments import build  # noqa: E402
from uav_isac.sensing import propagate_targets  # noqa: E402


def main() -> None:
    base = load_config(ROOT / "configs" / "reference_strict.json")
    rows = []
    for item in build("R11", base, {}):
        if "eps4" not in item.label:
            continue
        for margin in (base.physical.enforcement_margin, 0.0):
            cfg = item.config
            cfg = replace(cfg, physical=replace(cfg.physical, enforcement_margin=margin))
            seed = trapezoidal_seed(cfg.mission, cfg.physical, altitude_m=cfg.scheme.fixed_altitude_m)
            users = np.asarray(cfg.mission.users_m, float)
            means, covs = propagate_targets(cfg.mission, cfg.physical)
            jb = theory.joint_bound_report(
                seed.q, seed.v, means, covs,
                lambda slot: _enforcement_sets(seed, slot, users, means, covs, cfg),
                cfg.physical, [0], solver=cfg.solver.fallback_solver or "CLARABEL")[0]
            rows.append({"run": item.label, "rate": cfg.physical.min_spectral_efficiency,
                         "eps_m2": cfg.physical.pcrb_threshold_m2, "enforcement_margin": margin,
                         "eps_design_m2": cfg.physical.pcrb_threshold_m2 * (1.0 - margin), **jb})
            print(f"{item.label} margin={margin:.2f}: per-point {jb['per_point_max_pmin_w']:.4f} W, "
                  f"certified {jb['joint_bound_certified_w']:.4f} W, infeasible={jb['certified_infeasible']}",
                  flush=True)
    out = ROOT / "results" / "R11" / "joint_bound_margin_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
