"""Fair trajectory baselines: propulsion-only paths that ignore users and targets.

For each mission the script
  1. computes the certified propulsion lower bound LB (tools/fair_baselines.py),
  2. builds the fair baseline path (straight line for the reference mission,
     circle through the base for the return-to-base mission),
  3. solves the robust covariance problem on that path and audits it,
so the paper can state: on the energy-efficient path that a propulsion-only
planner would fly, the robust design guarantees ISAC feasibility, and the total
energy is within E_tot - LB of the optimum (Corollary 2).

Usage (from robustcov_rev/):
  python tools/run_fair_baselines.py                      # both missions, ISAC check
  python tools/run_fair_baselines.py --skip-isac          # energies and LB only (seconds)
  python tools/run_fair_baselines.py --headings 0 90 180 270
Outputs: results_rev2/fair/<mission>_<baseline>[_h<deg>].json and .npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from rev2_common import (MISSIONS, ROOT, energy_breakdown, evaluate_path,  # noqa: E402
                         mission_config, save_traj, write_json)
from fair_baselines import (circle_baseline, propulsion_lower_bound,  # noqa: E402
                            straight_line_baseline)
from uav_isac.dynamics import Trajectory, trapezoidal_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", default=list(MISSIONS), choices=MISSIONS)
    ap.add_argument("--headings", nargs="+", type=float, default=[90.0],
                    help="circle headings in degrees (return-to-base mission); 90 = +y")
    ap.add_argument("--skip-isac", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results_rev2" / "fair"))
    args = ap.parse_args()
    out = Path(args.out)

    for which in args.missions:
        cfg = mission_config(which)
        mis, phy = cfg.mission, cfg.physical
        lb = propulsion_lower_bound(mis, phy)
        seed = trapezoidal_seed(mis, phy, altitude_m=cfg.scheme.fixed_altitude_m)
        ref_energy = energy_breakdown(seed, mis, phy)["propulsion_j"]  # hover or straight seed
        common = {"mission": which, "propulsion_lb_j": lb["propulsion_lb_j"],
                  "seed_propulsion_j": ref_energy}
        if which == "reference":
            jobs = [("straight", None, *straight_line_baseline(mis, phy))]
        else:
            jobs = [(f"circle_h{int(h)}", h, *circle_baseline(mis, phy, heading=np.deg2rad(h)))
                    for h in args.headings]
        for name, _, traj, info in jobs:
            if (out / f"{which}_{name}.json").exists():
                print(f"[{which}/{name}] exists, skipped", flush=True)
                continue
            rec = {**common, **info,
                   # same definitions as the Table III columns
                   "max_displacement_from_seed_m": float(np.max(np.linalg.norm(traj.q - seed.q, axis=1))),
                   "max_horizontal_speed_mps": float(np.max(np.linalg.norm(traj.v[:, :2], axis=1)))}
            save_traj(out / f"{which}_{name}.npz", traj)
            if not args.skip_isac:
                print(f"[{which}/{name}] robust covariance + audit ...", flush=True)
                rec["isac"] = evaluate_path(cfg, traj, seed=cfg.solver.random_seed)
                if rec["isac"].get("total_j") is not None:
                    rec["certified_gap_j"] = rec["isac"]["total_j"] - lb["propulsion_lb_j"]
            write_json(out / f"{which}_{name}.json", rec)
            print(f"[{which}/{name}] E_prop={info['propulsion_j']:.2f} J, LB={lb['propulsion_lb_j']:.2f} J"
                  + ("" if args.skip_isac else
                     f", ISAC ok={rec['isac'].get('accepted')}, RF={rec['isac'].get('rf_j')}"),
                  flush=True)


if __name__ == "__main__":
    main()
