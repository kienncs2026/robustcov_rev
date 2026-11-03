"""Post-hoc analyses on completed run directories (no re-optimization).

* :func:`reaudit` re-evaluates a stored design under other evaluation settings
  (model mismatch R1.4, independent Monte Carlo seeds R1.9) and writes
  ``<run_dir>/reaudit/<tag>/``.
* :func:`lipschitz_table` estimates L_w of Lemma 4 per slot and target (R1.3).
* :func:`joint_bound_table` evaluates Proposition 5 on every slot of a stored design.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import audit_solution
from .config import load_config
from .dynamics import Trajectory
from .sensing import propagate_targets


def load_design(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    t = pd.read_csv(run_dir / "trajectory.csv")
    traj = Trajectory(q=t[["qx_m", "qy_m", "qz_m"]].to_numpy(),
                      v=t[["vx_mps", "vy_mps", "vz_mps"]].to_numpy(),
                      a=t[["ax_mps2", "ay_mps2", "az_mps2"]].to_numpy())
    with np.load(run_dir / "covariances.npz") as arr:
        return cfg, traj, arr["W"], arr["S"]


def reaudit(run_dir: str | Path, tag: str, evaluation: dict | None = None, seed_offset: int = 0,
            samples: int | None = None, channel_samples: int | None = None) -> dict:
    run_dir = Path(run_dir)
    cfg, traj, w, s = load_design(run_dir)
    cfg = deepcopy(cfg)
    for key, value in (evaluation or {}).items():
        setattr(cfg.evaluation, key, value)
    users = np.asarray(cfg.mission.users_m, float)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    audit = audit_solution(traj, w, s, users, means, covs, cfg.mission, cfg.physical, cfg.solver,
                           cfg.scheme, cfg.evaluation,
                           seed=cfg.solver.random_seed + 999_999 + 1_000_003 * int(seed_offset),
                           audit_samples=samples, channel_samples=channel_samples,
                           exact_robust=channel_samples is None or channel_samples > 0,
                           enforcement_pcrb_residual_m2=0.0)
    out = run_dir / "reaudit" / tag
    out.mkdir(parents=True, exist_ok=True)
    summary = {"tag": tag, "seed_offset": seed_offset, **(evaluation or {}), **audit.summary}
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(audit.pcrb_rows).to_csv(out / "pcrb_out_of_sample.csv", index=False)
    pd.DataFrame(audit.communication_rows).to_csv(out / "constraint_residuals.csv", index=False)
    for name, rows in audit.extra_tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    return summary


def lipschitz_table(run_dir: str | Path, samples: int = 100, slots: list[int] | None = None,
                    seed: int = 4242) -> pd.DataFrame:
    from .theory import whitened_lipschitz
    cfg, traj, w, s = load_design(run_dir)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    rng = np.random.default_rng(seed)
    kappa = cfg.physical.confidence_radius
    eps = cfg.physical.pcrb_threshold_m2
    eps_des = eps * (1.0 - cfg.physical.enforcement_margin)
    rows = []
    for n in (slots if slots is not None else range(cfg.mission.slots)):
        rx = w[n].sum(axis=0) + s[n]
        for l in range(means.shape[0]):
            lw = whitened_lipschitz(traj.q[n], traj.v[n], means[l, n], covs[l, n], rx,
                                    cfg.physical, kappa, samples, rng)
            rows.append({"slot": n, "target": l, "L_w_m2": lw, "kappa": kappa,
                         "required_margin_m2": lw * kappa, "available_margin_m2": eps - eps_des,
                         "lemma4_sufficient": bool(lw * kappa <= eps - eps_des)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(run_dir) / "lemma4_lipschitz.csv", index=False)
    return df


def joint_bound_table(run_dir: str | Path) -> pd.DataFrame:
    from .covariance import build_enforcement_sets
    from .theory import joint_bound_report
    cfg, traj, _, _ = load_design(run_dir)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    users = cfg.mission.users_m

    def sets(slot):
        return build_enforcement_sets(traj.q[slot], means[:, slot], covs[:, slot], cfg.physical,
                                      cfg.solver, cfg.scheme,
                                      np.random.default_rng(cfg.solver.random_seed + 1009 * slot),
                                      k_users=len(users))
    rows = joint_bound_report(traj.q, traj.v, means, covs, sets, cfg.physical,
                              list(range(cfg.mission.slots)))
    df = pd.DataFrame(rows)
    df.to_csv(Path(run_dir) / "joint_bound_all_slots.csv", index=False)
    return df


MISMATCH_GRID = (
    {"rician_k_db": 15.0}, {"rician_k_db": 10.0}, {"rician_k_db": 5.0},
    {"target_turn_rate_dps": 2.0}, {"target_turn_rate_dps": 5.0},
)


def mismatch_sweep(run_dirs: list[str | Path], samples: int | None = None) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        for ev in MISMATCH_GRID:
            tag = "mismatch_" + "_".join(f"{k}{v:g}" for k, v in ev.items())
            rows.append({"run_dir": Path(rd).name, **reaudit(rd, tag, ev, samples=samples)})
    return pd.DataFrame(rows)


def seed_sweep(run_dirs: list[str | Path], seeds: int = 3, samples: int | None = None) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        for k in range(1, seeds + 1):
            rows.append({"run_dir": Path(rd).name, **reaudit(rd, f"mcseed{k}", None, seed_offset=k,
                                                            samples=samples)})
    return pd.DataFrame(rows)
