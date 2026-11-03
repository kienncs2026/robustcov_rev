"""End-to-end smoke tests on a tiny mission (2x2 UPA, 6 slots).

Each test exercises one revision feature through ``run_simulation`` and checks
the behaviour the manuscript relies on. Total runtime: a few minutes.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")

from uav_isac.ao_solver import run_simulation  # noqa: E402
from uav_isac.config import load_config  # noqa: E402
from uav_isac.experiments import with_overrides  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    cfg = load_config(ROOT / "configs" / "smoke_tiny.json")
    cfg.output_root = str(tmp_path_factory.mktemp("smoke"))
    return cfg


def _run(base, name, **ov):
    cfg = with_overrides(base, {k.replace("__", "."): v for k, v in ov.items()})
    cfg.scheme.name = name
    res = run_simulation(cfg, make_plots=False)
    return cfg, res


MOBILE = dict(scheme__fixed_trajectory=False, solver__trajectory_updates=True, solver__outer_iterations=3)


def test_linear_equals_schur(base):
    _, a = _run(base, "schur")
    _, b = _run(base, "linear", solver__pcrb_formulation="linear")
    assert a.success and b.success
    assert np.isclose(a.audit.summary["rf_j"], b.audit.summary["rf_j"], rtol=1e-4)


def test_scenario_certified_meets_risk(base):
    cfg, r = _run(base, "scenario", solver__pcrb_formulation="linear",
                  solver__pcrb_enforcement="scenario_certified")
    assert r.success
    assert r.audit.summary["max_pcrb_ucb_95"] <= cfg.physical.pcrb_violation


def test_refinement_validates(base):
    cfg, r = _run(base, "refine", solver__pcrb_formulation="linear", solver__refinement_enabled=True,
                  solver__refinement_validation_samples=300)
    s = r.audit.summary
    assert r.success and bool(s["refinement_validated"]) and s["refinement_points_added"] > 0
    assert (Path(r.run_dir) / "refinement_log.csv").exists()


def test_mobile_reuse_screening_and_stopping(base):
    _, r = _run(base, "mobile", solver__pcrb_formulation="linear", solver__candidate_screening=True,
                solver__joint_bound_check=True, **MOBILE)
    s = r.audit.summary
    it = pd.read_csv(Path(r.run_dir) / "iteration_log.csv")
    cand = it[it.trajectory_status.astype(str).str.contains("multistart")]
    assert r.success
    # One pass per evaluated candidate plus the first pass: candidate solves are reused.
    assert int(s["covariance_passes"]) == 1 + int(cand["candidate_covariance_ok"].fillna(False).astype(bool).sum())
    lb, e = cand["candidate_energy_lower_bound_j"], cand["total_energy_j"]
    ok = np.isfinite(lb) & np.isfinite(e)
    assert np.all(lb[ok] <= e[ok] + 1e-6)  # Prop. 4 is a valid lower bound
    assert (Path(r.run_dir) / "joint_bound_certificate.csv").exists()


def test_propulsion_only_lowers_propulsion(base):
    _, seed = _run(base, "straight")
    _, r = _run(base, "proponly", scheme__fixed_trajectory=False, scheme__trajectory_objective="propulsion_only",
                 solver__propulsion_only_multistart=3)
    assert r.success
    assert r.audit.summary["propulsion_j"] < seed.audit.summary["propulsion_j"]


def test_random_search_and_gaussian_rank(base):
    _, r1 = _run(base, "rs", solver__trajectory_method="random_search", solver__random_search_samples=6, **MOBILE)
    _, r2 = _run(base, "gauss", solver__rank_recovery_mode="gaussian_randomization")
    assert r1.success and r2.success
    assert r2.audit.summary["max_rank_residual"] <= 1e-4


def test_mismatch_and_lemma_tables(base):
    _, r = _run(base, "mismatch", evaluation__rician_k_db=10.0, evaluation__target_turn_rate_dps=3.0)
    d = Path(r.run_dir)
    for name in ("mismatch_rician_outage.csv", "lemma2_bcrb_check.csv", "lemma3_closed_vs_open.csv"):
        assert (d / name).exists()
    assert bool(r.audit.summary["lemma2_lower_inequality_ok"])
    assert r.audit.summary["lemma3_max_closed_over_open"] <= 1 + 1e-9


def test_certified_boundary_infeasibility(base):
    """Choose P_max between the per-point and the joint bound: only Prop. 5 detects it."""
    from uav_isac import theory
    from uav_isac.covariance import build_enforcement_sets
    from uav_isac.dynamics import trapezoidal_seed
    from uav_isac.sensing import propagate_targets
    # Hover with static targets and no process noise: every slot has the same geometry,
    # so the per-point test passes everywhere while the joint bound can still fail.
    cfg = with_overrides(base, {"mission.q_initial_m": [300.0, 300.0, 80.0],
                                "mission.q_final_m": [300.0, 300.0, 80.0],
                                "mission.targets_initial": [[340.0, 300.0, 0.0, 0.0], [300.0, 360.0, 0.0, 0.0]],
                                "physical.process_acceleration_std_mps2": 1e-6,
                                "physical.initial_velocity_std_mps": 1e-3,
                                "physical.pcrb_threshold_m2": 12.0, "solver.joint_bound_check": True})
    traj = trapezoidal_seed(cfg.mission, cfg.physical)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    sets = lambda n: build_enforcement_sets(traj.q[n], means[:, n], covs[:, n], cfg.physical, cfg.solver,
                                            cfg.scheme, np.random.default_rng(0), k_users=1)
    row = theory.joint_bound_report(traj.q, traj.v, means, covs, sets, cfg.physical, [0])[0]
    assert row["joint_bound_certified_w"] > row["per_point_max_pmin_w"] * 1.01
    cfg.physical.p_max_w = 0.5 * (row["joint_bound_certified_w"] + row["per_point_max_pmin_w"])
    cfg.scheme.name = "certified"
    r = run_simulation(cfg, make_plots=False)
    assert not r.success and r.failure_code == "INFEASIBLE_CERTIFIED_BOUNDARY"


def test_posthoc_and_outputs(base, tmp_path):
    from uav_isac import posthoc
    from uav_isac.paper_outputs import build_outputs
    _, r = _run(base, "posthoc_src")
    lw = posthoc.lipschitz_table(r.run_dir, samples=5, slots=[0, 3])
    assert (lw.L_w_m2 >= 0).all()
    s = posthoc.reaudit(r.run_dir, "mcseed1", None, seed_offset=1, samples=200)
    assert np.isfinite(s["max_pcrb_ucb_95"])
    posthoc.mismatch_sweep([r.run_dir], samples=100)
    out = build_outputs(base.output_root, tmp_path / "paper")
    index = pd.read_csv(out / "INDEX.csv")
    assert "all_runs.csv" in set(index.file)
