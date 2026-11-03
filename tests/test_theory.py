"""Fast checks of the analytical results (no conic solver unless noted)."""
import numpy as np
import pytest

from uav_isac import theory
from uav_isac.config import SimulationConfig
from uav_isac.dynamics import enforce_terminal_with_last_two, reconstruct_from_accelerations, trapezoidal_seed
from uav_isac.geometry import upa_steering
from uav_isac.sensing import measurement_jacobian, pcrb, propagate_targets

CFG = SimulationConfig()
PH = CFG.physical
PH.p_max_w = 2.0


def _geometry(seed=0):
    rng = np.random.default_rng(seed)
    q = np.array([rng.uniform(50, 550), rng.uniform(50, 550), rng.uniform(50, 130)])
    v = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), 0.0])
    xi = np.array([rng.uniform(50, 550), rng.uniform(50, 550), rng.uniform(-6, 6), rng.uniform(-6, 6)])
    _, covs = propagate_targets(CFG.mission, PH)
    return q, v, xi, covs[0, int(rng.integers(0, CFG.mission.slots))]


@pytest.mark.parametrize("seed", range(5))
def test_prop1_scalar_form_matches_pcrb(seed):
    q, v, xi, P = _geometry(seed)
    c, lam = theory.psi_spectrum(q, v, xi, P, PH)
    a = upa_steering(q, np.array([xi[0], xi[1], 0.0]), PH.mx, PH.my, PH.wavelength_m)
    for power in (0.01, 0.3, 2.0):
        rx = power * np.outer(a, a.conj())
        r = np.linalg.norm(np.array([xi[0], xi[1], 0.0]) - q)
        gamma = PH.sensing_reference_snr_linear * (PH.sensing_reference_range_m / r) ** 4 * power / PH.sensing_reference_power_w
        assert np.isclose(theory.psi_of_gamma(c, lam, gamma), pcrb(q, v, xi, P, rx, PH), rtol=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_jacobian_rank_at_most_three_and_strict_decrease(seed):
    q, v, xi, P = _geometry(seed)
    assert np.linalg.matrix_rank(measurement_jacobian(q, v, xi, PH.fc_hz)) <= 3
    c, lam = theory.psi_spectrum(q, v, xi, P, PH)
    assert float(np.sum(c * lam)) > 0.0  # strict decrease condition of Prop. 1


@pytest.mark.parametrize("seed", range(5))
def test_prop2_threshold_is_exact(seed):
    q, v, xi, P = _geometry(seed)
    eps = 0.5 * float(np.trace(P[:2, :2]))
    pm, a = theory.pmin_point(q, v, xi, P, PH, eps)
    assert np.isfinite(pm) and pm > 0
    at = lambda s: pcrb(q, v, xi, P, s * pm * np.outer(a, a.conj()), PH)
    assert at(1.0) <= eps * (1 + 1e-6)
    assert at(0.999) > eps


def test_lemma4_covering_radius_equals_kappa():
    kappa = 3.08
    pts = np.vstack([np.zeros(4)] + [s * kappa * np.eye(4)[i] for i in range(4) for s in (1, -1)])
    diag = kappa * np.ones(4) / 2
    assert np.isclose(np.linalg.norm(diag), kappa)
    assert np.isclose(np.min(np.linalg.norm(pts - diag, axis=1)), kappa)
    mc = theory.nine_point_covering_radius(kappa, samples=20_000)
    assert 0.8 * kappa <= mc <= kappa + 1e-9


def test_terminal_reconstruction_exact():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(CFG.mission.slots, 3))
    a = enforce_terminal_with_last_two(a, CFG.mission)
    q, v = reconstruct_from_accelerations(a, CFG.mission)
    dt = CFG.mission.dt_s
    qf = q[-1] + dt * v[-1] + 0.5 * dt**2 * a[-1]
    vf = v[-1] + dt * a[-1]
    assert np.allclose(qf, CFG.mission.q_final_m, atol=1e-8)
    assert np.allclose(vf, CFG.mission.v_final_mps, atol=1e-10)


def test_scenario_sample_size_reference_value():
    d = theory.scenario_dimension(4, 16)
    assert d == 1284
    n = theory.scenario_sample_size(d, 0.05, 1e-6)
    assert n == 29_142
    assert theory.scenario_violation_bound(n, d, 0.05) <= 1e-6 < theory.scenario_violation_bound(n - 1, d, 0.05)


def test_lemma3_closed_loop_never_worse():
    cfg = SimulationConfig(); cfg.physical.p_max_w = 2.0
    traj = trapezoidal_seed(cfg.mission, cfg.physical)
    means, covs = propagate_targets(cfg.mission, cfg.physical)
    m = cfg.physical.antennas
    rx = np.repeat((0.5 / m * np.eye(m))[None], cfg.mission.slots, axis=0).astype(complex)
    rows = theory.closed_vs_open_loop(traj.q, traj.v, rx, means, covs, cfg.mission, cfg.physical)
    assert all(r["closed_over_open"] <= 1 + 1e-9 for r in rows)
    assert min(r["closed_over_open"] for r in rows) < 0.999


def test_lemma2_inequalities_on_samples():
    q, v, xi, P = _geometry(1)
    rng = np.random.default_rng(0)
    a = upa_steering(q, np.array([xi[0], xi[1], 0.0]), PH.mx, PH.my, PH.wavelength_m)
    rx = 0.5 * np.outer(a, a.conj())
    from uav_isac.sensing import measurement_information
    xs = rng.multivariate_normal(xi, P, size=400)
    infos = np.array([measurement_information(q, v, x, rx, PH) for x in xs])
    psis = np.array([pcrb(q, v, x, P, rx, PH) for x in xs])
    eps = float(np.quantile(psis, 0.95))
    res = theory.bcrb_from_samples(P, infos, psis, eps, rho=0.05)
    assert res["lemma2_lower_ok"]
    assert res["bcrb_m2"] <= res["lemma2_rhs_m2"] + 1e-9


def test_prop4_lower_bound_terms():
    q, v, xi, P = _geometry(2)
    users = np.asarray(CFG.mission.users_m)
    plb = theory.rf_lower_bound_slot(q, users, [np.array([0.1, 0.4])], PH)
    assert plb >= 0.4


def test_prop5_joint_bound_dual_is_valid():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(5)
    q = np.array([300.0, 300.0, 80.0])
    steer = np.array([upa_steering(q, np.array([rng.uniform(0, 600), rng.uniform(0, 600), 0.0]),
                                   PH.mx, PH.my, PH.wavelength_m) for _ in range(6)])
    p = rng.uniform(0.2, 1.0, size=6)
    jb = theory.joint_illumination_bound(steer, p)
    assert jb.certified_w >= jb.per_point_max_w - 1e-12
    assert jb.certified_w <= jb.primal_w * (1 + 1e-4) + 1e-9
