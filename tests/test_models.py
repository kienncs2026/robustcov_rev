import numpy as np
import pytest

from uav_isac.config import SimulationConfig
from uav_isac.dynamics import trapezoidal_seed, kinematic_residuals
from uav_isac.geometry import upa_steering
from uav_isac.sensing import finite_difference_jacobian, measurement_jacobian, propagate_targets


def test_upa_unit_norm():
    cfg = SimulationConfig()
    a = upa_steering(np.array([0.0, 0.0, 80.0]), np.array([100.0, 30.0, 0.0]),
                     cfg.physical.mx, cfg.physical.my, cfg.physical.wavelength_m)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-12)


def test_measurement_jacobian():
    cfg = SimulationConfig()
    q = np.array([250.0, 280.0, 80.0])
    v = np.array([15.0, 2.0, 0.0])
    xi = np.array([310.0, 420.0, 4.0, -1.5])
    analytic = measurement_jacobian(q, v, xi, cfg.physical.fc_hz)
    numeric = finite_difference_jacobian(q, v, xi, cfg.physical.fc_hz)
    scale = np.maximum(np.abs(numeric), 1e-8)
    assert np.max(np.abs(analytic - numeric) / scale) < 2e-4


def test_trajectory_terminal_and_dynamics():
    cfg = SimulationConfig()
    traj = trapezoidal_seed(cfg.mission, cfg.physical)
    residual = kinematic_residuals(traj, cfg.mission, cfg.physical)
    assert residual["dynamics"] < 1e-9
    assert residual["terminal_position"] < 1e-8
    assert residual["terminal_velocity"] < 1e-8
    assert residual["horizontal_speed"] >= -1e-8
    assert residual["acceleration"] >= -1e-8


def test_target_covariance_positive_definite():
    cfg = SimulationConfig()
    _, covariances = propagate_targets(cfg.mission, cfg.physical)
    assert np.min(np.linalg.eigvalsh(covariances)) > 0.0


def test_s7_expanded_topologies_are_distinct():
    from uav_isac.legacy.scenarios import sweep_configs
    cfgs = sweep_configs(SimulationConfig(), "S7")
    for cfg in cfgs:
        if cfg.scheme.name.startswith("proposed_K"):
            xy = np.asarray(cfg.mission.users_m)[:, :2]
            for i in range(len(xy)):
                for j in range(i + 1, len(xy)):
                    assert np.linalg.norm(xy[i] - xy[j]) > 1.0
        if cfg.scheme.name.startswith("proposed_L"):
            xy = np.asarray(cfg.mission.targets_initial)[:, :2]
            for i in range(len(xy)):
                for j in range(i + 1, len(xy)):
                    assert np.linalg.norm(xy[i] - xy[j]) > 1.0


def test_perfect_target_realization_follows_cv_process():
    pytest.importorskip("cvxpy")
    from uav_isac.ao_solver import _perfect_target_realization
    from uav_isac.sensing import propagate_targets, transition_matrix

    cfg = SimulationConfig()
    cfg.physical.process_acceleration_std_mps2 = 0.0  # zero process noise -> exact CV
    means, covariances = propagate_targets(cfg.mission, cfg.physical)
    rng = np.random.default_rng(7)
    states = _perfect_target_realization(cfg.mission, cfg.physical, means, covariances, rng)
    f = transition_matrix(cfg.mission.dt_s)
    for l in range(states.shape[0]):
        for n in range(states.shape[1] - 1):
            assert np.allclose(states[l, n + 1], f @ states[l, n], atol=1e-9)
