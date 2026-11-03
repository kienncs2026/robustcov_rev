from __future__ import annotations

import math
import numpy as np
from scipy.stats import chi2, beta

from .config import MissionConfig, PhysicalConfig
from .geometry import C_LIGHT, upa_steering


def transition_matrix(dt: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def process_covariance(dt: float, acceleration_std: float) -> np.ndarray:
    i2 = np.eye(2)
    return acceleration_std**2 * np.block([
        [dt**3 / 3.0 * i2, dt**2 / 2.0 * i2],
        [dt**2 / 2.0 * i2, dt * i2],
    ])


def propagate_targets(mission: MissionConfig, physical: PhysicalConfig,
                      covariance_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n = mission.slots
    target0 = np.asarray(mission.targets_initial, dtype=float)
    l_count = target0.shape[0]
    f = transition_matrix(mission.dt_s)
    q = process_covariance(mission.dt_s, physical.process_acceleration_std_mps2)
    means = np.zeros((l_count, n, 4), dtype=float)
    covariances = np.zeros((l_count, n, 4, 4), dtype=float)
    p0 = np.diag([
        physical.initial_position_std_m**2,
        physical.initial_position_std_m**2,
        physical.initial_velocity_std_mps**2,
        physical.initial_velocity_std_mps**2,
    ]) * covariance_scale
    means[:, 0] = target0
    covariances[:, 0] = p0
    for slot in range(n - 1):
        means[:, slot + 1] = (f @ means[:, slot].T).T
        for l in range(l_count):
            covariances[l, slot + 1] = f @ covariances[l, slot] @ f.T + q * covariance_scale
    return means, covariances


def measurement_function(q: np.ndarray, v: np.ndarray, xi: np.ndarray, fc_hz: float) -> np.ndarray:
    p = np.array([xi[0], xi[1], 0.0])
    vt = np.array([xi[2], xi[3], 0.0])
    delta = p - q
    r = float(np.linalg.norm(delta))
    u = delta / r
    radial_rate = float(u @ (vt - v))
    return np.array([2.0 * r / C_LIGHT, -2.0 * fc_hz / C_LIGHT * radial_rate, u[0], u[1]])


def measurement_jacobian(q: np.ndarray, v: np.ndarray, xi: np.ndarray, fc_hz: float) -> np.ndarray:
    p = np.array([xi[0], xi[1], 0.0])
    vt = np.array([xi[2], xi[3], 0.0])
    delta = p - q
    r = float(np.linalg.norm(delta))
    u = delta / r
    projection = (np.eye(3) - np.outer(u, u)) / r
    rel_v = vt - v
    pv = projection @ rel_v
    g = np.zeros((4, 4), dtype=float)
    g[0] = 2.0 / C_LIGHT * np.array([u[0], u[1], 0.0, 0.0])
    g[1] = -2.0 * fc_hz / C_LIGHT * np.array([pv[0], pv[1], u[0], u[1]])
    g[2] = np.array([(1.0 - u[0] ** 2) / r, -u[0] * u[1] / r, 0.0, 0.0])
    g[3] = np.array([-u[0] * u[1] / r, (1.0 - u[1] ** 2) / r, 0.0, 0.0])
    return g


def measurement_calibration(physical: PhysicalConfig) -> np.ndarray:
    gamma_ref = physical.sensing_reference_snr_linear
    sigma_tau = 2.0 * physical.range_std_ref_m / C_LIGHT
    sigma_doppler = 2.0 * physical.fc_hz * physical.radial_velocity_std_ref_mps / C_LIGHT
    sigma_mu = physical.direction_cosine_std_ref
    return gamma_ref * np.diag([sigma_tau**2, sigma_doppler**2, sigma_mu**2, sigma_mu**2])


def sensing_snr(q: np.ndarray, xi: np.ndarray, rx: np.ndarray, physical: PhysicalConfig,
                range_override_m: float | None = None) -> float:
    p = np.array([xi[0], xi[1], 0.0])
    r = float(np.linalg.norm(p - q)) if range_override_m is None else float(range_override_m)
    a = upa_steering(q, p, physical.mx, physical.my, physical.wavelength_m)
    illumination = max(float(np.vdot(a, rx @ a).real), 0.0)
    return physical.sensing_reference_snr_linear * (physical.sensing_reference_range_m / r) ** 4 * (
        illumination / physical.sensing_reference_power_w
    )


def measurement_information(q: np.ndarray, v: np.ndarray, xi: np.ndarray, rx: np.ndarray,
                            physical: PhysicalConfig, range_override_m: float | None = None,
                            snr_scale: float = 1.0) -> np.ndarray:
    gamma = snr_scale * sensing_snr(q, xi, rx, physical, range_override_m=range_override_m)
    g = measurement_jacobian(q, v, xi, physical.fc_hz)
    d_inv = np.linalg.inv(measurement_calibration(physical))
    return gamma * g.T @ d_inv @ g


def pcrb(q: np.ndarray, v: np.ndarray, xi: np.ndarray, prior_covariance: np.ndarray,
         rx: np.ndarray, physical: PhysicalConfig, range_override_m: float | None = None,
         snr_scale: float = 1.0) -> float:
    j_prior = np.linalg.inv(prior_covariance)
    j = j_prior + measurement_information(q, v, xi, rx, physical, range_override_m=range_override_m, snr_scale=snr_scale)
    b = np.linalg.inv((j + j.T) / 2.0)
    return float(np.trace(b[:2, :2]))


def confidence_radius(probability_mass: float, dimension: int = 4) -> float:
    return float(np.sqrt(chi2.ppf(probability_mass, df=dimension)))


def confidence_points(mean: np.ndarray, covariance: np.ndarray, radius: float,
                      mode: str = "principal_axes", scenario_count: int = 100,
                      rng: np.random.Generator | None = None,
                      extra_boundary_points: int = 0) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    l = np.linalg.cholesky(covariance + 1e-12 * np.eye(4))
    if mode == "mean":
        return mean[None, :]
    if mode == "scenario":
        if rng is None:
            rng = np.random.default_rng(0)
        return rng.multivariate_normal(mean, covariance, size=scenario_count)
    points = [mean]
    for i in range(4):
        points.append(mean + radius * l[:, i])
        points.append(mean - radius * l[:, i])
    if extra_boundary_points > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        directions = rng.normal(size=(extra_boundary_points, 4))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points.extend(mean + radius * (directions @ l.T))
    return np.asarray(points)


def worst_case_range(q: np.ndarray, mean: np.ndarray, covariance: np.ndarray, radius: float,
                     samples: int = 256) -> float:
    """Approximate the maximum range over the 2-D position projection of the ellipsoid."""
    cov2 = covariance[:2, :2]
    values, vectors = np.linalg.eigh((cov2 + cov2.T) / 2.0)
    transform = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    xy = mean[:2] + radius * circle @ transform.T
    xyz = np.column_stack([xy, np.zeros(samples)])
    return float(np.max(np.linalg.norm(xyz - q[None, :], axis=1)))


def finite_difference_jacobian(q: np.ndarray, v: np.ndarray, xi: np.ndarray, fc_hz: float,
                               step: float = 1e-5) -> np.ndarray:
    result = np.zeros((4, 4), dtype=float)
    for i in range(4):
        scale = max(1.0, abs(xi[i]))
        h = step * scale
        xp = xi.copy(); xp[i] += h
        xm = xi.copy(); xm[i] -= h
        result[:, i] = (measurement_function(q, v, xp, fc_hz) - measurement_function(q, v, xm, fc_hz)) / (2.0 * h)
    return result


def binomial_upper_bound(violations: int, samples: int, confidence: float = 0.95) -> float:
    if samples <= 0:
        return float("nan")
    if violations >= samples:
        return 1.0
    return float(beta.ppf(confidence, violations + 1, samples - violations))
