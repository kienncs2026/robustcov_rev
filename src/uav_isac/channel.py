from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize

from .geometry import csi_shape, hermitian_sqrt, nominal_channel
from .config import PhysicalConfig


def sinr(channel: np.ndarray, w_covariances: np.ndarray, sensing_covariance: np.ndarray,
         user_index: int, noise_w: float) -> float:
    desired = float(np.vdot(channel, w_covariances[user_index] @ channel).real)
    interference = float(np.vdot(channel, sensing_covariance @ channel).real)
    for j in range(w_covariances.shape[0]):
        if j != user_index:
            interference += float(np.vdot(channel, w_covariances[j] @ channel).real)
    return desired / max(interference + noise_w, 1e-18)


def robust_quadratic_matrix(w_covariances: np.ndarray, sensing_covariance: np.ndarray,
                            user_index: int, gamma: float) -> np.ndarray:
    a = w_covariances[user_index].copy()
    for j in range(w_covariances.shape[0]):
        if j != user_index:
            a -= gamma * w_covariances[j]
    a -= gamma * sensing_covariance
    return (a + a.conj().T) / 2.0


def _complex_to_real_quadratic(bmat: np.ndarray, bvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    br = bmat.real
    bi = bmat.imag
    h = np.block([[br, -bi], [bi, br]])
    g = np.concatenate([bvec.real, bvec.imag])
    return (h + h.T) / 2.0, g


def trust_region_minimum(bmat: np.ndarray, bvec: np.ndarray, constant: float) -> tuple[float, np.ndarray]:
    """Minimize e^H B e + 2 Re{b^H e} + c over ||e||_2 <= 1.

    A secular-equation solution is used in the regular case. A constrained SLSQP
    fallback covers the rare hard case where the linear term is orthogonal to the
    minimum-eigenvalue eigenspace.
    """
    h, g = _complex_to_real_quadratic((bmat + bmat.conj().T) / 2.0, bvec)
    eigenvalues = np.linalg.eigvalsh(h)
    min_eig = float(eigenvalues[0])

    # Unconstrained minimizer is valid only for positive semidefinite H.
    if min_eig >= -1e-12:
        x_unc = -np.linalg.pinv(h, rcond=1e-12) @ g
        if np.linalg.norm(x_unc) <= 1.0 + 1e-10:
            value = float(x_unc @ h @ x_unc + 2.0 * g @ x_unc + constant)
            m = bvec.size
            return value, x_unc[:m] + 1j * x_unc[m:]

    lower = max(0.0, -min_eig) + 1e-10

    def solution(lam: float) -> np.ndarray:
        return -np.linalg.solve(h + lam * np.eye(h.shape[0]), g)

    def secular(lam: float) -> float:
        return float(np.linalg.norm(solution(lam)) - 1.0)

    try:
        f_lower = secular(lower)
        if f_lower >= 0.0:
            upper = max(1.0, lower * 2.0)
            while secular(upper) > 0.0 and upper < 1e12:
                upper *= 2.0
            lam = brentq(secular, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=200)
            x = solution(lam)
            value = float(x @ h @ x + 2.0 * g @ x + constant)
            m = bvec.size
            return value, x[:m] + 1j * x[m:]
    except (np.linalg.LinAlgError, ValueError):
        pass

    # Hard-case fallback.
    def objective(x: np.ndarray) -> float:
        return float(x @ h @ x + 2.0 * g @ x + constant)

    constraints = ({"type": "ineq", "fun": lambda x: 1.0 - float(x @ x)},)
    values, vectors = np.linalg.eigh(h)
    starts = [np.zeros(h.shape[0]), vectors[:, 0], -vectors[:, 0]]
    best_result = None
    for x0 in starts:
        result = minimize(objective, x0, method="SLSQP", constraints=constraints,
                          options={"maxiter": 400, "ftol": 1e-12, "disp": False})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    assert best_result is not None
    x = best_result.x
    m = bvec.size
    return float(best_result.fun), x[:m] + 1j * x[m:]


def robust_constraint_minimum(q: np.ndarray, user: np.ndarray, w_covariances: np.ndarray,
                              sensing_covariance: np.ndarray, user_index: int,
                              gamma: float, physical: PhysicalConfig) -> tuple[float, np.ndarray]:
    h_hat = nominal_channel(q, user, mx=physical.mx, my=physical.my,
                            wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                            pathloss_exponent=physical.pathloss_exponent)
    c_sqrt = hermitian_sqrt(csi_shape(h_hat, physical.csi_error_radius, physical.csi_model))
    a = robust_quadratic_matrix(w_covariances, sensing_covariance, user_index, gamma)
    bmat = c_sqrt.conj().T @ a @ c_sqrt
    bvec = c_sqrt.conj().T @ a @ h_hat
    constant = float(np.vdot(h_hat, a @ h_hat).real - gamma * physical.user_noise_w)
    return trust_region_minimum(bmat, bvec, constant)


def worst_case_sinr(q: np.ndarray, user: np.ndarray, w_covariances: np.ndarray,
                    sensing_covariance: np.ndarray, user_index: int,
                    physical: PhysicalConfig) -> tuple[float, np.ndarray]:
    """Compute infimum SINR by bisection over the trust-region feasibility test."""
    h_hat = nominal_channel(q, user, mx=physical.mx, my=physical.my,
                            wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                            pathloss_exponent=physical.pathloss_exponent)
    nominal = sinr(h_hat, w_covariances, sensing_covariance, user_index, physical.user_noise_w)
    lower = 0.0
    upper = max(nominal * 1.25 + 1.0, physical.gamma_min * 2.0)
    best_e = np.zeros(h_hat.size, dtype=np.complex128)
    for _ in range(50):
        midpoint = 0.5 * (lower + upper)
        minimum, e = robust_constraint_minimum(
            q, user, w_covariances, sensing_covariance, user_index, midpoint, physical
        )
        if minimum >= 0.0:
            lower = midpoint
            best_e = e
        else:
            upper = midpoint
    return lower, best_e


def worst_case_sinr_margin(q: np.ndarray, user: np.ndarray, w_covariances: np.ndarray,
                           sensing_covariance: np.ndarray, user_index: int,
                           physical: PhysicalConfig) -> tuple[float, float]:
    gamma_wc, _ = worst_case_sinr(q, user, w_covariances, sensing_covariance, user_index, physical)
    quadratic_minimum, _ = robust_constraint_minimum(
        q, user, w_covariances, sensing_covariance, user_index, physical.gamma_min, physical
    )
    return gamma_wc - physical.gamma_min, quadratic_minimum


def random_channel_outage(q: np.ndarray, user: np.ndarray, w_covariances: np.ndarray,
                          sensing_covariance: np.ndarray, user_index: int,
                          physical: PhysicalConfig, samples: int,
                          rng: np.random.Generator, *,
                          threshold_tolerance: float = 1e-6) -> float:
    h_hat = nominal_channel(q, user, mx=physical.mx, my=physical.my,
                            wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                            pathloss_exponent=physical.pathloss_exponent)
    c_sqrt = hermitian_sqrt(csi_shape(h_hat, physical.csi_error_radius, physical.csi_model))
    m = h_hat.size
    z = rng.normal(size=(samples, 2 * m))
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    radii = rng.random(samples) ** (1.0 / (2.0 * m))
    z *= radii[:, None]
    errors = z[:, :m] + 1j * z[:, m:]
    outages = 0
    for error in errors:
        h = h_hat + c_sqrt @ error
        value = sinr(h, w_covariances, sensing_covariance, user_index, physical.user_noise_w)
        outages += value < physical.gamma_min - threshold_tolerance
    return outages / samples
