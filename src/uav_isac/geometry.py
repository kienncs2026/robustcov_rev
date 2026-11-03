from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0


def direction_cosines(q: np.ndarray, p: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    delta = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    r = float(np.linalg.norm(delta))
    if r <= 0.0:
        raise ValueError("UAV and ground point cannot coincide in 3D.")
    u = delta / r
    return float(u[0]), float(u[1]), r, u


def upa_steering(q: np.ndarray, p: np.ndarray, mx: int, my: int, wavelength_m: float) -> np.ndarray:
    mux, muy, _, _ = direction_cosines(q, p)
    dx = wavelength_m / 2.0
    dy = wavelength_m / 2.0
    ax = np.exp(-1j * 2.0 * np.pi * dx / wavelength_m * np.arange(mx) * mux) / np.sqrt(mx)
    ay = np.exp(-1j * 2.0 * np.pi * dy / wavelength_m * np.arange(my) * muy) / np.sqrt(my)
    return np.kron(ax, ay).astype(np.complex128)


def nominal_channel(q: np.ndarray, user: np.ndarray, *, mx: int, my: int,
                    wavelength_m: float, beta0: float, pathloss_exponent: float) -> np.ndarray:
    a = upa_steering(q, user, mx, my, wavelength_m)
    distance = float(np.linalg.norm(np.asarray(q) - np.asarray(user)))
    m = mx * my
    gain = np.sqrt(m * beta0 * distance ** (-pathloss_exponent))
    return gain * a


def csi_shape(h_hat: np.ndarray, radius: float, model: str = "isotropic") -> np.ndarray:
    m = h_hat.size
    scale = radius**2 * float(np.vdot(h_hat, h_hat).real)
    if model == "isotropic":
        return scale * np.eye(m, dtype=np.complex128)
    if model == "angle_dominated":
        phase_direction = 1j * np.arange(m, dtype=float) * h_hat
        norm = np.linalg.norm(phase_direction)
        if norm < 1e-14:
            return scale * np.eye(m, dtype=np.complex128)
        u = phase_direction / norm
        return scale * (0.15 * np.eye(m) + 0.85 * np.outer(u, u.conj()))
    raise ValueError(f"Unknown CSI model: {model}")


def hermitian_sqrt(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2.0)
    values = np.maximum(values, floor)
    return (vectors * np.sqrt(values)) @ vectors.conj().T
