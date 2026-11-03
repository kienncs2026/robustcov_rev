"""Analytical results of the revised manuscript, implemented as reusable routines.

Each function corresponds to a numbered statement in the revision:

* Proposition 1  -> :func:`psi_spectrum`, :func:`gamma_star`, :func:`pmin_point`
* Proposition 2  -> :func:`pmin_point` (threshold of the exact linear constraint)
* Proposition 3  -> :func:`scenario_sample_size`, :func:`scenario_dimension`
* Proposition 4  -> :func:`rf_lower_bound_slot`, :func:`energy_lower_bound`
* Proposition 5  -> :func:`joint_illumination_bound`, :func:`joint_bound_report`
* Lemma 2        -> :func:`bcrb_from_samples`
* Lemma 3        -> :func:`closed_vs_open_loop`
* Lemma 4        -> :func:`nine_point_covering_radius`, :func:`whitened_lipschitz`

Only numpy/scipy are needed except for :func:`joint_illumination_bound`, which
solves one small SDP with CVXPY.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import binom

from .config import MissionConfig, PhysicalConfig
from .geometry import nominal_channel, upa_steering
from .sensing import (
    measurement_calibration,
    measurement_jacobian,
    pcrb,
    process_covariance,
    transition_matrix,
)

EP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Proposition 1 / 2: scalar PCRB and the illumination threshold P_min
# ---------------------------------------------------------------------------
def info_direction(q: np.ndarray, v: np.ndarray, xi: np.ndarray,
                   physical: PhysicalConfig) -> np.ndarray:
    """B = G^T D^{-1} G, the measurement information per unit echo SNR."""
    g = measurement_jacobian(q, v, xi, physical.fc_hz)
    d_inv = np.linalg.inv(measurement_calibration(physical))
    b = g.T @ d_inv @ g
    return (b + b.T) / 2.0


def psi_spectrum(q: np.ndarray, v: np.ndarray, xi: np.ndarray, prior_cov: np.ndarray,
                 physical: PhysicalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (c, lambda) such that psi(gamma) = sum_i c_i / (1 + gamma lambda_i), Eq. (25)."""
    b = info_direction(q, v, xi, physical)
    w, vec = np.linalg.eigh((prior_cov + prior_cov.T) / 2.0)
    p_half = vec @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ vec.T
    lam, u = np.linalg.eigh(p_half @ b @ p_half)
    lam = np.maximum(lam, 0.0)
    c = np.sum((EP @ p_half @ u) ** 2, axis=0)
    return c, lam


def psi_of_gamma(c: np.ndarray, lam: np.ndarray, gamma: float) -> float:
    return float(np.sum(c / (1.0 + gamma * lam)))


def gamma_star(c: np.ndarray, lam: np.ndarray, eps: float, tol: float = 1e-12) -> float:
    """Smallest echo SNR with psi(gamma) <= eps (0 if already met, inf if unreachable)."""
    if psi_of_gamma(c, lam, 0.0) <= eps:
        return 0.0
    if float(np.sum(c[lam <= 1e-14])) >= eps:
        return math.inf
    lo, hi = 0.0, 1.0
    while psi_of_gamma(c, lam, hi) > eps:
        hi *= 2.0
        if hi > 1e30:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if psi_of_gamma(c, lam, mid) > eps:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1.0):
            break
    return hi


def snr_to_illumination(gamma: float, range_m: float, physical: PhysicalConfig) -> float:
    """Invert Eq. (19): illumination a^H R a needed for echo SNR gamma at range r."""
    if not np.isfinite(gamma):
        return math.inf
    return (gamma * physical.sensing_reference_power_w / physical.sensing_reference_snr_linear
            * (range_m / physical.sensing_reference_range_m) ** 4)


def pmin_point(q: np.ndarray, v: np.ndarray, xi: np.ndarray, prior_cov: np.ndarray,
               physical: PhysicalConfig, eps: float,
               range_override_m: float | None = None) -> tuple[float, np.ndarray]:
    """P_min(xi) of Eq. (26) and the unit steering vector a(xi).

    The constraint a^H R_x a >= P_min is exactly equivalent to psi(xi) <= eps
    (Proposition 2). The range used in the echo SNR follows the enforcement
    convention: the conservative ellipsoid range when ``range_override_m`` is set.
    """
    p = np.array([xi[0], xi[1], 0.0])
    a = upa_steering(q, p, physical.mx, physical.my, physical.wavelength_m)
    r = float(np.linalg.norm(p - q)) if range_override_m is None else float(range_override_m)
    c, lam = psi_spectrum(q, v, xi, prior_cov, physical)
    g = gamma_star(c, lam, eps)
    return snr_to_illumination(g, r, physical), a


def pmin_for_points(q: np.ndarray, v: np.ndarray, points: np.ndarray, prior_cov: np.ndarray,
                    physical: PhysicalConfig, eps: float,
                    range_override_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vector of P_min values and the (P, M) steering matrix for an enforcement set."""
    values = np.empty(len(points))
    steer = np.empty((len(points), physical.antennas), dtype=np.complex128)
    for j, xi in enumerate(points):
        values[j], steer[j] = pmin_point(q, v, xi, prior_cov, physical, eps, range_override_m)
    return values, steer


def illumination_rows(steering: np.ndarray) -> np.ndarray:
    """Rows phi_j with a_j^H R a_j = phi_j @ vec(R, order='F') for Hermitian R."""
    m = steering.shape[1]
    # vec(R) column-major: entry (i,k) at index i + k*m; coefficient conj(a_i) a_k.
    return np.stack([np.outer(a.conj(), a).reshape(m * m, order="F") for a in steering])


# ---------------------------------------------------------------------------
# Proposition 3: scenario sample size
# ---------------------------------------------------------------------------
def scenario_dimension(k_users: int, antennas: int, support: str = "full") -> int:
    """Number of real decision variables d used in Eq. (39)."""
    if support == "full":
        return (k_users + 1) * antennas**2 + k_users
    if support == "rx":
        # Support-rank argument (requires citation check before use in the paper).
        return antennas**2
    raise ValueError(f"unknown support option {support!r}")


def scenario_violation_bound(n_samples: int, d: int, rho: float) -> float:
    """Right-hand side of Eq. (39): binomial tail sum_{i<d} C(N,i) rho^i (1-rho)^(N-i)."""
    return float(binom.cdf(d - 1, n_samples, rho))


def scenario_sample_size(d: int, rho: float, beta: float) -> int:
    """Smallest N_s with scenario_violation_bound(N_s, d, rho) <= beta."""
    lo, hi = d, max(d, 16)
    while scenario_violation_bound(hi, d, rho) > beta:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if scenario_violation_bound(mid, d, rho) <= beta:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# ---------------------------------------------------------------------------
# Proposition 4: RF lower bound and screening
# ---------------------------------------------------------------------------
def rf_lower_bound_slot(q: np.ndarray, users: np.ndarray, pmins: list[np.ndarray],
                        physical: PhysicalConfig) -> float:
    """P_lb[n] of Eq. (44)."""
    comm = 0.0
    for user in users:
        h = nominal_channel(q, user, mx=physical.mx, my=physical.my,
                            wavelength_m=physical.wavelength_m, beta0=physical.beta0,
                            pathloss_exponent=physical.pathloss_exponent)
        comm += physical.gamma_min * physical.user_noise_w / float(np.vdot(h, h).real)
    sens = max((float(np.max(p)) for p in pmins if len(p)), default=0.0)
    return max(comm, sens)


def energy_lower_bound(propulsion_j: float, plb: np.ndarray, mission: MissionConfig,
                       physical: PhysicalConfig) -> float:
    return float(propulsion_j + mission.dt_s / physical.pa_efficiency * float(np.sum(plb)))


# ---------------------------------------------------------------------------
# Proposition 5: joint illumination bound with dual certificate
# ---------------------------------------------------------------------------
@dataclass
class JointBound:
    primal_w: float          # P_J from the primal SDP (solver-dependent)
    certified_w: float       # sum mu_j p_j with rescaled dual (solver-independent)
    per_point_max_w: float   # max_j p_j (per-point necessary condition)
    active: list[int]        # indices with mu_j > 0
    status: str


def joint_illumination_bound(steering: np.ndarray, pmins: np.ndarray,
                             solver: str = "CLARABEL") -> JointBound:
    """Eq. (45): min tr(R) s.t. a_j^H R a_j >= p_j, R >= 0, plus a dual certificate."""
    import cvxpy as cp

    pmins = np.asarray(pmins, dtype=float)
    if np.any(~np.isfinite(pmins)):
        return JointBound(math.inf, math.inf, math.inf, [int(np.argmax(~np.isfinite(pmins)))],
                          "unreachable_threshold")
    keep = pmins > 0.0
    if not np.any(keep):
        return JointBound(0.0, 0.0, 0.0, [], "trivial")
    a = steering[keep]
    p = pmins[keep]
    m = a.shape[1]
    phi = illumination_rows(a)
    r = cp.Variable((m, m), hermitian=True)
    con = cp.real(phi @ cp.vec(r, order="F")) >= p
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(r))), [r >> 0, con])
    try:
        prob.solve(solver=solver)
    except Exception as exc:  # pragma: no cover
        return JointBound(math.nan, float(np.max(p)), float(np.max(p)), [], f"error:{type(exc).__name__}")
    mu = np.maximum(np.asarray(con.dual_value, dtype=float).reshape(-1), 0.0)
    gram = sum(mu[j] * np.outer(a[j], a[j].conj()) for j in range(len(mu)))
    lam_max = float(np.max(np.linalg.eigvalsh((gram + gram.conj().T) / 2.0))) if np.any(mu > 0) else 0.0
    certified = float(mu @ p / lam_max) if lam_max > 0 else float(np.max(p))
    certified = max(certified, float(np.max(p)))  # a single point is always a valid dual
    idx = np.flatnonzero(keep)[mu > 1e-6 * max(mu.max(), 1e-30)].tolist()
    return JointBound(float(prob.value) if prob.value is not None else math.nan,
                      certified, float(np.max(p)), idx, str(prob.status))


def joint_bound_report(q_all: np.ndarray, v_all: np.ndarray, target_means: np.ndarray,
                       target_covariances: np.ndarray, enforcement_builder, physical: PhysicalConfig,
                       slots: list[int], solver: str = "CLARABEL") -> list[dict]:
    """Evaluate Proposition 5 on the listed slots.

    ``enforcement_builder(slot)`` returns the frozen enforcement sets of that slot
    (the same sets used by the covariance problem), so the bound refers exactly to
    the surrogate that is solved.
    """
    rows = []
    for slot in slots:
        sets = enforcement_builder(slot)
        p_all, a_all = [], []
        for l, es in enumerate(sets):
            pm, st = pmin_for_points(q_all[slot], v_all[slot], es.points,
                                     target_covariances[l, slot], physical,
                                     es.threshold_m2, es.range_override_m)
            p_all.append(pm)
            a_all.append(st)
        bound = joint_illumination_bound(np.vstack(a_all), np.concatenate(p_all), solver=solver)
        rows.append({
            "slot": slot,
            "per_point_max_pmin_w": bound.per_point_max_w,
            "joint_bound_primal_w": bound.primal_w,
            "joint_bound_certified_w": bound.certified_w,
            "p_max_w": physical.p_max_w,
            "certified_ratio": bound.certified_w / physical.p_max_w,
            "certified_infeasible": bool(bound.certified_w > physical.p_max_w),
            "active_points": ";".join(str(i) for i in bound.active),
            "sdp_status": bound.status,
        })
    return rows


# ---------------------------------------------------------------------------
# Lemma 2: Bayesian PCRB versus the state-conditioned bound
# ---------------------------------------------------------------------------
def bcrb_from_samples(prior_cov: np.ndarray, info_samples: np.ndarray, psi_samples: np.ndarray,
                      eps: float, rho: float) -> dict[str, float]:
    """Check Eq. (24) on Monte Carlo draws.

    ``info_samples`` has shape (S, 4, 4) and holds J_m at the sampled states.
    """
    j_b = np.linalg.inv(prior_cov) + info_samples.mean(axis=0)
    psi_b = float(np.trace(EP @ np.linalg.inv((j_b + j_b.T) / 2.0) @ EP.T))
    psi_mean = float(np.mean(psi_samples))
    psi0 = float(np.trace(EP @ prior_cov @ EP.T))
    return {
        "bcrb_m2": psi_b,
        "mean_state_conditioned_pcrb_m2": psi_mean,
        "prior_position_variance_m2": psi0,
        "lemma2_rhs_m2": (1.0 - rho) * eps + rho * psi0,
        "lemma2_lower_ok": bool(psi_b <= psi_mean * (1.0 + 1e-9) + 1e-12),
    }


# ---------------------------------------------------------------------------
# Lemma 3: closed-loop versus open-loop PCRB along the predicted mean path
# ---------------------------------------------------------------------------
def closed_vs_open_loop(q_all: np.ndarray, v_all: np.ndarray, rx_all: np.ndarray,
                        target_means: np.ndarray, target_covariances: np.ndarray,
                        mission: MissionConfig, physical: PhysicalConfig) -> list[dict]:
    """Position PCRB at the mean state with open-loop prior vs full recursion."""
    f = transition_matrix(mission.dt_s)
    qn = process_covariance(mission.dt_s, physical.process_acceleration_std_mps2)
    rows = []
    for l in range(target_means.shape[0]):
        j_bar = None
        for n in range(mission.slots):
            mu = target_means[l, n]
            p_open = target_covariances[l, n]
            jm = _measurement_info(q_all[n], v_all[n], mu, rx_all[n], physical)
            if j_bar is None:
                prior_info = np.linalg.inv(p_open)
            else:
                prior_info = np.linalg.inv(f @ np.linalg.inv(j_bar) @ f.T + qn)
            j_bar = prior_info + jm
            psi_closed = float(np.trace(EP @ np.linalg.inv((j_bar + j_bar.T) / 2) @ EP.T))
            psi_open = pcrb(q_all[n], v_all[n], mu, p_open, rx_all[n], physical)
            rows.append({"target": l, "slot": n, "pcrb_open_loop_m2": psi_open,
                         "pcrb_closed_loop_m2": psi_closed,
                         "closed_over_open": psi_closed / max(psi_open, 1e-300)})
    return rows


def _measurement_info(q, v, xi, rx, physical):
    from .sensing import measurement_information
    return measurement_information(q, v, xi, rx, physical)


# ---------------------------------------------------------------------------
# Lemma 4: covering radius of the nine-state set and the Lipschitz margin
# ---------------------------------------------------------------------------
def nine_point_covering_radius(kappa: float, samples: int = 200_000, seed: int = 0) -> float:
    """Monte Carlo estimate of Eq. (35); the exact value is kappa."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(samples, 4))
    s *= kappa * rng.random(samples)[:, None] ** 0.25 / np.linalg.norm(s, axis=1, keepdims=True)
    pts = np.vstack([np.zeros(4)] + [sgn * kappa * np.eye(4)[i] for i in range(4) for sgn in (1, -1)])
    dist = np.min(np.linalg.norm(s[:, None, :] - pts[None, :, :], axis=2), axis=1)
    return float(dist.max())


def whitened_lipschitz(q: np.ndarray, v: np.ndarray, mean: np.ndarray, prior_cov: np.ndarray,
                       rx: np.ndarray, physical: PhysicalConfig, kappa: float,
                       samples: int, rng: np.random.Generator, step: float = 1e-3) -> float:
    """Largest gradient norm of psi(mu + L s) over samples of ||s|| <= kappa.

    Central differences in whitened coordinates; the true (sampled) range is used,
    as in the out-of-sample test. The result is an empirical estimate of L_w.
    """
    chol = np.linalg.cholesky(prior_cov + 1e-12 * np.eye(4))
    s = rng.normal(size=(samples, 4))
    s *= kappa * rng.random(samples)[:, None] ** 0.25 / np.linalg.norm(s, axis=1, keepdims=True)
    best = 0.0
    for sv in s:
        grad = np.empty(4)
        for i in range(4):
            e = np.zeros(4); e[i] = step
            fp = pcrb(q, v, mean + chol @ (sv + e), prior_cov, rx, physical)
            fm = pcrb(q, v, mean + chol @ (sv - e), prior_cov, rx, physical)
            grad[i] = (fp - fm) / (2.0 * step)
        best = max(best, float(np.linalg.norm(grad)))
    return best
