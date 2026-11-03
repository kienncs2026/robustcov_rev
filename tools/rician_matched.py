"""R1.4: robust design with a CSI radius matched to the Rician factor K, audited under Rician K.

    python tools/rician_matched.py 15        # K in dB; radius = 95% quantile of ||h - h_LoS|| / ||h_LoS||

The design uses the fixed reference trajectory, the linear PCRB form and CLARABEL (SCS fallback), like the
other mismatch runs; only the CSI radius changes. Runs go to results/R9M/runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_isac.ao_solver import run_simulation  # noqa: E402
from uav_isac.config import load_config  # noqa: E402
from uav_isac.experiments import LINEAR, _fixed_geometry, with_overrides  # noqa: E402


def matched_radius(k_db: float, antennas: int, coverage: float = 0.95, samples: int = 400_000,
                   seed: int = 1) -> float:
    """Quantile of the relative Rician error, with the scattered model of audit._rician_outage."""
    k = 10.0 ** (k_db / 10.0)
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=(samples, antennas)) + 1j * rng.normal(size=(samples, antennas))) / np.sqrt(2.0)
    # |h_LoS,m| is equal for all elements, so the relative error does not depend on the geometry.
    e = (np.sqrt(k / (k + 1.0)) - 1.0) + np.sqrt(1.0 / (k + 1.0)) * z
    return float(np.quantile(np.linalg.norm(e, axis=1) / np.sqrt(antennas), coverage))


def main() -> None:
    k_db = float(sys.argv[1])
    base = load_config(ROOT / "configs" / "reference_strict.json")
    radius = matched_radius(k_db, base.physical.antennas)
    cfg = with_overrides(_fixed_geometry(base), {**LINEAR, "solver.solver": "CLARABEL", "solver.fallback_solver": "SCS",
                                                 "physical.csi_error_radius": round(radius, 4),
                                                 "evaluation.rician_k_db": k_db})
    cfg.scenario = "R9M"
    cfg.scheme.name = f"R9M_matched_k{k_db:g}"
    cfg.output_root = str(ROOT / "results" / "R9M" / "runs")
    print(f"K = {k_db:g} dB, matched CSI radius = {radius:.4f}", flush=True)
    res = run_simulation(cfg, make_plots=False)
    s = res.audit.summary if res.audit is not None else {}
    print(f"success={res.success} failure={res.failure_code} rf_j={s.get('rf_j')} "
          f"rician_max_outage={s.get('rician_max_outage')} rician_mean_outage={s.get('rician_mean_outage')} "
          f"ucb={s.get('max_pcrb_ucb_95')}", flush=True)


if __name__ == "__main__":
    main()
