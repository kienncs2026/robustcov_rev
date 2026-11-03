"""Experiment registry for the revision (reviewer-driven studies R1--R11).

Every experiment is a frozen list of configurations built from the locked
reference configuration. Nothing is selected after seeing results: changing a
value below changes the run fingerprint and therefore the run directory.

Legacy studies of the submission (S2, S3, S4, S5A, S5B, S9, S9R) are kept
unchanged in :mod:`uav_isac.legacy` and are referred to as ``LEGACY``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import SchemeConfig, SimulationConfig
from .legacy.postlock import _s5b_geometry, scheme_for_method
from .legacy.s9r import locally_perturbed_topology
from .topology import randomized_topology

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------
R1_LOCAL_SEEDS_START = 20_000          # local +/-25 m perturbations of T0
R1_UNRESTRICTED_SEEDS_START = 30_000   # unrestricted random topologies
R1_PERTURBATION_BOUND_M = 25.0
TIGHTENING_MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20)
SENSITIVITY_MULTISTART = (1, 3, 5, 8)
SENSITIVITY_SEGMENTS = (4, 6, 8, 12)
SENSITIVITY_MAXITER = (15, 35, 70)
S3_CORNER = {"scale": 4.0, "risk": 0.01, "radius": 3.6437211935036444}  # 8-m prior, 1 % risk


@dataclass
class PlannedRun:
    experiment: str
    label: str
    config: SimulationConfig
    tags: dict = field(default_factory=dict)


@dataclass
class Experiment:
    exp_id: str
    title: str
    reviewer_refs: str
    builder: Callable[[SimulationConfig, dict], list[PlannedRun]]
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def with_overrides(cfg: SimulationConfig, overrides: dict) -> SimulationConfig:
    out = deepcopy(cfg)
    for key, value in overrides.items():
        section, name = key.split(".", 1)
        target = getattr(out, section)
        if not hasattr(target, name):
            raise KeyError(f"unknown config field {key}")
        setattr(target, name, value)
    return out


def _proposed(base: SimulationConfig) -> SimulationConfig:
    cfg = deepcopy(base)
    cfg.scheme = scheme_for_method("proposed")
    return cfg


def _fixed_geometry(base: SimulationConfig) -> SimulationConfig:
    cfg = deepcopy(base)
    cfg.scheme = scheme_for_method("robust_fixed_geometry")
    return cfg


LINEAR = {"solver.pcrb_formulation": "linear"}
SCENARIO = {"solver.pcrb_formulation": "linear", "solver.pcrb_enforcement": "scenario_certified"}
REFINED = {"solver.pcrb_formulation": "linear", "solver.refinement_enabled": True}
REVISION_AO = {"solver.candidate_screening": True, "solver.joint_bound_check": True}


def _run(exp: str, label: str, cfg: SimulationConfig, **tags) -> PlannedRun:
    cfg = deepcopy(cfg)
    cfg.scenario = exp
    cfg.scheme.name = f"{exp}_{label}"
    return PlannedRun(exp, label, cfg, tags)


def local_topology(base: SimulationConfig, seed: int, bound_m: float = R1_PERTURBATION_BOUND_M) -> SimulationConfig:
    """Bounded perturbation of user/target XY (generalizes the S9R protocol)."""
    if seed in (11_000, 11_001):
        return locally_perturbed_topology(base, seed)
    cfg = deepcopy(base)
    rng = np.random.default_rng(seed)
    users = np.asarray(cfg.mission.users_m, float)
    targets = np.asarray(cfg.mission.targets_initial, float)
    users[:, :2] += rng.uniform(-bound_m, bound_m, size=(len(users), 2))
    targets[:, :2] += rng.uniform(-bound_m, bound_m, size=(len(targets), 2))
    cfg.mission.users_m = users.tolist()
    cfg.mission.targets_initial = targets.tolist()
    cfg.topology_seed = int(seed)
    return cfg


def s3_corner(base: SimulationConfig) -> SimulationConfig:
    cfg = deepcopy(base)
    s = S3_CORNER["scale"] ** 0.5
    cfg.physical.initial_position_std_m *= s
    cfg.physical.initial_velocity_std_mps *= s
    cfg.physical.process_acceleration_std_mps2 *= s
    cfg.physical.pcrb_violation = S3_CORNER["risk"]
    cfg.physical.confidence_radius = S3_CORNER["radius"]
    return cfg


def displaced_geometry(base: SimulationConfig) -> SimulationConfig:
    """Return-to-base y260 geometry of S5A/S5B with epsilon = 9 m^2."""
    cfg = _s5b_geometry(base)
    cfg.physical.pcrb_threshold_m2 = 9.0
    return cfg


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_r1(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    n_local = int(opts.get("n_local", 20))
    n_unres = int(opts.get("n_unrestricted", 10))
    methods = tuple(opts.get("r1_methods", ("nine_state", "refined")))
    runs: list[PlannedRun] = []
    topologies = [("local", R1_LOCAL_SEEDS_START + i) for i in range(n_local)]
    topologies += [("unrestricted", R1_UNRESTRICTED_SEEDS_START + i) for i in range(n_unres)]
    for kind, seed in topologies:
        topo = (local_topology(base, seed) if kind == "local"
                else randomized_topology(base, seed))
        for method in methods:
            ov = dict(REVISION_AO)
            if method == "scenario":
                ov.update(SCENARIO)
            elif method == "refined":
                ov.update(REFINED)
            ov["solver.phase1_enabled"] = True
            cfg = with_overrides(_proposed(topo), ov)
            runs.append(_run("R1", f"{kind}{seed}_{method}", cfg,
                             topology_kind=kind, topology_seed=seed, method=method))
    return runs


def build_r2(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    runs = []
    for geom_name, geom in (("reference", base), ("displaced", displaced_geometry(base))):
        prop = _proposed(geom)
        variants = {
            "propulsion_only": {"scheme.trajectory_objective": "propulsion_only"},
            "random_search": {"solver.trajectory_method": "random_search",
                              "solver.random_search_samples": 64},
            "powell": {"solver.trajectory_method": "Powell"},
        }
        for label, ov in variants.items():
            runs.append(_run("R2", f"{geom_name}_{label}", with_overrides(prop, ov),
                             geometry=geom_name, method=label))
    return runs


def build_r3(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    solvers = opts.get("solvers", ("SCS", "CLARABEL", "MOSEK"))
    runs = []
    fixed = _fixed_geometry(base)
    for solver in solvers:
        for form in ("schur", "linear"):
            cfg = with_overrides(fixed, {"solver.solver": solver, "solver.pcrb_formulation": form})
            runs.append(_run("R3", f"{solver.lower()}_{form}", cfg, solver=solver, formulation=form))
    return runs


def build_r4(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    runs = []
    for case_name, case in (("reference", base), ("s3corner", s3_corner(base))):
        fixed = _fixed_geometry(case)
        for mode in ("bank", "gaussian_randomization"):
            cfg = with_overrides(fixed, {"solver.rank_recovery_mode": mode, **LINEAR})
            runs.append(_run("R4", f"{case_name}_{mode}", cfg, case=case_name, rank_mode=mode))
    return runs


def build_r5(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    runs = []
    for frac in (0.1, 0.2, 0.3):
        cfg = _proposed(base)
        cfg.scheme.sensing_mode = "fixed_fraction"
        cfg.scheme.sensing_fraction = frac
        runs.append(_run("R5", f"robust_fixed_sensing{int(frac*100)}", cfg, sensing_fraction=frac))
    return runs


def build_r6(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    cases = (("reference", base), ("local1", local_topology(base, 11_000)),
             ("local2", local_topology(base, 11_001)), ("s3corner", s3_corner(base)))
    runs = []
    for case_name, case in cases:
        prop = _proposed(case)
        for label, ov in (("scenario", SCENARIO), ("refined", REFINED)):
            runs.append(_run("R6", f"{case_name}_{label}", with_overrides(prop, ov),
                             case=case_name, method=label))
    return runs


def build_r7(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    cases = (("reference", base), ("local1", local_topology(base, 11_000)),
             ("local2", local_topology(base, 11_001)))
    runs = []
    for case_name, case in cases:
        fixed = with_overrides(_fixed_geometry(case), LINEAR)
        for margin in TIGHTENING_MARGINS:
            cfg = with_overrides(fixed, {"physical.enforcement_margin": float(margin)})
            runs.append(_run("R7", f"{case_name}_margin{int(round(margin*100)):02d}", cfg,
                             case=case_name, margin=margin))
    return runs


def build_r8(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    geom = _proposed(displaced_geometry(base))
    ref = geom.solver
    runs = []
    for value in SENSITIVITY_MULTISTART:
        if value != ref.trajectory_multistart:
            runs.append(_run("R8", f"multistart{value}", with_overrides(geom, {"solver.trajectory_multistart": value}),
                             factor="multistart", value=value))
    for value in SENSITIVITY_SEGMENTS:
        if value != ref.trajectory_control_segments:
            runs.append(_run("R8", f"segments{value}", with_overrides(geom, {"solver.trajectory_control_segments": value}),
                             factor="segments", value=value))
    for value in SENSITIVITY_MAXITER:
        if value != ref.trajectory_powell_maxiter:
            runs.append(_run("R8", f"maxiter{value}", with_overrides(geom, {"solver.trajectory_powell_maxiter": value}),
                             factor="maxiter", value=value))
    return runs


def build_r11(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    """Proposition 5 certificates at epsilon = 4 m^2 for all rates (fast)."""
    runs = []
    for rate in (1.0, 2.0, 3.0, 4.0):
        cfg = with_overrides(_proposed(base), {"physical.min_spectral_efficiency": rate,
                                               "physical.pcrb_threshold_m2": 4.0,
                                               "solver.joint_bound_check": True})
        runs.append(_run("R11", f"rate{rate:g}_eps4", cfg, rate=rate, eps=4.0))
    cfg = with_overrides(_fixed_geometry(base), {"solver.joint_bound_check": True,
                                                 "solver.joint_bound_slots": "all", **LINEAR})
    runs.append(_run("R11", "reference_all_slots", cfg, rate=2.0, eps=9.0))
    return runs


def build_r12(base: SimulationConfig, opts: dict) -> list[PlannedRun]:
    """Propulsion-first design (Cor. 2): the AO starts from the propulsion-optimal path.

    Re-runs the configurations whose headline numbers depend on the trajectory:
    Table II rows at the reference geometry and the displaced-target studies.
    """
    pf = {"solver.seed_mode": "propulsion_optimal", **LINEAR, "solver.solver": "CLARABEL"}
    runs = []
    for method in ("proposed", "deterministic_joint", "trajectory_fixed_sensing"):
        cfg = deepcopy(base)
        cfg.scheme = scheme_for_method(method)
        runs.append(_run("R12", f"reference_{method}", with_overrides(cfg, pf), geometry="reference",
                         method=method))
    from .legacy.postlock import S5B_PCRB_THRESHOLDS_M2
    for eps in S5B_PCRB_THRESHOLDS_M2:
        cfg = with_overrides(_proposed(displaced_geometry(base)), {**pf, "physical.pcrb_threshold_m2": float(eps)})
        runs.append(_run("R12", f"displaced260_eps{eps:g}", cfg, geometry="displaced", offset=260.0, eps=eps))
    for offset in (180.0, 210.0, 240.0):
        cfg = _proposed(displaced_geometry(base))
        y = min(565.0, 300.0 + offset)
        cfg.mission.users_m = [[220.0, y - 20.0, 0.0], [380.0, y - 10.0, 0.0]]
        cfg.mission.targets_initial = [[300.0, y, 0.0, -2.0]]
        cfg.physical.min_spectral_efficiency = 1.5
        runs.append(_run("R12", f"displaced{int(offset)}_eps9", with_overrides(cfg, pf),
                         geometry="displaced", offset=offset, eps=9.0))
    return runs


EXPERIMENTS: dict[str, Experiment] = {
    "R1": Experiment("R1", "Random topologies (bounded local and unrestricted)", "R1.6, R1.9, Ed.5, R2.3",
                     build_r1, "Nine-state, scenario (Prop. 3) and refined (Alg. 2) designs per topology."),
    "R2": Experiment("R2", "Trajectory baselines and search strategies", "R1.7, R2.5, R3.3, Ed.3",
                     build_r2, "Proposed SLSQP and fixed trajectories come from LEGACY S9/S5B."),
    "R3": Experiment("R3", "Conic solver and PCRB formulation comparison", "R3.3, R2.6, Prop. 2",
                     build_r3, "MOSEK runs are skipped when MOSEK is not installed."),
    "R4": Experiment("R4", "Rank recovery: deterministic bank vs Gaussian randomization", "R3.3, R3.4",
                     build_r4),
    "R5": Experiment("R5", "Fixed sensing-power baseline with robust uncertainty treatment", "R3.3",
                     build_r5),
    "R6": Experiment("R6", "Reliability remedies: scenario certificate and refinement", "Ed.4, R2.3, R3.2",
                     build_r6),
    "R7": Experiment("R7", "Tightening sweep (Lemma 4); L_w is computed post hoc", "R1.3",
                     build_r7),
    "R8": Experiment("R8", "Trajectory-block sensitivity (starts, segments, iterations)", "R2.4, R1.10",
                     build_r8),
    "R12": Experiment("R12", "Propulsion-first design (AO seeded by the propulsion-optimal path)",
                      "Ed.3, R1.2, R2.5, R3.1", build_r12,
                      "Uses the linear PCRB form and CLARABEL; compare with LEGACY S9 T0, S5A, S5B."),
    "R11": Experiment("R11", "Joint illumination certificates (Prop. 5)", "Ed.3, R1.2, R3.1",
                      build_r11),
}


def build(exp_id: str, base: SimulationConfig, opts: dict | None = None) -> list[PlannedRun]:
    exp = EXPERIMENTS[exp_id.upper()]
    runs = exp.builder(base, opts or {})
    labels = [r.label for r in runs]
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"duplicate labels in {exp_id}")
    return runs
