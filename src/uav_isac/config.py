from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
import warnings
from pathlib import Path
from typing import Any


@dataclass
class PhysicalConfig:
    fc_hz: float = 30e9
    bandwidth_hz: float = 20e6
    mx: int = 4
    my: int = 4
    pathloss_exponent: float = 2.2
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    p_max_w: float = 1.0
    pa_efficiency: float = 0.35
    min_spectral_efficiency: float = 2.0
    csi_error_radius: float = 0.05
    csi_model: str = "isotropic"
    qz_min_m: float = 50.0
    qz_max_m: float = 130.0
    vh_max_mps: float = 30.0
    vz_max_mps: float = 5.0
    a_max_mps2: float = 5.0
    p0_w: float = 580.65
    pi_w: float = 790.67
    u_tip_mps: float = 200.0
    v0_mps: float = 7.2
    d0: float = 0.3
    air_density: float = 1.225
    rotor_solidity: float = 0.05
    rotor_area_m2: float = 0.79
    gravity_mps2: float = 9.81
    vertical_loss_coeff: float = 20.0
    pcrb_threshold_m2: float = 9.0
    pcrb_violation: float = 0.05
    confidence_radius: float = 3.08
    enforcement_margin: float = 0.10
    sensing_reference_range_m: float = 150.0
    sensing_reference_power_w: float = 0.25
    sensing_reference_snr_db: float = 10.0
    range_std_ref_m: float = 0.5
    radial_velocity_std_ref_mps: float = 0.2
    direction_cosine_std_ref: float = 8.7e-3
    process_acceleration_std_mps2: float = 0.5
    initial_position_std_m: float = 8.0
    initial_velocity_std_mps: float = 1.0

    @property
    def antennas(self) -> int:
        return self.mx * self.my

    @property
    def wavelength_m(self) -> float:
        return 299_792_458.0 / self.fc_hz

    @property
    def beta0(self) -> float:
        return (self.wavelength_m / (4.0 * 3.141592653589793)) ** 2

    @property
    def user_noise_w(self) -> float:
        dbm = self.noise_psd_dbm_hz + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return 10.0 ** ((dbm - 30.0) / 10.0)

    @property
    def gamma_min(self) -> float:
        return 2.0 ** self.min_spectral_efficiency - 1.0

    @property
    def mass_kg(self) -> float:
        weight = 2.0 * self.air_density * self.rotor_area_m2 * self.v0_mps**2
        return weight / self.gravity_mps2

    @property
    def sensing_reference_snr_linear(self) -> float:
        return 10.0 ** (self.sensing_reference_snr_db / 10.0)


@dataclass
class MissionConfig:
    duration_s: float = 24.0
    slots: int = 48
    q_initial_m: list[float] = field(default_factory=lambda: [50.0, 300.0, 80.0])
    q_final_m: list[float] = field(default_factory=lambda: [550.0, 300.0, 80.0])
    v_initial_mps: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    v_final_mps: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    users_m: list[list[float]] = field(default_factory=lambda: [
        [120.0, 100.0, 0.0],
        [480.0, 100.0, 0.0],
        [100.0, 500.0, 0.0],
        [500.0, 500.0, 0.0],
    ])
    targets_initial: list[list[float]] = field(default_factory=lambda: [
        [180.0, 420.0, 6.0, -3.0],
        [450.0, 180.0, -5.0, 4.0],
    ])

    @property
    def dt_s(self) -> float:
        return self.duration_s / self.slots


@dataclass
class SolverConfig:
    solver: str = "SCS"
    solver_eps: float = 1e-5
    solver_max_iters: int = 40_000
    verbose: bool = False
    outer_iterations: int = 4
    outer_tolerance: float = 1e-3
    trajectory_updates: bool = True
    trajectory_control_segments: int = 6
    trajectory_powell_maxiter: int = 35
    trajectory_method: str = "SLSQP"
    trajectory_multistart: int = 3
    trajectory_probe_scale: float = 0.25
    trajectory_trust_accel_mps2: float = 0.8
    trajectory_penalty: float = 5e5
    trajectory_smooth_weight: float = 0.1
    trajectory_backtracking_factors: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125, 0.0625])
    trajectory_feasibility_tolerance: float = 1e-6
    energy_acceptance_tolerance_j: float = 1e-6
    rank_refinement: bool = True
    rank_targets: list[float] = field(default_factory=lambda: [0.90, 0.99, 0.999, 0.9999, 0.99999])
    rank_tolerance: float = 1e-4
    rank_one_recovery: bool = True
    covariance_parallel_workers: int = 1
    pcrb_enforcement: str = "principal_axes"
    scenario_samples: int = 100
    extra_boundary_points: int = 0
    out_of_sample_count: int = 10_000
    channel_audit_samples: int = 2_000
    random_seed: int = 2026
    # Post-solve feasibility gate: the PCRB constraints are re-evaluated at the
    # enforced points from the numerical solution; if the violation exceeds this
    # tolerance (m^2) the slot is re-solved with the fallback interior-point solver.
    pcrb_residual_tol_m2: float = 1e-3
    fallback_solver: str = "CLARABEL"
    # Numerical covariance gates. A solver status is insufficient: every final
    # covariance must also satisfy the Hermitian PSD cone within this tolerance.
    psd_tolerance: float = 1e-6
    fallback_on_inaccurate: bool = True
    max_final_inaccurate_fraction: float = 0.25
    # CP5b trajectory-activation thresholds.
    trajectory_activation_min_displacement_m: float = 1.0
    trajectory_activation_min_energy_gain_j: float = 1.0
    # Phase-F0 feasibility restoration before the first covariance SDP.
    feasibility_restoration_enabled: bool = True
    feasibility_restoration_segments: int = 3
    feasibility_restoration_maxiter: int = 15
    feasibility_restoration_multistart: int = 3
    feasibility_restoration_ratio_tolerance: float = 1e-3
    feasibility_restoration_smooth_weight: float = 1e-3
    feasibility_restoration_attraction_strengths: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    # Reproducible power-loss recovery. State is written after every accepted AO state.
    resume_enabled: bool = False
    skip_completed: bool = True
    rerun_failed: bool = False
    # ----- Revision extensions (defaults reproduce the locked submission) -----
    # PCRB constraint form: "schur" (Eq. 36, submission) or "linear" (Prop. 2).
    pcrb_formulation: str = "schur"
    # pcrb_enforcement also accepts "scenario_certified" (Prop. 3): N_s is
    # computed from (d, rho, scenario_beta) and the linear form is required.
    scenario_beta: float = 1e-6
    scenario_support: str = "full"          # "full" or "rx" (support-rank, cite first)
    scenario_max_samples: int = 60_000      # safety cap per target and slot
    # Rank-one recovery bank: "bank" (submission) or "gaussian_randomization".
    rank_recovery_mode: str = "bank"
    gaussian_randomization_trials: int = 20
    # Stopping rule of the AO loop. "accepted_gain" (revision): stop when the last
    # accepted trajectory update lowered energy by <= outer_tolerance (relative).
    # "legacy": compare two re-solves of the same trajectory (submission code).
    stopping_rule: str = "accepted_gain"
    # Proposition 5 certificate on slots fixed by the boundary conditions.
    joint_bound_check: bool = False
    joint_bound_slots: str = "first"        # "first" or "all" (report only)
    # Proposition 4 screening of the trajectory candidate before its re-solve.
    candidate_screening: bool = False
    # Slack-based Phase I when the covariance problem fails on the current path.
    phase1_enabled: bool = False
    phase1_iterations: int = 3
    # Algorithm 2 (validation-driven refinement) after the AO loop.
    refinement_enabled: bool = False
    refinement_rounds: int = 5
    refinement_validation_samples: int = 2_000
    refinement_points_per_round: int = 4
    refinement_only_if_failed: bool = True
    # Trajectory search alternatives: trajectory_method in {"SLSQP", "Powell",
    # "random_search"}; random_search evaluates this many perturbations.
    random_search_samples: int = 64
    # Propulsion-only baseline trajectory (Scheme.trajectory_objective).
    propulsion_only_segments: int = 6
    propulsion_only_maxiter: int = 200
    propulsion_only_multistart: int = 30
    propulsion_only_method: str = "powell"     # "powell" (revision) or "slsqp"
    # "default": straight/hover seed (submission). "propulsion_optimal": the AO starts
    # from the propulsion-only trajectory (propulsion-first design, Cor. 2).
    seed_mode: str = "default"


@dataclass
class SchemeConfig:
    name: str = "proposed"
    robust_communication: bool = True
    chance_sensing: bool = True
    perfect_information: bool = False
    fixed_trajectory: bool = False
    fixed_altitude_m: float | None = None
    sensing_mode: str = "optimized"  # optimized, zero, fixed_fraction
    sensing_fraction: float = 0.2
    # "joint" (proposed AO) or "propulsion_only" (trajectory minimizing propulsion
    # energy under the kinematic limits only, then a single covariance solve).
    trajectory_objective: str = "joint"


@dataclass
class EvaluationConfig:
    actual_pathloss_exponent: float | None = None
    actual_process_acceleration_std_mps2: float | None = None
    rcs_log_std_db: float = 0.0
    # Model-mismatch evaluation (R1.4). None keeps the design model.
    rician_k_db: float | None = None          # Rician channel instead of the ellipsoid
    target_turn_rate_dps: float | None = None  # coordinated-turn truth vs CV predictor
    mismatch_channel_samples: int = 2_000
    # Extra diagnostics written by the audit (cheap).
    lemma_diagnostics: bool = True


@dataclass
class SimulationConfig:
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    scheme: SchemeConfig = field(default_factory=SchemeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_root: str = "outputs"
    scenario: str = "S0"
    topology_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _merge_dataclass(cls: type, values: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        warnings.warn(f"Ignoring unknown {cls.__name__} keys in config file: {unknown}",
                      stacklevel=2)
    return cls(**{k: v for k, v in values.items() if k in allowed})


def load_config(path: str | Path | None = None) -> SimulationConfig:
    if path is None:
        return SimulationConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimulationConfig(
        physical=_merge_dataclass(PhysicalConfig, data.get("physical", {})),
        mission=_merge_dataclass(MissionConfig, data.get("mission", {})),
        solver=_merge_dataclass(SolverConfig, data.get("solver", {})),
        scheme=_merge_dataclass(SchemeConfig, data.get("scheme", {})),
        evaluation=_merge_dataclass(EvaluationConfig, data.get("evaluation", {})),
        output_root=data.get("output_root", "outputs"),
        scenario=data.get("scenario", "S0"),
        topology_seed=data.get("topology_seed", 0),
    )
