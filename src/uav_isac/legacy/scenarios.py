from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import time

import pandas as pd

from ..config import SimulationConfig, SchemeConfig
from .cp6_quality import classify_run
from ..topology import randomized_topology

import numpy as np


def _expand_users(users: list[list[float]], count: int, seed: int,
                  min_separation_m: float = 30.0) -> list[list[float]]:
    """Truncate or extend the user list to ``count`` distinct ground positions."""
    if count <= len(users):
        return [list(u) for u in users[:count]]
    rng = np.random.default_rng(seed)
    result = [list(u) for u in users]
    attempts = 0
    while len(result) < count and attempts < 10_000:
        attempts += 1
        candidate = rng.uniform(50.0, 550.0, size=2)
        if all(np.linalg.norm(candidate - np.asarray(u[:2])) >= min_separation_m for u in result):
            result.append([float(candidate[0]), float(candidate[1]), 0.0])
    if len(result) != count:
        raise RuntimeError("Could not place additional distinct users for the S7 sweep.")
    return result


def _expand_targets(targets: list[list[float]], count: int, duration_s: float, seed: int,
                    min_separation_m: float = 50.0) -> list[list[float]]:
    """Truncate or extend the target list to ``count`` distinct, in-region CV tracks."""
    if count <= len(targets):
        return [list(t) for t in targets[:count]]
    rng = np.random.default_rng(seed)
    result = [list(t) for t in targets]
    attempts = 0
    while len(result) < count and attempts < 10_000:
        attempts += 1
        position = rng.uniform(150.0, 450.0, size=2)
        speed = rng.uniform(2.0, 8.0)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        velocity = speed * np.array([np.cos(heading), np.sin(heading)])
        final = position + duration_s * velocity
        if np.any(final < 50.0) or np.any(final > 550.0):
            continue
        if any(np.linalg.norm(position - np.asarray(t[:2])) < min_separation_m for t in result):
            continue
        result.append([float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1])])
    if len(result) != count:
        raise RuntimeError("Could not place additional distinct targets for the S7 sweep.")
    return result


def scheme_by_name(name: str) -> SchemeConfig:
    schemes = {
        "proposed": SchemeConfig(name="proposed"),
        "perfect": SchemeConfig(name="perfect_information", robust_communication=False,
                                chance_sensing=False, perfect_information=True),
        "nominal": SchemeConfig(name="nominal", robust_communication=False, chance_sensing=False),
        "robust_comm": SchemeConfig(name="robust_communication_only", robust_communication=True, chance_sensing=False),
        "chance_sensing": SchemeConfig(name="chance_sensing_only", robust_communication=False, chance_sensing=True),
        "straight": SchemeConfig(name="fixed_straight", fixed_trajectory=True),
        "fixed_altitude": SchemeConfig(name="fixed_altitude", fixed_altitude_m=80.0),
        "comm_only": SchemeConfig(name="communication_waveform_only", sensing_mode="zero"),
        "fixed_sensing_10": SchemeConfig(name="fixed_sensing_10", sensing_mode="fixed_fraction", sensing_fraction=0.1),
        "fixed_sensing_20": SchemeConfig(name="fixed_sensing_20", sensing_mode="fixed_fraction", sensing_fraction=0.2),
        "fixed_sensing_30": SchemeConfig(name="fixed_sensing_30", sensing_mode="fixed_fraction", sensing_fraction=0.3),
    }
    if name not in schemes:
        raise ValueError(f"Unknown scheme {name}. Available: {sorted(schemes)}")
    return schemes[name]


def s0_config(base: SimulationConfig) -> SimulationConfig:
    cfg = deepcopy(base)
    cfg.scenario = "S0"
    cfg.physical.mx = 2
    cfg.physical.my = 2
    cfg.physical.pcrb_threshold_m2 = 36.0
    cfg.mission.duration_s = 24.0
    cfg.mission.slots = 8
    cfg.mission.q_initial_m = [100.0, 300.0, 80.0]
    cfg.mission.q_final_m = [500.0, 300.0, 80.0]
    cfg.mission.users_m = [[300.0, 160.0, 0.0]]
    cfg.mission.targets_initial = [[300.0, 300.0, 0.0, 0.0]]
    cfg.physical.initial_position_std_m = 4.0
    cfg.physical.initial_velocity_std_mps = 0.5
    cfg.physical.process_acceleration_std_mps2 = 0.2
    cfg.solver.outer_iterations = 2
    cfg.solver.trajectory_control_segments = 3
    cfg.solver.trajectory_powell_maxiter = 10
    cfg.solver.out_of_sample_count = 1_000
    cfg.solver.channel_audit_samples = 200
    return cfg


def scenario_config(base: SimulationConfig, scenario: str) -> SimulationConfig:
    cfg = s0_config(base) if scenario.upper() == "S0" else deepcopy(base)
    cfg.scenario = scenario.upper()
    return cfg


def sweep_configs(base: SimulationConfig, scenario: str, quick: bool = False) -> list[SimulationConfig]:
    scenario = scenario.upper()
    configs: list[SimulationConfig] = []
    if scenario == "S0":
        return [s0_config(base)]
    if scenario == "S1":
        names = ["proposed", "perfect", "nominal", "robust_comm", "chance_sensing", "straight", "fixed_altitude", "comm_only"]
        if quick:
            names = ["proposed", "nominal", "straight"]
        for name in names:
            cfg = deepcopy(base); cfg.scenario = scenario; cfg.scheme = scheme_by_name(name); configs.append(cfg)
        return configs
    if scenario == "S2":
        radii = [0.0, 0.02, 0.05, 0.08, 0.10]
        if quick: radii = [0.0, 0.05, 0.10]
        for model in (["isotropic"] if quick else ["isotropic", "angle_dominated"]):
            for radius in radii:
                for name in ["proposed", "nominal"]:
                    cfg = deepcopy(base); cfg.scenario = scenario; cfg.scheme = scheme_by_name(name)
                    cfg.physical.csi_error_radius = radius; cfg.physical.csi_model = model
                    configs.append(cfg)
        return configs
    if scenario == "S3":
        # CP6-v2 isolates uncertainty/risk sensitivity at the validated PCRB
        # operating point. PCRB-threshold sensitivity belongs to S4; sweeping it
        # again here created a redundant 5 x 4 x 4 = 80-run Cartesian product.
        scales = [0.25, 0.5, 1.0, 2.0, 4.0]
        risks = [0.01, 0.05, 0.10, 0.20]
        if quick:
            scales, risks = [0.5, 1.0, 2.0], [0.05, 0.10]
        for scale in scales:
            for risk in risks:
                cfg = deepcopy(base); cfg.scenario = scenario
                cfg.physical.initial_position_std_m *= scale**0.5
                cfg.physical.initial_velocity_std_mps *= scale**0.5
                cfg.physical.process_acceleration_std_mps2 *= scale**0.5
                cfg.physical.pcrb_violation = risk
                # Protocol constants are frozen to avoid machine-precision changes in
                # scipy chi-square quantiles changing the run fingerprint across hosts.
                # Values correspond to the CP6-v2 paper-grade protocol.
                risk_radius = {
                    0.01: 3.6437211935036444,
                    0.05: 3.080215745168048,
                    0.10: 2.789164810428896,
                    0.20: 2.4471650320328306,
                }
                cfg.physical.confidence_radius = risk_radius[risk]
                # Keep base.physical.pcrb_threshold_m2 unchanged. S4 is the
                # dedicated rate--PCRB threshold tradeoff sweep.
                configs.append(cfg)
        return configs
    if scenario == "S4":
        rates = [1.0, 2.0, 3.0, 4.0]
        eps = [4.0, 9.0, 16.0, 25.0]
        if quick: rates, eps = [1.0, 2.0, 3.0], [9.0, 16.0]
        for rate in rates:
            for threshold in eps:
                cfg = deepcopy(base); cfg.scenario = scenario
                cfg.physical.min_spectral_efficiency = rate
                cfg.physical.pcrb_threshold_m2 = threshold
                configs.append(cfg)
        return configs
    if scenario == "S5A":
        # CP6-v3 trajectory-activation experiment inherited from CP5b.  The
        # return-to-base mission makes fixed hover a valid baseline, while the
        # proposed controller can use a closed excursion toward an asymmetric
        # northern sensing/communication cluster.  S5 (altitude sensitivity) is
        # intentionally left unchanged.
        # Final manuscript lock uses four asymmetric offsets. The 180 m and
        # 240 m pairs are backward-compatible with CP6-v3; 210 m and 260 m add
        # two points to establish a trajectory-activation/energy-gain trend.
        offsets = [180.0, 210.0, 240.0, 260.0]
        if quick:
            offsets = [180.0, 240.0]
        for offset in offsets:
            for fixed in [False, True]:
                cfg = deepcopy(base); cfg.scenario = scenario
                cfg.mission.duration_s = 24.0
                cfg.mission.slots = 48
                cfg.mission.q_initial_m = [300.0, 300.0, 80.0]
                cfg.mission.q_final_m = [300.0, 300.0, 80.0]
                cfg.mission.v_initial_mps = [0.0, 0.0, 0.0]
                cfg.mission.v_final_mps = [0.0, 0.0, 0.0]
                y_cluster = min(565.0, 300.0 + offset)
                cfg.mission.users_m = [
                    [220.0, y_cluster - 20.0, 0.0],
                    [380.0, y_cluster - 10.0, 0.0],
                ]
                cfg.mission.targets_initial = [
                    [300.0, y_cluster, 0.0, -2.0],
                ]
                cfg.physical.min_spectral_efficiency = 1.5
                cfg.physical.pcrb_threshold_m2 = 9.0
                cfg.physical.pcrb_violation = 0.05
                cfg.solver.outer_iterations = max(cfg.solver.outer_iterations, 5)
                cfg.solver.trajectory_control_segments = 6
                cfg.solver.trajectory_powell_maxiter = max(cfg.solver.trajectory_powell_maxiter, 50)
                cfg.solver.trajectory_trust_accel_mps2 = max(cfg.solver.trajectory_trust_accel_mps2, 1.0)
                cfg.solver.trajectory_multistart = max(cfg.solver.trajectory_multistart, 4)
                cfg.solver.trajectory_probe_scale = max(cfg.solver.trajectory_probe_scale, 0.35)
                if fixed:
                    cfg.scheme = scheme_by_name("straight")
                    cfg.scheme.name = f"fixed_hover_s5a_y{int(offset)}"
                else:
                    cfg.scheme = scheme_by_name("proposed")
                    cfg.scheme.name = f"proposed_s5a_y{int(offset)}"
                configs.append(cfg)
        return configs
    if scenario == "S5":
        kz_values = [0.0, 10.0, 20.0, 40.0]
        altitude_max = [100.0, 130.0, 160.0]
        if quick: kz_values, altitude_max = [0.0, 20.0], [100.0, 130.0]
        for kz in kz_values:
            for qmax in altitude_max:
                for fixed in [False, True]:
                    cfg = deepcopy(base); cfg.scenario = scenario
                    cfg.physical.vertical_loss_coeff = kz; cfg.physical.qz_max_m = qmax
                    cfg.scheme = scheme_by_name("fixed_altitude" if fixed else "proposed")
                    configs.append(cfg)
        return configs
    if scenario == "S6":
        names = ["proposed", "comm_only", "fixed_sensing_10", "fixed_sensing_20", "fixed_sensing_30"]
        if quick: names = ["proposed", "comm_only", "fixed_sensing_20"]
        for name in names:
            cfg = deepcopy(base); cfg.scenario = scenario; cfg.scheme = scheme_by_name(name); configs.append(cfg)
        return configs
    if scenario == "S7":
        # One-factor-at-a-time configurations. When K or L exceeds the base topology,
        # additional users/targets are placed at distinct deterministic positions;
        # tiling the base list would duplicate positions exactly, which makes the
        # per-user SINR constraints jointly infeasible for coincident channels
        # (identical h with gamma_min > 1) and silently reduces the effective L.
        values = [("K", x) for x in [2, 4, 6]] + [("L", x) for x in [1, 2, 3]] + \
                 [("M", x) for x in [9, 16, 25]] + [("N", x) for x in [24, 48, 72]]
        if quick: values = [("K", 2), ("K", 4), ("L", 1), ("M", 9), ("N", 24)]
        for factor, value in values:
            cfg = deepcopy(base); cfg.scenario = scenario
            if factor == "K":
                cfg.mission.users_m = _expand_users(cfg.mission.users_m, value, seed=20_260_707)
            elif factor == "L":
                cfg.mission.targets_initial = _expand_targets(
                    cfg.mission.targets_initial, value, cfg.mission.duration_s, seed=20_260_708
                )
            elif factor == "M":
                side = int(round(value**0.5)); cfg.physical.mx = side; cfg.physical.my = side
            else:
                cfg.mission.slots = value
            cfg.scheme.name = f"proposed_{factor}{value}"
            configs.append(cfg)
        return configs
    if scenario == "S8":
        alphas = [2.0, 2.2, 2.5]
        accels = [0.0, 0.5, 1.0]
        rcs_values = [0.0, 3.0]
        if quick: alphas, accels, rcs_values = [2.0, 2.5], [0.0, 1.0], [0.0, 3.0]
        for alpha in alphas:
            for accel in accels:
                for rcs_db in rcs_values:
                    cfg = deepcopy(base); cfg.scenario = scenario
                    cfg.evaluation.actual_pathloss_exponent = alpha
                    cfg.evaluation.actual_process_acceleration_std_mps2 = accel
                    cfg.evaluation.rcs_log_std_db = rcs_db
                    cfg.scheme.name = f"proposed_evalalpha{alpha}_acc{accel}_rcs{rcs_db}"
                    configs.append(cfg)
        return configs
    raise ValueError(f"Unsupported scenario: {scenario}")


def run_sweep(base: SimulationConfig, scenario: str, *, quick: bool = False,
              topologies: int = 1, make_plots: bool = False,
              warm_start_adjacent: bool = False,
              index_filter: set[int] | None = None) -> Path:
    from ..ao_solver import run_simulation
    from ..io_utils import config_fingerprint, run_directory_name

    configs = sweep_configs(base, scenario, quick=quick)
    rows: list[dict] = []
    sweep_root = Path(base.output_root) / f"sweep_{scenario.upper()}"
    aggregate_path = sweep_root / "aggregate.csv"
    existing_rows: dict[str, dict] = {}
    if aggregate_path.exists() and aggregate_path.stat().st_size > 0:
        try:
            for old_row in pd.read_csv(aggregate_path).to_dict("records"):
                key = str(old_row.get("config_fingerprint", ""))
                if key:
                    existing_rows[key] = old_row
        except Exception:
            # The run artifacts remain authoritative. A damaged aggregate will be
            # reconstructed as completed runs are loaded from their unique dirs.
            existing_rows = {}
    runs_root = sweep_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Freeze the planned sweep before the first expensive solve. This makes the
    # expected run count explicit and lets progress.py distinguish queued, active,
    # completed, and failed configurations without inferring from directory count.
    planned: list[tuple[int, SimulationConfig]] = []
    manifest_rows: list[dict] = []
    for cfg_index, cfg in enumerate(configs):
        for topo in range(topologies):
            run_cfg = randomized_topology(cfg, seed=10_000 + topo) if topologies > 1 else deepcopy(cfg)
            run_cfg.output_root = str(runs_root)
            planned.append((cfg_index, run_cfg))
            manifest_rows.append({
                "config_index": cfg_index,
                "scenario": scenario.upper(),
                "scheme": run_cfg.scheme.name,
                "topology_seed": run_cfg.topology_seed,
                "config_fingerprint": config_fingerprint(run_cfg, length=0),
                "run_id": config_fingerprint(run_cfg),
                "run_dir": str(runs_root / run_directory_name(run_cfg)),
                "csi_error_radius": run_cfg.physical.csi_error_radius,
                "csi_model": run_cfg.physical.csi_model,
                "rate": run_cfg.physical.min_spectral_efficiency,
                "pcrb_threshold": run_cfg.physical.pcrb_threshold_m2,
                "pcrb_risk": run_cfg.physical.pcrb_violation,
                "prior_position_std_m": run_cfg.physical.initial_position_std_m,
                "prior_velocity_std_mps": run_cfg.physical.initial_velocity_std_mps,
                "process_acceleration_std_mps2": run_cfg.physical.process_acceleration_std_mps2,
                "q_initial_x_m": run_cfg.mission.q_initial_m[0],
                "q_initial_y_m": run_cfg.mission.q_initial_m[1],
                "q_final_x_m": run_cfg.mission.q_final_m[0],
                "q_final_y_m": run_cfg.mission.q_final_m[1],
                "target_centroid_x_m": float(np.mean(np.asarray(run_cfg.mission.targets_initial)[:, 0])),
                "target_centroid_y_m": float(np.mean(np.asarray(run_cfg.mission.targets_initial)[:, 1])),
            })
    pd.DataFrame(manifest_rows).to_csv(sweep_root / "sweep_manifest.csv", index=False)

    s4_warm_starts: dict[float, tuple] = {}
    s3_warm_starts: dict[float, tuple[np.ndarray, np.ndarray, str]] = {}
    for cfg_index, run_cfg in planned:
        if index_filter is not None and cfg_index not in index_filter:
            # Shard support (code_paper_robustcov only): another process owns this
            # config. The full manifest above is still written; the aggregate is
            # rebuilt by a final unfiltered pass once every shard has completed.
            continue
        started = time.perf_counter()
        warm_pair = None
        warm_label = ""
        incumbent = None
        incumbent_label = ""
        if warm_start_adjacent and scenario.upper() == "S4":
            key = float(run_cfg.physical.min_spectral_efficiency)
            if key in s4_warm_starts:
                inc_traj, warm_w, warm_s, warm_label, source_threshold = s4_warm_starts[key]
                warm_pair = (warm_w, warm_s)
                if float(run_cfg.physical.pcrb_threshold_m2) >= float(source_threshold):
                    incumbent = (inc_traj, warm_w, warm_s)
                    incumbent_label = warm_label
        elif warm_start_adjacent and scenario.upper() == "S3":
            # Warm each uncertainty scale from the latest feasible point with the
            # same chance-risk level. This does not certify feasibility; it only
            # helps the SDP/rank-recovery at the strict scale=4, rho=0.01 corner.
            key = float(run_cfg.physical.pcrb_violation)
            if key in s3_warm_starts:
                warm_w, warm_s, warm_label = s3_warm_starts[key]
                warm_pair = (warm_w, warm_s)
        result = run_simulation(
            run_cfg, quick_audit=quick, make_plots=make_plots,
            covariance_warm_start=warm_pair, warm_start_label=warm_label,
            feasible_incumbent=incumbent, incumbent_label=incumbent_label,
        )
        wall_time_s = time.perf_counter() - started
        if (warm_start_adjacent and result.audit is not None
                and bool(result.audit.summary.get("accepted", False))
                and result.w is not None and result.s is not None):
            if scenario.upper() == "S4" and result.trajectory is not None:
                key = float(run_cfg.physical.min_spectral_efficiency)
                s4_warm_starts[key] = (
                    result.trajectory, result.w.copy(), result.s.copy(),
                    run_directory_name(run_cfg), float(run_cfg.physical.pcrb_threshold_m2),
                )
            elif scenario.upper() == "S3":
                key = float(run_cfg.physical.pcrb_violation)
                s3_warm_starts[key] = (
                    result.w.copy(), result.s.copy(), run_directory_name(run_cfg)
                )
        fingerprint_full = config_fingerprint(run_cfg, length=0)
        if result.reason in {"loaded_completed_run", "loaded_cached_failure"} and fingerprint_full in existing_rows:
            # Preserve the original measured wall time and provenance when a sweep
            # is resumed. Re-reading a completed run should not make ETA appear
            # artificially fast.
            cached_row = dict(existing_rows[fingerprint_full])
            cached_row["run_classification"] = classify_run(cached_row)
            rows.append(cached_row)
            aggregate_df = pd.DataFrame(rows)
            tmp = sweep_root / "aggregate.tmp.csv"
            aggregate_df.to_csv(tmp, index=False)
            tmp.replace(aggregate_path)
            continue
        row = {
            "config_index": cfg_index,
            "scenario": scenario.upper(),
            "scheme": run_cfg.scheme.name,
            "topology_seed": run_cfg.topology_seed,
            "config_fingerprint": fingerprint_full,
            "run_id": config_fingerprint(run_cfg),
            "success": result.success,
            "run_dir": str(result.run_dir),
            "wall_time_s": wall_time_s,
            "result_source": result.reason if result.reason == "loaded_completed_run" else "computed",
            "csi_error_radius": run_cfg.physical.csi_error_radius,
            "csi_model": run_cfg.physical.csi_model,
            "rate": run_cfg.physical.min_spectral_efficiency,
            "pcrb_threshold": run_cfg.physical.pcrb_threshold_m2,
            "pcrb_risk": run_cfg.physical.pcrb_violation,
            "prior_position_std_m": run_cfg.physical.initial_position_std_m,
            "prior_velocity_std_mps": run_cfg.physical.initial_velocity_std_mps,
            "process_acceleration_std_mps2": run_cfg.physical.process_acceleration_std_mps2,
            "vertical_loss": run_cfg.physical.vertical_loss_coeff,
            "eval_pathloss": run_cfg.evaluation.actual_pathloss_exponent,
            "eval_acceleration_std": run_cfg.evaluation.actual_process_acceleration_std_mps2,
            "eval_rcs_std_db": run_cfg.evaluation.rcs_log_std_db,
            "K": len(run_cfg.mission.users_m),
            "L": len(run_cfg.mission.targets_initial),
            "M": run_cfg.physical.antennas,
            "N": run_cfg.mission.slots,
            "q_initial_x_m": run_cfg.mission.q_initial_m[0],
            "q_initial_y_m": run_cfg.mission.q_initial_m[1],
            "q_final_x_m": run_cfg.mission.q_final_m[0],
            "q_final_y_m": run_cfg.mission.q_final_m[1],
            "target_centroid_x_m": float(np.mean(np.asarray(run_cfg.mission.targets_initial)[:, 0])),
            "target_centroid_y_m": float(np.mean(np.asarray(run_cfg.mission.targets_initial)[:, 1])),
        }
        if result.audit is not None:
            row.update(result.audit.summary)
        else:
            row["reason"] = result.reason
            row["failure_code"] = result.failure_code
        row["run_classification"] = classify_run(row)
        rows.append(row)
        # Atomic aggregate update so a power loss cannot leave a truncated CSV.
        aggregate_df = pd.DataFrame(rows)
        tmp = sweep_root / "aggregate.tmp.csv"
        aggregate_df.to_csv(tmp, index=False)
        tmp.replace(sweep_root / "aggregate.csv")
    return sweep_root / "aggregate.csv"

