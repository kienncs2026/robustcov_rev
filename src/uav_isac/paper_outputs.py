"""Build the CSV tables and figures used in the revised manuscript.

All tables are rebuilt from run directories (``config.json`` + ``summary.csv``),
so they do not depend on aggregate files or on the order in which shards ran.
``INDEX.csv`` maps every output to the reviewer comments and manuscript items.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

INDEX: list[dict] = []


def _register(name: str, description: str, reviewers: str, manuscript: str) -> None:
    INDEX.append({"file": name, "description": description, "reviewer_comments": reviewers,
                  "manuscript_item": manuscript})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _flatten(cfg: dict) -> dict:
    ph, mi, so, sc = cfg["physical"], cfg["mission"], cfg["solver"], cfg["scheme"]
    ev = cfg.get("evaluation", {})
    targets = np.asarray(mi["targets_initial"], float)
    return {
        "scenario": cfg.get("scenario"), "scheme_name": sc.get("name"),
        "topology_seed": cfg.get("topology_seed", 0), "random_seed": so.get("random_seed"),
        "rate": ph.get("min_spectral_efficiency"), "eps": ph.get("pcrb_threshold_m2"),
        "risk": ph.get("pcrb_violation"), "csi_radius": ph.get("csi_error_radius"),
        "csi_model": ph.get("csi_model"), "prior_pos_std": ph.get("initial_position_std_m"),
        "margin": ph.get("enforcement_margin"), "solver": so.get("solver"),
        "pcrb_formulation_cfg": so.get("pcrb_formulation", "schur"),
        "pcrb_enforcement_cfg": so.get("pcrb_enforcement", "principal_axes"),
        "rank_recovery_mode": so.get("rank_recovery_mode", "bank"),
        "trajectory_method": so.get("trajectory_method"),
        "trajectory_objective": sc.get("trajectory_objective", "joint"),
        "seed_mode": so.get("seed_mode", "default"),
        # Configs written before 31/10 lack the field; those propulsion-only runs used SLSQP.
        "propulsion_only_method": so.get("propulsion_only_method", "slsqp"),
        "multistart": so.get("trajectory_multistart"),
        "segments": so.get("trajectory_control_segments"),
        "maxiter": so.get("trajectory_powell_maxiter"),
        "robust_communication": sc.get("robust_communication"),
        "chance_sensing": sc.get("chance_sensing"), "fixed_trajectory": sc.get("fixed_trajectory"),
        "sensing_mode": sc.get("sensing_mode"), "sensing_fraction": sc.get("sensing_fraction"),
        "q0_x": mi["q_initial_m"][0], "q0_y": mi["q_initial_m"][1],
        "qF_x": mi["q_final_m"][0], "target0_y": float(targets[0, 1]),
        "K": len(mi["users_m"]), "L": len(targets),
        "eval_rician_k_db": ev.get("rician_k_db"), "eval_turn_rate": ev.get("target_turn_rate_dps"),
    }


def solver_totals(run_dir: Path) -> dict:
    path = run_dir / "solver_log.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"slot_solves": 0}
    out = {"solver_time_sum_s": float(df.get("total_solver_runtime_s", df.get("runtime_s")).fillna(0).sum()),
           "slot_solves": int(len(df))}
    for col in ("rank_recovery_call_count", "rank_refinement_call_count", "fallback_call_count"):
        if col in df:
            out[f"{col}_sum"] = int(df[col].fillna(0).sum())
    return out


def load_runs(results_root: str | Path) -> pd.DataFrame:
    rows = []
    for ident in Path(results_root).rglob("RUN_IDENTITY.json"):
        run_dir = ident.parent
        if "reaudit" in run_dir.parts:
            continue
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            continue
        try:
            cfg = json.loads(ident.read_text())["scientific_config"]
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
        except Exception:
            continue
        row = {"run_dir": str(run_dir), "run_name": run_dir.name,
               "group": run_dir.parent.parent.name,
               "completed": (run_dir / "state" / "COMPLETED").exists(),
               "failed": str(summary.get("status", "")).lower() == "failed"}
        row.update(_flatten(cfg))
        row.update(summary)
        row.update(solver_totals(run_dir))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "accepted" in df:
        df["accepted"] = df["accepted"].astype(str).str.lower().isin(["true", "1"])
    return df


def _cp_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


COLS = ["scenario", "scheme_name", "topology_seed", "total_j", "propulsion_j", "rf_j",
        "max_random_channel_outage", "max_empirical_pcrb_violation", "max_pcrb_ucb_95",
        "accepted", "failure_code", "wall_total_s", "solver_time_sum_s"]


def _cols(df: pd.DataFrame, extra: list[str] = ()) -> pd.DataFrame:
    keep = [c for c in list(extra) + COLS if c in df.columns]
    return df[list(dict.fromkeys(keep))]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def build_outputs(results_root: str | Path, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    INDEX.clear()
    df = load_runs(results_root)
    df.to_csv(out / "all_runs.csv", index=False)
    _register("all_runs.csv", "Every run (config + audit summary + solver totals)", "all", "source data")
    if df.empty:
        pd.DataFrame(INDEX).to_csv(out / "INDEX.csv", index=False)
        return out
    sc = df["scenario"].astype(str)

    # Table II: matched baselines at T0 (+ robust fixed sensing, trajectory baselines)
    t2 = df[((sc == "S9") & (df.topology_seed == 0)) | (sc == "R5")
            | ((sc == "R2") & (df.q0_x != df.qF_x))]
    _cols(t2, ["trajectory_objective", "trajectory_method", "sensing_mode", "sensing_fraction"]).to_csv(
        out / "T2_matched_baselines.csv", index=False)
    _register("T2_matched_baselines.csv", "Matched-system baselines at the reference geometry, incl. robust fixed sensing power and trajectory baselines",
              "R3.3, R1.7, Ed.5", "Table II")

    # Fig. 2 CSI, Fig. 3 prior/risk, Fig. 4 rate-PCRB
    _cols(df[sc == "S2"], ["csi_model", "csi_radius"]).sort_values(["csi_model", "csi_radius"]).to_csv(
        out / "F2_csi_robustness.csv", index=False)
    _register("F2_csi_robustness.csv", "CSI sweep, robust vs deterministic", "-", "Fig. 2")
    s3 = df[sc == "S3"].copy()
    if not s3.empty:
        s3["ucb_over_risk"] = s3["max_pcrb_ucb_95"] / s3["risk"]
    _cols(s3, ["prior_pos_std", "risk", "ucb_over_risk"]).to_csv(out / "F3_prior_risk.csv", index=False)
    _register("F3_prior_risk.csv", "Prior-uncertainty / risk sweep", "-", "Fig. 3")
    s4 = pd.concat([df[sc == "S4"], df[sc == "R11"]])
    _cols(s4, ["rate", "eps", "joint_bound_slot1_certified_w", "joint_bound_slot1_per_point_w"]).to_csv(
        out / "F4_rate_pcrb_and_certificates.csv", index=False)
    _register("F4_rate_pcrb_and_certificates.csv",
              "Rate-PCRB operating region and Prop. 5 certificates at eps = 4 m^2",
              "Ed.3, R1.2, R3.1", "Fig. 4, Sec. V-C, Prop. 5")

    # Mobility (S5A/S5B) with energy decomposition
    mob = df[sc.isin(["S5A", "S5B"])].copy()
    if not mob.empty:
        mob["offset_m"] = mob["target0_y"] - 300.0
        mob["mode"] = np.where(mob["fixed_trajectory"].astype(bool), "hover", "mobile")
        rows = []
        for (s, off, eps), g in mob.groupby(["scenario", "offset_m", "eps"]):
            h, m = g[g["mode"] == "hover"], g[g["mode"] == "mobile"]
            if len(h) and len(m):
                h, m = h.iloc[0], m.iloc[0]
                rows.append({"scenario": s, "offset_m": off, "eps": eps,
                             "hover_total_j": h.total_j, "mobile_total_j": m.total_j,
                             "saving_total_j": h.total_j - m.total_j,
                             "saving_propulsion_j": h.propulsion_j - m.propulsion_j,
                             "saving_rf_j": h.rf_j - m.rf_j,
                             "rf_share_bound_percent": 100 * h.rf_j / h.total_j,
                             "saving_percent": 100 * (h.total_j - m.total_j) / h.total_j,
                             "mobile_displacement_m": m.get("max_displacement_from_initial_seed_m", np.nan)})
        pd.DataFrame(rows).to_csv(out / "F5_mobility_decomposition.csv", index=False)
        _register("F5_mobility_decomposition.csv", "Displaced-target mobility: propulsion vs RF saving and Cor. 1 bound",
                  "R1.2, R2.5, R3.1", "Fig. 5, Cor. 1")

    # Table III and R1 random topologies
    t3 = df[(sc == "S9R") | ((sc == "S9") & (df.topology_seed == 0))]
    _cols(t3).to_csv(out / "T3_local_topologies.csv", index=False)
    _register("T3_local_topologies.csv", "Reference and two locally perturbed topologies", "-", "Table III")
    r1 = df[sc == "R1"].copy()
    if not r1.empty:
        r1["method"] = r1["scheme_name"].str.extract(r"^R1_(?:local|unrestricted)\d+_(.+)$")[0]
        r1["kind"] = np.where(r1["topology_seed"] >= 30_000, "unrestricted", "local")
        r1["meets_risk"] = r1["max_pcrb_ucb_95"] <= r1["risk"]
        _cols(r1, ["kind", "method", "meets_risk"]).to_csv(out / "R1_random_topologies_runs.csv", index=False)
        rows = []
        for (kind, method), g in r1.groupby(["kind", "method"]):
            feas = g[~g["failed"]]
            k = int(feas["meets_risk"].sum())
            lo, hi = _cp_interval(k, len(feas))
            rows.append({"kind": kind, "method": method, "topologies": len(g),
                         "feasible": len(feas), "meets_risk": k,
                         "meets_risk_fraction": k / len(feas) if len(feas) else np.nan,
                         "meets_risk_ci95_low": lo, "meets_risk_ci95_high": hi,
                         "median_ucb": feas["max_pcrb_ucb_95"].median(),
                         "p90_ucb": feas["max_pcrb_ucb_95"].quantile(0.9) if len(feas) else np.nan,
                         "max_outage": feas["max_random_channel_outage"].max(),
                         "mean_rf_j": feas["rf_j"].mean(), "mean_wall_s": feas.get("wall_total_s", pd.Series(dtype=float)).mean()})
        pd.DataFrame(rows).to_csv(out / "R1_random_topologies_summary.csv", index=False)
        _register("R1_random_topologies_summary.csv", "Reliability over random topologies with Clopper-Pearson interval on the pass fraction",
                  "R1.6, R1.9, Ed.5, R2.3", "new Table (topology statistics)")
        _fig_r1(r1, out)

    # R2 trajectory baselines incl. legacy proposed / fixed
    r2 = df[(sc == "R2") | ((sc == "S9") & (df.topology_seed == 0)
                            & df.scheme_name.isin(["cp7_proposed", "cp7_adapted_robust_fixed_geometry"]))
            | ((sc == "S5B") & (df.eps == 9.0))].copy()
    if not r2.empty:
        r2["geometry"] = np.where(r2.q0_x == r2.qF_x, "displaced", "reference")
        _cols(r2, ["geometry", "trajectory_objective", "trajectory_method", "propulsion_only_method",
                   "propulsion_only_level", "vertical_motion_j", "propulsion_only_max_displacement_m",
                   "fixed_trajectory", "max_displacement_from_initial_seed_m",
                   "accepted_trajectory_updates"]).to_csv(
            out / "R2_trajectory_baselines.csv", index=False)
        _register("R2_trajectory_baselines.csv", "Proposed vs fixed, propulsion-only, random search, Powell",
                  "R1.7, R2.5, R3.3, Ed.3", "new Table/Fig. (trajectory baselines)")
        _fig_trajectories(r2, out)

    # R3 solver / formulation, R4 rank recovery, R5 is in T2
    r3 = df[(sc == "R3") | ((sc == "S9") & (df.topology_seed == 0)
                            & (df.scheme_name == "cp7_adapted_robust_fixed_geometry"))]
    _cols(r3, ["solver", "pcrb_formulation_cfg", "slot_solves", "fallback_call_count_sum",
               "time_covariance_s"]).to_csv(out / "R3_solver_formulation.csv", index=False)
    _register("R3_solver_formulation.csv", "Solver and Schur vs linear (Prop. 2): energy, feasibility, solver time",
              "R3.3, R2.6", "new Table (solver comparison), Prop. 2 check")
    _cols(df[sc == "R4"], ["rank_recovery_mode", "max_rank_residual", "rank_recovery_call_count_sum"]).to_csv(
        out / "R4_rank_recovery.csv", index=False)
    _register("R4_rank_recovery.csv", "Rank recovery bank vs Gaussian randomization", "R3.3, R3.4", "Sec. IV-D / response")

    # R6 reliability remedies (+ nine-state legacy rows)
    r6 = df[(sc == "R6") | (sc == "S9R") | ((sc == "S9") & (df.topology_seed == 0))
            | ((sc == "S3") & (df.prior_pos_std > 7.9) & (df.risk < 0.011))]
    r6 = r6[r6.get("chance_sensing", True).astype(bool) & ~r6.get("fixed_trajectory", False).astype(bool)]
    _cols(r6, ["pcrb_enforcement_cfg", "pcrb_formulation_cfg", "risk", "refinement_validated",
               "refinement_rounds_used", "refinement_points_added"]).to_csv(out / "R6_reliability_remedies.csv", index=False)
    _register("R6_reliability_remedies.csv", "Nine-state vs scenario (Prop. 3) vs refinement (Alg. 2) on failing cases",
              "Ed.4, R2.3, R3.2", "new Table (reliability remedies)")

    # R7 tightening (+ Lemma 4 L_w if computed)
    r7 = df[sc == "R7"].copy()
    if not r7.empty:
        lw = []
        for rd in r7["run_dir"]:
            p = Path(rd) / "lemma4_lipschitz.csv"
            if p.exists():
                t = pd.read_csv(p)
                lw.append({"run_dir": rd, "max_L_w_m2": t.L_w_m2.max(),
                           "max_required_margin_m2": t.required_margin_m2.max(),
                           "lemma4_sufficient_all": bool(t.lemma4_sufficient.all())})
        if lw:
            r7 = r7.merge(pd.DataFrame(lw), on="run_dir", how="left")
        _cols(r7, ["margin", "max_L_w_m2", "max_required_margin_m2", "lemma4_sufficient_all"]).to_csv(
            out / "R7_tightening_lipschitz.csv", index=False)
        _register("R7_tightening_lipschitz.csv", "Tightening sweep with Lemma 4 margin check", "R1.3", "Lemma 4 discussion")
        _fig_r7(r7, out)

    # R8 sensitivity
    r8 = df[(sc == "R8") | ((sc == "S5B") & (df.eps == 9.0) & ~df.fixed_trajectory.astype(bool))]
    _cols(r8, ["multistart", "segments", "maxiter", "accepted_trajectory_updates",
               "max_displacement_from_initial_seed_m"]).to_csv(out / "R8_trajectory_sensitivity.csv", index=False)
    _register("R8_trajectory_sensitivity.csv", "Starts / segments / SLSQP iterations sensitivity", "R2.4, R1.10", "Sec. IV-D / response")

    # R12 propulsion-first design next to the submitted designs it replaces
    r12 = df[(sc == "R12") | (sc.isin(["S5A", "S5B"]))
             | ((sc == "S9") & (df.topology_seed == 0) & df.scheme_name.isin(
                 ["cp7_proposed", "cp7_adapted_deterministic_joint", "cp7_adapted_trajectory_fixed_sensing20"]))].copy()
    if (r12.scenario == "R12").any():
        r12["geometry"] = np.where(r12.q0_x == r12.qF_x, "displaced", "reference")
        r12["offset_m"] = np.where(r12.geometry == "displaced", r12.target0_y - 300.0, np.nan)
        _cols(r12, ["geometry", "offset_m", "eps", "rate", "seed_mode", "fixed_trajectory",
                    "propulsion_only_propulsion_j", "propulsion_only_level", "vertical_motion_j",
                    "propulsion_only_max_displacement_m", "max_displacement_from_initial_seed_m",
                    "accepted_trajectory_updates"]).sort_values(
            ["geometry", "offset_m", "eps", "scenario"]).to_csv(out / "R12_propulsion_first.csv", index=False)
        _register("R12_propulsion_first.csv", "Propulsion-first design (Cor. 2) vs submitted S9 T0, S5A and S5B designs",
                  "Ed.3, R1.2, R2.5, R3.1", "Table II, Sec. V mobility (NEXT STEP)")

    # Re-audits: mismatch (R9) and MC seeds
    re_rows = []
    for summ in Path(results_root).rglob("reaudit/*/summary.csv"):
        r = pd.read_csv(summ).iloc[0].to_dict()
        r["run_name"] = summ.parent.parent.parent.name
        re_rows.append(r)
    if re_rows:
        re = pd.DataFrame(re_rows)
        re[re["tag"].astype(str).str.startswith("mismatch")].to_csv(out / "R9_model_mismatch.csv", index=False)
        re[re["tag"].astype(str).str.startswith("mcseed")].to_csv(out / "R1_9_monte_carlo_seeds.csv", index=False)
        _register("R9_model_mismatch.csv", "Rician channel and coordinated-turn target mismatch", "R1.4", "new paragraph Sec. V")
        _register("R1_9_monte_carlo_seeds.csv", "Independent Monte Carlo seeds on stored designs", "R1.9", "response R1.9")

    # Convergence, runtime, lemma checks
    conv = []
    for rd in df["run_dir"]:
        p = Path(rd) / "iteration_log.csv"
        if p.exists():
            it = pd.read_csv(p)
            it.insert(0, "run_name", Path(rd).name)
            conv.append(it)
    if conv:
        pd.concat(conv).to_csv(out / "convergence_iterations.csv", index=False)
        _register("convergence_iterations.csv", "Energy per outer iteration for every run", "Ed.2, R1.10", "Prop. 6 / convergence figure")
    _cols(df, [c for c in df.columns if c.startswith("time_")] + ["covariance_passes", "slot_solves"]).to_csv(
        out / "runtime_breakdown.csv", index=False)
    _register("runtime_breakdown.csv", "Wall time per phase and solver time per run", "R2.6, R3.4", "Sec. V complexity paragraph")
    lem = [c for c in df.columns if c.startswith("lemma")]
    if lem:
        _cols(df, lem).to_csv(out / "lemma_checks.csv", index=False)
        _register("lemma_checks.csv", "Numerical checks of Lemma 2 and Lemma 3 per run", "R2.2, R3.2", "Sec. III-D")

    pd.DataFrame(INDEX).to_csv(out / "INDEX.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Figures (simple, publication-ready styling is left to the authors)
# ---------------------------------------------------------------------------
def _fig_trajectories(r2: pd.DataFrame, out: Path) -> None:
    for geom, g in r2.groupby("geometry"):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for _, row in g.iterrows():
            p = Path(row.run_dir) / "trajectory.csv"
            if p.exists():
                t = pd.read_csv(p)
                ax.plot(t.qx_m, t.qy_m, label=str(row.scheme_name), lw=1.2)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6); ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout(); fig.savefig(out / f"fig_trajectories_{geom}.pdf"); plt.close(fig)
        _register(f"fig_trajectories_{geom}.pdf", f"UAV paths, {geom} geometry", "R1.7, R2.5", "new trajectory figure")


def _fig_r1(r1: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for (kind, method), g in r1[~r1["failed"]].groupby(["kind", "method"]):
        v = np.sort((g["max_pcrb_ucb_95"] / g["risk"]).to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=f"{kind}/{method}")
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("UCB / rho"); ax.set_ylabel("empirical CDF")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_R1_ucb_cdf.pdf"); plt.close(fig)
    _register("fig_R1_ucb_cdf.pdf", "CDF of UCB/rho over random topologies", "R1.6", "new figure")


def _fig_r7(r7: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    r7 = r7.copy()
    r7["case"] = r7["scheme_name"].str.extract(r"R7_(\w+?)_margin")[0]
    for case, g in r7.groupby("case"):
        g = g.sort_values("margin")
        ax.plot(100 * g["margin"], g["max_pcrb_ucb_95"], marker="o", label=case)
    ax.axhline(float(r7["risk"].iloc[0]), color="k", lw=0.8, ls="--")
    ax.set_xlabel("tightening (%)"); ax.set_ylabel("max PCRB UCB"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_R7_tightening.pdf"); plt.close(fig)
    _register("fig_R7_tightening.pdf", "UCB vs tightening margin", "R1.3", "Lemma 4 discussion")
