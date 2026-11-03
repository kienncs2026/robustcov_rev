from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
# Figures are only ever written to PDF, never shown. Force the headless backend
# before pyplot is imported: with DISPLAY set (X11 forwarding over SSH) pyplot
# otherwise selects an interactive backend, and when that X connection drops
# mid-run Xlib raises a fatal IO error that terminates the process immediately,
# with no traceback and no results.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    trajectory = pd.read_csv(run_dir / "trajectory.csv")
    power = pd.read_csv(run_dir / "power_covariance.csv")
    pcrb = pd.read_csv(run_dir / "pcrb_out_of_sample.csv")
    iterations = pd.read_csv(run_dir / "iteration_log.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(trajectory.qx_m, trajectory.qy_m, marker="o", markersize=2)
    ax.set_xlabel("x-coordinate (m)")
    ax.set_ylabel("y-coordinate (m)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(run_dir / "trajectory_xy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(trajectory.slot, trajectory.qz_m)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "altitude_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(power.slot, power.total_power_w, label="Total")
    ax.plot(power.slot, power.sensing_power_w, label="Dedicated sensing")
    ax.set_xlabel("Slot")
    ax.set_ylabel("Radiated power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "power_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for target, group in pcrb.groupby("target"):
        ax.plot(group.slot, group.mean_pcrb_m2, label=f"Target {target}")
    ax.set_xlabel("Slot")
    ax.set_ylabel("Mean-state position PCRB (m$^2$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "pcrb_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    if not iterations.empty and "total_energy_j" in iterations:
        finite = iterations[np.isfinite(iterations.total_energy_j)]
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(finite.outer_iteration, finite.total_energy_j, marker="o")
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel("Total energy (J)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "convergence.pdf", bbox_inches="tight")
        plt.close(fig)
