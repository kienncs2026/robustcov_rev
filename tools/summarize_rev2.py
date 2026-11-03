"""Summarize the fair-baseline and seed studies.

Reads results_rev2/{fair,seeds}/*.json and writes results_rev2/summary_rev2.md,
results_rev2/summary_rev2.csv, and results_rev2/table_rev2.tex.

Certified gap (Corollary 2 with the lower bound of tools/fair_baselines.py):
    E_tot(theta) - E*  <=  E_tot(theta) - LB,   LB <= E*_prop <= E*.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_rev2"


def load(folder: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RES / folder).glob("*.json"))]


def pct(x, ref):
    return 100.0 * x / ref if ref else float("nan")


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "spread": max(vals) - min(vals)}


def main() -> None:
    fair, seeds = load("fair"), load("seeds")
    lb = {r["mission"]: r["propulsion_lb_j"] for r in fair}
    seed_ref = {r["mission"]: r["seed_propulsion_j"] for r in fair}
    rows, md = [], ["# Revision 2: fair baselines and independent seeds\n"]

    md.append("## Certified propulsion lower bound\n")
    md.append("| Mission | LB (J) | Seed path E_prop (J) |\n|---|---:|---:|")
    for m in lb:
        md.append(f"| {m} | {lb[m]:.1f} | {seed_ref[m]:.1f} |")

    md.append("\n## Fair baselines (no ISAC information in the path)\n")
    md.append("| Mission | Baseline | E_prop (J) | RF (J) | Total (J) | Outage | UCB95 | Accepted "
              "| Certified gap (J, %) |\n|---|---|---:|---:|---:|---:|---:|---|---:|")
    for r in fair:
        isac = r.get("isac", {})
        tot = isac.get("total_j")
        gap = r.get("certified_gap_j")
        name = r["baseline"] + (f" ({r['radius_m']:.1f} m, {r['max_speed_mps']:.1f} m/s, heading {r['heading_rad']*57.29578:.0f} deg)"
                                if r["baseline"] == "circle" else
                                f" (cruise {r['cruise_speed_mps']:.1f} m/s)")
        md.append(f"| {r['mission']} | {name} | {r['propulsion_j']:.2f} | {isac.get('rf_j', float('nan')):.2f} "
                  f"| {tot if tot is None else round(tot, 2)} | {isac.get('max_random_channel_outage')} "
                  f"| {isac.get('max_pcrb_ucb_95')} | {isac.get('accepted')} "
                  f"| {'' if gap is None else f'{gap:.1f} ({pct(gap, tot):.2f}%)'} |")
        rows.append({"study": "fair", "mission": r["mission"], "design": r["baseline"],
                     "propulsion_j": r["propulsion_j"], "rf_j": isac.get("rf_j"), "total_j": tot,
                     "outage": isac.get("max_random_channel_outage"),
                     "ucb95": isac.get("max_pcrb_ucb_95"), "accepted": isac.get("accepted"),
                     "certified_gap_j": gap, "lb_j": r["propulsion_lb_j"]})

    md.append("\n## Independent optimization seeds\n")
    for m in sorted({r["mission"] for r in seeds}):
        for design in sorted({r["design"] for r in seeds if r["mission"] == m}):
            rs = sorted([r for r in seeds if r["mission"] == m and r["design"] == design],
                        key=lambda r: r["seed_index"])
            md.append(f"### {m}: {design}\n")
            md.append("| k | E_prop (J) | RF (J) | Total (J) | Outage | UCB95 | Accepted | Gap vs LB (J) |"
                      "\n|---:|---:|---:|---:|---:|---:|---|---:|")
            tots, props, ucbs = [], [], []
            for r in rs:
                isac = r.get("isac", r)
                prop = r.get("propulsion_only_propulsion_j", isac.get("propulsion_j"))
                tot = isac.get("total_j")
                gap = None if tot is None or m not in lb else tot - lb[m]
                md.append(f"| {r['seed_index']} | {prop:.2f} | {isac.get('rf_j')} | {tot} "
                          f"| {isac.get('max_random_channel_outage')} | {isac.get('max_pcrb_ucb_95')} "
                          f"| {isac.get('accepted')} | {'' if gap is None else f'{gap:.1f}'} |")
                props.append(prop); tots.append(tot); ucbs.append(isac.get("max_pcrb_ucb_95"))
                rows.append({"study": "seeds", "mission": m, "design": design, "seed_index": r["seed_index"],
                             "propulsion_j": prop, "rf_j": isac.get("rf_j"), "total_j": tot,
                             "outage": isac.get("max_random_channel_outage"), "ucb95": isac.get("max_pcrb_ucb_95"),
                             "accepted": isac.get("accepted"), "certified_gap_j": gap, "lb_j": lb.get(m)})
            for label, vals in (("E_prop", props), ("Total", tots), ("UCB95", ucbs)):
                s = stats(vals)
                if s:
                    md.append(f"\n{label}: mean {s['mean']:.4g}, std {s['std']:.3g}, "
                              f"range [{s['min']:.6g}, {s['max']:.6g}], spread {s['spread']:.3g}"
                              + (f" ({pct(s['spread'], s['mean']):.3f}% of mean)" if label != "UCB95" else ""))
            md.append("")

    (RES / "summary_rev2.md").write_text("\n".join(md))
    with open(RES / "summary_rev2.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    tex = ["% Rows for a fair-baseline / seed table (values in J; gap = E_tot - LB).",
           "% Mission & Design & Propulsion & RF & Total & UCB95 & Certified gap \\\\"]
    for r in rows:
        if r["study"] == "fair" or r.get("seed_index") == 0:
            def ucb_cell(x):
                return "--" if x is None else f"{100 * x:.3f}\\%"

            def f(x, d=2):
                return "--" if x is None else f"{x:,.{d}f}".replace(",", "{,}")
            tex.append(f"{r['mission']} & {r['design']} & {f(r['propulsion_j'])} & {f(r['rf_j'])} & "
                       f"{f(r['total_j'])} & {ucb_cell(r['ucb95'])} & "
                       f"{f(r['certified_gap_j'], 1)} \\\\")
    (RES / "table_rev2.tex").write_text("\n".join(tex) + "\n")
    print((RES / "summary_rev2.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
