"""Command-line entry point for the revision experiments.

    python run_revision.py plan                      # list experiments and run counts
    python run_revision.py legacy --which S9,S9R     # submission studies (reuse or rerun)
    python run_revision.py run --exp R2 [--shard 0/4]
    python run_revision.py posthoc lipschitz|mismatch|seeds|jointbound
    python run_revision.py outputs                   # tables/figures for the paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _shard(text: str) -> tuple[int, int]:
    k, n = (int(x) for x in text.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError("shard must be k/n with 0 <= k < n")
    return k, n


def _opts(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="UAV-ISAC revision experiments")
    ap.add_argument("--config", default="configs/reference_strict.json")
    ap.add_argument("--results", default="results")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan"); p.add_argument("--exp", default="all"); p.add_argument("--opt", nargs="*")
    r = sub.add_parser("run"); r.add_argument("--exp", required=True); r.add_argument("--shard", type=_shard, default=(0, 1))
    r.add_argument("--opt", nargs="*"); r.add_argument("--only", nargs="*"); r.add_argument("--plots", action="store_true")
    lg = sub.add_parser("legacy"); lg.add_argument("--which", default="S2,S3,S4,S5A,S9,S5B,S9R")
    lg.add_argument("--shard", type=_shard, default=(0, 1)); lg.add_argument("--plots", action="store_true")
    c = sub.add_parser("collect"); c.add_argument("--exp", required=True)
    ph = sub.add_parser("posthoc"); ph.add_argument("what", choices=["lipschitz", "mismatch", "seeds", "jointbound"])
    ph.add_argument("--samples", type=int, default=None); ph.add_argument("--run-dir", nargs="*")
    ph.add_argument("--redo", action="store_true", help="recompute runs whose output already exists")
    o = sub.add_parser("outputs"); o.add_argument("--out", default="paper_outputs")
    args = ap.parse_args(argv)

    base = load_config(args.config)
    results = Path(args.results)

    if args.cmd == "plan":
        from .experiments import EXPERIMENTS, build
        ids = list(EXPERIMENTS) if args.exp == "all" else [x.strip().upper() for x in args.exp.split(",")]
        for exp_id in ids:
            runs = build(exp_id, base, _opts(args.opt))
            e = EXPERIMENTS[exp_id]
            print(f"{exp_id:4s} {len(runs):4d} runs  [{e.reviewer_refs}]  {e.title}")
        return
    if args.cmd == "run":
        from .runner import run_experiment
        path = run_experiment(args.exp, base, results, opts=_opts(args.opt), shard=args.shard,
                              make_plots=args.plots, only=args.only)
        print(f"aggregate={path}")
        return
    if args.cmd == "legacy":
        from .runner import run_legacy
        for path in run_legacy(base, results / "legacy", [x.strip() for x in args.which.split(",")],
                               shard=args.shard, make_plots=args.plots):
            print(f"aggregate={path}")
        return
    if args.cmd == "collect":
        from .runner import collect
        print(collect(results / args.exp.upper()))
        return
    if args.cmd == "posthoc":
        from . import posthoc
        from .paper_outputs import load_runs
        df = load_runs(results)
        if args.run_dir:
            dirs = list(args.run_dir)
        elif args.what == "lipschitz":
            sel = df[(df.scenario == "R7") | (df.scenario == "S9R") |
                     ((df.scenario == "S9") & (df.topology_seed == 0))]
            dirs = sel[~sel.failed & sel.chance_sensing.astype(bool)].run_dir.tolist()
        elif args.what == "mismatch":
            sel = df[(df.scenario == "S9") & (df.topology_seed == 0) & ~df.failed]
            dirs = sel.run_dir.tolist()
        elif args.what == "seeds":
            sel = df[((df.scenario == "R1") & df.scheme_name.str.endswith("nine_state")) |
                     ((df.scenario == "S9") & (df.topology_seed == 0) & (df.scheme_name == "cp7_proposed"))]
            dirs = sel[~sel.failed].run_dir.tolist()
        else:
            sel = df[((df.scenario == "S9") & (df.topology_seed == 0)) | (df.scenario == "S9R")]
            dirs = sel[~sel.failed].run_dir.tolist()
        # Last file each analysis writes; its presence means the run finished, so a rerun
        # after an interruption resumes instead of starting over.
        marker = {"lipschitz": "lemma4_lipschitz.csv", "jointbound": "joint_bound_all_slots.csv",
                  "mismatch": "reaudit/mismatch_target_turn_rate_dps5/constraint_residuals.csv",
                  "seeds": "reaudit/mcseed3/constraint_residuals.csv"}[args.what]
        skipped = [] if args.redo else [rd for rd in dirs if (Path(rd) / marker).exists()]
        dirs = [rd for rd in dirs if rd not in skipped]
        print(f"{args.what}: {len(dirs)} run directories ({len(skipped)} already done, skipped)")
        for rd in dirs:
            if args.what == "lipschitz":
                posthoc.lipschitz_table(rd, samples=args.samples or 100)
            elif args.what == "mismatch":
                posthoc.mismatch_sweep([rd], samples=args.samples)
            elif args.what == "seeds":
                posthoc.seed_sweep([rd], samples=args.samples)
            else:
                posthoc.joint_bound_table(rd)
            print(f"  done {Path(rd).name}", flush=True)
        return
    if args.cmd == "outputs":
        from .paper_outputs import build_outputs
        print(build_outputs(results, args.out))


if __name__ == "__main__":
    main()
