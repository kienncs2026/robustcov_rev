# Robust Transmit Covariance and Propulsion-First Trajectory Design for UAV-Enabled ISAC

Simulation code and results for manuscript (revision 1). Every number in the
revised manuscript and in the response letter is produced by the scripts in this package;
`docs/REVIEWER_DATA_MAP.md` maps each response item to the result files, and
`docs/LOGIC_MAP.md` maps each proposition, lemma and algorithm to the implementing function.

All results are included, so the tables can be regenerated in seconds without a conic solver
(Section 3). Re-running the optimization is optional and expensive (Section 4).

## 1. Installation

Python 3.10 or newer. Linux is used throughout; Windows and macOS work for everything except
`tools/run_queue.py` (fork-based multiprocessing).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy, scipy, pandas, cvxpy, scs, clarabel, matplotlib, pytest
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
```

The last line matters: CLARABEL opens one thread per core unless `RAYON_NUM_THREADS` is set, which
oversubscribes the machine when many single-slot solves run in parallel. MOSEK is optional (used only
by the R3 solver comparison when a license is present).

## 2. Smoke tests (2-3 minutes)

```bash
python -m pytest -q tests
```

* `tests/test_theory.py` (about 1 s): Proposition 1 scalar form and rank, Proposition 2 threshold
  `P_min`, Lemma 4 covering radius, Eq. (42) terminal reconstruction, scenario sample size,
  Lemmas 2-3, Proposition 4 screening bound and the Proposition 5 dual certificate.
* `tests/test_pipeline_smoke.py` (about 2 min, a 2x2 UPA / 6-slot mission): linear vs Schur PCRB form,
  scenario certificate, Algorithm 2 refinement, candidate screening, propulsion-only baseline, random
  search, Gaussian randomization, Rician and coordinated-turn mismatch audits, Proposition 5
  infeasibility certificate, post-hoc analysis and table export.
* `tests/test_plan.py`: every experiment plan has a unique fingerprint and the Table II configuration
  fingerprint is unchanged from the submission (`2137019914ce`).

## 3. Regenerate every table and figure from the included results (seconds)

```bash
python run_revision.py outputs --out paper_outputs
python tools/summarize_rev2.py                       # results_rev2/summary_rev2.{md,csv}, table_rev2.tex
```

`paper_outputs/INDEX.csv` lists each file, the response item it answers and its place in the manuscript
(Table II: `T2_matched_baselines.csv`; Table III: `R2_trajectory_baselines.csv`,
`R12_propulsion_first.csv`, `results_rev2/`; Table IV: `T3_local_topologies.csv`,
`R1_random_topologies_*.csv`; Figs. 2-5: `F2_*.csv` to `F5_*.csv`;
Section V-G/H/I: `R3_*`, `R4_*`, `R6_*`, `R7_*`, `R8_*`, `R9_*`, `runtime_breakdown.csv`,
`convergence_iterations.csv`, `lemma_checks.csv`). The figures of the manuscript are drawn from these
CSV files; the auxiliary PDF plots that `outputs` also writes are not part of the manuscript and are omitted here.

Layout of the result folders:

| Folder | Content |
|---|---|
| `results/legacy/` | Submitted-manuscript studies S2, S3, S4, S5A, S5B, S9, S9R (configuration unchanged, reloaded, not re-solved) |
| `results/R1` ... `results/R12`, `results/R9M` | Revision experiments, one directory per run: `config.json`, `summary.csv`, `solver_log.csv`, `trajectory.csv`, `power_covariance.csv`, `covariances.npz`, `pcrb_out_of_sample.csv`, `constraint_residuals.csv`, `iteration_log.csv`, Lemma 2/3 checks, `reaudit/<tag>/` post-hoc audits |
| `results/R9M_local_20261031/` | The same matched-radius design (R1.4, K = 15 dB) computed on a second machine; identical to `results/R9M/` |
| `results_rev2/` | Fair propulsion-only baselines, LP lower bound and the five-seed study (`fair/`, `seeds/`, AO runs in `seeds/runs/`) |
| `results_free_altitude_seed_20261031/` | Propulsion-first designs before the level-flight projection (Remark 3); the "free-altitude seed" rows of Table III |
| `results_slsqp_propulsion_only_20261031/` | Propulsion-only rows with the submitted SLSQP search (superseded, kept for traceability) |
| `logs/` | Queue logs, post-hoc logs, Rician matched-radius logs and the slot-0 radius scan |

The four per-sample scenario files `results/R6/runs/*_scenario_*/fixed_geometry_feasibility_bound.csv`
(0.3-0.6 GB each) are shipped separately; their `*_summary.csv` companions are included here.

## 4. Re-running the optimization (optional)

Every run directory is keyed by a configuration fingerprint. Re-running a command reloads a finished
run from its directory and resumes an unfinished one from its per-slot checkpoints, so the commands
below are safe to repeat. Delete (or rename) a run directory to force a fresh solve.

Step 4.1: plan (lists every run with its fingerprint and cost class)

```bash
python run_revision.py plan
```

Step 4.2: the submission studies (loaded from `results/legacy/` in a few seconds)

```bash
python run_revision.py legacy --which S2,S3,S4,S5A,S5B,S9R
```

Step 4.3: the revision experiments. Either one experiment at a time with static shards,

```bash
python run_revision.py run --exp R11                 # Proposition 5 certificates, seconds + 1 h
python run_revision.py run --exp R2 --shard 0/3      # shard k of 3, run shards in parallel processes
python run_revision.py run --exp R6 --only reference_scenario
python run_revision.py collect --exp R2              # aggregate.csv after all shards finish
```

or with the queue runner (one process per run, N workers, collect done automatically):

```bash
python tools/run_queue.py --workers 14 --exp R11,R12,R2,R5,R8,R7,R3,R4,R6,R1
```

| Exp | Runs | Content | Response items | Cost per run, 1 thread |
|---|---|---|---|---|
| R11 | 5 | Proposition 5 infeasibility certificate at eps = 4 m^2, all 48 slots | E.3, R1.2, R3.1 | seconds (4 runs), 1 h (1 run) |
| R12 | 11 | Level propulsion-first designs: Table II methods and the offset/threshold sweep | E.3, R1.2, R2.5, R3.1 | 1 h (reference), 1.7 h (return-to-base; 1.5 h is the 30-start Powell search) |
| R2 | 6 | Trajectory baselines: propulsion-only, random search, AO-Powell | R1.7, R2.5, R3.3 | 1-1.7 h |
| R5 | 3 | Robust fixed sensing fractions 10/20/30 % | R3.3 | 1-3 h |
| R6 | 8 | Scenario enforcement (Prop. 3) and refinement (Alg. 2) on four geometries | E.4, R2.3, R3.2 | 2-6 h |
| R7 | 15 | Tightening sweep 0-20 % x three geometries, Lipschitz estimates | R1.3 | 0.5-1 h |
| R3 | 6 | SCS / CLARABEL / MOSEK x Schur / linear on the fixed trajectory | R3.3, R2.6, E.5 | 0.5-2 h |
| R4 | 4 | Rank recovery: bank vs Gaussian randomization | R3.3, R3.4 | 0.5-2 h |
| R8 | 9 | Sensitivity to starts, segments, SLSQP iterations | R2.4 | 5-15 min |
| R1 | 60 | 20 local and 10 unrestricted topologies x {nine-state, refined} | R1.6, R1.9, E.4 | 1-4 h each |

Step 4.4: post-hoc analyses on stored designs (no re-optimization; minutes)

```bash
python run_revision.py posthoc lipschitz     # Lemma 4 margins for R7 and the reference designs (R1.3)
python run_revision.py posthoc mismatch      # Rician K = 15/10/5 dB and 2/5 deg/s turns on the Table II designs (R1.4)
python run_revision.py posthoc seeds         # three independent Monte Carlo audit seeds (R1.9)
python run_revision.py posthoc jointbound    # Proposition 5 on every slot
python tools/joint_bound_margin_check.py     # slot-1 joint bound at eps = 4 m^2 with and without the 10 % margin (2.023 W / 1.794 W)
```

Step 4.5: matched Rician radius (R1.4; 1-2 h per value of K)

```bash
python tools/rician_matched.py 15            # r = 0.2112, feasible   -> results/R9M/
python tools/rician_matched.py 10            # r = 0.3665, infeasible
python tools/rician_matched.py 5             # r = 0.6086, infeasible
python tools/rician_radius_scan.py 0.2112,0.25,0.26,0.3665,0.6086 0     # slot-0 radius scan with SCS
```

Step 4.6: fair baselines, LP lower bound and the five-seed study (E.3, R1.7, R1.9, R2.5)

```bash
bash tools/run_rev2.sh 10                    # everything below, 10 parallel workers, about 2 h
# or step by step
python tools/run_fair_baselines.py --headings 0 90 180 270              # 5 baselines, 15-30 min each
python tools/run_seed_study.py --seeds 0 1 2 3 4 --ao-reference --jobs 15
python tools/summarize_rev2.py
```

Finished tasks are skipped on rerun; a level task whose Powell path (`results_rev2/seeds/<task>.npz`)
exists reuses it and only redoes the covariance solve and audit.

Step 4.7: export

```bash
python run_revision.py outputs --out paper_outputs
```

## 5. Reproducibility notes

* All randomness is seeded (`solver.random_seed = 2026`; topology seeds 11000-11001, 20000-20019,
  30000-30009; Powell seeds `random_seed + 61000 + k`; audit seeds derived from the run seed).
* Conic solves are deterministic given the solver; CLARABEL and SCS results agree to the reported
  precision (Section V-G). The derivative-free Powell search can return a different local optimum on a
  different floating-point platform (observed: 25,961.8 J vs 25,816.4 J on the return-to-base mission);
  the five-seed study in `results_rev2/` quantifies this spread.
* `README_REVISION_VI.md` and `README_REV2_VI.md` are the authors' working notes in Vietnamese and
  contain the same commands with additional operational detail.
