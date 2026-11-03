# Removed from the original package

Kept in the original archive, not needed to reproduce or extend the paper:

* Checkpoint drivers and validators: `scripts/run_cp5_*`, `run_cp6_*`, `run_cp7_postlock.py`,
  `run_s9r.py`, `validate_*`, `progress_*`, `reuse_*`, `lock_*`, `finalize_paper_lock.py`,
  `preflight_*`, `make_cp7_figures.py`, `make_s9r_figures.py`, `run_all.py`, `compare_results.py`.
  Replaced by `run_revision.py legacy|run|outputs`.
* Modules `cp5b.py`, `checkpoints.py`, `protocol.py`, `validation.py`, `grid_reference.py`, `cli.py`.
* Configs for CP5/CP6 v2/v3 and local shards; only the locked reference
  (`configs/reference_strict.json`, identical to cp6_lock/cp7_postlock/cp7_s9r) and smoke configs remain.
* CHANGELOG/README/VERIFICATION files of the CP5–CP7 checkpoints.

Moved unchanged (imports only) to `uav_isac/legacy/`: `scenarios.py`, `postlock.py`,
`s9r.py`, `cp6_quality.py`. They define the frozen S2–S9R protocols, so archived run
directories keep their fingerprints.
