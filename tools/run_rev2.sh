#!/usr/bin/env bash
# Revision 2: fair baselines + independent seeds, then the summary.
# Usage: bash tools/run_rev2.sh [jobs]    (from robustcov_rev/)
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
JOBS="${1:-4}"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results_rev2
{
  echo "== fair baselines";  python tools/run_fair_baselines.py --headings 0 90 180 270
  echo "== seed study";      python tools/run_seed_study.py --seeds 0 1 2 3 4 --jobs "$JOBS"
  echo "== AO reference";    python tools/run_seed_study.py --seeds 0 1 2 --missions reference --ao-reference --jobs "$JOBS"
  echo "== summary";         python tools/summarize_rev2.py
} 2>&1 | tee -a results_rev2/run_rev2.log
