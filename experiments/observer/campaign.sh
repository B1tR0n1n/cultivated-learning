#!/usr/bin/env bash
# Run the pre-registered matrix in order: by seed, then arm. Resumable: a run
# with summary.json is skipped; a run with only progress.jsonl stops the
# campaign (run.py refuses to overwrite it) so a crash is never hidden.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OBSERVER_OUT:-/mnt/vault/projects/cultivated-learning/observer/runs}
mkdir -p "$OUT"

for seed in 1 2 3 4 5 6 7 8; do
  for arm in SELF SELF_PLUS PEER OTHER; do
    if [[ $seed -gt 4 && ( $arm == SELF || $arm == OTHER ) ]]; then
      continue
    fi
    dir="$OUT/$arm-s$seed"
    if [[ -f "$dir/summary.json" ]]; then
      echo "skip $arm s$seed (done)"
      continue
    fi
    echo "=== $(date '+%F %T') start $arm s$seed"
    .venv/bin/python -m experiments.observer.run --arm "$arm" --seed "$seed" --out "$dir" > "$OUT/$arm-s$seed.log" 2>&1
    echo "=== $(date '+%F %T') done  $arm s$seed"
  done
done
echo "=== campaign complete $(date '+%F %T')"
