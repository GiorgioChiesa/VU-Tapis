#!/usr/bin/env bash
# EXP-21 phase 2 jobs (2026-11-19): long models on the arm M features (each run evaluates itself on inner val and
# test, see run_files/exp21_long.sh), then a CPU summary against the reference.
cd "$(dirname "$0")/../.."
Q=run_files/gpuq/q
FEAT="count:50:results/M/fold0/untrimmed/RARP*.npz"
ids=()
p=200
for s in 0 1; do for r in a3 sed a3_causal sed_causal; do
  $Q add --id exp21_long_${r}_s$s --priority $p --group exp21 --mem 8000 --deps "$FEAT" \
    --log results/M/fold0/long/gpuq_${r}_s$s.log --cmd "RUNS=$r SEEDS=$s bash run_files/exp21_long.sh M {gpu} && [ -e results/M/fold0/long/${r}_s$s/eval/held_out.json ]"
  ids+=(exp21_long_${r}_s$s); p=$((p - 1))
done; done
$Q add --id exp21_long_summary --priority 150 --group exp21 --cpu --deps "${ids[@]}" \
  --log results/M/fold0/long/gpuq_summary.log --cmd "bash run_files/exp21_long_summary.sh"
