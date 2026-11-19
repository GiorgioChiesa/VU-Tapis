#!/usr/bin/env bash
# EXP-19 resume at the lowest priority (GPU_HANDOVER.md (c)): each job drops its own HOLD_EXP21 placeholder (only if
# that job never ran) and runs one EXP-19 lane for that single job on the GPU gpuq gives it. The jobs wait while any
# EXP-21 job is pending (wait_for_group exp21) and need a fully free GPU (exclusive, < 1500 MiB: the EXP-19 lane and
# cv_finetune.py refuse a GPU with any process on it, e.g. the other user idle process on GPU3).
cd "$(dirname "$0")/../.."
Q=run_files/gpuq/q
SLC=/home/gchie/workspace/wt/short-long-context
p=5
for j in S_fold3 AUG_none_fold2 S_fold4 AUG_none_fold3 AUG_none_fold4; do
  $Q add --id exp19_$j --priority $p --group exp19 --wait-for-group exp21 --exclusive --max-used 1500 --mem 20000 \
    --cwd $SLC --log results/queue/gpuq_$j.log \
    --cmd "d=results/queue/$j; if [ -e \$d/HOLD_EXP21 ] && [ ! -e \$d/job.log ]; then rm -r \$d; fi; export NUMPY_MADVISE_HUGEPAGE=0; JOBS=$j bash run_files/exp19_queue.sh {gpu} && [ -e \$d/finished ]"
  p=$((p - 1))
done
