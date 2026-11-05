#!/usr/bin/env bash
# Zero-shot VLM event spotting on the 6 validation videos, 3 models on 2 GPUs, then evaluation.
# Run from src/VU-Tapis inside tmux:  bash tools/vlm/run_val.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/gchie/miniforge3/envs/vlm/bin/python
EVAL_PY=/home/gchie/miniforge3/envs/Vu-Tapis/bin/python
M=/home/gchie/workspace/nas_private/models
FD=/home/gchie/workspace/data/orsi_tensors

until grep -q "^RARP64" /home/gchie/workspace/data/orsi_frames448.log 2>/dev/null && ! tmux has-session -t frames 2>/dev/null; do sleep 30; done

run() {  # name model gpu
    $PY tools/vlm/infer_vlm_events.py --model "$2" --gpu "$3" --out-dir "outputs/vlm/$1/val" 2>&1 \
        | grep -v -iE "warn|cap_pixels|Loading weights" | tee "outputs/vlm/$1.log"
    $EVAL_PY tools/eval_untrimmed.py --pred-dir "outputs/vlm/$1/val" --frame-dir $FD 2>&1 | tail -40 >> "outputs/vlm/$1.log"
    $EVAL_PY tools/event_confusion.py --pred-dir "outputs/vlm/$1/val" --frame-dir $FD --threshold 0 2>&1 | tail -25 >> "outputs/vlm/$1.log"
}
mkdir -p outputs/vlm
( run qwen3vl8b "$M/Qwen__Qwen3-VL-8B-Instruct" 3; run surgvidlm "$M/jwang01__surgvidlm/surgvidlm-stage2" 3 ) &
( run lingshu7b "$M/lingshu-medical-mllm__Lingshu-7B" 2 ) &
wait
echo ALL_DONE
