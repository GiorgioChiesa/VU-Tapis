# AGENTS.md - VU-Tapis Project Conventions

This file highlights high-signal operational details and process quirks often missed by agents.

## 🏃‍♂️ Execution Flow (The Core Task)
The primary entry point for training is running `run_files/orsi_long.sh`. This script orchestrates the full pipeline lifecycle (datasets processing, model building, training, and evaluation).

### Why `run_files/orsi_long.sh` matters
- It constructs the Orsi patient split used by default: 43 train, 6 val, 1 test.
- It sets `CONFIG_PATH=configs/Orsi/$ARCH/TAPIS_ALL.yaml` and passes the config to `tools/run_net.py`.
- It overrides dataset options via `ENDOVIS_DATASET.*` values and exposes the main data roots:
  - `ENDOVIS_DATASET.FRAME_DIR` -> `/data/orsi_tensors`
  - `ENDOVIS_DATASET.FRAME_LIST_DIR` -> `/data/coco`
  - `ENDOVIS_DATASET.ANNOTATION_DIR` -> `/data/coco`
  - `ENDOVIS_DATASET.VAL_COCO_ANNS` -> `/data/coco/all_merged.json`
- It exports `PYTHONPATH` for `tapis/` and `region_proposals/`, which are required by the training launcher.

### Launcher and training flow
- `tools/run_net.py` is the repository launcher. It changes the working directory to repo root, loads config via `tapis.config.defaults`, and calls `launch_job()`.
- If `cfg.FEATURES.USE_RPN` is enabled, `tools/run_net.py` additionally loads Detectron2 + Mask2Former config.
- The actual training/evaluation logic is implemented in `tools/train_net.py`.

### Command Pitfalls / Required Steps:
1. **Dataset Processing:** The full pipeline depends on `run_files/orsi_long.sh` handling the data preparation (e.g., event collapsing, ADD_IDLE frame addition) before training starts.
2. **Training Overrides (Crucial):** When manually running `tools/run_net.py`, remember that the necessary batch and iteration count overrides are:
   `TRAIN.BATCH_SIZE 12` and `VAL.BATCH_SIZE 24`.
3. **Config Structure:** Use spaces, not colons, for overrides (e.g., `TRAIN.BATCH_SIZE 12` ✅, not `TRAIN.BATCH_SIZE=12` ❌).
4. **Model/Task Separation:**
    *   **Frame Tasks (Steps/Phases):** Use global classification.
    *   **Region Tasks (Instruments):** Use RPN + dedicated region head.
5. **Data Paths:** Global data paths are `/data/orsi_tensors` (Frames) and `/data/coco` (Lists).

## ⚠️ Gotchas & Quirks
*   **State Management:** The Memory Bank **must** be reset between evaluating different videos/patients (`tools/train_net.py:439-452`).
*   **OOM Fix:** CUDA Out-of-Memory errors usually require reducing `VAL.BATCH_SIZE` (e.g., reducing 24 to 16).
*   **Temporal Reasoning:** TAPISWithMemory requires explicit reset logic between video evaluations.

## 🧩 Architecture Notes
*   **Paths:** The `tapis/` and `region_proposals/` packages are automatically added to `PYTHONPATH` during execution (`run_files/*.sh` and `orsi.py:1-16`).
*   **Checkpoint:** Best model checkpoints are saved in `.pyth` format.

*(Last updated: May 2026)*
