# AGENTS.md - TAPIS Project Context

## Quick Start

```bash
# LONG task (steps + phases) - uses YAML defaults overridden by command line
bash run_files/orsi_long.sh

# Direct execution with overrides
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_LONG.yaml \
    TRAIN.ENABLE True VAL.ENABLE True OUTPUT_DIR outputs/orsi/LONG/run1
```

## Execution Flow

```
run_files/*.sh → tools/run_net.py → tools/train_net.py:train() (line 525)
  ├── build_model()        # tapis/models/build.py
  ├── construct_loader()  # tapis/datasets/loader.py
  ├── train_epoch()       # tools/train_net.py:163
  └── eval_epoch()        # tools/train_net.py:393
```

## Run Script: orsi_long.sh

### Patient Splits (lines 19-54)
- **Train**: 43 patients (dynamic via `n_train=43`)
- **Val**: 7 patients (dynamic via `n_val=7`)
- **Test**: 0 patients (`n_test=0`)
- Splits generated from `all_patients` array; uses `VAL_FOLDS/VAL_FOLDS_STR` (not TEST)

### Key Config Lines (60-100)
- `NAME="Idlex10"` (output subdirectory name)
- PYTHONPATH includes `tapis` and `region_proposals` (lines 76-77)
- GPU device via `GPUIDS` (default: 1)
- Data paths: `FRAME_DIR=/data/orsi_tensors`, `FRAME_LIST=/data/coco`
- **Command-line overrides** (lines 82-100): `TRAIN.ACCUM_STEPS 10`, `TRAIN.BATCH_SIZE 12`, `VAL.BATCH_SIZE 24`, `SOLVER.MAX_ITER 7000`

## Training & Validation

### tools/train_net.py:train() (lines 525-714)
- Distributed training init: `du.init_distributed_training(cfg)`
- Random seed from `cfg.RNG_SEED`
- Model: `build_model(cfg)`, optional `TRAIN.FREEZE_ENCODER`
- Optimizer: `optim.construct_optimizer(model, cfg)` (Adam or SGD)
- Mixed precision: `torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)`
- Checkpoint: Resume via `cu.load_train_checkpoint()`
- Early stopping: `SOLVER.EARLY_STOP_ep_th: [5, -1.0]` (5 epochs, no mAP threshold)

### tools/train_net.py:eval_epoch() (lines 393-489)
- Model eval mode: `model.eval()`
- Memory bank reset between videos (lines 439-452)
- Per-task predictions: steps, phases
- Metrics: mAP via `SurgeryMeter`

## Dataset (tapis/datasets/orsi.py)

### Data Paths
- Frames: `{ORSI_ROOT_DIR}/{patient}/Video_1fps/*.pt`
- Labels: `{patient}/Label/{patient}_all_labels.csv`
- CSV columns: `patient_id, patient_name, frame_id, frame_path, second, event_id, event_name, phase_id, phase_name`

### Key Processing Steps
- **Event collapsing** (line 275): Maps related events (e.g., left/right variants → same event)
- **ADD_IDLE** (lines 380-467): Adds up to N idle frames per patient, filtered to avoid non-idle events within 30 frames
- **Label selection** (line 469): Clip label = nearest non-Idle label to central frame; Idle used only if no non-idle available
- **Event duration**: Labels last 3 seconds (3 frames at 1 fps), only first frame stored

### Output CSV (line 273)
- Saves to: `OUTPUT_DIR/{split}_data.csv`
- Contains dataframe after filtering, collapse, and add_idle processing

## Config Files

| Config | Task | Classes |
|--------|------|---------|
| `TAPIS_STEPS.yaml` | Steps | 33 |
| `TAPIS_PHASES.yaml` | Phases | 15 |
| `TAPIS_LONG.yaml` | Steps + Phases | 33 + 15 |

NUM_CLASSES includes background class.

## Key Configs (configs/Orsi/TAPIS/TAPIS_LONG.yaml)

```yaml
DATA.NUM_FRAMES: 16          # Clip length = NUM_FRAMES * SAMPLING_RATE
DATA.SAMPLING_RATE: 1
TRAIN.ACCUM_STEPS: 1         # Effective batch = BATCH * ACCUM (overridden to 10 in orsi_long.sh)
TRAIN.BATCH_SIZE: 16         # Overridden to 12 in orsi_long.sh
VAL.BATCH_SIZE: 20           # Overridden to 24 in orsi_long.sh
SOLVER.BASE_LR: 0.0001
SOLVER.MAX_EPOCH: 50
SOLVER.MAX_ITER: 10000       # Overridden to 7000 in orsi_long.sh
SOLVER.LR_POLICY: cosine
TASKS.TASKS: ["steps", "phases"]
TASKS.NUM_CLASSES: [33, 15]
TASKS.LOSS_FUNC: ["cross_entropy", "cross_entropy"]
TASKS.WEIGHT_LOSS_BY_CLASS: [steps_distribution.csv, phases_distribution.csv]
ENDOVIS_DATASET.ADD_IDLE: 10
ENDOVIS_DATASET.ORSI_ROOT_DIR: /data/orsi_tensors
```

Override YAML defaults via command line: `python -B tools/run_net.py --cfg <yaml> TRAIN.BATCH_SIZE 12 VAL.BATCH_SIZE 24`

## Output Locations

```
outputs/{DATASET}/{TASK}/{NAME}/totale/
├── checkpoint_best_mean.pyth      # Best overall mAP
├── checkpoint_best_{task}.pyth    # Best per-task mAP
├── checkpoints/                   # checkpoint_epoch_*.pyth
├── stdout.log                     # Training logs
├── {train,val}_data.csv          # Processed dataframes
├── {train,val}_confusion_matrix_{task}.png
└── distributions/                # Class distribution CSVs for loss weighting
```

## Common Issues & Fixes

1. **CUDA OOM**: Reduce `VAL.BATCH_SIZE` (20→24 in orsi_long.sh, lower if needed)
2. **Missing data**: Verify `/data/orsi_tensors` and `/data/coco` exist
3. **GPU unavailable**: Check `nvidia-smi`
4. **PYTHONPATH errors**: `tapis/` and `region_proposals/` added automatically in run_files/*.sh and orsi.py
