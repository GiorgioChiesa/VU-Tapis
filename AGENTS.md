# AGENTS.md - TAPIS Project Context

## Quick Start

```bash
# LONG task (steps + phases)
bash run_files/orsi_long.sh

# STEPS task only
bash run_files/orsi_steps.sh

# Direct execution
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_LONG.yaml \
    TRAIN.ENABLE True VAL.ENABLE True OUTPUT_DIR outputs/orsi/LONG/run1
```

## Execution Flow

```
run_files/*.sh → tools/run_net.py → tools/train_net.py:train() (525)
  ├── build_model()        # tapis/models/build.py
  ├── construct_loader()  # tapis/datasets/loader.py
  ├── train_epoch()       # tools/train_net.py:163
  └── eval_epoch()        # tools/train_net.py:393
```

## Run Script: orsi_long.sh

### Patient Splits (lines 19-54)
- **Train**: 43 patients (dynamic via n_train=43)
- **Val**: 7 patients (dynamic via n_val=7)
- **Test**: 0 patients (n_test=0)
- Note: Splits generated from all_patients array, uses VAL_FOLDS/VAL_FOLDS_STR (not TEST)

### Key Config Lines (75-98)
- PYTHONPATH must include `tapis` and `region_proposals` (lines 75-76)
- GPU device: configurable via `GPUIDS` (default: 1)
- Data paths: `FRAME_DIR=/data/orsi_tensors`, `FRAME_LIST=/data/coco`


## Training & Validation

### tools/train_net.py:train() (lines 525-639)
- Distributed training init: `du.init_distributed_training(cfg)`
- Random seed set from `cfg.RNG_SEED`
- Model: `build_model(cfg)`, optional `TRAIN.FREEZE_ENCODER`
- Optimizer: `optim.construct_optimizer(model, cfg)` (Adam or SGD)
- Mixed precision: `torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)`
- Checkpoint: Resume via `cu.load_train_checkpoint()`

### tools/train_net.py:eval_epoch() (lines 393-489)
- Model eval mode: `model.eval()`
- Memory bank reset between videos (lines 442-455)
- Per-task predictions: steps, phases
- Metrics: mAP calculation via `SurgeryMeter`


## Dataset (tapis/datasets/orsi.py)

### Data Paths
- Frames: `{ORSI_ROOT_DIR}/{patient}/Video_1fps/*.pt`
- Labels: `{patient}/Label/{patient}_all_labels.csv`
- CSV columns: patient_id, patient_name, frame_id, frame_path, second, event_id, event_name, phase_id, phase_name

### ADD_IDLE Feature (lines 382-468)
- Config: `ENDOVIS_DATASET.ADD_IDLE` (default: 10 in TAPIS_LONG.yaml)
- Selects Idle frames evenly distributed across patient videos
- Filters out Idle within 30 frames of non-Idle events
- Patient mapping: Uses `patient_name` (string "RARP01") mapped to `patient_id` (int) via regex extraction (line 394)

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
DATA.NUM_FRAMES: 16          # Actual clip length = NUM_FRAMES * SAMPLING_RATE
DATA.SAMPLING_RATE: 1
TRAIN.ACCUM_STEPS: 1         # Effective batch = BATCH * ACCUM
TRAIN.BATCH_SIZE: 16
VAL.BATCH_SIZE: 64             # Override via command line for OOM
SOLVER.BASE_LR: 0.0001
SOLVER.MAX_EPOCH: 50
SOLVER.MAX_ITER: 10000
SOLVER.EARLY_STOP_ep_th: [5, -1.0]
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
├── checkpoints/          # checkpoint_epoch_*.pyth
├── stdout.log            # Training logs
├── metrics/              # Evaluation results and confusion matrices
└── distributions/        # Class distribution CSVs for loss weighting
```


## Common Issues & Fixes

1. **CUDA OOM**: Reduce `VAL.BATCH_SIZE` (64→24 or lower)
2. **Missing data**: Verify `/data/orsi_tensors` and `/data/coco` exist
3. **GPU unavailable**: Check `nvidia-smi`
4. **PYTHONPATH errors**: Ensure `tapis/` and `region_proposals/` are in PYTHONPATH (done in run_files/*.sh)
