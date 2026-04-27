# AGENTS.md - TAPIS Project Context

## Quick Start

```bash
# LONG task (steps + phases)
bash run_files/orsi_long.sh

# STEPS task only
bash run_files/orsi_steps.sh

# Direct execution
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_LONG.yaml \
    TRAIN.ENABLE True TEST.ENABLE True OUTPUT_DIR outputs/orsi/LONG/run1
```

## Execution Flow

```
run_files/*.sh → tools/run_net.py → train_net.py:train() (525)
  ├── build_model()        # tapis/models/build.py
  ├── construct_loader()   # tapis/datasets/loader.py
  ├── train_epoch()        # train_net.py:163
  └── eval_epoch()         # train_net.py:400
```

## Architecture Layers

| Layer | Entry Point | Description |
|-------|-------------|-------------|
| Shell | `run_files/*.sh` | Generates patient splits, configures paths |
| Main | `tools/run_net.py` | Parses args, loads config, calls train |
| Training | `tools/train_net.py` | train() (525), train_epoch() (163), eval_epoch() (400) |
| Model | `tapis/models/build.py` | MODEL_REGISTRY.get(name)(cfg) |
| Dataset | `tapis/datasets/build.py` | DATASET_REGISTRY.get(name)(cfg, split) |
| Loader | `tapis/datasets/loader.py` | construct_loader(cfg, split) |

## Config Files

| Config | Task | Classes |
|--------|------|---------|
| `TAPIS_STEPS.yaml` | Steps | 32 |
| `TAPIS_PHASES.yaml` | Phases | 14 |
| `TAPIS_LONG.yaml` | Steps + Phases | 32 + 14 |
| `TAPIS_SHORT.yaml` | Short-term | - |
| `TAPIS_ACTIONS.yaml` | Actions | - |
| `TAPIS_INSTRUMENTS.yaml` | Instruments | - |

## Data Paths (Local Setup)

```bash
FRAME_DIR="/data/orsi_tensors"
FRAME_LIST="/data/coco"
ANNOT_DIR="/data/coco"
COCO_ANN_PATH="/data/coco/all_merged.json"
CHECKPOINT="data/pretrained_models/fold1/LONG.pyth"
```

## Environment Setup

```bash
export PYTHONPATH=$TAPIS_DIR/tapis:$PYTHONPATH
export PYTHONPATH=$TAPIS_DIR/region_proposals:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # Check GPU availability first
```

## Key Configs

```yaml
DATA.NUM_FRAMES: 16          # LONG task uses 32 frames (NUM_FRAMES * SAMPLING_RATE)
DATA.SAMPLING_RATE: 1        # Increase to 2 for longer temporal context
TRAIN.ACCUM_STEPS: 10        # Effective batch = BATCH_SIZE * ACCUM_STEPS
TRAIN.BATCH_SIZE: 12         # Adjust based on GPU memory
TEST.BATCH_SIZE: 24          # May need reduction to avoid OOM
SOLVER.BASE_LR: 0.0001       # STEPS uses 5e-5
SOLVER.MAX_EPOCH: 50
SOLVER.EARLY_STOP_ep_th: [5, -1.0]  # Stop after 5 epochs no improvement
SOLVER.LR_POLICY: cosine
TASKS.TASKS: ["steps", "phases"]
TASKS.WEIGHT_LOSS_BY_CLASS: [False, False]  # Must be False if CSV files don't match data
```

## Patient Splits

```
all_patients: RARP01-RARP65 (50 total, some gaps)
n_train: 43  n_val: 0  n_test: 7
```

## Common Issues & Fixes

1. **CUDA OOM**: Reduce TEST.BATCH_SIZE (60→24) or TRAIN.BATCH_SIZE
2. **Missing data**: Verify FRAME_DIR exists at `/data/orsi_tensors`
3. **GPU unavailable**: Check `nvidia-smi` and set correct CUDA_VISIBLE_DEVICES
4. **Weight loss error**: `tapis/datasets/orsi.py:138` - `df` undefined bug. Use `self.counter[task]['total_count'].sum()`
5. **WEIGHT_LOSS_BY_CLASS**: Files must match actual class distribution or set to `[False, False]`

## Testing

```bash
pytest tests/test_dataset_orsi.py -v
```

## Output Locations

```
outputs/orsi/LONG/Container/totale/
├── checkpoints/     # checkpoint_epoch_*.pyth
├── stdout.log      # Training logs
└── metrics/        # Evaluation results
```