# AGENTS.md - TAPIS Project Context

## Execution Flow

```
run_files/orsi_steps.sh
  └── tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_STEPS.yaml
        └── train(cfg)            # tools/train_net.py:525
              ├── build_model(cfg)  # tapis/models/build.py
              ├── loader.construct_loader  # tapis/datasets/loader.py
              ├── train_epoch()     # line 163
              └── eval_epoch()     # line 400
```

## Run Commands

```bash
# From run_files/ (uses bash scripts that configure paths and PYTHONPATH)
bash run_files/orsi_steps.sh         # Orsi STEPS task
bash run_files/orsi_phases.sh        # Orsi PHASES task
bash run_files/grasp_long-term.sh   # GraSP long-term tasks

# Direct execution (use -B to skip .pyc caching)
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_STEPS.yaml \
    TRAIN.ENABLE True TEST.ENABLE True \
    OUTPUT_DIR outputs/orsi/steps/run1
```

## Environment Setup

```bash
# Required PYTHONPATH (set in run_files/*.sh)
export PYTHONPATH=$TAPIS_DIR/tapis:$PYTHONPATH
export PYTHONPATH=$TAPIS_DIR/region_proposals:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1  # set GPU
```

## Architecture Layers

| Layer | Entry Point | Description |
|-------|-----------|-------------|
| Shell | `run_files/*.sh` | Generates patient splits, configures paths |
| Main | `tools/run_net.py` | Parses args, loads config, calls train |
| Training | `tools/train_net.py` | train() (line 525), train_epoch() (line 163), eval_epoch() (line 400) |
| Model | `tapis/models/build.py` | MODEL_REGISTRY.get(name)(cfg) |
| Dataset | `tapis/datasets/build.py` | DATASET_REGISTRY.get(name)(cfg, split) |
| Loader | `tapis/datasets/loader.py` | construct_loader(cfg, split) |

## Config Structure

Key YAML configs in `configs/Orsi/TAPIS/`:
- `TAPIS_STEPS.yaml` - Steps classification (72 classes)
- `TAPIS_PHASES.yaml` - Phases classification
- `TAPIS_LONG.yaml` - Long-term tasks
- `TAPIS_SHORT.yaml` - Short-term tasks
- `TAPIS_ACTIONS.yaml` - Actions classification
- `TAPIS_INSTRUMENTS.yaml` - Instrument segmentation

## Data Dependencies

External data required (paths configured in run scripts):
- Frame images: `FRAME_DIR`
- Frame lists (CSV): `FRAME_LIST`
- Annotations (JSON): `ANNOT_DIR`
- COCO annotations: `COCO_ANN_PATH`
- Pretrained checkpoints: `CHECKPOINT`

## Training Control

```yaml
TRAIN.ENABLE: True/False        # Enable/disable training
TEST.ENABLE: True/False        # Enable/disable evaluation
TRAIN.BATCH_SIZE: 16            # Batch size
SOLVER.MAX_EPOCH: 50           # Max epochs
SOLVER.BASE_LR: 0.0001          # Learning rate
TRAIN.FREEZE_ENCODER: False    # Freeze encoder weights
TRAIN.MIXED_PRECISION: True    # Use FP16
```

## Patient Splits (default from orsi_steps.sh)

```
n_train: 43 patients
n_val: 0 patients
n_test: 7 patients
all_patients: RARP01-RARP65 (50 total, some gaps)
```

## Common Issues

1. **Missing data paths**: Run scripts reference external paths (`/scratch/...`, `/home/gchie/...`)
2. **PYTHONPATH**: Must include both `tapis` and `region_proposals`
3. **CUDA_VISIBLE_DEVICES**: Set GPU IDs before running

## Testing

```bash
# Run specific test
pytest tests/test_dataset_orsi.py -v
```