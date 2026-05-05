# VU-TAPIS: Transformers for Actions, Phases, Steps and Instrument Segmentation

<div align="center">
  <img src="../Images/TAPIS.jpg"/>
</div><br/>

VU-TAPIS is an implementation of the TAPIS (Transformers for Actions, Phases, Steps, and Instrument Segmentation) model, adapted for the Orsi dataset. It performs multi-task learning on endoscopic surgical videos to classify surgical steps (33 classes), phases (15 classes), and optionally instruments via instance segmentation.

The model utilizes a generalized architecture with:
- **Primary Backbone**: MViT (MultiScale Vision Transformer)
- **Alternative Backbones**: SlowFast, VideoSwinTransformer, LEMON
- **Task Types**: Frame tasks (global classification), Region tasks (RPN + region head)

## Project Overview

This repository contains the VU-TAPIS implementation, focusing on the Orsi dataset for surgical video analysis. It includes training scripts, configuration files, and evaluation tools for multi-task surgical scene understanding.

## Installation

Please follow these steps to set up VU-TAPIS:

```sh
# Create conda environment
conda create --name vu-tapis python=3.8 -y
conda activate vu-tapis

# Install PyTorch (adjust for your CUDA version)
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Clone and setup
git clone <this-repo>
cd VU-Tapis
pip install -r requirements.txt

# Install additional dependencies
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/facebookresearch/fairscale'
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Run setup script
bash install.sh
```

## Data Preparation

The VU-TAPIS project uses the Orsi dataset. Ensure you have access to:
- Video frames: `/data/orsi_tensors/{patient}/Video_1fps/*.pt`
- Annotations: `/data/coco/` (COCO format JSON files)
- Labels: `{patient}/Label/{patient}_all_labels.csv`

The expected directory structure is:
```
data/
├── orsi_tensors/
│   ├── RARP01/
│   │   └── Video_1fps/
│   │       ├── 000000.pt
│   │       └── ...
│   └── ...
└── coco/
    ├── all_merged.json
    ├── RARP01_coco.json
    └── ...
```

## Running the Code

### Quick Start

For the main multi-task training (steps + phases):

```bash
bash run_files/orsi_long.sh
```

This runs training with default parameters, using dynamic patient splits (43 train, 7 val, 1 test).

### Available Tasks

| Task | Classes | Config | Run Script |
|------|---------|--------|------------|
| Steps | 33 | `configs/Orsi/TAPIS/TAPIS_STEPS.yaml` | `orsi_steps.sh` |
| Phases | 15 | `configs/Orsi/TAPIS/TAPIS_PHASES.yaml` | `orsi_phases.sh` |
| Steps + Phases | 33 + 15 | `configs/Orsi/TAPIS/TAPIS_LONG.yaml` | `orsi_long.sh` |
| With Memory | - | - | `orsi_memory.sh` |
| LEMON Backbone | - | - | `orsi_lemon.sh` |

### Custom Training

Modify run scripts or use direct command:

```bash
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_LONG.yaml \
    TRAIN.ENABLE True VAL.ENABLE True \
    TRAIN.BATCH_SIZE 12 VAL.BATCH_SIZE 24 \
    SOLVER.MAX_ITER 7000 \
    OUTPUT_DIR outputs/orsi/custom_run
```

### Key Configuration Options

- **Data**: `DATA.NUM_FRAMES` (clip length), `DATA.SAMPLING_RATE`
- **Training**: `TRAIN.BATCH_SIZE`, `TRAIN.ACCUM_STEPS`, `SOLVER.MAX_ITER`
- **Tasks**: `TASKS.TASKS` (list of tasks), `TASKS.NUM_CLASSES`
- **Dataset**: `ENDOVIS_DATASET.ORSI_ROOT_DIR`, `ENDOVIS_DATASET.ADD_IDLE`

## Output Structure

Training outputs are saved to:
```
outputs/orsi/{TASK}/{NAME}/totale/
├── checkpoint_best_mean.pyth      # Best overall mAP
├── checkpoint_best_{task}.pyth    # Per-task best
├── checkpoints/                   # Epoch checkpoints
├── stdout.log                     # Training logs
├── {train,val}_data.csv          # Processed datasets
├── {train,val}_confusion_matrix_{task}.png
└── distributions/                 # Class distributions
```

## Evaluation

The model uses mAP (mean Average Precision) as the primary metric, computed via `SurgeryMeter`.

To evaluate a trained model:

```bash
python tapis/evaluate.py \
    --coco_anns_path /data/coco/all_merged.json \
    --pred_path outputs/orsi/LONG/run1/predictions.json \
    --output_path outputs/orsi/LONG/run1/eval_results \
    --tasks steps phases \
    --metrics mAP
```

## Common Issues

1. **CUDA OOM**: Reduce `VAL.BATCH_SIZE` or `TRAIN.BATCH_SIZE`
2. **Missing data**: Verify `/data/orsi_tensors` and `/data/coco` exist
3. **NaN loss**: Check data validity, learning rate, gradient clipping
4. **Config overrides**: Use spaces: `TRAIN.BATCH_SIZE 12` (not `=`)

## Project Structure

- `tapis/`: Core TAPIS implementation
  - `models/`: Model architectures (MViT, SlowFast, etc.)
  - `datasets/`: Data loaders (Orsi dataset)
  - `utils/`: Utilities for training, metrics, etc.
- `tools/`: Training and inference scripts
- `configs/`: YAML configuration files
- `run_files/`: Bash scripts for different experiments
- `region_proposals/`: Instrument segmentation baseline

## Contributing

This is the VU-TAPIS adaptation. For issues related to the original TAPIS model, refer to the [BCV-Uniandes/GraSP](https://github.com/BCV-Uniandes/GraSP) repository.

## License

[Add license information if available]
