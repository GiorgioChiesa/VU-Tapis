# TAPIS Training Pipeline Guide (For AI Agents)

This document outlines the standard, end-to-end workflow for training the TAPIS model, synthesizing information from `run_files/orsi_long.sh`, `tools/run_net.py`, and the dataset definitions.

## 🚀 Primary Execution Command

The standard execution command for a long-term training run is:
```bash
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_LONG.yaml \
    TRAIN.ENABLE True TEST.ENABLE True OUTPUT_DIR outputs/orsi/LONG/run1
```

## ⚙️ Training Workflow (`tools/train_net.py`)

The training process follows a structured flow:

1.  **Initialization:** The process begins with distributed training initialization (`du.init_distributed_training(cfg)`).
2.  **Model Setup:** The model is built using `build_model(cfg)`. The system supports optional freezing of the encoder (`TRAIN.FREEZE_ENCODER`).
3.  **Optimization:** An optimizer (Adam or SGD) is constructed via `optim.construct_optimizer(model, cfg)`.
4.  **Precision:** Mixed precision training is supported and controlled by `cfg.TRAIN.MIXED_PRECISION`.
5.  **Checkpointing:** Training can resume using `cu.load_train_checkpoint()`.
6.  **Epoch Loop:** The core training loop executes `train_epoch()` and evaluation via `eval_epoch()`.

## 💾 Dataset Handling (`tapis/datasets/orsi.py`)

The dataset preparation is critical and involves several steps:

*   **Data Paths:**
    *   Frames: `{ORSI_ROOT_DIR}/{patient}/Video_1fps/*.pt`
    *   Labels: `{patient}/Label/{patient}_all_labels.csv`
*   **Key Feature: ADD_IDLE:** The system includes an `ADD_IDLE` feature (default: 10 frames). This function selects and distributes idle frames across patient videos, while filtering out idle frames that are too close to non-idle events.
*   **Output:** The processed data is saved to `OUTPUT_DIR/{split}_data.csv`.

## 🗄️ Configuration Management

All training parameters are managed via YAML configuration files.

*   **Primary Config:** Use `configs/Orsi/TAPIS/TAPIS_LONG.yaml` as the base.
*   **Key Parameters:**
    *   `DATA.NUM_FRAMES`: Defines the clip length.
    *   `TRAIN.BATCH_SIZE`: The batch size for training.
    *   `TASKS.TASKS`: Must be set to `["steps", "phases"]`.
    *   `TASKS.NUM_CLASSES`: Must be set to `[33, 15]` (Steps, Phases).

## 📈 Evaluation Metrics

Evaluation is performed in `tools/train_net.py:eval_epoch()`. The primary metric calculated is **mAP** using the `SurgeryMeter` class.

---
*This guide is based on the current codebase structure and should be cross-referenced with the official documentation in the `docs/` directory.*