# TAPIS Function Pathway

## Execution Flow (from run_files/orsi_steps.sh)

### Layer 1: Shell Entry Point (`run_files/orsi_steps.sh`)
- **Purpose**: Configures data paths and runs experiment
- **Key actions**:
  - Sets patient splits (43 train, 0 val, 7 test) from `all_patients` array
  - Sets `CONFIG_PATH`, `OUTPUT_DIR`, data directories
  - Exports PYTHONPATH for `tapis` and `region_proposals`
  - Executes `python tools/run_net.py` with config overrides

### Layer 2: Main Entry (`tools/run_net.py`)
- **Purpose**: Wrapper that parses args and invokes training
- **Key actions**:
  1. `parse_args()` - parses command line arguments
  2. `load_config(args)` - loads YAML config
  3. `assert_and_infer_cfg(cfg)` - validates and prepares config
  4. `launch_job(cfg, func=train)` - initiates training process

### Layer 3: Training Logic (`tools/train_net.py`)
- **`train(cfg)`** - Main orchestrator (line 525)
  - Initializes distributed training
  - Sets random seed, configures cudnn
  - Builds model via `build_model(cfg)`
  - Constructs optimizer via `optim.construct_optimizer()`
  - Creates GradScaler for mixed precision
  - Loads checkpoint (resume training)
  - Creates loaders: `loader.construct_loader(cfg, "train")`, `construct_loader(cfg, "val")`
  - Training loop: epochs → `train_epoch()` → `eval_epoch()`

- **`train_epoch()`** (line 163)
  - Sets model to train mode
  - Creates loss functions from `losses.get_loss_func()`
  - Learning rate scheduling: `optim.get_epoch_lr()`, `optim.set_lr()`
  - Forward pass: `model(inputs)`
  - Loss computation: weighted combination of task losses
  - Backward pass: `scaler.scale().backward()`
  - Optimizer step: `scaler.step()`, `scaler.update()`
  - Metrics update: `train_meter.update_stats()`

- **`eval_epoch()`** (line 401)
  - Sets model to eval mode
  - Inference: `model(inputs)`
  - Metrics computation: mAP, confusion matrix
  - Returns task_map, mean_map, out_files, early_stop

### Layer 4: Model Building (`tapis/models/build.py`)
- **`build_model(cfg, gpu_id=None)`** (line 18)
  - Looks up MODEL_REGISTRY by `cfg.MODEL.MODEL_NAME` (e.g., "MViT")
  - Instantiates model: `MODEL_REGISTRY.get(name)(cfg)`
  - Transfers to GPU: `model.cuda(device=cur_device)`
  - Wraps in DDP if multi-GPU: `DistributedDataParallel()`

### Layer 5: Dataset Building (`tapis/datasets/build.py`)
- **`build_dataset(dataset_name, cfg, split)`** (line 15)
  - Capitalizes name, looks up in DATASET_REGISTRY
  - Registered datasets: "orsi", "grasp", "endovis_2017", "endovis_2018"

### Layer 6: Data Loading (`tapis/datasets/loader.py`)
- **`construct_loader(cfg, split)`** (line 68)
  - Builds dataset: `build_dataset()`
  - Creates DataLoader with appropriate batch_size, shuffle, sampler

---

## Registry Pattern

| Registry | Registered Names | Build Function |
|----------|-----------------|----------------|
| MODEL_REGISTRY | MViT, SlowFast, etc. | `build_model()` |
| DATASET_REGISTRY | Orsi, Grasp, EndoVis2017, EndoVis2018 | `build_dataset()` |
| LOSS_REGISTRY | cross_entropy, focal_loss, bce | `losses.get_loss_func()` |

---

## Key Config Groups (from `tapis/config/defaults.py`)

| Config Group | Purpose |
|------------|---------|
| TRAIN | ENABLE, BATCH_SIZE, CHECKPOINT_FILE_PATH, FREEZE_ENCODER |
| TEST | ENABLE, BATCH_SIZE |
| DATA | NUM_FRAMES, SAMPLING_RATE, SEQ_MODE, CROP_SIZE |
| MVIT | EMBED_DIM, DEPTH, NUM_HEADS, POOL_Q_STRIDE |
| TASKS | TASKS (steps/phases/actions), NUM_CLASSES, LOSS_WEIGHTS |
| SOLVER | BASE_LR, MAX_EPOCH, LR_POLICY, WEIGHT_DECAY |
| AUG | ENABLE, COLOR_JITTER, AA_TYPE |

---

## Data Dependencies (from orsi_steps.sh)

```
FRAME_DIR: /home/gchie/workspace/nas_private/data/orsi
FRAME_LIST: /scratch/.../frame_lists/*.csv
ANNOT_DIR: /scratch/.../annotations/*.json
COCO_ANN_PATH: /scratch/.../grasp_long-term_*.json
CHECKPOINT: /scratch/.../pretrained_models/fold1/*.pyth
```