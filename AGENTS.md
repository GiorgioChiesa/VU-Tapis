# Project Identity
- **Thesis**: Surgical events classification (steps 33 + phases 15) from endoscopic video (RARP dataset).
- **Three frameworks co-exist**:
  - `src/AI-loop/reset/` — ReSET (PyTorch, this thesis work)
  - `src/VU-Tapis/` — Git submodule (fork of upstream TAPIS, `container` branch)
  - `src/TAPIS/` — Reference copy of upstream TAPIS (BCV-Uniandes/GraSP, not a submodule)
- **Deps**: `uv` package manager, `poe` task runner. Never `pip install`.

# Developer Commands
```bash
# Train ReSET (run from repo root, --config path is relative)
python -m src.AI-loop.reset.train --config src/AI-loop/configs/reset/vit_small.yaml --name my_exp
# CLI overrides: --lr, --batch_size, --max_epochs, --n_splits, --device

# Streaming test
python -m src.AI-loop.reset.test_streaming --checkpoint outputs/.../checkpoints/best_mean.pth

# Lint, test, docs
poe lint           # pre-commit: ruff check --fix-only → ruff format → mypy → cz check (commit-msg)
poe test           # coverage run (pytest) → coverage report → coverage xml
poe docs --serve   # mkdocs serve
poe notebook       # jupyter lab --ip=0.0.0.0 --no-browser
poe check_gpu      # torch.cuda.is_available()

# Deps
uv add <pkg>                   # runtime
uv add --dev <pkg>             # dev
uv sync --upgrade              # upgrade all

# Commit (always use this, never git commit -m)
uv run cz commit
```

# Submodules
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Mantyx-Research/thesis_GiorgioChiesa.git

# Update if already cloned without --recurse-submodules
git submodule update --init --recursive
```

# Architecture (ReSET)
Two **exclusive** modes selected by `frame_encoder` config key (`src/AI-loop/configs/reset/*.yaml`):
1. **Frame encoder + transformer**: Per-frame timm/HF backbone → `TemporalPositionalEncoding` → `nn.TransformerEncoder` → center-frame head
2. **3D video backbone** (`mvit_v2_s`): `torchvision.models.video.mvit_v2_s` processes whole clip at once

Wandb logging IS implemented (via `WandbConfig.enable` in config YAML).

# Data
- Frames: `/data/orsi_tensors/{RARPXX}/Video_1fps/{frame_id:06d}.pt` (1 fps)
- Labels: `/data/coco/{RARPXX}.csv` or `{RARPXX}_coco.json`
- 50 patients (`RARP01`–`RARP65` with gaps)

# Gotchas
- `train.py` / `test_streaming.py` modify `sys.path` — run them via `python -m src.AI-loop.reset.train` from repo root
- **No mypy config** exists — mypy in pre-commit runs with default (loose) settings
- `src/giorgiochiesa/` is an empty placeholder package (`__init__.py` only)
- Training artifacts (*.pt, *.pth, wandb/, outputs/) are gitignored
- VU-Tapis config overrides use spaces, not `=`: `TRAIN.BATCH_SIZE 12` ✓
- pretrained SegNeXt weights: `bash src/AI-loop/reset/download_pretrained_weights.sh`
