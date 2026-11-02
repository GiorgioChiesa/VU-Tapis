#!/usr/bin/env python3
"""
Same as tools/run_net.py for frame tasks, but loads `tools/train_net.py` by path.

`tapis/datasets/orsi.py` prepends `region_proposals/` to sys.path when imported,
and that folder has its own `train_net.py` (which needs the compiled detectron2
extension). On the host env, `from train_net import train` in run_net.py then picks
the wrong module; loading by path avoids it. RPN/region tasks are not supported here.
"""

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
for p in (REPO / "detectron2", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tapis.config.defaults import assert_and_infer_cfg  # noqa: E402
from tapis.utils.misc import launch_job  # noqa: E402
from tapis.utils.parser import load_config, parse_args  # noqa: E402

spec = importlib.util.spec_from_file_location("train_net", REPO / "tools" / "train_net.py")
train_net = importlib.util.module_from_spec(spec)
# tools/train_net.py itself runs `from train_net import train` at module level:
# registering it first makes that import resolve to this very module.
sys.modules["train_net"] = train_net
spec.loader.exec_module(train_net)


def main():
    args = parse_args()
    cfg = assert_and_infer_cfg(load_config(args))
    if cfg.FEATURES.USE_RPN:
        raise NotImplementedError("use tools/run_net.py inside the container for RPN configs")
    if cfg.TRAIN.ENABLE or cfg.VAL.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_net.train)


if __name__ == "__main__":
    main()
