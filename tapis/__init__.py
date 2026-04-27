#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys
from pathlib import Path

# Add repo root to sys.path to enable 'tapis' module import
_repo_root = Path(__file__).parent.parent
_str_root = str(_repo_root)
if _str_root not in sys.path:
    sys.path.insert(0, _str_root)

# Add other required paths
for _p in [str(_repo_root / "region_proposals"), str(_repo_root / "detectron2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Add tapis as an alias to support imports like 'from tapis.config import ...'
if "tapis" not in sys.modules:
    sys.modules["tapis"] = sys.modules.get("tapis", None)

from tapis.utils.env import setup_environment

setup_environment()
