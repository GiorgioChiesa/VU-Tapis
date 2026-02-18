#!/usr/bin/env python3
"""
GraSP Dataset Analysis Script.
Comprehensive analysis of surgical steps and phases.

Usage:
    python analize_dataset.py
    python analize_dataset.py --output-dir /custom/path
    python analize_dataset.py --data-dir /custom/data --output-dir /custom/results
"""

import sys
import os
from pathlib import Path

# Get the analysis directory
script_dir = Path(__file__).parent
analysis_dir = script_dir / 'analysis'

# Add analysis modules to path
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

# Import and run main
from main_analysis import main

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
