"""
Centralized configuration for GraSP dataset analysis.
All paths and parameters can be overridden via CLI arguments.
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATHS
# ============================================================================

# Root data directory
DATA_ROOT = '/scratch/Video_Understanding/GraSP/TAPIS/data/GraSP'

# Output directory (analysis results)
ANALYSIS_ROOT = os.path.join("/scratch/Video_Understanding/GraSP/TAPIS", 'dataset_analisis')

# ============================================================================
# INPUT DATA CONFIGURATION
# ============================================================================

INPUT_CONFIG = {
    'annotations': {
        'train': os.path.join(DATA_ROOT, 'annotations/grasp_long-term_train.json'),
        'test': os.path.join(DATA_ROOT, 'annotations/grasp_long-term_test.json'),
    },
    'frame_lists': {
        'train': os.path.join(DATA_ROOT, 'frame_lists/train.csv'),
        'test': os.path.join(DATA_ROOT, 'frame_lists/test.csv'),
    },
    'frames_dir': os.path.join(DATA_ROOT, 'frames'),
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    'base_dir': ANALYSIS_ROOT,
    'tables_dir': os.path.join(ANALYSIS_ROOT, 'tables'),
    'plots_dir': os.path.join(ANALYSIS_ROOT, 'plots'),
    'reports_dir': os.path.join(ANALYSIS_ROOT, 'reports'),
    'logs_dir': os.path.join(ANALYSIS_ROOT, 'logs'),
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'dpi': 300,
    'figure_size': (14, 8),
    'small_figure_size': (10, 6),
    'heatmap_size': (16, 12),
    'font_size': 10,
    'title_size': 14,
    'label_size': 11,
    'color_palette': 'tab20',  # Changed from 'husl' (deprecated in newer matplotlib)
    'save_formats': ['png'],  # Could add 'pdf' here
}

# ============================================================================
# STEP CATEGORIES (21 total, IDs 0-20)
# ============================================================================

STEP_CATEGORIES = {
    0: 'Idle',
    1: 'Id_Illiac_Vein_Artery',
    2: 'Dissection_Illiac_Lymph_Nodes',
    3: 'Dissection_Obturator_Lymph_Nodes',
    4: 'Pack_Lymph_Nodes',
    5: 'Prevessical_Dissection',
    6: 'Ligation_Dorsal_Venous_Complex',
    7: 'Prostate_Dissection',
    8: 'Seminal_Vessicle_Dissection',
    9: 'Denon_Dissection',
    10: 'Cut_Prostate',
    11: 'Hold_Prostate',
    12: 'Pack_Prostate',
    13: 'Pass_Suture_Urethra',
    14: 'Pass_Suture_Neck',
    15: 'Pull_Suture',
    16: 'Tie_Suture',
    17: 'Suction',
    18: 'Cut',
    19: 'Cut_Bladder',
    20: 'Clip_Pedicles',
}

# ============================================================================
# PHASE CATEGORIES (11 total, IDs 0-10)
# ============================================================================

PHASE_CATEGORIES = {
    0: 'Idle',
    1: 'LPIL',  # Left pelvic isolated lymphadenectomy
    2: 'RPIL',  # Right pelvic isolated lymphadenectomy
    3: 'Retzius_Space',
    4: 'Dorsal_Venous_Complex',
    5: 'Id_Bladder_Neck',
    6: 'Seminal_Vesicles',
    7: 'Denonvilliers_Fascia',
    8: 'Pedicle_Control',
    9: 'Severing_Prostate_Urethra',
    10: 'Bladder_Neck_Rec',
}

# ============================================================================
# STEP TAXONOMY & CLASSIFICATION
# ============================================================================

STEP_TAXONOMY = {
    'idle': [0],
    'identification': [1],
    'dissection': [2, 3, 5, 7, 8, 9],
    'packing': [4, 12],
    'ligation': [6],
    'cutting': [10, 18, 19],
    'holding_pulling': [11, 15],
    'suturing': [13, 14, 16],
    'instruments': [17, 20],
}

# ============================================================================
# STEP-PHASE MAPPING
# ============================================================================

STEP_PHASE_MAPPING = {
    # Format: step_id: phase_id
    # This can be extracted/inferred from the data during analysis
    # For now, we'll compute this dynamically from annotations
}

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

ANALYSIS_CONFIG = {
    'analyze_train': True,
    'analyze_test': True,
    'analyze_phases': True,
    'analyze_steps': True,
    'compute_temporal_analysis': True,
    'compute_statistics': True,
    'verbose': True,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(ANALYSIS_ROOT, 'logs/analysis.log'),
}

# ============================================================================
# VALIDATION & ERROR HANDLING
# ============================================================================

VALIDATION_CONFIG = {
    'check_file_existence': True,
    'validate_json_structure': True,
    'validate_data_integrity': True,
    'skip_missing_frames': True,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_step_name(step_id: int) -> str:
    """Get step name from ID."""
    return STEP_CATEGORIES.get(step_id, f'Unknown_Step_{step_id}')


def get_phase_name(phase_id: int) -> str:
    """Get phase name from ID."""
    return PHASE_CATEGORIES.get(phase_id, f'Unknown_Phase_{phase_id}')


def get_step_class(step_id: int) -> str:
    """Get step classification (taxonomy category)."""
    for category, steps in STEP_TAXONOMY.items():
        if step_id in steps:
            return category
    return 'unknown'


def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    for dir_path in OUTPUT_CONFIG.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # Quick test
    print("Configuration loaded successfully")
    print(f"Data root: {DATA_ROOT}")
    print(f"Analysis root: {ANALYSIS_ROOT}")
    print(f"Total steps: {len(STEP_CATEGORIES)}")
    print(f"Total phases: {len(PHASE_CATEGORIES)}")
    ensure_output_dirs()
    print("Output directories created")
