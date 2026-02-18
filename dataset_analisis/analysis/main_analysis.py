"""
Main Analysis Orchestration Module.
Coordinates the entire analysis pipeline.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add analysis directory to path
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from config.analysis_config import (
    INPUT_CONFIG,
    OUTPUT_CONFIG,
    LOGGING_CONFIG,
    ensure_output_dirs,
)
from core.data_loader import DataLoader, DataLoaderException
from core.step_analyzer import StepAnalyzer
from visualization.plots import StepVisualizer
from visualization.tables import TableExporter


def setup_logging():
    """Setup logging configuration."""
    ensure_output_dirs()

    log_file = LOGGING_CONFIG['log_file']
    log_level = getattr(logging, LOGGING_CONFIG['level'])
    log_format = LOGGING_CONFIG['format']

    # Create logger
    logger = logging.getLogger('GraSP_Analysis')
    logger.setLevel(log_level)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main(args=None):
    """
    Main analysis execution function.

    Args:
        args: Command-line arguments
    """
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 40)
    logger.info("GRASP DATASET ANALYSIS - MAIN EXECUTION")
    logger.info("=" * 40)

    try:
        # ====================================================================
        # PHASE 1: DATA LOADING
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 1: DATA LOADING")
        logger.info("=" * 40)

        data_loader = DataLoader(logger)
        train_df, test_df = data_loader.load_and_prepare()
        data_loader.save_dataframes()
        
        logger.info(f"\nTrain DataFrame shape: {train_df.shape}")
        logger.info(f"Test DataFrame shape: {test_df.shape}")

        # ====================================================================
        # PHASE 2: COMPREHENSIVE ANALYSIS
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 2: COMPREHENSIVE ANALYSIS")
        logger.info("=" * 40)

        analyzer = StepAnalyzer(train_df, test_df, logger)
        analysis_results = analyzer.execute_all_analysis()

        logger.info(f"\nCompleted analysis with {len(analysis_results)} result types")

        # ====================================================================
        # PHASE 3: VISUALIZATION
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 3: VISUALIZATION")
        logger.info("=" * 40)

        visualizer = StepVisualizer(logger)
        visualizer.generate_all_plots(analysis_results)

        # ====================================================================
        # PHASE 4: TABLE EXPORT
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 4: TABLE EXPORT")
        logger.info("=" * 40)

        exporter = TableExporter(logger)
        exporter.export_all(analysis_results)

        # ====================================================================
        # SUMMARY
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 40)
        logger.info(f"\nResults saved in: {OUTPUT_CONFIG['base_dir']}")
        logger.info(f"  - Tables: {OUTPUT_CONFIG['tables_dir']}")
        logger.info(f"  - Plots: {OUTPUT_CONFIG['plots_dir']}")
        logger.info(f"  - Logs: {OUTPUT_CONFIG['logs_dir']}")

        logger.info("\n" + "=" * 40)
        return 0

    except DataLoaderException as e:
        logger.error(f"\nData Loading Error: {e}")
        logger.error("=" * 40)
        return 1

    except Exception as e:
        logger.error(f"\nUnexpected Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 40)
        return 1

    finally:
        logger.info("\nAnalysis session ended\n")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GraSP Dataset Analysis - Comprehensive step and phase analysis'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override data root directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output root directory',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Override config if needed
    if args.data_dir:
        import config.analysis_config as config
        config.DATA_ROOT = args.data_dir
        config.INPUT_CONFIG['annotations']['train'] = f"{args.data_dir}/annotations/grasp_long-term_train.json"
        config.INPUT_CONFIG['annotations']['test'] = f"{args.data_dir}/annotations/grasp_long-term_test.json"

    if args.output_dir:
        import config.analysis_config as config
        config.ANALYSIS_ROOT = args.output_dir
        config.OUTPUT_CONFIG['base_dir'] = args.output_dir
        config.OUTPUT_CONFIG['tables_dir'] = f"{args.output_dir}/tables"
        config.OUTPUT_CONFIG['plots_dir'] = f"{args.output_dir}/plots"
        config.OUTPUT_CONFIG['logs_dir'] = f"{args.output_dir}/logs"

    exit_code = main(args)
    sys.exit(exit_code)
