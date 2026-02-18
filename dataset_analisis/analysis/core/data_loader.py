"""
Data Loader for GraSP Dataset.
Loads JSON annotations and CSV frame lists with validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import pandas as pd

from config.analysis_config import INPUT_CONFIG, VALIDATION_CONFIG, OUTPUT_CONFIG


class DataLoaderException(Exception):
    """Custom exception for data loading errors."""
    pass


class DataLoader:
    """
    Loads and preprocesses GraSP dataset annotations and frame lists.
    """

    def __init__(self, logger=None):
        """
        Initialize DataLoader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.train_data = None
        self.test_data = None
        self.train_df = None
        self.test_df = None
        self.frame_lists = {}

    def load_json_dataset(self, split: str) -> dict:
        """
        Load JSON annotation file.

        Args:
            split: 'train' or 'test'

        Returns:
            Dictionary with annotation data

        Raises:
            DataLoaderException: If file not found or JSON is invalid
        """
        try:
            json_path = INPUT_CONFIG['annotations'][split]

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Annotation file not found: {json_path}")

            self.logger.info(f"Loading {split} annotations from {json_path}")

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Validate structure
            required_keys = ['info', 'images', 'annotations']
            if split == 'train':
                required_keys.extend(['steps_categories', 'phases_categories'])

            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                raise ValueError(f"Missing required keys in JSON: {missing_keys}")

            num_annotations = len(data.get('annotations', []))
            self.logger.info(f"Loaded {num_annotations} {split} annotations")

            return data

        except json.JSONDecodeError as e:
            raise DataLoaderException(f"Invalid JSON format in {split}: {e}")
        except Exception as e:
            raise DataLoaderException(f"Error loading {split} data: {e}")

    def load_frame_list(self, split: str) -> pd.DataFrame:
        """
        Load frame list CSV file.

        Args:
            split: 'train' or 'test'

        Returns:
            DataFrame with frame list data

        Raises:
            DataLoaderException: If file not found or CSV is invalid
        """
        try:
            csv_path = INPUT_CONFIG['frame_lists'][split]

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Frame list not found: {csv_path}")

            self.logger.info(f"Loading {split} frame list from {csv_path}")

            # Load CSV without header (space-separated)
            df = pd.read_csv(
                csv_path,
                sep=' ',
                header=None,
                names=['video_name', 'video_id', 'frame_index', 'frame_path'],
            )

            self.logger.info(f"Loaded {len(df)} frames for {split}")
            return df

        except Exception as e:
            raise DataLoaderException(f"Error loading {split} frame list: {e}")

    def convert_annotations_to_dataframe(
        self, data: dict, split: str
    ) -> pd.DataFrame:
        """
        Convert annotation data to DataFrame.

        Args:
            data: Raw annotation data from JSON
            split: 'train' or 'test'

        Returns:
            DataFrame with annotations
        """
        try:
            self.logger.info(f"Converting {split} annotations to DataFrame")

            # Extract basic annotation info
            annotations = data['annotations']
            images = {img['id']: img for img in data['images']}

            # Build rows for DataFrame
            rows = []
            for ann in annotations:
                image_id = ann['image_id']
                image_info = images.get(image_id, {})

                # Extract video name and frame number from path
                file_name = image_info.get('file_name', '')
                parts = file_name.split('/')
                video_name = parts[0] if parts else ''
                frame_name = parts[1] if len(parts) > 1 else ''

                try:
                    frame_num = int(frame_name.replace('.jpg', ''))
                except (ValueError, IndexError):
                    frame_num = -1

                rows.append({
                    'annotation_id': ann['id'],
                    'image_id': image_id,
                    'image_name': file_name,
                    'video_name': video_name,
                    'frame_num': frame_num,
                    'phase': ann.get('phases', -1),
                    'step': ann.get('steps', -1),
                })

            df = pd.DataFrame(rows)
            self.logger.info(f"Created DataFrame with {len(df)} rows for {split}")

            return df

        except Exception as e:
            raise DataLoaderException(
                f"Error converting {split} annotations to DataFrame: {e}"
            )

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all data and prepare DataFrames.

        Returns:
            Tuple of (train_df, test_df)

        Raises:
            DataLoaderException: If any loading fails
        """
        try:
            # Load train data
            self.logger.info("=" * 60)
            self.logger.info("LOADING TRAIN DATA")
            self.logger.info("=" * 60)
            train_data = self.load_json_dataset('train')
            self.train_df = self.convert_annotations_to_dataframe(train_data, 'train')

            # Load test data
            self.logger.info("=" * 60)
            self.logger.info("LOADING TEST DATA")
            self.logger.info("=" * 60)
            test_data = self.load_json_dataset('test')
            self.test_df = self.convert_annotations_to_dataframe(test_data, 'test')

            # Load frame lists
            self.logger.info("=" * 60)
            self.logger.info("LOADING FRAME LISTS")
            self.logger.info("=" * 60)
            train_frames = self.load_frame_list('train')
            test_frames = self.load_frame_list('test')

            # Store references to raw data
            self.train_data = train_data
            self.test_data = test_data

            # Validation
            if VALIDATION_CONFIG['validate_data_integrity']:
                self._validate_data()

            self.logger.info("=" * 60)
            self.logger.info("DATA LOADING COMPLETE")
            self.logger.info("=" * 60)

            return self.train_df, self.test_df

        except Exception as e:
            self.logger.error(f"Fatal error during data loading: {e}")
            raise

    def _validate_data(self):
        """Validate data integrity."""
        self.logger.info("Validating data integrity...")

        # Check for missing values
        train_nulls = self.train_df.isnull().sum()
        test_nulls = self.test_df.isnull().sum()

        if train_nulls.any():
            self.logger.warning(f"Missing values in train data:\n{train_nulls}")
        if test_nulls.any():
            self.logger.warning(f"Missing values in test data:\n{test_nulls}")

        # Check step and phase ranges
        train_steps = self.train_df['step']
        test_steps = self.test_df['step']

        train_phases = self.train_df['phase']
        test_phases = self.test_df['phase']

        self.logger.info(f"Train steps range: {train_steps.min()} - {train_steps.max()}")
        self.logger.info(f"Test steps range: {test_steps.min()} - {test_steps.max()}")
        self.logger.info(f"Train phases range: {train_phases.min()} - {train_phases.max()}")
        self.logger.info(f"Test phases range: {test_phases.min()} - {test_phases.max()}")

        self.logger.info("Data validation complete")

    def get_step_names(self) -> dict:
        """Get step names from raw data."""
        if self.train_data and 'steps_categories' in self.train_data:
            return self.train_data['steps_categories']
        return {}

    def get_phase_names(self) -> dict:
        """Get phase names from raw data."""
        if self.train_data and 'phases_categories' in self.train_data:
            return self.train_data['phases_categories']
        return {}
    
    def save_dataframes(self):
        """Save DataFrames to CSV for debugging."""
        if self.train_df is not None:
            train_csv_path = os.path.join(OUTPUT_CONFIG['tables_dir'], 'train_annotations.csv')
            self.train_df.to_csv(train_csv_path, index=False)
            self.logger.info(f"Saved train DataFrame to {train_csv_path}")
        if self.test_df is not None:
            test_csv_path = os.path.join(OUTPUT_CONFIG['tables_dir'], 'test_annotations.csv')
            self.test_df.to_csv(test_csv_path, index=False)
            self.logger.info(f"Saved test DataFrame to {test_csv_path}")
        if self.train_df is not None and self.test_df is not None:
            combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
            combined_csv_path = os.path.join(OUTPUT_CONFIG['tables_dir'], 'total_annotations.csv')
            combined_df.to_csv(combined_csv_path, index=False)
            self.logger.info(f"Saved combined DataFrame to {combined_csv_path}")


if __name__ == '__main__':
    # Test
    import sys
    sys.path.insert(0, '/scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/analisis/analysis')

    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    train_df, test_df = loader.load_and_prepare()
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nTrain columns: {train_df.columns.tolist()}")
    print(f"\nTrain sample:\n{train_df.head()}")
