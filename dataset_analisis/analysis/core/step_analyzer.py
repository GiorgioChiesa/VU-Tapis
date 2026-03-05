"""
Step Analyzer for GraSP Dataset.
Implements comprehensive analysis of steps and phases across 6 categories.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from utils.step_taxonomy import StepTaxonomy, PhaseTaxonomy
from config.analysis_config import STEP_CATEGORIES, PHASE_CATEGORIES


class StepAnalyzer:
    """
    Performs comprehensive analysis on surgical steps and phases.
    Analyzes from 6 perspectives: quantitative, spatial, temporal, correlation, quality, comparative.
    """

    def __init__(self, dfs: Dict[str, pd.DataFrame], logger=None):
        """
        Initialize analyzer with train and test dataframes.

        Args:
            dfs: Dictionary containing train and test dataframes
            logger: Optional logger instance
        """
        # self.train_df = dfs.get('train')
        # self.test_df = dfs.get('test')
        self.dfs = dfs
        self.total_df = pd.concat([d for d in dfs.values()], ignore_index=True)
        self.logger = logger or logging.getLogger(__name__)

        self.step_taxonomy = StepTaxonomy()
        self.phase_taxonomy = PhaseTaxonomy()

        # Cache for analysis results
        self.results = {}

    # ========================================================================
    # CATEGORY A: QUANTITATIVE DISTRIBUTION
    # ========================================================================

    def distribution_by_step(self) -> pd.DataFrame:
        """
        Analyze distribution of annotations by step.

        Returns:
            DataFrame with step distribution statistics
        """
        self.logger.info("Analyzing step distribution...")

        rows = []
        for step_id in sorted(STEP_CATEGORIES.keys()):
            step_name = STEP_CATEGORIES[step_id]

            counts = [len(d[d['step'] == step_id]) for d in self.dfs.values()]
            # train_count = len(self.train_df[self.train_df['step'] == step_id])
            # test_count = len(self.test_df[self.test_df['step'] == step_id])
            total_count = sum(counts)

            pct = []
            for dataset, count in zip(self.dfs.keys(), counts):
                pct.append(count / len(self.dfs[dataset]) * 100 if len(self.dfs[dataset]) > 0 else 0)
            
            total_pct = total_count / len(self.total_df) * 100 if len(self.total_df) > 0 else 0
            # train_pct = (
            #     train_count / len(self.train_df) * 100 if len(self.train_df) > 0 else 0
            # )
            # test_pct = test_count / len(self.test_df) * 100 if len(self.test_df) > 0 else 0
            # total_pct = (
            #     total_count / (len(self.train_df) + len(self.test_df)) * 100
            #     if (len(self.train_df) + len(self.test_df)) > 0
            #     else 0
            # )

            step_class = self.step_taxonomy.classify_step(step_id)

            row = {
                'step_id': step_id,
                'step_name': step_name,
                # 'train_count': train_count,
                # 'test_count': test_count,
                'total_count': total_count,
                # 'train_pct': train_pct,
                # 'test_pct': test_pct,
                'total_pct': total_pct,
                'step_class': step_class,
            }
            for dataset, count, percent in zip(self.dfs.keys(), counts, pct):
                row.update({
                    f'{dataset}_count': count,
                    f'{dataset}_pct': percent,
                })
            rows.append(row)

        df = pd.DataFrame(rows)
        self.results['step_distribution'] = df
        return df

    def distribution_by_phase(self) -> pd.DataFrame:
        """
        Analyze distribution of annotations by phase.

        Returns:
            DataFrame with phase distribution statistics
        """
        self.logger.info("Analyzing phase distribution...")

        rows = []
        for phase_id in sorted(PHASE_CATEGORIES.keys()):
            phase_name = PHASE_CATEGORIES[phase_id]
            
            counts = []
            pct = []
            for dataset, df in self.dfs.items():
                    count = len(df[df['phase'] == phase_id])
                    counts.append(count)
                    pct.append(count / len(df) * 100 if len(df) > 0 else 0)
            # train_count = len(self.train_df[self.train_df['phase'] == phase_id])
            # test_count = len(self.test_df[self.test_df['phase'] == phase_id])
            total_count = sum(counts)

            # train_pct = (
            #     train_count / len(self.train_df) * 100 if len(self.train_df) > 0 else 0
            # )
            # test_pct = test_count / len(self.test_df) * 100 if len(self.test_df) > 0 else 0
            total_pct = (
                total_count / len(self.total_df) * 100
                if len(self.total_df) > 0
                else 0
            )
            row = {
                'phase_id': phase_id,
                'phase_name': phase_name,
                # 'train_count': train_count,
                # 'test_count': test_count,
                'total_count': total_count,
                # 'train_pct': train_pct,
                # 'test_pct': test_pct,
                'total_pct': total_pct,
            }
            for dataset, count, percent in zip(self.dfs.keys(), counts, pct):
                row.update({
                    f'{dataset}_count': count,
                    f'{dataset}_pct': percent,
                })
            rows.append(row)

        df = pd.DataFrame(rows)
        self.results['phase_distribution'] = df
        return df

    # ========================================================================
    # CATEGORY B: SPATIAL DISTRIBUTION (Video-level)
    # ========================================================================

    def step_distribution_by_video(self) -> pd.DataFrame:
        """
        Analyze step distribution across videos.

        Returns:
            DataFrame with step counts per video
        """
        self.logger.info("Analyzing step distribution by video...")

        # Get unique videos from both train and test
        all_videos = sorted(self.total_df['video_name'].unique())

        rows = []
        for step_id in sorted(STEP_CATEGORIES.keys()):
            step_name = STEP_CATEGORIES[step_id]
            row = {'step_id': step_id, 'step_name': step_name}

            for video in all_videos:
                count = len(
                    self.total_df[
                    (self.total_df['video_name'] == video)
                    & (self.total_df['step'] == step_id)
                    ]
                )
                row[f'{video}'] = count

                # row['total_count'] = len(self.total_df[self.total_df['step'] == step_id])
            rows.append(row)

        df = pd.DataFrame(rows)
        self.results['step_video_matrix'] = df
        return df

    def phase_distribution_by_video(self) -> pd.DataFrame:
        """
        Analyze phase distribution across videos.

        Returns:
            DataFrame with phase counts per video
        """
        self.logger.info("Analyzing phase distribution by video...")


        # Get unique videos from both train and test
        all_videos = sorted(self.total_df['video_name'].unique())

        rows = []
        for phase_id in sorted(PHASE_CATEGORIES.keys()):
            phase_name = PHASE_CATEGORIES[phase_id]
            row = {'phase_id': phase_id, 'phase_name': phase_name}

            for video in all_videos:
                count = len(
                    self.total_df[
                    (self.total_df['video_name'] == video)
                    & (self.total_df['phase'] == phase_id)
                    ]
                )
                row[f'{video}'] = count

                # total_train = len(self.train_df[self.train_df['phase'] == phase_id])
                # row['total_train'] = total_train
                # total_test = len(self.test_df[self.test_df['phase'] == phase_id])
                # row['total_test'] = total_test
                # row['total'] = total_train + total_test
            rows.append(row)

        df = pd.DataFrame(rows)
        self.results['phase_video_matrix'] = df
        return df

    def video_statistics(self) -> pd.DataFrame:  # TODO - add phase statistics as well include test
        """
        Compute statistics per video.

        Returns:
            DataFrame with video-level statistics
        """
        self.logger.info("Computing video statistics...")

        videos = sorted(self.total_df['video_name'].unique())
        rows = []

        for video in videos:
            video_df = self.total_df[self.total_df['video_name'] == video]

            total_frames = len(video_df)
            unique_steps = video_df['step'].nunique()
            unique_phases = video_df['phase'].nunique()
            step_diversity = unique_steps / len(STEP_CATEGORIES)
            phase_diversity = unique_phases / len(PHASE_CATEGORIES)

            rows.append({
                'video_name': video,
                'total_frames': total_frames,
                'unique_steps': unique_steps,
                'unique_phases': unique_phases,
                'step_diversity': step_diversity,
                'phase_diversity': phase_diversity,
            })

        df = pd.DataFrame(rows)
        self.results['video_statistics'] = df
        return df

    # ========================================================================
    # CATEGORY C: TEMPORAL DISTRIBUTION
    # ========================================================================

    def temporal_step_progression(self) -> pd.DataFrame:
        """
        Analyze temporal progression of steps within videos.

        Returns:
            DataFrame with temporal patterns
        """
        self.logger.info("Analyzing temporal step progression...")

        rows = []
        videos = sorted(self.total_df['video_name'].unique())

        for video in videos:
            video_df = self.total_df[self.total_df['video_name'] == video].sort_values(
                'frame_num'
            )

            # Get contiguous segments
            segments = self._extract_contiguous_segments(video_df['step'].values)

            for idx, seg in enumerate(segments):
                rows.append({
                    'video_name': video,
                    'segment_idx': idx,
                    'step_id': seg['step'],
                    'step_name': STEP_CATEGORIES.get(seg['step'], 'Unknown'),
                    'start_frame': seg['start'],
                    'end_frame': seg['end'],
                    'duration_frames': seg['end'] - seg['start'],
                    'segment_position': idx,
                })

        df = pd.DataFrame(rows)
        self.results['temporal_steps_progression'] = df
        return df


    def temporal_phases_progression(self) -> pd.DataFrame:
        """
        Analyze temporal progression of phases within videos.

        Returns:
            DataFrame with temporal patterns
        """
        self.logger.info("Analyzing temporal phase progression...")

        rows = []
        videos = sorted(self.total_df['video_name'].unique())

        for video in videos:
            video_df = self.total_df[self.total_df['video_name'] == video].sort_values(
                'frame_num'
            )

            # Get contiguous segments
            segments = self._extract_contiguous_segments(video_df['phase'].values)

            for idx, seg in enumerate(segments):
                rows.append({
                    'video_name': video,
                    'segment_idx': idx,
                    'phase_id': seg['step'],
                    'phase_name': PHASE_CATEGORIES.get(seg['step'], 'Unknown'), # Corretto perche la funzione sopra restituice step, ma è passato un array di phases
                    'start_frame': seg['start'],
                    'end_frame': seg['end'],
                    'duration_frames': seg['end'] - seg['start'],
                    'segment_position': idx,
                })

        df = pd.DataFrame(rows)
        self.results['temporal_phases_progression'] = df
        return df


    def step_transitions(self) -> pd.DataFrame:
        """
        Analyze transitions between steps within videos.

        Returns:
            DataFrame with transition counts and probabilities
        """
        self.logger.info("Analyzing step transitions...")

        # Build transition matrix
        transition_matrix = np.zeros((len(STEP_CATEGORIES), len(STEP_CATEGORIES)))

        videos = sorted(self.total_df['video_name'].unique())
        for video in videos:
            video_df = self.total_df[self.total_df['video_name'] == video].sort_values(
                'frame_num'
            )

            steps = video_df['step'].values

            # Count transitions
            for i in range(len(steps) - 1):
                from_step = int(steps[i])
                to_step = int(steps[i + 1])
                if from_step != to_step:  # Only count actual transitions
                    transition_matrix[from_step, to_step] += 1

        # Convert to DataFrame
        rows = []
        for from_step in range(len(STEP_CATEGORIES)):
            for to_step in range(len(STEP_CATEGORIES)):
                count = int(transition_matrix[from_step, to_step])
                if count > 0:
                    rows.append({
                        'from_step_id': from_step,
                        'from_step_name': STEP_CATEGORIES.get(from_step, 'Unknown'),
                        'to_step_id': to_step,
                        'to_step_name': STEP_CATEGORIES.get(to_step, 'Unknown'),
                        'count': count,
                    })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self.results['step_transitions'] = df
        return df

    # ========================================================================
    # CATEGORY D: STEP-PHASE CORRELATION
    # ========================================================================

    def step_phase_correlation(self) -> pd.DataFrame:
        """
        Analyze which steps appear in which phases.

        Returns:
            DataFrame with step-phase relationships
        """
        self.logger.info("Analyzing step-phase correlation...")

        rows = []
        for step_id in sorted(STEP_CATEGORIES.keys()):
            step_name = STEP_CATEGORIES[step_id]

            for phase_id in sorted(PHASE_CATEGORIES.keys()):
                phase_name = PHASE_CATEGORIES[phase_id]

                count = len(
                    self.total_df[
                        (self.total_df['step'] == step_id)
                        & (self.total_df['phase'] == phase_id)
                    ]
                )

                if count > 0:
                    rows.append({
                        'step_id': step_id,
                        'step_name': step_name,
                        'phase_id': phase_id,
                        'phase_name': phase_name,
                        'count': count,
                    })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self.results['step_phase_correlation'] = df
        return df

    def phase_composition(self) -> pd.DataFrame:
        """
        Analyze which steps comprise each phase.

        Returns:
            DataFrame with phase composition
        """
        self.logger.info("Analyzing phase composition...")

        rows = []
        for phase_id in sorted(PHASE_CATEGORIES.keys()):
            phase_name = PHASE_CATEGORIES[phase_id]
            phase_df = self.total_df[self.total_df['phase'] == phase_id]

            unique_steps = sorted(phase_df['step'].unique())

            for step_id in unique_steps:
                count = len(phase_df[phase_df['step'] == step_id])
                pct = count / len(phase_df) * 100 if len(phase_df) > 0 else 0

                rows.append({
                    'phase_id': phase_id,
                    'phase_name': phase_name,
                    'step_id': step_id,
                    'step_name': STEP_CATEGORIES.get(step_id, 'Unknown'),
                    'count': count,
                    'percentage': pct,
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self.results['phase_composition'] = df
        return df

    # ========================================================================
    # CATEGORY E: QUALITY ASSESSMENT
    # ========================================================================

    def step_coverage_analysis(self) -> dict:
        """
        Analyze which steps are present/missing in train vs test.

        Returns:
            Dictionary with coverage metrics
        """
        self.logger.info("Analyzing step coverage...")

        steps_in_datasets = {dataset: set(df['step'].unique()) for dataset, df in self.dfs.items()}
        for dataset, df in self.dfs.items():
            self.logger.info(f"\n{dataset} DataFrame shape: {df.shape}")
        # train_steps = set(self.train_df['step'].unique())
        # test_steps = set(self.test_df['step'].unique())

        all_steps = set(STEP_CATEGORIES.keys())
        missing_in = {dataset: all_steps - steps for dataset, steps in steps_in_datasets.items()}
        # missing_in_train = all_steps - train_steps
        # missing_in_test = all_steps - test_steps
        # missing_in_both = missing_in_train & missing_in_test
        # present_in_both = train_steps & test_steps

        coverage = {
            'total_steps': len(all_steps),
            # 'steps_in_train': len(train_steps),
            # 'steps_in_test': len(test_steps),
            # 'steps_in_both': len(present_in_both),
            # 'missing_in_train': list(missing_in_train),
            # 'missing_in_test': list(missing_in_test),
            # 'missing_in_both': list(missing_in_both),
            # 'coverage_train': len(train_steps) / len(all_steps),
            # 'coverage_test': len(test_steps) / len(all_steps),
        }
        for dataset, steps in steps_in_datasets.items():
            coverage[f'steps_in_{dataset}'] = len(steps)
            coverage[f'coverage_{dataset}'] = len(steps) / len(all_steps)
            coverage[f'missing_in_{dataset}'] = list(missing_in[dataset])
        # coverage = coverage.update({f'missing_in_{dataset}': list(steps) for dataset, steps in missing_in.items()})
        # coverage = coverage.update({f'steps_in_{dataset}': len(steps) for dataset, steps in steps_in_datasets.items()})
        # coverage.update({f'coverage_{dataset}': len(steps) / len(all_steps) for dataset, steps in steps_in_datasets.items()})

        self.results['step_coverage'] = coverage
        return coverage

    def imbalance_assessment(self) -> dict:
        """
        Assess dataset imbalance metrics.

        Returns:
            Dictionary with imbalance statistics
        """
        self.logger.info("Assessing dataset imbalance...")

        step_dist = self.distribution_by_step()
        counts = [step_dist[f'{dataset}_count'].values for dataset in self.dfs.keys()]  
        # train_counts = step_dist['train_count'].values
        # test_counts = step_dist['test_count'].values

        # Remove zeros for better calculation
        non_zero_counts = [counts[i][counts[i] > 0] for i in range(len(counts))]
        # train_nonzero = train_counts[train_counts > 0]
        # test_nonzero = test_counts[test_counts > 0]
        imbalance = {}
        for dataset, count, non_zero in zip(self.dfs.keys(), counts, non_zero_counts):
            imbalance[f'{dataset}_min'] = int(non_zero.min()) if len(non_zero) > 0 else 0
            imbalance[f'{dataset}_max'] = int(non_zero.max()) if len(non_zero) > 0 else 0
            imbalance[f'{dataset}_ratio'] = (
                int(non_zero.max()) / int(non_zero.min())
                if len(non_zero) > 0 and non_zero.min() > 0
                else 1
            )
            imbalance[f'{dataset}_cv'] = (
                non_zero.std() / non_zero.mean() if len(non_zero) > 0 else 0
            )
        # imbalance = {
        #     'train_min': int(train_nonzero.min()) if len(train_nonzero) > 0 else 0,
        #     'train_max': int(train_nonzero.max()) if len(train_nonzero) > 0 else 0,
        #     'train_ratio': (
        #         int(train_nonzero.max()) / int(train_nonzero.min())
        #         if len(train_nonzero) > 0 and train_nonzero.min() > 0
        #         else 1
        #     ),
        #     'train_cv': (
        #         train_nonzero.std() / train_nonzero.mean()
        #         if len(train_nonzero) > 0
        #         else 0
        #     ),
        #     'test_min': int(test_nonzero.min()) if len(test_nonzero) > 0 else 0,
        #     'test_max': int(test_nonzero.max()) if len(test_nonzero) > 0 else 0,
        #     'test_ratio': (
        #         int(test_nonzero.max()) / int(test_nonzero.min())
        #         if len(test_nonzero) > 0 and test_nonzero.min() > 0
        #         else 1
        #     ),
        #     'test_cv': (
        #         test_nonzero.std() / test_nonzero.mean() if len(test_nonzero) > 0 else 0
        #     ),
        # }

        self.results['imbalance_metrics'] = imbalance
        return imbalance

    # ========================================================================
    # CATEGORY F: COMPARATIVE STATISTICS (Train vs Test)
    # ========================================================================

    def train_test_comparison(self) -> pd.DataFrame:
        """
        Compare step distribution between train and test.

        Returns:
            DataFrame with comparative statistics
        """
        self.logger.info("Comparing train and test distributions...")

        step_dist = self.distribution_by_step()

        # Add comparison metrics
        step_dist['chi2_contribution'] = (
            ((step_dist['train_pct'] - step_dist['test_pct']) ** 2)
            / (step_dist['train_pct'] + step_dist['test_pct'] + 1e-6)
        )

        self.results['train_test_comparison'] = step_dist

        return step_dist

    def statistical_test_chi_square(self) -> dict:
        """
        Perform chi-square test on step distribution.

        Returns:
            Dictionary with test results
        """
        self.logger.info("Performing chi-square test...")

        step_dist = self.distribution_by_step()

        # Create contingency table
        observed = np.array([step_dist['train_count'].values, step_dist['test_count'].values]).T

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(observed)

        test_result = {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
        }

        self.results['chi_square_test'] = test_result
        return test_result

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _extract_contiguous_segments(self, sequence: np.ndarray) -> List[Dict]:
        """
        Extract contiguous segments from a sequence.

        Args:
            sequence: Array of step IDs

        Returns:
            List of segment dictionaries
        """
        if len(sequence) == 0:
            return []

        segments = []
        current_step = sequence[0]
        start = 0

        for i in range(1, len(sequence)):
            if sequence[i] != current_step:
                segments.append({
                    'step': int(current_step),
                    'start': start,
                    'end': i,
                })
                current_step = sequence[i]
                start = i

        # Add last segment
        segments.append({
            'step': int(current_step),
            'start': start,
            'end': len(sequence),
        })

        return segments

    def execute_all_analysis(self) -> Dict:
        """
        Execute all 6 categories of analysis.

        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 70)
        self.logger.info("EXECUTING COMPREHENSIVE ANALYSIS")
        self.logger.info("=" * 70)

        # Category A: Quantitative
        self.distribution_by_step()
        self.distribution_by_phase()

        # Category B: Spatial
        self.step_distribution_by_video()
        self.phase_distribution_by_video()
        self.video_statistics()

        # Category C: Temporal
        self.temporal_step_progression()
        self.temporal_phases_progression()
        self.step_transitions()

        # Category D: Correlation
        self.step_phase_correlation()
        self.phase_composition()

        # Category E: Quality
        self.step_coverage_analysis()
        self.imbalance_assessment()

        # Category F: Comparative
        # self.train_test_comparison()
        # self.statistical_test_chi_square()

        self.logger.info("=" * 70)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("=" * 70)

        return self.results

    def get_results(self) -> Dict:
        """Get cached analysis results."""
        return self.results
