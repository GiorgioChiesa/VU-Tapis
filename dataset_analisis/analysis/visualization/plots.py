"""
Visualization Module for GraSP Analysis.
Generates 12+ plots for comprehensive visualization of step and phase analysis.
Uses matplotlib and seaborn for static PNG/PDF output.
"""

import logging
import os
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from config.analysis_config import (
    OUTPUT_CONFIG,
    VISUALIZATION_CONFIG,
    STEP_CATEGORIES,
    PHASE_CATEGORIES,
)


class StepVisualizer:
    """
    Generates visualizations for step and phase analysis.
    """

    def __init__(self, logger=None):
        """Initialize visualizer with configuration."""
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = OUTPUT_CONFIG['plots_dir']
        self.dpi = VISUALIZATION_CONFIG['dpi']
        self.figure_size = VISUALIZATION_CONFIG['figure_size']
        self.small_figure_size = VISUALIZATION_CONFIG['small_figure_size']
        self.heatmap_size = VISUALIZATION_CONFIG['heatmap_size']
        self.font_size = VISUALIZATION_CONFIG['font_size']
        self.title_size = VISUALIZATION_CONFIG['title_size']
        self.label_size = VISUALIZATION_CONFIG['label_size']
        self.save_formats = VISUALIZATION_CONFIG['save_formats']

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette(VISUALIZATION_CONFIG['color_palette'])

        os.makedirs(self.output_dir, exist_ok=True)

    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure to disk in configured formats."""
        for fmt in self.save_formats:
            output_path = os.path.join(self.output_dir, f"{filename}.{fmt}")
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {output_path}")
        plt.close(fig)

    # ========================================================================
    # PLOT 1: STEP DISTRIBUTION HISTOGRAM
    # ========================================================================

    def plot_step_distribution_histogram(self, step_dist_df: pd.DataFrame):
        """
        Generate histogram of step distribution (train vs test).

        Args:
            step_dist_df: DataFrame with step distribution
        """
        self.logger.info("Generating step distribution histogram...")

        fig, ax = plt.subplots(figsize=self.figure_size)

        x = np.arange(len(step_dist_df))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            step_dist_df['train_count'],
            width,
            label='Train',
            alpha=0.8,
            color='steelblue',
        )
        bars2 = ax.bar(
            x + width / 2,
            step_dist_df['test_count'],
            width,
            label='Test',
            alpha=0.8,
            color='coral',
        )

        ax.set_xlabel('Step ID', fontsize=self.label_size)
        ax.set_ylabel('Count', fontsize=self.label_size)
        ax.set_title('Step Distribution: Train vs Test', fontsize=self.title_size)
        ax.set_xticks(x)
        ax.set_xticklabels(step_dist_df['step_name'], rotation=45, ha='right')
        ax.legend(fontsize=self.font_size)
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, '01_step_distribution_histogram')

    # ========================================================================
    # PLOT 2: STEP DISTRIBUTION BARPLOT
    # ========================================================================

    def plot_step_distribution_barplot(self, step_dist_df: pd.DataFrame):
        """
        Generate barplot of step counts ordered by frequency.

        Args:
            step_dist_df: DataFrame with step distribution
        """
        self.logger.info("Generating step distribution barplot...")

        df_sorted = step_dist_df.sort_values('total_count', ascending=False)

        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = sns.color_palette(VISUALIZATION_CONFIG['color_palette'], len(df_sorted))
        bars = ax.barh(df_sorted['step_name'], df_sorted['total_count'], color=colors)

        ax.set_xlabel('Total Count', fontsize=self.label_size)
        ax.set_ylabel('Step', fontsize=self.label_size)
        ax.set_title('Step Distribution (Total Counts)', fontsize=self.title_size)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f' {int(width)}',
                   ha='left', va='center', fontsize=self.font_size - 2)

        fig.tight_layout()
        self._save_figure(fig, '02_step_distribution_barplot')

    # ========================================================================
    # PLOT 3: STEP DISTRIBUTION PIE CHART
    # ========================================================================

    def plot_step_distribution_pie(self, step_dist_df: pd.DataFrame):
        """
        Generate pie chart of step distribution.

        Args:
            step_dist_df: DataFrame with step distribution
        """
        self.logger.info("Generating step distribution pie chart...")

        fig, ax = plt.subplots(figsize=self.small_figure_size)

        # Only plot non-zero values
        non_zero_df = step_dist_df[step_dist_df['total_count'] > 0]

        colors = sns.color_palette(VISUALIZATION_CONFIG['color_palette'], len(non_zero_df))

        wedges, texts, autotexts = ax.pie(
            non_zero_df['total_count'],
            labels=non_zero_df['step_name'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': self.font_size - 2},
        )

        ax.set_title('Step Distribution (Pie Chart)', fontsize=self.title_size)

        fig.tight_layout()
        self._save_figure(fig, '03_step_distribution_pie')

    # ========================================================================
    # PLOT 4: STEP × VIDEO HEATMAP
    # ========================================================================

    def plot_step_video_heatmap(self, step_video_df: pd.DataFrame):
        """
        Generate heatmap of step × video occurrence matrix.

        Args:
            step_video_df: DataFrame with step-video matrix
        """
        self.logger.info("Generating step × video heatmap...")

        # Extract video columns (all except step_id, step_name, total)
        video_cols = [c for c in step_video_df.columns if c not in ['step_id', 'step_name', 'total']]

        # Build heatmap data
        heatmap_data = step_video_df[video_cols].copy()
        heatmap_data.index = step_video_df['step_name']

        fig, ax = plt.subplots(figsize=self.heatmap_size)
        #Normalize for better color scaling
        heatmap_data = heatmap_data / heatmap_data.sum()  # Shift to zero baseline for better color scaling

        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            cbar_kws={'label': '% Count'},
            ax=ax,
            annot=True,
            fmt=".3f",
            linewidth=.5,
            min=0,
        )

        ax.set_title('Step × Video Occurrence Matrix', fontsize=self.title_size)
        ax.set_xlabel('Video', fontsize=self.label_size)
        ax.set_ylabel('Step', fontsize=self.label_size)

        fig.tight_layout()
        self._save_figure(fig, '04_step_video_heatmap')



    # ========================================================================
    # PLOT 5: STEP TEMPORAL SEQUENCES
    # ========================================================================

    def plot_step_temporal_sequences(self, temporal_df: pd.DataFrame, sample_videos: int = 3):
        """
        Generate temporal sequence plot for sample videos.

        Args:
            temporal_df: DataFrame with temporal progression
            sample_videos: Number of videos to plot
        """
        self.logger.info("Generating step temporal sequences...")

        videos = temporal_df['video_name'].unique()[:sample_videos]

        num_videos = len(videos)
        fig, axes = plt.subplots(num_videos, 1, figsize=(16, 3 * num_videos))
        if num_videos == 1:
            axes = [axes]

        for idx, (ax, video) in enumerate(zip(axes, videos)):
            video_data = temporal_df[temporal_df['video_name'] == video]

            # Create timeline with colors for each step
            colors_map = sns.color_palette(VISUALIZATION_CONFIG['color_palette'], len(STEP_CATEGORIES))
            step_colors = [colors_map[int(row['step_id'])] for _, row in video_data.iterrows()]

            bars = ax.barh(
            [0] * len(video_data),
            video_data['duration_frames'],
            left=video_data['start_frame'],
            color=step_colors,
            edgecolor='black',
            linewidth=0.5,
            )

            ax.set_xlim(0, video_data['end_frame'].max() + 10)
            ax.set_xlabel('Frame', fontsize=self.label_size)
            ax.set_title(f'Step Sequence: {video}', fontsize=self.title_size)
            ax.set_yticks([])
            ax.grid(axis='x', alpha=0.3)

        # Add color legend for steps
        legend_patches = [mpatches.Patch(color=colors_map[i], label=STEP_CATEGORIES[i])
                 for i in range(len(STEP_CATEGORIES))]
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=min(4, len(STEP_CATEGORIES)), fontsize=self.font_size - 2)

        fig.tight_layout()
        self._save_figure(fig, '05_step_temporal_sequences')

    # ========================================================================
    # PLOT 6: PHASE DISTRIBUTION BARPLOT
    # ========================================================================

    def plot_phase_distribution_barplot(self, phase_dist_df: pd.DataFrame):
        """Generate barplot of phase distribution."""
        self.logger.info("Generating phase distribution barplot...")

        df_sorted = phase_dist_df.sort_values('total_count', ascending=False)

        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = sns.color_palette(VISUALIZATION_CONFIG['color_palette'], len(df_sorted))
        bars = ax.barh(df_sorted['phase_name'], df_sorted['total_count'], color=colors)

        ax.set_xlabel('Total Count', fontsize=self.label_size)
        ax.set_ylabel('Phase', fontsize=self.label_size)
        ax.set_title('Phase Distribution (Total Counts)', fontsize=self.title_size)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f' {int(width)}',
                   ha='left', va='center', fontsize=self.font_size - 2)

        fig.tight_layout()
        self._save_figure(fig, '06_phase_distribution_barplot')

    # ========================================================================
    # PLOT 7: PHASE × VIDEO HEATMAP
    # ========================================================================

    def plot_phase_video_heatmap(self, phase_video_df: pd.DataFrame):
        """Generate heatmap of phase × video occurrence matrix."""
        self.logger.info("Generating phase × video heatmap...")

        video_cols = [c for c in phase_video_df.columns if c not in ['phase_id', 'phase_name', 'total']]
        heatmap_data = phase_video_df[video_cols].copy()
        heatmap_data.index = phase_video_df['phase_name']

        #Normalize for better color scaling
        heatmap_data = heatmap_data / heatmap_data.sum()  # Shift to zero baseline for better color scaling

        fig, ax = plt.subplots(figsize=self.heatmap_size)

        sns.heatmap(
            heatmap_data,
            cmap='RdPu',
            cbar_kws={'label': '% Count'},
            ax=ax,
            annot=True,
            fmt=".3f",
            linewidth=.5,
        )

        ax.set_title('Phase × Video Occurrence Matrix', fontsize=self.title_size)
        ax.set_xlabel('Video', fontsize=self.label_size)
        ax.set_ylabel('Phase', fontsize=self.label_size)

        fig.tight_layout()
        self._save_figure(fig, '07_phase_video_heatmap')

    # ========================================================================
    # PLOT 8: STEP-PHASE CORRELATION HEATMAP
    # ========================================================================

    def plot_step_phase_correlation(self, step_phase_df: pd.DataFrame):
        """Generate heatmap of step-phase correlation."""
        self.logger.info("Generating step-phase correlation heatmap...")

        # Build pivot table
        pivot_df = step_phase_df.pivot_table(
            index='step_name',
            columns='phase_name',
            values='count',
            fill_value=0,
        )
        #Normalize for better color scaling
        pivot_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min())  # Shift to zero baseline for better color scaling

        fig, ax = plt.subplots(figsize=self.heatmap_size)

        sns.heatmap(
            pivot_df,
            cmap='viridis',
            cbar_kws={'label': '% Count'},
            ax=ax,
            annot=True,
            fmt=".3f",
            linewidth=.5,
        )

        ax.set_title('Step-Phase Correlation Matrix', fontsize=self.title_size)
        ax.set_xlabel('Phase', fontsize=self.label_size)
        ax.set_ylabel('Step', fontsize=self.label_size)

        fig.tight_layout()
        self._save_figure(fig, '08_step_phase_correlation_heatmap')


    # ========================================================================
    # PLOT 8-1: STEP-STEP CORRELATION HEATMAP
    # ========================================================================

    def plot_temporal_progression(self, temporal_progression_df: pd.DataFrame):
        """Generate heatmap of step-step correlation."""
        self.logger.info("Generating temporal progression heatmap...")

        # Build pivot table
        pivot_df = temporal_progression_df.pivot_table(
            index='from_step_name',
            columns='to_step_name',
            values='count',
            fill_value=0,
        )
        #Normalize for better color scaling
        pivot_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min())  # Shift to zero baseline for better color scaling

        fig, ax = plt.subplots(figsize=self.heatmap_size)

        sns.heatmap(
            pivot_df,
            cmap='viridis',
            cbar_kws={'label': '% Count'},
            ax=ax,
            annot=True,
            fmt=".2f",
            linewidth=.5,
        )

        ax.set_title('Step-Step Correlation Matrix', fontsize=self.title_size)
        ax.set_xlabel('To Step', fontsize=self.label_size)
        ax.set_ylabel('From Step', fontsize=self.label_size)

        fig.tight_layout()
        self._save_figure(fig, '08-1_step_step_correlation_heatmap')


    # ========================================================================
    # PLOT 9: TRAIN VS TEST COMPARISON BOXPLOT
    # ========================================================================

    def plot_train_test_comparison(self, step_dist_df: pd.DataFrame):
        """Generate boxplot comparing train and test distributions."""
        self.logger.info("Generating train vs test comparison boxplot...")

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Prepare data for boxplot
        data_to_plot = [step_dist_df['train_count'].values, step_dist_df['test_count'].values]

        bp = ax.boxplot(
            data_to_plot,
            labels=['Train', 'Test'],
            patch_artist=True,
            widths=0.5,
        )

        # Color the boxes
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Count', fontsize=self.label_size)
        ax.set_title('Train vs Test Distribution Comparison', fontsize=self.title_size)
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, '09_train_test_step_comparison_boxplot')

    # ========================================================================
    # PLOT 10: STEP IMBALANCE ANALYSIS
    # ========================================================================

    def plot_step_imbalance_analysis(self, step_dist_df: pd.DataFrame):
        """Generate visualization of step imbalance."""
        self.logger.info("Generating step imbalance analysis...")

        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)

        # Left: Range (min to max)
        ax = axes[0]
        sorted_df = step_dist_df.sort_values('total_count', ascending=True)
        ax.barh(range(len(sorted_df)), sorted_df['total_count'], color='steelblue')
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['step_name'], fontsize=self.font_size - 2)
        ax.set_xlabel('Count', fontsize=self.label_size)
        ax.set_title('Step Count Range (Sorted)', fontsize=self.title_size)
        ax.set_yscale('log')
        ax.grid(axis='x', alpha=0.3)

        # Right: Ratio statistics
        ax = axes[1]
        min_count = step_dist_df[step_dist_df['total_count'] > 0]['total_count'].min()
        max_count = step_dist_df['total_count'].max()
        ratio = max_count / min_count if min_count > 0 else 0

        stats_text = f"""
        Dataset Imbalance Statistics:

        Min count: {int(min_count)}
        Max count: {int(max_count)}
        Ratio (max/min): {ratio:.1f}x

        Mean: {step_dist_df['total_count'].mean():.1f}
        Std Dev: {step_dist_df['total_count'].std():.1f}
        CV: {step_dist_df['total_count'].std() / step_dist_df['total_count'].mean():.3f}
        """

        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=self.font_size, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')

        fig.tight_layout()
        self._save_figure(fig, '10_step_imbalance_analysis')

    # ========================================================================
    # PLOT 11: MISSING STEPS HEATMAP
    # ========================================================================

    def plot_missing_steps_heatmap(self, step_video_df: pd.DataFrame):
        """Generate heatmap showing which steps are missing in which videos."""
        self.logger.info("Generating missing steps heatmap...")

        video_cols = [c for c in step_video_df.columns if c not in ['step_id', 'step_name', 'total']]

        # Create binary matrix: 1 if step present, 0 if missing
        binary_matrix = (step_video_df[video_cols] > 0).astype(int)
        binary_matrix.index = step_video_df['step_name']

        fig, ax = plt.subplots(figsize=self.heatmap_size)

        sns.heatmap(
            binary_matrix,
            cmap='RdYlGn',
            cbar_kws={'label': 'Present'},
            ax=ax,
            annot=True,
            fmt='d',
            cbar=True,
            vmin=0,
            vmax=1,
        )

        ax.set_title('Step Presence by Video (Missing Steps Analysis)', fontsize=self.title_size)
        ax.set_xlabel('Video', fontsize=self.label_size)
        ax.set_ylabel('Step', fontsize=self.label_size)

        fig.tight_layout()
        self._save_figure(fig, '11_missing_steps_heatmap')

    # ========================================================================
    # PLOT 12: PHASE COMPOSITION STACKED BARS
    # ========================================================================

    def plot_phase_composition_stacked(self, phase_composition_df: pd.DataFrame):
        """Generate stacked barplot showing step composition per phase."""
        self.logger.info("Generating phase composition stacked bars...")

        # Pivot: phase rows, step columns, percentages
        pivot_df = phase_composition_df.pivot_table(
            index='phase_name',
            columns='step_name',
            values='percentage',
            fill_value=0,
        )

        fig, ax = plt.subplots(figsize=self.figure_size)

        colors = sns.color_palette(VISUALIZATION_CONFIG['color_palette'], len(pivot_df.columns))
        pivot_df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            legend=True,
            color=colors,
        )
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fontsize=self.font_size, fmt='%d')



        ax.set_xlabel('Percentage (%)', fontsize=self.label_size)
        ax.set_ylabel('Phase', fontsize=self.label_size)
        ax.set_title('Phase Composition (Percentage Step Distribution within Phases)', fontsize=self.title_size)
        ax.legend(title='Step', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=self.font_size - 2)
        ax.grid(axis='x', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, '12_phase_composition_stacked_bars')

    # ========================================================================
    # MASTER GENERATION METHOD
    # ========================================================================

    def generate_all_plots(self, analysis_results: dict):
        """
        Generate all plots from analysis results.

        Args:
            analysis_results: Dictionary with analysis results from StepAnalyzer
        """
        self.logger.info("=" * 70)
        self.logger.info("GENERATING ALL VISUALIZATIONS")
        self.logger.info("=" * 70)

        try:
            # Plot 1-3: Step distributions
            if 'step_distribution' in analysis_results:
                step_dist = analysis_results['step_distribution']
                self.plot_step_distribution_histogram(step_dist)
                self.plot_step_distribution_barplot(step_dist)
                self.plot_step_distribution_pie(step_dist)

            # Plot 4-5: Step spatial and temporal
            if 'step_video_matrix' in analysis_results:
                self.plot_step_video_heatmap(analysis_results['step_video_matrix'])
            if 'temporal_progression' in analysis_results:
                self.plot_step_temporal_sequences(analysis_results['temporal_progression'])

            # Plot 6-7: Phase distributions
            if 'phase_distribution' in analysis_results:
                self.plot_phase_distribution_barplot(analysis_results['phase_distribution'])
            if 'phase_video_matrix' in analysis_results:
                self.plot_phase_video_heatmap(analysis_results['phase_video_matrix'])

            # Plot 8: Step-Phase correlation
            if 'step_phase_correlation' in analysis_results:
                self.plot_step_phase_correlation(analysis_results['step_phase_correlation'])
            
            # Plot 8-1: Step-Step correlation
            if 'temporal_progression' in analysis_results:
                self.plot_temporal_progression(analysis_results['step_transitions'])

            # Plot 9: Comparison
            # if 'train_test_comparison' in analysis_results:
            #     self.plot_train_test_comparison(analysis_results['train_test_comparison'])

            # Plot 10-11: Quality metrics
            # if 'step_distribution' in analysis_results:
            #     self.plot_step_imbalance_analysis(analysis_results['step_distribution'])
            if 'step_video_matrix' in analysis_results:
                self.plot_missing_steps_heatmap(analysis_results['step_video_matrix'])

            # Plot 12: Phase composition
            if 'phase_composition' in analysis_results:
                self.plot_phase_composition_stacked(analysis_results['phase_composition'])

            self.logger.info("=" * 70)
            self.logger.info("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            raise
