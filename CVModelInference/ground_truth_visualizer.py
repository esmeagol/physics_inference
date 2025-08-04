"""
Ground Truth Visualization Components

This module provides comprehensive visualization capabilities for ground truth
evaluation results, including timeline plots, accuracy analysis, and comparative charts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
try:
    from .ground_truth_evaluator import GroundTruthEvaluator, EvaluationSummary
except ImportError:
    # Handle case when running as script or module
    try:
        from CVModelInference.ground_truth_evaluator import GroundTruthEvaluator, EvaluationSummary
    except ImportError:
        from ground_truth_evaluator import GroundTruthEvaluator, EvaluationSummary


@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance."""
    figure_size: Tuple[int, int] = (15, 10)
    color_palette: str = 'Set2'
    font_size: int = 12
    title_font_size: int = 14
    dpi: int = 300
    style: str = 'whitegrid'


class GroundTruthVisualizer:
    """
    Comprehensive visualization system for ground truth evaluation results.
    
    Provides timeline visualizations, accuracy analysis, error distribution plots,
    and comparative analysis charts for multiple trackers.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Optional visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Set up matplotlib and seaborn styling
        plt.style.use('default')
        sns.set_style(self.config.style)
        sns.set_palette(self.config.color_palette)
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_font_size,
            'figure.dpi': self.config.dpi
        })
    
    def create_ground_truth_timeline(self, evaluator: GroundTruthEvaluator, 
                                   tracker_name: str = "Tracker",
                                   save_path: Optional[str] = None) -> None:
        """
        Create a timeline visualization showing ground truth events and evaluation results.
        
        Args:
            evaluator: GroundTruthEvaluator instance with results
            tracker_name: Name of the tracker for the title
            save_path: Optional path to save the visualization
        """
        if not evaluator.evaluation_results:
            print("No evaluation results available for timeline visualization")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.config.figure_size[0], 12), 
                                           sharex=True, height_ratios=[2, 1, 1])
        
        # Extract timeline data
        timestamps = [eval_result.timestamp for eval_result in evaluator.evaluation_results]
        accuracies = []
        error_counts = []
        suppressed_moments = []
        
        for eval_result in evaluator.evaluation_results:
            if eval_result.suppressed:
                accuracies.append(None)
                error_counts.append(0)
                suppressed_moments.append(eval_result.timestamp)
            else:
                # Calculate moment accuracy
                total_errors = len(eval_result.count_errors)
                correct_errors = sum(1 for error in eval_result.count_errors 
                                   if error.error_type.value == 'correct')
                moment_accuracy = (correct_errors / total_errors * 100) if total_errors > 0 else 100
                accuracies.append(moment_accuracy)
                
                # Count total errors
                total_moment_errors = (
                    len([e for e in eval_result.count_errors if e.error_type.value != 'correct']) +
                    len(eval_result.illegal_changes) +
                    len(eval_result.duplication_errors)
                )
                error_counts.append(total_moment_errors)
        
        # Plot 1: Accuracy over time with events
        ax1.plot(timestamps, accuracies, 'o-', linewidth=2, markersize=6, label='Accuracy')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Ground Truth Timeline - {tracker_name}', fontsize=self.config.title_font_size)
        
        # Add ground truth events as vertical lines
        for event in evaluator.processed_events:
            color = self._get_event_color(event.event_type.value)
            ax1.axvline(x=event.start_time, color=color, linestyle='--', alpha=0.7, 
                       label=f'{event.event_type.value}' if event.start_time == evaluator.processed_events[0].start_time else "")
            
            if event.end_time is not None:
                # Add shaded region for range events
                ax1.axvspan(event.start_time, event.end_time, color=color, alpha=0.2)
        
        # Mark suppressed moments
        if suppressed_moments:
            ax1.scatter(suppressed_moments, [50] * len(suppressed_moments), 
                       color='red', marker='x', s=100, label='Suppressed', zorder=5)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Error counts over time
        ax2.bar(timestamps, error_counts, width=evaluator.moment_duration*0.8, 
               alpha=0.7, color='coral')
        ax2.set_ylabel('Error Count')
        ax2.set_title('Errors per Moment')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Event timeline
        self._plot_event_timeline(ax3, evaluator.processed_events)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Ground Truth Events')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Timeline visualization saved to {save_path}")
        else:
            plt.show()
    
    def create_accuracy_analysis(self, evaluation_summary: EvaluationSummary,
                               tracker_name: str = "Tracker",
                               save_path: Optional[str] = None) -> None:
        """
        Create comprehensive accuracy analysis visualizations.
        
        Args:
            evaluation_summary: Evaluation summary with results
            tracker_name: Name of the tracker
            save_path: Optional path to save the visualization
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        # Plot 1: Per-ball accuracy
        ball_types = list(evaluation_summary.per_ball_accuracy.keys())
        accuracies = [stats['accuracy'] for stats in evaluation_summary.per_ball_accuracy.values()]
        
        bars1 = ax1.bar(ball_types, accuracies, color=sns.color_palette(self.config.color_palette, len(ball_types)))
        ax1.set_title('Per-Ball Type Accuracy')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Context-based accuracy
        if evaluation_summary.context_accuracy:
            contexts = list(evaluation_summary.context_accuracy.keys())
            context_accs = [stats['accuracy'] for stats in evaluation_summary.context_accuracy.values()]
            
            bars2 = ax2.bar(contexts, context_accs, color='lightblue')
            ax2.set_title('Context-Based Accuracy')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, acc in zip(bars2, context_accs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{acc:.1f}%', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No context data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Context-Based Accuracy')
        
        # Plot 3: Error distribution
        error_types = ['Over Detection', 'Under Detection', 'Illegal Changes', 'Duplications']
        error_counts = [
            sum(stats.get('over_detections', 0) for stats in evaluation_summary.per_ball_accuracy.values()),
            sum(stats.get('under_detections', 0) for stats in evaluation_summary.per_ball_accuracy.values()),
            evaluation_summary.continuity_stats.get('total_illegal_disappearances', 0) + 
            evaluation_summary.continuity_stats.get('total_illegal_reappearances', 0),
            evaluation_summary.duplication_summary.get('total_duplication_errors', 0)
        ]
        
        colors = ['orange', 'red', 'purple', 'brown']
        wedges, texts, autotexts = ax3.pie(error_counts, labels=error_types, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Error Distribution')
        
        # Plot 4: Temporal accuracy trend
        if evaluation_summary.temporal_analysis.get('accuracy_over_time'):
            accuracy_over_time = evaluation_summary.temporal_analysis['accuracy_over_time']
            timestamps = evaluation_summary.temporal_analysis.get('timestamps', range(len(accuracy_over_time)))
            
            # Filter out None values
            valid_data = [(t, a) for t, a in zip(timestamps, accuracy_over_time) if a is not None]
            if valid_data:
                times, accs = zip(*valid_data)
                ax4.plot(times, accs, 'o-', linewidth=2, markersize=4)
                ax4.set_title('Accuracy Trend Over Time')
                ax4.set_xlabel('Time (seconds)')
                ax4.set_ylabel('Accuracy (%)')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 100)
            else:
                ax4.text(0.5, 0.5, 'No temporal data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Accuracy Trend Over Time')
        else:
            ax4.text(0.5, 0.5, 'No temporal data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Accuracy Trend Over Time')
        
        plt.suptitle(f'Accuracy Analysis - {tracker_name}', fontsize=self.config.title_font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Accuracy analysis saved to {save_path}")
        else:
            plt.show()
    
    def create_error_distribution_charts(self, evaluation_summary: EvaluationSummary,
                                       tracker_name: str = "Tracker",
                                       save_path: Optional[str] = None) -> None:
        """
        Create detailed error distribution visualizations.
        
        Args:
            evaluation_summary: Evaluation summary with results
            tracker_name: Name of the tracker
            save_path: Optional path to save the visualization
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        # Plot 1: Error magnitude distribution
        per_ball_stats = evaluation_summary.per_ball_accuracy
        ball_types = list(per_ball_stats.keys())
        
        over_detections = [stats.get('over_detections', 0) for stats in per_ball_stats.values()]
        under_detections = [stats.get('under_detections', 0) for stats in per_ball_stats.values()]
        
        x = np.arange(len(ball_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, over_detections, width, label='Over Detection', color='orange', alpha=0.8)
        bars2 = ax1.bar(x + width/2, under_detections, width, label='Under Detection', color='red', alpha=0.8)
        
        ax1.set_title('Detection Errors by Ball Type')
        ax1.set_xlabel('Ball Type')
        ax1.set_ylabel('Error Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels(ball_types, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Continuity analysis
        continuity_stats = evaluation_summary.continuity_stats
        continuity_metrics = [
            continuity_stats.get('total_illegal_disappearances', 0),
            continuity_stats.get('total_illegal_reappearances', 0),
            continuity_stats.get('moments_with_illegal_changes', 0)
        ]
        continuity_labels = ['Illegal\nDisappearances', 'Illegal\nReappearances', 'Moments with\nIllegal Changes']
        
        bars3 = ax2.bar(continuity_labels, continuity_metrics, color=['purple', 'magenta', 'indigo'])
        ax2.set_title('Tracking Continuity Issues')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars3, continuity_metrics):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom')
        
        # Plot 3: Duplication analysis
        duplication_summary = evaluation_summary.duplication_summary
        if duplication_summary.get('duplications_by_ball_type'):
            dup_ball_types = list(duplication_summary['duplications_by_ball_type'].keys())
            dup_counts = list(duplication_summary['duplications_by_ball_type'].values())
            
            bars4 = ax3.bar(dup_ball_types, dup_counts, color='brown', alpha=0.7)
            ax3.set_title('Duplication Errors by Ball Type')
            ax3.set_xlabel('Ball Type')
            ax3.set_ylabel('Duplication Count')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars4, dup_counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No duplication errors detected', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Duplication Errors by Ball Type')
        
        # Plot 4: Error periods analysis
        temporal_analysis = evaluation_summary.temporal_analysis
        if temporal_analysis.get('error_periods'):
            error_periods = temporal_analysis['error_periods']
            period_durations = [period['duration_moments'] for period in error_periods]
            period_errors = [period['total_errors'] for period in error_periods]
            
            scatter = ax4.scatter(period_durations, period_errors, 
                                c=range(len(period_durations)), cmap='viridis', 
                                s=100, alpha=0.7)
            ax4.set_title('Error Periods Analysis')
            ax4.set_xlabel('Period Duration (moments)')
            ax4.set_ylabel('Total Errors in Period')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax4, label='Period Index')
        else:
            ax4.text(0.5, 0.5, 'No error periods identified', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Error Periods Analysis')
        
        plt.suptitle(f'Error Distribution Analysis - {tracker_name}', fontsize=self.config.title_font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Error distribution analysis saved to {save_path}")
        else:
            plt.show()
    
    def create_comparative_analysis(self, tracker_summaries: Dict[str, EvaluationSummary],
                                  save_path: Optional[str] = None) -> None:
        """
        Create comparative analysis plots for multiple trackers.
        
        Args:
            tracker_summaries: Dictionary mapping tracker names to evaluation summaries
            save_path: Optional path to save the visualization
        """
        if len(tracker_summaries) < 2:
            print("Need at least 2 trackers for comparative analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        tracker_names = list(tracker_summaries.keys())
        
        # Plot 1: Overall accuracy comparison
        overall_accuracies = [summary.overall_accuracy for summary in tracker_summaries.values()]
        bars1 = ax1.bar(tracker_names, overall_accuracies, 
                       color=sns.color_palette(self.config.color_palette, len(tracker_names)))
        ax1.set_title('Overall Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars1, overall_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Continuity comparison
        continuity_percentages = [
            summary.continuity_stats.get('continuity_percentage', 0) 
            for summary in tracker_summaries.values()
        ]
        bars2 = ax2.bar(tracker_names, continuity_percentages, color='lightgreen')
        ax2.set_title('Tracking Continuity Comparison')
        ax2.set_ylabel('Continuity (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, cont in zip(bars2, continuity_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{cont:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Per-ball accuracy heatmap
        # Get all ball types across all trackers
        all_ball_types = set()
        for summary in tracker_summaries.values():
            all_ball_types.update(summary.per_ball_accuracy.keys())
        all_ball_types = sorted(list(all_ball_types))
        
        # Create accuracy matrix
        accuracy_matrix = []
        for tracker_name in tracker_names:
            summary = tracker_summaries[tracker_name]
            row = []
            for ball_type in all_ball_types:
                if ball_type in summary.per_ball_accuracy:
                    row.append(summary.per_ball_accuracy[ball_type]['accuracy'])
                else:
                    row.append(0)  # No data for this ball type
            accuracy_matrix.append(row)
        
        im = ax3.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax3.set_title('Per-Ball Accuracy Heatmap')
        ax3.set_xticks(range(len(all_ball_types)))
        ax3.set_xticklabels(all_ball_types, rotation=45)
        ax3.set_yticks(range(len(tracker_names)))
        ax3.set_yticklabels(tracker_names)
        
        # Add text annotations
        for i in range(len(tracker_names)):
            for j in range(len(all_ball_types)):
                text = ax3.text(j, i, f'{accuracy_matrix[i][j]:.0f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='Accuracy (%)')
        
        # Plot 4: Error comparison radar chart
        self._create_error_radar_chart(ax4, tracker_summaries)
        
        plt.suptitle('Comparative Tracker Analysis', fontsize=self.config.title_font_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Comparative analysis saved to {save_path}")
        else:
            plt.show()
    
    def _create_error_radar_chart(self, ax, tracker_summaries: Dict[str, EvaluationSummary]) -> None:
        """Create a radar chart comparing error metrics across trackers."""
        # Define error metrics to compare (normalized to 0-100 scale)
        metrics = ['Overall Accuracy', 'Continuity', 'Low Over-Detection', 'Low Under-Detection', 'Low Duplications']
        
        # Calculate normalized scores for each tracker
        tracker_scores = {}
        for name, summary in tracker_summaries.items():
            # Calculate scores (higher is better)
            overall_acc = summary.overall_accuracy
            continuity = summary.continuity_stats.get('continuity_percentage', 0)
            
            # For error metrics, convert to "low error" scores
            total_over = sum(stats.get('over_detections', 0) for stats in summary.per_ball_accuracy.values())
            total_under = sum(stats.get('under_detections', 0) for stats in summary.per_ball_accuracy.values())
            total_dup = summary.duplication_summary.get('total_duplication_errors', 0)
            
            # Normalize error counts (assuming max reasonable errors for scaling)
            max_errors = max(1, summary.moments_evaluated * 7)  # 7 ball types
            low_over = max(0, 100 - (total_over / max_errors * 100))
            low_under = max(0, 100 - (total_under / max_errors * 100))
            low_dup = max(0, 100 - (total_dup / max_errors * 100))
            
            tracker_scores[name] = [overall_acc, continuity, low_over, low_under, low_dup]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        
        colors = sns.color_palette(self.config.color_palette, len(tracker_scores))
        
        for i, (name, scores) in enumerate(tracker_scores.items()):
            scores += scores[:1]  # Complete the circle
            ax.plot(angles, scores, 'o-', linewidth=2, label=name, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        ax.set_ylim(0, 100)
        ax.set_title('Error Metrics Comparison', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_event_timeline(self, ax, events: List) -> None:
        """Plot ground truth events on a timeline."""
        event_types = {}
        y_pos = 0
        
        for event in events:
            event_type = event.event_type.value
            if event_type not in event_types:
                event_types[event_type] = y_pos
                y_pos += 1
            
            y = event_types[event_type]
            color = self._get_event_color(event_type)
            
            if event.end_time is not None:
                # Range event
                rect = Rectangle((event.start_time, y - 0.4), 
                               event.end_time - event.start_time, 0.8,
                               facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
            else:
                # Point event
                ax.scatter(event.start_time, y, color=color, s=100, marker='|', linewidth=3)
        
        ax.set_yticks(range(len(event_types)))
        ax.set_yticklabels(list(event_types.keys()))
        ax.set_ylim(-0.5, len(event_types) - 0.5)
        ax.grid(True, alpha=0.3)
    
    def _get_event_color(self, event_type: str) -> str:
        """Get color for event type."""
        color_map = {
            'initial_state': 'blue',
            'ball_potted': 'red',
            'ball_placed_back': 'green',
            'balls_occluded': 'orange',
            'ignore_errors': 'gray'
        }
        return color_map.get(event_type, 'black')
    
    def export_visualization_data(self, evaluation_summary: EvaluationSummary,
                                tracker_name: str, output_path: str) -> None:
        """
        Export visualization data to CSV for external analysis.
        
        Args:
            evaluation_summary: Evaluation summary with results
            tracker_name: Name of the tracker
            output_path: Path to save the CSV file
        """
        # Prepare data for export
        data = {
            'tracker_name': tracker_name,
            'overall_accuracy': evaluation_summary.overall_accuracy,
            'total_moments': evaluation_summary.total_moments,
            'moments_evaluated': evaluation_summary.moments_evaluated,
            'moments_suppressed': evaluation_summary.moments_suppressed,
            'continuity_percentage': evaluation_summary.continuity_stats.get('continuity_percentage', 0),
            'total_duplications': evaluation_summary.duplication_summary.get('total_duplication_errors', 0)
        }
        
        # Add per-ball accuracy data
        for ball_type, stats in evaluation_summary.per_ball_accuracy.items():
            data[f'{ball_type}_accuracy'] = stats['accuracy']
            data[f'{ball_type}_total_moments'] = stats['total_moments']
            data[f'{ball_type}_correct_moments'] = stats['correct_moments']
            data[f'{ball_type}_over_detections'] = stats.get('over_detections', 0)
            data[f'{ball_type}_under_detections'] = stats.get('under_detections', 0)
        
        # Add context-based accuracy
        for context, stats in evaluation_summary.context_accuracy.items():
            data[f'{context}_accuracy'] = stats['accuracy']
            data[f'{context}_total_evaluations'] = stats['total_evaluations']
        
        # Convert to DataFrame and save
        df = pd.DataFrame([data])
        df.to_csv(output_path, index=False)
        print(f"Visualization data exported to {output_path}")
    
    def create_summary_dashboard(self, tracker_summaries: Dict[str, EvaluationSummary],
                               save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all key metrics.
        
        Args:
            tracker_summaries: Dictionary mapping tracker names to evaluation summaries
            save_path: Optional path to save the visualization
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create a complex grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Overall accuracy comparison (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        tracker_names = list(tracker_summaries.keys())
        overall_accuracies = [summary.overall_accuracy for summary in tracker_summaries.values()]
        bars = ax1.bar(tracker_names, overall_accuracies, 
                      color=sns.color_palette(self.config.color_palette, len(tracker_names)))
        ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels
        for bar, acc in zip(bars, overall_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Continuity comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        continuity_percentages = [
            summary.continuity_stats.get('continuity_percentage', 0) 
            for summary in tracker_summaries.values()
        ]
        ax2.bar(tracker_names, continuity_percentages, color='lightgreen')
        ax2.set_title('Tracking Continuity', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Continuity (%)')
        ax2.set_ylim(0, 100)
        
        # Per-ball accuracy heatmap (middle left, spans 2 rows)
        ax3 = fig.add_subplot(gs[1:3, :2])
        all_ball_types = set()
        for summary in tracker_summaries.values():
            all_ball_types.update(summary.per_ball_accuracy.keys())
        all_ball_types = sorted(list(all_ball_types))
        
        accuracy_matrix = []
        for tracker_name in tracker_names:
            summary = tracker_summaries[tracker_name]
            row = []
            for ball_type in all_ball_types:
                if ball_type in summary.per_ball_accuracy:
                    row.append(summary.per_ball_accuracy[ball_type]['accuracy'])
                else:
                    row.append(0)
            accuracy_matrix.append(row)
        
        im = ax3.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax3.set_title('Per-Ball Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(all_ball_types)))
        ax3.set_xticklabels(all_ball_types, rotation=45)
        ax3.set_yticks(range(len(tracker_names)))
        ax3.set_yticklabels(tracker_names)
        
        # Add text annotations
        for i in range(len(tracker_names)):
            for j in range(len(all_ball_types)):
                ax3.text(j, i, f'{accuracy_matrix[i][j]:.0f}',
                        ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='Accuracy (%)')
        
        # Error summary (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        error_categories = ['Over Detection', 'Under Detection', 'Illegal Changes', 'Duplications']
        
        # Calculate total errors for each category across all trackers
        total_errors = [0, 0, 0, 0]
        for summary in tracker_summaries.values():
            total_errors[0] += sum(stats.get('over_detections', 0) for stats in summary.per_ball_accuracy.values())
            total_errors[1] += sum(stats.get('under_detections', 0) for stats in summary.per_ball_accuracy.values())
            total_errors[2] += (summary.continuity_stats.get('total_illegal_disappearances', 0) + 
                              summary.continuity_stats.get('total_illegal_reappearances', 0))
            total_errors[3] += summary.duplication_summary.get('total_duplication_errors', 0)
        
        colors = ['orange', 'red', 'purple', 'brown']
        wedges, texts, autotexts = ax4.pie(total_errors, labels=error_categories, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Overall Error Distribution', fontsize=14, fontweight='bold')
        
        # Radar chart (middle right, bottom)
        ax5 = fig.add_subplot(gs[2, 2:], projection='polar')
        self._create_error_radar_chart(ax5, tracker_summaries)
        
        # Summary statistics table (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary table data
        table_data = []
        headers = ['Tracker', 'Overall Acc.', 'Continuity', 'Moments', 'Errors', 'Best Ball Type']
        
        for name, summary in tracker_summaries.items():
            # Find best performing ball type
            best_ball = max(summary.per_ball_accuracy.items(), 
                          key=lambda x: x[1]['accuracy'])
            
            total_errors = (
                sum(stats.get('over_detections', 0) + stats.get('under_detections', 0) 
                   for stats in summary.per_ball_accuracy.values()) +
                summary.continuity_stats.get('total_illegal_disappearances', 0) +
                summary.continuity_stats.get('total_illegal_reappearances', 0) +
                summary.duplication_summary.get('total_duplication_errors', 0)
            )
            
            row = [
                name,
                f"{summary.overall_accuracy:.1f}%",
                f"{summary.continuity_stats.get('continuity_percentage', 0):.1f}%",
                f"{summary.moments_evaluated}/{summary.total_moments}",
                str(total_errors),
                f"{best_ball[0]} ({best_ball[1]['accuracy']:.1f}%)"
            ]
            table_data.append(row)
        
        # Create table
        table = ax6.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.12, 0.12, 0.12, 0.08, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Ground Truth Evaluation Dashboard', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        else:
            plt.show()