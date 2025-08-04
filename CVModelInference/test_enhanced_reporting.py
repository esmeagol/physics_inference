"""
Tests for enhanced visualization and reporting capabilities.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock missing dependencies
sys.modules['tracker'] = MagicMock()
sys.modules['inference_runner'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.patches'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['pandas'] = MagicMock()

from tracker_benchmark import SnookerTrackerBenchmark
from ground_truth_visualizer import VisualizationConfig


class TestEnhancedReporting(unittest.TestCase):
    """Test enhanced reporting and visualization capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = SnookerTrackerBenchmark()
        
        # Sample ground truth events
        self.sample_events = [
            {
                'time': 0,
                'red': 15,
                'yellow': 1,
                'green': 1,
                'blue': 1,
                'pink': 1,
                'black': 1,
                'white': 1
            },
            {
                'time_range': '10.0-12.0',
                'ball_potted': 'red'
            }
        ]
        
        # Set up ground truth
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        # Mock ground truth results
        mock_gt_result = MagicMock()
        mock_gt_result.overall_accuracy = 85.5
        mock_gt_result.moments_evaluated = 8
        mock_gt_result.total_moments = 10
        mock_gt_result.moments_suppressed = 2
        mock_gt_result.continuity_stats = {
            'continuity_percentage': 92.3,
            'total_illegal_disappearances': 1,
            'total_illegal_reappearances': 0
        }
        mock_gt_result.duplication_summary = {'total_duplication_errors': 2}
        mock_gt_result.per_ball_accuracy = {
            'red': {'accuracy': 80.0, 'correct_moments': 4, 'total_moments': 5, 'over_detections': 1, 'under_detections': 0},
            'white': {'accuracy': 90.0, 'correct_moments': 9, 'total_moments': 10, 'over_detections': 0, 'under_detections': 1}
        }
        mock_gt_result.context_accuracy = {
            'stable_period': {'accuracy': 88.0, 'total_evaluations': 50, 'correct_evaluations': 44},
            'transition_period': {'accuracy': 75.0, 'total_evaluations': 20, 'correct_evaluations': 15}
        }
        mock_gt_result.temporal_analysis = {
            'accuracy_over_time': [85, 90, 80, 85, 88],
            'timestamps': [0, 5, 10, 15, 20]
        }
        
        self.benchmark.ground_truth_results = {'test_tracker': mock_gt_result}
        
        # Mock standard results
        self.benchmark.results = {
            'test_tracker': {
                'processing_time': 10.0,
                'frames_processed': 100,
                'avg_fps': 25.0,
                'tracks': [],
                'metrics': {
                    'track_count': 2,
                    'track_switches': 1,
                    'track_fragmentations': 0,
                    'lost_tracks': 0,
                    'new_tracks': 2,
                    'avg_track_length': 10.0
                },
                'snooker_metrics': {
                    'ball_id_consistency': 95.0,
                    'cue_ball_tracking': 100.0,
                    'color_ball_tracking': 90.0,
                    'red_ball_tracking': 85.0
                }
            }
        }
    
    def test_set_visualization_config(self):
        """Test setting custom visualization configuration."""
        config = VisualizationConfig(
            figure_size=(12, 8),
            color_palette='viridis',
            font_size=10
        )
        
        self.benchmark.set_visualization_config(config)
        self.assertEqual(self.benchmark.visualizer.config.figure_size, (12, 8))
        self.assertEqual(self.benchmark.visualizer.config.color_palette, 'viridis')
        self.assertEqual(self.benchmark.visualizer.config.font_size, 10)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_ground_truth_timeline(self, mock_savefig, mock_show):
        """Test timeline visualization."""
        # Test without save path
        self.benchmark.visualize_ground_truth_timeline('test_tracker')
        mock_show.assert_called()
        
        # Test with save path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name
        
        try:
            self.benchmark.visualize_ground_truth_timeline('test_tracker', save_path)
            mock_savefig.assert_called()
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_visualize_ground_truth_timeline_no_results(self):
        """Test timeline visualization with no results."""
        benchmark = SnookerTrackerBenchmark()
        
        with self.assertRaises(ValueError):
            benchmark.visualize_ground_truth_timeline('nonexistent_tracker')
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_accuracy_analysis(self, mock_savefig, mock_show):
        """Test accuracy analysis visualization."""
        self.benchmark.visualize_accuracy_analysis('test_tracker')
        mock_show.assert_called()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_error_distribution(self, mock_savefig, mock_show):
        """Test error distribution visualization."""
        self.benchmark.visualize_error_distribution('test_tracker')
        mock_show.assert_called()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_comparative_analysis(self, mock_savefig, mock_show):
        """Test comparative analysis visualization."""
        # Add second tracker for comparison
        mock_gt_result2 = MagicMock()
        mock_gt_result2.overall_accuracy = 78.2
        mock_gt_result2.continuity_stats = {'continuity_percentage': 85.0}
        mock_gt_result2.duplication_summary = {'total_duplication_errors': 1}
        mock_gt_result2.moments_evaluated = 9
        mock_gt_result2.per_ball_accuracy = {
            'red': {'accuracy': 75.0},
            'white': {'accuracy': 82.0}
        }
        
        self.benchmark.ground_truth_results['test_tracker2'] = mock_gt_result2
        
        self.benchmark.visualize_comparative_analysis()
        mock_show.assert_called()
    
    def test_visualize_comparative_analysis_insufficient_data(self):
        """Test comparative analysis with insufficient data."""
        with self.assertRaises(ValueError):
            self.benchmark.visualize_comparative_analysis()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_comprehensive_dashboard(self, mock_savefig, mock_show):
        """Test comprehensive dashboard creation."""
        self.benchmark.create_comprehensive_dashboard()
        mock_show.assert_called()
    
    def test_create_comprehensive_dashboard_no_results(self):
        """Test dashboard creation with no results."""
        benchmark = SnookerTrackerBenchmark()
        
        with self.assertRaises(ValueError):
            benchmark.create_comprehensive_dashboard()
    
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_generate_comprehensive_report(self, mock_exists, mock_makedirs, mock_open):
        """Test comprehensive report generation."""
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock the ground truth evaluator's methods
        self.benchmark.ground_truth_evaluator.export_results_to_json = MagicMock()
        self.benchmark.ground_truth_evaluator.export_results_to_csv = MagicMock()
        self.benchmark.ground_truth_evaluator.generate_detailed_report = MagicMock()
        self.benchmark.ground_truth_evaluator.generate_detailed_report.return_value = {
            'recommendations': [
                {'priority': 'high', 'issue': 'Test issue', 'recommendation': 'Test recommendation'}
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock visualization methods to avoid matplotlib issues
            with patch.object(self.benchmark, 'visualize_ground_truth_timeline'), \
                 patch.object(self.benchmark, 'visualize_accuracy_analysis'), \
                 patch.object(self.benchmark, 'visualize_error_distribution'), \
                 patch.object(self.benchmark.visualizer, 'export_visualization_data'):
                
                self.benchmark.generate_comprehensive_report(temp_dir)
                
                # Verify methods were called
                mock_makedirs.assert_called_with(temp_dir)
                self.benchmark.ground_truth_evaluator.export_results_to_json.assert_called()
                self.benchmark.ground_truth_evaluator.export_results_to_csv.assert_called()
    
    def test_generate_comprehensive_report_no_results(self):
        """Test report generation with no results."""
        benchmark = SnookerTrackerBenchmark()
        
        with self.assertRaises(ValueError):
            benchmark.generate_comprehensive_report('/tmp')
    
    def test_enhanced_print_results(self):
        """Test enhanced results printing."""
        # Should not crash when printing enhanced results
        with patch('builtins.print'):
            self.benchmark.print_results()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_enhanced_visualize_results(self, mock_savefig, mock_show):
        """Test enhanced visualization results."""
        # Mock parent visualization method
        with patch('tracker_benchmark.TrackerBenchmark.visualize_results'):
            self.benchmark.visualize_results()
            mock_show.assert_called()
    
    def test_enhanced_visualize_results_single_tracker(self):
        """Test enhanced visualization with single tracker."""
        with patch('tracker_benchmark.TrackerBenchmark.visualize_results'), \
             patch.object(self.benchmark, 'visualize_accuracy_analysis') as mock_viz:
            
            self.benchmark.visualize_results()
            mock_viz.assert_called_with('test_tracker', None)
    
    def test_enhanced_visualize_results_multiple_trackers(self):
        """Test enhanced visualization with multiple trackers."""
        # Add second tracker
        mock_gt_result2 = MagicMock()
        mock_gt_result2.overall_accuracy = 78.2
        self.benchmark.ground_truth_results['test_tracker2'] = mock_gt_result2
        
        with patch('tracker_benchmark.TrackerBenchmark.visualize_results'), \
             patch.object(self.benchmark, 'create_comprehensive_dashboard') as mock_dashboard:
            
            self.benchmark.visualize_results()
            mock_dashboard.assert_called_with(None)
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization methods."""
        # Test with failed ground truth result
        self.benchmark.ground_truth_results['test_tracker'] = None
        
        with self.assertRaises(ValueError):
            self.benchmark.visualize_ground_truth_timeline('test_tracker')
        
        with self.assertRaises(ValueError):
            self.benchmark.visualize_accuracy_analysis('test_tracker')
        
        with self.assertRaises(ValueError):
            self.benchmark.visualize_error_distribution('test_tracker')


class TestVisualizationConfig(unittest.TestCase):
    """Test visualization configuration."""
    
    def test_default_config(self):
        """Test default visualization configuration."""
        config = VisualizationConfig()
        
        self.assertEqual(config.figure_size, (15, 10))
        self.assertEqual(config.color_palette, 'Set2')
        self.assertEqual(config.font_size, 12)
        self.assertEqual(config.title_font_size, 14)
        self.assertEqual(config.dpi, 300)
        self.assertEqual(config.style, 'whitegrid')
    
    def test_custom_config(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            figure_size=(20, 12),
            color_palette='viridis',
            font_size=14,
            title_font_size=16,
            dpi=150,
            style='darkgrid'
        )
        
        self.assertEqual(config.figure_size, (20, 12))
        self.assertEqual(config.color_palette, 'viridis')
        self.assertEqual(config.font_size, 14)
        self.assertEqual(config.title_font_size, 16)
        self.assertEqual(config.dpi, 150)
        self.assertEqual(config.style, 'darkgrid')


if __name__ == '__main__':
    unittest.main()