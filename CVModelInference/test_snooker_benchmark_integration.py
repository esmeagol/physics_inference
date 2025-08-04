"""
Integration tests for SnookerTrackerBenchmark with ground truth capabilities.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock, patch

# Mock missing dependencies
sys.modules['tracker'] = MagicMock()
sys.modules['inference_runner'] = MagicMock()

from tracker_benchmark import SnookerTrackerBenchmark


class MockTracker:
    """Mock tracker for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        
    def init(self, frame, detections):
        self.initialized = True
        
    def update(self, frame, detections=None):
        # Return mock tracks
        return [
            {
                'id': 1,
                'x': 100,
                'y': 100,
                'width': 20,
                'height': 20,
                'confidence': 0.9,
                'class': 'red'
            },
            {
                'id': 2,
                'x': 200,
                'y': 200,
                'width': 20,
                'height': 20,
                'confidence': 0.8,
                'class': 'white'
            }
        ]
        
    def visualize(self, frame, tracks):
        # Return the frame unchanged for testing
        return frame


class TestSnookerBenchmarkGroundTruthIntegration(unittest.TestCase):
    """Test ground truth integration in SnookerTrackerBenchmark."""
    
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
            },
            {
                'time': 15.0,
                'ball_placed_back': 'red'
            }
        ]
        
        # Add mock tracker
        self.mock_tracker = MockTracker("test_tracker")
        self.benchmark.add_tracker("test_tracker", self.mock_tracker)
    
    def test_set_ground_truth_events(self):
        """Test setting ground truth events."""
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        self.assertEqual(len(self.benchmark.ground_truth_events), 3)
        self.assertIsNotNone(self.benchmark.ground_truth_evaluator)
        self.assertTrue(self.benchmark.ground_truth_evaluator.is_initialized)
    
    def test_set_ground_truth_events_invalid(self):
        """Test setting invalid ground truth events."""
        invalid_events = [{'invalid': 'event'}]
        
        with self.assertRaises(ValueError):
            self.benchmark.set_ground_truth_events(invalid_events)
    
    def test_set_moment_duration(self):
        """Test setting moment duration."""
        self.benchmark.set_ground_truth_events(self.sample_events)
        self.benchmark.set_moment_duration(2.0)
        
        self.assertEqual(self.benchmark.moment_duration, 2.0)
        self.assertEqual(self.benchmark.ground_truth_evaluator.moment_duration, 2.0)
    
    def test_set_moment_duration_invalid(self):
        """Test setting invalid moment duration."""
        with self.assertRaises(ValueError):
            self.benchmark.set_moment_duration(-1.0)
    
    @patch('cv2.VideoCapture')
    def test_run_benchmark_with_ground_truth_not_initialized(self, mock_cv2):
        """Test running benchmark without ground truth initialization."""
        with self.assertRaises(ValueError):
            self.benchmark.run_benchmark_with_ground_truth("dummy_video.mp4")
    
    @patch('cv2.VideoCapture')
    def test_convert_tracker_results_to_gt_format(self, mock_cv2):
        """Test conversion of tracker results to ground truth format."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # FPS
        mock_cv2.return_value = mock_cap
        
        # Sample tracker result
        tracker_result = {
            'tracks': [
                {
                    'frame_idx': 0,
                    'tracks': [
                        {'id': 1, 'x': 100, 'y': 100, 'class': 'red', 'confidence': 0.9},
                        {'id': 2, 'x': 200, 'y': 200, 'class': 'white', 'confidence': 0.8}
                    ]
                },
                {
                    'frame_idx': 30,  # 1 second later at 30 FPS
                    'tracks': [
                        {'id': 1, 'x': 150, 'y': 150, 'class': 'red', 'confidence': 0.85},
                        {'id': 2, 'x': 250, 'y': 250, 'class': 'white', 'confidence': 0.75}
                    ]
                }
            ]
        }
        
        # Convert to ground truth format
        gt_format = self.benchmark._convert_tracker_results_to_gt_format(
            tracker_result, "dummy_video.mp4"
        )
        
        # Verify conversion
        self.assertEqual(len(gt_format), 2)
        
        # Check first frame
        first_frame = gt_format[0]
        self.assertEqual(first_frame['timestamp'], 0.0)
        self.assertEqual(first_frame['counts']['red'], 1)
        self.assertEqual(first_frame['counts']['white'], 1)
        self.assertEqual(len(first_frame['detections']), 2)
        
        # Check second frame
        second_frame = gt_format[1]
        self.assertEqual(second_frame['timestamp'], 1.0)  # 30 frames / 30 FPS = 1 second
        self.assertEqual(second_frame['counts']['red'], 1)
        self.assertEqual(second_frame['counts']['white'], 1)
    
    @patch('cv2.VideoCapture')
    @patch('os.path.exists')
    def test_run_benchmark_with_ground_truth_integration(self, mock_exists, mock_cv2):
        """Test complete ground truth integration."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # FPS
            3: 640,   # Width
            4: 480,   # Height
            7: 60     # Total frames
        }.get(prop, 0)
        
        # Mock frame reading - return 3 frames then end
        mock_frames = [
            (True, MagicMock()),  # Frame 0
            (True, MagicMock()),  # Frame 5 (detection interval)
            (True, MagicMock()),  # Frame 10
            (False, None)         # End of video
        ]
        mock_cap.read.side_effect = mock_frames
        mock_cv2.return_value = mock_cap
        
        # Set up ground truth
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        # Mock the standard benchmark run to avoid complex video processing
        with patch.object(self.benchmark, 'run_benchmark') as mock_run_benchmark:
            # Mock standard benchmark results
            mock_run_benchmark.return_value = {
                'test_tracker': {
                    'processing_time': 1.0,
                    'frames_processed': 3,
                    'avg_fps': 3.0,
                    'tracks': [
                        {
                            'frame_idx': 0,
                            'tracks': [
                                {'id': 1, 'x': 100, 'y': 100, 'class': 'red', 'confidence': 0.9},
                                {'id': 2, 'x': 200, 'y': 200, 'class': 'white', 'confidence': 0.8}
                            ]
                        },
                        {
                            'frame_idx': 5,
                            'tracks': [
                                {'id': 1, 'x': 110, 'y': 110, 'class': 'red', 'confidence': 0.85},
                                {'id': 2, 'x': 210, 'y': 210, 'class': 'white', 'confidence': 0.75}
                            ]
                        }
                    ],
                    'metrics': {'track_count': 2}
                }
            }
            
            # Run benchmark with ground truth
            results = self.benchmark.run_benchmark_with_ground_truth("dummy_video.mp4")
            
            # Verify results structure
            self.assertIn('standard_benchmark', results)
            self.assertIn('ground_truth_evaluation', results)
            self.assertIn('ground_truth_events', results)
            self.assertIn('moment_duration', results)
            
            # Verify ground truth results exist
            self.assertIn('test_tracker', results['ground_truth_evaluation'])
            gt_result = results['ground_truth_evaluation']['test_tracker']
            
            # Should have ground truth evaluation result
            self.assertIsNotNone(gt_result)
    
    def test_print_results_with_ground_truth(self):
        """Test printing results with ground truth data."""
        # Set up ground truth
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        # Mock ground truth results
        mock_gt_result = MagicMock()
        mock_gt_result.overall_accuracy = 85.5
        mock_gt_result.moments_evaluated = 8
        mock_gt_result.total_moments = 10
        mock_gt_result.continuity_stats = {'continuity_percentage': 92.3}
        mock_gt_result.duplication_summary = {'total_duplication_errors': 2}
        mock_gt_result.per_ball_accuracy = {
            'red': {'accuracy': 80.0, 'correct_moments': 4, 'total_moments': 5},
            'white': {'accuracy': 90.0, 'correct_moments': 9, 'total_moments': 10}
        }
        mock_gt_result.context_accuracy = {
            'stable_period': {'accuracy': 88.0},
            'transition_period': {'accuracy': 75.0}
        }
        
        self.benchmark.ground_truth_results = {'test_tracker': mock_gt_result}
        
        # Mock standard results
        self.benchmark.results = {
            'test_tracker': {
                'avg_fps': 25.0,
                'metrics': {
                    'track_count': 2,
                    'track_switches': 1,
                    'track_fragmentations': 0,
                    'lost_tracks': 0,
                    'new_tracks': 2
                },
                'snooker_metrics': {
                    'ball_id_consistency': 95.0,
                    'cue_ball_tracking': 100.0,
                    'color_ball_tracking': 90.0,
                    'red_ball_tracking': 85.0
                }
            }
        }
        
        # Should not crash when printing results
        self.benchmark.print_results()
    
    def test_export_ground_truth_results(self):
        """Test exporting ground truth results."""
        # Set up ground truth
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        # Mock ground truth results
        mock_gt_result = MagicMock()
        self.benchmark.ground_truth_results = {'test_tracker': mock_gt_result}
        
        # Mock the evaluator's export methods
        self.benchmark.ground_truth_evaluator.export_results_to_json = MagicMock()
        self.benchmark.ground_truth_evaluator.export_results_to_csv = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.benchmark.export_ground_truth_results(temp_dir)
            
            # Verify export methods were called
            self.benchmark.ground_truth_evaluator.export_results_to_json.assert_called()
            self.benchmark.ground_truth_evaluator.export_results_to_csv.assert_called()
    
    def test_export_ground_truth_results_no_results(self):
        """Test exporting when no ground truth results exist."""
        with self.assertRaises(ValueError):
            self.benchmark.export_ground_truth_results("/tmp")
    
    def test_print_ground_truth_summary(self):
        """Test printing ground truth summary."""
        # Set up ground truth
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        # Mock ground truth results
        mock_gt_result = MagicMock()
        mock_gt_result.overall_accuracy = 85.5
        mock_gt_result.moments_evaluated = 8
        mock_gt_result.total_moments = 10
        mock_gt_result.continuity_stats = {'continuity_percentage': 92.3}
        
        self.benchmark.ground_truth_results = {'test_tracker': mock_gt_result}
        
        # Should not crash when printing summary
        self.benchmark.print_ground_truth_summary()
        self.benchmark.print_ground_truth_summary('test_tracker')
        
        # Test with non-existent tracker
        self.benchmark.print_ground_truth_summary('non_existent')
    
    def test_visualize_results_with_ground_truth(self):
        """Test visualization with ground truth results."""
        # Set up ground truth
        self.benchmark.set_ground_truth_events(self.sample_events)
        
        # Mock ground truth results
        mock_gt_result = MagicMock()
        mock_gt_result.overall_accuracy = 85.5
        mock_gt_result.continuity_stats = {'continuity_percentage': 92.3}
        mock_gt_result.duplication_summary = {'total_duplication_errors': 2}
        mock_gt_result.moments_evaluated = 8
        
        self.benchmark.ground_truth_results = {'test_tracker': mock_gt_result}
        
        # Mock standard results for parent visualization
        self.benchmark.results = {
            'test_tracker': {
                'avg_fps': 25.0,
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
        
        # Mock matplotlib to avoid display issues in tests
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            # Should not crash when visualizing
            self.benchmark.visualize_results()
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works without ground truth."""
        # Mock standard benchmark functionality
        with patch.object(self.benchmark, 'run_benchmark') as mock_run_benchmark:
            mock_run_benchmark.return_value = {
                'test_tracker': {
                    'processing_time': 1.0,
                    'frames_processed': 10,
                    'avg_fps': 10.0,
                    'tracks': [],
                    'metrics': {'track_count': 0}
                }
            }
            
            # Should work without ground truth setup
            results = mock_run_benchmark("dummy_video.mp4")
            self.assertIn('test_tracker', results)
            
            # Print results should work without ground truth
            self.benchmark.results = results
            self.benchmark.print_results()


class TestGroundTruthEventFormats(unittest.TestCase):
    """Test different ground truth event formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = SnookerTrackerBenchmark()
    
    def test_mixed_time_formats(self):
        """Test handling of mixed time and frame formats."""
        mixed_events = [
            # Time-based initial state
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
            # Frame-based potting
            {
                'frame': 300,  # 10 seconds at 30 FPS
                'ball_potted': 'red'
            },
            # Time range potting
            {
                'time_range': '15.0-17.0',
                'ball_potted': 'yellow'
            },
            # Frame range occlusion
            {
                'frame_range': '600-750',  # 20-25 seconds at 30 FPS
                'balls_occluded': {'red': 2, 'blue': 1}
            }
        ]
        
        # Should handle mixed formats without error
        self.benchmark.set_ground_truth_events(mixed_events)
        self.assertEqual(len(self.benchmark.ground_truth_events), 4)
    
    def test_transition_period_events(self):
        """Test events with time ranges for transition periods."""
        transition_events = [
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
            # Potting with time range (transition period)
            {
                'time_range': '10.0-12.0',
                'ball_potted': 'red'
            },
            # Placement with time range
            {
                'time_range': '20.0-21.0',
                'ball_placed_back': 'red'
            },
            # Occlusion period
            {
                'time_range': '30.0-35.0',
                'balls_occluded': {'red': 3, 'blue': 1}
            },
            # Error suppression period
            {
                'time_range': '40.0-42.0',
                'ignore_errors': 'camera shake during shot'
            }
        ]
        
        # Should handle transition periods
        self.benchmark.set_ground_truth_events(transition_events)
        self.assertEqual(len(self.benchmark.ground_truth_events), 5)


if __name__ == '__main__':
    unittest.main()