"""
Integration tests for GroundTruthEvaluator class.
"""

import unittest
import tempfile
import os
import json
from ground_truth_evaluator import GroundTruthEvaluator, EvaluationSummary


class TestGroundTruthEvaluator(unittest.TestCase):
    """Test cases for GroundTruthEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = GroundTruthEvaluator(
            video_fps=30.0,
            detection_interval=1,
            distance_threshold=50.0
        )
        
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
        
        # Sample tracker results
        self.sample_tracker_results = [
            {
                'timestamp': 5.0,
                'counts': {'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},
                'detections': [
                    {'ball_type': 'red', 'x': 100, 'y': 100, 'confidence': 0.9},
                    {'ball_type': 'yellow', 'x': 200, 'y': 200, 'confidence': 0.8}
                ]
            },
            {
                'timestamp': 11.0,  # During potting event
                'counts': {'red': 14, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},
                'detections': [
                    {'ball_type': 'red', 'x': 150, 'y': 150, 'confidence': 0.85},
                    {'ball_type': 'yellow', 'x': 200, 'y': 200, 'confidence': 0.8}
                ]
            },
            {
                'timestamp': 16.0,  # After placement
                'counts': {'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},
                'detections': [
                    {'ball_type': 'red', 'x': 300, 'y': 300, 'confidence': 0.9},
                    {'ball_type': 'yellow', 'x': 200, 'y': 200, 'confidence': 0.8}
                ]
            }
        ]
    
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.video_fps, 30.0)
        self.assertEqual(self.evaluator.detection_interval, 1)
        self.assertEqual(self.evaluator.distance_threshold, 50.0)
        self.assertFalse(self.evaluator.is_initialized)
        self.assertEqual(len(self.evaluator.evaluation_results), 0)
    
    def test_set_ground_truth_events(self):
        """Test setting ground truth events."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        
        self.assertTrue(self.evaluator.is_initialized)
        self.assertEqual(len(self.evaluator.ground_truth_events), 3)
        self.assertEqual(len(self.evaluator.processed_events), 3)
    
    def test_set_ground_truth_events_invalid(self):
        """Test setting invalid ground truth events."""
        invalid_events = [
            {'invalid': 'event'}  # Missing time specification
        ]
        
        with self.assertRaises(ValueError):
            self.evaluator.set_ground_truth_events(invalid_events)
    
    def test_set_moment_duration(self):
        """Test setting moment duration."""
        self.evaluator.set_moment_duration(2.0)
        self.assertEqual(self.evaluator.moment_duration, 2.0)
        
        with self.assertRaises(ValueError):
            self.evaluator.set_moment_duration(-1.0)
    
    def test_evaluate_tracker_output_not_initialized(self):
        """Test evaluation without initialization."""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
    
    def test_evaluate_tracker_output_empty(self):
        """Test evaluation with empty tracker results."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_tracker_output([])
    
    def test_evaluate_tracker_output_basic(self):
        """Test basic tracker output evaluation."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        self.evaluator.set_moment_duration(5.0)  # 5-second moments
        
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        self.assertIsInstance(summary, EvaluationSummary)
        self.assertGreater(summary.total_moments, 0)
        self.assertGreaterEqual(summary.moments_evaluated, 0)
        self.assertGreaterEqual(summary.overall_accuracy, 0)
        self.assertIsInstance(summary.per_ball_accuracy, dict)
    
    def test_validate_tracker_output_format(self):
        """Test tracker output format validation."""
        # Valid format
        errors = self.evaluator.validate_tracker_output_format(self.sample_tracker_results)
        self.assertEqual(len(errors), 0)
        
        # Invalid format - not a list
        errors = self.evaluator.validate_tracker_output_format("not a list")
        self.assertGreater(len(errors), 0)
        
        # Invalid format - missing timestamp
        invalid_results = [{'counts': {'red': 15}}]
        errors = self.evaluator.validate_tracker_output_format(invalid_results)
        self.assertGreater(len(errors), 0)
        
        # Invalid format - wrong timestamp type
        invalid_results = [{'timestamp': 'not a number', 'counts': {'red': 15}}]
        errors = self.evaluator.validate_tracker_output_format(invalid_results)
        self.assertGreater(len(errors), 0)
    
    def test_get_state_at_time(self):
        """Test getting state at specific time."""
        # Before initialization
        state = self.evaluator.get_state_at_time(10.0)
        self.assertIsNone(state)
        
        # After initialization
        self.evaluator.set_ground_truth_events(self.sample_events)
        state = self.evaluator.get_state_at_time(5.0)
        self.assertIsNotNone(state)
        self.assertEqual(state.ball_counts['red'], 15)
    
    def test_get_moment_evaluation(self):
        """Test getting specific moment evaluation."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        # Valid moment index
        evaluation = self.evaluator.get_moment_evaluation(0)
        self.assertIsNotNone(evaluation)
        
        # Invalid moment index
        evaluation = self.evaluator.get_moment_evaluation(999)
        self.assertIsNone(evaluation)
    
    def test_get_evaluations_for_time_range(self):
        """Test getting evaluations for time range."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        evaluations = self.evaluator.get_evaluations_for_time_range(0.0, 10.0)
        self.assertIsInstance(evaluations, list)
        
        # All evaluations should be within the time range
        for evaluation in evaluations:
            self.assertGreaterEqual(evaluation.timestamp, 0.0)
            self.assertLessEqual(evaluation.timestamp, 10.0)
    
    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        # Before evaluation
        stats = self.evaluator.get_summary_statistics()
        self.assertIn('error', stats)
        
        # After evaluation
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        stats = self.evaluator.get_summary_statistics()
        self.assertIn('total_moments', stats)
        self.assertIn('overall_accuracy', stats)
        self.assertIn('evaluation_time_range', stats)
    
    def test_generate_detailed_report(self):
        """Test generating detailed report."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        # Basic report
        report = self.evaluator.generate_detailed_report(include_moment_details=False)
        self.assertIn('evaluation_metadata', report)
        self.assertIn('summary_statistics', report)
        self.assertIn('accuracy_analysis', report)
        self.assertIn('error_analysis', report)
        self.assertIn('recommendations', report)
        self.assertNotIn('moment_details', report)
        
        # Detailed report
        detailed_report = self.evaluator.generate_detailed_report(include_moment_details=True)
        self.assertIn('moment_details', detailed_report)
        self.assertIsInstance(detailed_report['moment_details'], list)
    
    def test_export_results_to_csv(self):
        """Test exporting results to CSV."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            self.evaluator.export_results_to_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            
            # Check file has content
            with open(csv_path, 'r') as f:
                content = f.read()
                self.assertIn('moment_idx', content)  # Header should be present
                self.assertGreater(len(content.split('\n')), 1)  # Should have data rows
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_export_results_to_json(self):
        """Test exporting results to JSON."""
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            self.evaluator.export_results_to_json(json_path)
            self.assertTrue(os.path.exists(json_path))
            
            # Check file has valid JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.assertIn('evaluation_metadata', data)
                self.assertIn('summary_statistics', data)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_print_summary_report(self):
        """Test printing summary report."""
        # Should not crash with no results
        self.evaluator.print_summary_report()
        
        # Should not crash with results
        self.evaluator.set_ground_truth_events(self.sample_events)
        summary = self.evaluator.evaluate_tracker_output(self.sample_tracker_results)
        self.evaluator.print_summary_report()
    
    def test_transition_period_handling(self):
        """Test handling of transition periods."""
        # Events with time ranges
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
            {
                'time_range': '5.0-7.0',  # 2-second potting event
                'ball_potted': 'red'
            }
        ]
        
        # Tracker results during transition
        transition_tracker_results = [
            {
                'timestamp': 4.0,  # Before transition
                'counts': {'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            },
            {
                'timestamp': 6.0,  # During transition
                'counts': {'red': 14, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            },
            {
                'timestamp': 8.0,  # After transition
                'counts': {'red': 14, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            }
        ]
        
        self.evaluator.set_ground_truth_events(transition_events)
        self.evaluator.set_moment_duration(2.0)  # 2-second moments
        
        summary = self.evaluator.evaluate_tracker_output(transition_tracker_results)
        
        # Should handle transition periods correctly
        self.assertGreater(summary.total_moments, 0)
        
        # Check that transition period analysis is included
        report = self.evaluator.generate_detailed_report()
        self.assertIn('transition_period_analysis', report)
    
    def test_error_categorization(self):
        """Test error categorization functionality."""
        # Create events and tracker results that will generate different error types
        error_events = [
            {
                'time': 0,
                'red': 15,
                'yellow': 1,
                'green': 1,
                'blue': 1,
                'pink': 1,
                'black': 1,
                'white': 1
            }
        ]
        
        error_tracker_results = [
            {
                'timestamp': 5.0,
                'counts': {'red': 18, 'yellow': 0, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},  # Over/under detection
                'detections': [
                    {'ball_type': 'red', 'x': 100, 'y': 100, 'confidence': 0.9},
                    {'ball_type': 'red', 'x': 110, 'y': 105, 'confidence': 0.8},  # Potential duplicate
                ]
            },
            {
                'timestamp': 10.0,
                'counts': {'red': 12, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}  # Illegal change
            }
        ]
        
        self.evaluator.set_ground_truth_events(error_events)
        summary = self.evaluator.evaluate_tracker_output(error_tracker_results)
        
        # Should categorize different types of errors
        self.assertIsInstance(summary.error_categorization, dict)
        
        # Check that error analysis is comprehensive
        report = self.evaluator.generate_detailed_report()
        error_analysis = report['error_analysis']
        
        self.assertIn('error_distribution', error_analysis)
        self.assertIn('by_ball_type', error_analysis['error_distribution'])
        self.assertIn('by_context', error_analysis['error_distribution'])
    
    def test_recommendations_generation(self):
        """Test recommendation generation."""
        # Create scenario that should generate recommendations
        poor_performance_events = [
            {
                'time': 0,
                'red': 15,
                'yellow': 1,
                'green': 1,
                'blue': 1,
                'pink': 1,
                'black': 1,
                'white': 1
            }
        ]
        
        poor_performance_results = [
            {
                'timestamp': 5.0,
                'counts': {'red': 10, 'yellow': 0, 'green': 2, 'blue': 0, 'pink': 2, 'black': 0, 'white': 3}  # Many errors
            }
        ]
        
        self.evaluator.set_ground_truth_events(poor_performance_events)
        summary = self.evaluator.evaluate_tracker_output(poor_performance_results)
        
        report = self.evaluator.generate_detailed_report()
        recommendations = report['recommendations']
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('category', rec)
            self.assertIn('priority', rec)
            self.assertIn('issue', rec)
            self.assertIn('recommendation', rec)


class TestGroundTruthEvaluatorIntegration(unittest.TestCase):
    """Integration tests for complete evaluation pipeline."""
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline with realistic data."""
        evaluator = GroundTruthEvaluator(video_fps=30.0)
        
        # Realistic snooker scenario
        events = [
            # Initial state
            {
                'time': 0,
                'red': 15,
                'yellow': 1,
                'green': 1,
                'brown': 1,
                'blue': 1,
                'pink': 1,
                'black': 1,
                'white': 1
            },
            # Red ball potted over time range
            {
                'time_range': '30.0-32.0',
                'ball_potted': 'red'
            },
            # Occlusion period
            {
                'time_range': '45.0-47.0',
                'balls_occluded': {'red': 2, 'blue': 1}
            },
            # Another red potted
            {
                'time': 60.0,
                'ball_potted': 'red'
            },
            # Yellow potted
            {
                'time_range': '75.0-76.0',
                'ball_potted': 'yellow'
            }
        ]
        
        # Simulate tracker results over time
        tracker_results = []
        for t in range(0, 80, 5):  # Every 5 seconds
            # Simulate realistic detection with some errors
            if t < 30:
                counts = {'red': 15, 'yellow': 1, 'green': 1, 'brown': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            elif t < 45:
                counts = {'red': 14, 'yellow': 1, 'green': 1, 'brown': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            elif t < 60:
                # During occlusion - some balls not detected
                counts = {'red': 12, 'yellow': 1, 'green': 1, 'brown': 1, 'blue': 0, 'pink': 1, 'black': 1, 'white': 1}
            elif t < 75:
                counts = {'red': 13, 'yellow': 1, 'green': 1, 'brown': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            else:
                counts = {'red': 13, 'yellow': 0, 'green': 1, 'brown': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
            
            tracker_results.append({
                'timestamp': float(t),
                'counts': counts,
                'detections': [
                    {'ball_type': ball_type, 'x': 100 + i * 50, 'y': 100 + i * 30, 'confidence': 0.9}
                    for i, (ball_type, count) in enumerate(counts.items())
                    for _ in range(min(count, 3))  # Limit detections for simplicity
                ]
            })
        
        # Run evaluation
        evaluator.set_ground_truth_events(events)
        evaluator.set_moment_duration(10.0)  # 10-second moments
        
        summary = evaluator.evaluate_tracker_output(tracker_results)
        
        # Verify comprehensive evaluation
        self.assertGreater(summary.total_moments, 0)
        self.assertGreaterEqual(summary.overall_accuracy, 0)
        self.assertLessEqual(summary.overall_accuracy, 100)
        
        # Verify different analysis components
        self.assertIsInstance(summary.per_ball_accuracy, dict)
        self.assertIsInstance(summary.context_accuracy, dict)
        self.assertIsInstance(summary.continuity_stats, dict)
        self.assertIsInstance(summary.temporal_analysis, dict)
        
        # Generate and verify detailed report
        report = evaluator.generate_detailed_report(include_moment_details=True)
        
        self.assertIn('evaluation_metadata', report)
        self.assertIn('transition_period_analysis', report)
        self.assertIn('recommendations', report)
        self.assertIn('moment_details', report)
        
        # Verify moment details structure
        moment_details = report['moment_details']
        self.assertGreater(len(moment_details), 0)
        
        for detail in moment_details:
            self.assertIn('moment_idx', detail)
            self.assertIn('timestamp', detail)
            self.assertIn('expected_state', detail)
            self.assertIn('detected_counts', detail)
            self.assertIn('count_errors', detail)
        
        # Test export functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'results.csv')
            json_path = os.path.join(temp_dir, 'results.json')
            
            evaluator.export_results_to_csv(csv_path)
            evaluator.export_results_to_json(json_path)
            
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))
            
            # Verify JSON export
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
                self.assertIn('evaluation_metadata', exported_data)
                self.assertIn('summary_statistics', exported_data)


if __name__ == '__main__':
    unittest.main()