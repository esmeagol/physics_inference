"""
Unit tests for MomentEvaluator class.
"""

import unittest
from unittest.mock import Mock, MagicMock
import math
from moment_evaluator import (
    MomentEvaluator, ErrorType, CountError, IllegalChange, DuplicationError,
    MomentEvaluation
)
from event_processor import EventProcessor, ProcessedEvent, EventType
from state_reconstructor import StateReconstructor, BallState


class TestMomentEvaluator(unittest.TestCase):
    """Test cases for MomentEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock event processor and state reconstructor
        self.mock_event_processor = Mock(spec=EventProcessor)
        self.mock_state_reconstructor = Mock(spec=StateReconstructor)
        self.mock_state_reconstructor.event_processor = self.mock_event_processor
        self.mock_state_reconstructor.processed_events = []
        
        # Create evaluator
        self.evaluator = MomentEvaluator(
            state_reconstructor=self.mock_state_reconstructor,
            distance_threshold=50.0
        )
        
        # Sample ball state
        self.sample_state = BallState(
            time=10.0,
            ball_counts={'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},
            occluded_balls={},
            ignore_errors=False
        )
    
    def test_evaluate_moment_basic(self):
        """Test basic moment evaluation."""
        # Setup mocks
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Test data
        detected_counts = {'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
        
        # Evaluate moment
        result = self.evaluator.evaluate_moment(
            moment_idx=0,
            timestamp=10.0,
            detected_counts=detected_counts
        )
        
        # Verify result
        self.assertIsInstance(result, MomentEvaluation)
        self.assertEqual(result.moment_idx, 0)
        self.assertEqual(result.timestamp, 10.0)
        self.assertEqual(result.detected_counts, detected_counts)
        self.assertFalse(result.suppressed)
        
        # Should have no errors for perfect match
        self.assertEqual(len(result.count_errors), 7)  # One for each ball type
        for error in result.count_errors:
            self.assertEqual(error.error_type, ErrorType.CORRECT)
    
    def test_evaluate_moment_with_errors_suppressed(self):
        """Test moment evaluation with error suppression."""
        # Setup state with error suppression
        suppressed_state = BallState(
            time=10.0,
            ball_counts=self.sample_state.ball_counts,
            occluded_balls={},
            ignore_errors=True
        )
        
        self.mock_state_reconstructor.get_state_at_time.return_value = suppressed_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Test with incorrect counts
        detected_counts = {'red': 10, 'yellow': 0}  # Wrong counts
        
        # Evaluate moment
        result = self.evaluator.evaluate_moment(
            moment_idx=0,
            timestamp=10.0,
            detected_counts=detected_counts
        )
        
        # Should be suppressed
        self.assertTrue(result.suppressed)
        self.assertEqual(len(result.count_errors), 0)  # No errors when suppressed
    
    def test_count_comparison_over_detection(self):
        """Test over-detection error identification."""
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # More balls detected than expected
        detected_counts = {'red': 18, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
        
        result = self.evaluator.evaluate_moment(0, 10.0, detected_counts)
        
        # Find red ball error
        red_errors = [e for e in result.count_errors if e.ball_type == 'red']
        self.assertEqual(len(red_errors), 1)
        
        error = red_errors[0]
        self.assertEqual(error.error_type, ErrorType.OVER_DETECTION)
        self.assertEqual(error.expected, 15)
        self.assertEqual(error.detected, 18)
        self.assertEqual(error.error_magnitude, 3)
    
    def test_count_comparison_under_detection(self):
        """Test under-detection error identification."""
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Fewer balls detected than expected
        detected_counts = {'red': 12, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
        
        result = self.evaluator.evaluate_moment(0, 10.0, detected_counts)
        
        # Find red ball error
        red_errors = [e for e in result.count_errors if e.ball_type == 'red']
        self.assertEqual(len(red_errors), 1)
        
        error = red_errors[0]
        self.assertEqual(error.error_type, ErrorType.UNDER_DETECTION)
        self.assertEqual(error.expected, 15)
        self.assertEqual(error.detected, 12)
        self.assertEqual(error.error_magnitude, 3)
    
    def test_count_comparison_with_occlusion(self):
        """Test count comparison with occluded balls."""
        # State with occluded balls
        occluded_state = BallState(
            time=10.0,
            ball_counts={'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},
            occluded_balls={'red': 3, 'blue': 1},  # 3 reds and 1 blue occluded
            ignore_errors=False
        )
        
        self.mock_state_reconstructor.get_state_at_time.return_value = occluded_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Detected counts should account for occlusion
        detected_counts = {'red': 12, 'yellow': 1, 'green': 1, 'blue': 0, 'pink': 1, 'black': 1, 'white': 1}
        
        result = self.evaluator.evaluate_moment(0, 10.0, detected_counts)
        
        # Should be correct (15-3=12 reds, 1-1=0 blue)
        red_errors = [e for e in result.count_errors if e.ball_type == 'red']
        blue_errors = [e for e in result.count_errors if e.ball_type == 'blue']
        
        self.assertEqual(len(red_errors), 1)
        self.assertEqual(red_errors[0].error_type, ErrorType.CORRECT)
        
        self.assertEqual(len(blue_errors), 1)
        self.assertEqual(blue_errors[0].error_type, ErrorType.CORRECT)
    
    def test_transition_period_flexibility(self):
        """Test flexible count validation during transition periods."""
        # Create mock potting event
        potting_event = Mock(spec=ProcessedEvent)
        potting_event.event_type = EventType.BALL_POTTED
        potting_event.data = {'ball_potted': 'red'}
        potting_event.end_time = 10.5  # Range event
        potting_event.start_time = 10.0
        
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = [potting_event]
        
        # Mock the before state for transition logic
        before_state = BallState(
            time=9.9,
            ball_counts={'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1},
            occluded_balls={},
            ignore_errors=False
        )
        self.mock_state_reconstructor.get_expected_state_for_event.return_value = before_state
        
        # During potting: 14 or 15 reds should be acceptable
        detected_counts = {'red': 14, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
        
        result = self.evaluator.evaluate_moment(0, 10.25, detected_counts)
        
        # Should be correct during transition
        red_errors = [e for e in result.count_errors if e.ball_type == 'red']
        self.assertEqual(len(red_errors), 1)
        self.assertEqual(red_errors[0].error_type, ErrorType.CORRECT)
        self.assertEqual(red_errors[0].context, "transition_period_potting")
    
    def test_illegal_change_detection(self):
        """Test illegal change detection."""
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # First moment
        detected_counts_1 = {'red': 15, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
        result_1 = self.evaluator.evaluate_moment(0, 10.0, detected_counts_1)
        
        # Second moment with illegal disappearance
        detected_counts_2 = {'red': 13, 'yellow': 1, 'green': 1, 'blue': 1, 'pink': 1, 'black': 1, 'white': 1}
        result_2 = self.evaluator.evaluate_moment(1, 10.5, detected_counts_2)
        
        # Should detect illegal disappearance
        self.assertEqual(len(result_2.illegal_changes), 1)
        
        change = result_2.illegal_changes[0]
        self.assertEqual(change.change_type, ErrorType.ILLEGAL_DISAPPEARANCE)
        self.assertEqual(change.ball_type, 'red')
        self.assertEqual(change.previous_count, 15)
        self.assertEqual(change.current_count, 13)
    
    def test_duplication_detection(self):
        """Test duplication detection using spatial analysis."""
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Detected positions with duplicates (two reds very close together)
        detected_positions = [
            {'ball_type': 'red', 'x': 100, 'y': 100, 'confidence': 0.9},
            {'ball_type': 'red', 'x': 110, 'y': 105, 'confidence': 0.8},  # Too close (distance ~11)
            {'ball_type': 'red', 'x': 200, 'y': 200, 'confidence': 0.95},  # Far enough
            {'ball_type': 'yellow', 'x': 300, 'y': 300, 'confidence': 0.85}
        ]
        
        detected_counts = {'red': 3, 'yellow': 1}
        
        result = self.evaluator.evaluate_moment(
            0, 10.0, detected_counts, detected_positions
        )
        
        # Should detect duplication
        self.assertEqual(len(result.duplication_errors), 1)
        
        dup_error = result.duplication_errors[0]
        self.assertEqual(dup_error.ball_type, 'red')
        self.assertEqual(len(dup_error.positions), 2)  # Two close positions
        self.assertTrue(all(d < 50 for d in dup_error.distances))  # All distances below threshold
    
    def test_per_ball_accuracy_calculation(self):
        """Test per-ball-type accuracy calculation."""
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Simulate several moments with different accuracy
        test_cases = [
            {'red': 15, 'yellow': 1},  # Correct
            {'red': 14, 'yellow': 1},  # Red under-detection
            {'red': 15, 'yellow': 2},  # Yellow over-detection
            {'red': 15, 'yellow': 1},  # Correct
        ]
        
        for i, detected_counts in enumerate(test_cases):
            self.evaluator.evaluate_moment(i, 10.0 + i, detected_counts)
        
        # Calculate accuracy
        accuracy = self.evaluator.calculate_per_ball_accuracy()
        
        # Red: 3/4 correct = 75%
        self.assertAlmostEqual(accuracy['red']['accuracy'], 75.0)
        self.assertEqual(accuracy['red']['total_moments'], 4)
        self.assertEqual(accuracy['red']['correct_moments'], 3)
        self.assertEqual(accuracy['red']['under_detections'], 1)
        
        # Yellow: 3/4 correct = 75%
        self.assertAlmostEqual(accuracy['yellow']['accuracy'], 75.0)
        self.assertEqual(accuracy['yellow']['over_detections'], 1)
    
    def test_spatial_distribution_analysis(self):
        """Test spatial distribution analysis."""
        detected_positions = [
            {'ball_type': 'red', 'x': 100, 'y': 100, 'confidence': 0.9},
            {'ball_type': 'red', 'x': 200, 'y': 150, 'confidence': 0.8},
            {'ball_type': 'red', 'x': 150, 'y': 200, 'confidence': 0.95},
            {'ball_type': 'yellow', 'x': 300, 'y': 300, 'confidence': 0.85}
        ]
        
        analysis = self.evaluator.analyze_spatial_distribution(detected_positions)
        
        # Check red ball analysis
        red_analysis = analysis['red']
        self.assertEqual(red_analysis['count'], 3)
        self.assertEqual(red_analysis['x_range'], (100, 200))
        self.assertEqual(red_analysis['y_range'], (100, 200))
        self.assertAlmostEqual(red_analysis['x_mean'], 150.0)
        self.assertAlmostEqual(red_analysis['y_mean'], 150.0)
        
        # Check yellow ball analysis
        yellow_analysis = analysis['yellow']
        self.assertEqual(yellow_analysis['count'], 1)
        self.assertEqual(yellow_analysis['x_mean'], 300)
        self.assertEqual(yellow_analysis['y_mean'], 300)
    
    def test_missing_object_detection(self):
        """Test missing object detection in expected regions."""
        detected_positions = [
            {'ball_type': 'red', 'x': 100, 'y': 100, 'confidence': 0.9},
            {'ball_type': 'yellow', 'x': 300, 'y': 300, 'confidence': 0.85}
        ]
        
        expected_regions = {
            'red': [(95, 105), (500, 500)],  # One near detection, one far
            'yellow': [(295, 305)],  # Near detection
            'blue': [(400, 400)]  # No detection nearby
        }
        
        missing = self.evaluator.detect_missing_objects_in_regions(
            detected_positions, expected_regions
        )
        
        # Should find missing red at (500, 500) and missing blue at (400, 400)
        self.assertEqual(len(missing), 2)
        
        missing_types = [obj['ball_type'] for obj in missing]
        self.assertIn('red', missing_types)
        self.assertIn('blue', missing_types)
    
    def test_tracking_continuity_analysis(self):
        """Test tracking continuity analysis."""
        self.mock_state_reconstructor.get_state_at_time.return_value = self.sample_state
        self.mock_event_processor.get_events_at_time.return_value = []
        
        # Simulate moments with continuity breaks
        test_cases = [
            {'red': 15, 'yellow': 1},  # Correct
            {'red': 13, 'yellow': 1},  # Illegal disappearance
            {'red': 15, 'yellow': 1},  # Illegal reappearance
            {'red': 15, 'yellow': 1},  # Correct
        ]
        
        for i, detected_counts in enumerate(test_cases):
            self.evaluator.evaluate_moment(i, 10.0 + i, detected_counts)
        
        # Analyze continuity
        continuity = self.evaluator.analyze_tracking_continuity()
        
        self.assertEqual(continuity['total_moments'], 4)
        self.assertEqual(continuity['moments_with_illegal_changes'], 2)
        self.assertEqual(continuity['total_illegal_disappearances'], 1)
        self.assertEqual(continuity['total_illegal_reappearances'], 1)
        self.assertAlmostEqual(continuity['continuity_percentage'], 50.0)  # 2/4 moments had issues
    
    def test_context_based_accuracy(self):
        """Test accuracy calculation by evaluation context."""
        # Mock different contexts
        stable_state = BallState(10.0, {'red': 15}, {}, False)
        
        # Mock event for transition context
        transition_event = Mock(spec=ProcessedEvent)
        transition_event.event_type = EventType.BALL_POTTED
        transition_event.data = {'ball_potted': 'red'}
        transition_event.end_time = 10.5
        transition_event.start_time = 10.0
        
        # Simulate different contexts
        test_cases = [
            # Stable period
            (stable_state, [], {'red': 15}, "stable_period"),
            (stable_state, [], {'red': 14}, "stable_period"),  # Error in stable
            # Transition period
            (stable_state, [transition_event], {'red': 14}, "transition_period_potting"),
            (stable_state, [transition_event], {'red': 15}, "transition_period_potting"),
        ]
        
        for i, (state, events, counts, expected_context) in enumerate(test_cases):
            self.mock_state_reconstructor.get_state_at_time.return_value = state
            self.mock_event_processor.get_events_at_time.return_value = events
            
            if events:  # Mock transition logic
                before_state = BallState(9.9, {'red': 15}, {}, False)
                self.mock_state_reconstructor.get_expected_state_for_event.return_value = before_state
            
            self.evaluator.evaluate_moment(i, 10.0 + i, counts)
        
        # Get context-based accuracy
        context_accuracy = self.evaluator.get_context_based_accuracy()
        
        # Should have entries for both contexts
        self.assertIn("stable_period", context_accuracy)
        if "transition_period_potting" in context_accuracy:
            # Both transition evaluations should be correct
            self.assertAlmostEqual(context_accuracy["transition_period_potting"]["accuracy"], 100.0)


class TestDuplicationDetection(unittest.TestCase):
    """Specific tests for duplication detection algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        mock_state_reconstructor = Mock(spec=StateReconstructor)
        self.evaluator = MomentEvaluator(mock_state_reconstructor, distance_threshold=50.0)
    
    def test_find_duplicate_groups_simple(self):
        """Test finding duplicate groups with simple case."""
        detections = [
            {'x': 100, 'y': 100, 'confidence': 0.9},
            {'x': 110, 'y': 105, 'confidence': 0.8},  # Close to first
            {'x': 200, 'y': 200, 'confidence': 0.95}  # Far from others
        ]
        
        groups = self.evaluator._find_duplicate_groups(detections)
        
        # Should find one group with two close detections
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
    
    def test_find_duplicate_groups_multiple(self):
        """Test finding multiple duplicate groups."""
        detections = [
            {'x': 100, 'y': 100, 'confidence': 0.9},
            {'x': 110, 'y': 105, 'confidence': 0.8},  # Group 1
            {'x': 200, 'y': 200, 'confidence': 0.95},
            {'x': 205, 'y': 195, 'confidence': 0.85},  # Group 2
            {'x': 400, 'y': 400, 'confidence': 0.9}   # Isolated
        ]
        
        groups = self.evaluator._find_duplicate_groups(detections)
        
        # Should find two groups
        self.assertEqual(len(groups), 2)
        for group in groups:
            self.assertEqual(len(group), 2)
    
    def test_calculate_pairwise_distances(self):
        """Test pairwise distance calculation."""
        positions = [(0, 0), (3, 4), (6, 8)]
        
        distances = self.evaluator._calculate_pairwise_distances(positions)
        
        # Should have 3 distances for 3 points
        self.assertEqual(len(distances), 3)
        
        # Check specific distances
        self.assertAlmostEqual(distances[0], 5.0)  # (0,0) to (3,4)
        self.assertAlmostEqual(distances[1], 10.0)  # (0,0) to (6,8)
        self.assertAlmostEqual(distances[2], 5.0)  # (3,4) to (6,8)


if __name__ == '__main__':
    unittest.main()