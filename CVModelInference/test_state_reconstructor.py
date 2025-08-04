"""
Tests for StateReconstructor class.
"""

import unittest
from event_processor import EventProcessor
from state_reconstructor import StateReconstructor, BallState


class TestStateReconstructor(unittest.TestCase):
    
    def setUp(self):
        self.processor = EventProcessor(video_fps=30.0)
        self.reconstructor = StateReconstructor(self.processor)
        
        self.events = [
            {"time": 0, "reds": 15, "yellow": 1, "green": 1, "brown": 1, 
             "blue": 1, "pink": 1, "black": 1, "white": 1},
            {"time": 30, "ball_potted": "red"},
            {"time_range": "60-65", "ball_potted": "black"},
            {"frame": 2100, "ball_placed_back": "black"},  # 70s
            {"time_range": "80-85", "balls_occluded": {"reds": 2}},
            {"time_range": "90-95", "ignore_errors": "poor_lighting"}
        ]
        
    def test_initial_state(self):
        """Test getting initial state."""
        self.reconstructor.set_events(self.events)
        state = self.reconstructor.get_state_at_time(0)
        
        self.assertEqual(state.ball_counts['reds'], 15)
        self.assertEqual(state.ball_counts['black'], 1)
        self.assertEqual(len(state.occluded_balls), 0)
        self.assertFalse(state.ignore_errors)
        
    def test_after_potting(self):
        """Test state after ball potted."""
        self.reconstructor.set_events(self.events)
        state = self.reconstructor.get_state_at_time(35)  # After red potted
        
        self.assertEqual(state.ball_counts['reds'], 14)  # One less red
        
    def test_during_range_potting(self):
        """Test state during range potting event."""
        self.reconstructor.set_events(self.events)
        state = self.reconstructor.get_state_at_time(62)  # During black potting
        
        self.assertEqual(state.ball_counts['black'], 0)  # Black being potted
        
    def test_after_placement(self):
        """Test state after ball placed back."""
        self.reconstructor.set_events(self.events)
        state = self.reconstructor.get_state_at_time(75)  # After black placed back
        
        self.assertEqual(state.ball_counts['black'], 1)  # Black back on table
        
    def test_during_occlusion(self):
        """Test state during occlusion."""
        self.reconstructor.set_events(self.events)
        state = self.reconstructor.get_state_at_time(82)  # During occlusion
        
        self.assertEqual(state.occluded_balls['reds'], 2)
        
    def test_during_error_suppression(self):
        """Test state during error suppression."""
        self.reconstructor.set_events(self.events)
        state = self.reconstructor.get_state_at_time(92)  # During error suppression
        
        self.assertTrue(state.ignore_errors)
        
    def test_validate_state(self):
        """Test state validation."""
        valid_state = BallState(0, {'reds': 15, 'black': 1}, {}, False)
        errors = self.reconstructor.validate_state(valid_state)
        self.assertEqual(len(errors), 0)
        
        invalid_state = BallState(0, {'reds': -1, 'black': 2}, {}, False)
        errors = self.reconstructor.validate_state(invalid_state)
        self.assertGreater(len(errors), 0)
    
    def test_before_during_after_logic(self):
        """Test before/during/after state logic for events."""
        self.reconstructor.set_events(self.events)
        
        # Find the range potting event (black ball 60-65s)
        range_event = None
        for event in self.reconstructor.processed_events:
            if (event.event_type.value == 'ball_potted' and 
                event.data.get('ball_potted') == 'black' and 
                event.end_time is not None):
                range_event = event
                break
        
        self.assertIsNotNone(range_event)
        
        # Test before/during/after states
        before_state = self.reconstructor.get_expected_state_for_event(range_event, 'before')
        during_state = self.reconstructor.get_expected_state_for_event(range_event, 'during')
        after_state = self.reconstructor.get_expected_state_for_event(range_event, 'after')
        
        # Before: should have 1 black ball (after red was potted)
        self.assertEqual(before_state.ball_counts['black'], 1)
        # During: should have 0 black balls (being potted)
        self.assertEqual(during_state.ball_counts['black'], 0)
        # After: should have 0 black balls (potted)
        self.assertEqual(after_state.ball_counts['black'], 0)
    
    def test_validate_event_transition(self):
        """Test event transition validation."""
        self.reconstructor.set_events(self.events)
        
        # Find a valid potting event
        potting_event = None
        for event in self.reconstructor.processed_events:
            if event.event_type.value == 'ball_potted':
                potting_event = event
                break
        
        self.assertIsNotNone(potting_event)
        
        # Should be valid
        errors = self.reconstructor.validate_event_transition(potting_event)
        self.assertEqual(len(errors), 0)
    
    def test_occlusion_handling(self):
        """Test occlusion event handling."""
        self.reconstructor.set_events(self.events)
        
        # During occlusion (80-85s), check at 82s
        state = self.reconstructor.get_state_at_time(82)
        self.assertEqual(state.occluded_balls['reds'], 2)
        
        # After occlusion, should be clear
        state = self.reconstructor.get_state_at_time(87)
        self.assertEqual(len(state.occluded_balls), 0)
    
    def test_error_suppression_handling(self):
        """Test error suppression event handling."""
        self.reconstructor.set_events(self.events)
        
        # During error suppression (90-95s), check at 92s
        state = self.reconstructor.get_state_at_time(92)
        self.assertTrue(state.ignore_errors)
        
        # After error suppression, should be false
        state = self.reconstructor.get_state_at_time(97)
        self.assertFalse(state.ignore_errors)


if __name__ == '__main__':
    unittest.main()