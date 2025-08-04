"""
Unit tests for EventProcessor class.

Tests cover event parsing, validation, chronological sorting, and filtering
with mixed time formats including both single time and time range events.
"""

import unittest
from typing import Dict, List, Any
from event_processor import EventProcessor, EventType, ProcessedEvent


class TestEventProcessor(unittest.TestCase):
    """Test cases for EventProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = EventProcessor(video_fps=30.0, detection_interval=1)
        
        # Sample events with mixed time formats
        self.sample_events = [
            # Initial state at time 0
            {"time": 0, "reds": 15, "yellow": 1, "green": 1, "brown": 1, 
             "blue": 1, "pink": 1, "black": 1, "white": 1},
            
            # Point event - ball potted at specific time
            {"time": 42.0, "ball_potted": "red"},
            
            # Range event - ball potted over time range
            {"time_range": "120-125", "ball_potted": "black"},
            
            # Frame-based point event
            {"frame": 1500, "ball_placed_back": "black"},
            
            # Frame-based range event
            {"frame_range": "2000-2150", "balls_occluded": {"reds": 2, "blue": 1}},
            
            # Error suppression range
            {"time_range": "180-185", "ignore_errors": "poor_lighting"}
        ]
    
    def test_parse_time_ranges_point_time(self):
        """Test parsing point time events."""
        event = {"time": 42.5}
        start_time, end_time = self.processor.parse_time_ranges(event)
        
        self.assertEqual(start_time, 42.5)
        self.assertIsNone(end_time)
    
    def test_parse_time_ranges_time_range(self):
        """Test parsing time range events."""
        event = {"time_range": "120.5-125.0"}
        start_time, end_time = self.processor.parse_time_ranges(event)
        
        self.assertEqual(start_time, 120.5)
        self.assertEqual(end_time, 125.0)
    
    def test_parse_time_ranges_point_frame(self):
        """Test parsing point frame events."""
        event = {"frame": 1500}
        start_time, end_time = self.processor.parse_time_ranges(event)
        
        expected_time = 1500 / 30.0  # 50.0 seconds
        self.assertEqual(start_time, expected_time)
        self.assertIsNone(end_time)
    
    def test_parse_time_ranges_frame_range(self):
        """Test parsing frame range events."""
        event = {"frame_range": "1500-1650"}
        start_time, end_time = self.processor.parse_time_ranges(event)
        
        expected_start = 1500 / 30.0  # 50.0 seconds
        expected_end = 1650 / 30.0    # 55.0 seconds
        self.assertEqual(start_time, expected_start)
        self.assertEqual(end_time, expected_end)
    
    def test_parse_time_ranges_invalid_formats(self):
        """Test parsing invalid time formats."""
        invalid_events = [
            {"time": "invalid"},
            {"time_range": "invalid-format"},
            {"time_range": "125-120"},  # End before start
            {"frame": -1},
            {"frame": "invalid"},
            {"frame_range": "invalid"},
            {"frame_range": "1650-1500"},  # End before start
            {}  # No time specification
        ]
        
        for event in invalid_events:
            with self.assertRaises(ValueError):
                self.processor.parse_time_ranges(event)
    
    def test_identify_event_type(self):
        """Test event type identification."""
        test_cases = [
            ({"time": 0, "reds": 15, "yellow": 1}, EventType.INITIAL_STATE),
            ({"time": 42, "ball_potted": "red"}, EventType.BALL_POTTED),
            ({"time": 50, "ball_placed_back": "black"}, EventType.BALL_PLACED_BACK),
            ({"time_range": "120-125", "balls_occluded": {"reds": 2}}, EventType.BALLS_OCCLUDED),
            ({"time_range": "180-185", "ignore_errors": "poor_lighting"}, EventType.IGNORE_ERRORS)
        ]
        
        for event, expected_type in test_cases:
            identified_type = self.processor._identify_event_type(event)
            self.assertEqual(identified_type, expected_type)
    
    def test_identify_event_type_invalid(self):
        """Test event type identification with invalid events."""
        invalid_events = [
            {"time": 42},  # No event data
            {"time": 42, "unknown_key": "value"}  # Unknown event type
        ]
        
        for event in invalid_events:
            with self.assertRaises(ValueError):
                self.processor._identify_event_type(event)
    
    def test_validate_event_valid_events(self):
        """Test validation of valid events."""
        for event in self.sample_events:
            errors = self.processor.validate_event(event)
            self.assertEqual(len(errors), 0, f"Valid event should have no errors: {event}")
    
    def test_validate_event_invalid_ball_types(self):
        """Test validation of invalid ball types."""
        invalid_events = [
            {"time": 42, "ball_potted": "invalid_ball"},
            {"time": 42, "ball_placed_back": "another_invalid"},
            {"time_range": "120-125", "balls_occluded": {"invalid_ball": 2}}
        ]
        
        for event in invalid_events:
            errors = self.processor.validate_event(event)
            self.assertGreater(len(errors), 0, f"Invalid event should have errors: {event}")
    
    def test_validate_event_invalid_counts(self):
        """Test validation of invalid ball counts."""
        invalid_events = [
            {"time": 0, "reds": -1},  # Negative count
            {"time": 0, "yellow": "invalid"},  # Non-integer count
            {"time_range": "120-125", "balls_occluded": {"reds": -2}}  # Negative occlusion
        ]
        
        for event in invalid_events:
            errors = self.processor.validate_event(event)
            self.assertGreater(len(errors), 0, f"Invalid event should have errors: {event}")
    
    def test_validate_events_multiple_initial_states(self):
        """Test validation catches multiple initial states."""
        events_with_duplicates = [
            {"time": 0, "reds": 15, "yellow": 1},
            {"time": 0, "reds": 14, "yellow": 1},  # Duplicate initial state
            {"time": 42, "ball_potted": "red"}
        ]
        
        errors = self.processor.validate_events(events_with_duplicates)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Multiple initial state" in error for error in errors))
    
    def test_process_events_success(self):
        """Test successful event processing."""
        processed_events = self.processor.process_events(self.sample_events)
        
        self.assertEqual(len(processed_events), len(self.sample_events))
        
        # Check first event (initial state)
        initial_event = processed_events[0]
        self.assertEqual(initial_event.event_type, EventType.INITIAL_STATE)
        self.assertEqual(initial_event.start_time, 0.0)
        self.assertIsNone(initial_event.end_time)
        self.assertEqual(initial_event.data['reds'], 15)
        
        # Check range event
        range_events = [e for e in processed_events if e.end_time is not None]
        self.assertGreater(len(range_events), 0)
        
        # Check frame conversion
        frame_events = [e for e in processed_events if 'frame' in e.original_event or 'frame_range' in e.original_event]
        self.assertGreater(len(frame_events), 0)
    
    def test_process_events_validation_failure(self):
        """Test event processing with validation errors."""
        invalid_events = [
            {"time": 0, "reds": 15},
            {"time": 42, "ball_potted": "invalid_ball"}  # Invalid ball type
        ]
        
        with self.assertRaises(ValueError):
            self.processor.process_events(invalid_events)
    
    def test_sort_events_chronologically(self):
        """Test chronological sorting of events."""
        # Create events in non-chronological order
        unsorted_events = [
            ProcessedEvent(EventType.BALL_POTTED, 100.0, None, 3000, None, {}, {}),
            ProcessedEvent(EventType.INITIAL_STATE, 0.0, None, 0, None, {}, {}),
            ProcessedEvent(EventType.BALL_PLACED_BACK, 50.0, None, 1500, None, {}, {})
        ]
        
        sorted_events = self.processor.sort_events_chronologically(unsorted_events)
        
        # Check that events are sorted by start time
        times = [event.start_time for event in sorted_events]
        self.assertEqual(times, [0.0, 50.0, 100.0])
    
    def test_get_events_for_timerange(self):
        """Test filtering events by time range."""
        processed_events = self.processor.process_events(self.sample_events)
        
        # Get events that overlap with 40-45 second range
        overlapping_events = self.processor.get_events_for_timerange(40.0, 45.0, processed_events)
        
        # Should include the ball_potted event at time 42
        self.assertGreater(len(overlapping_events), 0)
        ball_potted_events = [e for e in overlapping_events if e.event_type == EventType.BALL_POTTED]
        self.assertGreater(len(ball_potted_events), 0)
    
    def test_get_events_at_time(self):
        """Test getting events active at specific time."""
        processed_events = self.processor.process_events(self.sample_events)
        
        # Get events active at time 122 (should include the 120-125 range event)
        active_events = self.processor.get_events_at_time(122.0, processed_events)
        
        # Should include the ball_potted range event
        range_events = [e for e in active_events if e.end_time is not None]
        self.assertGreater(len(range_events), 0)
    
    def test_filter_events_by_type(self):
        """Test filtering events by type."""
        processed_events = self.processor.process_events(self.sample_events)
        
        # Filter for ball potted events
        potted_events = self.processor.filter_events_by_type(EventType.BALL_POTTED, processed_events)
        
        # Should have 2 ball potted events (one point, one range)
        self.assertEqual(len(potted_events), 2)
        
        # All should be ball potted events
        for event in potted_events:
            self.assertEqual(event.event_type, EventType.BALL_POTTED)
    
    def test_get_initial_state_event(self):
        """Test getting initial state event."""
        processed_events = self.processor.process_events(self.sample_events)
        
        initial_event = self.processor.get_initial_state_event(processed_events)
        
        self.assertIsNotNone(initial_event)
        self.assertEqual(initial_event.event_type, EventType.INITIAL_STATE)
        self.assertEqual(initial_event.start_time, 0.0)
        self.assertEqual(initial_event.data['reds'], 15)
    
    def test_get_initial_state_event_missing(self):
        """Test getting initial state when none exists."""
        events_without_initial = [
            {"time": 42, "ball_potted": "red"}
        ]
        
        processed_events = self.processor.process_events(events_without_initial)
        initial_event = self.processor.get_initial_state_event(processed_events)
        
        self.assertIsNone(initial_event)
    
    def test_frame_time_conversion(self):
        """Test frame/time conversion methods."""
        # Test time to frame
        frame = self.processor.convert_time_to_frame(50.0)
        self.assertEqual(frame, 1500)  # 50 * 30 fps
        
        # Test frame to time
        time = self.processor.convert_frame_to_time(1500)
        self.assertEqual(time, 50.0)  # 1500 / 30 fps
    
    def test_get_summary(self):
        """Test getting event summary."""
        processed_events = self.processor.process_events(self.sample_events)
        summary = self.processor.get_summary()
        
        self.assertEqual(summary['total_events'], len(self.sample_events))
        self.assertTrue(summary['has_initial_state'])
        self.assertIn('event_type_counts', summary)
        self.assertIn('time_range', summary)
        
        # Check event type counts
        counts = summary['event_type_counts']
        self.assertEqual(counts['initial_state'], 1)
        self.assertEqual(counts['ball_potted'], 2)  # One point, one range
        self.assertEqual(counts['ball_placed_back'], 1)
        self.assertEqual(counts['balls_occluded'], 1)
        self.assertEqual(counts['ignore_errors'], 1)
    
    def test_get_summary_empty(self):
        """Test getting summary with no events."""
        summary = self.processor.get_summary()
        self.assertEqual(summary['total_events'], 0)
    
    def test_mixed_time_formats_integration(self):
        """Test integration with mixed time formats."""
        # Events with all supported time formats
        mixed_events = [
            {"time": 0, "reds": 15, "yellow": 1},           # Point time
            {"time": 30.5, "ball_potted": "red"},           # Point time with decimal
            {"time_range": "60-65", "ball_potted": "black"}, # Time range
            {"frame": 2100, "ball_placed_back": "black"},    # Point frame
            {"frame_range": "2400-2550", "balls_occluded": {"reds": 1}}, # Frame range
            {"time_range": "90.5-95.0", "ignore_errors": "camera_shake"} # Time range with decimals
        ]
        
        # Process events
        processed_events = self.processor.process_events(mixed_events)
        
        # Sort chronologically
        sorted_events = self.processor.sort_events_chronologically(processed_events)
        
        # Verify chronological order
        times = [event.start_time for event in sorted_events]
        self.assertEqual(times, sorted(times))
        
        # Verify all events processed correctly
        self.assertEqual(len(processed_events), len(mixed_events))
        
        # Verify frame conversions
        frame_event = next(e for e in processed_events if e.original_event.get('frame') == 2100)
        expected_time = 2100 / 30.0  # 70.0 seconds
        self.assertEqual(frame_event.start_time, expected_time)
        
        # Verify range events have both start and end times
        range_events = [e for e in processed_events if e.end_time is not None]
        self.assertEqual(len(range_events), 3)  # 2 time ranges + 1 frame range
        
        # Test filtering by time range
        events_in_range = self.processor.get_events_for_timerange(60.0, 70.0, processed_events)
        self.assertGreater(len(events_in_range), 0)
    
    def test_potting_placement_time_range_support(self):
        """Test that potting and placement events support both point and range formats."""
        events_with_ranges = [
            {"time": 0, "reds": 15, "yellow": 1},
            {"time": 30, "ball_potted": "red"},                    # Point potting
            {"time_range": "60-65", "ball_potted": "black"},       # Range potting
            {"frame": 2100, "ball_placed_back": "black"},          # Point placement
            {"frame_range": "2400-2550", "ball_placed_back": "red"} # Range placement
        ]
        
        # Should process without errors
        processed_events = self.processor.process_events(events_with_ranges)
        
        # Verify potting events
        potting_events = self.processor.filter_events_by_type(EventType.BALL_POTTED, processed_events)
        self.assertEqual(len(potting_events), 2)
        
        # One should be point event, one should be range event
        point_potting = [e for e in potting_events if e.end_time is None]
        range_potting = [e for e in potting_events if e.end_time is not None]
        self.assertEqual(len(point_potting), 1)
        self.assertEqual(len(range_potting), 1)
        
        # Verify placement events
        placement_events = self.processor.filter_events_by_type(EventType.BALL_PLACED_BACK, processed_events)
        self.assertEqual(len(placement_events), 2)
        
        # One should be point event, one should be range event
        point_placement = [e for e in placement_events if e.end_time is None]
        range_placement = [e for e in placement_events if e.end_time is not None]
        self.assertEqual(len(point_placement), 1)
        self.assertEqual(len(range_placement), 1)


if __name__ == '__main__':
    unittest.main()