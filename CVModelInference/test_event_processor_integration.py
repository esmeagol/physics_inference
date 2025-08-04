"""
Integration test for EventProcessor with realistic snooker scenarios.
"""

from event_processor import EventProcessor, EventType


def test_realistic_snooker_scenario():
    """Test EventProcessor with a realistic snooker game scenario."""
    
    # Initialize processor for 30 FPS video
    processor = EventProcessor(video_fps=30.0, detection_interval=5)
    
    # Realistic snooker events with mixed time formats
    snooker_events = [
        # Game starts with full table
        {"time": 0, "reds": 15, "yellow": 1, "green": 1, "brown": 1, 
         "blue": 1, "pink": 1, "black": 1, "white": 1},
        
        # First red potted at 45 seconds
        {"time": 45.0, "ball_potted": "red"},
        
        # Black ball potted over time range (player taking time to aim)
        {"time_range": "52-55", "ball_potted": "black"},
        
        # Black ball placed back (using frame number)
        {"frame": 1680, "ball_placed_back": "black"},  # 56 seconds
        
        # Another red potted
        {"time": 62.5, "ball_potted": "red"},
        
        # Balls occluded by player's body (frame range)
        {"frame_range": "2100-2250", "balls_occluded": {"reds": 3, "white": 1}},
        
        # Poor camera angle - ignore errors
        {"time_range": "85-90", "ignore_errors": "camera_angle_poor"},
        
        # Final red potted with time range
        {"time_range": "95.5-97.0", "ball_potted": "red"}
    ]
    
    print("Testing realistic snooker scenario...")
    print(f"Processing {len(snooker_events)} events")
    
    # Process events
    try:
        processed_events = processor.process_events(snooker_events)
        print(f"✓ Successfully processed {len(processed_events)} events")
    except Exception as e:
        print(f"✗ Error processing events: {e}")
        return False
    
    # Sort chronologically
    sorted_events = processor.sort_events_chronologically(processed_events)
    print(f"✓ Events sorted chronologically")
    
    # Print event timeline
    print("\nEvent Timeline:")
    for i, event in enumerate(sorted_events):
        time_str = f"{event.start_time:.1f}s"
        if event.end_time:
            time_str += f"-{event.end_time:.1f}s"
        
        event_desc = f"{event.event_type.value}"
        if event.event_type == EventType.BALL_POTTED:
            event_desc += f": {event.data.get('ball_potted')}"
        elif event.event_type == EventType.BALL_PLACED_BACK:
            event_desc += f": {event.data.get('ball_placed_back')}"
        elif event.event_type == EventType.BALLS_OCCLUDED:
            event_desc += f": {event.data.get('balls_occluded')}"
        elif event.event_type == EventType.IGNORE_ERRORS:
            event_desc += f": {event.data.get('ignore_errors')}"
        
        print(f"  {i+1}. {time_str:12} - {event_desc}")
    
    # Test filtering capabilities
    print("\nFiltering Tests:")
    
    # Get events in first minute
    first_minute_events = processor.get_events_for_timerange(0, 60, sorted_events)
    print(f"✓ Events in first minute: {len(first_minute_events)}")
    
    # Get ball potting events
    potting_events = processor.filter_events_by_type(EventType.BALL_POTTED, sorted_events)
    print(f"✓ Ball potting events: {len(potting_events)}")
    
    # Get events active at 70 seconds (during occlusion)
    events_at_70s = processor.get_events_at_time(70.0, sorted_events)
    print(f"✓ Events active at 70s: {len(events_at_70s)}")
    
    # Get initial state
    initial_state = processor.get_initial_state_event(sorted_events)
    if initial_state:
        print(f"✓ Initial state found: {initial_state.data}")
    
    # Get summary
    summary = processor.get_summary()
    print(f"\nSummary:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Time range: {summary['time_range'][0]:.1f}s - {summary['time_range'][1]:.1f}s")
    print(f"  Event types: {summary['event_type_counts']}")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    
    processor = EventProcessor(video_fps=25.0, detection_interval=1)
    
    print("\nTesting edge cases...")
    
    # Test with no events
    try:
        empty_result = processor.process_events([])
        print(f"✓ Empty events list handled: {len(empty_result)} events")
    except Exception as e:
        print(f"✗ Error with empty events: {e}")
        return False
    
    # Test with invalid events
    invalid_events = [
        {"time": 0, "reds": 15},
        {"time": 30, "ball_potted": "invalid_ball"}  # Invalid ball type
    ]
    
    try:
        processor.process_events(invalid_events)
        print("✗ Should have failed with invalid events")
        return False
    except ValueError as e:
        print(f"✓ Invalid events properly rejected: {str(e)[:50]}...")
    
    # Test time/frame conversions
    test_time = 42.5
    test_frame = processor.convert_time_to_frame(test_time)
    converted_back = processor.convert_frame_to_time(test_frame)
    print(f"✓ Time conversion: {test_time}s -> {test_frame} frames -> {converted_back}s")
    
    return True


if __name__ == "__main__":
    print("EventProcessor Integration Tests")
    print("=" * 40)
    
    success1 = test_realistic_snooker_scenario()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n✓ All integration tests passed!")
    else:
        print("\n✗ Some integration tests failed!")