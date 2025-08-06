"""
Event Processor Module for Ground Truth Event Handling

This module provides functionality to parse, validate, and process ground truth events
for snooker tracking evaluation. It supports both time-based and frame-based events
with flexible time range parsing.
"""

import re
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Enumeration of supported event types."""
    INITIAL_STATE = "initial_state"
    BALL_POTTED = "ball_potted"
    BALL_PLACED_BACK = "ball_placed_back"
    BALLS_OCCLUDED = "balls_occluded"
    IGNORE_ERRORS = "ignore_errors"


@dataclass
class ProcessedEvent:
    """Processed event with normalized time information."""
    event_type: EventType
    start_time: float
    end_time: Optional[float]  # None for point events
    start_frame: int
    end_frame: Optional[int]  # None for point events
    data: Dict[str, Any]
    original_event: Dict[str, Any]


class EventProcessor:
    """
    Processes ground truth events for snooker tracking evaluation.
    
    Supports flexible event parsing with both single time and time range formats,
    automatic conversion between time-based and frame-based events, and comprehensive
    event validation.
    """
    
    # Valid ball types in snooker
    VALID_BALL_TYPES = {
        'red', 'yellow', 'green', 'brown', 'blue', 'pink', 'black', 'white', 'cue'
    }
    
    # Ball count keys for initial state
    BALL_COUNT_KEYS = {
        'reds', 'red', 'yellow', 'green', 'brown', 'blue', 'pink', 'black', 'white', 'cue'
    }
    
    def __init__(self, video_fps: float = 30.0, detection_interval: int = 1):
        """
        Initialize the event processor.
        
        Args:
            video_fps: Video frame rate for time/frame conversion
            detection_interval: Detection interval in frames
        """
        self.video_fps = video_fps
        self.detection_interval = detection_interval
        self.processed_events: List[ProcessedEvent] = []
        self.validation_errors: List[str] = []
        
    def parse_time_ranges(self, event: Dict[str, Any]) -> Tuple[float, Optional[float]]:
        """
        Parse time range strings and convert to start/end times.
        
        Args:
            event: Event dictionary containing time information
            
        Returns:
            Tuple of (start_time, end_time). end_time is None for point events.
            
        Raises:
            ValueError: If time format is invalid
        """
        # Handle time-based events
        if 'time' in event:
            time_val = event['time']
            if isinstance(time_val, (int, float)):
                return float(time_val), None
            else:
                raise ValueError(f"Invalid time format: {time_val}. Expected number.")
                
        # Handle time range events
        if 'time_range' in event:
            time_range = event['time_range']
            if isinstance(time_range, str):
                # Parse "start-end" format
                match = re.match(r'^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$', time_range.strip())
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    if end_time <= start_time:
                        raise ValueError(f"End time ({end_time}) must be greater than start time ({start_time})")
                    return start_time, end_time
                else:
                    raise ValueError(f"Invalid time_range format: {time_range}. Expected 'start-end' format.")
            else:
                raise ValueError(f"Invalid time_range type: {type(time_range)}. Expected string.")
                
        # Handle frame-based events
        if 'frame' in event:
            frame_val = event['frame']
            if isinstance(frame_val, int) and frame_val >= 0:
                time_val = frame_val / self.video_fps
                return time_val, None
            else:
                raise ValueError(f"Invalid frame format: {frame_val}. Expected non-negative integer.")
                
        # Handle frame range events
        if 'frame_range' in event:
            frame_range = event['frame_range']
            if isinstance(frame_range, str):
                # Parse "start-end" format
                match = re.match(r'^(\d+)-(\d+)$', frame_range.strip())
                if match:
                    start_frame = int(match.group(1))
                    end_frame = int(match.group(2))
                    if end_frame <= start_frame:
                        raise ValueError(f"End frame ({end_frame}) must be greater than start frame ({start_frame})")
                    start_time = start_frame / self.video_fps
                    end_time = end_frame / self.video_fps
                    return start_time, end_time
                else:
                    raise ValueError(f"Invalid frame_range format: {frame_range}. Expected 'start-end' format.")
            else:
                raise ValueError(f"Invalid frame_range type: {type(frame_range)}. Expected string.")
                
        # No time specification found
        raise ValueError("Event must specify time, frame, time_range, or frame_range")
    
    def _time_to_frame(self, time_val: float) -> int:
        """Convert time in seconds to frame number."""
        return int(time_val * self.video_fps)
    
    def _identify_event_type(self, event: Dict[str, Any]) -> EventType:
        """
        Identify the type of event based on its contents.
        
        Args:
            event: Event dictionary
            
        Returns:
            EventType enum value
            
        Raises:
            ValueError: If event type cannot be determined
        """
        # Check for initial state (time=0 with ball counts)
        if event.get('time') == 0 and any(key in event for key in self.BALL_COUNT_KEYS):
            return EventType.INITIAL_STATE
            
        # Check for ball potting
        if 'ball_potted' in event:
            return EventType.BALL_POTTED
            
        # Check for ball placement
        if 'ball_placed_back' in event:
            return EventType.BALL_PLACED_BACK
            
        # Check for occlusion
        if 'balls_occluded' in event:
            return EventType.BALLS_OCCLUDED
            
        # Check for error suppression
        if 'ignore_errors' in event:
            return EventType.IGNORE_ERRORS
            
        raise ValueError(f"Cannot determine event type for event: {event}")
    
    def validate_event(self, event: Dict[str, Any]) -> List[str]:
        """
        Validate a single event and return any error messages.
        
        Args:
            event: Event dictionary to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        try:
            # Validate time specification
            start_time, end_time = self.parse_time_ranges(event)
            
            # Validate event type
            event_type = self._identify_event_type(event)
            
            # Type-specific validation
            if event_type == EventType.INITIAL_STATE:
                # Validate ball counts
                for key, value in event.items():
                    if key in self.BALL_COUNT_KEYS:
                        if not isinstance(value, int) or value < 0:
                            errors.append(f"Invalid ball count for {key}: {value}. Expected non-negative integer.")
                            
            elif event_type == EventType.BALL_POTTED:
                ball_type = event.get('ball_potted')
                if not isinstance(ball_type, str):
                    errors.append(f"Invalid ball_potted type: {type(ball_type)}. Expected string.")
                elif ball_type.lower() not in self.VALID_BALL_TYPES:
                    errors.append(f"Invalid ball type: {ball_type}. Valid types: {self.VALID_BALL_TYPES}")
                    
            elif event_type == EventType.BALL_PLACED_BACK:
                ball_type = event.get('ball_placed_back')
                if not isinstance(ball_type, str):
                    errors.append(f"Invalid ball_placed_back type: {type(ball_type)}. Expected string.")
                elif ball_type.lower() not in self.VALID_BALL_TYPES:
                    errors.append(f"Invalid ball type: {ball_type}. Valid types: {self.VALID_BALL_TYPES}")
                    
            elif event_type == EventType.BALLS_OCCLUDED:
                balls_occluded = event.get('balls_occluded')
                if not isinstance(balls_occluded, dict):
                    errors.append(f"Invalid balls_occluded type: {type(balls_occluded)}. Expected dictionary.")
                else:
                    for ball_type, count in balls_occluded.items():
                        if ball_type not in self.BALL_COUNT_KEYS:
                            errors.append(f"Invalid ball type in occlusion: {ball_type}")
                        if not isinstance(count, int) or count < 0:
                            errors.append(f"Invalid occlusion count for {ball_type}: {count}. Expected non-negative integer.")
                            
            elif event_type == EventType.IGNORE_ERRORS:
                ignore_reason = event.get('ignore_errors')
                if not isinstance(ignore_reason, str):
                    errors.append(f"Invalid ignore_errors type: {type(ignore_reason)}. Expected string.")
                    
            # Validate that potting/placement events can use time ranges
            if event_type in [EventType.BALL_POTTED, EventType.BALL_PLACED_BACK]:
                # These events can be either point events or range events
                pass  # No additional validation needed
                
        except ValueError as e:
            errors.append(str(e))
            
        return errors
    
    def validate_events(self, events: List[Dict[str, Any]]) -> List[str]:
        """
        Validate a list of events and return any error messages.
        
        Args:
            events: List of event dictionaries to validate
            
        Returns:
            List of error messages (empty if all valid)
        """
        all_errors = []
        
        for i, event in enumerate(events):
            event_errors = self.validate_event(event)
            for error in event_errors:
                all_errors.append(f"Event {i}: {error}")
                
        # Check for duplicate initial states
        initial_states = [i for i, event in enumerate(events) 
                         if event.get('time') == 0 and any(key in event for key in self.BALL_COUNT_KEYS)]
        if len(initial_states) > 1:
            all_errors.append(f"Multiple initial state events found at indices: {initial_states}")
            
        return all_errors
    
    def process_events(self, events: List[Dict[str, Any]]) -> List[ProcessedEvent]:
        """
        Process and normalize a list of events.
        
        Args:
            events: List of raw event dictionaries
            
        Returns:
            List of processed events
            
        Raises:
            ValueError: If events contain validation errors
        """
        # Validate events first
        validation_errors = self.validate_events(events)
        if validation_errors:
            self.validation_errors = validation_errors
            raise ValueError(f"Event validation failed: {validation_errors}")
            
        processed_events = []
        
        for event in events:
            try:
                # Parse time information
                start_time, end_time = self.parse_time_ranges(event)
                start_frame = self._time_to_frame(start_time)
                end_frame = self._time_to_frame(end_time) if end_time is not None else None
                
                # Identify event type
                event_type = self._identify_event_type(event)
                
                # Extract event data (everything except time/frame keys)
                data = {k: v for k, v in event.items() 
                       if k not in ['time', 'frame', 'time_range', 'frame_range']}
                
                # Create processed event
                processed_event = ProcessedEvent(
                    event_type=event_type,
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    data=data,
                    original_event=event.copy()
                )
                
                processed_events.append(processed_event)
                
            except ValueError as e:
                raise ValueError(f"Error processing event {event}: {e}")
                
        self.processed_events = processed_events
        return processed_events
    
    def sort_events_chronologically(self, events: List[ProcessedEvent]) -> List[ProcessedEvent]:
        """
        Sort events chronologically by start time.
        
        Args:
            events: List of processed events
            
        Returns:
            List of events sorted by start time
        """
        return sorted(events, key=lambda e: e.start_time)
    
    def get_events_for_timerange(self, start_time: float, end_time: float, 
                                events: Optional[List[ProcessedEvent]] = None) -> List[ProcessedEvent]:
        """
        Get all events that affect a specific time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            events: Optional list of events to filter (uses processed_events if None)
            
        Returns:
            List of events that overlap with the specified time range
        """
        if events is None:
            events = self.processed_events
            
        overlapping_events = []
        
        for event in events:
            # Check if event overlaps with time range
            event_start = event.start_time
            event_end = event.end_time if event.end_time is not None else event.start_time
            
            # Events overlap if: event_start <= end_time AND event_end >= start_time
            if event_start <= end_time and event_end >= start_time:
                overlapping_events.append(event)
                
        return overlapping_events
    
    def get_events_at_time(self, time: float, 
                          events: Optional[List[ProcessedEvent]] = None) -> List[ProcessedEvent]:
        """
        Get all events that are active at a specific time.
        
        Args:
            time: Time in seconds
            events: Optional list of events to filter (uses processed_events if None)
            
        Returns:
            List of events active at the specified time
        """
        if events is None:
            events = self.processed_events
            
        active_events = []
        
        for event in events:
            # Check if event is active at the specified time
            event_start = event.start_time
            event_end = event.end_time if event.end_time is not None else event.start_time
            
            # Event is active if: event_start <= time <= event_end
            if event_start <= time <= event_end:
                active_events.append(event)
                
        return active_events
    
    def filter_events_by_type(self, event_type: EventType, 
                             events: Optional[List[ProcessedEvent]] = None) -> List[ProcessedEvent]:
        """
        Filter events by type.
        
        Args:
            event_type: Type of events to filter
            events: Optional list of events to filter (uses processed_events if None)
            
        Returns:
            List of events of the specified type
        """
        if events is None:
            events = self.processed_events
            
        return [event for event in events if event.event_type == event_type]
    
    def get_initial_state_event(self, events: Optional[List[ProcessedEvent]] = None) -> Optional[ProcessedEvent]:
        """
        Get the initial state event (time=0 with ball counts).
        
        Args:
            events: Optional list of events to search (uses processed_events if None)
            
        Returns:
            Initial state event or None if not found
        """
        if events is None:
            events = self.processed_events
            
        initial_events = self.filter_events_by_type(EventType.INITIAL_STATE, events)
        return initial_events[0] if initial_events else None
    
    def convert_frame_to_time(self, frame: int) -> float:
        """Convert frame number to time in seconds."""
        return frame / self.video_fps
    
    def convert_time_to_frame(self, time: float) -> int:
        """Convert time in seconds to frame number."""
        return self._time_to_frame(time)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of processed events.
        
        Returns:
            Dictionary containing event summary statistics
        """
        if not self.processed_events:
            return {'total_events': 0}
            
        event_type_counts = {}
        for event_type in EventType:
            count = len(self.filter_events_by_type(event_type))
            event_type_counts[event_type.value] = count
            
        time_range = (
            min(e.start_time for e in self.processed_events),
            max(e.end_time or e.start_time for e in self.processed_events)
        )
        
        return {
            'total_events': len(self.processed_events),
            'event_type_counts': event_type_counts,
            'time_range': time_range,
            'has_initial_state': self.get_initial_state_event() is not None,
            'validation_errors': self.validation_errors
        }