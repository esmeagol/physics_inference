"""
State Reconstructor for Ground Truth Ball State Management

Reconstructs expected ball states at any given time based on ground truth events.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
try:
    from .event_processor import EventProcessor, ProcessedEvent, EventType
except ImportError:
    # Handle case when running as script or module
    try:
        from CVModelInference.event_processor import EventProcessor, ProcessedEvent, EventType
    except ImportError:
        from event_processor import EventProcessor, ProcessedEvent, EventType


@dataclass
class BallState:
    """Expected ball state at a specific time."""
    time: float
    ball_counts: Dict[str, int]
    occluded_balls: Dict[str, int]
    ignore_errors: bool


class StateReconstructor:
    """Reconstructs ball states from ground truth events."""
    
    def __init__(self, event_processor: EventProcessor):
        self.event_processor = event_processor
        self.processed_events: List[ProcessedEvent] = []
        
    def set_events(self, events: List[Dict]) -> None:
        """Set and process ground truth events."""
        self.processed_events = self.event_processor.process_events(events)
        self.processed_events = self.event_processor.sort_events_chronologically(self.processed_events)
        
    def get_state_at_time(self, time: float) -> BallState:
        """Get expected ball state at specific time."""
        return self._get_state_at_time_with_transition_logic(time)
    
    def _get_state_at_time_with_transition_logic(self, time: float) -> BallState:
        """Get expected ball state at specific time."""
        if not self.processed_events:
            raise ValueError("No events set")
            
        # Start with initial state
        initial_event = self.event_processor.get_initial_state_event(self.processed_events)
        if not initial_event:
            raise ValueError("No initial state event found")
            
        ball_counts = initial_event.data.copy()
        occluded_balls = {}
        ignore_errors = False
        
        # Apply events up to the requested time
        for event in self.processed_events:
            if event.start_time > time:
                break
                
            if event.event_type == EventType.BALL_POTTED:
                ball_type = event.data['ball_potted']
                if event.end_time is None:
                    # Point event - apply if at or after start time
                    if event.start_time <= time:
                        self._apply_ball_change(ball_counts, ball_type, -1)
                else:
                    # Range event - apply if at or after start time (effect persists)
                    if event.start_time <= time:
                        self._apply_ball_change(ball_counts, ball_type, -1)
                        
            elif event.event_type == EventType.BALL_PLACED_BACK:
                ball_type = event.data['ball_placed_back']
                if event.end_time is None:
                    # Point event
                    if event.start_time <= time:
                        self._apply_ball_change(ball_counts, ball_type, 1)
                else:
                    # Range event - apply if at or after start time (effect persists)
                    if event.start_time <= time:
                        self._apply_ball_change(ball_counts, ball_type, 1)
                        
            elif event.event_type == EventType.BALLS_OCCLUDED:
                if event.start_time <= time <= (event.end_time or event.start_time):
                    occluded_balls = event.data['balls_occluded'].copy()
                    
            elif event.event_type == EventType.IGNORE_ERRORS:
                if event.start_time <= time <= (event.end_time or event.start_time):
                    ignore_errors = True
                    
        return BallState(time, ball_counts, occluded_balls, ignore_errors)
    
    def get_expected_state_for_event(self, event: ProcessedEvent, phase: str) -> BallState:
        """Get expected state before/during/after an event.
        
        Args:
            event: The event to check
            phase: 'before', 'during', or 'after'
        """
        if phase == 'before':
            # State just before event starts
            return self.get_state_at_time(max(0, event.start_time - 0.001))
        elif phase == 'during':
            # State during event (for range events) or at event time (for point events)
            if event.end_time:
                mid_time = (event.start_time + event.end_time) / 2
            else:
                mid_time = event.start_time
            return self.get_state_at_time(mid_time)
        elif phase == 'after':
            # State just after event ends
            end_time = event.end_time if event.end_time else event.start_time
            return self.get_state_at_time(end_time + 0.001)
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 'before', 'during', or 'after'")
    
    def validate_event_transition(self, event: ProcessedEvent) -> List[str]:
        """Validate that an event transition is possible."""
        errors = []
        
        if event.event_type == EventType.BALL_POTTED:
            ball_type = event.data['ball_potted']
            before_state = self.get_expected_state_for_event(event, 'before')
            during_state = self.get_expected_state_for_event(event, 'during')
            after_state = self.get_expected_state_for_event(event, 'after')
            
            # Get ball count (handle red/reds naming)
            before_count = self._get_ball_count(before_state.ball_counts, ball_type)
            during_count = self._get_ball_count(during_state.ball_counts, ball_type)
            after_count = self._get_ball_count(after_state.ball_counts, ball_type)
            
            # Validation: n balls before, n-1 or n during, n-1 after
            if before_count <= 0:
                errors.append(f"Cannot pot {ball_type}: no balls available before event")
            if event.end_time:  # Range event
                if during_count not in [before_count - 1, before_count]:
                    errors.append(f"Invalid {ball_type} count during potting: expected {before_count-1} or {before_count}, got {during_count}")
            if after_count != before_count - 1:
                errors.append(f"Invalid {ball_type} count after potting: expected {before_count-1}, got {after_count}")
                
        elif event.event_type == EventType.BALL_PLACED_BACK:
            ball_type = event.data['ball_placed_back']
            before_state = self.get_expected_state_for_event(event, 'before')
            after_state = self.get_expected_state_for_event(event, 'after')
            
            before_count = self._get_ball_count(before_state.ball_counts, ball_type)
            after_count = self._get_ball_count(after_state.ball_counts, ball_type)
            
            # Validation: after count should be before count + 1
            if after_count != before_count + 1:
                errors.append(f"Invalid {ball_type} count after placement: expected {before_count+1}, got {after_count}")
                
        return errors
    
    def _get_ball_count(self, ball_counts: Dict[str, int], ball_type: str) -> int:
        """Get ball count handling red/reds naming."""
        if ball_type in ball_counts:
            return ball_counts[ball_type]
        elif ball_type == 'red' and 'reds' in ball_counts:
            return ball_counts['reds']
        elif ball_type == 'reds' and 'red' in ball_counts:
            return ball_counts['red']
        return 0
    
    def _apply_ball_change(self, ball_counts: Dict[str, int], ball_type: str, change: int) -> None:
        """Apply ball count change, handling red/reds naming."""
        if ball_type in ball_counts:
            ball_counts[ball_type] = max(0, ball_counts[ball_type] + change)
        elif ball_type == 'red' and 'reds' in ball_counts:
            ball_counts['reds'] = max(0, ball_counts['reds'] + change)
        elif ball_type == 'reds' and 'red' in ball_counts:
            ball_counts['red'] = max(0, ball_counts['red'] + change)
        
    def validate_state(self, state: BallState) -> List[str]:
        """Validate that ball state is possible."""
        errors = []
        
        for ball_type, count in state.ball_counts.items():
            if count < 0:
                errors.append(f"Negative count for {ball_type}: {count}")
                
            # Check maximum possible counts
            max_counts = {'reds': 15, 'red': 15}
            for color in ['yellow', 'green', 'brown', 'blue', 'pink', 'black', 'white', 'cue']:
                max_counts[color] = 1
                
            if ball_type in max_counts and count > max_counts[ball_type]:
                errors.append(f"Too many {ball_type} balls: {count} > {max_counts[ball_type]}")
                
        return errors