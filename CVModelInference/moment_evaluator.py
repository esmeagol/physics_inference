"""
Moment Evaluator for Tracker Output Evaluation

This module provides sophisticated evaluation of tracker output against ground truth
states, including count comparison, illegal change detection, and spatial analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
try:
    from .event_processor import EventProcessor, ProcessedEvent, EventType
    from .state_reconstructor import StateReconstructor, BallState
except ImportError:
    # Handle case when running as script or module
    try:
        from CVModelInference.event_processor import EventProcessor, ProcessedEvent, EventType
        from CVModelInference.state_reconstructor import StateReconstructor, BallState
    except ImportError:
        from event_processor import EventProcessor, ProcessedEvent, EventType
        from state_reconstructor import StateReconstructor, BallState


class ErrorType(Enum):
    """Types of tracking errors."""
    CORRECT = "correct"
    OVER_DETECTION = "over_detection"
    UNDER_DETECTION = "under_detection"
    ILLEGAL_DISAPPEARANCE = "illegal_disappearance"
    ILLEGAL_REAPPEARANCE = "illegal_reappearance"
    DUPLICATION = "duplication"
    MISSING_OBJECT = "missing_object"


@dataclass
class CountError:
    """Count comparison error details."""
    ball_type: str
    expected: int
    detected: int
    error_type: ErrorType
    error_magnitude: int
    context: str  # 'stable_period', 'transition_period', 'during_event'
    event_context: Optional[Dict[str, Any]] = None


@dataclass
class IllegalChange:
    """Illegal change detection result."""
    ball_type: str
    change_type: ErrorType  # ILLEGAL_DISAPPEARANCE or ILLEGAL_REAPPEARANCE
    previous_count: int
    current_count: int
    expected_events: List[str]  # Events that could explain this change
    context: str


@dataclass
class DuplicationError:
    """Duplication detection result."""
    ball_type: str
    positions: List[Tuple[float, float]]
    distances: List[float]
    confidence_scores: List[float]


@dataclass
class MomentEvaluation:
    """Complete evaluation result for a single moment."""
    moment_idx: int
    timestamp: float
    expected_state: BallState
    detected_counts: Dict[str, int]
    detected_positions: List[Dict[str, Any]]
    count_errors: List[CountError]
    illegal_changes: List[IllegalChange]
    duplication_errors: List[DuplicationError]
    suppressed: bool
    active_events: List[ProcessedEvent]


class MomentEvaluator:
    """
    Evaluates tracker output against expected ground truth states.
    
    Provides sophisticated count comparison with time range awareness,
    illegal change detection, and spatial analysis for duplication detection.
    """
    
    def __init__(self, state_reconstructor: StateReconstructor, 
                 distance_threshold: float = 50.0):
        """
        Initialize the moment evaluator.
        
        Args:
            state_reconstructor: StateReconstructor instance for ground truth
            distance_threshold: Distance threshold for duplication detection (pixels)
        """
        self.state_reconstructor = state_reconstructor
        self.distance_threshold = distance_threshold
        self.previous_detected_counts: Optional[Dict[str, int]] = None
        self.evaluation_history: List[MomentEvaluation] = []
        
    def evaluate_moment(self, moment_idx: int, timestamp: float,
                       detected_counts: Dict[str, int],
                       detected_positions: Optional[List[Dict[str, Any]]] = None) -> MomentEvaluation:
        """
        Evaluate tracker output for a single moment.
        
        Args:
            moment_idx: Index of the moment being evaluated
            timestamp: Time in seconds for this moment
            detected_counts: Dictionary of detected ball counts by type
            detected_positions: Optional list of detection positions with metadata
            
        Returns:
            Complete evaluation result for the moment
        """
        # Get expected state at this time
        expected_state = self.state_reconstructor.get_state_at_time(timestamp)
        
        # Get active events at this time
        active_events = self.state_reconstructor.event_processor.get_events_at_time(
            timestamp, self.state_reconstructor.processed_events
        )
        
        # Initialize evaluation result
        evaluation = MomentEvaluation(
            moment_idx=moment_idx,
            timestamp=timestamp,
            expected_state=expected_state,
            detected_counts=detected_counts,
            detected_positions=detected_positions or [],
            count_errors=[],
            illegal_changes=[],
            duplication_errors=[],
            suppressed=expected_state.ignore_errors,
            active_events=active_events
        )
        
        # Skip evaluation if errors are suppressed
        if expected_state.ignore_errors:
            self.previous_detected_counts = detected_counts
            self.evaluation_history.append(evaluation)
            return evaluation
        
        # Perform count comparison evaluation
        evaluation.count_errors = self._evaluate_counts(
            expected_state, detected_counts, active_events, timestamp
        )
        
        # Perform illegal change detection
        if self.previous_detected_counts is not None:
            evaluation.illegal_changes = self._detect_illegal_changes(
                self.previous_detected_counts, detected_counts, 
                expected_state, active_events, timestamp
            )
        
        # Perform duplication detection
        if detected_positions:
            evaluation.duplication_errors = self._detect_duplications(detected_positions)
        
        # Update history
        self.previous_detected_counts = detected_counts
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _evaluate_counts(self, expected_state: BallState, 
                        detected_counts: Dict[str, int],
                        active_events: List[ProcessedEvent],
                        timestamp: float) -> List[CountError]:
        """
        Evaluate detected counts against expected counts with time range awareness.
        
        Args:
            expected_state: Expected ball state
            detected_counts: Detected ball counts
            active_events: Events active at this time
            timestamp: Current timestamp
            
        Returns:
            List of count errors
        """
        count_errors = []
        
        # Get all ball types to evaluate
        all_ball_types = set(expected_state.ball_counts.keys()) | set(detected_counts.keys())
        
        for ball_type in all_ball_types:
            expected_count = self._get_expected_count_with_occlusion(
                expected_state, ball_type
            )
            detected_count = detected_counts.get(ball_type, 0)
            
            # Determine context and allowed flexibility
            context, allowed_counts = self._get_evaluation_context(
                ball_type, expected_state, active_events, timestamp
            )
            
            # Check if detected count is within allowed range
            if detected_count in allowed_counts:
                # Correct detection
                count_errors.append(CountError(
                    ball_type=ball_type,
                    expected=expected_count,
                    detected=detected_count,
                    error_type=ErrorType.CORRECT,
                    error_magnitude=0,
                    context=context,
                    event_context=self._get_event_context(active_events, ball_type)
                ))
            else:
                # Error detected
                if detected_count > max(allowed_counts):
                    error_type = ErrorType.OVER_DETECTION
                    error_magnitude = detected_count - expected_count
                else:
                    error_type = ErrorType.UNDER_DETECTION
                    error_magnitude = expected_count - detected_count
                
                count_errors.append(CountError(
                    ball_type=ball_type,
                    expected=expected_count,
                    detected=detected_count,
                    error_type=error_type,
                    error_magnitude=abs(error_magnitude),
                    context=context,
                    event_context=self._get_event_context(active_events, ball_type)
                ))
        
        return count_errors
    
    def _get_expected_count_with_occlusion(self, expected_state: BallState, 
                                          ball_type: str) -> int:
        """
        Get expected count accounting for occlusion.
        
        Args:
            expected_state: Expected ball state
            ball_type: Type of ball
            
        Returns:
            Expected visible count (total - occluded)
        """
        total_count = self._get_ball_count(expected_state.ball_counts, ball_type)
        occluded_count = self._get_ball_count(expected_state.occluded_balls, ball_type)
        return max(0, total_count - occluded_count)
    
    def _get_evaluation_context(self, ball_type: str, expected_state: BallState,
                               active_events: List[ProcessedEvent], 
                               timestamp: float) -> Tuple[str, List[int]]:
        """
        Determine evaluation context and allowed count flexibility.
        
        Args:
            ball_type: Type of ball being evaluated
            expected_state: Expected ball state
            active_events: Events active at this time
            timestamp: Current timestamp
            
        Returns:
            Tuple of (context_description, allowed_counts_list)
        """
        expected_count = self._get_expected_count_with_occlusion(expected_state, ball_type)
        
        # Check if we're in a transition period for this ball type
        transition_events = [
            event for event in active_events
            if self._event_affects_ball_type(event, ball_type) and event.end_time is not None
        ]
        
        if transition_events:
            # We're during a transition event
            for event in transition_events:
                if event.event_type == EventType.BALL_POTTED:
                    # During potting: allow n-1 or n balls
                    before_state = self.state_reconstructor.get_expected_state_for_event(event, 'before')
                    before_count = self._get_expected_count_with_occlusion(before_state, ball_type)
                    allowed_counts = [before_count - 1, before_count]
                    return "transition_period_potting", allowed_counts
                    
                elif event.event_type == EventType.BALL_PLACED_BACK:
                    # During placement: allow n or n+1 balls
                    before_state = self.state_reconstructor.get_expected_state_for_event(event, 'before')
                    before_count = self._get_expected_count_with_occlusion(before_state, ball_type)
                    allowed_counts = [before_count, before_count + 1]
                    return "transition_period_placement", allowed_counts
        
        # Check if we're immediately after a point event (within tolerance)
        recent_events = [
            event for event in active_events
            if (self._event_affects_ball_type(event, ball_type) and 
                event.end_time is None and 
                abs(event.start_time - timestamp) < 0.1)  # 100ms tolerance
        ]
        
        if recent_events:
            return "during_event", [expected_count]
        
        # Stable period - only expected count allowed
        return "stable_period", [expected_count]
    
    def _event_affects_ball_type(self, event: ProcessedEvent, ball_type: str) -> bool:
        """Check if an event affects a specific ball type."""
        if event.event_type == EventType.BALL_POTTED:
            return self._normalize_ball_type(event.data['ball_potted']) == self._normalize_ball_type(ball_type)
        elif event.event_type == EventType.BALL_PLACED_BACK:
            return self._normalize_ball_type(event.data['ball_placed_back']) == self._normalize_ball_type(ball_type)
        elif event.event_type == EventType.BALLS_OCCLUDED:
            return ball_type in event.data['balls_occluded']
        return False
    
    def _normalize_ball_type(self, ball_type: str) -> str:
        """Normalize ball type (handle red/reds)."""
        if ball_type.lower() in ['red', 'reds']:
            return 'red'
        return ball_type.lower()
    
    def _get_ball_count(self, ball_counts: Dict[str, int], ball_type: str) -> int:
        """Get ball count handling red/reds naming."""
        if ball_type in ball_counts:
            return ball_counts[ball_type]
        elif ball_type == 'red' and 'reds' in ball_counts:
            return ball_counts['reds']
        elif ball_type == 'reds' and 'red' in ball_counts:
            return ball_counts['red']
        return 0
    
    def _get_event_context(self, active_events: List[ProcessedEvent], 
                          ball_type: str) -> Optional[Dict[str, Any]]:
        """Get event context for a specific ball type."""
        relevant_events = [
            event for event in active_events
            if self._event_affects_ball_type(event, ball_type)
        ]
        
        if not relevant_events:
            return None
        
        return {
            'active_events': [
                {
                    'type': event.event_type.value,
                    'ball_type': ball_type,
                    'time_range': f"{event.start_time}-{event.end_time}" if event.end_time else str(event.start_time),
                    'phase': self._determine_event_phase(event, self.evaluation_history[-1].timestamp if self.evaluation_history else 0)
                }
                for event in relevant_events
            ]
        }
    
    def _determine_event_phase(self, event: ProcessedEvent, timestamp: float) -> str:
        """Determine which phase of an event we're in."""
        if event.end_time is None:
            return "at_event"
        
        duration = event.end_time - event.start_time
        progress = (timestamp - event.start_time) / duration
        
        if progress < 0.33:
            return "early"
        elif progress < 0.67:
            return "middle"
        else:
            return "late"
    
    def calculate_per_ball_accuracy(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-ball-type accuracy metrics.
        
        Returns:
            Dictionary with accuracy metrics for each ball type
        """
        if not self.evaluation_history:
            return {}
        
        ball_type_stats = {}
        
        # Get all ball types from evaluation history
        all_ball_types = set()
        for evaluation in self.evaluation_history:
            all_ball_types.update(evaluation.expected_state.ball_counts.keys())
            all_ball_types.update(evaluation.detected_counts.keys())
        
        for ball_type in all_ball_types:
            stats = {
                'total_moments': 0,
                'correct_moments': 0,
                'over_detections': 0,
                'under_detections': 0,
                'total_error_magnitude': 0,
                'accuracy': 0.0,
                'avg_error_magnitude': 0.0
            }
            
            for evaluation in self.evaluation_history:
                if evaluation.suppressed:
                    continue
                
                stats['total_moments'] += 1
                
                # Find errors for this ball type
                ball_errors = [
                    error for error in evaluation.count_errors
                    if self._normalize_ball_type(error.ball_type) == self._normalize_ball_type(ball_type)
                ]
                
                if not ball_errors or all(error.error_type == ErrorType.CORRECT for error in ball_errors):
                    stats['correct_moments'] += 1
                else:
                    for error in ball_errors:
                        if error.error_type == ErrorType.OVER_DETECTION:
                            stats['over_detections'] += 1
                        elif error.error_type == ErrorType.UNDER_DETECTION:
                            stats['under_detections'] += 1
                        stats['total_error_magnitude'] += error.error_magnitude
            
            # Calculate derived metrics
            if stats['total_moments'] > 0:
                stats['accuracy'] = (stats['correct_moments'] / stats['total_moments']) * 100
                
                total_errors = stats['over_detections'] + stats['under_detections']
                if total_errors > 0:
                    stats['avg_error_magnitude'] = stats['total_error_magnitude'] / total_errors
            
            ball_type_stats[ball_type] = stats
        
        return ball_type_stats
    
    def get_error_magnitude_distribution(self) -> Dict[str, List[int]]:
        """
        Get distribution of error magnitudes by ball type.
        
        Returns:
            Dictionary mapping ball types to lists of error magnitudes
        """
        magnitude_distribution = {}
        
        for evaluation in self.evaluation_history:
            if evaluation.suppressed:
                continue
                
            for error in evaluation.count_errors:
                if error.error_type in [ErrorType.OVER_DETECTION, ErrorType.UNDER_DETECTION]:
                    ball_type = self._normalize_ball_type(error.ball_type)
                    if ball_type not in magnitude_distribution:
                        magnitude_distribution[ball_type] = []
                    magnitude_distribution[ball_type].append(error.error_magnitude)
        
        return magnitude_distribution
    
    def get_context_based_accuracy(self) -> Dict[str, Dict[str, float]]:
        """
        Get accuracy metrics broken down by evaluation context.
        
        Returns:
            Dictionary with accuracy by context type
        """
        context_stats = {}
        
        for evaluation in self.evaluation_history:
            if evaluation.suppressed:
                continue
                
            for error in evaluation.count_errors:
                context = error.context
                if context not in context_stats:
                    context_stats[context] = {'total': 0, 'correct': 0}
                
                context_stats[context]['total'] += 1
                if error.error_type == ErrorType.CORRECT:
                    context_stats[context]['correct'] += 1
        
        # Calculate accuracy percentages
        context_accuracy = {}
        for context, stats in context_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                context_accuracy[context] = {
                    'accuracy': accuracy,
                    'total_evaluations': stats['total'],
                    'correct_evaluations': stats['correct']
                }
        
        return context_accuracy
    
    def _detect_illegal_changes(self, previous_counts: Dict[str, int],
                               current_counts: Dict[str, int],
                               expected_state: BallState,
                               active_events: List[ProcessedEvent],
                               timestamp: float) -> List[IllegalChange]:
        """
        Detect illegal ball disappearances and reappearances.
        
        Args:
            previous_counts: Ball counts from previous moment
            current_counts: Ball counts from current moment
            expected_state: Expected state at current time
            active_events: Events active at this time
            timestamp: Current timestamp
            
        Returns:
            List of illegal changes detected
        """
        illegal_changes = []
        
        # Get all ball types to check
        all_ball_types = set(previous_counts.keys()) | set(current_counts.keys())
        
        for ball_type in all_ball_types:
            prev_count = previous_counts.get(ball_type, 0)
            curr_count = current_counts.get(ball_type, 0)
            
            # Skip if no change
            if prev_count == curr_count:
                continue
            
            # Get expected events that could explain this change
            expected_events = self._get_expected_events_for_change(
                ball_type, prev_count, curr_count, active_events, timestamp
            )
            
            # Determine context
            context = self._get_change_context(active_events, ball_type)
            
            # Check for illegal disappearance
            if curr_count < prev_count:
                # Ball(s) disappeared - this is illegal if no events explain it
                if len(expected_events) == 0:  # No events to explain the disappearance
                    illegal_changes.append(IllegalChange(
                        ball_type=ball_type,
                        change_type=ErrorType.ILLEGAL_DISAPPEARANCE,
                        previous_count=prev_count,
                        current_count=curr_count,
                        expected_events=expected_events,
                        context=context
                    ))
            
            # Check for illegal reappearance
            elif curr_count > prev_count:
                # Ball(s) appeared - this is illegal if no events explain it
                if len(expected_events) == 0:  # No events to explain the reappearance
                    illegal_changes.append(IllegalChange(
                        ball_type=ball_type,
                        change_type=ErrorType.ILLEGAL_REAPPEARANCE,
                        previous_count=prev_count,
                        current_count=curr_count,
                        expected_events=expected_events,
                        context=context
                    ))
        
        return illegal_changes
    
    def _get_expected_events_for_change(self, ball_type: str, prev_count: int, 
                                       curr_count: int, active_events: List[ProcessedEvent],
                                       timestamp: float) -> List[str]:
        """
        Get list of events that could explain a ball count change.
        
        Args:
            ball_type: Type of ball that changed
            prev_count: Previous count
            curr_count: Current count
            active_events: Events active at this time
            timestamp: Current timestamp
            
        Returns:
            List of event descriptions that could explain the change
        """
        expected_events = []
        
        # Check if there are actual events that could explain this change
        for event in active_events:
            if event.event_type == EventType.BALL_POTTED:
                if (self._normalize_ball_type(event.data['ball_potted']) == self._normalize_ball_type(ball_type) 
                    and curr_count < prev_count):
                    expected_events.append(f"ball_potted:{ball_type}")
            elif event.event_type == EventType.BALL_PLACED_BACK:
                if (self._normalize_ball_type(event.data['ball_placed_back']) == self._normalize_ball_type(ball_type) 
                    and curr_count > prev_count):
                    expected_events.append(f"ball_placed_back:{ball_type}")
            elif event.event_type == EventType.BALLS_OCCLUDED:
                occluded_balls = event.data.get('balls_occluded', {})
                if ball_type in occluded_balls:
                    if curr_count < prev_count:
                        expected_events.append(f"balls_occluded:{ball_type} (start)")
                    else:
                        expected_events.append(f"balls_occluded:{ball_type} (end)")
        
        return expected_events
    

    
    def _get_change_context(self, active_events: List[ProcessedEvent], 
                           ball_type: str) -> str:
        """
        Get context description for a ball count change.
        
        Args:
            active_events: Events active at this time
            ball_type: Type of ball that changed
            
        Returns:
            Context description string
        """
        relevant_events = [
            event for event in active_events
            if self._event_affects_ball_type(event, ball_type)
        ]
        
        if relevant_events:
            return "during_event"
        elif active_events:
            return "during_other_events"
        else:
            return "stable_period"
    
    def analyze_tracking_continuity(self) -> Dict[str, Any]:
        """
        Analyze tracking continuity across all moments.
        
        Returns:
            Dictionary with continuity analysis results
        """
        if len(self.evaluation_history) < 2:
            return {'error': 'Insufficient data for continuity analysis'}
        
        continuity_stats = {
            'total_moments': len(self.evaluation_history),
            'moments_with_illegal_changes': 0,
            'total_illegal_disappearances': 0,
            'total_illegal_reappearances': 0,
            'per_ball_illegal_changes': {},
            'continuity_breaks': []
        }
        
        for evaluation in self.evaluation_history:
            if evaluation.suppressed:
                continue
            
            if evaluation.illegal_changes:
                continuity_stats['moments_with_illegal_changes'] += 1
                
                for change in evaluation.illegal_changes:
                    ball_type = self._normalize_ball_type(change.ball_type)
                    
                    if ball_type not in continuity_stats['per_ball_illegal_changes']:
                        continuity_stats['per_ball_illegal_changes'][ball_type] = {
                            'disappearances': 0,
                            'reappearances': 0
                        }
                    
                    if change.change_type == ErrorType.ILLEGAL_DISAPPEARANCE:
                        continuity_stats['total_illegal_disappearances'] += 1
                        continuity_stats['per_ball_illegal_changes'][ball_type]['disappearances'] += 1
                    elif change.change_type == ErrorType.ILLEGAL_REAPPEARANCE:
                        continuity_stats['total_illegal_reappearances'] += 1
                        continuity_stats['per_ball_illegal_changes'][ball_type]['reappearances'] += 1
                    
                    # Record continuity break
                    continuity_stats['continuity_breaks'].append({
                        'moment_idx': evaluation.moment_idx,
                        'timestamp': evaluation.timestamp,
                        'ball_type': ball_type,
                        'change_type': change.change_type.value,
                        'previous_count': change.previous_count,
                        'current_count': change.current_count,
                        'context': change.context
                    })
        
        # Calculate continuity percentage
        total_non_suppressed = sum(1 for e in self.evaluation_history if not e.suppressed)
        if total_non_suppressed > 0:
            continuity_percentage = (
                (total_non_suppressed - continuity_stats['moments_with_illegal_changes']) 
                / total_non_suppressed * 100
            )
            continuity_stats['continuity_percentage'] = continuity_percentage
        
        return continuity_stats
    
    def distinguish_game_events_from_tracking_failures(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Distinguish between expected game events and tracking failures.
        
        Returns:
            Dictionary categorizing changes as game events or tracking failures
        """
        categorized_changes = {
            'expected_game_events': [],
            'tracking_failures': [],
            'ambiguous_changes': []
        }
        
        for evaluation in self.evaluation_history:
            if evaluation.suppressed:
                continue
            
            for change in evaluation.illegal_changes:
                change_info = {
                    'moment_idx': evaluation.moment_idx,
                    'timestamp': evaluation.timestamp,
                    'ball_type': change.ball_type,
                    'change_type': change.change_type.value,
                    'previous_count': change.previous_count,
                    'current_count': change.current_count,
                    'context': change.context,
                    'expected_events': change.expected_events
                }
                
                # Categorize based on context and expected events
                if change.context == "during_event" and change.expected_events:
                    # Likely a game event that wasn't properly annotated
                    categorized_changes['expected_game_events'].append(change_info)
                elif change.context == "stable_period" and not change.expected_events:
                    # Likely a tracking failure
                    categorized_changes['tracking_failures'].append(change_info)
                else:
                    # Ambiguous case
                    categorized_changes['ambiguous_changes'].append(change_info)
        
        return categorized_changes
    
    def _detect_duplications(self, detected_positions: List[Dict[str, Any]]) -> List[DuplicationError]:
        """
        Detect duplicate objects using spatial analysis.
        
        Args:
            detected_positions: List of detection positions with metadata
                Expected format: [{'ball_type': str, 'x': float, 'y': float, 'confidence': float}, ...]
            
        Returns:
            List of duplication errors detected
        """
        duplication_errors = []
        
        # Group detections by ball type
        detections_by_type = {}
        for detection in detected_positions:
            ball_type = self._normalize_ball_type(detection.get('ball_type', ''))
            if ball_type not in detections_by_type:
                detections_by_type[ball_type] = []
            detections_by_type[ball_type].append(detection)
        
        # Check for duplications within each ball type
        for ball_type, detections in detections_by_type.items():
            if len(detections) <= 1:
                continue
            
            # Find pairs of detections that are too close together
            duplicate_groups = self._find_duplicate_groups(detections)
            
            for group in duplicate_groups:
                if len(group) > 1:
                    positions = [(d.get('x', 0), d.get('y', 0)) for d in group]
                    confidences = [d.get('confidence', 0.0) for d in group]
                    distances = self._calculate_pairwise_distances(positions)
                    
                    duplication_errors.append(DuplicationError(
                        ball_type=ball_type,
                        positions=positions,
                        distances=distances,
                        confidence_scores=confidences
                    ))
        
        return duplication_errors
    
    def _find_duplicate_groups(self, detections: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Find groups of detections that are duplicates based on distance.
        
        Args:
            detections: List of detections for a single ball type
            
        Returns:
            List of groups, where each group contains duplicate detections
        """
        if len(detections) <= 1:
            return []
        
        # Create distance matrix
        n = len(detections)
        distance_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = detections[i].get('x', 0), detections[i].get('y', 0)
                x2, y2 = detections[j].get('x', 0), detections[j].get('y', 0)
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        # Find connected components (groups of close detections)
        visited = [False] * n
        duplicate_groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start a new group
            group = []
            stack = [i]
            
            while stack:
                current = stack.pop()
                if visited[current]:
                    continue
                
                visited[current] = True
                group.append(detections[current])
                
                # Add neighbors within threshold
                for j in range(n):
                    if not visited[j] and distance_matrix[current][j] <= self.distance_threshold:
                        stack.append(j)
            
            # Only consider groups with multiple detections as duplicates
            if len(group) > 1:
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def _calculate_pairwise_distances(self, positions: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate all pairwise distances between positions.
        
        Args:
            positions: List of (x, y) position tuples
            
        Returns:
            List of distances between all pairs
        """
        distances = []
        n = len(positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances.append(distance)
        
        return distances
    
    def analyze_spatial_distribution(self, detected_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze spatial distribution of detected objects.
        
        Args:
            detected_positions: List of detection positions with metadata
            
        Returns:
            Dictionary with spatial distribution analysis
        """
        if not detected_positions:
            return {'error': 'No position data available'}
        
        # Group by ball type
        positions_by_type = {}
        for detection in detected_positions:
            ball_type = self._normalize_ball_type(detection.get('ball_type', ''))
            if ball_type not in positions_by_type:
                positions_by_type[ball_type] = []
            positions_by_type[ball_type].append((
                detection.get('x', 0), 
                detection.get('y', 0)
            ))
        
        analysis = {}
        
        for ball_type, positions in positions_by_type.items():
            if not positions:
                continue
            
            # Calculate basic statistics
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            analysis[ball_type] = {
                'count': len(positions),
                'x_range': (min(x_coords), max(x_coords)) if x_coords else (0, 0),
                'y_range': (min(y_coords), max(y_coords)) if y_coords else (0, 0),
                'x_mean': sum(x_coords) / len(x_coords) if x_coords else 0,
                'y_mean': sum(y_coords) / len(y_coords) if y_coords else 0,
                'positions': positions
            }
            
            # Calculate clustering metrics
            if len(positions) > 1:
                # Calculate average distance between all pairs
                total_distance = 0
                pair_count = 0
                
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        total_distance += distance
                        pair_count += 1
                
                analysis[ball_type]['avg_pairwise_distance'] = (
                    total_distance / pair_count if pair_count > 0 else 0
                )
                
                # Find minimum distance (potential duplicates)
                min_distance = float('inf')
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        min_distance = min(min_distance, distance)
                
                analysis[ball_type]['min_distance'] = min_distance
                analysis[ball_type]['potential_duplicates'] = min_distance < self.distance_threshold
        
        return analysis
    
    def detect_missing_objects_in_regions(self, detected_positions: List[Dict[str, Any]],
                                         expected_regions: Optional[Dict[str, List[Tuple[float, float]]]] = None) -> List[Dict[str, Any]]:
        """
        Detect missing objects in expected regions.
        
        Args:
            detected_positions: List of detection positions with metadata
            expected_regions: Optional dictionary mapping ball types to expected regions
                Format: {'red': [(x1, y1), (x2, y2), ...], ...}
            
        Returns:
            List of missing object reports
        """
        if expected_regions is None:
            # Without expected regions, we can't detect missing objects
            return []
        
        missing_objects = []
        
        # Group detections by ball type
        detections_by_type = {}
        for detection in detected_positions:
            ball_type = self._normalize_ball_type(detection.get('ball_type', ''))
            if ball_type not in detections_by_type:
                detections_by_type[ball_type] = []
            detections_by_type[ball_type].append((
                detection.get('x', 0), 
                detection.get('y', 0)
            ))
        
        # Check each expected region
        for ball_type, expected_positions in expected_regions.items():
            detected_positions_for_type = detections_by_type.get(ball_type, [])
            
            for expected_pos in expected_positions:
                # Check if there's a detection near this expected position
                found_nearby = False
                min_distance = float('inf')
                
                for detected_pos in detected_positions_for_type:
                    distance = math.sqrt(
                        (detected_pos[0] - expected_pos[0]) ** 2 + 
                        (detected_pos[1] - expected_pos[1]) ** 2
                    )
                    min_distance = min(min_distance, distance)
                    
                    if distance <= self.distance_threshold:
                        found_nearby = True
                        break
                
                if not found_nearby:
                    missing_objects.append({
                        'ball_type': ball_type,
                        'expected_position': expected_pos,
                        'nearest_detection_distance': min_distance if min_distance != float('inf') else None,
                        'search_radius': self.distance_threshold
                    })
        
        return missing_objects
    
    def get_duplication_summary(self) -> Dict[str, Any]:
        """
        Get summary of duplication errors across all evaluations.
        
        Returns:
            Dictionary with duplication error summary
        """
        duplication_summary = {
            'total_moments_with_duplications': 0,
            'total_duplication_errors': 0,
            'duplications_by_ball_type': {},
            'average_duplicate_distance': 0.0,
            'duplicate_confidence_analysis': {}
        }
        
        all_distances = []
        all_confidences_by_type = {}
        
        for evaluation in self.evaluation_history:
            if evaluation.suppressed or not evaluation.duplication_errors:
                continue
            
            if evaluation.duplication_errors:
                duplication_summary['total_moments_with_duplications'] += 1
            
            for dup_error in evaluation.duplication_errors:
                ball_type = self._normalize_ball_type(dup_error.ball_type)
                
                # Count by ball type
                if ball_type not in duplication_summary['duplications_by_ball_type']:
                    duplication_summary['duplications_by_ball_type'][ball_type] = 0
                duplication_summary['duplications_by_ball_type'][ball_type] += 1
                
                # Collect distances and confidences
                all_distances.extend(dup_error.distances)
                
                if ball_type not in all_confidences_by_type:
                    all_confidences_by_type[ball_type] = []
                all_confidences_by_type[ball_type].extend(dup_error.confidence_scores)
        
        duplication_summary['total_duplication_errors'] = sum(
            duplication_summary['duplications_by_ball_type'].values()
        )
        
        # Calculate average distance
        if all_distances:
            duplication_summary['average_duplicate_distance'] = sum(all_distances) / len(all_distances)
        
        # Analyze confidence scores
        for ball_type, confidences in all_confidences_by_type.items():
            if confidences:
                duplication_summary['duplicate_confidence_analysis'][ball_type] = {
                    'avg_confidence': sum(confidences) / len(confidences),
                    'min_confidence': min(confidences),
                    'max_confidence': max(confidences),
                    'count': len(confidences)
                }
        
        return duplication_summary