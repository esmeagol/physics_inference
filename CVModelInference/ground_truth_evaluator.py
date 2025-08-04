"""
Ground Truth Evaluator for Snooker Tracker Evaluation

This module provides the main orchestration class for evaluating tracker output
against ground truth events with comprehensive moment-based analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math
try:
    from .event_processor import EventProcessor, ProcessedEvent
    from .state_reconstructor import StateReconstructor
    from .moment_evaluator import MomentEvaluator, MomentEvaluation, ErrorType
except ImportError:
    # Handle case when running as script or module
    try:
        from CVModelInference.event_processor import EventProcessor, ProcessedEvent
        from CVModelInference.state_reconstructor import StateReconstructor
        from CVModelInference.moment_evaluator import MomentEvaluator, MomentEvaluation, ErrorType
    except ImportError:
        from event_processor import EventProcessor, ProcessedEvent
        from state_reconstructor import StateReconstructor
        from moment_evaluator import MomentEvaluator, MomentEvaluation, ErrorType


@dataclass
class EvaluationSummary:
    """Summary of complete evaluation results."""
    total_moments: int
    moments_evaluated: int
    moments_suppressed: int
    overall_accuracy: float
    per_ball_accuracy: Dict[str, Dict[str, Any]]
    context_accuracy: Dict[str, Dict[str, float]]
    continuity_stats: Dict[str, Any]
    duplication_summary: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    error_categorization: Dict[str, List[Dict[str, Any]]]


class GroundTruthEvaluator:
    """
    Main orchestration class for ground truth evaluation.
    
    Coordinates EventProcessor, StateReconstructor, and MomentEvaluator to provide
    comprehensive tracker evaluation against ground truth events with time range awareness.
    """
    
    def __init__(self, video_fps: float = 30.0, detection_interval: int = 1,
                 distance_threshold: float = 50.0):
        """
        Initialize the ground truth evaluator.
        
        Args:
            video_fps: Video frame rate for time/frame conversion
            detection_interval: Detection interval in frames
            distance_threshold: Distance threshold for duplication detection (pixels)
        """
        self.video_fps = video_fps
        self.detection_interval = detection_interval
        self.distance_threshold = distance_threshold
        
        # Initialize components
        self.event_processor = EventProcessor(video_fps, detection_interval)
        self.state_reconstructor = StateReconstructor(self.event_processor)
        self.moment_evaluator = MomentEvaluator(self.state_reconstructor, distance_threshold)
        
        # Evaluation state
        self.ground_truth_events: List[Dict[str, Any]] = []
        self.processed_events: List[ProcessedEvent] = []
        self.moment_duration: float = 1.0  # Default 1 second per moment
        self.evaluation_results: List[MomentEvaluation] = []
        self.is_initialized: bool = False
    
    def set_ground_truth_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Set ground truth events for evaluation.
        
        Args:
            events: List of ground truth event dictionaries
            
        Raises:
            ValueError: If events are invalid
        """
        try:
            # Process and validate events
            self.processed_events = self.event_processor.process_events(events)
            self.ground_truth_events = events
            
            # Set events in state reconstructor
            self.state_reconstructor.set_events(events)
            
            self.is_initialized = True
            
        except Exception as e:
            raise ValueError(f"Failed to set ground truth events: {e}")
    
    def set_moment_duration(self, duration: float) -> None:
        """
        Set the duration for each evaluation moment.
        
        Args:
            duration: Duration in seconds for each moment
        """
        if duration <= 0:
            raise ValueError("Moment duration must be positive")
        self.moment_duration = duration
    
    def evaluate_tracker_output(self, tracker_results: List[Dict[str, Any]],
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> EvaluationSummary:
        """
        Evaluate tracker output against ground truth events.
        
        Args:
            tracker_results: List of tracker output dictionaries
                Expected format: [{'timestamp': float, 'detections': [...], 'counts': {...}}, ...]
            start_time: Optional start time for evaluation (defaults to first event)
            end_time: Optional end time for evaluation (defaults to last event)
            
        Returns:
            Complete evaluation summary
            
        Raises:
            ValueError: If evaluator is not initialized or input is invalid
        """
        if not self.is_initialized:
            raise ValueError("Evaluator not initialized. Call set_ground_truth_events first.")
        
        if not tracker_results:
            raise ValueError("No tracker results provided")
        
        # Determine evaluation time range
        eval_start, eval_end = self._determine_evaluation_range(
            tracker_results, start_time, end_time
        )
        
        # Group tracker results into moments
        moments = self._group_tracker_results_into_moments(
            tracker_results, eval_start, eval_end
        )
        
        # Evaluate each moment
        self.evaluation_results = []
        for moment_idx, moment_data in enumerate(moments):
            evaluation = self._evaluate_moment(moment_idx, moment_data)
            self.evaluation_results.append(evaluation)
        
        # Generate comprehensive summary
        return self._generate_evaluation_summary()
    
    def _determine_evaluation_range(self, tracker_results: List[Dict[str, Any]],
                                   start_time: Optional[float],
                                   end_time: Optional[float]) -> Tuple[float, float]:
        """
        Determine the time range for evaluation.
        
        Args:
            tracker_results: Tracker output data
            start_time: Optional explicit start time
            end_time: Optional explicit end time
            
        Returns:
            Tuple of (start_time, end_time)
        """
        # Get time range from tracker results
        tracker_times = [result.get('timestamp', 0) for result in tracker_results]
        tracker_start = min(tracker_times) if tracker_times else 0
        tracker_end = max(tracker_times) if tracker_times else 0
        
        # Get time range from ground truth events
        event_times = []
        for event in self.processed_events:
            event_times.append(event.start_time)
            if event.end_time is not None:
                event_times.append(event.end_time)
        
        gt_start = min(event_times) if event_times else 0
        gt_end = max(event_times) if event_times else 0
        
        # Use explicit times if provided, otherwise use intersection of tracker and GT ranges
        if start_time is not None:
            eval_start = start_time
        else:
            # Use the later of the two start times to ensure we have both GT and tracker data
            eval_start = max(tracker_start, gt_start)
        
        if end_time is not None:
            eval_end = end_time
        else:
            # Use the earlier of the two end times to ensure we have both GT and tracker data
            eval_end = min(tracker_end, gt_end)
            # If no overlap, extend the range to include both
            if eval_start >= eval_end:
                eval_start = min(tracker_start, gt_start)
                eval_end = max(tracker_end, gt_end)
        
        if eval_start >= eval_end:
            raise ValueError(f"Invalid evaluation range: {eval_start} >= {eval_end}")
        
        return eval_start, eval_end
    
    def _group_tracker_results_into_moments(self, tracker_results: List[Dict[str, Any]],
                                           start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Group tracker results into evaluation moments.
        
        Args:
            tracker_results: Tracker output data
            start_time: Start time for evaluation
            end_time: End time for evaluation
            
        Returns:
            List of moment data dictionaries
        """
        moments = []
        current_time = start_time
        
        while current_time < end_time:
            moment_end = min(current_time + self.moment_duration, end_time)
            
            # Find tracker results within this moment
            moment_results = [
                result for result in tracker_results
                if current_time <= result.get('timestamp', 0) < moment_end
            ]
            
            # Aggregate results for this moment
            moment_data = self._aggregate_moment_results(
                moment_results, current_time, moment_end
            )
            
            moments.append(moment_data)
            current_time = moment_end
        
        return moments
    
    def _aggregate_moment_results(self, moment_results: List[Dict[str, Any]],
                                 start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Aggregate tracker results within a single moment.
        
        Args:
            moment_results: Tracker results within the moment
            start_time: Moment start time
            end_time: Moment end time
            
        Returns:
            Aggregated moment data
        """
        if not moment_results:
            # No tracker data for this moment
            return {
                'timestamp': (start_time + end_time) / 2,
                'start_time': start_time,
                'end_time': end_time,
                'detected_counts': {},
                'detected_positions': [],
                'num_frames': 0
            }
        
        # Use the middle timestamp as representative
        mid_timestamp = (start_time + end_time) / 2
        
        # Aggregate counts (use the most recent result in the moment)
        latest_result = max(moment_results, key=lambda x: x.get('timestamp', 0))
        detected_counts = latest_result.get('counts', {})
        
        # Aggregate positions (use all detections from the latest frame)
        detected_positions = latest_result.get('detections', [])
        
        return {
            'timestamp': mid_timestamp,
            'start_time': start_time,
            'end_time': end_time,
            'detected_counts': detected_counts,
            'detected_positions': detected_positions,
            'num_frames': len(moment_results),
            'raw_results': moment_results
        }
    
    def _evaluate_moment(self, moment_idx: int, moment_data: Dict[str, Any]) -> MomentEvaluation:
        """
        Evaluate a single moment using the MomentEvaluator.
        
        Args:
            moment_idx: Index of the moment
            moment_data: Aggregated moment data
            
        Returns:
            Moment evaluation result
        """
        return self.moment_evaluator.evaluate_moment(
            moment_idx=moment_idx,
            timestamp=moment_data['timestamp'],
            detected_counts=moment_data['detected_counts'],
            detected_positions=moment_data['detected_positions']
        )
    
    def _generate_evaluation_summary(self) -> EvaluationSummary:
        """
        Generate comprehensive evaluation summary from all moment evaluations.
        
        Returns:
            Complete evaluation summary
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        # Basic statistics
        total_moments = len(self.evaluation_results)
        moments_suppressed = sum(1 for eval_result in self.evaluation_results if eval_result.suppressed)
        moments_evaluated = total_moments - moments_suppressed
        
        # Calculate overall accuracy
        overall_accuracy = self._calculate_overall_accuracy()
        
        # Get detailed metrics from moment evaluator
        per_ball_accuracy = self.moment_evaluator.calculate_per_ball_accuracy()
        context_accuracy = self.moment_evaluator.get_context_based_accuracy()
        continuity_stats = self.moment_evaluator.analyze_tracking_continuity()
        duplication_summary = self.moment_evaluator.get_duplication_summary()
        
        # Generate temporal analysis
        temporal_analysis = self._analyze_temporal_patterns()
        
        # Categorize errors
        error_categorization = self.moment_evaluator.distinguish_game_events_from_tracking_failures()
        
        return EvaluationSummary(
            total_moments=total_moments,
            moments_evaluated=moments_evaluated,
            moments_suppressed=moments_suppressed,
            overall_accuracy=overall_accuracy,
            per_ball_accuracy=per_ball_accuracy,
            context_accuracy=context_accuracy,
            continuity_stats=continuity_stats,
            duplication_summary=duplication_summary,
            temporal_analysis=temporal_analysis,
            error_categorization=error_categorization
        )
    
    def _calculate_overall_accuracy(self) -> float:
        """
        Calculate overall accuracy across all moments and ball types.
        
        Returns:
            Overall accuracy percentage
        """
        total_evaluations = 0
        correct_evaluations = 0
        
        for evaluation in self.evaluation_results:
            if evaluation.suppressed:
                continue
            
            for error in evaluation.count_errors:
                total_evaluations += 1
                if error.error_type == ErrorType.CORRECT:
                    correct_evaluations += 1
        
        if total_evaluations == 0:
            return 0.0
        
        return (correct_evaluations / total_evaluations) * 100
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in evaluation results.
        
        Returns:
            Dictionary with temporal analysis results
        """
        if not self.evaluation_results:
            return {}
        
        # Accuracy over time
        accuracy_over_time = []
        error_counts_over_time = []
        
        for evaluation in self.evaluation_results:
            if evaluation.suppressed:
                accuracy_over_time.append(None)  # Skip suppressed moments
                error_counts_over_time.append(0)
                continue
            
            # Calculate accuracy for this moment
            total_errors = len(evaluation.count_errors)
            correct_errors = sum(1 for error in evaluation.count_errors 
                               if error.error_type == ErrorType.CORRECT)
            
            moment_accuracy = (correct_errors / total_errors * 100) if total_errors > 0 else 100
            accuracy_over_time.append(moment_accuracy)
            
            # Count different types of errors
            error_count = len([error for error in evaluation.count_errors 
                             if error.error_type != ErrorType.CORRECT])
            error_count += len(evaluation.illegal_changes)
            error_count += len(evaluation.duplication_errors)
            error_counts_over_time.append(error_count)
        
        # Find periods of consistent errors
        error_periods = self._identify_error_periods(error_counts_over_time)
        
        # Analyze accuracy trends
        accuracy_trend = self._analyze_accuracy_trend(accuracy_over_time)
        
        return {
            'accuracy_over_time': accuracy_over_time,
            'error_counts_over_time': error_counts_over_time,
            'error_periods': error_periods,
            'accuracy_trend': accuracy_trend,
            'timestamps': [eval_result.timestamp for eval_result in self.evaluation_results]
        }
    
    def _identify_error_periods(self, error_counts: List[int]) -> List[Dict[str, Any]]:
        """
        Identify periods of consistent errors.
        
        Args:
            error_counts: List of error counts per moment
            
        Returns:
            List of error period descriptions
        """
        error_periods = []
        current_period_start = None
        current_period_errors = 0
        
        for i, error_count in enumerate(error_counts):
            if error_count > 0:
                if current_period_start is None:
                    current_period_start = i
                    current_period_errors = error_count
                else:
                    current_period_errors += error_count
            else:
                if current_period_start is not None:
                    # End of error period
                    error_periods.append({
                        'start_moment': current_period_start,
                        'end_moment': i - 1,
                        'duration_moments': i - current_period_start,
                        'total_errors': current_period_errors,
                        'avg_errors_per_moment': current_period_errors / (i - current_period_start)
                    })
                    current_period_start = None
                    current_period_errors = 0
        
        # Handle case where errors continue to the end
        if current_period_start is not None:
            error_periods.append({
                'start_moment': current_period_start,
                'end_moment': len(error_counts) - 1,
                'duration_moments': len(error_counts) - current_period_start,
                'total_errors': current_period_errors,
                'avg_errors_per_moment': current_period_errors / (len(error_counts) - current_period_start)
            })
        
        return error_periods
    
    def _analyze_accuracy_trend(self, accuracy_values: List[Optional[float]]) -> Dict[str, Any]:
        """
        Analyze accuracy trend over time.
        
        Args:
            accuracy_values: List of accuracy values (None for suppressed moments)
            
        Returns:
            Dictionary with trend analysis
        """
        # Filter out None values
        valid_accuracies = [acc for acc in accuracy_values if acc is not None]
        
        if len(valid_accuracies) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple linear trend analysis
        n = len(valid_accuracies)
        x_values = list(range(n))
        
        # Calculate linear regression slope
        x_mean = sum(x_values) / n
        y_mean = sum(valid_accuracies) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, valid_accuracies))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Categorize trend
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'slope': slope,
            'initial_accuracy': valid_accuracies[0],
            'final_accuracy': valid_accuracies[-1],
            'accuracy_change': valid_accuracies[-1] - valid_accuracies[0],
            'min_accuracy': min(valid_accuracies),
            'max_accuracy': max(valid_accuracies),
            'avg_accuracy': sum(valid_accuracies) / len(valid_accuracies)
        }
    
    def get_moment_evaluation(self, moment_idx: int) -> Optional[MomentEvaluation]:
        """
        Get evaluation result for a specific moment.
        
        Args:
            moment_idx: Index of the moment
            
        Returns:
            Moment evaluation result or None if not found
        """
        if 0 <= moment_idx < len(self.evaluation_results):
            return self.evaluation_results[moment_idx]
        return None
    
    def get_evaluations_for_time_range(self, start_time: float, 
                                      end_time: float) -> List[MomentEvaluation]:
        """
        Get evaluation results for a specific time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of evaluation results within the time range
        """
        return [
            evaluation for evaluation in self.evaluation_results
            if start_time <= evaluation.timestamp <= end_time
        ]
    
    def get_state_at_time(self, timestamp: float) -> Optional[Any]:
        """
        Get expected ground truth state at a specific time.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Expected ball state or None if evaluator not initialized
        """
        if not self.is_initialized:
            return None
        
        try:
            return self.state_reconstructor.get_state_at_time(timestamp)
        except Exception:
            return None
    
    def validate_tracker_output_format(self, tracker_results: List[Dict[str, Any]]) -> List[str]:
        """
        Validate the format of tracker output data.
        
        Args:
            tracker_results: Tracker output to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(tracker_results, list):
            errors.append("Tracker results must be a list")
            return errors
        
        for i, result in enumerate(tracker_results):
            if not isinstance(result, dict):
                errors.append(f"Result {i}: Must be a dictionary")
                continue
            
            # Check required fields
            if 'timestamp' not in result:
                errors.append(f"Result {i}: Missing 'timestamp' field")
            elif not isinstance(result['timestamp'], (int, float)):
                errors.append(f"Result {i}: 'timestamp' must be a number")
            
            # Check optional fields
            if 'counts' in result and not isinstance(result['counts'], dict):
                errors.append(f"Result {i}: 'counts' must be a dictionary")
            
            if 'detections' in result and not isinstance(result['detections'], list):
                errors.append(f"Result {i}: 'detections' must be a list")
        
        return errors
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the evaluation.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.evaluation_results:
            return {'error': 'No evaluation results available'}
        
        # Count different types of issues
        total_count_errors = sum(len(eval_result.count_errors) for eval_result in self.evaluation_results)
        total_illegal_changes = sum(len(eval_result.illegal_changes) for eval_result in self.evaluation_results)
        total_duplications = sum(len(eval_result.duplication_errors) for eval_result in self.evaluation_results)
        
        # Count correct evaluations
        correct_count_evaluations = sum(
            sum(1 for error in eval_result.count_errors if error.error_type == ErrorType.CORRECT)
            for eval_result in self.evaluation_results
        )
        
        return {
            'total_moments': len(self.evaluation_results),
            'moments_with_data': sum(1 for eval_result in self.evaluation_results if not eval_result.suppressed),
            'moments_suppressed': sum(1 for eval_result in self.evaluation_results if eval_result.suppressed),
            'total_count_evaluations': total_count_errors,
            'correct_count_evaluations': correct_count_evaluations,
            'total_illegal_changes': total_illegal_changes,
            'total_duplications': total_duplications,
            'overall_accuracy': self._calculate_overall_accuracy(),
            'evaluation_time_range': (
                min(eval_result.timestamp for eval_result in self.evaluation_results),
                max(eval_result.timestamp for eval_result in self.evaluation_results)
            ) if self.evaluation_results else (0, 0)
        }
    
    def generate_detailed_report(self, include_moment_details: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report with time range context.
        
        Args:
            include_moment_details: Whether to include detailed moment-by-moment results
            
        Returns:
            Detailed evaluation report dictionary
        """
        if not self.evaluation_results:
            return {'error': 'No evaluation results available'}
        
        # Get evaluation summary
        summary = self._generate_evaluation_summary()
        
        # Create detailed report structure
        report = {
            'evaluation_metadata': {
                'video_fps': self.video_fps,
                'detection_interval': self.detection_interval,
                'moment_duration': self.moment_duration,
                'distance_threshold': self.distance_threshold,
                'total_ground_truth_events': len(self.ground_truth_events),
                'evaluation_time_range': (
                    min(eval_result.timestamp for eval_result in self.evaluation_results),
                    max(eval_result.timestamp for eval_result in self.evaluation_results)
                )
            },
            
            'summary_statistics': {
                'total_moments': summary.total_moments,
                'moments_evaluated': summary.moments_evaluated,
                'moments_suppressed': summary.moments_suppressed,
                'overall_accuracy': summary.overall_accuracy,
                'suppression_rate': (summary.moments_suppressed / summary.total_moments * 100) if summary.total_moments > 0 else 0
            },
            
            'accuracy_analysis': {
                'per_ball_accuracy': summary.per_ball_accuracy,
                'context_based_accuracy': summary.context_accuracy,
                'temporal_patterns': summary.temporal_analysis
            },
            
            'error_analysis': {
                'continuity_analysis': summary.continuity_stats,
                'duplication_analysis': summary.duplication_summary,
                'error_categorization': summary.error_categorization,
                'error_distribution': self._analyze_error_distribution()
            },
            
            'transition_period_analysis': self._analyze_transition_periods(),
            
            'recommendations': self._generate_recommendations(summary)
        }
        
        # Add moment-by-moment details if requested
        if include_moment_details:
            report['moment_details'] = self._generate_moment_details()
        
        return report
    
    def _analyze_error_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of different error types.
        
        Returns:
            Dictionary with error distribution analysis
        """
        error_distribution = {
            'count_errors': {
                'over_detection': 0,
                'under_detection': 0,
                'correct': 0
            },
            'illegal_changes': {
                'disappearances': 0,
                'reappearances': 0
            },
            'duplications': 0,
            'by_ball_type': {},
            'by_context': {}
        }
        
        for evaluation in self.evaluation_results:
            if evaluation.suppressed:
                continue
            
            # Count error types
            for error in evaluation.count_errors:
                if error.error_type == ErrorType.OVER_DETECTION:
                    error_distribution['count_errors']['over_detection'] += 1
                elif error.error_type == ErrorType.UNDER_DETECTION:
                    error_distribution['count_errors']['under_detection'] += 1
                elif error.error_type == ErrorType.CORRECT:
                    error_distribution['count_errors']['correct'] += 1
                
                # By ball type
                ball_type = error.ball_type
                if ball_type not in error_distribution['by_ball_type']:
                    error_distribution['by_ball_type'][ball_type] = {
                        'over_detection': 0, 'under_detection': 0, 'correct': 0
                    }
                
                if error.error_type == ErrorType.OVER_DETECTION:
                    error_distribution['by_ball_type'][ball_type]['over_detection'] += 1
                elif error.error_type == ErrorType.UNDER_DETECTION:
                    error_distribution['by_ball_type'][ball_type]['under_detection'] += 1
                else:
                    error_distribution['by_ball_type'][ball_type]['correct'] += 1
                
                # By context
                context = error.context
                if context not in error_distribution['by_context']:
                    error_distribution['by_context'][context] = {
                        'over_detection': 0, 'under_detection': 0, 'correct': 0
                    }
                
                if error.error_type == ErrorType.OVER_DETECTION:
                    error_distribution['by_context'][context]['over_detection'] += 1
                elif error.error_type == ErrorType.UNDER_DETECTION:
                    error_distribution['by_context'][context]['under_detection'] += 1
                else:
                    error_distribution['by_context'][context]['correct'] += 1
            
            # Count illegal changes
            for change in evaluation.illegal_changes:
                if change.change_type == ErrorType.ILLEGAL_DISAPPEARANCE:
                    error_distribution['illegal_changes']['disappearances'] += 1
                elif change.change_type == ErrorType.ILLEGAL_REAPPEARANCE:
                    error_distribution['illegal_changes']['reappearances'] += 1
            
            # Count duplications
            error_distribution['duplications'] += len(evaluation.duplication_errors)
        
        return error_distribution
    
    def _analyze_transition_periods(self) -> Dict[str, Any]:
        """
        Analyze performance during transition periods (events with time ranges).
        
        Returns:
            Dictionary with transition period analysis
        """
        transition_analysis = {
            'total_transition_moments': 0,
            'transition_accuracy': 0.0,
            'stable_period_accuracy': 0.0,
            'transition_vs_stable_comparison': {},
            'event_type_performance': {}
        }
        
        transition_evaluations = 0
        transition_correct = 0
        stable_evaluations = 0
        stable_correct = 0
        
        event_performance = {}
        
        for evaluation in self.evaluation_results:
            if evaluation.suppressed:
                continue
            
            # Determine if this moment is during a transition
            is_transition = any(
                'transition_period' in error.context
                for error in evaluation.count_errors
            )
            
            # Count evaluations and correct ones
            moment_evaluations = len(evaluation.count_errors)
            moment_correct = sum(1 for error in evaluation.count_errors 
                               if error.error_type == ErrorType.CORRECT)
            
            if is_transition:
                transition_analysis['total_transition_moments'] += 1
                transition_evaluations += moment_evaluations
                transition_correct += moment_correct
                
                # Analyze by event type
                for event in evaluation.active_events:
                    event_key = f"{event.event_type.value}"
                    if event_key not in event_performance:
                        event_performance[event_key] = {'total': 0, 'correct': 0}
                    event_performance[event_key]['total'] += moment_evaluations
                    event_performance[event_key]['correct'] += moment_correct
            else:
                stable_evaluations += moment_evaluations
                stable_correct += moment_correct
        
        # Calculate accuracies
        if transition_evaluations > 0:
            transition_analysis['transition_accuracy'] = (transition_correct / transition_evaluations) * 100
        
        if stable_evaluations > 0:
            transition_analysis['stable_period_accuracy'] = (stable_correct / stable_evaluations) * 100
        
        # Compare transition vs stable
        if transition_evaluations > 0 and stable_evaluations > 0:
            accuracy_diff = transition_analysis['transition_accuracy'] - transition_analysis['stable_period_accuracy']
            transition_analysis['transition_vs_stable_comparison'] = {
                'accuracy_difference': accuracy_diff,
                'transition_is_better': accuracy_diff > 0,
                'performance_impact': 'positive' if accuracy_diff > 5 else 'negative' if accuracy_diff < -5 else 'minimal'
            }
        
        # Event type performance
        for event_type, stats in event_performance.items():
            if stats['total'] > 0:
                transition_analysis['event_type_performance'][event_type] = {
                    'accuracy': (stats['correct'] / stats['total']) * 100,
                    'total_evaluations': stats['total'],
                    'correct_evaluations': stats['correct']
                }
        
        return transition_analysis
    
    def _generate_recommendations(self, summary: EvaluationSummary) -> List[Dict[str, str]]:
        """
        Generate recommendations based on evaluation results.
        
        Args:
            summary: Evaluation summary
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Overall accuracy recommendations
        if summary.overall_accuracy < 70:
            recommendations.append({
                'category': 'accuracy',
                'priority': 'high',
                'issue': f'Low overall accuracy ({summary.overall_accuracy:.1f}%)',
                'recommendation': 'Consider improving detection model or adjusting detection thresholds'
            })
        elif summary.overall_accuracy < 85:
            recommendations.append({
                'category': 'accuracy',
                'priority': 'medium',
                'issue': f'Moderate accuracy ({summary.overall_accuracy:.1f}%)',
                'recommendation': 'Fine-tune detection parameters for better performance'
            })
        
        # Per-ball accuracy recommendations
        for ball_type, stats in summary.per_ball_accuracy.items():
            if stats['accuracy'] < 60:
                recommendations.append({
                    'category': 'ball_specific',
                    'priority': 'high',
                    'issue': f'Poor {ball_type} ball detection ({stats["accuracy"]:.1f}%)',
                    'recommendation': f'Focus on improving {ball_type} ball detection - check lighting, color calibration, or model training'
                })
        
        # Continuity recommendations
        if summary.continuity_stats.get('continuity_percentage', 100) < 80:
            recommendations.append({
                'category': 'continuity',
                'priority': 'high',
                'issue': f'Poor tracking continuity ({summary.continuity_stats.get("continuity_percentage", 0):.1f}%)',
                'recommendation': 'Implement tracking smoothing or improve object association between frames'
            })
        
        # Duplication recommendations
        if summary.duplication_summary.get('total_duplication_errors', 0) > 0:
            recommendations.append({
                'category': 'duplication',
                'priority': 'medium',
                'issue': f'{summary.duplication_summary["total_duplication_errors"]} duplication errors detected',
                'recommendation': 'Implement non-maximum suppression or adjust detection confidence thresholds'
            })
        
        # Context-based recommendations
        context_accuracies = summary.context_accuracy
        if 'stable_period' in context_accuracies and 'transition_period_potting' in context_accuracies:
            stable_acc = context_accuracies['stable_period']['accuracy']
            transition_acc = context_accuracies['transition_period_potting']['accuracy']
            
            if stable_acc - transition_acc > 20:
                recommendations.append({
                    'category': 'transition_handling',
                    'priority': 'medium',
                    'issue': f'Poor performance during transitions ({transition_acc:.1f}% vs {stable_acc:.1f}% stable)',
                    'recommendation': 'Implement temporal smoothing or event-aware detection during ball movements'
                })
        
        # Temporal trend recommendations
        temporal_analysis = summary.temporal_analysis
        if temporal_analysis.get('accuracy_trend', {}).get('trend') == 'declining':
            recommendations.append({
                'category': 'temporal',
                'priority': 'medium',
                'issue': 'Accuracy declining over time',
                'recommendation': 'Check for model drift, lighting changes, or accumulating tracking errors'
            })
        
        return recommendations
    
    def _generate_moment_details(self) -> List[Dict[str, Any]]:
        """
        Generate detailed information for each evaluation moment.
        
        Returns:
            List of moment detail dictionaries
        """
        moment_details = []
        
        for evaluation in self.evaluation_results:
            detail = {
                'moment_idx': evaluation.moment_idx,
                'timestamp': evaluation.timestamp,
                'suppressed': evaluation.suppressed,
                'expected_state': {
                    'ball_counts': evaluation.expected_state.ball_counts,
                    'occluded_balls': evaluation.expected_state.occluded_balls,
                    'ignore_errors': evaluation.expected_state.ignore_errors
                },
                'detected_counts': evaluation.detected_counts,
                'active_events': [
                    {
                        'type': event.event_type.value,
                        'start_time': event.start_time,
                        'end_time': event.end_time,
                        'data': event.data
                    }
                    for event in evaluation.active_events
                ],
                'count_errors': [
                    {
                        'ball_type': error.ball_type,
                        'expected': error.expected,
                        'detected': error.detected,
                        'error_type': error.error_type.value,
                        'error_magnitude': error.error_magnitude,
                        'context': error.context
                    }
                    for error in evaluation.count_errors
                ],
                'illegal_changes': [
                    {
                        'ball_type': change.ball_type,
                        'change_type': change.change_type.value,
                        'previous_count': change.previous_count,
                        'current_count': change.current_count,
                        'context': change.context
                    }
                    for change in evaluation.illegal_changes
                ],
                'duplication_errors': [
                    {
                        'ball_type': dup.ball_type,
                        'num_duplicates': len(dup.positions),
                        'positions': dup.positions,
                        'min_distance': min(dup.distances) if dup.distances else 0
                    }
                    for dup in evaluation.duplication_errors
                ]
            }
            
            moment_details.append(detail)
        
        return moment_details
    
    def export_results_to_csv(self, filepath: str) -> None:
        """
        Export evaluation results to CSV format.
        
        Args:
            filepath: Path to save the CSV file
        """
        import csv
        
        if not self.evaluation_results:
            raise ValueError("No evaluation results to export")
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'moment_idx', 'timestamp', 'suppressed', 'ball_type', 
                'expected_count', 'detected_count', 'error_type', 
                'error_magnitude', 'context', 'active_events'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for evaluation in self.evaluation_results:
                active_events_str = ';'.join([
                    f"{event.event_type.value}@{event.start_time}"
                    for event in evaluation.active_events
                ])
                
                if evaluation.count_errors:
                    for error in evaluation.count_errors:
                        writer.writerow({
                            'moment_idx': evaluation.moment_idx,
                            'timestamp': evaluation.timestamp,
                            'suppressed': evaluation.suppressed,
                            'ball_type': error.ball_type,
                            'expected_count': error.expected,
                            'detected_count': error.detected,
                            'error_type': error.error_type.value,
                            'error_magnitude': error.error_magnitude,
                            'context': error.context,
                            'active_events': active_events_str
                        })
                else:
                    # Write a row even if no errors (for completeness)
                    writer.writerow({
                        'moment_idx': evaluation.moment_idx,
                        'timestamp': evaluation.timestamp,
                        'suppressed': evaluation.suppressed,
                        'ball_type': '',
                        'expected_count': '',
                        'detected_count': '',
                        'error_type': '',
                        'error_magnitude': '',
                        'context': '',
                        'active_events': active_events_str
                    })
    
    def export_results_to_json(self, filepath: str, include_moment_details: bool = True) -> None:
        """
        Export evaluation results to JSON format.
        
        Args:
            filepath: Path to save the JSON file
            include_moment_details: Whether to include detailed moment information
        """
        import json
        
        report = self.generate_detailed_report(include_moment_details)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(report, jsonfile, indent=2, default=str)
    
    def print_summary_report(self) -> None:
        """
        Print a human-readable summary report to console.
        """
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        summary = self._generate_evaluation_summary()
        
        print("=" * 60)
        print("GROUND TRUTH EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nEvaluation Overview:")
        print(f"  Total moments: {summary.total_moments}")
        print(f"  Moments evaluated: {summary.moments_evaluated}")
        print(f"  Moments suppressed: {summary.moments_suppressed}")
        print(f"  Overall accuracy: {summary.overall_accuracy:.1f}%")
        
        print(f"\nPer-Ball Accuracy:")
        for ball_type, stats in summary.per_ball_accuracy.items():
            print(f"  {ball_type.capitalize()}: {stats['accuracy']:.1f}% "
                  f"({stats['correct_moments']}/{stats['total_moments']} moments)")
        
        print(f"\nContext-Based Performance:")
        for context, stats in summary.context_accuracy.items():
            print(f"  {context.replace('_', ' ').title()}: {stats['accuracy']:.1f}% "
                  f"({stats['correct_evaluations']}/{stats['total_evaluations']} evaluations)")
        
        print(f"\nTracking Continuity:")
        continuity = summary.continuity_stats
        print(f"  Continuity: {continuity.get('continuity_percentage', 0):.1f}%")
        print(f"  Illegal disappearances: {continuity.get('total_illegal_disappearances', 0)}")
        print(f"  Illegal reappearances: {continuity.get('total_illegal_reappearances', 0)}")
        
        print(f"\nDuplication Analysis:")
        duplication = summary.duplication_summary
        print(f"  Total duplications: {duplication.get('total_duplication_errors', 0)}")
        print(f"  Moments with duplications: {duplication.get('total_moments_with_duplications', 0)}")
        
        # Print recommendations
        recommendations = self._generate_recommendations(summary)
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"  {i}. [{rec['priority'].upper()}] {rec['issue']}")
                print(f"     → {rec['recommendation']}")
        
        print("=" * 60)