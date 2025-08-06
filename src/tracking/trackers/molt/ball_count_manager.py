"""
Ball counting and verification logic for snooker/billiards tracking.

This module provides the BallCountManager class for tracking ball counts
and handling count violations in snooker and billiards games.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .types import Track


class BallCountManager:
    """
    Manages ball counting and verification logic specific to snooker.
    
    This class tracks the current number of balls of each type and compares
    them against expected counts. It provides methods for handling count
    violations such as lost balls or duplicate detections.
    """
    
    def __init__(self, expected_counts: Dict[str, int]) -> None:
        """
        Initialize the BallCountManager with expected ball counts for the game.
        
        The BallCountManager is responsible for monitoring the number of tracked
        balls of each type and ensuring they match the expected counts for the
        specific game being played (snooker, pool, etc.). It detects violations
        and provides suggestions for resolving count discrepancies.
        
        Args:
            expected_counts (Dict[str, int]): Dictionary mapping ball colors/types
                to their expected counts. Keys should be ball color names (e.g.,
                'red', 'white', 'yellow') and values should be non-negative integers
                representing the expected number of balls of that type.
                
                Examples:
                - Snooker: {'white': 1, 'red': 15, 'yellow': 1, 'green': 1, ...}
                - 8-ball pool: {'white': 1, 'yellow': 7, 'red': 7, 'black': 1}
                - 9-ball pool: {'white': 1, 'yellow': 9}
        
        Raises:
            ValueError: If expected_counts is empty, contains invalid ball class
                names (non-string or empty), or invalid counts (negative integers).
                
        Example:
            >>> # Snooker ball count manager
            >>> snooker_counts = {
            ...     'white': 1, 'red': 15, 'yellow': 1, 'green': 1,
            ...     'brown': 1, 'blue': 1, 'pink': 1, 'black': 1
            ... }
            >>> manager = BallCountManager(snooker_counts)
            
            >>> # Pool ball count manager
            >>> pool_counts = {'white': 1, 'yellow': 7, 'red': 7, 'black': 1}
            >>> manager = BallCountManager(pool_counts)
        
        Initialization Effects:
            - Sets up expected counts for all ball types
            - Initializes current counts to zero for all ball types
            - Clears track assignment mappings
            - Resets all violation statistics and history
        """
        if not expected_counts:
            raise ValueError("Expected counts cannot be empty")
        
        # Validate expected counts
        for ball_class, count in expected_counts.items():
            if not isinstance(ball_class, str) or not ball_class.strip():
                raise ValueError(f"Ball class must be a non-empty string, got: {ball_class}")
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Expected count must be a non-negative integer, got: {count}")
        
        # Store expected counts
        self.expected_counts = expected_counts.copy()
        
        # Initialize current counts (all start at 0)
        self.current_counts: Dict[str, int] = {
            ball_class: 0 for ball_class in expected_counts.keys()
        }
        
        # Track assignments between track IDs and ball classes
        self.track_assignments: Dict[int, str] = {}
        
        # Statistics for monitoring
        self.total_count_violations = 0
        self.lost_ball_recoveries = 0
        self.duplicate_ball_merges = 0
        
        # History of count violations for analysis
        self.violation_history: List[Dict[str, Any]] = []
    
    def update_counts_from_tracks(self, tracks: List[Track]) -> None:
        """
        Update current ball counts from tracking results.
        
        Args:
            tracks: List of current tracking results with 'id' and 'class' fields
        """
        # Reset current counts
        self.current_counts = {
            ball_class: 0 for ball_class in self.expected_counts.keys()
        }
        
        # Clear old track assignments
        self.track_assignments.clear()
        
        # Count balls from current tracks
        for track in tracks:
            track_id = track.get('id')
            ball_class = track.get('class')
            
            if track_id is not None and ball_class is not None:
                # Update count for this ball class
                if ball_class in self.current_counts:
                    self.current_counts[ball_class] += 1
                else:
                    # Handle unknown ball class
                    self.current_counts[ball_class] = 1
                
                # Update track assignment
                self.track_assignments[track_id] = ball_class
    
    def verify_counts(self) -> bool:
        """
        Verify that current ball counts match expected counts for all ball types.
        
        This method performs a comprehensive check of all ball counts, comparing
        the current number of tracked balls against the expected counts. It
        identifies violations (over-count or under-count) and records them for
        analysis and potential corrective action.
        
        The verification process:
        1. Compares current vs. expected counts for each ball type
        2. Identifies violations (over-count or under-count scenarios)
        3. Records violation details in the violation history
        4. Updates violation statistics
        5. Returns overall validity status
        
        Returns:
            bool: True if all ball counts match expected values exactly,
                False if any violations are detected.
                
        Side Effects:
            - Updates violation_history with any detected violations
            - Increments total_count_violations counter if violations found
            - Maintains rolling history of recent violations (last 100)
            
        Example:
            >>> # Update counts from current tracking results
            >>> manager.update_counts_from_tracks(current_tracks)
            >>> 
            >>> # Verify counts
            >>> if manager.verify_counts():
            ...     print("All ball counts are correct")
            ... else:
            ...     print("Ball count violations detected")
            ...     violations = manager.get_count_violations()
            ...     for ball_type, info in violations.items():
            ...         if info['violation_type'] != 'none':
            ...             print(f"  {ball_type}: {info['violation_type']}")
        
        Use Cases:
            - Continuous monitoring during tracking
            - Quality assurance for tracking accuracy
            - Triggering corrective actions (track merging, recovery)
            - Performance evaluation and debugging
        
        Note:
            - Should be called after update_counts_from_tracks()
            - Violations are automatically logged with timestamps
            - Use get_count_violations() for detailed violation information
        """
        violations_found = False
        
        for ball_class, expected_count in self.expected_counts.items():
            current_count = self.current_counts.get(ball_class, 0)
            
            if current_count != expected_count:
                violations_found = True
                
                # Record violation
                violation = {
                    'ball_class': ball_class,
                    'expected': expected_count,
                    'current': current_count,
                    'violation_type': 'over_count' if current_count > expected_count else 'under_count'
                }
                self.violation_history.append(violation)
                
                # Keep only recent violations (last 100)
                if len(self.violation_history) > 100:
                    self.violation_history.pop(0)
        
        if violations_found:
            self.total_count_violations += 1
        
        return not violations_found
    
    def get_count_violations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about current count violations.
        
        Returns:
            Dict mapping ball classes to violation details:
            {
                'ball_class': {
                    'expected': int,
                    'current': int,
                    'difference': int,
                    'violation_type': str  # 'over_count', 'under_count', or 'none'
                }
            }
        """
        violations = {}
        
        for ball_class, expected_count in self.expected_counts.items():
            current_count = self.current_counts.get(ball_class, 0)
            difference = current_count - expected_count
            
            if difference > 0:
                violation_type = 'over_count'
            elif difference < 0:
                violation_type = 'under_count'
            else:
                violation_type = 'none'
            
            violations[ball_class] = {
                'expected': expected_count,
                'current': current_count,
                'difference': difference,
                'violation_type': violation_type
            }
        
        return violations
    
    def handle_lost_ball(self, ball_class: str) -> Optional[int]:
        """
        Handle the case when a ball is lost (count below expected).
        
        This method provides guidance for track recovery by identifying
        which ball class needs to be recovered.
        
        Args:
            ball_class: The ball class that is missing
            
        Returns:
            Optional[int]: Suggested track ID to reassign, or None if no suggestion
        """
        if ball_class not in self.expected_counts:
            return None
        
        expected_count = self.expected_counts[ball_class]
        current_count = self.current_counts.get(ball_class, 0)
        
        if current_count >= expected_count:
            # No ball is actually lost
            return None
        
        # Record recovery attempt
        self.lost_ball_recoveries += 1
        
        # For now, return None as track reassignment logic would be complex
        # In a full implementation, this could analyze track history to suggest
        # which track might have been misclassified
        return None
    
    def handle_duplicate_ball(self, ball_class: str, track_ids: List[int]) -> List[int]:
        """
        Handle the case when there are too many balls of a given class.
        
        This method identifies which tracks should be merged or reassigned
        to resolve count violations.
        
        Args:
            ball_class: The ball class that has too many instances
            track_ids: List of track IDs for this ball class
            
        Returns:
            List[int]: List of track IDs that should be merged or reassigned
        """
        if ball_class not in self.expected_counts:
            return []
        
        expected_count = self.expected_counts[ball_class]
        current_count = len(track_ids)
        
        if current_count <= expected_count:
            # No duplicates to handle
            return []
        
        # Record merge attempt
        self.duplicate_ball_merges += 1
        
        # Return excess track IDs (keep the first expected_count tracks)
        excess_tracks = track_ids[expected_count:]
        
        return excess_tracks
    
    def suggest_track_merges(self, tracks: List[Track]) -> List[Tuple[int, int]]:
        """
        Suggest track pairs that should be merged to resolve count violations.
        
        This method analyzes tracks with the same ball class and suggests
        which pairs should be merged based on spatial proximity and appearance similarity.
        
        Args:
            tracks: List of current tracking results
            
        Returns:
            List[Tuple[int, int]]: List of (track_id1, track_id2) pairs to merge
        """
        merge_suggestions = []
        
        # Group tracks by ball class
        tracks_by_class: Dict[str, List[Track]] = {}
        for track in tracks:
            ball_class = track.get('class')
            if ball_class:
                if ball_class not in tracks_by_class:
                    tracks_by_class[ball_class] = []
                tracks_by_class[ball_class].append(track)
        
        # Find classes with too many tracks
        for ball_class, class_tracks in tracks_by_class.items():
            expected_count = self.expected_counts.get(ball_class, 1)
            
            if len(class_tracks) > expected_count:
                # Find closest track pairs for merging
                merge_pairs = self._find_closest_track_pairs(class_tracks, len(class_tracks) - expected_count)
                merge_suggestions.extend(merge_pairs)
        
        return merge_suggestions
    
    def _find_closest_track_pairs(self, tracks: List[Track], num_merges: int) -> List[Tuple[int, int]]:
        """
        Find the closest pairs of tracks for merging.
        
        Args:
            tracks: List of tracks of the same class
            num_merges: Number of merge pairs needed
            
        Returns:
            List[Tuple[int, int]]: List of track ID pairs to merge
        """
        if len(tracks) < 2 or num_merges <= 0:
            return []
        
        # Calculate distances between all track pairs
        distances = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track1, track2 = tracks[i], tracks[j]
                
                # Calculate spatial distance
                x1, y1 = track1.get('x', 0), track1.get('y', 0)
                x2, y2 = track2.get('x', 0), track2.get('y', 0)
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                distances.append((distance, track1.get('id'), track2.get('id')))
        
        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])
        
        # Return the closest pairs up to num_merges
        merge_pairs: List[Tuple[int, int]] = []
        used_tracks = set()
        
        for distance, track_id1, track_id2 in distances:
            if len(merge_pairs) >= num_merges:
                break
            
            # Only merge if neither track is already involved in a merge
            if (track_id1 is not None and track_id2 is not None and 
                track_id1 not in used_tracks and track_id2 not in used_tracks):
                merge_pairs.append((track_id1, track_id2))
                used_tracks.add(track_id1)
                used_tracks.add(track_id2)
        
        return merge_pairs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ball count management.
        
        Returns:
            Dict[str, Any]: Statistics including violations, recoveries, and merges
        """
        return {
            'expected_counts': self.expected_counts.copy(),
            'current_counts': self.current_counts.copy(),
            'total_count_violations': self.total_count_violations,
            'lost_ball_recoveries': self.lost_ball_recoveries,
            'duplicate_ball_merges': self.duplicate_ball_merges,
            'recent_violations': len(self.violation_history),
            'track_assignments': self.track_assignments.copy()
        }
    
    def reset(self) -> None:
        """
        Reset the ball count manager state.
        
        This clears all current counts and statistics while preserving
        the expected counts configuration.
        """
        # Reset current counts to zero
        self.current_counts = {
            ball_class: 0 for ball_class in self.expected_counts.keys()
        }
        
        # Clear track assignments
        self.track_assignments.clear()
        
        # Reset statistics
        self.total_count_violations = 0
        self.lost_ball_recoveries = 0
        self.duplicate_ball_merges = 0
        
        # Clear violation history
        self.violation_history.clear()
    
    def update_expected_counts(self, new_expected_counts: Dict[str, int]) -> None:
        """
        Update the expected ball counts (e.g., when balls are potted in snooker).
        
        Args:
            new_expected_counts: New expected counts dictionary
            
        Raises:
            ValueError: If new_expected_counts is invalid
        """
        # Validate new expected counts
        if not new_expected_counts:
            raise ValueError("Expected counts cannot be empty")
        
        for ball_class, count in new_expected_counts.items():
            if not isinstance(ball_class, str) or not ball_class.strip():
                raise ValueError(f"Ball class must be a non-empty string, got: {ball_class}")
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Expected count must be a non-negative integer, got: {count}")
        
        # Update expected counts
        self.expected_counts = new_expected_counts.copy()
        
        # Update current counts to include new ball classes
        for ball_class in new_expected_counts.keys():
            if ball_class not in self.current_counts:
                self.current_counts[ball_class] = 0
        
        # Remove ball classes that are no longer expected
        classes_to_remove = []
        for ball_class in self.current_counts.keys():
            if ball_class not in new_expected_counts:
                classes_to_remove.append(ball_class)
        
        for ball_class in classes_to_remove:
            del self.current_counts[ball_class]
    
    def __str__(self) -> str:
        """String representation of the ball count manager."""
        return (f"BallCountManager(expected={self.expected_counts}, current={self.current_counts}, "
                f"violations={self.total_count_violations}, total_tracks={len(self.track_assignments)})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the ball count manager."""
        return (f"BallCountManager(expected={self.expected_counts}, "
                f"current={self.current_counts}, "
                f"violations={self.total_count_violations})")