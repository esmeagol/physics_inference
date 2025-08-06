"""
Tracker population management for MOLT algorithm.

This module provides the TrackerPopulation class which manages a collection
of local trackers for a single object.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .types import Frame, Position, Size, Histogram
from .local_tracker import LocalTracker

if TYPE_CHECKING:
    from .histogram_extractor import HistogramExtractor


class TrackerPopulation:
    """
    Manages a population of local trackers for a single object.
    
    This class maintains a collection of LocalTracker instances that collectively
    track a single object. It handles population initialization, updates, ranking,
    and regeneration with diversity strategies.
    """
    
    def __init__(self, object_id: int, object_class: str, population_size: int,
                 initial_center: Position, initial_size: Size,
                 reference_histogram: Histogram) -> None:
        """
        Initialize a tracker population for a single object.
        
        Args:
            object_id: Unique identifier for the tracked object
            object_class: Class/type of the object (e.g., 'red', 'white', 'yellow')
            population_size: Number of local trackers in the population
            initial_center: (x, y) initial center position of the object
            initial_size: (width, height) initial size of the object
            reference_histogram: Reference appearance histogram for the object
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if population_size <= 0:
            raise ValueError(f"Population size must be positive, got {population_size}")
        
        if reference_histogram is None or reference_histogram.size == 0:
            raise ValueError("Reference histogram cannot be None or empty")
        
        if len(initial_center) != 2 or len(initial_size) != 2:
            raise ValueError("Initial center and size must be tuples of length 2")
        
        if initial_size[0] <= 0 or initial_size[1] <= 0:
            raise ValueError("Initial size dimensions must be positive")
        
        if not isinstance(object_class, str) or not object_class.strip():
            raise ValueError("Object class must be a non-empty string")
        
        # Store object information
        self.object_id = object_id
        self.object_class = object_class
        self.population_size = population_size
        self.initial_center = initial_center
        self.initial_size = initial_size
        
        # Store reference appearance model
        self.reference_histogram = reference_histogram.copy()
        
        # Initialize tracker list
        self.trackers: List[LocalTracker] = []
        
        # Best tracker from current frame
        self.best_tracker: Optional[LocalTracker] = None
        
        # Population statistics
        self.frame_count = 0
        self.total_updates = 0
        self.best_weights_history: List[float] = []
        
        # Initialize the population with trackers around the initial position
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """
        Initialize the population with trackers around the initial position.
        
        Creates the initial set of local trackers distributed around the
        initial object position.
        """
        self.trackers.clear()
        
        # Create trackers with small random offsets around initial position
        for i in range(self.population_size):
            # Add small random offset to initial position
            offset_x = np.random.normal(0, 5.0)  # Small standard deviation
            offset_y = np.random.normal(0, 5.0)
            
            tracker_center = (
                self.initial_center[0] + offset_x,
                self.initial_center[1] + offset_y
            )
            
            # Create local tracker
            tracker = LocalTracker(
                center=tracker_center,
                size=self.initial_size,
                histogram=self.reference_histogram,
                tracker_id=i
            )
            
            self.trackers.append(tracker)
        
        # Set initial best tracker (first one)
        if self.trackers:
            self.best_tracker = self.trackers[0]
    
    def update(self, frame: Frame, histogram_extractor: 'HistogramExtractor',
               similarity_weights: Dict[str, float], reference_position: Optional[Position] = None) -> Optional[LocalTracker]:
        """
        Update all trackers in the population for the current frame.
        
        This method processes all local trackers, computes their similarities,
        ranks them by total weight, and returns the best tracker.
        
        Args:
            frame: Current frame as numpy array with shape (H, W, C) in BGR format
            histogram_extractor: HistogramExtractor instance for computing histograms
            similarity_weights: Dict with 'histogram' and 'spatial' weight values
            reference_position: Optional reference position for spatial distance computation
            
        Returns:
            Optional[LocalTracker]: The best tracker from this population, or None if no trackers
            
        Raises:
            ValueError: If frame is invalid or no trackers exist
        """
        if frame is None or len(frame.shape) != 3:
            raise ValueError("Frame must be a 3-channel image")
        
        if not self.trackers:
            raise ValueError("No trackers in population")
        
        self.frame_count += 1
        self.total_updates += 1
        
        # Use best tracker position as reference if not provided
        if reference_position is None and self.best_tracker is not None:
            reference_position = self.best_tracker.center
        elif reference_position is None:
            reference_position = self.initial_center
        
        frame_height, frame_width = frame.shape[:2]
        
        # Update each tracker in the population
        for tracker in self.trackers:
            try:
                # Extract patch around tracker position
                patch = self._extract_patch(frame, tracker.center, tracker.size)
                
                if patch is not None and patch.size > 0:
                    # Compute histogram similarity
                    hist_similarity = tracker.compute_similarity(patch, histogram_extractor)
                    
                    # Compute spatial distance
                    spatial_distance = tracker.compute_distance(reference_position)
                    
                    # Update tracker weights
                    tracker.update_weights(hist_similarity, spatial_distance, similarity_weights)
                else:
                    # If patch extraction fails, set low weights
                    tracker.update_weights(0.0, float('inf'), similarity_weights)
                    
            except Exception as e:
                # Handle individual tracker update failures gracefully
                print(f"Warning: Failed to update tracker {tracker.tracker_id} in population {self.object_id}: {e}")
                tracker.update_weights(0.0, float('inf'), similarity_weights)
        
        # Sort trackers by total weight in descending order
        self.trackers.sort(key=lambda t: t.total_weight, reverse=True)
        
        # Update best tracker
        self.best_tracker = self.trackers[0] if self.trackers else None
        
        # Store best weight for statistics
        if self.best_tracker:
            self.best_weights_history.append(self.best_tracker.total_weight)
            # Keep only recent history (last 100 frames)
            if len(self.best_weights_history) > 100:
                self.best_weights_history.pop(0)
        
        return self.best_tracker
    
    def get_best_tracker(self) -> Optional[LocalTracker]:
        """
        Get the highest-weighted tracker from the population.
        
        Returns:
            Optional[LocalTracker]: The best tracker, or None if no trackers exist
        """
        return self.best_tracker
    
    def get_top_trackers(self, n: int = 3) -> List[LocalTracker]:
        """
        Get the top N trackers from the population.
        
        Args:
            n: Number of top trackers to return
            
        Returns:
            List[LocalTracker]: List of top N trackers sorted by weight (descending)
        """
        if not self.trackers:
            return []
        
        # Ensure trackers are sorted by weight
        sorted_trackers = sorted(self.trackers, key=lambda t: t.total_weight, reverse=True)
        return sorted_trackers[:min(n, len(sorted_trackers))]
    
    def _extract_patch(self, frame: Frame, center: Position, 
                      size: Size) -> Optional[NDArray[np.uint8]]:
        """
        Extract an image patch around the specified center position.
        
        Args:
            frame: Input frame as numpy array
            center: (x, y) center position for patch extraction
            size: (width, height) size of the patch
            
        Returns:
            Optional[NDArray[np.uint8]]: Extracted patch or None if extraction fails
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate patch boundaries
            half_width = int(size[0] / 2)
            half_height = int(size[1] / 2)
            
            x1 = max(0, int(center[0] - half_width))
            y1 = max(0, int(center[1] - half_height))
            x2 = min(frame_width, int(center[0] + half_width))
            y2 = min(frame_height, int(center[1] + half_height))
            
            # Check if patch has valid dimensions
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract patch
            patch = frame[y1:y2, x1:x2]
            
            # Ensure patch has minimum size
            if patch.shape[0] < 5 or patch.shape[1] < 5:
                return None
            
            return patch
            
        except Exception as e:
            print(f"Warning: Failed to extract patch at {center}: {e}")
            return None
    
    def generate_new_population(self, exploration_radius: float, 
                               diversity_distribution: List[float] = [0.5, 0.3, 0.2]) -> None:
        """
        Generate a new population of trackers with diversity strategy.
        
        This method creates a new population around the best trackers using
        a configurable distribution strategy (default: 50% around best,
        30% around second best, 20% around third best).
        
        Args:
            exploration_radius: Maximum radius for scattering new trackers
            diversity_distribution: List of ratios for distributing trackers
                                  around top performers [best, second, third]
        
        Raises:
            ValueError: If diversity distribution is invalid
        """
        if not self.trackers:
            raise ValueError("Cannot regenerate population: no existing trackers")
        
        if len(diversity_distribution) != 3:
            raise ValueError("Diversity distribution must have exactly 3 values")
        
        if abs(sum(diversity_distribution) - 1.0) > 1e-6:
            raise ValueError("Diversity distribution must sum to 1.0")
        
        if any(ratio < 0 for ratio in diversity_distribution):
            raise ValueError("All diversity distribution values must be non-negative")
        
        if exploration_radius <= 0:
            raise ValueError("Exploration radius must be positive")
        
        # Get top 3 trackers (or as many as available)
        top_trackers = self.get_top_trackers(3)
        
        # Calculate number of trackers for each group
        group_sizes = []
        remaining_size = self.population_size
        
        for i, ratio in enumerate(diversity_distribution):
            if i == len(diversity_distribution) - 1:
                # Last group gets remaining trackers
                group_sizes.append(remaining_size)
            else:
                group_size = int(self.population_size * ratio)
                group_sizes.append(group_size)
                remaining_size -= group_size
        
        # Generate new population
        new_trackers = []
        tracker_id = 0
        
        for group_idx, group_size in enumerate(group_sizes):
            if group_idx >= len(top_trackers):
                # If we don't have enough top trackers, use the best available
                reference_tracker = top_trackers[-1]
            else:
                reference_tracker = top_trackers[group_idx]
            
            # Generate trackers around this reference tracker
            for _ in range(group_size):
                new_center = self._generate_random_position(
                    reference_tracker.center, exploration_radius
                )
                
                new_tracker = LocalTracker(
                    center=new_center,
                    size=reference_tracker.size,
                    histogram=self.reference_histogram,
                    tracker_id=tracker_id
                )
                
                new_trackers.append(new_tracker)
                tracker_id += 1
        
        # Replace old population with new one
        self.trackers = new_trackers
        
        # Update best tracker reference (will be updated in next update() call)
        if self.trackers:
            self.best_tracker = self.trackers[0]
    
    def _generate_random_position(self, center: Position, 
                                 radius: float) -> Position:
        """
        Generate a random position within the specified radius of the center.
        
        Uses uniform distribution within a circle to ensure even spatial distribution.
        
        Args:
            center: (x, y) center position
            radius: Maximum distance from center
            
        Returns:
            Position: New random position (x, y)
        """
        # Generate random angle and distance
        angle = np.random.uniform(0, 2 * np.pi)
        # Use sqrt for uniform distribution within circle
        distance = radius * np.sqrt(np.random.uniform(0, 1))
        
        # Calculate new position
        new_x = center[0] + distance * np.cos(angle)
        new_y = center[1] + distance * np.sin(angle)
        
        return (float(new_x), float(new_y))
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current population.
        
        Returns:
            Dict[str, Any]: Dictionary containing population statistics
        """
        if not self.trackers:
            return {
                'object_id': self.object_id,
                'object_class': self.object_class,
                'population_size': 0,
                'best_weight': 0.0,
                'average_weight': 0.0,
                'weight_std': 0.0,
                'frame_count': self.frame_count,
                'total_updates': self.total_updates,
                'best_weights_history': []
            }
        
        weights = [t.total_weight for t in self.trackers]
        
        return {
            'object_id': self.object_id,
            'object_class': self.object_class,
            'population_size': len(self.trackers),
            'best_weight': max(weights) if weights else 0.0,
            'average_weight': np.mean(weights) if weights else 0.0,
            'weight_std': np.std(weights) if weights else 0.0,
            'frame_count': self.frame_count,
            'total_updates': self.total_updates,
            'best_weights_history': self.best_weights_history.copy()
        }
    
    def reset(self) -> None:
        """
        Reset the population state while preserving configuration.
        
        This clears all trackers and statistics but keeps the object
        information and reference histogram.
        """
        self.trackers.clear()
        self.best_tracker = None
        self.frame_count = 0
        self.total_updates = 0
        self.best_weights_history.clear()
        
        # Reinitialize the population
        self._initialize_population()
    
    def is_valid(self) -> bool:
        """
        Check if the population is in a valid state.
        
        Returns:
            bool: True if population is valid, False otherwise
        """
        try:
            # Check basic parameters
            if self.population_size <= 0:
                return False
            
            if self.reference_histogram is None or self.reference_histogram.size == 0:
                return False
            
            # Check trackers
            if len(self.trackers) != self.population_size:
                return False
            
            # Check if all trackers are valid
            for tracker in self.trackers:
                if not tracker.is_valid():
                    return False
            
            return True
            
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the population."""
        best_weight = self.best_tracker.total_weight if self.best_tracker else 0.0
        return (f"TrackerPopulation(id={self.object_id}, class={self.object_class}, "
                f"size={len(self.trackers)}, best_weight={best_weight:.3f})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the population."""
        return (f"TrackerPopulation(object_id={self.object_id}, "
                f"object_class='{self.object_class}', "
                f"population_size={self.population_size}, "
                f"current_size={len(self.trackers)}, "
                f"frame_count={self.frame_count})")