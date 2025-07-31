"""
MOLT (Multiple Object Local Tracker) Implementation

This module implements the MOLT algorithm for robust tracking of multiple small,
similar objects in low-quality, low-frame-rate video. MOLT uses a population-based
approach where each object is tracked by multiple local trackers that combine
appearance features with motion constraints.

The algorithm is specifically designed for scenarios like tracking multiple balls
in cue sports videos where traditional trackers struggle due to object similarity
and challenging video conditions.
"""

from typing import Dict, List, Optional, Any, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray
import cv2

# Import the Tracker interface from CVModelInference
from CVModelInference.tracker import Tracker, TrackerInfo, Detection, Track, Frame


class MOLTTrackerConfig:
    """Configuration schema for MOLT tracker with default parameters."""
    
    # Default population sizes for different ball types
    DEFAULT_POPULATION_SIZES = {
        'white': 1500,      # Larger population for fast-moving white ball
        'red': 300,         # Medium population for red balls
        'yellow': 200,      # Smaller population for colored balls
        'green': 200,
        'brown': 200,
        'blue': 200,
        'pink': 200,
        'black': 200
    }
    
    # Default exploration radii for different ball types
    DEFAULT_EXPLORATION_RADII = {
        'white': 30,        # Larger search radius for white ball
        'red': 20,          # Medium radius for red balls
        'default': 15       # Default radius for other balls
    }
    
    # Expected ball counts for snooker (configurable based on game state)
    DEFAULT_EXPECTED_BALL_COUNTS = {
        'white': 1,
        'red': 15,          # Configurable based on game state
        'yellow': 1,
        'green': 1,
        'brown': 1,
        'blue': 1,
        'pink': 1,
        'black': 1
    }
    
    # Default histogram configuration
    DEFAULT_HISTOGRAM_BINS = 16
    
    # Default similarity weights
    DEFAULT_SIMILARITY_WEIGHTS = {
        'histogram': 0.6,   # Weight for appearance similarity
        'spatial': 0.4      # Weight for spatial consistency
    }
    
    # Default diversity distribution for population generation
    DEFAULT_DIVERSITY_DISTRIBUTION = [0.5, 0.3, 0.2]  # Best, second, third tracker ratios
    
    # Default color space for histogram extraction
    DEFAULT_COLOR_SPACE = 'HSV'  # HSV or RGB
    
    # Default minimum confidence threshold
    DEFAULT_MIN_CONFIDENCE = 0.1
    
    # Default maximum frames to track without detection
    DEFAULT_MAX_FRAMES_WITHOUT_DETECTION = 30


class MOLTTracker(Tracker[Dict[str, Any]]):
    """
    MOLT (Multiple Object Local Tracker) implementation.
    
    This tracker uses a population-based approach where each object is tracked
    by multiple local trackers that combine appearance features (color histograms)
    with motion constraints to maintain robust tracking in challenging conditions.
    
    The tracker is specifically designed for tracking multiple small, similar
    objects like balls in cue sports videos.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the MOLT tracker with configurable parameters.
        
        Args:
            **kwargs: Configuration parameters including:
                - population_sizes: Dict mapping ball colors to population sizes
                - exploration_radii: Dict mapping ball colors to search radii
                - expected_ball_counts: Dict mapping ball colors to expected counts
                - histogram_bins: Number of bins for color histograms
                - similarity_weights: Weights for combining histogram and spatial similarities
                - diversity_distribution: Distribution ratios for population generation
                - color_space: Color space for histogram extraction ('HSV' or 'RGB')
                - min_confidence: Minimum confidence threshold for tracking
                - max_frames_without_detection: Maximum frames to track without detection
        """
        # Configuration parameters
        self.population_sizes = kwargs.get('population_sizes', MOLTTrackerConfig.DEFAULT_POPULATION_SIZES.copy())
        self.exploration_radii = kwargs.get('exploration_radii', MOLTTrackerConfig.DEFAULT_EXPLORATION_RADII.copy())
        self.expected_ball_counts = kwargs.get('expected_ball_counts', MOLTTrackerConfig.DEFAULT_EXPECTED_BALL_COUNTS.copy())
        self.histogram_bins = kwargs.get('histogram_bins', MOLTTrackerConfig.DEFAULT_HISTOGRAM_BINS)
        self.similarity_weights = kwargs.get('similarity_weights', MOLTTrackerConfig.DEFAULT_SIMILARITY_WEIGHTS.copy())
        self.diversity_distribution = kwargs.get('diversity_distribution', MOLTTrackerConfig.DEFAULT_DIVERSITY_DISTRIBUTION.copy())
        self.color_space = kwargs.get('color_space', MOLTTrackerConfig.DEFAULT_COLOR_SPACE)
        self.min_confidence = kwargs.get('min_confidence', MOLTTrackerConfig.DEFAULT_MIN_CONFIDENCE)
        self.max_frames_without_detection = kwargs.get('max_frames_without_detection', MOLTTrackerConfig.DEFAULT_MAX_FRAMES_WITHOUT_DETECTION)
        
        # Internal state
        self.populations: List['TrackerPopulation'] = []
        self.ball_counts: Dict[str, int] = {}
        self.frame_count: int = 0
        self.next_track_id: int = 1
        self.is_initialized: bool = False
        
        # Statistics for tracker info
        self.total_tracks_created: int = 0
        self.total_tracks_lost: int = 0
        self.total_fragmentations: int = 0
        
        # Initialize histogram extractor and ball count manager
        self.histogram_extractor = HistogramExtractor(
            num_bins=self.histogram_bins,
            color_space=self.color_space
        )
        self.ball_count_manager = BallCountManager(
            expected_counts=self.expected_ball_counts
        )
    
    def init(self, frame: Frame, detections: List[Detection]) -> bool:
        """
        Initialize the tracker with the first frame and initial detections.
        
        Args:
            frame: First frame as numpy array with shape (H, W, C) in BGR format
            detections: List of detection dictionaries containing object information
                        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Reset tracker state
            self.reset()
            
            # Validate inputs
            if frame is None or len(frame.shape) != 3:
                return False
            
            if not detections:
                return False
            
            # Store frame dimensions
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # Initialize populations for each detection (placeholder for now)
            # This will be implemented in later tasks
            self._init_populations(detections, frame)
            
            self.frame_count = 1
            self.is_initialized = True
            
            return True
            
        except Exception as e:
            print(f"Error initializing MOLT tracker: {e}")
            return False
    
    def update(self, frame: Frame, detections: Optional[List[Detection]] = None) -> List[Track]:
        """
        Update the tracker with a new frame and optional new detections.
        
        Args:
            frame: New frame as numpy array with shape (H, W, C) in BGR format
            detections: Optional list of new detections to incorporate
                        
        Returns:
            List[Track]: List of tracked objects with updated positions and IDs
        """
        if not self.is_initialized:
            return []
        
        try:
            self.frame_count += 1
            
            # Update populations (placeholder for now)
            # This will be implemented in later tasks
            self._update_populations(frame)
            
            # Verify ball counts (placeholder for now)
            # This will be implemented in later tasks
            self._verify_ball_counts()
            
            # Generate tracking results (placeholder for now)
            # This will be implemented in later tasks
            tracks = self._generate_tracking_results()
            
            return tracks
            
        except Exception as e:
            print(f"Error updating MOLT tracker: {e}")
            return []
    
    def reset(self) -> None:
        """
        Reset the tracker state.
        
        This method clears all internal state and prepares the tracker
        for a new sequence of frames.
        """
        self.populations.clear()
        self.ball_counts.clear()
        self.frame_count = 0
        self.next_track_id = 1
        self.is_initialized = False
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_fragmentations = 0
    
    def visualize(self, frame: Frame, tracks: Sequence[Track], 
                 output_path: Optional[str] = None) -> Frame:
        """
        Visualize the tracks on the input frame.
        
        Args:
            frame: Input frame as numpy array with shape (H, W, C) in BGR format
            tracks: Tracking results from update()
            output_path: Optional path to save the visualization
            
        Returns:
            Frame: A copy of the input frame with tracks visualized
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Placeholder visualization - will be implemented in later tasks
        # For now, just draw basic bounding boxes
        for track in tracks:
            x, y = int(track.get('x', 0)), int(track.get('y', 0))
            w, h = int(track.get('width', 0)), int(track.get('height', 0))
            track_id = track.get('id', 0)
            ball_class = track.get('class', 'unknown')
            
            # Calculate bounding box coordinates
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and class
            label = f"ID:{track_id} {ball_class}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_frame)
        
        return vis_frame
    
    def get_tracker_info(self) -> TrackerInfo:
        """
        Get information about the tracker.
        
        Returns:
            TrackerInfo: Dictionary containing tracker information
        """
        return TrackerInfo(
            name="MOLT",
            type="Multiple Object Local Tracker",
            parameters={
                'population_sizes': self.population_sizes,
                'exploration_radii': self.exploration_radii,
                'expected_ball_counts': self.expected_ball_counts,
                'histogram_bins': self.histogram_bins,
                'similarity_weights': self.similarity_weights,
                'diversity_distribution': self.diversity_distribution,
                'color_space': self.color_space,
                'min_confidence': self.min_confidence,
                'max_frames_without_detection': self.max_frames_without_detection
            },
            frame_count=self.frame_count,
            active_tracks=len([p for p in self.populations if p is not None]),  # Placeholder
            total_tracks_created=self.total_tracks_created,
            total_tracks_lost=self.total_tracks_lost,
            total_fragmentations=self.total_fragmentations
        )
    
    # Private helper methods (placeholders for future implementation)
    
    def _init_populations(self, detections: List[Detection], frame: Frame) -> None:
        """
        Initialize tracker populations from initial detections.
        
        Args:
            detections: List of initial detections
            frame: Initial frame
        """
        self.populations.clear()
        self.ball_counts.clear()
        
        # Initialize ball count manager (already initialized in constructor with expected counts)
        
        # Create populations for each detection
        for detection in detections:
            try:
                # Extract detection information
                center = (detection.get('x', 0), detection.get('y', 0))
                size = (detection.get('width', 20), detection.get('height', 20))
                object_class = detection.get('class', 'unknown')
                
                # Skip if detection has invalid dimensions
                if size[0] <= 0 or size[1] <= 0:
                    print(f"Warning: Skipping detection with invalid size: {size}")
                    continue
                
                # Extract initial histogram from detection patch
                reference_histogram = self._extract_initial_histogram(frame, center, size)
                
                if reference_histogram is None:
                    print(f"Warning: Failed to extract histogram for detection at {center}")
                    continue
                
                # Get population size for this object class
                population_size = self.population_sizes.get(object_class, 
                                                          self.population_sizes.get('red', 300))
                
                # Create tracker population
                population = TrackerPopulation(
                    object_id=self.next_track_id,
                    object_class=object_class,
                    population_size=population_size,
                    initial_center=center,
                    initial_size=size,
                    reference_histogram=reference_histogram
                )
                
                self.populations.append(population)
                
                # Update ball counts
                if object_class not in self.ball_counts:
                    self.ball_counts[object_class] = 0
                self.ball_counts[object_class] += 1
                
                # Increment track ID for next object
                self.next_track_id += 1
                self.total_tracks_created += 1
                
            except Exception as e:
                print(f"Error initializing population for detection {detection}: {e}")
                continue
        
        print(f"Initialized {len(self.populations)} tracker populations")
        print(f"Ball counts: {self.ball_counts}")
    
    def _extract_initial_histogram(self, frame: Frame, center: Tuple[float, float], 
                                  size: Tuple[float, float]) -> Optional[NDArray[np.float32]]:
        """
        Extract initial histogram from detection patch.
        
        Args:
            frame: Input frame
            center: (x, y) center position of detection
            size: (width, height) size of detection
            
        Returns:
            Optional[NDArray[np.float32]]: Extracted histogram or None if extraction fails
        """
        try:
            # Calculate patch boundaries
            half_width = int(size[0] / 2)
            half_height = int(size[1] / 2)
            
            frame_height, frame_width = frame.shape[:2]
            
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
            
            # Extract histogram using histogram extractor
            histogram = self.histogram_extractor.extract_histogram(patch)
            
            return histogram
            
        except Exception as e:
            print(f"Error extracting initial histogram: {e}")
            return None
    
    def _update_populations(self, frame: Frame) -> None:
        """
        Update all tracker populations for the current frame.
        
        Args:
            frame: Current frame
        """
        # Placeholder - will be implemented in later tasks
        pass
    
    def _verify_ball_counts(self) -> None:
        """
        Verify and correct ball count inconsistencies.
        """
        # Placeholder - will be implemented in later tasks
        pass
    
    def _generate_tracking_results(self) -> List[Track]:
        """
        Generate tracking results from current population states.
        
        Returns:
            List[Track]: List of tracking results
        """
        # Placeholder - will be implemented in later tasks
        return []


class TrackerPopulation:
    """
    Manages a population of local trackers for a single object.
    
    This class maintains a collection of LocalTracker instances that collectively
    track a single object. It handles population initialization, updates, ranking,
    and regeneration with diversity strategies.
    """
    
    def __init__(self, object_id: int, object_class: str, population_size: int,
                 initial_center: Tuple[float, float], initial_size: Tuple[float, float],
                 reference_histogram: NDArray[np.float32]) -> None:
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
               similarity_weights: Dict[str, float], reference_position: Optional[Tuple[float, float]] = None) -> 'LocalTracker':
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
            LocalTracker: The best tracker from this population
            
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
    
    def get_best_tracker(self) -> Optional['LocalTracker']:
        """
        Get the highest-weighted tracker from the population.
        
        Returns:
            Optional[LocalTracker]: The best tracker, or None if no trackers exist
        """
        return self.best_tracker
    
    def _extract_patch(self, frame: Frame, center: Tuple[float, float], 
                      size: Tuple[float, float]) -> Optional[NDArray[np.uint8]]:
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
        
        # Get top 3 trackers (or as many as available)
        top_trackers = self.trackers[:min(3, len(self.trackers))]
        
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
    
    def _generate_random_position(self, center: Tuple[float, float], 
                                 radius: float) -> Tuple[float, float]:
        """
        Generate a random position within the specified radius of the center.
        
        Uses uniform distribution within a circle to ensure even spatial distribution.
        
        Args:
            center: (x, y) center position
            radius: Maximum distance from center
            
        Returns:
            Tuple[float, float]: New random position (x, y)
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
                'population_size': 0,
                'best_weight': 0.0,
                'average_weight': 0.0,
                'weight_std': 0.0,
                'frame_count': self.frame_count,
                'total_updates': self.total_updates
            }
        
        weights = [t.total_weight for t in self.trackers]
        
        return {
            'population_size': len(self.trackers),
            'best_weight': max(weights) if weights else 0.0,
            'average_weight': np.mean(weights) if weights else 0.0,
            'weight_std': np.std(weights) if weights else 0.0,
            'frame_count': self.frame_count,
            'total_updates': self.total_updates,
            'best_weights_history': self.best_weights_history.copy()
        }
    
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


class LocalTracker:
    """
    Individual tracker within a population that maintains position and appearance information.
    
    Each LocalTracker represents a hypothesis about where an object might be located
    in the current frame. It stores position, size, appearance features (histogram),
    and computes similarity scores for tracking decisions.
    """
    
    def __init__(self, center: Tuple[float, float], size: Tuple[float, float], 
                 histogram: NDArray[np.float32], tracker_id: int) -> None:
        """
        Initialize a local tracker with position and appearance information.
        
        Args:
            center: (x, y) center position of the tracker
            size: (width, height) size of the tracked object
            histogram: Reference appearance histogram as 1D numpy array
            tracker_id: Unique identifier for this tracker within the population
        """
        # Position and size information
        self.center = center
        self.size = size
        self.tracker_id = tracker_id
        
        # Appearance model
        if histogram is None or histogram.size == 0:
            raise ValueError("Histogram cannot be None or empty")
        self.histogram = histogram.copy()
        
        # Similarity weights (will be updated by compute methods)
        self.hist_weight: float = 0.0
        self.dist_weight: float = 0.0
        self.total_weight: float = 0.0
        
        # Additional tracking information
        self.last_update_frame: int = 0
        self.confidence: float = 0.0
    
    def compute_similarity(self, patch: NDArray[np.uint8], 
                          histogram_extractor: 'HistogramExtractor',
                          method: str = 'bhattacharyya') -> float:
        """
        Calculate histogram similarity between tracker's reference and current patch.
        
        Args:
            patch: Current image patch as numpy array with shape (H, W, C) in BGR format
            histogram_extractor: HistogramExtractor instance for computing histograms
            method: Distance metric for histogram comparison
            
        Returns:
            float: Similarity score in [0, 1] range where 1.0 = identical appearance
            
        Raises:
            ValueError: If patch is invalid or histogram extraction fails
        """
        try:
            # Extract histogram from current patch
            current_histogram = histogram_extractor.extract_histogram(patch)
            
            # Compare with reference histogram
            distance = histogram_extractor.compare_histograms(
                self.histogram, current_histogram, method=method
            )
            
            # Convert distance to similarity score [0, 1]
            similarity = histogram_extractor.normalize_similarity(distance, method=method)
            
            return float(similarity)
            
        except Exception as e:
            # Return low similarity if histogram computation fails
            print(f"Warning: Failed to compute histogram similarity for tracker {self.tracker_id}: {e}")
            return 0.0
    
    def compute_distance(self, reference_pos: Tuple[float, float]) -> float:
        """
        Calculate spatial distance from tracker position to reference position.
        
        Args:
            reference_pos: (x, y) reference position to compare against
            
        Returns:
            float: Euclidean distance between tracker center and reference position
        """
        if reference_pos is None or len(reference_pos) != 2:
            raise ValueError("Reference position must be a tuple of (x, y) coordinates")
        
        # Calculate Euclidean distance
        dx = self.center[0] - reference_pos[0]
        dy = self.center[1] - reference_pos[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        return float(distance)
    
    def update_weights(self, hist_similarity: float, spatial_distance: float,
                      similarity_weights: Dict[str, float], 
                      max_distance: float = 100.0) -> None:
        """
        Update tracker weights by combining histogram and spatial similarities.
        
        Args:
            hist_similarity: Histogram similarity score in [0, 1] range
            spatial_distance: Spatial distance from reference position
            similarity_weights: Dict with 'histogram' and 'spatial' weight values
            max_distance: Maximum expected spatial distance for normalization
        """
        # Validate inputs
        if not (0.0 <= hist_similarity <= 1.0):
            raise ValueError(f"Histogram similarity must be in [0, 1], got {hist_similarity}")
        
        if spatial_distance < 0:
            raise ValueError(f"Spatial distance must be non-negative, got {spatial_distance}")
        
        if 'histogram' not in similarity_weights or 'spatial' not in similarity_weights:
            raise ValueError("similarity_weights must contain 'histogram' and 'spatial' keys")
        
        # Store individual weights
        self.hist_weight = float(hist_similarity)
        
        # Convert spatial distance to similarity (closer = higher similarity)
        # Use exponential decay: similarity = exp(-distance / max_distance)
        spatial_similarity = np.exp(-spatial_distance / max_distance)
        self.dist_weight = float(spatial_similarity)
        
        # Combine weights using configured ratios
        hist_weight_ratio = similarity_weights['histogram']
        spatial_weight_ratio = similarity_weights['spatial']
        
        # Normalize weight ratios to sum to 1.0
        total_ratio = hist_weight_ratio + spatial_weight_ratio
        if total_ratio <= 0:
            raise ValueError("Weight ratios must sum to a positive value")
        
        hist_weight_norm = hist_weight_ratio / total_ratio
        spatial_weight_norm = spatial_weight_ratio / total_ratio
        
        # Calculate combined total weight
        self.total_weight = (hist_weight_norm * self.hist_weight + 
                           spatial_weight_norm * self.dist_weight)
        
        # Update confidence based on total weight
        self.confidence = self.total_weight
    
    @property
    def total_weight_property(self) -> float:
        """
        Get the total weight for tracker ranking.
        
        Returns:
            float: Combined weight score for ranking trackers
        """
        return self.total_weight
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get the current tracker position.
        
        Returns:
            Tuple[float, float]: (x, y) center position
        """
        return self.center
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box coordinates.
        
        Returns:
            Tuple[float, float, float, float]: (x, y, width, height) where (x, y) is center
        """
        return (self.center[0], self.center[1], self.size[0], self.size[1])
    
    def __str__(self) -> str:
        """String representation of the tracker."""
        return (f"LocalTracker(id={self.tracker_id}, center={self.center}, "
                f"size={self.size}, total_weight={self.total_weight:.3f})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the tracker."""
        return (f"LocalTracker(id={self.tracker_id}, center={self.center}, "
                f"size={self.size}, hist_weight={self.hist_weight:.3f}, "
                f"dist_weight={self.dist_weight:.3f}, total_weight={self.total_weight:.3f})")


class HistogramExtractor:
    """
    Handles appearance feature extraction and comparison using color histograms.
    
    This class provides methods for extracting color histograms from image patches
    and comparing them using various distance metrics for object tracking.
    """
    
    def __init__(self, num_bins: int = 16, color_space: str = 'HSV') -> None:
        """
        Initialize the histogram extractor.
        
        Args:
            num_bins: Number of bins for histogram computation
            color_space: Color space for histogram extraction ('HSV' or 'RGB')
        """
        self.num_bins = num_bins
        self.color_space = color_space.upper()
        
        if self.color_space not in ['HSV', 'RGB']:
            raise ValueError(f"Unsupported color space: {color_space}. Use 'HSV' or 'RGB'.")
        
        # Set up histogram parameters based on color space
        if self.color_space == 'HSV':
            # HSV ranges: H[0,180], S[0,255], V[0,255] in OpenCV
            self.ranges = [0, 180, 0, 256, 0, 256]
            self.channels = [0, 1, 2]  # Use all three channels
        else:  # RGB
            # RGB ranges: R[0,255], G[0,255], B[0,255]
            self.ranges = [0, 256, 0, 256, 0, 256]
            self.channels = [0, 1, 2]  # Use all three channels
        
        # Histogram size for each channel
        self.hist_size = [num_bins, num_bins, num_bins]
    
    def extract_histogram(self, patch: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Extract color histogram from an image patch.
        
        Args:
            patch: Image patch as numpy array with shape (H, W, C) in BGR format
            
        Returns:
            NDArray[np.float32]: Normalized color histogram as 1D array
            
        Raises:
            ValueError: If patch is invalid or empty
        """
        if patch is None or patch.size == 0:
            raise ValueError("Patch cannot be None or empty")
        
        if len(patch.shape) != 3 or patch.shape[2] != 3:
            raise ValueError("Patch must be a 3-channel color image")
        
        # Convert color space if needed
        if self.color_space == 'HSV':
            # Convert from BGR to HSV
            converted_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        else:  # RGB
            # Convert from BGR to RGB
            converted_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        
        # Calculate 3D histogram
        hist = cv2.calcHist(
            [converted_patch],
            self.channels,
            None,  # No mask
            self.hist_size,
            self.ranges
        )
        
        # Normalize histogram
        normalized_hist = self.normalize_histogram(hist.astype(np.float32))
        
        # Flatten to 1D array for easier handling
        return normalized_hist.flatten().astype(np.float32)
    
    def normalize_histogram(self, hist: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize histogram to unit sum.
        
        Args:
            hist: Input histogram as numpy array
            
        Returns:
            NDArray[np.float32]: Normalized histogram with sum = 1.0
        """
        if hist is None or hist.size == 0:
            raise ValueError("Histogram cannot be None or empty")
        
        # Calculate sum and avoid division by zero
        hist_sum = np.sum(hist)
        if hist_sum == 0:
            # Return uniform distribution if histogram is empty
            return np.ones_like(hist, dtype=np.float32) / hist.size
        
        # Normalize to unit sum
        return (hist / hist_sum).astype(np.float32)
    
    def compare_histograms(self, hist1: NDArray[np.float32], hist2: NDArray[np.float32], 
                          method: str = 'bhattacharyya') -> float:
        """
        Compare two histograms using the specified distance metric.
        
        Args:
            hist1: First histogram as 1D numpy array
            hist2: Second histogram as 1D numpy array
            method: Distance metric ('bhattacharyya', 'intersection', 'chi_square')
            
        Returns:
            float: Distance between histograms (lower values indicate higher similarity)
            
        Raises:
            ValueError: If histograms have different shapes or invalid method
        """
        if hist1 is None or hist2 is None:
            raise ValueError("Histograms cannot be None")
        
        if hist1.shape != hist2.shape:
            raise ValueError("Histograms must have the same shape")
        
        method = method.lower()
        
        if method == 'bhattacharyya':
            return self._bhattacharyya_distance(hist1, hist2)
        elif method == 'intersection':
            return self._intersection_distance(hist1, hist2)
        elif method == 'chi_square':
            return self._chi_square_distance(hist1, hist2)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'bhattacharyya', 'intersection', or 'chi_square'.")
    
    def normalize_similarity(self, distance: float, method: str = 'bhattacharyya') -> float:
        """
        Map distance to similarity score in [0,1] range.
        
        Args:
            distance: Distance value from compare_histograms
            method: Distance metric used ('bhattacharyya', 'intersection', 'chi_square')
            
        Returns:
            float: Similarity score where 1.0 = identical, 0.0 = completely different
        """
        method = method.lower()
        
        if method == 'bhattacharyya':
            # Bhattacharyya distance is in [0, inf), convert to similarity [0, 1]
            # Use exponential decay: similarity = exp(-distance)
            return float(np.exp(-distance))
        elif method == 'intersection':
            # Intersection distance is in [0, 1], where 0 = identical
            # Convert to similarity: similarity = 1 - distance
            return float(1.0 - distance)
        elif method == 'chi_square':
            # Chi-square distance is in [0, inf), convert to similarity [0, 1]
            # Use exponential decay with scaling factor
            return float(np.exp(-distance / 2.0))
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _bhattacharyya_distance(self, hist1: NDArray[np.float32], hist2: NDArray[np.float32]) -> float:
        """
        Calculate Bhattacharyya distance between two histograms.
        
        Args:
            hist1: First normalized histogram
            hist2: Second normalized histogram
            
        Returns:
            float: Bhattacharyya distance
        """
        # Ensure histograms are normalized
        hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
        
        # Calculate Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1_norm * hist2_norm))
        
        # Clamp to avoid numerical issues
        bc = np.clip(bc, 1e-10, 1.0)
        
        # Calculate Bhattacharyya distance
        distance = -np.log(bc)
        
        return float(distance)
    
    def _intersection_distance(self, hist1: NDArray[np.float32], hist2: NDArray[np.float32]) -> float:
        """
        Calculate histogram intersection distance between two histograms.
        
        Args:
            hist1: First normalized histogram
            hist2: Second normalized histogram
            
        Returns:
            float: Intersection distance (1 - intersection)
        """
        # Ensure histograms are normalized
        hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
        
        # Calculate intersection (sum of minimum values)
        intersection = np.sum(np.minimum(hist1_norm, hist2_norm))
        
        # Convert to distance (1 - intersection)
        distance = 1.0 - intersection
        
        return float(distance)
    
    def _chi_square_distance(self, hist1: NDArray[np.float32], hist2: NDArray[np.float32]) -> float:
        """
        Calculate Chi-square distance between two histograms.
        
        Args:
            hist1: First normalized histogram
            hist2: Second normalized histogram
            
        Returns:
            float: Chi-square distance
        """
        # Ensure histograms are normalized
        hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
        
        # Calculate Chi-square distance
        # chi2 = 0.5 * sum((h1 - h2)^2 / (h1 + h2))
        denominator = hist1_norm + hist2_norm + 1e-10
        numerator = (hist1_norm - hist2_norm) ** 2
        
        distance = 0.5 * np.sum(numerator / denominator)
        
        return float(distance)


class BallCountManager:
    """
    Manages ball counting and verification logic specific to snooker.
    
    This class tracks the current number of balls of each type and compares
    them against expected counts. It provides methods for handling count
    violations such as lost balls or duplicate detections.
    """
    
    def __init__(self, expected_counts: Dict[str, int]) -> None:
        """
        Initialize the BallCountManager with expected ball counts.
        
        Args:
            expected_counts: Dictionary mapping ball colors to expected counts
                           e.g., {'red': 15, 'white': 1, 'yellow': 1, ...}
        
        Raises:
            ValueError: If expected_counts is invalid
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
        Check if current counts match expected counts.
        
        Returns:
            bool: True if all counts are within expected ranges, False otherwise
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
    
    def get_count_violations(self) -> Dict[str, Dict[str, int]]:
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
        
        # Sort by distance and return closest pairs
        distances.sort(key=lambda x: x[0])
        
        merge_pairs = []
        used_tracks = set()
        
        for distance, track_id1, track_id2 in distances:
            if len(merge_pairs) >= num_merges:
                break
            
            # Only merge if neither track is already involved in a merge
            if track_id1 not in used_tracks and track_id2 not in used_tracks:
                merge_pairs.append((track_id1, track_id2))
                used_tracks.add(track_id1)
                used_tracks.add(track_id2)
        
        return merge_pairs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ball count management.
        
        Returns:
            Dict containing various statistics about count violations and corrections
        """
        return {
            'expected_counts': self.expected_counts.copy(),
            'current_counts': self.current_counts.copy(),
            'track_assignments': self.track_assignments.copy(),
            'total_count_violations': self.total_count_violations,
            'lost_ball_recoveries': self.lost_ball_recoveries,
            'duplicate_ball_merges': self.duplicate_ball_merges,
            'recent_violations': self.violation_history[-10:] if self.violation_history else []
        }
    
    def reset(self) -> None:
        """
        Reset all counts and statistics.
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
        self.violation_history.clear()
    
    def update_expected_counts(self, new_expected_counts: Dict[str, int]) -> None:
        """
        Update the expected ball counts (useful for different game states).
        
        Args:
            new_expected_counts: New expected counts dictionary
            
        Raises:
            ValueError: If new_expected_counts is invalid
        """
        if not new_expected_counts:
            raise ValueError("New expected counts cannot be empty")
        
        # Validate new expected counts
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
        
        # Remove counts for ball classes that are no longer expected
        self.current_counts = {
            ball_class: count for ball_class, count in self.current_counts.items()
            if ball_class in self.expected_counts
        }
    
    def __str__(self) -> str:
        """String representation of the ball count manager."""
        violations = sum(1 for ball_class in self.expected_counts.keys()
                        if self.current_counts.get(ball_class, 0) != self.expected_counts[ball_class])
        return f"BallCountManager(violations={violations}, total_tracks={len(self.track_assignments)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the ball count manager."""
        return (f"BallCountManager(expected={self.expected_counts}, "
                f"current={self.current_counts}, "
                f"violations={self.total_count_violations})")