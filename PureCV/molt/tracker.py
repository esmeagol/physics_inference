"""
Main MOLT tracker implementation.

This module provides the MOLTTracker class which implements the MOLT
(Multiple Object Local Tracker) algorithm for robust tracking of multiple
small, similar objects in challenging video conditions.
"""

from typing import Dict, List, Optional, Any, Sequence
import numpy as np
import cv2

# Import the Tracker interface from CVModelInference
from CVModelInference.tracker import Tracker, TrackerInfo, Detection, Track, Frame

from .config import MOLTTrackerConfig
from .types import Position, Size
from .population import TrackerPopulation
from .histogram_extractor import HistogramExtractor
from .ball_count_manager import BallCountManager


class MOLTTracker(Tracker[Dict[str, Any]]):
    """
    MOLT (Multiple Object Local Tracker) implementation.
    
    This tracker uses a population-based approach where each object is tracked
    by multiple local trackers that combine appearance features (color histograms)
    with motion constraints to maintain robust tracking in challenging conditions.
    
    The tracker is specifically designed for tracking multiple small, similar
    objects like balls in cue sports videos.
    """
    
    def __init__(self, config: Optional[MOLTTrackerConfig] = None, **kwargs: Any) -> None:
        """
        Initialize the MOLT tracker with configurable parameters.
        
        Args:
            config: MOLTTrackerConfig instance with tracker parameters
            **kwargs: Additional configuration parameters that override config values
        """
        # Use provided config or create default
        if config is None:
            config = MOLTTrackerConfig.create_default()
        
        # Override config values with any provided kwargs
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        
        # Store configuration parameters
        self.population_sizes = config_dict['population_sizes']
        self.exploration_radii = config_dict['exploration_radii']
        self.expected_ball_counts = config_dict['expected_ball_counts']
        self.histogram_bins = config_dict['histogram_bins']
        self.similarity_weights = config_dict['similarity_weights']
        self.diversity_distribution = config_dict['diversity_distribution']
        self.color_space = config_dict['color_space']
        self.min_confidence = config_dict['min_confidence']
        self.max_frames_without_detection = config_dict['max_frames_without_detection']
        
        # Internal state
        self.populations: List[TrackerPopulation] = []
        self.ball_counts: Dict[str, int] = {}
        self.frame_count: int = 0
        self.next_track_id: int = 1
        self.is_initialized: bool = False
        
        # Frame dimensions (set during initialization)
        self.frame_width: int = 0
        self.frame_height: int = 0
        
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
            
            # Initialize populations for each detection
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
        self.frame_width = 0
        self.frame_height = 0
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_fragmentations = 0
        
        # Reset components
        if hasattr(self, 'ball_count_manager'):
            self.ball_count_manager.reset()
    
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
        
        # Define colors for different ball classes
        colors = {
            'white': (255, 255, 255),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'brown': (42, 42, 165),
            'blue': (255, 0, 0),
            'pink': (203, 192, 255),
            'black': (0, 0, 0),
            'unknown': (128, 128, 128)
        }
        
        # Draw tracks
        for track in tracks:
            x, y = int(track.get('x', 0)), int(track.get('y', 0))
            w, h = int(track.get('width', 0)), int(track.get('height', 0))
            track_id = track.get('id', 0)
            ball_class = track.get('class', 'unknown')
            confidence = track.get('confidence', 0.0)
            
            # Calculate bounding box coordinates
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            
            # Get color for this ball class
            color = colors.get(ball_class, colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(vis_frame, (x, y), 3, color, -1)
            
            # Draw track ID and class
            label = f"ID:{track_id} {ball_class} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            text_color = (0, 0, 0) if ball_class == 'white' else (255, 255, 255)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Draw frame information
        info_text = f"Frame: {self.frame_count}, Tracks: {len(tracks)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
            active_tracks=len([p for p in self.populations if p is not None]),
            total_tracks_created=self.total_tracks_created,
            total_tracks_lost=self.total_tracks_lost,
            total_fragmentations=self.total_fragmentations
        )
    
    # Private helper methods
    
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
    
    def _extract_initial_histogram(self, frame: Frame, center: Position, 
                                  size: Size) -> Optional[np.ndarray]:
        """
        Extract initial histogram from detection patch.
        
        Args:
            frame: Input frame
            center: (x, y) center position of detection
            size: (width, height) size of detection
            
        Returns:
            Optional[np.ndarray]: Extracted histogram or None if extraction fails
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
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all populations.
        
        Returns:
            Dict[str, Any]: Dictionary containing population statistics
        """
        stats: Dict[str, Any] = {
            'total_populations': len(self.populations),
            'ball_counts': self.ball_counts.copy(),
            'populations': []
        }
        
        for population in self.populations:
            pop_stats = population.get_population_statistics()
            stats['populations'].append(pop_stats)
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current tracker configuration.
        
        Returns:
            Dict[str, Any]: Configuration parameters
        """
        return {
            'population_sizes': self.population_sizes,
            'exploration_radii': self.exploration_radii,
            'expected_ball_counts': self.expected_ball_counts,
            'histogram_bins': self.histogram_bins,
            'similarity_weights': self.similarity_weights,
            'diversity_distribution': self.diversity_distribution,
            'color_space': self.color_space,
            'min_confidence': self.min_confidence,
            'max_frames_without_detection': self.max_frames_without_detection
        }