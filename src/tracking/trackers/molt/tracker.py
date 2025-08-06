"""
Main MOLT tracker implementation.

This module provides the MOLTTracker class which implements the MOLT
(Multiple Object Local Tracker) algorithm for robust tracking of multiple
small, similar objects in challenging video conditions.
"""

from typing import Dict, List, Optional, Any, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2

# Import the Tracker interface from CVModelInference
from src.tracking.tracker import Tracker, TrackerInfo, Detection, Track, Frame

from .config import MOLTTrackerConfig
from .types import Position, Size, Histogram
from .population import TrackerPopulation
from .histogram_extractor import HistogramExtractor
from .ball_count_manager import BallCountManager


from typing import TypeVar, Dict, Any, Optional, List, Sequence

# Type variable for the tracker configuration
ConfigType = TypeVar('ConfigType', bound=Dict[str, Any])

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
        
        The MOLT tracker uses a population-based approach where each object is tracked
        by multiple local trackers. This constructor sets up the tracker with either
        a provided configuration or default parameters, with optional parameter overrides.
        
        Args:
            config (MOLTTrackerConfig, optional): Configuration object containing all
                tracker parameters. If None, uses default configuration optimized for snooker.
            **kwargs: Additional configuration parameters that override config values.
                Common parameters include:
                - population_sizes (Dict[str, int]): Number of trackers per ball type
                - exploration_radii (Dict[str, int]): Search radius per ball type
                - histogram_bins (int): Number of histogram bins (8-32 typical)
                - color_space (str): Color space for histograms ('HSV', 'RGB', 'LAB')
                - similarity_weights (Dict[str, float]): Weights for histogram/spatial similarity
                - min_confidence (float): Minimum confidence threshold [0, 1]
                - expected_ball_counts (Dict[str, int]): Expected number of balls per type
        
        Example:
            >>> # Default configuration
            >>> tracker = MOLTTracker()
            
            >>> # Custom configuration
            >>> config = MOLTTrackerConfig.create_for_snooker()
            >>> tracker = MOLTTracker(config=config)
            
            >>> # Parameter overrides
            >>> tracker = MOLTTracker(
            ...     histogram_bins=16,
            ...     color_space='HSV',
            ...     population_sizes={'red': 300, 'white': 1500}
            ... )
        
        Raises:
            ValueError: If configuration parameters are invalid
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
        
        # Trail tracking for visualization
        self.trails: Dict[int, List[Tuple[float, float]]] = {}
        self.max_trail_length: int = 30  # Keep last 30 positions
        
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
        
        This method sets up the tracker for a new video sequence by creating
        tracker populations for each detected object. Each population consists
        of multiple local trackers distributed around the initial object position.
        
        The initialization process:
        1. Validates input frame and detections
        2. Extracts appearance histograms from detection patches
        3. Creates tracker populations with configured sizes
        4. Sets up ball counting with expected counts
        5. Initializes tracking statistics and trails
        
        Args:
            frame (NDArray[np.uint8]): First frame as numpy array with shape (H, W, C) 
                in BGR format. Must be a valid 3-channel color image.
            detections (List[Dict]): List of detection dictionaries. Each detection
                must contain:
                - 'x' (float): Center x coordinate
                - 'y' (float): Center y coordinate  
                - 'width' (float): Bounding box width (must be > 0)
                - 'height' (float): Bounding box height (must be > 0)
                - 'class' (str): Ball color/type ('red', 'white', 'yellow', etc.)
                - 'confidence' (float, optional): Detection confidence [0, 1]
                
        Returns:
            bool: True if initialization was successful, False otherwise.
                Initialization can fail due to:
                - Invalid frame (None, wrong shape, or empty)
                - Empty or invalid detections
                - Histogram extraction failures
                - Configuration errors
        
        Example:
            >>> detections = [
            ...     {'x': 100, 'y': 150, 'width': 30, 'height': 30, 'class': 'red'},
            ...     {'x': 200, 'y': 250, 'width': 30, 'height': 30, 'class': 'white'}
            ... ]
            >>> success = tracker.init(first_frame, detections)
            >>> if success:
            ...     print(f"Initialized {len(tracker.populations)} populations")
            ... else:
            ...     print("Initialization failed")
        
        Note:
            - The tracker must be initialized before calling update()
            - Call reset() before re-initializing with a new video sequence
            - Use high-quality detections from the first few frames for best results
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
        
        This is the main tracking method that processes each new frame in the video
        sequence. It updates all tracker populations, verifies ball count consistency,
        and returns the current tracking results.
        
        The update process:
        1. Validates the input frame
        2. Updates all tracker populations with the current frame
        3. Ranks trackers within each population by combined similarity scores
        4. Selects the best tracker from each population as the object position
        5. Generates new tracker populations around the best positions
        6. Verifies ball counts and handles violations
        7. Updates trajectory trails for visualization
        8. Returns formatted tracking results
        
        Args:
            frame (NDArray[np.uint8]): New frame as numpy array with shape (H, W, C) 
                in BGR format. Must be a valid 3-channel color image.
            detections (List[Dict], optional): Optional list of new detections to 
                incorporate. Currently not used but reserved for future enhancement.
                When implemented, will help recover lost tracks and improve accuracy.
                
        Returns:
            List[Dict]: List of tracking results, one per tracked object. Each result
                contains:
                - 'id' (int): Unique track identifier
                - 'x' (float): Center x coordinate
                - 'y' (float): Center y coordinate
                - 'width' (float): Bounding box width
                - 'height' (float): Bounding box height
                - 'class' (str): Ball color/type
                - 'class_id' (int): Numeric class identifier
                - 'confidence' (float): Tracking confidence [0, 1]
                - 'trail' (List[Tuple]): Recent position history for visualization
                - 'population_size' (int): Number of trackers for this object
                - 'best_weight' (float): Weight of the best tracker
                - 'frames_tracked' (int): Number of frames this object has been tracked
        
        Example:
            >>> tracks = tracker.update(current_frame)
            >>> for track in tracks:
            ...     print(f"Track {track['id']}: {track['class']} at "
            ...           f"({track['x']:.1f}, {track['y']:.1f}) "
            ...           f"confidence={track['confidence']:.3f}")
        
        Note:
            - Returns empty list if tracker is not initialized
            - Handles frame processing errors gracefully by returning previous results
            - Updates internal statistics and ball count information
            - Automatically manages tracker population regeneration
        """
        if not self.is_initialized:
            return []
        
        # Validate frame
        if frame is None or len(frame.shape) != 3:
            print("Error: Invalid frame provided to update method")
            return []
        
        try:
            self.frame_count += 1
            
            # Update all tracker populations with current frame
            self._update_populations(frame)
            
            # Generate tracking results from current population states
            tracks = self._generate_tracking_results()

            # Regenerate populations for next frame (after extracting results)
            self._regenerate_populations()
            
            # Clean up trails for inactive tracks
            self._cleanup_old_trails(tracks)
            
            # Verify ball counts and handle violations
            self._verify_ball_counts()
            
            # TODO: In future tasks, we may incorporate new detections here
            # if detections is not None:
            #     self._incorporate_new_detections(detections, frame)
            
            return tracks
            
        except Exception as e:
            print(f"Error updating MOLT tracker: {e}")
            return []
    
    def reset(self) -> None:
        """
        Reset the tracker state for a new video sequence.
        
        This method performs a comprehensive reset of all internal tracking state
        while preserving the configuration parameters. Use this method when starting
        a new video sequence or when you need to clear tracking history.
        
        The reset process clears:
        - All tracker populations and their local trackers
        - Ball count statistics and violation history
        - Trajectory trails and visualization data
        - Frame counters and tracking statistics
        - Track ID assignments and object mappings
        
        Configuration parameters are preserved:
        - Population sizes and exploration radii
        - Histogram parameters and similarity weights
        - Expected ball counts and confidence thresholds
        - Color space and other algorithm settings
        
        Example:
            >>> # Process first video
            >>> tracker.init(frame1, detections1)
            >>> for frame in video1_frames:
            ...     tracks = tracker.update(frame)
            
            >>> # Reset for second video
            >>> tracker.reset()
            >>> tracker.init(frame2, detections2)
            >>> for frame in video2_frames:
            ...     tracks = tracker.update(frame)
        
        Note:
            - Must call init() again after reset() before using update()
            - Reset is automatically called during failed initialization attempts
            - Does not affect configuration parameters or algorithm settings
        """
        # Clear population state
        for population in self.populations:
            if hasattr(population, 'reset'):
                population.reset()
        self.populations.clear()
        
        # Clear tracking state
        self.ball_counts.clear()
        self.frame_count = 0
        self.next_track_id = 1
        self.is_initialized = False
        
        # Clear frame information
        self.frame_width = 0
        self.frame_height = 0
        
        # Clear visualization state
        self.trails.clear()
        
        # Reset statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_fragmentations = 0
        
        # Reset components
        if hasattr(self, 'ball_count_manager'):
            self.ball_count_manager.reset()
        
        # Reset histogram extractor if needed
        if hasattr(self, 'histogram_extractor') and hasattr(self.histogram_extractor, 'reset'):
            self.histogram_extractor.reset()
        
        print("MOLT tracker reset completed")
    
    def visualize(self, frame: Frame, tracks: Sequence[Track], 
                 output_path: Optional[str] = None) -> Frame:
        """
        Create a comprehensive visualization of tracking results on the input frame.
        
        This method generates a detailed visualization that shows all aspects of the
        MOLT tracking algorithm's operation, including individual tracks, population
        statistics, ball count information, and trajectory trails.
        
        Visualization elements include:
        - Bounding boxes with confidence-based line thickness
        - Center points with distinctive markers
        - Trajectory trails with fading effects (newer positions brighter)
        - Comprehensive track labels showing:
          * Track ID and ball class
          * Tracking confidence score
          * Population size and best tracker weight
          * Number of frames tracked
        - Frame-level information:
          * Current frame number and total active tracks
          * Ball count violations and statistics
          * Population performance metrics
        - Color coding based on ball types for easy identification
        
        Args:
            frame (NDArray[np.uint8]): Input frame as numpy array with shape (H, W, C) 
                in BGR format. The original frame is not modified.
            tracks (List[Dict]): Tracking results from update() method containing
                track information and MOLT-specific statistics.
            output_path (str, optional): Optional file path to save the visualization.
                If provided, the visualization will be saved as an image file.
                Supports common formats (.jpg, .png, .bmp, etc.).
                
        Returns:
            NDArray[np.uint8]: A copy of the input frame with comprehensive tracking
                visualization overlaid. Same shape and format as input frame.
        
        Example:
            >>> tracks = tracker.update(frame)
            >>> vis_frame = tracker.visualize(frame, tracks, "output/frame_001.jpg")
            >>> cv2.imshow("MOLT Tracking", vis_frame)
            >>> cv2.waitKey(1)
        
        Color Coding:
            - White balls: White bounding boxes and labels
            - Red balls: Red bounding boxes and labels  
            - Yellow balls: Yellow bounding boxes and labels
            - Green balls: Green bounding boxes and labels
            - Brown balls: Brown bounding boxes and labels
            - Blue balls: Blue bounding boxes and labels
            - Pink balls: Pink bounding boxes and labels
            - Black balls: Dark gray bounding boxes and labels
            - Unknown balls: Gray bounding boxes and labels
        
        Note:
            - Visualization performance scales with number of tracks and trail length
            - Labels are positioned to avoid overlapping with bounding boxes
            - Trail visualization shows last 30 positions by default
            - Ball count violations are highlighted in red text
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Define colors for different ball classes (BGR format for OpenCV)
        colors = {
            'white': (255, 255, 255),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'brown': (42, 42, 165),
            'blue': (255, 0, 0),
            'pink': (203, 192, 255),
            'black': (50, 50, 50),  # Use dark gray instead of pure black for visibility
            'unknown': (128, 128, 128)
        }
        
        # Draw trajectory trails first (so they appear behind other elements)
        for track in tracks:
            track_id = track.get('id', 0)
            ball_class = track.get('class', 'unknown')
            trail = track.get('trail', [])
            
            if len(trail) > 1:
                color = colors.get(ball_class, colors['unknown'])
                
                # Draw trail as connected lines with fading effect
                for i in range(1, len(trail)):
                    # Calculate alpha based on position in trail (newer points are brighter)
                    alpha = i / len(trail)
                    trail_color = tuple(int(c * alpha) for c in color)
                    
                    pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    
                    # Draw trail line
                    cv2.line(vis_frame, pt1, pt2, trail_color, 2)
        
        # Draw tracks with enhanced information
        for track in tracks:
            x, y = int(track.get('x', 0)), int(track.get('y', 0))
            w, h = int(track.get('width', 0)), int(track.get('height', 0))
            track_id = track.get('id', 0)
            ball_class = track.get('class', 'unknown')
            confidence = track.get('confidence', 0.0)
            
            # MOLT-specific information
            population_size = track.get('population_size', 0)
            best_weight = track.get('best_weight', 0.0)
            frames_tracked = track.get('frames_tracked', 0)
            
            # Calculate bounding box coordinates
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            
            # Get color for this ball class
            color = colors.get(ball_class, colors['unknown'])
            
            # Adjust line thickness based on confidence
            thickness = max(1, int(confidence * 4))
            
            # Draw bounding box with confidence-based thickness
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            cv2.circle(vis_frame, (x, y), 4, color, -1)
            cv2.circle(vis_frame, (x, y), 6, (255, 255, 255), 1)  # White outline
            
            # Create comprehensive label with MOLT information
            main_label = f"ID:{track_id} {ball_class}"
            conf_label = f"Conf:{confidence:.3f}"
            pop_label = f"Pop:{population_size} W:{best_weight:.3f}"
            frames_label = f"Frames:{frames_tracked}"
            
            # Calculate label positions
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            
            # Get text sizes
            main_size = cv2.getTextSize(main_label, font, font_scale, font_thickness)[0]
            conf_size = cv2.getTextSize(conf_label, font, font_scale, font_thickness)[0]
            pop_size = cv2.getTextSize(pop_label, font, font_scale, font_thickness)[0]
            frames_size = cv2.getTextSize(frames_label, font, font_scale, font_thickness)[0]
            
            # Calculate label background size
            max_width = max(main_size[0], conf_size[0], pop_size[0], frames_size[0])
            total_height = main_size[1] + conf_size[1] + pop_size[1] + frames_size[1] + 20  # 5px padding each
            
            # Draw label background
            label_x1 = x1
            label_y1 = y1 - total_height - 5
            label_x2 = x1 + max_width + 10
            label_y2 = y1
            
            # Ensure label stays within frame bounds
            if label_y1 < 0:
                label_y1 = y2 + 5
                label_y2 = y2 + total_height + 10
            
            # Draw semi-transparent background
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
            
            # Draw label border
            cv2.rectangle(vis_frame, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), 1)
            
            # Draw text labels
            text_color = (0, 0, 0) if ball_class == 'white' else (255, 255, 255)
            y_offset = label_y1 + main_size[1] + 5
            
            cv2.putText(vis_frame, main_label, (label_x1 + 5, y_offset), 
                       font, font_scale, text_color, font_thickness)
            y_offset += conf_size[1] + 5
            
            cv2.putText(vis_frame, conf_label, (label_x1 + 5, y_offset), 
                       font, font_scale, text_color, font_thickness)
            y_offset += pop_size[1] + 5
            
            cv2.putText(vis_frame, pop_label, (label_x1 + 5, y_offset), 
                       font, font_scale, text_color, font_thickness)
            y_offset += frames_size[1] + 5
            
            cv2.putText(vis_frame, frames_label, (label_x1 + 5, y_offset), 
                       font, font_scale, text_color, font_thickness)
        
        # Draw comprehensive frame information
        info_y = 25
        info_font = cv2.FONT_HERSHEY_SIMPLEX
        info_scale = 0.6
        info_thickness = 2
        info_color = (255, 255, 255)
        
        # Main frame info
        frame_info = f"Frame: {self.frame_count} | Tracks: {len(tracks)} | MOLT Tracker"
        cv2.putText(vis_frame, frame_info, (10, info_y), info_font, info_scale, info_color, info_thickness)
        info_y += 25
        
        # Ball count information
        if hasattr(self, 'ball_count_manager'):
            ball_stats = self.ball_count_manager.get_statistics()
            violations = ball_stats['total_count_violations']
            
            count_info = f"Ball Count Violations: {violations}"
            cv2.putText(vis_frame, count_info, (10, info_y), info_font, info_scale, info_color, info_thickness)
            info_y += 20
            
            # Current vs expected counts
            current_counts = ball_stats['current_counts']
            expected_counts = ball_stats['expected_counts']
            
            for ball_class in ['white', 'red', 'yellow', 'green', 'brown', 'blue', 'pink', 'black']:
                if ball_class in expected_counts:
                    current = current_counts.get(ball_class, 0)
                    expected = expected_counts[ball_class]
                    
                    if current != expected:
                        count_color = (0, 0, 255)  # Red for violations
                    else:
                        count_color = (0, 255, 0)  # Green for correct counts
                    
                    count_text = f"{ball_class}: {current}/{expected}"
                    cv2.putText(vis_frame, count_text, (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, count_color, 1)
                    info_y += 15
        
        # Draw population statistics
        if self.populations:
            pop_info = f"Active Populations: {len(self.populations)}"
            cv2.putText(vis_frame, pop_info, (10, info_y), info_font, 0.5, info_color, 1)
            info_y += 20
            
            # Average population performance
            total_weight: float = 0.0
            total_trackers: int = 0
            for population in self.populations:
                if population.best_tracker:
                    total_weight += population.best_tracker.total_weight
                    total_trackers += len(population.trackers)
            
            if len(self.populations) > 0:
                avg_weight = total_weight / len(self.populations)
                avg_pop_size = total_trackers / len(self.populations)
                
                perf_info = f"Avg Weight: {avg_weight:.3f} | Avg Pop Size: {avg_pop_size:.0f}"
                cv2.putText(vis_frame, perf_info, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, info_color, 1)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_frame)
        
        return vis_frame
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current tracker configuration as a dictionary.
        
        This method returns the current configuration parameters used by the tracker,
        including both the original configuration and any runtime modifications.
        
        Returns:
            Dict[str, Any]: Dictionary containing current configuration parameters
            
        Example:
            >>> config = tracker.get_config()
            >>> print(f"Histogram bins: {config['histogram_bins']}")
            >>> print(f"Color space: {config['color_space']}")
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
    
    def get_ball_count_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive ball count statistics and violation information.
        
        Returns:
            Dict[str, Any]: Dictionary containing ball count statistics
        """
        if hasattr(self, 'ball_count_manager'):
            return self.ball_count_manager.get_statistics()
        else:
            return {
                'expected_counts': self.expected_ball_counts,
                'current_counts': self.ball_counts,
                'total_count_violations': 0,
                'lost_ball_recoveries': 0,
                'duplicate_ball_merges': 0
            }
    
    def get_tracker_info(self) -> TrackerInfo:
        """
        Get comprehensive information about the tracker state and performance.
        
        This method returns detailed information about the MOLT tracker's current
        state, configuration parameters, performance statistics, and operational
        metrics. Use this for debugging, performance analysis, and system monitoring.
        
        The returned information includes:
        - Basic tracker identification (name, type)
        - Configuration parameters (population sizes, exploration radii, etc.)
        - Performance metrics (average weights, population statistics)
        - Ball counting information (current counts, violations, recoveries)
        - Tracking statistics (frames processed, active tracks, fragmentations)
        
        Returns:
            Dict: Comprehensive tracker information dictionary containing:
                - 'name' (str): Tracker algorithm name ("MOLT")
                - 'type' (str): Full algorithm description
                - 'parameters' (Dict): All configuration and performance parameters:
                  * Core MOLT parameters (population_sizes, exploration_radii, etc.)
                  * Performance metrics (avg_best_weight, avg_population_weight)
                  * Ball count information (current_ball_counts, violations, etc.)
                - 'frame_count' (int): Number of frames processed
                - 'active_tracks' (int): Number of currently active tracks
                - 'total_tracks_created' (int): Total tracks created since initialization
                - 'total_tracks_lost' (int): Total tracks lost during tracking
                - 'total_fragmentations' (int): Total track fragmentation events
        
        Example:
            >>> info = tracker.get_tracker_info()
            >>> print(f"Tracker: {info['name']} ({info['type']})")
            >>> print(f"Frames processed: {info['frame_count']}")
            >>> print(f"Active tracks: {info['active_tracks']}")
            >>> print(f"Average best weight: {info['parameters']['avg_best_weight']:.3f}")
            >>> print(f"Ball count violations: {info['parameters']['ball_count_violations']}")
        
        Use Cases:
            - Performance monitoring and optimization
            - Debugging tracking issues
            - System health checks
            - Algorithm parameter tuning
            - Logging and analytics
        
        Note:
            - Information reflects the current state at time of call
            - Performance metrics are calculated from active populations
            - Ball count statistics include historical violation data
        """
        # Calculate population statistics
        active_populations = len([p for p in self.populations if p is not None and p.best_tracker is not None])
        total_population_size = sum(len(p.trackers) for p in self.populations)
        
        # Calculate average performance metrics
        avg_best_weight = 0.0
        avg_population_weight = 0.0
        if self.populations:
            total_best_weight = sum(p.best_tracker.total_weight for p in self.populations if p.best_tracker)
            avg_best_weight = total_best_weight / len(self.populations) if self.populations else 0.0
            
            # Get average population weights
            pop_stats = [p.get_population_statistics() for p in self.populations]
            if pop_stats:
                avg_population_weight = sum(stats['average_weight'] for stats in pop_stats) / len(pop_stats)
        
        # Get ball count statistics
        ball_count_stats = self.get_ball_count_statistics() if hasattr(self, 'ball_count_manager') else {}
        
        return TrackerInfo(
            name="MOLT",
            type="Multiple Object Local Tracker",
            parameters={
                # Core MOLT parameters
                'population_sizes': self.population_sizes,
                'exploration_radii': self.exploration_radii,
                'expected_ball_counts': self.expected_ball_counts,
                'histogram_bins': self.histogram_bins,
                'similarity_weights': self.similarity_weights,
                'diversity_distribution': self.diversity_distribution,
                'color_space': self.color_space,
                'min_confidence': self.min_confidence,
                'max_frames_without_detection': self.max_frames_without_detection,
                'max_trail_length': self.max_trail_length,
                
                # Performance metrics
                'avg_best_weight': avg_best_weight,
                'avg_population_weight': avg_population_weight,
                'total_population_size': total_population_size,
                
                # Ball count information
                'current_ball_counts': self.ball_counts.copy(),
                'ball_count_violations': ball_count_stats.get('total_count_violations', 0),
                'lost_ball_recoveries': ball_count_stats.get('lost_ball_recoveries', 0),
                'duplicate_ball_merges': ball_count_stats.get('duplicate_ball_merges', 0)
            },
            frame_count=self.frame_count,
            active_tracks=active_populations,
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
                                  size: Size) -> Optional[Histogram]:
        """
        Extract initial histogram from detection patch.
        
        Args:
            frame: Input frame as a numpy array with shape (H, W, C) in BGR format
            center: (x, y) center position of detection
            size: (width, height) size of detection
            
        Returns:
            Optional[NDArray[np.float64]]: Extracted histogram or None if extraction fails
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
        
        This method updates each population by:
        1. Computing similarities for all trackers in each population
        2. Ranking trackers by combined weight
        3. Regenerating populations around best trackers
        
        Args:
            frame: Current frame
        """
        if not self.populations:
            return
        
        # Update each population
        for population in self.populations:
            try:
                # Update population with current frame
                best_tracker = population.update(
                    frame=frame,
                    histogram_extractor=self.histogram_extractor,
                    similarity_weights=self.similarity_weights
                )
                
                if best_tracker is None:
                    print(f"Warning: No best tracker found for population {population.object_id}")
                    continue
                
            except Exception as e:
                print(f"Error updating population {population.object_id}: {e}")
                continue
    
    def _regenerate_populations(self) -> None:
        """
        Regenerate all populations for the next frame.
        
        This method creates new tracker populations around the best trackers
        from the current frame. It should be called AFTER extracting tracking
        results to avoid resetting the weights before they are used.
        """
        if not self.populations:
            return
        
        # Regenerate each population
        for population in self.populations:
            try:
                best_tracker = population.get_best_tracker()
                
                if best_tracker is None:
                    continue
                
                # Get exploration radius for this object class
                exploration_radius = self.exploration_radii.get(
                    population.object_class,
                    self.exploration_radii.get('default', 15)
                )
                
                # Regenerate population around best trackers for next frame
                population.generate_new_population(
                    exploration_radius=exploration_radius,
                    diversity_distribution=self.diversity_distribution
                )
                
            except Exception as e:
                print(f"Error updating population {population.object_id}: {e}")
                continue
    
    def _verify_ball_counts(self) -> None:
        """
        Verify and correct ball count inconsistencies.
        
        This method uses the BallCountManager to check if the current
        number of tracked balls matches expected counts and handles
        violations by suggesting track merges or flagging lost balls.
        """
        if not self.populations:
            return
        
        try:
            # Generate current tracks for count verification
            current_tracks = []
            for population in self.populations:
                best_tracker = population.get_best_tracker()
                if best_tracker is not None and best_tracker.total_weight >= self.min_confidence:
                    track = {
                        'id': population.object_id,
                        'class': population.object_class,
                        'x': float(best_tracker.center[0]),
                        'y': float(best_tracker.center[1]),
                        'confidence': float(best_tracker.total_weight)
                    }
                    current_tracks.append(track)
            
            # Update ball count manager with current tracks
            self.ball_count_manager.update_counts_from_tracks(current_tracks)
            
            # Verify counts
            counts_valid = self.ball_count_manager.verify_counts()
            
            if not counts_valid:
                # Handle count violations
                violations = self.ball_count_manager.get_count_violations()
                
                for ball_class, violation_info in violations.items():
                    violation_type = violation_info['violation_type']
                    
                    if violation_type == 'over_count':
                        # Too many balls of this class - suggest merges
                        class_tracks = [t for t in current_tracks if t.get('class') == ball_class]
                        # Ensure track_ids is a List[int] by explicitly converting to int
                        from typing import cast
                        track_ids = [cast(int, t['id']) for t in class_tracks]
                        
                        excess_tracks = self.ball_count_manager.handle_duplicate_ball(ball_class, track_ids)
                        
                        if excess_tracks:
                            print(f"Warning: Too many {ball_class} balls detected. "
                                  f"Expected: {violation_info['expected']}, "
                                  f"Current: {violation_info['current']}. "
                                  f"Consider merging tracks: {excess_tracks}")
                            
                            # In a full implementation, we could actually merge populations here
                            # For now, we just log the issue
                    
                    elif violation_type == 'under_count':
                        # Too few balls of this class - attempt recovery
                        self.ball_count_manager.handle_lost_ball(ball_class)
                        
                        print(f"Warning: Missing {ball_class} balls. "
                              f"Expected: {violation_info['expected']}, "
                              f"Current: {violation_info['current']}. "
                              f"Missing: {abs(violation_info['difference'])}")
                        
                        # In a full implementation, we could increase exploration radius
                        # or lower confidence thresholds for this ball class
                
                # Get merge suggestions for spatial proximity
                merge_suggestions = self.ball_count_manager.suggest_track_merges(current_tracks)
                
                if merge_suggestions:
                    print(f"Suggested track merges based on proximity: {merge_suggestions}")
                    # In a full implementation, we could merge the suggested populations
            
            # Update internal ball counts for statistics
            self.ball_counts = self.ball_count_manager.current_counts.copy()
            
        except Exception as e:
            print(f"Error in ball count verification: {e}")
    
    def _generate_tracking_results(self) -> List[Track]:
        """
        Generate tracking results from current population states.
        
        Extracts the best tracker from each population and formats
        the results according to the standard tracking output schema.
        Includes trajectory trails, confidence scores, and comprehensive
        tracking statistics.
        
        Returns:
            List[Track]: List of tracking results with positions, IDs, trails, and metadata
        """
        tracks = []
        
        for population in self.populations:
            try:
                best_tracker = population.get_best_tracker()
                
                if best_tracker is None:
                    continue
                
                # Check if tracker meets minimum confidence threshold
                if best_tracker.total_weight < self.min_confidence:
                    continue
                
                track_id = population.object_id
                center_x = float(best_tracker.center[0])
                center_y = float(best_tracker.center[1])
                
                # Update trail for this track
                if track_id not in self.trails:
                    self.trails[track_id] = []
                
                # Add current position to trail
                self.trails[track_id].append((center_x, center_y))
                
                # Limit trail length
                if len(self.trails[track_id]) > self.max_trail_length:
                    self.trails[track_id] = self.trails[track_id][-self.max_trail_length:]
                
                # Get population statistics for enhanced tracking info
                pop_stats = population.get_population_statistics()
                
                # Calculate normalized confidence (0-1 range)
                normalized_confidence = min(1.0, max(0.0, best_tracker.total_weight))
                
                # Create comprehensive tracking result
                track = {
                    'id': track_id,
                    'x': center_x,
                    'y': center_y,
                    'width': float(best_tracker.size[0]),
                    'height': float(best_tracker.size[1]),
                    'class': population.object_class,
                    'class_id': self._get_class_id(population.object_class),
                    'confidence': normalized_confidence,
                    'trail': self.trails[track_id].copy(),
                    
                    # MOLT-specific tracking information
                    'population_size': len(population.trackers),
                    'best_weight': float(best_tracker.total_weight),
                    'frames_tracked': population.frame_count,
                    'histogram_weight': float(best_tracker.hist_weight),
                    'spatial_weight': float(best_tracker.dist_weight),
                    'average_population_weight': float(pop_stats.get('average_weight', 0.0)),
                    'weight_std': float(pop_stats.get('weight_std', 0.0)),
                    
                    # Additional metadata
                    'tracker_type': 'MOLT',
                    'algorithm_version': '1.0'
                }
                
                tracks.append(track)
                
            except Exception as e:
                print(f"Error generating tracking result for population {population.object_id}: {e}")
                continue
        
        return tracks
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all populations.
        
        This method returns detailed statistics about the performance and state
        of all tracker populations, useful for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Dictionary containing population statistics
            
        Example:
            >>> stats = tracker.get_population_statistics()
            >>> print(f"Total populations: {stats['total_populations']}")
            >>> print(f"Ball counts: {stats['ball_counts']}")
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
    
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for the MOLT tracker.
        
        Returns:
            Dict[str, Any]: Dictionary containing performance metrics including
                          population statistics and tracking metrics
        """
        # Initialize metrics with default values
        metrics: Dict[str, Any] = {
            'frame_count': int(self.frame_count),
            'total_populations': int(len(self.populations)),
            'active_populations': int(len([p for p in self.populations if hasattr(p, 'best_tracker') and p.best_tracker is not None])),
            'total_tracks_created': int(self.total_tracks_created),
            'total_tracks_lost': int(self.total_tracks_lost),
            'total_fragmentations': int(self.total_fragmentations),
            'active_trails': int(len(self.trails)),
            'is_initialized': bool(self.is_initialized),
            'population_metrics': [],
            'ball_count_metrics': {}
        }
        
        # Population metrics
        if self.populations:
            population_metrics: List[Dict[str, Any]] = []
            total_trackers = 0
            total_best_weight = 0.0
            total_avg_weight = 0.0
            
            for population in self.populations:
                pop_stats = population.get_population_statistics()
                if isinstance(pop_stats, dict):
                    population_metrics.append(pop_stats)
                    total_trackers += int(pop_stats.get('population_size', 0))
                    total_best_weight += float(pop_stats.get('best_weight', 0.0))
                    total_avg_weight += float(pop_stats.get('average_weight', 0.0))
            
            if population_metrics:  # Only update if we have valid metrics
                metrics.update({
                    'population_metrics': population_metrics,
                    'total_trackers': int(total_trackers),
                    'avg_best_weight': float(total_best_weight / len(population_metrics)) if population_metrics else 0.0,
                    'avg_population_weight': float(total_avg_weight / len(population_metrics)) if population_metrics else 0.0,
                    'avg_trackers_per_population': float(total_trackers / len(population_metrics)) if population_metrics else 0.0
                })
        
        # Ball count metrics
        if hasattr(self, 'ball_count_manager') and self.ball_count_manager is not None:
            ball_metrics = self.ball_count_manager.get_statistics()
            if ball_metrics is not None:
                try:
                    # Try to convert to dict if it has dict-like interface
                    if hasattr(ball_metrics, 'items'):
                        metrics['ball_count_metrics'] = dict(ball_metrics)
                    else:
                        # Handle other types by converting to string representation
                        metrics['ball_count_metrics'] = str(ball_metrics)
                except (TypeError, AttributeError):
                    # Fallback to string representation
                    metrics['ball_count_metrics'] = str(ball_metrics)
        
        return metrics
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """
        Get current tracking status and health information.
        
        Returns:
            Dict[str, Any]: Status information including health indicators
                          and potential issues
        """
        status: Dict[str, Any] = {
            'is_initialized': bool(self.is_initialized),
            'frame_count': int(self.frame_count),
            'tracking_health': 'unknown',
            'issues': [],
            'recommendations': [],
            'active_populations': 0,
            'total_populations': 0
        }
        
        if not self.is_initialized:
            status['tracking_health'] = 'not_initialized'
            status['issues'].append('Tracker not initialized')
            status['recommendations'].append('Call init() with first frame and detections')
            return status
        
        # Check population health
        active_populations = [p for p in self.populations if hasattr(p, 'best_tracker') and p.best_tracker is not None]
        
        if not active_populations:
            status['tracking_health'] = 'critical'
            status['issues'].append('No active populations')
            status['recommendations'].append('Check detection quality and confidence thresholds')
        elif self.populations and len(active_populations) < len(self.populations) * 0.5:
            status['tracking_health'] = 'poor'
            status['issues'].append(f'Only {len(active_populations)}/{len(self.populations)} populations active')
            status['recommendations'].append('Consider lowering confidence threshold or increasing exploration radius')
        else:
            status['tracking_health'] = 'good'
        
        # Check ball count violations
        if hasattr(self, 'ball_count_manager') and self.ball_count_manager is not None:
            ball_stats = self.ball_count_manager.get_statistics()
            if isinstance(ball_stats, dict):
                violations = int(ball_stats.get('total_count_violations', 0))
                
                if violations > self.frame_count * 0.1:  # More than 10% of frames have violations
                    status['issues'].append(f'High ball count violations: {violations}')
                    status['recommendations'].append('Check detection accuracy and tracking parameters')
        
        # Check average confidence
        if active_populations:
            try:
                total_weight = sum(float(getattr(p.best_tracker, 'total_weight', 0.0)) for p in active_populations)
                avg_confidence = total_weight / len(active_populations)
                
                if avg_confidence < 0.3:
                    status['issues'].append(f'Low average confidence: {avg_confidence:.3f}')
                    status['recommendations'].append('Improve lighting conditions or adjust histogram parameters')
            except (AttributeError, ZeroDivisionError):
                pass  # Skip confidence check if attributes are missing or division by zero
        
        status['active_populations'] = int(len(active_populations))
        status['total_populations'] = int(len(self.populations))
        
        return status
    

    
    def _get_class_id(self, class_name: str) -> int:
        """
        Map class name to numeric class ID.
        
        Args:
            class_name: String class name (e.g., 'red', 'white', 'yellow')
            
        Returns:
            int: Numeric class ID
        """
        class_mapping = {
            'white': 0,
            'red': 1,
            'yellow': 2,
            'green': 3,
            'brown': 4,
            'blue': 5,
            'pink': 6,
            'black': 7
        }
        
        return class_mapping.get(class_name, -1)
    
    def _cleanup_old_trails(self, current_tracks: List[Track]) -> None:
        """
        Clean up trails for tracks that are no longer active.
        
        Args:
            current_tracks: List of currently active tracks
        """
        # Get set of active track IDs
        active_track_ids = {track['id'] for track in current_tracks}
        
        # Remove trails for inactive tracks
        inactive_track_ids = set(self.trails.keys()) - active_track_ids
        for track_id in inactive_track_ids:
            del self.trails[track_id]