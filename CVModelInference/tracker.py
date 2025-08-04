"""
Tracker Interface Module for CVModelInference

This module defines the Tracker interface for classes that implement
object tracking across video frames.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, TypedDict, TypeVar, Generic, Sequence
import numpy as np
from numpy.typing import NDArray

# Type aliases for common types
Detection = Dict[str, Any]  # Type for detection dictionaries
Track = Dict[str, Any]  # Type for tracking result dictionaries
Frame = NDArray[np.uint8]  # Type for image frames (H, W, C) in BGR format

# Generic type variable for tracker configuration
T = TypeVar('T', bound=Dict[str, Any])


class TrackerInfo(TypedDict, total=False):
    """Type definition for tracker information dictionary."""
    name: str
    type: str
    parameters: Dict[str, Any]
    frame_count: int
    active_tracks: int
    total_tracks_created: int
    total_tracks_lost: int
    total_fragmentations: int


class Tracker(ABC, Generic[T]):
    """
    Abstract base class defining the interface for object trackers.
    
    This interface provides a consistent API for tracking objects across
    video frames using different tracking algorithms.
    
    Type Variables:
        T: Type of the configuration dictionary used to initialize the tracker
    """
    
    def normalize_ball_class(self, class_name: str) -> str:
        """
        Normalize ball class names to match ground truth expectations.
        
        Args:
            class_name: Original class name from detection
            
        Returns:
            Normalized class name
        """
        # Convert to lowercase for case-insensitive comparison
        ball_type = class_name.lower()
        
        # Handle hyphenated class names (e.g., 'red-ball' -> 'red')
        if '-' in ball_type:
            parts = ball_type.split('-')
            if len(parts) == 2 and parts[1] == 'ball':
                ball_type = parts[0]  # Extract color name from 'color-ball'
        
        # Map common variations to standard names
        if ball_type in ['cue', 'cue_ball', 'cueball', 'white_ball', 'white-ball', 'white']:
            return 'white'
        elif ball_type in ['red', 'red_ball', 'redball', 'red-ball']:
            return 'red'
        elif ball_type in ['yellow', 'yellow_ball', 'yellowball', 'yellow-ball']:
            return 'yellow'
        elif ball_type in ['green', 'green_ball', 'greenball', 'green-ball']:
            return 'green'
        elif ball_type in ['brown', 'brown_ball', 'brownball', 'brown-ball']:
            return 'brown'
        elif ball_type in ['blue', 'blue_ball', 'blueball', 'blue-ball']:
            return 'blue'
        elif ball_type in ['pink', 'pink_ball', 'pinkball', 'pink-ball']:
            return 'pink'
        elif ball_type in ['black', 'black_ball', 'blackball', 'black-ball']:
            return 'black'
        
        # If no mapping found, return the original name in lowercase
        return ball_type
    
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the tracker.
        
        Args:
            **kwargs: Implementation-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def init(self, frame: Frame, detections: List[Detection]) -> bool:
        """
        Initialize the tracker with the first frame and initial detections.
        
        Args:
            frame: First frame as numpy array with shape (H, W, C) in BGR format
            detections: List of detection dictionaries, each containing at minimum:
                       {
                           'x': float,  # center x coordinate
                           'y': float,  # center y coordinate
                           'width': float,  # bounding box width
                           'height': float,  # bounding box height
                           'class': str,  # class name
                           'confidence': float,  # detection confidence
                           'id': Optional[Union[int, str]]  # optional track ID
                       }
                        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update(self, frame: Frame, detections: Optional[List[Detection]] = None) -> List[Track]:
        """
        Update the tracker with a new frame and optional new detections.
        
        Args:
            frame: New frame as numpy array with shape (H, W, C) in BGR format
            detections: Optional list of new detections to incorporate.
                       If None, the tracker will only update existing tracks.
                        
        Returns:
            List[Track]: List of tracked objects with updated positions and IDs.
                         Each track is a dictionary containing at least:
                         {
                             'id': int,  # unique track ID
                             'x': float,  # center x coordinate
                             'y': float,  # center y coordinate
                             'width': float,  # bounding box width
                             'height': float,  # bounding box height
                             'class': str,  # class name
                             'confidence': float,  # detection confidence
                             'age': int,  # number of frames since first detection
                             'active': bool  # whether the track is currently active
                         }
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the tracker state.
        
        This method should clear all internal state and prepare the tracker
        for a new sequence of frames.
        """
        pass
    
    @abstractmethod
    def visualize(self, frame: Frame, tracks: Sequence[Track], 
                 output_path: Optional[str] = None) -> Frame:
        """
        Visualize the tracks on the input frame.
        
        Args:
            frame: Input frame as numpy array with shape (H, W, C) in BGR format
            tracks: Tracking results from update()
            output_path: Optional path to save the visualization.
                        If provided, the visualization will be saved to this path.
            
        Returns:
            Frame: A copy of the input frame with tracks visualized.
                   The frame will have the same shape as the input (H, W, C).
        
        Note:
            The input frame should not be modified directly. Instead, create a copy,
            modify the copy, and return it.
        """
        pass
    
    @abstractmethod
    def get_tracker_info(self) -> TrackerInfo:
        """
        Get information about the tracker.
        
        Returns:
            TrackerInfo: Dictionary containing tracker information with the following
                        optional keys:
                        {
                            'name': str,  # tracker name
                            'type': str,  # tracker type/algorithm
                            'parameters': Dict[str, Any],  # tracker parameters
                            'frame_count': int,  # number of frames processed
                            'active_tracks': int,  # number of currently active tracks
                            'total_tracks_created': int,  # total tracks created so far
                            'total_tracks_lost': int,  # total tracks lost so far
                            'total_fragmentations': int  # total track fragmentations
                        }
        """
        pass
