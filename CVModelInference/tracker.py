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
