"""
Type definitions and constants for MOLT tracker.

This module provides comprehensive type definitions, type aliases, and constants
used throughout the MOLT tracker implementation. It ensures type safety,
consistency, and provides clear documentation of data structures and parameters.

The module defines:
- Type aliases for common data structures (Detection, Track, Frame, etc.)
- Constants for algorithm parameters and supported options
- Default values for configuration parameters
- Type hints for complex data structures

Type Safety:
All types are designed to work with mypy static type checking and provide
clear documentation of expected data formats and structures.

Example Usage:
    >>> from tracking.trackers.molt.types import Detection, Track, Frame
    >>> from typing import List
    >>> 
    >>> def process_detections(detections: List[Detection]) -> List[Track]:
    ...     # Type-safe detection processing
    ...     pass
    
    >>> # Use constants for validation
    >>> from tracking.trackers.molt.types import HISTOGRAM_METHODS, COLOR_SPACES
    >>> 
    >>> def validate_method(method: str) -> bool:
    ...     return method in HISTOGRAM_METHODS
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray

# Type aliases for common data structures

Detection = Dict[str, Any]
"""
Type alias for detection dictionaries.

Detection dictionaries contain object detection information with the following structure:
{
    'x': float,           # Center x coordinate
    'y': float,           # Center y coordinate
    'width': float,       # Bounding box width (must be > 0)
    'height': float,      # Bounding box height (must be > 0)
    'class': str,         # Object class/ball color ('red', 'white', 'yellow', etc.)
    'class_id': int,      # Numeric class identifier (optional)
    'confidence': float   # Detection confidence [0, 1] (optional)
}
"""

Track = Dict[str, Any]
"""
Type alias for tracking result dictionaries.

Track dictionaries contain tracking results with the following structure:
{
    'id': int,                    # Unique track identifier
    'x': float,                   # Center x coordinate
    'y': float,                   # Center y coordinate
    'width': float,               # Bounding box width
    'height': float,              # Bounding box height
    'class': str,                 # Object class/ball color
    'class_id': int,              # Numeric class identifier
    'confidence': float,          # Tracking confidence [0, 1]
    'trail': List[Tuple[float, float]], # Position history for visualization
    'population_size': int,       # Number of trackers for this object
    'best_weight': float,         # Weight of the best tracker
    'frames_tracked': int         # Number of frames this object has been tracked
}
"""

Frame = NDArray[np.uint8]
"""
Type alias for image frames.

Represents image frames as numpy arrays with shape (H, W, C) in BGR format
(OpenCV standard). Frames must be 3-channel color images with uint8 data type.
"""

Histogram = NDArray[np.float32]
"""
Type alias for color histograms.

Represents normalized color histograms as 1D float32 numpy arrays.
Histograms are typically flattened 3D histograms with length num_bins^3.
Values sum to 1.0 and represent color distribution probabilities.
"""

Position = Tuple[float, float]
"""
Type alias for 2D positions.

Represents (x, y) coordinates as a tuple of floats. Used for object centers,
tracker positions, and other 2D coordinate pairs.
"""

Size = Tuple[float, float]
"""
Type alias for 2D dimensions.

Represents (width, height) dimensions as a tuple of floats. Used for bounding
box sizes, object dimensions, and other 2D size measurements.
"""

BoundingBox = Tuple[float, float, float, float]
"""
Type alias for bounding boxes.

Represents bounding boxes as (x, y, width, height) tuples where:
- x, y: Center coordinates
- width, height: Box dimensions
All values are floats.
"""

# Constants for histogram methods
HISTOGRAM_METHODS = ['bhattacharyya', 'intersection', 'chi_square']
"""
Supported histogram distance methods.

- 'bhattacharyya': Bhattacharyya distance (robust statistical measure, default)
- 'intersection': Histogram intersection distance (fast and intuitive)
- 'chi_square': Chi-square distance (good for normalized histograms)
"""

DEFAULT_HISTOGRAM_METHOD = 'bhattacharyya'
"""Default histogram distance method (most robust for tracking applications)."""

# Constants for color spaces
COLOR_SPACES = ['HSV', 'RGB', 'LAB']
"""
Supported color spaces for histogram extraction.

- 'HSV': Hue-Saturation-Value (default, good for color-based tracking)
- 'RGB': Red-Green-Blue (faster conversion from BGR)
- 'LAB': L*a*b* (perceptually uniform color space)
"""

DEFAULT_COLOR_SPACE = 'HSV'
"""Default color space (best for color-based object tracking)."""

# Default similarity weights
DEFAULT_SIMILARITY_WEIGHTS = {
    'histogram': 0.6,   # Weight for appearance similarity
    'spatial': 0.4      # Weight for spatial consistency
}
"""
Default weights for combining histogram and spatial similarities.

The weights determine the relative importance of appearance vs. motion consistency:
- Higher histogram weight favors appearance matching
- Higher spatial weight favors motion consistency
- Weights should sum to 1.0 for proper normalization
"""

# Default diversity distribution for population generation
DEFAULT_DIVERSITY_DISTRIBUTION = [0.5, 0.3, 0.2]
"""
Default distribution ratios for population regeneration.

Defines how new trackers are distributed around top performers:
- 50% around best tracker (highest weight)
- 30% around second best tracker
- 20% around third best tracker

This strategy balances exploitation of good positions with exploration diversity.
"""