"""
Type definitions and common types for MOLT tracker.

This module defines the common types used throughout the MOLT tracker
implementation to ensure type safety and consistency.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray

# Type aliases for common data structures
Detection = Dict[str, Any]  # Type for detection dictionaries
Track = Dict[str, Any]      # Type for tracking result dictionaries
Frame = NDArray[np.uint8]   # Type for image frames (H, W, C) in BGR format
Histogram = NDArray[np.float32]  # Type for color histograms
Position = Tuple[float, float]   # Type for (x, y) positions
Size = Tuple[float, float]       # Type for (width, height) dimensions
BoundingBox = Tuple[float, float, float, float]  # Type for (x, y, width, height)

# Constants for histogram methods
HISTOGRAM_METHODS = ['bhattacharyya', 'intersection', 'chi_square']
DEFAULT_HISTOGRAM_METHOD = 'bhattacharyya'

# Constants for color spaces
COLOR_SPACES = ['HSV', 'RGB', 'LAB']
DEFAULT_COLOR_SPACE = 'HSV'

# Default similarity weights
DEFAULT_SIMILARITY_WEIGHTS = {
    'histogram': 0.6,   # Weight for appearance similarity
    'spatial': 0.4      # Weight for spatial consistency
}

# Default diversity distribution for population generation
DEFAULT_DIVERSITY_DISTRIBUTION = [0.5, 0.3, 0.2]  # Best, second, third tracker ratios