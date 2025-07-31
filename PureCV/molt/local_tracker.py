"""
Individual local tracker implementation for MOLT algorithm.

This module provides the LocalTracker class which represents a single
hypothesis about object location within a tracker population.
"""

from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .types import Position, Size, Histogram, BoundingBox

if TYPE_CHECKING:
    from .histogram_extractor import HistogramExtractor


class LocalTracker:
    """
    Individual tracker within a population that maintains position and appearance information.
    
    Each LocalTracker represents a hypothesis about where an object might be located
    in the current frame. It stores position, size, appearance features (histogram),
    and computes similarity scores for tracking decisions.
    """
    
    def __init__(self, center: Position, size: Size, 
                 histogram: Histogram, tracker_id: int) -> None:
        """
        Initialize a local tracker with position and appearance information.
        
        Args:
            center: (x, y) center position of the tracker
            size: (width, height) size of the tracked object
            histogram: Reference appearance histogram as 1D numpy array
            tracker_id: Unique identifier for this tracker within the population
            
        Raises:
            ValueError: If histogram is invalid
        """
        # Validate inputs
        if len(center) != 2:
            raise ValueError("Center must be a tuple of (x, y) coordinates")
        if len(size) != 2:
            raise ValueError("Size must be a tuple of (width, height)")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Size dimensions must be positive")
        
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
            # Validate patch
            if patch is None or patch.size == 0:
                return 0.0
            
            if len(patch.shape) != 3 or patch.shape[2] != 3:
                return 0.0
            
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
    
    def compute_distance(self, reference_pos: Position) -> float:
        """
        Calculate spatial distance from tracker position to reference position.
        
        Args:
            reference_pos: (x, y) reference position to compare against
            
        Returns:
            float: Euclidean distance between tracker center and reference position
            
        Raises:
            ValueError: If reference position is invalid
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
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not (0.0 <= hist_similarity <= 1.0):
            raise ValueError(f"Histogram similarity must be in [0, 1], got {hist_similarity}")
        
        if spatial_distance < 0:
            raise ValueError(f"Spatial distance must be non-negative, got {spatial_distance}")
        
        if 'histogram' not in similarity_weights or 'spatial' not in similarity_weights:
            raise ValueError("similarity_weights must contain 'histogram' and 'spatial' keys")
        
        if max_distance <= 0:
            raise ValueError(f"Max distance must be positive, got {max_distance}")
        
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
    
    def update_position(self, new_center: Position) -> None:
        """
        Update the tracker's position.
        
        Args:
            new_center: New (x, y) center position
            
        Raises:
            ValueError: If new_center is invalid
        """
        if new_center is None or len(new_center) != 2:
            raise ValueError("New center must be a tuple of (x, y) coordinates")
        
        self.center = new_center
    
    def update_size(self, new_size: Size) -> None:
        """
        Update the tracker's size.
        
        Args:
            new_size: New (width, height) size
            
        Raises:
            ValueError: If new_size is invalid
        """
        if new_size is None or len(new_size) != 2:
            raise ValueError("New size must be a tuple of (width, height)")
        
        if new_size[0] <= 0 or new_size[1] <= 0:
            raise ValueError("Size dimensions must be positive")
        
        self.size = new_size
    
    def update_histogram(self, new_histogram: Histogram) -> None:
        """
        Update the tracker's reference histogram.
        
        Args:
            new_histogram: New reference histogram
            
        Raises:
            ValueError: If new_histogram is invalid
        """
        if new_histogram is None or new_histogram.size == 0:
            raise ValueError("Histogram cannot be None or empty")
        
        self.histogram = new_histogram.copy()
    
    @property
    def total_weight_property(self) -> float:
        """
        Get the total weight for tracker ranking.
        
        Returns:
            float: Combined weight score for ranking trackers
        """
        return self.total_weight
    
    def get_position(self) -> Position:
        """
        Get the current tracker position.
        
        Returns:
            Position: (x, y) center position
        """
        return self.center
    
    def get_size(self) -> Size:
        """
        Get the current tracker size.
        
        Returns:
            Size: (width, height) size
        """
        return self.size
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box coordinates.
        
        Returns:
            BoundingBox: (x, y, width, height) where (x, y) is center
        """
        return (self.center[0], self.center[1], self.size[0], self.size[1])
    
    def get_corner_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box with corner coordinates.
        
        Returns:
            Tuple[float, float, float, float]: (x1, y1, x2, y2) where (x1, y1) is top-left
        """
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2
        
        x1 = self.center[0] - half_width
        y1 = self.center[1] - half_height
        x2 = self.center[0] + half_width
        y2 = self.center[1] + half_height
        
        return (x1, y1, x2, y2)
    
    def is_valid(self) -> bool:
        """
        Check if the tracker is in a valid state.
        
        Returns:
            bool: True if tracker is valid, False otherwise
        """
        try:
            # Check size validity (most likely to fail)
            if self.size[0] <= 0 or self.size[1] <= 0:
                return False
            
            # Check histogram validity
            if self.histogram is None or self.histogram.size == 0:
                return False
            
            # Check weights validity
            if not (0.0 <= self.total_weight <= 1.0):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_state_dict(self) -> Dict:
        """
        Get the tracker state as a dictionary.
        
        Returns:
            Dict: Dictionary containing tracker state
        """
        return {
            'tracker_id': self.tracker_id,
            'center': self.center,
            'size': self.size,
            'hist_weight': self.hist_weight,
            'dist_weight': self.dist_weight,
            'total_weight': self.total_weight,
            'confidence': self.confidence,
            'last_update_frame': self.last_update_frame,
            'histogram_shape': self.histogram.shape,
            'is_valid': self.is_valid()
        }
    
    def __str__(self) -> str:
        """String representation of the tracker."""
        return (f"LocalTracker(id={self.tracker_id}, center={self.center}, "
                f"size={self.size}, total_weight={self.total_weight:.3f})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the tracker."""
        return (f"LocalTracker(id={self.tracker_id}, center={self.center}, "
                f"size={self.size}, hist_weight={self.hist_weight:.3f}, "
                f"dist_weight={self.dist_weight:.3f}, total_weight={self.total_weight:.3f})")