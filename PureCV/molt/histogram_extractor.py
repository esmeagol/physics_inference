"""
Histogram extraction and comparison for appearance-based tracking.

This module provides the HistogramExtractor class for extracting color histograms
from image patches and comparing them using various distance metrics.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
import cv2

from .types import Frame, Histogram, HISTOGRAM_METHODS, DEFAULT_HISTOGRAM_METHOD


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
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_bins <= 0:
            raise ValueError(f"Number of bins must be positive, got {num_bins}")
        
        self.num_bins = num_bins
        self.color_space = color_space.upper()
        
        if self.color_space not in ['HSV', 'RGB', 'LAB']:
            raise ValueError(f"Unsupported color space: {color_space}. Use 'HSV', 'RGB', or 'LAB'.")
        
        # Set up histogram parameters based on color space
        if self.color_space == 'HSV':
            # HSV ranges: H[0,180], S[0,255], V[0,255] in OpenCV
            self.ranges = [0, 180, 0, 256, 0, 256]
            self.channels = [0, 1, 2]  # Use all three channels
        elif self.color_space == 'LAB':
            # LAB ranges: L[0,255], A[0,255], B[0,255] in OpenCV
            self.ranges = [0, 256, 0, 256, 0, 256]
            self.channels = [0, 1, 2]
        else:  # RGB
            # RGB ranges: R[0,255], G[0,255], B[0,255]
            self.ranges = [0, 256, 0, 256, 0, 256]
            self.channels = [0, 1, 2]  # Use all three channels
        
        # Histogram size for each channel
        self.hist_size = [num_bins, num_bins, num_bins]
    
    def extract_histogram(self, patch: NDArray[np.uint8]) -> Histogram:
        """
        Extract color histogram from an image patch.
        
        Args:
            patch: Image patch as numpy array with shape (H, W, C) in BGR format
            
        Returns:
            Histogram: Normalized color histogram as 1D array
            
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
        elif self.color_space == 'LAB':
            # Convert from BGR to LAB
            converted_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
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
    
    def normalize_histogram(self, hist: Histogram) -> Histogram:
        """
        Normalize histogram to unit sum.
        
        Args:
            hist: Input histogram as numpy array
            
        Returns:
            Histogram: Normalized histogram with sum = 1.0
            
        Raises:
            ValueError: If histogram is invalid
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
    
    def compare_histograms(self, hist1: Histogram, hist2: Histogram, 
                          method: str = DEFAULT_HISTOGRAM_METHOD) -> float:
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
            raise ValueError(f"Histograms must have the same shape: {hist1.shape} vs {hist2.shape}")
        
        method = method.lower()
        
        if method not in HISTOGRAM_METHODS:
            raise ValueError(f"Unsupported method: {method}. Use one of: {HISTOGRAM_METHODS}")
        
        if method == 'bhattacharyya':
            return self._bhattacharyya_distance(hist1, hist2)
        elif method == 'intersection':
            return self._intersection_distance(hist1, hist2)
        elif method == 'chi_square':
            return self._chi_square_distance(hist1, hist2)
        else:
            # This should never be reached due to earlier validation
            raise ValueError(f"Unsupported method: {method}")
    
    def normalize_similarity(self, distance: float, method: str = DEFAULT_HISTOGRAM_METHOD) -> float:
        """
        Map distance to similarity score in [0,1] range.
        
        Args:
            distance: Distance value from compare_histograms
            method: Distance metric used ('bhattacharyya', 'intersection', 'chi_square')
            
        Returns:
            float: Similarity score where 1.0 = identical, 0.0 = completely different
            
        Raises:
            ValueError: If method is unsupported
        """
        method = method.lower()
        
        if method not in HISTOGRAM_METHODS:
            raise ValueError(f"Unsupported method: {method}. Use one of: {HISTOGRAM_METHODS}")
        
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
            # This should never be reached due to earlier validation
            raise ValueError(f"Unsupported method: {method}")
    
    def _bhattacharyya_distance(self, hist1: Histogram, hist2: Histogram) -> float:
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
    
    def _intersection_distance(self, hist1: Histogram, hist2: Histogram) -> float:
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
    
    def _chi_square_distance(self, hist1: Histogram, hist2: Histogram) -> float:
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
    
    def get_config(self) -> dict:
        """
        Get the current configuration of the histogram extractor.
        
        Returns:
            dict: Configuration parameters
        """
        return {
            'num_bins': self.num_bins,
            'color_space': self.color_space,
            'hist_size': self.hist_size,
            'ranges': self.ranges,
            'channels': self.channels
        }