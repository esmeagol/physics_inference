"""
Standalone unit tests for HistogramExtractor class.

Tests histogram extraction, normalization, and comparison methods
to ensure correct functionality for the MOLT tracker.
"""

import unittest
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from numpy.typing import NDArray


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
            return 1.0 - distance
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


class TestHistogramExtractor(unittest.TestCase):
    """Test cases for HistogramExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor_hsv = HistogramExtractor(num_bins=8, color_space='HSV')
        self.extractor_rgb = HistogramExtractor(num_bins=8, color_space='RGB')
        
        # Create test images
        self.test_patch_red = self._create_solid_color_patch((0, 0, 255), 32, 32)  # Red in BGR
        self.test_patch_blue = self._create_solid_color_patch((255, 0, 0), 32, 32)  # Blue in BGR
        self.test_patch_green = self._create_solid_color_patch((0, 255, 0), 32, 32)  # Green in BGR
        self.test_patch_identical = self._create_solid_color_patch((0, 0, 255), 32, 32)  # Identical to red
    
    def _create_solid_color_patch(self, color, width, height):
        """Create a solid color patch for testing."""
        patch = np.full((height, width, 3), color, dtype=np.uint8)
        return patch
    
    def test_init_valid_color_spaces(self):
        """Test initialization with valid color spaces."""
        extractor_hsv = HistogramExtractor(num_bins=16, color_space='HSV')
        self.assertEqual(extractor_hsv.color_space, 'HSV')
        self.assertEqual(extractor_hsv.num_bins, 16)
        
        extractor_rgb = HistogramExtractor(num_bins=16, color_space='RGB')
        self.assertEqual(extractor_rgb.color_space, 'RGB')
        self.assertEqual(extractor_rgb.num_bins, 16)
    
    def test_init_invalid_color_space(self):
        """Test initialization with invalid color space."""
        with self.assertRaises(ValueError):
            HistogramExtractor(num_bins=16, color_space='INVALID')
    
    def test_extract_histogram_valid_patch(self):
        """Test histogram extraction with valid patches."""
        # Test HSV extraction
        hist_hsv = self.extractor_hsv.extract_histogram(self.test_patch_red)
        self.assertIsInstance(hist_hsv, np.ndarray)
        self.assertEqual(hist_hsv.dtype, np.float32)
        self.assertEqual(len(hist_hsv.shape), 1)  # Should be flattened
        self.assertEqual(hist_hsv.shape[0], 8 * 8 * 8)  # 8 bins per channel, 3 channels
        
        # Test RGB extraction
        hist_rgb = self.extractor_rgb.extract_histogram(self.test_patch_red)
        self.assertIsInstance(hist_rgb, np.ndarray)
        self.assertEqual(hist_rgb.dtype, np.float32)
        self.assertEqual(len(hist_rgb.shape), 1)  # Should be flattened
        self.assertEqual(hist_rgb.shape[0], 8 * 8 * 8)  # 8 bins per channel, 3 channels
    
    def test_extract_histogram_invalid_patch(self):
        """Test histogram extraction with invalid patches."""
        # Test None patch
        with self.assertRaises(ValueError):
            self.extractor_hsv.extract_histogram(None)
        
        # Test empty patch
        with self.assertRaises(ValueError):
            self.extractor_hsv.extract_histogram(np.array([]))
        
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            self.extractor_hsv.extract_histogram(np.zeros((32, 32)))  # Missing channel dimension
        
        # Test wrong number of channels
        with self.assertRaises(ValueError):
            self.extractor_hsv.extract_histogram(np.zeros((32, 32, 4)))  # 4 channels instead of 3
    
    def test_normalize_histogram(self):
        """Test histogram normalization."""
        # Create test histogram
        test_hist = np.array([1, 2, 3, 4], dtype=np.float32)
        normalized = self.extractor_hsv.normalize_histogram(test_hist)
        
        # Check normalization
        self.assertAlmostEqual(np.sum(normalized), 1.0, places=6)
        self.assertEqual(normalized.dtype, np.float32)
        
        # Test with zero histogram
        zero_hist = np.zeros(4, dtype=np.float32)
        normalized_zero = self.extractor_hsv.normalize_histogram(zero_hist)
        self.assertAlmostEqual(np.sum(normalized_zero), 1.0, places=6)
        
        # Test with invalid input
        with self.assertRaises(ValueError):
            self.extractor_hsv.normalize_histogram(None)
        
        with self.assertRaises(ValueError):
            self.extractor_hsv.normalize_histogram(np.array([]))
    
    def test_compare_histograms_bhattacharyya(self):
        """Test Bhattacharyya distance comparison."""
        hist1 = self.extractor_hsv.extract_histogram(self.test_patch_red)
        hist2 = self.extractor_hsv.extract_histogram(self.test_patch_blue)
        hist3 = self.extractor_hsv.extract_histogram(self.test_patch_identical)
        
        # Test different histograms
        distance_different = self.extractor_hsv.compare_histograms(hist1, hist2, 'bhattacharyya')
        self.assertIsInstance(distance_different, float)
        self.assertGreater(distance_different, 0)
        
        # Test identical histograms
        distance_identical = self.extractor_hsv.compare_histograms(hist1, hist3, 'bhattacharyya')
        self.assertAlmostEqual(distance_identical, 0.0, places=5)
        
        # Test self-comparison
        distance_self = self.extractor_hsv.compare_histograms(hist1, hist1, 'bhattacharyya')
        self.assertAlmostEqual(distance_self, 0.0, places=5)
    
    def test_compare_histograms_intersection(self):
        """Test intersection distance comparison."""
        hist1 = self.extractor_hsv.extract_histogram(self.test_patch_red)
        hist2 = self.extractor_hsv.extract_histogram(self.test_patch_blue)
        hist3 = self.extractor_hsv.extract_histogram(self.test_patch_identical)
        
        # Test different histograms
        distance_different = self.extractor_hsv.compare_histograms(hist1, hist2, 'intersection')
        self.assertIsInstance(distance_different, float)
        self.assertGreater(distance_different, 0)
        self.assertLessEqual(distance_different, 1.0)
        
        # Test identical histograms
        distance_identical = self.extractor_hsv.compare_histograms(hist1, hist3, 'intersection')
        self.assertAlmostEqual(distance_identical, 0.0, places=5)
        
        # Test self-comparison
        distance_self = self.extractor_hsv.compare_histograms(hist1, hist1, 'intersection')
        self.assertAlmostEqual(distance_self, 0.0, places=5)
    
    def test_compare_histograms_chi_square(self):
        """Test Chi-square distance comparison."""
        hist1 = self.extractor_hsv.extract_histogram(self.test_patch_red)
        hist2 = self.extractor_hsv.extract_histogram(self.test_patch_blue)
        hist3 = self.extractor_hsv.extract_histogram(self.test_patch_identical)
        
        # Test different histograms
        distance_different = self.extractor_hsv.compare_histograms(hist1, hist2, 'chi_square')
        self.assertIsInstance(distance_different, float)
        self.assertGreater(distance_different, 0)
        
        # Test identical histograms
        distance_identical = self.extractor_hsv.compare_histograms(hist1, hist3, 'chi_square')
        self.assertAlmostEqual(distance_identical, 0.0, places=5)
        
        # Test self-comparison
        distance_self = self.extractor_hsv.compare_histograms(hist1, hist1, 'chi_square')
        self.assertAlmostEqual(distance_self, 0.0, places=5)
    
    def test_compare_histograms_invalid_inputs(self):
        """Test histogram comparison with invalid inputs."""
        hist1 = self.extractor_hsv.extract_histogram(self.test_patch_red)
        hist2 = self.extractor_hsv.extract_histogram(self.test_patch_blue)
        
        # Test None inputs
        with self.assertRaises(ValueError):
            self.extractor_hsv.compare_histograms(None, hist2, 'bhattacharyya')
        
        with self.assertRaises(ValueError):
            self.extractor_hsv.compare_histograms(hist1, None, 'bhattacharyya')
        
        # Test mismatched shapes
        hist_wrong_shape = np.zeros(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.extractor_hsv.compare_histograms(hist1, hist_wrong_shape, 'bhattacharyya')
        
        # Test invalid method
        with self.assertRaises(ValueError):
            self.extractor_hsv.compare_histograms(hist1, hist2, 'invalid_method')
    
    def test_normalize_similarity(self):
        """Test similarity normalization."""
        # Test Bhattacharyya normalization
        distance = 0.5
        similarity = self.extractor_hsv.normalize_similarity(distance, 'bhattacharyya')
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        self.assertAlmostEqual(similarity, np.exp(-0.5), places=6)
        
        # Test intersection normalization
        distance = 0.3
        similarity = self.extractor_hsv.normalize_similarity(distance, 'intersection')
        self.assertAlmostEqual(similarity, 0.7, places=6)
        
        # Test chi-square normalization
        distance = 1.0
        similarity = self.extractor_hsv.normalize_similarity(distance, 'chi_square')
        self.assertAlmostEqual(similarity, np.exp(-0.5), places=6)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            self.extractor_hsv.normalize_similarity(0.5, 'invalid_method')
    
    def test_histogram_consistency_across_color_spaces(self):
        """Test that histograms are consistent across different color spaces."""
        # Extract histograms in both color spaces
        hist_hsv = self.extractor_hsv.extract_histogram(self.test_patch_red)
        hist_rgb = self.extractor_rgb.extract_histogram(self.test_patch_red)
        
        # Both should be normalized
        self.assertAlmostEqual(np.sum(hist_hsv), 1.0, places=5)
        self.assertAlmostEqual(np.sum(hist_rgb), 1.0, places=5)
        
        # Both should have the same shape
        self.assertEqual(hist_hsv.shape, hist_rgb.shape)
    
    def test_distance_properties(self):
        """Test mathematical properties of distance metrics."""
        hist1 = self.extractor_hsv.extract_histogram(self.test_patch_red)
        hist2 = self.extractor_hsv.extract_histogram(self.test_patch_blue)
        hist3 = self.extractor_hsv.extract_histogram(self.test_patch_green)
        
        for method in ['bhattacharyya', 'intersection', 'chi_square']:
            # Test symmetry: d(A,B) = d(B,A)
            dist_ab = self.extractor_hsv.compare_histograms(hist1, hist2, method)
            dist_ba = self.extractor_hsv.compare_histograms(hist2, hist1, method)
            self.assertAlmostEqual(dist_ab, dist_ba, places=6)
            
            # Test identity: d(A,A) = 0
            dist_aa = self.extractor_hsv.compare_histograms(hist1, hist1, method)
            self.assertAlmostEqual(dist_aa, 0.0, places=5)
            
            # Test non-negativity: d(A,B) >= 0
            self.assertGreaterEqual(dist_ab, 0.0)


if __name__ == '__main__':
    unittest.main()