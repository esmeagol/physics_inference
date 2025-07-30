"""
Unit tests for HistogramExtractor class.

Tests histogram extraction, normalization, and comparison methods
to ensure correct functionality for the MOLT tracker.
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add the current directory to the path to import the HistogramExtractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the HistogramExtractor class to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location("molt_tracker", "molt_tracker.py")
if spec is not None:
    molt_tracker_module = importlib.util.module_from_spec(spec)
else:
    raise ImportError("Could not load molt_tracker module")

# Mock the missing dependencies
class MockTracker:
    pass

class MockTrackerInfo:
    pass

class MockDetection:
    pass

class MockTrack:
    pass

class MockFrame:
    pass

# Add mocks to sys.modules
sys.modules['CVModelInference'] = type(sys)('CVModelInference')
sys.modules['CVModelInference.tracker'] = type(sys)('tracker')
sys.modules['CVModelInference.tracker'].Tracker = MockTracker
sys.modules['CVModelInference.tracker'].TrackerInfo = MockTrackerInfo
sys.modules['CVModelInference.tracker'].Detection = MockDetection
sys.modules['CVModelInference.tracker'].Track = MockTrack
sys.modules['CVModelInference.tracker'].Frame = MockFrame

# Now import the module
if spec is not None and spec.loader is not None:
    spec.loader.exec_module(molt_tracker_module)
HistogramExtractor = molt_tracker_module.HistogramExtractor


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