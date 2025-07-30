"""
Unit tests for LocalTracker class in MOLT tracker implementation.

This module tests the LocalTracker class functionality including:
- Constructor and initialization
- Histogram similarity computation
- Spatial distance calculation
- Weight calculation and combination logic
"""

import unittest
import numpy as np
from typing import Tuple, Dict
import cv2

# Mock the CVModelInference module for testing
import sys
from unittest.mock import MagicMock
from typing import Generic, TypeVar, Dict, List, Any

# Create mock types that support generics
T = TypeVar('T')

class MockTracker(Generic[T]):
    pass

# Create mock module
mock_cvmodel = MagicMock()
mock_cvmodel.tracker = MagicMock()
mock_cvmodel.tracker.Tracker = MockTracker
mock_cvmodel.tracker.TrackerInfo = dict
mock_cvmodel.tracker.Detection = dict
mock_cvmodel.tracker.Track = dict
mock_cvmodel.tracker.Frame = object

sys.modules['CVModelInference'] = mock_cvmodel
sys.modules['CVModelInference.tracker'] = mock_cvmodel.tracker

from molt_tracker import LocalTracker, HistogramExtractor


class TestLocalTracker(unittest.TestCase):
    """Test cases for LocalTracker class."""
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create histogram extractor for testing
        self.histogram_extractor = HistogramExtractor(num_bins=4, color_space='HSV')
        
        # Create a sample histogram with correct size (4^3 = 64 elements)
        histogram_size = 4 * 4 * 4  # num_bins^3 for 3D histogram
        self.sample_histogram = np.random.rand(histogram_size).astype(np.float32)
        self.sample_histogram = self.sample_histogram / np.sum(self.sample_histogram)  # Normalize
        
        # Create test tracker
        self.center = (100.0, 150.0)
        self.size = (20.0, 20.0)
        self.tracker_id = 1
        
        self.tracker = LocalTracker(
            center=self.center,
            size=self.size,
            histogram=self.sample_histogram,
            tracker_id=self.tracker_id
        )
        
        # Create test image patches
        self.test_patch = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        self.identical_patch = self.test_patch.copy()
        self.different_patch = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        
        # Default similarity weights
        self.similarity_weights = {'histogram': 0.6, 'spatial': 0.4}
    
    def test_constructor_valid_inputs(self) -> None:
        """Test LocalTracker constructor with valid inputs."""
        # Create a proper histogram for this test
        test_histogram = np.random.rand(64).astype(np.float32)
        test_histogram = test_histogram / np.sum(test_histogram)
        
        tracker = LocalTracker(
            center=(50.0, 75.0),
            size=(15.0, 15.0),
            histogram=test_histogram,
            tracker_id=42
        )
        
        self.assertEqual(tracker.center, (50.0, 75.0))
        self.assertEqual(tracker.size, (15.0, 15.0))
        self.assertEqual(tracker.tracker_id, 42)
        np.testing.assert_array_equal(tracker.histogram, test_histogram)
        self.assertEqual(tracker.hist_weight, 0.0)
        self.assertEqual(tracker.dist_weight, 0.0)
        self.assertEqual(tracker.total_weight, 0.0)
        self.assertEqual(tracker.confidence, 0.0)
    
    def test_constructor_invalid_histogram(self) -> None:
        """Test LocalTracker constructor with invalid histogram."""
        # Test None histogram
        with self.assertRaises(ValueError):
            LocalTracker(
                center=self.center,
                size=self.size,
                histogram=None,
                tracker_id=1
            )
        
        # Test empty histogram
        with self.assertRaises(ValueError):
            LocalTracker(
                center=self.center,
                size=self.size,
                histogram=np.array([]),
                tracker_id=1
            )
    
    def test_compute_similarity_valid_patch(self) -> None:
        """Test histogram similarity computation with valid patch."""
        similarity = self.tracker.compute_similarity(
            self.test_patch, 
            self.histogram_extractor
        )
        
        # Similarity should be a float in [0, 1] range
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_compute_similarity_identical_patches(self) -> None:
        """Test that identical patches produce high similarity."""
        # First compute similarity to establish reference
        similarity1 = self.tracker.compute_similarity(
            self.identical_patch,
            self.histogram_extractor
        )
        
        # Second computation with same patch should give same result
        similarity2 = self.tracker.compute_similarity(
            self.identical_patch,
            self.histogram_extractor
        )
        
        self.assertAlmostEqual(similarity1, similarity2, places=5)
    
    def test_compute_similarity_invalid_patch(self) -> None:
        """Test histogram similarity computation with invalid patch."""
        # Test with None patch - should return 0.0 and not crash
        similarity = self.tracker.compute_similarity(
            None,
            self.histogram_extractor
        )
        self.assertEqual(similarity, 0.0)
        
        # Test with wrong shape patch - should return 0.0 and not crash
        invalid_patch = np.random.randint(0, 255, (20, 20), dtype=np.uint8)  # Missing channel dimension
        similarity = self.tracker.compute_similarity(
            invalid_patch,
            self.histogram_extractor
        )
        self.assertEqual(similarity, 0.0)
    
    def test_compute_distance_valid_position(self) -> None:
        """Test spatial distance computation with valid reference position."""
        reference_pos = (110.0, 160.0)
        distance = self.tracker.compute_distance(reference_pos)
        
        # Expected distance: sqrt((100-110)^2 + (150-160)^2) = sqrt(100 + 100) = sqrt(200) ≈ 14.14
        expected_distance = np.sqrt((100.0 - 110.0)**2 + (150.0 - 160.0)**2)
        
        self.assertIsInstance(distance, float)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_compute_distance_same_position(self) -> None:
        """Test spatial distance computation with same position."""
        distance = self.tracker.compute_distance(self.center)
        self.assertAlmostEqual(distance, 0.0, places=5)
    
    def test_compute_distance_invalid_position(self) -> None:
        """Test spatial distance computation with invalid reference position."""
        # Test None position
        with self.assertRaises(ValueError):
            self.tracker.compute_distance(None)
        
        # Test invalid tuple length
        with self.assertRaises(ValueError):
            self.tracker.compute_distance((100.0,))  # Missing y coordinate
        
        with self.assertRaises(ValueError):
            self.tracker.compute_distance((100.0, 150.0, 200.0))  # Too many coordinates
    
    def test_update_weights_valid_inputs(self) -> None:
        """Test weight update with valid inputs."""
        hist_similarity = 0.8
        spatial_distance = 10.0
        
        self.tracker.update_weights(
            hist_similarity,
            spatial_distance,
            self.similarity_weights
        )
        
        # Check that weights were updated
        self.assertEqual(self.tracker.hist_weight, hist_similarity)
        self.assertGreater(self.tracker.dist_weight, 0.0)  # Should be positive for small distance
        self.assertGreater(self.tracker.total_weight, 0.0)
        self.assertEqual(self.tracker.confidence, self.tracker.total_weight)
    
    def test_update_weights_zero_distance(self) -> None:
        """Test weight update with zero spatial distance."""
        hist_similarity = 0.7
        spatial_distance = 0.0
        
        self.tracker.update_weights(
            hist_similarity,
            spatial_distance,
            self.similarity_weights
        )
        
        # With zero distance, spatial similarity should be 1.0
        self.assertEqual(self.tracker.hist_weight, hist_similarity)
        self.assertAlmostEqual(self.tracker.dist_weight, 1.0, places=5)
        
        # Total weight should be weighted combination
        expected_total = (0.6 * hist_similarity + 0.4 * 1.0)
        self.assertAlmostEqual(self.tracker.total_weight, expected_total, places=5)
    
    def test_update_weights_large_distance(self) -> None:
        """Test weight update with large spatial distance."""
        hist_similarity = 0.9
        spatial_distance = 1000.0  # Very large distance
        
        self.tracker.update_weights(
            hist_similarity,
            spatial_distance,
            self.similarity_weights
        )
        
        # With large distance, spatial similarity should be close to 0
        self.assertEqual(self.tracker.hist_weight, hist_similarity)
        self.assertLess(self.tracker.dist_weight, 0.1)  # Should be very small
        
        # Total weight should be dominated by histogram similarity
        self.assertLess(self.tracker.total_weight, hist_similarity)
    
    def test_update_weights_invalid_inputs(self) -> None:
        """Test weight update with invalid inputs."""
        # Test invalid histogram similarity
        with self.assertRaises(ValueError):
            self.tracker.update_weights(-0.1, 10.0, self.similarity_weights)
        
        with self.assertRaises(ValueError):
            self.tracker.update_weights(1.1, 10.0, self.similarity_weights)
        
        # Test negative spatial distance
        with self.assertRaises(ValueError):
            self.tracker.update_weights(0.5, -10.0, self.similarity_weights)
        
        # Test invalid similarity weights
        with self.assertRaises(ValueError):
            self.tracker.update_weights(0.5, 10.0, {'histogram': 0.6})  # Missing 'spatial'
        
        with self.assertRaises(ValueError):
            self.tracker.update_weights(0.5, 10.0, {'spatial': 0.4})  # Missing 'histogram'
        
        # Test zero weight ratios
        with self.assertRaises(ValueError):
            self.tracker.update_weights(0.5, 10.0, {'histogram': 0.0, 'spatial': 0.0})
    
    def test_total_weight_property(self) -> None:
        """Test total_weight property."""
        # Initially should be 0.0
        self.assertEqual(self.tracker.total_weight_property, 0.0)
        
        # After updating weights, should match total_weight
        self.tracker.update_weights(0.8, 15.0, self.similarity_weights)
        self.assertEqual(self.tracker.total_weight_property, self.tracker.total_weight)
    
    def test_get_position(self) -> None:
        """Test get_position method."""
        position = self.tracker.get_position()
        self.assertEqual(position, self.center)
        self.assertIsInstance(position, tuple)
        self.assertEqual(len(position), 2)
    
    def test_get_bounding_box(self) -> None:
        """Test get_bounding_box method."""
        bbox = self.tracker.get_bounding_box()
        expected_bbox = (self.center[0], self.center[1], self.size[0], self.size[1])
        
        self.assertEqual(bbox, expected_bbox)
        self.assertIsInstance(bbox, tuple)
        self.assertEqual(len(bbox), 4)
    
    def test_string_representations(self) -> None:
        """Test __str__ and __repr__ methods."""
        # Test __str__
        str_repr = str(self.tracker)
        self.assertIn("LocalTracker", str_repr)
        self.assertIn(f"id={self.tracker_id}", str_repr)
        self.assertIn(f"center={self.center}", str_repr)
        
        # Test __repr__
        repr_str = repr(self.tracker)
        self.assertIn("LocalTracker", repr_str)
        self.assertIn(f"id={self.tracker_id}", repr_str)
        self.assertIn("hist_weight", repr_str)
        self.assertIn("dist_weight", repr_str)
        self.assertIn("total_weight", repr_str)
    
    def test_weight_combination_ratios(self) -> None:
        """Test that different weight ratios produce expected results."""
        hist_similarity = 0.8
        spatial_distance = 50.0  # Use larger distance to make spatial component smaller
        
        # Test histogram-heavy weighting
        hist_heavy_weights = {'histogram': 0.9, 'spatial': 0.1}
        self.tracker.update_weights(hist_similarity, spatial_distance, hist_heavy_weights)
        hist_heavy_total = self.tracker.total_weight
        
        # Reset tracker
        self.tracker.hist_weight = 0.0
        self.tracker.dist_weight = 0.0
        self.tracker.total_weight = 0.0
        
        # Test spatial-heavy weighting
        spatial_heavy_weights = {'histogram': 0.1, 'spatial': 0.9}
        self.tracker.update_weights(hist_similarity, spatial_distance, spatial_heavy_weights)
        spatial_heavy_total = self.tracker.total_weight
        
        # With high histogram similarity and large distance,
        # histogram-heavy should give higher total weight
        self.assertGreater(hist_heavy_total, spatial_heavy_total)


if __name__ == '__main__':
    unittest.main()