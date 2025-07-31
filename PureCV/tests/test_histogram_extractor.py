#!/usr/bin/env python3
"""
Unit tests for HistogramExtractor class.

This module tests the histogram extraction and comparison functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
from PureCV.molt.histogram_extractor import HistogramExtractor


def create_test_patch(color=(255, 0, 0), size=(20, 20)):
    """Create a test image patch with a solid color."""
    patch = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return patch


def test_histogram_extractor_initialization():
    """Test HistogramExtractor initialization."""
    # Test default initialization
    extractor = HistogramExtractor()
    assert extractor.num_bins == 16
    assert extractor.color_space == 'HSV'
    
    # Test custom initialization
    extractor = HistogramExtractor(num_bins=32, color_space='RGB')
    assert extractor.num_bins == 32
    assert extractor.color_space == 'RGB'
    
    # Test invalid color space
    try:
        HistogramExtractor(color_space='INVALID')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test invalid num_bins
    try:
        HistogramExtractor(num_bins=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_histogram_extraction():
    """Test histogram extraction from image patches."""
    extractor = HistogramExtractor(num_bins=8, color_space='HSV')
    
    # Create test patches
    red_patch = create_test_patch((0, 0, 255))  # Red in BGR
    blue_patch = create_test_patch((255, 0, 0))  # Blue in BGR
    
    # Extract histograms
    red_hist = extractor.extract_histogram(red_patch)
    blue_hist = extractor.extract_histogram(blue_patch)
    
    # Check histogram properties
    assert red_hist is not None
    assert blue_hist is not None
    assert red_hist.shape == (8 * 8 * 8,)  # Flattened 3D histogram
    assert blue_hist.shape == (8 * 8 * 8,)
    
    # Check normalization (should sum to 1)
    assert abs(np.sum(red_hist) - 1.0) < 1e-6
    assert abs(np.sum(blue_hist) - 1.0) < 1e-6
    
    # Histograms of different colors should be different
    assert not np.allclose(red_hist, blue_hist)


def test_histogram_extraction_edge_cases():
    """Test histogram extraction with edge cases."""
    extractor = HistogramExtractor()
    
    # Test with None patch
    try:
        extractor.extract_histogram(None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with empty patch
    try:
        extractor.extract_histogram(np.array([]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with invalid shape
    try:
        extractor.extract_histogram(np.zeros((10, 10)))  # Missing channel dimension
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_histogram_comparison():
    """Test histogram comparison methods."""
    extractor = HistogramExtractor(num_bins=8)
    
    # Create identical patches
    patch1 = create_test_patch((0, 0, 255))
    patch2 = create_test_patch((0, 0, 255))
    
    # Create different patch
    patch3 = create_test_patch((255, 0, 0))
    
    # Extract histograms
    hist1 = extractor.extract_histogram(patch1)
    hist2 = extractor.extract_histogram(patch2)
    hist3 = extractor.extract_histogram(patch3)
    
    # Test Bhattacharyya distance
    distance_same = extractor.compare_histograms(hist1, hist2, 'bhattacharyya')
    distance_diff = extractor.compare_histograms(hist1, hist3, 'bhattacharyya')
    
    # Identical histograms should have very small distance
    assert distance_same < 0.1
    # Different histograms should have larger distance
    assert distance_diff > distance_same
    
    # Test intersection distance
    distance_same_int = extractor.compare_histograms(hist1, hist2, 'intersection')
    distance_diff_int = extractor.compare_histograms(hist1, hist3, 'intersection')
    
    assert distance_same_int < 0.1
    assert distance_diff_int > distance_same_int
    
    # Test chi-square distance
    distance_same_chi = extractor.compare_histograms(hist1, hist2, 'chi_square')
    distance_diff_chi = extractor.compare_histograms(hist1, hist3, 'chi_square')
    
    assert distance_same_chi < 0.1
    assert distance_diff_chi > distance_same_chi


def test_similarity_normalization():
    """Test similarity score normalization."""
    extractor = HistogramExtractor()
    
    # Test Bhattacharyya normalization
    similarity = extractor.normalize_similarity(0.0, 'bhattacharyya')
    assert abs(similarity - 1.0) < 1e-6  # Distance 0 should give similarity 1
    
    similarity = extractor.normalize_similarity(float('inf'), 'bhattacharyya')
    assert abs(similarity - 0.0) < 1e-6  # Infinite distance should give similarity 0
    
    # Test intersection normalization
    similarity = extractor.normalize_similarity(0.0, 'intersection')
    assert abs(similarity - 1.0) < 1e-6
    
    similarity = extractor.normalize_similarity(1.0, 'intersection')
    assert abs(similarity - 0.0) < 1e-6
    
    # Test invalid method
    try:
        extractor.normalize_similarity(0.5, 'invalid_method')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_histogram_normalization():
    """Test histogram normalization."""
    extractor = HistogramExtractor()
    
    # Create unnormalized histogram
    hist = np.array([1, 2, 3, 4], dtype=np.float32)
    normalized = extractor.normalize_histogram(hist)
    
    # Check normalization
    assert abs(np.sum(normalized) - 1.0) < 1e-6
    assert np.allclose(normalized, hist / np.sum(hist))
    
    # Test with zero histogram
    zero_hist = np.zeros(4, dtype=np.float32)
    normalized_zero = extractor.normalize_histogram(zero_hist)
    
    # Should return uniform distribution
    assert abs(np.sum(normalized_zero) - 1.0) < 1e-6
    assert np.allclose(normalized_zero, np.ones(4) / 4)


def test_get_config():
    """Test getting extractor configuration."""
    extractor = HistogramExtractor(num_bins=16, color_space='HSV')
    config = extractor.get_config()
    
    assert config['num_bins'] == 16
    assert config['color_space'] == 'HSV'
    assert 'hist_size' in config
    assert 'ranges' in config
    assert 'channels' in config


if __name__ == "__main__":
    test_histogram_extractor_initialization()
    test_histogram_extraction()
    test_histogram_extraction_edge_cases()
    test_histogram_comparison()
    test_similarity_normalization()
    test_histogram_normalization()
    test_get_config()
    print("✓ All histogram extractor tests passed!")