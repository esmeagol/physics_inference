#!/usr/bin/env python3
"""
Integration tests for the reorganized MOLT tracker.

This module tests that all components work together correctly
after the code reorganization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
from typing import List, Dict, Any

from PureCV.molt import MOLTTracker, MOLTTrackerConfig
from PureCV.molt.histogram_extractor import HistogramExtractor
from PureCV.molt.ball_count_manager import BallCountManager
from PureCV.molt.population import TrackerPopulation
from PureCV.molt.local_tracker import LocalTracker


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with colored regions for testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored regions to simulate balls
    cv2.circle(frame, (100, 100), 15, (0, 0, 255), -1)  # Red ball
    cv2.circle(frame, (200, 150), 15, (255, 255, 255), -1)  # White ball
    cv2.circle(frame, (300, 200), 15, (0, 255, 255), -1)  # Yellow ball
    
    return frame


def create_test_detections() -> List[Dict[str, Any]]:
    """Create test detections for initialization."""
    return [
        {'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'red', 'confidence': 0.9},
        {'x': 200, 'y': 150, 'width': 30, 'height': 30, 'class': 'white', 'confidence': 0.95},
        {'x': 300, 'y': 200, 'width': 30, 'height': 30, 'class': 'yellow', 'confidence': 0.85}
    ]


def test_component_integration():
    """Test that all components work together."""
    print("Testing component integration...")
    
    # Test HistogramExtractor
    extractor = HistogramExtractor(num_bins=8, color_space='HSV')
    frame = create_test_frame()
    patch = frame[85:115, 85:115]  # Extract patch around red ball
    histogram = extractor.extract_histogram(patch)
    assert histogram is not None
    assert histogram.shape == (8 * 8 * 8,)
    
    # Test BallCountManager
    expected_counts = {'red': 1, 'white': 1, 'yellow': 1}
    ball_manager = BallCountManager(expected_counts)
    assert ball_manager.expected_counts == expected_counts
    
    # Test LocalTracker
    tracker = LocalTracker(
        center=(100, 100),
        size=(30, 30),
        histogram=histogram,
        tracker_id=1
    )
    assert tracker.center == (100, 100)
    assert tracker.is_valid()
    
    # Test TrackerPopulation
    population = TrackerPopulation(
        object_id=1,
        object_class='red',
        population_size=50,
        initial_center=(100, 100),
        initial_size=(30, 30),
        reference_histogram=histogram
    )
    assert len(population.trackers) == 50
    assert population.object_class == 'red'
    
    print("✓ Component integration test passed!")


def test_molt_tracker_with_config():
    """Test MOLTTracker with custom configuration."""
    print("Testing MOLT tracker with custom configuration...")
    
    # Create custom configuration
    config = MOLTTrackerConfig.create_for_snooker()
    config.histogram_bins = 8  # Smaller for faster testing
    config.population_sizes = {'red': 50, 'white': 50, 'yellow': 50}
    
    # Create tracker with config
    tracker = MOLTTracker(config=config)
    
    # Test initialization
    frame = create_test_frame()
    detections = create_test_detections()
    success = tracker.init(frame, detections)
    
    assert success
    assert tracker.is_initialized
    assert len(tracker.populations) == 3
    assert tracker.histogram_bins == 8
    
    # Test tracker info
    info = tracker.get_tracker_info()
    assert info['name'] == 'MOLT'
    assert info['parameters']['histogram_bins'] == 8
    
    print("✓ MOLT tracker with config test passed!")


def test_molt_tracker_with_kwargs():
    """Test MOLTTracker with kwargs override."""
    print("Testing MOLT tracker with kwargs override...")
    
    # Create tracker with kwargs
    tracker = MOLTTracker(
        histogram_bins=12,
        color_space='RGB',
        min_confidence=0.2
    )
    
    # Verify overrides
    assert tracker.histogram_bins == 12
    assert tracker.color_space == 'RGB'
    assert tracker.min_confidence == 0.2
    
    # Test initialization
    frame = create_test_frame()
    detections = create_test_detections()
    success = tracker.init(frame, detections)
    
    assert success
    assert tracker.is_initialized
    
    print("✓ MOLT tracker with kwargs test passed!")


def test_visualization():
    """Test tracker visualization functionality."""
    print("Testing tracker visualization...")
    
    tracker = MOLTTracker()
    frame = create_test_frame()
    detections = create_test_detections()
    
    # Initialize tracker
    tracker.init(frame, detections)
    
    # Create mock tracks for visualization
    tracks = [
        {'id': 1, 'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'red', 'confidence': 0.9},
        {'id': 2, 'x': 200, 'y': 150, 'width': 30, 'height': 30, 'class': 'white', 'confidence': 0.95},
        {'id': 3, 'x': 300, 'y': 200, 'width': 30, 'height': 30, 'class': 'yellow', 'confidence': 0.85}
    ]
    
    # Test visualization
    vis_frame = tracker.visualize(frame, tracks)
    
    # Check that visualization frame has same shape as input
    assert vis_frame.shape == frame.shape
    
    # Check that visualization frame is different from input (has drawings)
    assert not np.array_equal(vis_frame, frame)
    
    print("✓ Tracker visualization test passed!")


def test_population_statistics():
    """Test population statistics functionality."""
    print("Testing population statistics...")
    
    tracker = MOLTTracker()
    frame = create_test_frame()
    detections = create_test_detections()
    
    # Initialize tracker
    tracker.init(frame, detections)
    
    # Get population statistics
    stats = tracker.get_population_statistics()
    
    assert stats['total_populations'] == 3
    assert len(stats['populations']) == 3
    assert 'ball_counts' in stats
    
    # Check individual population stats
    for pop_stats in stats['populations']:
        assert 'object_id' in pop_stats
        assert 'object_class' in pop_stats
        assert 'population_size' in pop_stats
        assert pop_stats['population_size'] > 0
    
    print("✓ Population statistics test passed!")


def test_error_handling():
    """Test error handling in integrated system."""
    print("Testing error handling...")
    
    tracker = MOLTTracker()
    
    # Test initialization with invalid inputs
    assert not tracker.init(None, [])  # None frame
    assert not tracker.init(create_test_frame(), [])  # Empty detections
    
    # Test update without initialization
    tracks = tracker.update(create_test_frame())
    assert tracks == []
    
    # Test reset functionality
    frame = create_test_frame()
    detections = create_test_detections()
    tracker.init(frame, detections)
    assert tracker.is_initialized
    
    tracker.reset()
    assert not tracker.is_initialized
    assert len(tracker.populations) == 0
    
    print("✓ Error handling test passed!")


if __name__ == "__main__":
    try:
        test_component_integration()
        test_molt_tracker_with_config()
        test_molt_tracker_with_kwargs()
        test_visualization()
        test_population_statistics()
        test_error_handling()
        print("\n🎉 All integration tests passed!")
        print("✅ Code reorganization successful!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()