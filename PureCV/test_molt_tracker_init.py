#!/usr/bin/env python3
"""
Integration tests for MOLT tracker initialization.

This module tests the initialization process of the MOLT tracker,
including population creation and histogram extraction.
"""

import numpy as np
import cv2
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PureCV.molt import MOLTTracker


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a test frame with colored regions for testing.
    
    Args:
        width: Frame width
        height: Frame height
        
    Returns:
        np.ndarray: Test frame in BGR format
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored regions to simulate balls
    # Red ball at (100, 100)
    cv2.circle(frame, (100, 100), 15, (0, 0, 255), -1)
    
    # White ball at (200, 150)
    cv2.circle(frame, (200, 150), 15, (255, 255, 255), -1)
    
    # Yellow ball at (300, 200)
    cv2.circle(frame, (300, 200), 15, (0, 255, 255), -1)
    
    return frame


def create_test_detections() -> List[Dict[str, Any]]:
    """
    Create test detections for initialization.
    
    Returns:
        List[Dict[str, Any]]: List of test detections
    """
    return [
        {
            'x': 100,
            'y': 100,
            'width': 30,
            'height': 30,
            'class': 'red',
            'confidence': 0.9
        },
        {
            'x': 200,
            'y': 150,
            'width': 30,
            'height': 30,
            'class': 'white',
            'confidence': 0.95
        },
        {
            'x': 300,
            'y': 200,
            'width': 30,
            'height': 30,
            'class': 'yellow',
            'confidence': 0.85
        }
    ]


def test_molt_tracker_initialization():
    """Test MOLT tracker initialization with test data."""
    print("Testing MOLT tracker initialization...")
    
    # Create tracker with default configuration
    tracker = MOLTTracker()
    
    # Create test frame and detections
    frame = create_test_frame()
    detections = create_test_detections()
    
    # Test initialization
    success = tracker.init(frame, detections)
    
    # Verify initialization
    assert success, "Tracker initialization should succeed"
    assert tracker.is_initialized, "Tracker should be marked as initialized"
    assert len(tracker.populations) == 3, f"Expected 3 populations, got {len(tracker.populations)}"
    assert tracker.frame_count == 1, f"Expected frame_count=1, got {tracker.frame_count}"
    assert tracker.next_track_id == 4, f"Expected next_track_id=4, got {tracker.next_track_id}"
    
    # Verify ball counts
    expected_ball_counts = {'red': 1, 'white': 1, 'yellow': 1}
    for ball_class, expected_count in expected_ball_counts.items():
        actual_count = tracker.ball_counts.get(ball_class, 0)
        assert actual_count == expected_count, f"Expected {expected_count} {ball_class} balls, got {actual_count}"
    
    # Verify populations
    for i, population in enumerate(tracker.populations):
        assert population is not None, f"Population {i} should not be None"
        assert population.object_id == i + 1, f"Population {i} should have object_id {i + 1}"
        assert len(population.trackers) > 0, f"Population {i} should have trackers"
        assert population.reference_histogram is not None, f"Population {i} should have reference histogram"
        assert population.reference_histogram.size > 0, f"Population {i} histogram should not be empty"
    
    print("✓ MOLT tracker initialization test passed!")


def test_molt_tracker_initialization_edge_cases():
    """Test MOLT tracker initialization with edge cases."""
    print("Testing MOLT tracker initialization edge cases...")
    
    tracker = MOLTTracker()
    
    # Test with empty detections
    frame = create_test_frame()
    success = tracker.init(frame, [])
    assert not success, "Initialization should fail with empty detections"
    
    # Test with invalid frame
    detections = create_test_detections()
    success = tracker.init(None, detections)
    assert not success, "Initialization should fail with None frame"
    
    # Test with invalid detection dimensions
    invalid_detections = [
        {
            'x': 100,
            'y': 100,
            'width': 0,  # Invalid width
            'height': 30,
            'class': 'red',
            'confidence': 0.9
        }
    ]
    success = tracker.init(frame, invalid_detections)
    # Should still succeed but skip invalid detections
    assert success, "Initialization should succeed but skip invalid detections"
    assert len(tracker.populations) == 0, "Should have no populations for invalid detections"
    
    print("✓ MOLT tracker initialization edge cases test passed!")


def test_tracker_info():
    """Test tracker info functionality."""
    print("Testing tracker info...")
    
    tracker = MOLTTracker()
    frame = create_test_frame()
    detections = create_test_detections()
    
    # Initialize tracker
    tracker.init(frame, detections)
    
    # Get tracker info
    info = tracker.get_tracker_info()
    
    # Verify info structure
    assert info['name'] == 'MOLT', f"Expected name 'MOLT', got {info['name']}"
    assert info['type'] == 'Multiple Object Local Tracker', f"Unexpected type: {info['type']}"
    assert 'parameters' in info, "Info should contain parameters"
    assert info['frame_count'] == 1, f"Expected frame_count=1, got {info['frame_count']}"
    assert info['active_tracks'] == 3, f"Expected 3 active tracks, got {info['active_tracks']}"
    
    print("✓ Tracker info test passed!")


if __name__ == "__main__":
    try:
        test_molt_tracker_initialization()
        test_molt_tracker_initialization_edge_cases()
        test_tracker_info()
        print("\n🎉 All MOLT tracker initialization tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()