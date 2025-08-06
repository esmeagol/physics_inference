#!/usr/bin/env python3
"""
Test script for Task 8.2: Tracker information and statistics methods.

This script tests the enhanced tracker information methods including
get_tracker_info, get_performance_metrics, get_tracking_status, and reset.
"""

import numpy as np
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, cast
from numpy.typing import NDArray

# Add PureCV to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.trackers.molt import MOLTTracker, MOLTTrackerConfig


def create_test_frame(width: int = 640, height: int = 480) -> NDArray[np.uint8]:
    """Create a test frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_test_detections() -> List[Dict[str, Any]]:
    """Create test detections for initialization."""
    return [
        {
            'x': 100, 'y': 100, 'width': 30, 'height': 30,
            'class': 'red', 'class_id': 1, 'confidence': 0.9
        },
        {
            'x': 200, 'y': 150, 'width': 30, 'height': 30,
            'class': 'white', 'class_id': 0, 'confidence': 0.95
        }
    ]


def test_tracker_info() -> None:
    """Test the get_tracker_info method."""
    print("Testing get_tracker_info method...")
    
    config = MOLTTrackerConfig.create_for_snooker()
    tracker = MOLTTracker(config=config)
    
    # Test info before initialization
    info = tracker.get_tracker_info()
    
    assert info['name'] == 'MOLT', "Tracker name should be MOLT"
    assert info['type'] == 'Multiple Object Local Tracker', "Tracker type incorrect"
    assert info['frame_count'] == 0, "Frame count should be 0 before initialization"
    assert info['active_tracks'] == 0, "Active tracks should be 0 before initialization"
    
    print("  ✅ Pre-initialization info correct")
    
    # Initialize tracker
    frame = create_test_frame()
    detections = create_test_detections()
    
    if tracker.init(frame, detections):
        info = tracker.get_tracker_info()
        
        assert info['frame_count'] == 1, "Frame count should be 1 after initialization"
        assert info['active_tracks'] >= 0, "Active tracks should be non-negative"
        assert 'population_sizes' in info['parameters'], "Population sizes should be in parameters"
        assert 'current_ball_counts' in info['parameters'], "Current ball counts should be in parameters"
        
        print("  ✅ Post-initialization info correct")
        
        # Update tracker and check info again
        tracker.update(frame)
        info = tracker.get_tracker_info()
        
        assert info['frame_count'] == 2, "Frame count should be 2 after update"
        
        print("  ✅ Post-update info correct")
    


def test_performance_metrics() -> None:
    """Test the get_performance_metrics method."""
    print("Testing get_performance_metrics method...")
    
    tracker = MOLTTracker()
    
    # Test metrics before initialization
    metrics = tracker.get_performance_metrics()
    
    assert 'frame_count' in metrics, "Frame count should be in metrics"
    assert 'total_populations' in metrics, "Total populations should be in metrics"
    assert 'is_initialized' in metrics, "Initialization status should be in metrics"
    assert metrics['is_initialized'] == False, "Should not be initialized initially"
    
    print("  ✅ Pre-initialization metrics correct")
    
    # Initialize and test metrics
    frame = create_test_frame()
    detections = create_test_detections()
    
    if tracker.init(frame, detections):
        metrics = tracker.get_performance_metrics()
        
        assert metrics['is_initialized'] == True, "Should be initialized after init"
        assert metrics['total_populations'] > 0, "Should have populations after init"
        assert 'population_metrics' in metrics, "Should have population metrics"
        assert 'ball_count_metrics' in metrics, "Should have ball count metrics"
        
        print("  ✅ Post-initialization metrics correct")
        
        # Update and check metrics
        tracker.update(frame)
        metrics = tracker.get_performance_metrics()
        
        assert metrics['frame_count'] == 2, "Frame count should be updated"
        
        print("  ✅ Post-update metrics correct")
    


def test_tracking_status() -> None:
    """Test the get_tracking_status method."""
    print("Testing get_tracking_status method...")
    
    tracker = MOLTTracker()
    
    # Test status before initialization
    status = tracker.get_tracking_status()
    
    assert status['is_initialized'] == False, "Should not be initialized"
    assert status['tracking_health'] == 'not_initialized', "Health should be not_initialized"
    assert len(status['issues']) > 0, "Should have issues when not initialized"
    assert len(status['recommendations']) > 0, "Should have recommendations"
    
    print("  ✅ Pre-initialization status correct")
    
    # Initialize and test status
    frame = create_test_frame()
    detections = create_test_detections()
    
    if tracker.init(frame, detections):
        status = tracker.get_tracking_status()
        
        assert status['is_initialized'] == True, "Should be initialized"
        assert status['tracking_health'] in ['good', 'poor', 'critical'], "Health should be valid"
        assert 'active_populations' in status, "Should have active populations count"
        assert 'total_populations' in status, "Should have total populations count"
        
        print(f"  ✅ Post-initialization status: {status['tracking_health']}")
        
        # Update and check status
        tracker.update(frame)
        status = tracker.get_tracking_status()
        
        print(f"  ✅ Post-update status: {status['tracking_health']}")
    


def test_reset_functionality() -> None:
    """Test the reset method."""
    print("Testing reset functionality...")
    
    tracker = MOLTTracker()
    
    # Initialize tracker
    frame = create_test_frame()
    detections = create_test_detections()
    
    if tracker.init(frame, detections):
        # Update a few times
        for _ in range(3):
            tracker.update(frame)
        
        # Check state before reset
        assert tracker.frame_count > 1, "Frame count should be > 1"
        assert tracker.is_initialized == True, "Should be initialized"
        assert len(tracker.populations) > 0, "Should have populations"
        
        print("  ✅ Tracker state established")
        
        # Reset tracker
        tracker.reset()
        
        # Check state after reset
        assert tracker.frame_count == 0, "Frame count should be 0 after reset"
        assert tracker.is_initialized == False, "Should not be initialized after reset"
        assert len(tracker.populations) == 0, "Should have no populations after reset"
        assert len(tracker.trails) == 0, "Should have no trails after reset"
        assert tracker.total_tracks_created == 0, "Statistics should be reset"
        
        print("  ✅ Reset completed successfully")
        
        # Test that tracker can be reinitialized
        if tracker.init(frame, detections):
            assert tracker.is_initialized == True, "Should be able to reinitialize"
            print("  ✅ Reinitialization successful")
    


def test_statistics_integration() -> None:
    """Test integration of all statistics methods."""
    print("Testing statistics integration...")
    
    tracker = MOLTTracker()
    frame = create_test_frame()
    detections = create_test_detections()
    
    if tracker.init(frame, detections):
        # Update tracker
        tracker.update(frame)
        
        # Get all statistics
        tracker_info = tracker.get_tracker_info()
        performance_metrics = tracker.get_performance_metrics()
        tracking_status = tracker.get_tracking_status()
        population_stats = tracker.get_population_statistics()
        ball_count_stats = tracker.get_ball_count_statistics()
        
        # Verify consistency between different statistics
        assert tracker_info['frame_count'] == performance_metrics['frame_count'], \
            "Frame count should be consistent"
        
        assert tracker_info['active_tracks'] == performance_metrics['active_populations'], \
            "Active tracks should be consistent"
        
        assert tracking_status['is_initialized'] == performance_metrics['is_initialized'], \
            "Initialization status should be consistent"
        
        print("  ✅ Statistics consistency verified")
        
        # Print sample statistics for verification
        print(f"  Frame count: {tracker_info['frame_count']}")
        print(f"  Active tracks: {tracker_info['active_tracks']}")
        print(f"  Tracking health: {tracking_status['tracking_health']}")
        print(f"  Ball count violations: {ball_count_stats['total_count_violations']}")
        
        print("  ✅ Statistics integration successful")
    


if __name__ == "__main__":
    print("=" * 60)
    print("MOLT Tracker Task 8.2 Test Suite")
    print("=" * 60)
    
    try:
        test_tracker_info()
        test_performance_metrics()
        test_tracking_status()
        test_reset_functionality()
        test_statistics_integration()
        
        print("\n All tests passed!")
        
    except Exception as e:
        print(f"\n Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)