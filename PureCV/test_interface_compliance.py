#!/usr/bin/env python3
"""
Test script for MOLT tracker interface compliance.

This script tests that the MOLTTracker properly implements the Tracker
abstract base class interface and behaves correctly according to the
interface contract.
"""

import numpy as np
import sys
import os
from abc import ABC

# Add PureCV to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PureCV.molt import MOLTTracker, MOLTTrackerConfig

# Import the Tracker interface
try:
    from CVModelInference.tracker import Tracker, TrackerInfo, Detection, Track, Frame
    TRACKER_INTERFACE_AVAILABLE = True
except ImportError:
    print("Warning: CVModelInference.tracker not available, skipping interface compliance tests")
    TRACKER_INTERFACE_AVAILABLE = False


def test_interface_inheritance():
    """Test that MOLTTracker properly inherits from Tracker."""
    if not TRACKER_INTERFACE_AVAILABLE:
        print("Skipping interface inheritance test - interface not available")
        return True
    
    print("Testing interface inheritance...")
    
    # Check that MOLTTracker is a subclass of Tracker
    assert issubclass(MOLTTracker, Tracker), "MOLTTracker should inherit from Tracker"
    
    # Create instance and check isinstance
    tracker = MOLTTracker()
    assert isinstance(tracker, Tracker), "MOLTTracker instance should be instance of Tracker"
    
    print("  ✅ Interface inheritance correct")
    return True


def test_required_methods():
    """Test that all required methods are implemented."""
    print("Testing required methods implementation...")
    
    tracker = MOLTTracker()
    
    # Check that all required methods exist and are callable
    required_methods = ['init', 'update', 'reset', 'visualize', 'get_tracker_info']
    
    for method_name in required_methods:
        assert hasattr(tracker, method_name), f"MOLTTracker should have {method_name} method"
        method = getattr(tracker, method_name)
        assert callable(method), f"{method_name} should be callable"
    
    print("  ✅ All required methods implemented")
    return True


def test_method_signatures():
    """Test that method signatures match the interface."""
    print("Testing method signatures...")
    
    tracker = MOLTTracker()
    
    # Test init method signature
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = [
        {'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'red', 'class_id': 1, 'confidence': 0.9}
    ]
    
    # init should accept frame and detections, return bool
    result = tracker.init(frame, detections)
    assert isinstance(result, bool), "init should return bool"
    
    # update should accept frame and optional detections, return list
    tracks = tracker.update(frame)
    assert isinstance(tracks, list), "update should return list"
    
    # reset should accept no arguments, return None
    result = tracker.reset()
    assert result is None, "reset should return None"
    
    # visualize should accept frame, tracks, and optional output_path, return frame
    vis_frame = tracker.visualize(frame, tracks)
    assert isinstance(vis_frame, np.ndarray), "visualize should return numpy array"
    assert vis_frame.shape == frame.shape, "visualize should return frame with same shape"
    
    # get_tracker_info should return TrackerInfo (dict)
    info = tracker.get_tracker_info()
    assert isinstance(info, dict), "get_tracker_info should return dict"
    
    print("  ✅ Method signatures correct")
    return True


def test_tracker_info_format():
    """Test that TrackerInfo has the expected format."""
    print("Testing TrackerInfo format...")
    
    tracker = MOLTTracker()
    info = tracker.get_tracker_info()
    
    # Check required fields
    required_fields = ['name', 'type', 'parameters', 'frame_count', 'active_tracks']
    
    for field in required_fields:
        assert field in info, f"TrackerInfo should have {field} field"
    
    # Check field types
    assert isinstance(info['name'], str), "name should be string"
    assert isinstance(info['type'], str), "type should be string"
    assert isinstance(info['parameters'], dict), "parameters should be dict"
    assert isinstance(info['frame_count'], int), "frame_count should be int"
    assert isinstance(info['active_tracks'], int), "active_tracks should be int"
    
    # Check specific values
    assert info['name'] == 'MOLT', "name should be 'MOLT'"
    assert 'Multiple Object Local Tracker' in info['type'], "type should mention MOLT"
    
    print("  ✅ TrackerInfo format correct")
    return True


def test_track_format():
    """Test that Track objects have the expected format."""
    print("Testing Track format...")
    
    tracker = MOLTTracker()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = [
        {'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'red', 'class_id': 1, 'confidence': 0.9}
    ]
    
    if tracker.init(frame, detections):
        tracks = tracker.update(frame)
        
        if tracks:  # Only test if we have tracks
            track = tracks[0]
            
            # Check required fields
            required_fields = ['id', 'x', 'y', 'width', 'height', 'confidence']
            
            for field in required_fields:
                assert field in track, f"Track should have {field} field"
            
            # Check field types
            assert isinstance(track['id'], int), "id should be int"
            assert isinstance(track['x'], (int, float)), "x should be numeric"
            assert isinstance(track['y'], (int, float)), "y should be numeric"
            assert isinstance(track['width'], (int, float)), "width should be numeric"
            assert isinstance(track['height'], (int, float)), "height should be numeric"
            assert isinstance(track['confidence'], (int, float)), "confidence should be numeric"
            
            # Check value ranges
            assert 0 <= track['confidence'] <= 1, "confidence should be in [0, 1]"
            assert track['width'] > 0, "width should be positive"
            assert track['height'] > 0, "height should be positive"
            
            print("  ✅ Track format correct")
        else:
            print("  ℹ️  No tracks generated (may be expected)")
    
    return True


def test_state_management():
    """Test proper state management through the interface."""
    print("Testing state management...")
    
    tracker = MOLTTracker()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = [
        {'x': 100, 'y': 100, 'width': 30, 'height': 30, 'class': 'red', 'class_id': 1, 'confidence': 0.9}
    ]
    
    # Test initial state
    info = tracker.get_tracker_info()
    assert info['frame_count'] == 0, "Initial frame count should be 0"
    assert info['active_tracks'] == 0, "Initial active tracks should be 0"
    
    # Test after initialization
    success = tracker.init(frame, detections)
    assert success, "Initialization should succeed"
    
    info = tracker.get_tracker_info()
    assert info['frame_count'] == 1, "Frame count should be 1 after init"
    
    # Test after update
    tracks = tracker.update(frame)
    info = tracker.get_tracker_info()
    assert info['frame_count'] == 2, "Frame count should be 2 after update"
    
    # Test after reset
    tracker.reset()
    info = tracker.get_tracker_info()
    assert info['frame_count'] == 0, "Frame count should be 0 after reset"
    assert info['active_tracks'] == 0, "Active tracks should be 0 after reset"
    
    print("  ✅ State management correct")
    return True


def test_error_handling():
    """Test proper error handling through the interface."""
    print("Testing error handling...")
    
    tracker = MOLTTracker()
    
    # Test with invalid frame
    result = tracker.init(None, [])
    assert result == False, "init should return False for invalid frame"
    
    # Test with empty detections
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = tracker.init(frame, [])
    assert result == False, "init should return False for empty detections"
    
    # Test update before initialization
    tracks = tracker.update(frame)
    assert isinstance(tracks, list), "update should return list even when not initialized"
    assert len(tracks) == 0, "update should return empty list when not initialized"
    
    # Test visualize with empty tracks
    vis_frame = tracker.visualize(frame, [])
    assert isinstance(vis_frame, np.ndarray), "visualize should handle empty tracks"
    
    print("  ✅ Error handling correct")
    return True


def test_configuration_interface():
    """Test configuration through the interface."""
    print("Testing configuration interface...")
    
    # Test with default configuration
    tracker1 = MOLTTracker()
    info1 = tracker1.get_tracker_info()
    assert 'parameters' in info1, "Should have parameters in tracker info"
    
    # Test with custom configuration
    config = MOLTTrackerConfig.create_for_snooker()
    tracker2 = MOLTTracker(config=config)
    info2 = tracker2.get_tracker_info()
    
    # Both should have the same interface structure
    assert set(info1.keys()) == set(info2.keys()), "TrackerInfo structure should be consistent"
    
    # Test with parameter overrides
    tracker3 = MOLTTracker(histogram_bins=32, min_confidence=0.5)
    info3 = tracker3.get_tracker_info()
    
    # Should have updated parameters
    assert info3['parameters']['histogram_bins'] == 32, "Parameter override should work"
    assert info3['parameters']['min_confidence'] == 0.5, "Parameter override should work"
    
    print("  ✅ Configuration interface correct")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("MOLT Tracker Interface Compliance Test")
    print("=" * 60)
    
    success = True
    
    try:
        success &= test_interface_inheritance()
        success &= test_required_methods()
        success &= test_method_signatures()
        success &= test_tracker_info_format()
        success &= test_track_format()
        success &= test_state_management()
        success &= test_error_handling()
        success &= test_configuration_interface()
        
        if success:
            print("\n🎉 All interface compliance tests passed!")
            print("✅ MOLTTracker properly implements the Tracker interface")
        else:
            print("\n❌ Some interface compliance tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)