#!/usr/bin/env python3
"""
Basic usage example for MOLT tracker.

This example demonstrates how to use the MOLT tracker with
the reorganized code structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
from typing import List, Dict, Any

from PureCV.molt import MOLTTracker, MOLTTrackerConfig


def create_sample_video_frame(frame_num: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Create a sample video frame with moving balls."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add moving balls
    # Red ball moving horizontally
    red_x = 50 + (frame_num * 2) % (width - 100)
    cv2.circle(frame, (red_x, 100), 15, (0, 0, 255), -1)
    
    # White ball moving diagonally
    white_x = 100 + (frame_num * 1) % (width - 200)
    white_y = 150 + int(50 * np.sin(frame_num * 0.1))
    cv2.circle(frame, (white_x, white_y), 15, (255, 255, 255), -1)
    
    # Yellow ball moving in circle
    center_x, center_y = width // 2, height // 2
    radius = 100
    angle = frame_num * 0.05
    yellow_x = int(center_x + radius * np.cos(angle))
    yellow_y = int(center_y + radius * np.sin(angle))
    cv2.circle(frame, (yellow_x, yellow_y), 15, (0, 255, 255), -1)
    
    return frame


def create_initial_detections() -> List[Dict[str, Any]]:
    """Create initial detections for the first frame."""
    return [
        {'x': 50, 'y': 100, 'width': 30, 'height': 30, 'class': 'red', 'confidence': 0.9},
        {'x': 100, 'y': 150, 'width': 30, 'height': 30, 'class': 'white', 'confidence': 0.95},
        {'x': 420, 'y': 240, 'width': 30, 'height': 30, 'class': 'yellow', 'confidence': 0.85}
    ]


def basic_tracking_example() -> None:
    """Demonstrate basic tracking functionality."""
    print("=== Basic MOLT Tracker Usage Example ===\n")
    
    # 1. Create tracker with default configuration
    print("1. Creating MOLT tracker with default configuration...")
    tracker = MOLTTracker()
    print(f"   ✓ Tracker created: {tracker.get_tracker_info()['name']}")
    
    # 2. Initialize tracker with first frame
    print("\n2. Initializing tracker with first frame...")
    first_frame = create_sample_video_frame(0)
    initial_detections = create_initial_detections()
    
    success = tracker.init(first_frame, initial_detections)
    if success:
        print(f"   ✓ Tracker initialized successfully")
        print(f"   ✓ Number of populations: {len(tracker.populations)}")
        print(f"   ✓ Ball counts: {tracker.ball_counts}")
    else:
        print("   ❌ Tracker initialization failed")
        return
    
    # 3. Process several frames
    print("\n3. Processing video frames...")
    num_frames = 10
    
    for frame_num in range(1, num_frames + 1):
        frame = create_sample_video_frame(frame_num)
        tracks = tracker.update(frame)
        
        print(f"   Frame {frame_num}: {len(tracks)} tracks")
        
        # Visualize every few frames
        if frame_num % 3 == 0:
            vis_frame = tracker.visualize(frame, tracks)
            print(f"   ✓ Visualization created for frame {frame_num}")
    
    # 4. Get tracker statistics
    print("\n4. Getting tracker statistics...")
    info = tracker.get_tracker_info()
    stats = tracker.get_population_statistics()
    
    print(f"   ✓ Frames processed: {info['frame_count']}")
    print(f"   ✓ Active tracks: {info['active_tracks']}")
    print(f"   ✓ Total populations: {stats['total_populations']}")
    
    print("\n✅ Basic tracking example completed successfully!")


def custom_configuration_example() -> None:
    """Demonstrate usage with custom configuration."""
    print("\n=== Custom Configuration Example ===\n")
    
    # 1. Create custom configuration
    print("1. Creating custom configuration...")
    config = MOLTTrackerConfig.create_for_snooker()
    
    # Customize parameters
    config.histogram_bins = 12
    config.color_space = 'RGB'
    config.population_sizes = {'red': 100, 'white': 150, 'yellow': 100}
    config.min_confidence = 0.3
    
    print(f"   ✓ Custom config created with {config.histogram_bins} histogram bins")
    print(f"   ✓ Color space: {config.color_space}")
    print(f"   ✓ Population sizes: {config.population_sizes}")
    
    # 2. Create tracker with custom config
    print("\n2. Creating tracker with custom configuration...")
    tracker = MOLTTracker(config=config)
    
    # Verify configuration
    tracker_config = tracker.get_config()
    print(f"   ✓ Tracker histogram bins: {tracker_config['histogram_bins']}")
    print(f"   ✓ Tracker color space: {tracker_config['color_space']}")
    
    # 3. Test with sample data
    print("\n3. Testing with sample data...")
    frame = create_sample_video_frame(0)
    detections = create_initial_detections()
    
    success = tracker.init(frame, detections)
    if success:
        print("   ✓ Custom configured tracker works correctly")
    else:
        print("   ❌ Custom configured tracker failed")
    
    print("\n✅ Custom configuration example completed!")


def kwargs_override_example() -> None:
    """Demonstrate parameter override with kwargs."""
    print("\n=== Kwargs Override Example ===\n")
    
    print("1. Creating tracker with kwargs overrides...")
    
    # Create tracker with parameter overrides
    tracker = MOLTTracker(
        histogram_bins=16,
        color_space='HSV',
        min_confidence=0.1,
        population_sizes={'red': 200, 'white': 300, 'yellow': 150}
    )
    
    # Verify overrides
    config = tracker.get_config()
    print(f"   ✓ Histogram bins: {config['histogram_bins']}")
    print(f"   ✓ Color space: {config['color_space']}")
    print(f"   ✓ Min confidence: {config['min_confidence']}")
    print(f"   ✓ Population sizes: {config['population_sizes']}")
    
    print("\n✅ Kwargs override example completed!")


def error_handling_example() -> None:
    """Demonstrate error handling."""
    print("\n=== Error Handling Example ===\n")
    
    tracker = MOLTTracker()
    
    print("1. Testing error conditions...")
    
    # Test invalid initialization
    print("   Testing invalid frame...")
    success = tracker.init(None, create_initial_detections())
    print(f"   ✓ Invalid frame handled: success = {success}")
    
    print("   Testing empty detections...")
    frame = create_sample_video_frame(0)
    success = tracker.init(frame, [])
    print(f"   ✓ Empty detections handled: success = {success}")
    
    print("   Testing update without initialization...")
    tracks = tracker.update(frame)
    print(f"   ✓ Update without init handled: {len(tracks)} tracks returned")
    
    # Test reset functionality
    print("\n2. Testing reset functionality...")
    tracker.init(frame, create_initial_detections())
    print(f"   Before reset - initialized: {tracker.is_initialized}")
    
    tracker.reset()
    print(f"   After reset - initialized: {tracker.is_initialized}")
    print(f"   After reset - populations: {len(tracker.populations)}")
    
    print("\n✅ Error handling example completed!")


if __name__ == "__main__":
    try:
        basic_tracking_example()
        custom_configuration_example()
        kwargs_override_example()
        error_handling_example()
        
        print("\n🎉 All examples completed successfully!")
        print("📚 The reorganized MOLT tracker is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()