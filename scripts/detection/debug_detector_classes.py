#!/usr/bin/env python3
"""
Debug script to see what classes the detectors are returning.
"""

import os
import sys
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from detection.local_pt_inference import LocalPT


def debug_detector_classes() -> bool:
    """Debug what classes each detector returns."""
    
    model1_path = "/Users/abhinavrai/Playground/snooker_data/trained models/snookers-gkqap-yolov11-medium-weights.pt"
    model2_path = "/Users/abhinavrai/Playground/snooker_data/trained models/ar-snkr_objd-lolhi-4-yolov11-weights.pt"
    video_path = "assets/test_videos/hires_video_65s.mp4"
    
    # Check if files exist
    for path in [model1_path, model2_path, video_path]:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return False
    
    try:
        # Initialize detectors
        detector1 = LocalPT(model1_path, confidence=0.3)
        detector2 = LocalPT(model2_path, confidence=0.3)
        
        # Get first frame from video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read frame from video")
            return False
        
        # Test individual detectors
        result1 = detector1.predict(frame)
        result2 = detector2.predict(frame)
        
        print("Detector 1 classes:")
        for pred in result1.get('predictions', []):
            print(f"  Class: '{pred['class']}', Confidence: {pred['confidence']:.3f}")
        
        print(f"\nDetector 1 unique classes: {set(pred['class'] for pred in result1.get('predictions', []))}")
        
        print("\nDetector 2 classes:")
        for pred in result2.get('predictions', []):
            print(f"  Class: '{pred['class']}', Confidence: {pred['confidence']:.3f}")
        
        print(f"\nDetector 2 unique classes: {set(pred['class'] for pred in result2.get('predictions', []))}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    debug_detector_classes()