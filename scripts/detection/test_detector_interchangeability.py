#!/usr/bin/env python3
"""
Test script to demonstrate that SnookerBallDetector is interchangeable with raw detectors.
"""

import os
import sys
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from detection.local_pt_inference import LocalPT
from detection.snooker_ball_detector import SnookerBallDetector
from typing import Any, Dict


def test_detector_interchangeability() -> bool:
    """Test that SnookerBallDetector has the same interface as raw detectors."""
    
    print("Testing Detector Interchangeability...")
    print("=" * 50)
    
    # Model paths
    model1_path = "/Users/abhinavrai/Playground/snooker_data/trained models/snookers-gkqap-yolov11-medium-weights.pt"
    model2_path = "/Users/abhinavrai/Playground/snooker_data/trained models/ar-snkr_objd-lolhi-4-yolov11-weights.pt"
    video_path = "assets/test_videos/hires_video_65s.mp4"
    
    # Check if files exist
    for path in [model1_path, model2_path, video_path]:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return False
    
    try:
        # Initialize raw detectors
        raw_detector1 = LocalPT(model1_path, confidence=0.3)
        raw_detector2 = LocalPT(model2_path, confidence=0.3)
        
        # Initialize SnookerBallDetector (sanitized detector)
        snooker_detector = SnookerBallDetector([raw_detector1, raw_detector2])
        
        # Get test frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read frame from video")
            return False
        
        print("1. Testing predict() method interface...")
        
        # Test all detectors with the same interface
        detectors = [
            ("Raw Detector 1", raw_detector1),
            ("Raw Detector 2", raw_detector2), 
            ("Snooker Detector", snooker_detector)
        ]
        
        results = {}
        
        for name, detector in detectors:
            print(f"\n   Testing {name}:")
            
            # Test predict method
            result = detector.predict(frame)
            results[name] = result
            
            # Check required keys
            required_keys = ['predictions', 'image', 'model']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"     ❌ Missing keys: {missing_keys}")
                return False
            else:
                print(f"     ✅ Has all required keys: {required_keys}")
            
            # Check predictions format
            predictions = result['predictions']
            print(f"     ✅ Found {len(predictions)} predictions")
            
            if predictions:
                pred = predictions[0]
                pred_keys = ['x', 'y', 'width', 'height', 'confidence', 'class']
                missing_pred_keys = [key for key in pred_keys if key not in pred]
                
                if missing_pred_keys:
                    print(f"     ❌ Prediction missing keys: {missing_pred_keys}")
                    return False
                else:
                    print(f"     ✅ Predictions have correct format")
            
            # Check image info
            image_info = result['image']
            if 'width' in image_info and 'height' in image_info:
                print(f"     ✅ Image info: {image_info['width']}x{image_info['height']}")
            else:
                print(f"     ❌ Image info missing width/height")
                return False
        
        print("\n2. Testing other InferenceRunner methods...")
        
        # Test get_model_info
        for name, detector in detectors:
            try:
                model_info = detector.get_model_info()
                print(f"   {name}: {model_info.get('name', 'Unknown')}")
            except Exception as e:
                print(f"   ❌ {name} get_model_info failed: {e}")
                return False
        
        # Test visualize_predictions
        print("\n3. Testing visualize_predictions method...")
        for name, detector in detectors:
            try:
                result = results[name]
                vis_img = detector.visualize_predictions(frame, result)
                print(f"   ✅ {name}: Visualization successful ({vis_img.shape})")
            except Exception as e:
                print(f"   ❌ {name} visualize_predictions failed: {e}")
                return False
        
        print("\n4. Testing batch prediction interface...")
        
        # Test predict_batch (if available)
        test_frames = [frame, frame]  # Use same frame twice for testing
        
        for name, detector in detectors:
            try:
                if hasattr(detector, 'predict_batch'):
                    batch_results = detector.predict_batch(test_frames)
                    print(f"   ✅ {name}: Batch prediction successful ({len(batch_results)} results)")
                else:
                    print(f"   ⚠️  {name}: No predict_batch method")
            except Exception as e:
                print(f"   ❌ {name} predict_batch failed: {e}")
                return False
        
        print("\n5. Comparing output formats...")
        
        # Compare the structure of outputs
        raw_result = results["Raw Detector 1"]
        snooker_result = results["Snooker Detector"]
        
        print(f"   Raw detector keys: {list(raw_result.keys())}")
        print(f"   Snooker detector keys: {list(snooker_result.keys())}")
        
        # Check if snooker detector has additional info (which is expected)
        common_keys = set(raw_result.keys()) & set(snooker_result.keys())
        snooker_extra_keys = set(snooker_result.keys()) - set(raw_result.keys())
        
        print(f"   Common keys: {list(common_keys)}")
        print(f"   Snooker extra keys: {list(snooker_extra_keys)}")
        
        # Verify core compatibility
        if 'predictions' in common_keys and 'image' in common_keys and 'model' in common_keys:
            print("   ✅ Core interface compatibility confirmed")
        else:
            print("   ❌ Core interface compatibility failed")
            return False
        
        print("\n6. Testing as drop-in replacement...")
        
        def process_with_detector(detector: Any, frame: Any) -> Dict[str, Any]:
            """Function that works with any detector following the interface."""
            result = detector.predict(frame)
            predictions = result['predictions']
            image_info = result['image']
            model_name = result['model']
            
            return {
                'ball_count': len(predictions),
                'image_size': f"{image_info['width']}x{image_info['height']}",
                'model': model_name
            }
        
        # Test with all detectors using the same function
        for name, detector in detectors:
            try:
                processed = process_with_detector(detector, frame)
                print(f"   ✅ {name}: {processed}")
            except Exception as e:
                print(f"   ❌ {name} failed as drop-in replacement: {e}")
                return False
        
        print("\n" + "=" * 50)
        print("✅ INTERCHANGEABILITY TEST PASSED!")
        print("\nSnookerBallDetector is fully interchangeable with raw detectors:")
        print("- Same predict() method signature")
        print("- Same return format with 'predictions', 'image', 'model' keys")
        print("- Same prediction format with x, y, width, height, confidence, class")
        print("- Same InferenceRunner interface methods")
        print("- Can be used as drop-in replacement in any code expecting a detector")
        print("- Provides additional sanitization info as bonus")
        
        return True
        
    except Exception as e:
        print(f"Error in interchangeability test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_detector_interchangeability()
    if not success:
        sys.exit(1)