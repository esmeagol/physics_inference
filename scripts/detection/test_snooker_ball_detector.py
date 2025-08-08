#!/usr/bin/env python3
"""
Test script for SnookerBallDetector with multiple models.

This script tests the SnookerBallDetector sanitization layer using two different
PyTorch models on the specified video file.
"""

import os
import sys
import cv2
import argparse
import logging
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from detection.local_pt_inference import LocalPT
from detection.snooker_ball_detector import SnookerBallDetector


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('assets/output/snooker_detector_test.log')
        ]
    )


def test_snooker_ball_detector() -> bool:
    """Test the SnookerBallDetector with two models."""
    
    # Model paths
    model1_path = "/Users/abhinavrai/Playground/snooker_data/trained models/snookers-gkqap-yolov11-medium-weights.pt"
    model2_path = "/Users/abhinavrai/Playground/snooker_data/trained models/ar-snkr_objd-lolhi-4-yolov11-weights.pt"
    video_path = "assets/test_videos/hires_video_65s.mp4"
    
    # Check if files exist
    if not os.path.exists(model1_path):
        print(f"Error: Model 1 not found at {model1_path}")
        return False
    
    if not os.path.exists(model2_path):
        print(f"Error: Model 2 not found at {model2_path}")
        return False
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return False
    
    print("Setting up detectors...")
    
    try:
        # Initialize individual detectors
        detector1 = LocalPT(model1_path, confidence=0.2)
        detector2 = LocalPT(model2_path, confidence=0.2)
        
        print(f"Detector 1: {detector1.get_model_info()['name']}")
        print(f"Detector 2: {detector2.get_model_info()['name']}")
        
        # Initialize SnookerBallDetector with both models
        snooker_detector = SnookerBallDetector(
            detectors=[detector1, detector2],
            confidence_threshold=0.2,
            temporal_window=5
        )
        
        print(f"SnookerBallDetector initialized with {len(snooker_detector.detectors)} detectors")
        
    except Exception as e:
        print(f"Error initializing detectors: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video writer
    output_path = "assets/output/snooker_detector_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    process_every_n_frames = 1  # Process every 5th frame for speed
    
    print("Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % process_every_n_frames == 0:
                print(f"Processing frame {frame_count}/{total_frames}")
                
                try:
                    # Run detection
                    results = snooker_detector.predict(frame)
                    
                    # Visualize results
                    annotated_frame = snooker_detector.visualize_predictions(
                        frame, results
                    )
                    
                    # Log results
                    sanitization_info = results.get('sanitization_info', {})
                    print(f"  Frame {frame_count}: "
                          f"Original: {sanitization_info.get('original_count', 0)}, "
                          f"Final: {sanitization_info.get('final_count', 0)}, "
                          f"Phase: {sanitization_info.get('game_phase', 'unknown')}")
                    
                    # Write annotated frame
                    out.write(annotated_frame)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    # Write original frame on error
                    out.write(frame)
            else:
                # Write original frame for skipped frames
                out.write(frame)
            
            # Break early for testing (process first 100 frames)
            if frame_count >= 100:
                print("Stopping after 100 frames for testing")
                break
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"Output video saved to: {output_path}")
    print(f"Log file saved to: assets/output/snooker_detector_test.log")
    
    return True


def test_single_frame() -> bool:
    """Test the detector on a single frame for debugging."""
    
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
        snooker_detector = SnookerBallDetector([detector1, detector2])
        
        # Get first frame from video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read frame from video")
            return False
        
        # Test individual detectors for reference
        result1 = detector1.predict(frame)
        result2 = detector2.predict(frame)
        
        # print(f"Detector 1 found {len(result1.get('predictions', []))} balls")
        # print(f"Detector 2 found {len(result2.get('predictions', []))} balls")
        
        # Test combined detector
        print("Testing SnookerBallDetector...")
        combined_result = snooker_detector.predict(frame)
        
        print(f"Combined detector found {len(combined_result.get('predictions', []))} balls")
        
        # Show sanitization info
        sanitization_info = combined_result.get('sanitization_info', {})
        print(f"Sanitization info: {sanitization_info}")
        
        # Save visualizations
        vis1 = detector1.visualize_predictions(frame, result1, "detector1_output.jpg")
        vis2 = detector2.visualize_predictions(frame, result2, "detector2_output.jpg")
        vis_combined = snooker_detector.visualize_predictions(frame, combined_result, "combined_output.jpg")
        
        print("Visualizations saved:")
        print("  detector1_output.jpg")
        print("  detector2_output.jpg")
        print("  combined_output.jpg")
        
        return True
        
    except Exception as e:
        print(f"Error in single frame test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description='Test SnookerBallDetector')
    parser.add_argument('--single-frame', action='store_true',
                       help='Test on single frame only')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    print("Testing SnookerBallDetector...")
    print("=" * 50)
    
    if args.single_frame:
        success = test_single_frame()
    else:
        success = test_snooker_ball_detector()
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()