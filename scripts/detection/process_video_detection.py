#!/usr/bin/env python3
"""
Video Object Detection Script

This script processes the first 5 frames of a video file using a YOLO object detection model 
and outputs detection results in a JSON format with frame index, object ID, and bounding boxes.
"""

import os
import json
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent  # Go up to scripts directory
sys.path.insert(0, str(project_root.parent))  # Add physics_inference to path

# Import using absolute import from src package
# This avoids the mypy error about duplicate module names
from src.detection.local_pt_inference import LocalPT


def convert_to_bbox(x_center: float, y_center: float, width: float, height: float) -> Tuple[int, int, int, int]:
    """Convert center-based coordinates to [xmin, ymin, xmax, ymax] format."""
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return x_min, y_min, x_max, y_max


def process_video(
    video_path: str,
    model_path: str,
    max_frames: int = 5,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Process the first N frames of a video file and return detections in JSON format.
    
    Args:
        video_path: Path to the input video file
        model_path: Path to the YOLO model weights (.pt file)
        max_frames: Maximum number of frames to process (default: 5)
        confidence: Minimum confidence threshold for detections (0-1)
        iou: IoU threshold for NMS (0-1)
        
    Returns:
        List of detection dictionaries with bounding box coordinates
    """
    # Initialize the model
    detector = LocalPT(
        model_path=model_path,
        confidence=confidence,
        iou=iou
    )
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {min(max_frames, total_frames)} (processing first {max_frames} frames)")
    
    # Initialize results list
    results = []
    object_id_counter = 1  # Start object IDs from 1
    
    # Process only the first N frames
    for frame_idx in tqdm(range(min(max_frames, total_frames)), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        detections = detector.predict(frame)
        
        # Process detections for this frame
        for det in detections.get('predictions', []):
            # Skip if confidence is below threshold
            if det['confidence'] < confidence:
                continue
                
            # Convert center-based coordinates to [xmin, ymin, xmax, ymax]
            bbox = convert_to_bbox(
                det['x'], 
                det['y'], 
                det['width'], 
                det['height']
            )
            
            # Create detection entry with bounding box
            detection = {
                'frame_index': frame_idx,
                'object_id': object_id_counter,
                'box': list(bbox),  # Convert tuple to list for JSON serialization
            }
            
            results.append(detection)
            object_id_counter += 1
    
    # Release video capture
    cap.release()
    
    return results


def save_results_to_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save detection results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main() -> None:
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Configuration with absolute paths
    # Video is in the root assets directory
    video_path = str(project_root.parent / "assets" / "test_videos" / "video_1.mp4")
    # Model is in the root trained_models directory
    model_path = str(project_root.parent / "trained_models" / "ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt")
    output_json = "detection_results_bbox.json"
    max_frames = 5  # Process only first 5 frames
    confidence = 0.5
    iou = 0.5
    
    # Process the video
    results = process_video(
        video_path=video_path,
        model_path=model_path,
        max_frames=max_frames,
        confidence=confidence,
        iou=iou
    )
    
    # Save results
    save_results_to_json(results, output_json)
    
    # Print a sample of the results
    print("\nSample of the first 3 detections:")
    for det in results[:3]:
        print(json.dumps(det, indent=2))
    
    print(f"\nProcessing complete. Found {len(results)} detections in {max_frames} frames.")
    print(f"Results saved to: {os.path.abspath(output_json)}")


if __name__ == "__main__":
    main()
