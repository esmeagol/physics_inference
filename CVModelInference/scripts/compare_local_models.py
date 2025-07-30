#!/usr/bin/env python3
"""
Script to compare object detection results between two local PyTorch models.
Processes a video file and generates annotated output video with side-by-side comparison.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import numpy as np
from tqdm import tqdm

# Import the InferenceRunner implementations
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CVModelInference.local_pt_inference import LocalPT


def load_models(model1_path: str, model2_path: str, confidence: float = 0.5, iou: float = 0.5):
    """
    Load two local PyTorch models.
    
    Args:
        model1_path: Path to first model weights (.pt file)
        model2_path: Path to second model weights (.pt file)
        confidence: Confidence threshold for detections
        iou: IoU threshold for NMS
        
    Returns:
        Tuple of (model1, model2) as LocalPT instances
    """
    print(f"Loading model 1: {os.path.basename(model1_path)}")
    model1 = LocalPT(model_path=model1_path, confidence=confidence, iou=iou)
    
    print(f"Loading model 2: {os.path.basename(model2_path)}")
    model2 = LocalPT(model_path=model2_path, confidence=confidence, iou=iou)
    
    return model1, model2


def process_frame(model, frame: np.ndarray, confidence: float = 0.5) -> Dict[Any, Any]:
    """
    Process a single frame with the given model.
    
    Args:
        model: LocalPT model instance
        frame: Input frame as numpy array
        confidence: Confidence threshold
        
    Returns:
        Dictionary containing detection results
    """
    result = model.predict(frame, confidence=confidence)
    return dict(result) if result else {}


def create_side_by_side_frame(frame: np.ndarray, results1: Dict, results2: Dict, 
                             model1_name: str, model2_name: str) -> np.ndarray:
    """
    Create a side-by-side comparison frame with detection results.
    
    Args:
        frame: Original input frame
        results1: Detection results from model 1
        results2: Detection results from model 2
        model1_name: Display name for model 1
        model2_name: Display name for model 2
        
    Returns:
        Side-by-side comparison frame with annotations
    """
    # Create a copy of the frame for each model
    frame1 = frame.copy()
    frame2 = frame.copy()
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Draw detections on each frame
    draw_detections(frame1, results1, model1_name)
    draw_detections(frame2, results2, model2_name)
    
    # Create side-by-side comparison frame
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    comparison[:, :width] = frame1
    comparison[:, width:] = frame2
    
    # Add a vertical line separating the two frames
    cv2.line(comparison, (width, 0), (width, height), (255, 255, 255), 2)
    
    return comparison


def draw_detections(frame: np.ndarray, results: Dict, model_name: str) -> None:
    """
    Draw detection results on a frame.
    
    Args:
        frame: Frame to draw on
        results: Detection results
        model_name: Name of the model to display
    """
    predictions = results.get('predictions', [])
    
    # Count objects by class
    class_counts: dict[str, int] = {}
    for pred in predictions:
        cls = pred['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Draw each prediction
    for pred in predictions:
        # Get class and confidence
        cls = pred['class']
        conf = pred['confidence']
        
        # Determine color based on class (simple hash function for consistent colors)
        color_hash = hash(cls) % 0xFFFFFF
        color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
        
        # Draw bounding box
        x = int(pred['x'] - pred['width']/2)
        y = int(pred['y'] - pred['height']/2)
        w = int(pred['width'])
        h = int(pred['height'])
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Add label with confidence
        label = f"{cls} {conf:.2f}"
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add summary text
    summary = [f"{model_name}:", f"Total: {len(predictions)}"]
    for cls, count in sorted(class_counts.items()):
        summary.append(f"{cls}: {count}")
    
    # Draw semi-transparent background for text
    text_height = len(summary) * 25 + 10
    cv2.rectangle(frame, (10, 10), 
                 (200, text_height), 
                 (0, 0, 0), -1)
    
    # Draw text
    for i, line in enumerate(summary):
        y = 30 + i * 25
        cv2.putText(frame, line, (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process_video(model1, model2, video_path: str, output_path: str, 
                 confidence: float = 0.5, fps_limit: Optional[float] = None,
                 start_time: float = 0, duration: Optional[float] = None,
                 show_preview: bool = False):
    """
    Process a video with two models and generate a comparison video.
    
    Args:
        model1: First LocalPT model
        model2: Second LocalPT model
        video_path: Path to input video
        output_path: Path to save output video
        confidence: Confidence threshold for detections
        fps_limit: Optional limit for processing FPS
        start_time: Time in seconds to start processing from
        duration: Optional duration in seconds to process
        show_preview: Whether to show preview window during processing
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames to skip for start_time
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate end frame if duration is specified
    end_frame = total_frames
    if duration is not None:
        end_frame = start_frame + int(duration * fps)
    
    # Create output video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Get model names for display
    model1_name = os.path.basename(model1.model_path)
    model2_name = os.path.basename(model2.model_path)
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    # Initialize FPS tracking
    fps_delay = 0.0
    if fps_limit:
        fps_delay = 1.0 / fps_limit
    
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Output: {output_path}")
    print(f"Start time: {start_time}s, Duration: {duration if duration else 'until end'}s")
    print(f"FPS limit: {fps_limit if fps_limit else 'None'}")
    
    pbar = tqdm(total=end_frame - start_frame, desc="Processing frames")
    
    try:
        while cap.isOpened():
            # Check if we've reached the end frame
            if frame_count + start_frame >= end_frame:
                break
            
            # Track processing time for FPS limiting
            start_process_time = time.time()
            
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process with both models
            results1 = process_frame(model1, frame, confidence)
            results2 = process_frame(model2, frame, confidence)
            
            # Create side-by-side comparison
            comparison = create_side_by_side_frame(
                frame, results1, results2, model1_name, model2_name
            )
            
            # Write to output video
            out.write(comparison)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('Comparison', comparison)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Update counters
            frame_count += 1
            processed_count += 1
            pbar.update(1)
            
            # Limit FPS if specified
            if fps_limit:
                process_time = time.time() - start_process_time
                sleep_time = max(0, fps_delay - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        pbar.close()
    
    print(f"Processed {processed_count} frames")
    print(f"Output saved to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description='Compare two local PyTorch models on a video file')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model weights (.pt file)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model weights (.pt file)')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output video')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--fps-limit', type=float, default=None,
                       help='Limit processing FPS')
    parser.add_argument('--start-time', type=float, default=0,
                       help='Start time in seconds')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration to process in seconds')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview window during processing')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.isfile(args.model1):
        print(f"Error: Model 1 file does not exist: {args.model1}")
        return
    
    if not os.path.isfile(args.model2):
        print(f"Error: Model 2 file does not exist: {args.model2}")
        return
    
    if not os.path.isfile(args.video):
        print(f"Error: Video file does not exist: {args.video}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    model1, model2 = load_models(
        args.model1,
        args.model2,
        confidence=args.confidence,
        iou=args.iou
    )
    
    # Process video
    process_video(
        model1, model2,
        args.video,
        args.output,
        confidence=args.confidence,
        fps_limit=args.fps_limit,
        start_time=args.start_time,
        duration=args.duration,
        show_preview=args.preview
    )


if __name__ == "__main__":
    main()
