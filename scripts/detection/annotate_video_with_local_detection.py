#!/usr/bin/env python3
"""
Test script for local model video processing.

This script processes a video file, runs inference on each frame using a local PyTorch model,
and creates an annotated output video.
"""

import os
import argparse
import time
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root.parent))

from src.detection.local_pt_inference import LocalPT
from collections import Counter

# Default values
DEFAULT_CONFIG = {
    'model_path': "/Users/abhinavrai/Playground/snooker_data/trained models/ar-snkr_objd-lolhi-4-yolov11-weights.pt",
    'confidence': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
    'iou': float(os.getenv('IOU_THRESHOLD', '0.5')),
}

def generate_summary_text(predictions: Dict[str, Any]) -> str:
    """
    Generate a summary text string from predictions.
    
    Args:
        predictions: Dictionary containing prediction results
        
    Returns:
        Formatted summary string
    """
    pred_list = predictions.get('predictions', [])
    
    if not pred_list:
        return "No objects detected"
    
    # Count total objects
    total_count = len(pred_list)
    
    # Count objects by class
    class_counts = Counter(pred['class'] for pred in pred_list)
    
    # Format class counts
    class_parts = [f"{class_name}: {count}" for class_name, count in sorted(class_counts.items())]
    class_str = ", ".join(class_parts)
    
    return f"Total: {total_count} ({class_str})"

def process_video(
    input_video_path: str,
    output_dir: str,
    debug_dir: str,
    model_path: str | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    frame_skip: int = 0,
) -> None:
    """
    Process a video file with a local PyTorch model.
    
    Args:
        input_video_path: Path to the input video file
        output_dir: Directory to save the output video
        debug_dir: Directory to save debug frames
        model_path: Path to the local PyTorch model weights (.pt file)
        confidence: Minimum confidence threshold for detections (0-1)
        iou: IoU threshold for NMS (0-1)
        frame_skip: Number of frames to skip between processing
    """
    # Apply defaults if not provided
    # Apply defaults with proper type assertions
    default_model_path = DEFAULT_CONFIG['model_path']
    default_confidence = DEFAULT_CONFIG['confidence']
    default_iou = DEFAULT_CONFIG['iou']
    
    # Use type assertions to help mypy understand the types
    assert isinstance(default_model_path, str)
    assert isinstance(default_confidence, float)
    assert isinstance(default_iou, float)
    
    # Now assign with the correct types
    model_path_str: str = default_model_path if model_path is None else str(model_path)
    confidence_float: float = default_confidence if confidence is None else float(confidence)
    iou_float: float = default_iou if iou is None else float(iou)
    
    # Validate required parameters
    if not os.path.exists(model_path_str):
        raise FileNotFoundError(f"Model file not found: {model_path_str}")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize the local model
    model = LocalPT(
        model_path=model_path_str,
        confidence=confidence_float,
        iou=iou_float
    )
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video
    input_filename = Path(input_video_path).stem
    output_video_path = os.path.join(output_dir, f"{input_filename}_annotated.mp4")
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    print(f"Processing video: {input_video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
    print(f"Skipping {frame_skip} frames between processing")
    
    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames if needed
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue
            
            # Run inference
            try:
                predictions = model.predict(frame)
                
                # Create a copy of the frame for annotations
                annotated_frame = frame.copy()
                
                # Generate and draw summary text at the top of the frame
                summary_text = generate_summary_text(predictions)
                
                # Draw text with black outline for visibility
                text_position = (10, 30)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Draw black outline
                cv2.putText(
                    annotated_frame,
                    summary_text,
                    text_position,
                    font,
                    font_scale,
                    (0, 0, 0),  # Black color for outline
                    thickness + 1
                )
                
                # Draw white text
                cv2.putText(
                    annotated_frame,
                    summary_text,
                    text_position,
                    font,
                    font_scale,
                    (255, 255, 255),  # White color for text
                    thickness
                )
                
                # Draw bounding boxes for each prediction
                for pred in predictions.get('predictions', []):
                    # Extract prediction details
                    x_center = pred['x']
                    y_center = pred['y']
                    width = pred['width']
                    height = pred['height']
                    confidence = pred['confidence']
                    class_name = pred['class']
                    
                    # Convert to top-left and bottom-right coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}"
                    cv2.putText(
                        annotated_frame, 
                        label, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )
                
                # Save debug frame periodically
                if frame_count % 30 == 0:  # Save every 30th frame for debugging
                    debug_frame_path = os.path.join(
                        debug_dir,
                        f"{input_filename}_frame_{frame_count:06d}.jpg"
                    )
                    cv2.imwrite(debug_frame_path, annotated_frame)
                
                # Write the frame to the output video
                out.write(annotated_frame)
                processed_frames += 1
                
                # Print progress
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = processed_frames / elapsed if elapsed > 0 else 0
                    print(
                        f"Processed {frame_count}/{total_frames} frames "
                        f"({frame_count/total_frames*100:.1f}%) - "
                        f"Processing FPS: {fps_processed:.2f}",
                        end="\r"
                    )
                
            except Exception as e:
                print(f"\nError processing frame {frame_count}: {str(e)}")
                # Write the original frame if processing fails
                out.write(frame)
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        # Release resources
        cap.release()
        out.release()
        
        # Print summary
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"Processing complete!")
        print(f"Total frames processed: {processed_frames}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {processed_frames/max(1, total_time):.2f}")
        print(f"Output video saved to: {output_video_path}")
        print("="*50)

def main() -> None:
    """Main function to parse arguments and run the video processing."""
    parser = argparse.ArgumentParser(description="Process video with local PyTorch model")
    parser.add_argument(
        "--input",
        type=str,
        default="assets/test_videos/hires_video_cropped_60s.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/output",
        help="Directory to save output video"
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="assets/debug",
        help="Directory to save debug frames"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_CONFIG['model_path'],
        help=f"Path to local PyTorch model weights (default: {DEFAULT_CONFIG['model_path']})"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIG['confidence'],
        help=f"Minimum confidence threshold (0-1) (default: {DEFAULT_CONFIG['confidence']})"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_CONFIG['iou'],
        help=f"IoU threshold for NMS (0-1) (default: {DEFAULT_CONFIG['iou']})"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip between processing (0 = process every frame)"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    if not os.path.isabs(args.input):
        args.input = str(Path(project_root.parent) / args.input)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(Path(project_root.parent) / args.output_dir)
    if not os.path.isabs(args.debug_dir):
        args.debug_dir = str(Path(project_root.parent) / args.debug_dir)
    if not os.path.isabs(args.model_path):
        args.model_path = str(Path(project_root.parent) / args.model_path)
    
    process_video(
        input_video_path=args.input,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir,
        model_path=args.model_path,
        confidence=args.confidence,
        iou=args.iou,
        frame_skip=args.frame_skip,
    )

if __name__ == "__main__":
    main()
