#!/usr/bin/env python3
"""
Video Table Trimming Script

This script processes video files to detect table regions and crops the video
to show only the table area, removing non-table parts of the frame.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from detection.table_detection import TableDetector


def get_stable_table_region(detector: TableDetector, video_path: str, num_frames: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """
    Analyze multiple frames from a video to determine a stable table region.
    
    Args:
        detector: TableDetector instance
        video_path: Path to video file
        num_frames: Number of frames to sample for stability
        
    Returns:
        Tuple of (x, y, width, height) for the stable table region, or None if no table found
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // (num_frames + 1))
    
    bboxes: List[Tuple[int, int, int, int]] = []
    
    # Sample frames throughout the video
    for i in range(1, num_frames + 1):
        frame_pos = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Get table bounding box
        bbox = detector.get_table_bounding_box(frame)
        if bbox is not None:
            bboxes.append(bbox)
    
    cap.release()
    
    if not bboxes:
        print("No table detected in sampled frames")
        return None
    
    # Calculate median bounding box for stability
    bboxes_array = np.array(bboxes)
    median_bbox = np.median(bboxes_array, axis=0).astype(int)
    
    return tuple(median_bbox)


def trim_video_to_table(input_path: str, 
                       output_path: str,
                       model_path: str = "snkr_segm-egvem-3-roboflow-weights.pt",
                       confidence: float = 0.5,
                       padding: int = 20,
                       preview_frames: int = 5) -> bool:
    """
    Trim video to show only the table region.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output trimmed video
        model_path: Path to table segmentation model
        confidence: Detection confidence threshold
        padding: Padding around detected table in pixels
        preview_frames: Number of frames to analyze for stable region
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize table detector
        print(f"Loading table detection model: {model_path}")
        detector = TableDetector(model_path=model_path, confidence=confidence)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input video: {total_frames} frames at {fps:.2f} FPS")
        
        # Get stable table region
        print(f"Analyzing {preview_frames} frames to determine stable table region...")
        stable_bbox = get_stable_table_region(detector, input_path, preview_frames)
        
        if stable_bbox is None:
            print("Error: Could not detect table in video")
            cap.release()
            return False
        
        x, y, w, h = stable_bbox
        
        # Add padding and ensure bounds
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame_width, x + w + padding)
        y_end = min(frame_height, y + h + padding)
        
        crop_width = x_end - x_start
        crop_height = y_end - y_start
        
        print(f"Table region: ({x_start}, {y_start}) to ({x_end}, {y_end})")
        print(f"Output dimensions: {crop_width}x{crop_height}")
        
        # Get original video codec and settings
        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # If we can't get the original codec, fall back to a common one
        if original_fourcc == 0:
            # Try to determine codec from file extension
            _, ext = os.path.splitext(input_path)
            if ext.lower() in ['.mp4', '.mov']:
                original_fourcc = int(cv2.VideoWriter.fourcc(*'avc1'))  # H.264
            elif ext.lower() in ['.avi']:
                original_fourcc = int(cv2.VideoWriter.fourcc(*'XVID'))
            else:
                original_fourcc = int(cv2.VideoWriter.fourcc(*'mp4v'))  # Default fallback
        
        # Setup video writer with original codec
        out = cv2.VideoWriter(output_path, original_fourcc, fps, (crop_width, crop_height))
        
        # If original codec doesn't work, try H.264 which is widely supported
        if not out.isOpened():
            print("Warning: Could not use original codec, falling back to H.264")
            out.release()
            out = cv2.VideoWriter(output_path, int(cv2.VideoWriter.fourcc(*'avc1')), fps, (crop_width, crop_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            cap.release()
            return False
        
        # Process video frame by frame
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        print("Processing video frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame to table region
            cropped_frame = frame[y_start:y_end, x_start:x_end]
            
            # Write cropped frame
            out.write(cropped_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Successfully created trimmed video: {output_path}")
        print(f"Processed {frame_count} frames")
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False


def main() -> None:
    """Main function to handle command line arguments and execute video trimming."""
    parser = argparse.ArgumentParser(
        description="Trim video to show only table region using segmentation model"
    )
    
    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "output_video", 
        help="Path for output trimmed video"
    )
    
    parser.add_argument(
        "--model",
        default="trained_models/snkr_segm-egvem-3-roboflow-weights.pt",
        help="Path to table segmentation model (default: snkr_segm-egvem-3-roboflow-weights.pt)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around detected table in pixels (default: 20)"
    )
    
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=5,
        help="Number of frames to analyze for stable region (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        sys.exit(1)
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please ensure the segmentation model is available in the current directory")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the video
    success = trim_video_to_table(
        input_path=args.input_video,
        output_path=args.output_video,
        model_path=args.model,
        confidence=args.confidence,
        padding=args.padding,
        preview_frames=args.preview_frames
    )
    
    if success:
        print("Video trimming completed successfully!")
        sys.exit(0)
    else:
        print("Video trimming failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()