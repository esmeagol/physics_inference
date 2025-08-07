#!/usr/bin/env python3
"""
Video Table Trimming Script

This script processes video files to detect table regions and blacks out
areas outside the table, keeping only the table visible in the output video.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.detection.table_detection import TableDetector
import cv2
import numpy as np
from typing import Optional


def process_video_with_table_detection(
    detector: TableDetector,
    input_path: str,
    output_path: str,
    sample_frames: int = 10,
    buffer_pixels: int = 20,
    save_mask_frames: bool = False
) -> bool:
    """Process video to trim to table region using table detection.
    
    Args:
        detector: TableDetector instance
        input_path: Path to input video
        output_path: Path to output video
        sample_frames: Number of frames to sample for table detection
        buffer_pixels: Buffer pixels around detected table
        save_mask_frames: Whether to save mask frames
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Sample frames to detect table region
        print(f"Sampling {sample_frames} frames to detect table region...")
        table_bbox = None
        
        for i in range(min(sample_frames, total_frames)):
            frame_idx = i * (total_frames // sample_frames) if sample_frames < total_frames else i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Detect table in this frame
            bbox = detector.get_table_bounding_box(frame)
            if bbox is not None:
                table_bbox = bbox
                print(f"Table detected at frame {frame_idx}: {bbox}")
                break
        
        if table_bbox is None:
            print("Error: No table detected in sampled frames")
            cap.release()
            return False
        
        # Add buffer to bounding box
        x, y, w, h = table_bbox
        x = max(0, x - buffer_pixels)
        y = max(0, y - buffer_pixels)
        w = min(width - x, w + 2 * buffer_pixels)
        h = min(height - y, h + 2 * buffer_pixels)
        
        print(f"Table region with buffer: ({x}, {y}, {w}, {h})")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Reset to beginning and process all frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        
        print("Processing video frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop to table region
            cropped_frame = frame[y:y+h, x:x+w]
            out.write(cropped_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video processing completed: {frame_count} frames processed")
        return True
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False


def main() -> None:
    """Main function to handle command line arguments and execute video processing."""
    parser = argparse.ArgumentParser(
        description="Process video to show only table region by blacking out areas outside the table"
    )
    
    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "output_video", 
        help="Path for output processed video"
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
        "--buffer",
        type=int,
        default=15,
        help="Buffer around detected table in pixels (default: 15)"
    )
    
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=10,
        help="Number of frames to analyze for stable region (default: 10)"
    )
    
    parser.add_argument(
        "--save-mask-frames",
        action="store_true",
        help="Save images of the first 10 frames with table mask overlay"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        sys.exit(1)
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please ensure the segmentation model is available in the specified path")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize table detector
    print(f"Loading table detection model: {args.model}")
    detector = TableDetector(model_path=args.model, confidence=args.confidence)
    
    # Process the video
    success = process_video_with_table_detection(
        detector=detector,
        input_path=args.input_video,
        output_path=args.output_video,
        sample_frames=args.sample_frames,
        buffer_pixels=args.buffer,
        save_mask_frames=args.save_mask_frames
    )
    
    if success:
        print("Video trimming completed successfully!")
        sys.exit(0)
    else:
        print("Video trimming failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()