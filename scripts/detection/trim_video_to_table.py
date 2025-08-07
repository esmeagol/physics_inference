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

from detection.table_detection import TableDetector


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
    success = detector.process_video(
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