#!/usr/bin/env python3
"""
Script to track objects in a video using YOLOv11 and supervision.

This script processes a video file, detects objects using YOLOv11,
tracks them across frames, and generates an annotated output video.
"""

import os
import argparse
import time
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from CVModelInference.tracking import Tracker


def main():
    parser = argparse.ArgumentParser(description="Track objects in a video using YOLOv11")
    parser.add_argument(
        "--model",
        type=str,
        default="CVModelInference/trained_models/ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt",
        help="Path to model weights (.pt file)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/test_videos/filtered_ROS-Frame-2.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/output/tracked_video.mp4",
        help="Path to save output video"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (0-1)"
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=None,
        help="Limit processing FPS"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to process in seconds"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window during processing"
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Disable drawing motion traces"
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.isfile(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    if not os.path.isfile(args.video):
        print(f"Error: Video file does not exist: {args.video}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracker
    print(f"Initializing tracker with model: {args.model}")
    tracker = Tracker(
        model_path=args.model,
        confidence=args.confidence,
        iou=args.iou
    )
    
    # Process video
    print(f"Processing video: {args.video}")
    print(f"Output will be saved to: {args.output}")
    
    start_time = time.time()
    
    tracker.process_video(
        input_video_path=args.video,
        output_video_path=args.output,
        confidence=args.confidence,
        fps_limit=args.fps_limit,
        start_time=args.start_time,
        duration=args.duration,
        show_preview=args.preview,
        draw_traces=not args.no_traces
    )
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
