#!/usr/bin/env python3
"""
Test snooker perspective transformation with real video from assets.

This script tests the perspective transformation setup using one of the
available test videos in the assets directory.
"""

import sys
import os
import cv2
import argparse
from pathlib import Path
from typing import List
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.snooker.perspective_transformer import PerspectiveTransformer
from src.snooker.table.table_generator import SnookerTableGenerator
from src.snooker.reference_validator import validate_transformation


def test_perspective_with_video(video_path: str, output_dir: str = "assets/output/snooker/video_test") -> bool:
    """
    Test perspective transformation setup with a real video.
    
    Args:
        video_path: Path to test video
        output_dir: Directory to save outputs
    """
    print(f"Testing perspective transformation with video: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Read first frame
    ret, frame_raw = cap.read()
    if not ret or frame_raw is None:
        print("Error: Could not read first frame")
        cap.release()
        return False
    
    frame = frame_raw.astype(np.uint8)
    
    print(f"Video loaded successfully: {frame.shape[1]}x{frame.shape[0]}")
    
    # Save original frame
    original_path = os.path.join(output_dir, "original_frame.png")
    cv2.imwrite(original_path, frame)
    print(f"Original frame saved to: {original_path}")
    
    # Create reference table for comparison
    generator = SnookerTableGenerator()
    reference_table = generator.load_base_table_image()
    reference_path = os.path.join(output_dir, "reference_table.png")
    cv2.imwrite(reference_path, reference_table)
    print(f"Reference table saved to: {reference_path}")
    
    # Setup perspective transformer
    transformer = PerspectiveTransformer()
    
    print("\nSetting up perspective transformation...")
    print("Please select the 4 corners of the snooker table in clockwise order:")
    print("1. Top-left corner of the playing area")
    print("2. Top-right corner of the playing area")
    print("3. Bottom-right corner of the playing area")
    print("4. Bottom-left corner of the playing area")
    
    # Setup transformation with manual point selection
    success = transformer.setup_transformation(frame)
    
    if success:
        print("✅ Perspective transformation setup successful!")
        
        # Save transformation config
        config_path = os.path.join(output_dir, "transformation_config.json")
        transformer.save_transformation_config(config_path)
        print(f"Transformation config saved to: {config_path}")
        
        # Transform the frame
        transformed = transformer.transform_frame(frame)
        if transformed is not None:
            transformed_path = os.path.join(output_dir, "transformed_frame.png")
            cv2.imwrite(transformed_path, transformed)
            print(f"Transformed frame saved to: {transformed_path}")
            
            # Validate the transformation
            print("\nValidating transformation quality...")
            validation_report = validate_transformation(
                transformed, 
                output_dir=os.path.join(output_dir, "validation")
            )
            
            if validation_report['validation_passed']:
                print("✅ Transformation validation PASSED")
            else:
                print("⚠️ Transformation validation FAILED")
                print("Check validation outputs for details")
            
            # Create visualization
            viz_path = os.path.join(output_dir, "transformation_visualization.png")
            transformer.visualize_transformation(frame, viz_path)
            
            # Process a few more frames to test consistency
            print("\nTesting transformation consistency with additional frames...")
            frame_count = 1
            for i in range(5):
                ret, frame_raw = cap.read()
                if ret and frame_raw is not None:
                    frame = frame_raw.astype(np.uint8)
                    transformed = transformer.transform_frame(frame)
                    if transformed is not None:
                        frame_path = os.path.join(output_dir, f"transformed_frame_{frame_count+1}.png")
                        cv2.imwrite(frame_path, transformed)
                        frame_count += 1
            
            print(f"Processed {frame_count} frames successfully")
            print("✅ Video test completed successfully!")
            print(f"Check the '{output_dir}' directory for all results.")
            
        else:
            print("❌ Failed to transform frame")
            return False
    else:
        print("❌ Perspective transformation setup failed")
        return False
    
    cap.release()
    return True


def list_available_videos() -> List[Path]:
    """List available test videos."""
    video_dir = Path("assets/test_videos")
    if not video_dir.exists():
        print("No test videos directory found")
        return []
    
    video_files = list(video_dir.glob("*.mp4"))
    print(f"\nAvailable test videos in {video_dir}:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {video.name}")
    
    return video_files


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test perspective transformation with real snooker videos"
    )
    
    parser.add_argument(
        "--video",
        help="Path to specific video file (if not provided, will list available videos)"
    )
    
    parser.add_argument(
        "--output",
        default="assets/output/snooker/video_test",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test videos and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_videos()
        return
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        video_path = args.video
    else:
        # List available videos and let user choose
        video_files = list_available_videos()
        if not video_files:
            print("No test videos found")
            return
        
        try:
            choice = input(f"\nEnter video number (1-{len(video_files)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return
            
            video_index = int(choice) - 1
            if 0 <= video_index < len(video_files):
                video_path = str(video_files[video_index])
            else:
                print("Invalid choice")
                return
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled")
            return
    
    print(f"\nTesting with video: {video_path}")
    success = test_perspective_with_video(video_path, args.output)
    
    if success:
        print("\n🎯 Video test completed successfully!")
    else:
        print("\n❌ Video test failed!")


if __name__ == "__main__":
    main()