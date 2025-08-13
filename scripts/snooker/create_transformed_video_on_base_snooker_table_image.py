#!/usr/bin/env python3
"""
Create a video from transformed snooker detections overlaid on base table image.

This script loads transformed detection data from JSON, overlays colored balls
on the base table image for each frame, and generates a video output.
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray

import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.snooker.table.table_constants import (
    PLAY_AREA_TOP_LEFT_X, PLAY_AREA_TOP_LEFT_Y,
    BALL_SIZE, BALL_COLORS
)


def draw_ball_on_image(image: NDArray, x: float, y: float, ball_class: str, 
                      confidence: float = 1.0, ball_size: int | None = None) -> NDArray:
    """
    Draw a colored ball on the image at the specified coordinates.
    
    Args:
        image: Input image to draw on
        x: X coordinate (already in transformed coordinate system 0-2022)
        y: Y coordinate (already in transformed coordinate system 0-4056)
        ball_class: Ball class name (e.g., 'red', 'blue', 'white', etc.)
        confidence: Confidence score (0.0 to 1.0)
        ball_size: Size of the ball in pixels (defaults to BALL_SIZE)
        
    Returns:
        Image with ball drawn on it
    """
    if ball_size is None:
        ball_size = BALL_SIZE
    
    # The coordinates are already in the transformed coordinate system (0-2022 x 0-4056)
    # We need to map them to the full table image coordinates
    # The transformed area maps to the playing area on the full table
    table_x = int(x + PLAY_AREA_TOP_LEFT_X)
    table_y = int(y + PLAY_AREA_TOP_LEFT_Y)
    
    # Get ball color from constants
    ball_color: Tuple[int, int, int]
    if ball_class in BALL_COLORS:
        ball_color_tuple = BALL_COLORS[ball_class]["color"]
        ball_color = (int(ball_color_tuple[0]), int(ball_color_tuple[1]), int(ball_color_tuple[2]))  # type: ignore[index]
    else:
        # Default to white if unknown class
        ball_color = (255, 255, 255)
    
    # Adjust color based on confidence (darker for lower confidence)
    if confidence < 1.0:
        ball_color = (int(ball_color[0] * confidence), int(ball_color[1] * confidence), int(ball_color[2] * confidence))
    
    # Draw filled circle for the ball
    cv2.circle(image, (table_x, table_y), ball_size // 2, ball_color, -1)
    
    # Draw border for better visibility
    border_color = (0, 0, 0) if ball_class != "black" else (255, 255, 255)
    cv2.circle(image, (table_x, table_y), ball_size // 2, border_color, 2)
    
    # Add confidence text if confidence is low
    if confidence < 0.8:
        text = f"{confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text above the ball
        text_x = table_x - text_width // 2
        text_y = table_y - ball_size // 2 - 5
        
        # Draw text background
        cv2.rectangle(image, 
                     (text_x - 2, text_y - text_height - 2),
                     (text_x + text_width + 2, text_y + baseline + 2),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return image


def create_frame_from_detections(base_image: NDArray, detections: List[Dict], 
                                frame_number: int, time: float) -> NDArray:
    """
    Create a frame by overlaying detections on the base table image.
    
    Args:
        base_image: Base table image to overlay on
        detections: List of detection dictionaries for this frame
        frame_number: Frame number for display
        time: Time in seconds for display
        
    Returns:
        Frame with detections overlaid
    """
    # Create a copy of the base image
    frame = base_image.copy()
    
    # Draw each detection as a ball
    for detection in detections:
        x = detection.get("x", 0)
        y = detection.get("y", 0)
        ball_class = detection.get("class", "unknown")
        confidence = detection.get("confidence", 1.0)
        
        # Skip if coordinates are invalid
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            continue
        
        frame = draw_ball_on_image(frame, x, y, ball_class, confidence)
    
    # Add frame information overlay
    info_text = f"Frame: {frame_number} | Time: {time:.2f}s | Balls: {len(detections)}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def create_detection_video(transformed_detections_path: str, 
                          base_table_image_path: str,
                          output_video_path: str,
                          fps: int = 30,
                          ball_size: int | None = None) -> None:
    """
    Create a video from transformed detections overlaid on base table image.
    
    Args:
        transformed_detections_path: Path to transformed detections JSON file
        base_table_image_path: Path to base table image
        output_video_path: Path to output video file
        fps: Frames per second for the output video
        ball_size: Size of balls to draw (defaults to BALL_SIZE from constants)
    """
    print(f"Loading transformed detections from: {transformed_detections_path}")
    
    # Load transformed detections
    with open(transformed_detections_path, 'r') as f:
        all_detections = json.load(f)
    
    print(f"Loaded {len(all_detections)} frames")
    
    # Load base table image
    print(f"Loading base table image from: {base_table_image_path}")
    base_image = cv2.imread(base_table_image_path)
    if base_image is None:
        raise ValueError(f"Could not load base table image from {base_table_image_path}")
    
    print(f"Base image dimensions: {base_image.shape[1]}x{base_image.shape[0]}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    height, width = base_image.shape[:2]
    
    print(f"Creating video writer: {output_video_path} ({width}x{height}, {fps} fps)")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise ValueError(f"Could not create video writer for {output_video_path}")
    
    # Process each frame
    frame_count = 0
    total_frames = len(all_detections)
    
    for frame_data in all_detections:
        frame_count += 1
        
        frame_number = frame_data.get("frame", frame_count)
        time = frame_data.get("time", frame_count / fps)
        predictions = frame_data.get("predictions", [])
        
        # Create frame with detections overlaid
        frame = create_frame_from_detections(base_image, predictions, frame_number, time)
        
        # Write frame to video
        video_writer.write(frame)
        
        # Progress update
        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Processed frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # Release video writer
    video_writer.release()
    
    print(f"Video created successfully: {output_video_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Video duration: {frame_count / fps:.2f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create video from transformed snooker detections")
    parser.add_argument("--detections", 
                       default="/Users/abhinavrai/Playground/physics_inference/assets/output/transformed_detections.json",
                       help="Path to transformed detections JSON file")
    parser.add_argument("--base-image",
                       default="/Users/abhinavrai/Playground/physics_inference/src/snooker/table/base_table_image.png",
                       help="Path to base table image")
    parser.add_argument("--output",
                       default="/Users/abhinavrai/Playground/physics_inference/assets/output/detection_video.mp4",
                       help="Path to output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument("--ball-size", type=int, default=None, help="Size of balls to draw (defaults to BALL_SIZE)")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.detections):
        print(f"Error: Detections file not found: {args.detections}")
        sys.exit(1)
    
    if not os.path.exists(args.base_image):
        print(f"Error: Base image file not found: {args.base_image}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_detection_video(
            args.detections,
            args.base_image,
            args.output,
            args.fps,
            args.ball_size
        )
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
