#!/usr/bin/env python3
"""
Test script for Roboflow video processing.

This script processes a video file, runs inference on each frame using Roboflow,
and creates an annotated output video.
"""

import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import cv2
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from CVModelInference.roboflow_local_inference import RoboflowLocal

# Load environment variables from .env file
load_dotenv()

# Default values from environment variables
DEFAULT_CONFIG = {
    'api_key': os.getenv('ROBOFLOW_API_KEY'),
    'workspace': os.getenv('ROBOFLOW_WORKSPACE', 'cvueplay'),
    'project': os.getenv('ROBOFLOW_PROJECT', 'snookers-gkqap'),
    'version': int(os.getenv('ROBOFLOW_MODEL_VERSION', '1')),
    'server_url': os.getenv('ROBOFLOW_SERVER_URL', 'http://localhost:9001'),
    'confidence': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
}

def process_video(
    input_video_path: str,
    output_dir: str,
    debug_dir: str,
    api_key: str = None,
    model_id: str = None,
    version: int = None,
    confidence: float = None,
    frame_skip: int = 0,
    server_url: str = None,
):
    # Apply defaults from environment variables if not provided
    if api_key is None:
        api_key = DEFAULT_CONFIG['api_key']
    if model_id is None:
        model_id = DEFAULT_CONFIG['project']
    if version is None:
        version = DEFAULT_CONFIG['version']
    if confidence is None:
        confidence = DEFAULT_CONFIG['confidence']
    if server_url is None:
        server_url = DEFAULT_CONFIG['server_url']
        
    # Validate required parameters
    if not api_key:
        raise ValueError("Roboflow API key is required. Set it in .env file or pass as argument.")
    if not model_id:
        raise ValueError("Model ID is required. Set it in .env file or pass as argument.")
    """
    Process a video file with Roboflow inference.
    
    Args:
        input_video_path: Path to input video file
        output_dir: Directory to save the output video
        debug_dir: Directory to save debug information
        api_key: Roboflow API key
        model_id: Roboflow model ID
        version: Model version number
        confidence: Minimum confidence threshold
        frame_skip: Number of frames to skip between processing (0 = process every frame)
        server_url: URL of the Roboflow inference server
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize Roboflow client
    roboflow_client = RoboflowLocal(
        api_key=api_key,
        model_id=model_id,
        version=version,
        server_url=server_url,
        confidence=confidence
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            
            # Convert BGR to RGB (Roboflow expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            try:
                predictions = roboflow_client.predict(rgb_frame)
                
                # Draw predictions on the frame
                annotated_frame = roboflow_client.visualize_predictions(
                    rgb_frame, predictions
                )
                
                # Convert back to BGR for video writing
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Save debug frame if needed
                if frame_count % 30 == 0:  # Save every 30th frame for debugging
                    debug_frame_path = os.path.join(
                        debug_dir,
                        f"{input_filename}_frame_{frame_count:06d}.jpg"
                    )
                    cv2.imwrite(debug_frame_path, annotated_frame_bgr)
                
                # Write the frame to the output video
                out.write(annotated_frame_bgr)
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

def main():
    parser = argparse.ArgumentParser(description="Process video with Roboflow inference")
    parser.add_argument(
        "--input",
        type=str,
        default="assets/test_videos/filtered_ROS-Frame-2.mp4",
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
        "--api-key",
        type=str,
        default=None,
        help=f"Roboflow API key (default: from .env: {DEFAULT_CONFIG['api_key'][:4]}...{DEFAULT_CONFIG['api_key'][-4:] if DEFAULT_CONFIG['api_key'] else ''})"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=f"Roboflow model ID (default: from .env: {DEFAULT_CONFIG['project']})"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help=f"Model version number (default: from .env: {DEFAULT_CONFIG['version']})"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=f"Minimum confidence threshold (0-1) (default: from .env: {DEFAULT_CONFIG['confidence']})"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Number of frames to skip between processing (0 = process every frame)"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help=f"URL of the Roboflow inference server (default: from .env: {DEFAULT_CONFIG['server_url']})"
    )
    
    args = parser.parse_args()
    
    process_video(
        input_video_path=args.input,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir,
        api_key=args.api_key,
        model_id=args.model_id,
        version=args.version,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        server_url=args.server_url
    )

if __name__ == "__main__":
    main()
