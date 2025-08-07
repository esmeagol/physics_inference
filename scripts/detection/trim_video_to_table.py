#!/usr/bin/env python3
"""
Video Table Trimming Script

This script processes video files to detect table regions and blacks out
areas outside the table, keeping only the table visible in the output video.

Supports two modes:
1. Automatic detection using a trained model
2. Manual point selection by the user
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detection.table_detection import TableDetector
import cv2
import numpy as np
from typing import Optional, Tuple, List


class PointSelector:
    """Interactive point selector for manual table region definition."""
    
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []
        self.window_name = "Select Points"
        self.image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Optional[object]) -> None:
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and self.image is not None:
            self.points.append((x, y))
            # Draw point on image
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.image, f"{len(self.points)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)
            
    def select_points(self, image: np.ndarray, num_points: int, instruction: str) -> List[Tuple[int, int]]:
        """Select points interactively from an image."""
        self.points = []
        self.original_image = image.copy()
        self.image = image.copy()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\n{instruction}")
        print(f"Click {num_points} points on the image. Press 'r' to reset, 'q' to quit, ENTER to confirm.")
        
        cv2.imshow(self.window_name, self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset points
                self.points = []
                self.image = self.original_image.copy()
                cv2.imshow(self.window_name, self.image)
                print("Points reset. Click again.")
                
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return []
                
            elif key == 13 or key == 10:  # Enter key
                if len(self.points) == num_points:
                    cv2.destroyAllWindows()
                    return self.points
                else:
                    print(f"Please select exactly {num_points} points. Currently selected: {len(self.points)}")


def get_detector_table_info(detector: TableDetector, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Get table bounding box and mask from detector."""
    bbox = detector.get_table_bounding_box(frame)
    mask = detector.detect_table_mask(frame)
    return bbox, mask


def get_manual_table_info(frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Get table bounding box and mask from manual point selection."""
    selector = PointSelector()
    
    # Get bounding box points (2 points: top-left, bottom-right)
    bbox_points = selector.select_points(frame, 2, 
        "Select 2 points for bounding box: 1) Top-left corner, 2) Bottom-right corner")
    
    if len(bbox_points) != 2:
        print("Bounding box selection cancelled.")
        return None, None
    
    # Calculate bounding box
    x1, y1 = bbox_points[0]
    x2, y2 = bbox_points[1]
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    bbox = (x, y, w, h)
    
    # Get quadrilateral points for mask
    quad_points = selector.select_points(frame, 4,
        "Select 4 points for table quadrilateral (clockwise from top-left)")
    
    if len(quad_points) != 4:
        print("Quadrilateral selection cancelled.")
        return bbox, None
    
    # Create mask from quadrilateral
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    quad_array = np.array(quad_points, dtype=np.int32)
    cv2.fillPoly(mask, [quad_array], 255)
    
    return bbox, mask


def apply_mask_to_frame(frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply mask to frame, blacking out areas outside the mask."""
    if mask is None:
        return frame
    
    # Ensure mask is the same size as frame
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply mask (black out areas where mask is 0)
    masked_frame = np.where(mask_3ch > 0, frame, 0)
    return masked_frame.astype(np.uint8)


def process_video(
    input_path: str,
    output_path: str,
    bbox: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    buffer_pixels: int = 0
) -> bool:
    """Process video with given bounding box and optional mask.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        bbox: Bounding box (x, y, width, height)
        mask: Optional mask for blacking out areas
        buffer_pixels: Buffer pixels around bounding box
        
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
        
        # Apply buffer to bounding box
        x, y, w, h = bbox
        x = max(0, x - buffer_pixels)
        y = max(0, y - buffer_pixels)
        w = min(width - x, w + 2 * buffer_pixels)
        h = min(height - y, h + 2 * buffer_pixels)
        
        print(f"Processing region: ({x}, {y}, {w}, {h})")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Process all frames
        frame_count = 0
        print("Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply mask if provided
            if mask is not None:
                frame = apply_mask_to_frame(frame, mask)
            
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
        "--mode",
        choices=["manual", "detector"],
        default="manual",
        help="Mode for table detection: 'manual' for point selection, 'detector' for automatic detection (default: manual)"
    )
    
    parser.add_argument(
        "--model",
        default="/Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-3-roboflow-weights.pt",
        help="Path to table segmentation model (used only in detector mode)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (used only in detector mode, default: 0.5)"
    )
    
    parser.add_argument(
        "--buffer",
        type=int,
        default=15,
        help="Buffer around detected/selected table in pixels (default: 15)"
    )
    
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=10,
        help="Number of frames to analyze for stable region (used only in detector mode, default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video to get first frame
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        sys.exit(1)
    
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read first frame from video")
        sys.exit(1)
    
    # Get table information based on mode
    bbox = None
    mask = None
    
    if args.mode == "detector":
        # Validate model file
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            print("Please ensure the segmentation model is available in the specified path")
            sys.exit(1)
        
        print(f"Loading table detection model: {args.model}")
        detector = TableDetector(model_path=args.model, confidence=args.confidence)
        
        # Sample frames to detect table region
        print(f"Sampling {args.sample_frames} frames to detect table region...")
        cap = cv2.VideoCapture(args.input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(min(args.sample_frames, total_frames)):
            frame_idx = i * (total_frames // args.sample_frames) if args.sample_frames < total_frames else i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Detect table in this frame
            frame_bbox, frame_mask = get_detector_table_info(detector, frame)
            if frame_bbox is not None:
                bbox = frame_bbox
                mask = frame_mask
                print(f"Table detected at frame {frame_idx}: {bbox}")
                break
        
        cap.release()
        
        if bbox is None:
            print("Error: No table detected in sampled frames")
            sys.exit(1)
            
    else:  # manual mode
        print("Manual point selection mode")
        bbox, mask = get_manual_table_info(first_frame)
        
        if bbox is None:
            print("Error: No bounding box selected")
            sys.exit(1)
    
    # Process the video
    success = process_video(
        input_path=args.input_video,
        output_path=args.output_video,
        bbox=bbox,
        mask=mask,
        buffer_pixels=args.buffer
    )
    
    if success:
        print("Video trimming completed successfully!")
        sys.exit(0)
    else:
        print("Video trimming failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()