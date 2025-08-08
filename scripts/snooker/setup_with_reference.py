#!/usr/bin/env python3
"""
Setup perspective transformation using a reference snooker table image.

This script helps you set up perspective transformation by comparing
your video frame with a reference snooker table markings image.
"""

import sys
import os
import cv2
import argparse
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Optional
from numpy.typing import NDArray

from src.snooker.perspective_transformer import PerspectiveTransformer
from src.snooker.table.table_generator import SnookerTableGenerator


def load_reference_image(reference_path: str) -> Optional[NDArray[np.uint8]]:
    """
    Load and display the reference snooker table image.
    
    Args:
        reference_path: Path to the reference image
        
    Returns:
        Reference image as numpy array
    """
    if not os.path.exists(reference_path):
        print(f"Error: Reference image not found at {reference_path}")
        return None
    
    loaded_image = cv2.imread(reference_path)
    if loaded_image is None:
        print(f"Error: Could not load reference image from {reference_path}")
        return None
    
    reference = loaded_image.astype(np.uint8)
    
    print(f"Reference image loaded: {reference.shape[1]}x{reference.shape[0]}")
    return reference


def show_reference_and_frame(reference: NDArray[np.uint8], frame: NDArray[np.uint8], window_name: str = "Reference vs Frame") -> None:
    """
    Display reference image and video frame side by side for comparison.
    
    Args:
        reference: Reference snooker table image
        frame: Video frame to compare
        window_name: Window name for display
    """
    # Resize images to same height for comparison
    target_height = 600
    
    ref_height, ref_width = reference.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    
    ref_scale = target_height / ref_height
    frame_scale = target_height / frame_height
    
    ref_resized = cv2.resize(reference, (int(ref_width * ref_scale), target_height)).astype(np.uint8)
    frame_resized = cv2.resize(frame, (int(frame_width * frame_scale), target_height)).astype(np.uint8)
    
    # Create side-by-side comparison
    combined_width = ref_resized.shape[1] + frame_resized.shape[1] + 20
    combined = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
    
    # Place reference image
    combined[:, :ref_resized.shape[1]] = ref_resized
    
    # Place frame
    start_x = ref_resized.shape[1] + 20
    combined[:, start_x:start_x + frame_resized.shape[1]] = frame_resized
    
    # Add labels
    cv2.putText(combined, "Reference Table", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Your Video Frame", (start_x + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the comparison
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)
    
    print("\nReference comparison displayed.")
    print("Press any key to continue with point selection...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def setup_transformation_with_reference(video_path: str, 
                                       reference_path: str, 
                                       output_dir: str = "assets/output/snooker/reference") -> bool:
    """
    Setup perspective transformation using reference image guidance.
    
    Args:
        video_path: Path to input video
        reference_path: Path to reference snooker table image
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference image
    reference = load_reference_image(reference_path)
    if reference is None:
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
    
    print(f"Video loaded: {frame.shape[1]}x{frame.shape[0]}")
    
    # Save original frame
    original_path = os.path.join(output_dir, "original_frame.png")
    cv2.imwrite(original_path, frame)
    
    # Save reference for comparison
    ref_copy_path = os.path.join(output_dir, "reference_copy.png")
    cv2.imwrite(ref_copy_path, reference)
    
    # Show reference and frame for comparison
    print("\nShowing reference table vs your video frame...")
    show_reference_and_frame(reference, frame)
    
    # Setup perspective transformer
    transformer = PerspectiveTransformer()
    
    print("\nNow select the 4 corners of the snooker table in your video frame.")
    print("Use the reference image as a guide to identify:")
    print("1. Top-left corner of the playing area")
    print("2. Top-right corner of the playing area")
    print("3. Bottom-right corner of the playing area")
    print("4. Bottom-left corner of the playing area")
    print("\nNote: Select the corners of the green playing surface, not the cushions!")
    
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
            
            # Create our standard table for comparison
            generator = SnookerTableGenerator()
            standard_table = generator.create_base_table()
            standard_path = os.path.join(output_dir, "standard_table.png")
            cv2.imwrite(standard_path, standard_table)
            
            # Create comparison with reference, transformed, and standard
            comparison = create_three_way_comparison(reference, transformed, standard_table)
            comparison_path = os.path.join(output_dir, "three_way_comparison.png")
            cv2.imwrite(comparison_path, comparison)
            print(f"Three-way comparison saved to: {comparison_path}")
            
            # Create visualization
            viz_path = os.path.join(output_dir, "transformation_visualization.png")
            transformer.visualize_transformation(frame, viz_path)
            
            print("✅ Setup completed successfully!")
            print(f"Check the '{output_dir}' directory for all results.")
            
        else:
            print("❌ Failed to transform frame")
            return False
    else:
        print("❌ Perspective transformation setup failed")
        return False
    
    cap.release()
    return True


def create_three_way_comparison(reference: NDArray[np.uint8], 
                               transformed: NDArray[np.uint8], 
                               standard: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Create a three-way comparison image.
    
    Args:
        reference: Reference snooker table image
        transformed: Transformed video frame
        standard: Our generated standard table
        
    Returns:
        Combined comparison image
    """
    target_height = 400
    
    # Resize all images to same height
    images = []
    for img in [reference, transformed, standard]:
        height, width = img.shape[:2]
        scale = target_height / height
        resized = cv2.resize(img, (int(width * scale), target_height)).astype(np.uint8)
        images.append(resized)
    
    # Calculate combined width
    total_width = sum(img.shape[1] for img in images) + 40  # 20px spacing between images
    combined = np.zeros((target_height + 50, total_width, 3), dtype=np.uint8)
    
    # Place images
    x_offset = 0
    labels = ["Reference", "Transformed", "Standard"]
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Place image
        combined[40:40+target_height, x_offset:x_offset+img.shape[1]] = img
        
        # Add label
        cv2.putText(combined, label, (x_offset + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        x_offset += img.shape[1] + 20
    
    return combined


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Setup perspective transformation using reference snooker table image"
    )
    
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--reference",
        default="/Users/abhinavrai/Playground/snooker/game_info/snooker table markings copy.png",
        help="Path to reference snooker table image"
    )
    
    parser.add_argument(
        "--output",
        default="assets/output/snooker/reference",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    print(f"Setting up perspective transformation...")
    print(f"Video: {args.video}")
    print(f"Reference: {args.reference}")
    print(f"Output: {args.output}")
    
    success = setup_transformation_with_reference(args.video, args.reference, args.output)
    
    if success:
        print("\n🎯 Setup completed successfully!")
        print("You can now use the saved transformation config for processing the full video.")
    else:
        print("\n❌ Setup failed!")


if __name__ == "__main__":
    main()