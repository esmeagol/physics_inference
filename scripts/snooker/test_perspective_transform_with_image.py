#!/usr/bin/env python3
"""
Test snooker perspective transformation with test images from assets.

This script tests the perspective transformation setup using test images
from the assets/test_images directory.
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List
from numpy.typing import NDArray

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.snooker.perspective_transformer import PerspectiveTransformer
from src.snooker.table.table_generator import SnookerTableGenerator
from src.snooker.reference_validator import validate_transformation


def test_perspective_with_image(image_path: str, output_dir: str = "assets/output/snooker/image_test") -> bool:
    """
    Test perspective transformation setup with a test image.
    
    Args:
        image_path: Path to test image
        output_dir: Directory to save outputs
    """
    print(f"Testing perspective transformation with image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False
    
    # Load image
    loaded_image = cv2.imread(image_path)
    if loaded_image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    frame = loaded_image.astype(np.uint8)
    
    print(f"Image loaded successfully: {frame.shape[1]}x{frame.shape[0]}")
    
    # Save original frame to output directory
    original_path = os.path.join(output_dir, "original_image.png")
    cv2.imwrite(original_path, frame)
    print(f"Original image copied to: {original_path}")
    
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
    print("\nNote: Look for the green playing surface boundaries, not the outer cushions!")
    
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
            transformed_path = os.path.join(output_dir, "transformed_image.png")
            cv2.imwrite(transformed_path, transformed)
            print(f"Transformed image saved to: {transformed_path}")
            
            # Validate the transformation
            print("\nValidating transformation quality...")
            validation_report = validate_transformation(
                transformed, 
                output_dir=os.path.join(output_dir, "validation")
            )
            
            print(f"Validation score: {validation_report['overall_score']:.2f}")
            if validation_report['validation_passed']:
                print("✅ Transformation validation PASSED")
            else:
                print("⚠️ Transformation validation FAILED")
                print("Check validation outputs for details")
            
            # Create visualization
            viz_path = os.path.join(output_dir, "transformation_visualization.png")
            transformer.visualize_transformation(frame, viz_path)
            # Overlay transformed playing area onto base table
            composite_image = transformer.overlay_transformed_on_base_table(transformed, reference_table)
            composite_path = os.path.join(output_dir, "composite_overlay.png")
            cv2.imwrite(composite_path, composite_image)
            print(f"Composite overlay saved to: {composite_path}")

            # Create comparison with reference table
            comparison_path = os.path.join(output_dir, "comparison_with_reference.png")
            create_side_by_side_comparison(composite_image, reference_table, comparison_path)
            
            print("✅ Image test completed successfully!")
            print(f"Check the '{output_dir}' directory for all results.")
            
        else:
            print("❌ Failed to transform image")
            return False
    else:
        print("❌ Perspective transformation setup failed")
        return False
    
    return True


def create_side_by_side_comparison(transformed: NDArray[np.uint8], reference: NDArray[np.uint8], output_path: str) -> None:
    """Create a side-by-side comparison of transformed and reference images."""
    # Resize images to same height
    target_height = 800
    
    trans_height, trans_width = transformed.shape[:2]
    ref_height, ref_width = reference.shape[:2]
    
    trans_scale = target_height / trans_height
    ref_scale = target_height / ref_height
    
    trans_resized = cv2.resize(transformed, (int(trans_width * trans_scale), target_height)).astype(np.uint8)
    ref_resized = cv2.resize(reference, (int(ref_width * ref_scale), target_height)).astype(np.uint8)
    
    # Create combined image
    combined_width = trans_resized.shape[1] + ref_resized.shape[1] + 20
    combined = np.zeros((target_height + 50, combined_width, 3), dtype=np.uint8)
    
    # Place transformed image
    combined[40:40+target_height, :trans_resized.shape[1]] = trans_resized
    
    # Place reference image
    start_x = trans_resized.shape[1] + 20
    combined[40:40+target_height, start_x:start_x + ref_resized.shape[1]] = ref_resized
    
    # Add labels
    cv2.putText(combined, "Transformed", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Reference", (start_x + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, combined)
    print(f"Comparison image saved to: {output_path}")


def list_available_images() -> List[Path]:
    """List available test images."""
    image_dir = Path("assets/test_images")
    if not image_dir.exists():
        print("No test images directory found")
        return []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(image_dir.glob(ext)))
    
    print(f"\nAvailable test images in {image_dir}:")
    for i, image in enumerate(image_files, 1):
        print(f"{i}. {image.name}")
    
    return image_files


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test perspective transformation with test images"
    )
    
    parser.add_argument(
        "--image",
        help="Path to specific image file (if not provided, will list available images)"
    )
    
    parser.add_argument(
        "--output",
        default="assets/output/snooker/image_test",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test images and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_images()
        return
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        image_path = args.image
    else:
        # List available images and let user choose
        image_files = list_available_images()
        if not image_files:
            print("No test images found")
            return
        
        try:
            choice = input(f"\nEnter image number (1-{len(image_files)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return
            
            image_index = int(choice) - 1
            if 0 <= image_index < len(image_files):
                image_path = str(image_files[image_index])
            else:
                print("Invalid choice")
                return
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled")
            return
    
    print(f"\nTesting with image: {image_path}")
    success = test_perspective_with_image(image_path, args.output)
    
    if success:
        print("\n🎯 Image test completed successfully!")
    else:
        print("\n❌ Image test failed!")


if __name__ == "__main__":
    main()