"""
Process Test Images

This script processes all images in the assets/test_images directory using the table detection
and saves the results to the output directory.
"""

import os
import glob
import cv2
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

# Import the table detection function
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.test_table_detection import highlight_table

def process_test_images():
    """Process all test images in the assets/test_images directory."""
    # Create output directory if it doesn't exist
    output_dir = "output/test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg/jpeg/png images from the test_images directory
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(f"assets/test_images/{ext}"))
    
    if not image_paths:
        print("No test images found in assets/test_images/")
        return
    
    print(f"Found {len(image_paths)} test images.")
    print("Processing images...")
    
    # Process each image
    for i, image_path in enumerate(sorted(image_paths), 1):
        try:
            # Create output filename
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"detected_{filename}")
            
            print(f"\nProcessing {i}/{len(image_paths)}: {filename}")
            
            # Process with default HSV values first
            highlight_table(
                image_path=image_path,
                output_path=output_path,
                hsv_lower=(35, 40, 40),
                hsv_upper=(85, 255, 255)
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print("\nProcessing complete!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    process_test_images()
