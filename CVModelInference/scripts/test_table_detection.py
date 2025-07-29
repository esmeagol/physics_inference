"""
Table Detection Test Script

This script demonstrates how to use the PureCV table detection module to detect
table boundaries in an image and save the result with the detected boundaries highlighted.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

# Import the PureCV module
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from PureCV import detect_green_table, perspective_transform

def highlight_table(image_path, output_path, hsv_lower=(35, 40, 40), hsv_upper=(85, 255, 255)):
    """
    Detect table in an image and save the result with highlighted boundaries.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
        hsv_lower: Lower bound for HSV threshold (H, S, V)
        hsv_upper: Upper bound for HSV threshold (H, S, V)
    """
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Detect the table
    result = detect_green_table(
        image,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        debug=False
    )
    
    if result is None:
        print("No table detected in the image.")
        return
    
    # Create a copy of the image to draw on
    output_image = image.copy()
    
    # Draw the detected contour
    cv2.polylines(output_image, [result.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)
    
    # Add corner points with numbers
    for i, point in enumerate(result):
        x, y = point.astype(int)
        # Draw a filled circle at the corner
        cv2.circle(output_image, (x, y), 10, (0, 0, 255), -1)
        # Add corner number
        cv2.putText(output_image, str(i+1), (x-5, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add a legend
    legend = "Green: Detected Table Boundary | Red: Corner Points"
    cv2.putText(output_image, legend, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the result
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, output_image)
    print(f"Saved result to: {output_path}")
    
    # Also save a warped version (top-down view)
    if len(result) == 4:
        warped = perspective_transform(image, result)
        warped_path = os.path.splitext(output_path)[0] + "_warped.jpg"
        cv2.imwrite(warped_path, warped)
        print(f"Saved warped table to: {warped_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect table boundaries in an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, default='output/detected_table.jpg', 
                       help='Path to save the output image (default: output/detected_table.jpg)')
    parser.add_argument('--hue-min', type=int, default=35, 
                       help='Minimum hue value (0-180), default: 35')
    parser.add_argument('--hue-max', type=int, default=85, 
                       help='Maximum hue value (0-180), default: 85')
    parser.add_argument('--sat-min', type=int, default=40, 
                       help='Minimum saturation (0-255), default: 40')
    parser.add_argument('--val-min', type=int, default=40, 
                       help='Minimum value (0-255), default: 40')
    
    args = parser.parse_args()
    
    # Define HSV range
    hsv_lower = (args.hue_min, args.sat_min, args.val_min)
    hsv_upper = (args.hue_max, 255, 255)
    
    print(f"Using HSV range: {hsv_lower} to {hsv_upper}")
    
    # Process the image
    try:
        highlight_table(
            image_path=args.image_path,
            output_path=args.output,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper
        )
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
