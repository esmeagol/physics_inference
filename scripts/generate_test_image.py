"""
Generate Test Image with Green Table

This script generates a test image with a green table that can be used to test
the table detection functionality.
"""

import cv2
import numpy as np

def create_test_image(output_path, width=800, height=600):
    """
    Create a test image with a green table.
    
    Args:
        output_path: Path to save the generated image
        width: Width of the output image
        height: Height of the output image
    """
    # Create a black background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define the table corners (slightly perspective distorted)
    table_corners = np.array([
        [width * 0.2, height * 0.2],   # Top-left
        [width * 0.8, height * 0.15],  # Top-right
        [width * 0.9, height * 0.8],   # Bottom-right
        [width * 0.1, height * 0.85]   # Bottom-left
    ], dtype=np.int32)
    
    # Draw the table surface (green)
    cv2.fillPoly(image, [table_corners], (0, 100, 0))
    
    # Add some texture to the table
    for i in range(100):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if cv2.pointPolygonTest(table_corners, (x, y), False) > 0:
            # Add small green variations for texture
            cv2.circle(image, (x, y), 1, 
                      (np.random.randint(0, 50), 
                       np.random.randint(100, 150), 
                       np.random.randint(0, 50)), -1)
    
    # Add a border around the table
    cv2.polylines(image, [table_corners], isClosed=True, color=(139, 69, 19), thickness=10)
    
    # Add some balls on the table
    ball_colors = [
        (255, 255, 255),  # White
        (255, 255, 0),    # Yellow
        (0, 0, 255),      # Red
        (0, 0, 0),        # Black
    ]
    
    for i, color in enumerate(ball_colors):
        # Position balls in a line
        x = int(width * 0.4 + i * 40)
        y = int(height * 0.5)
        
        # Only draw if inside the table
        if cv2.pointPolygonTest(table_corners, (x, y), False) > 0:
            cv2.circle(image, (x, y), 15, color, -1)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Generated test image: {output_path}")

if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Generate the test image
    output_path = os.path.join("test_images", "pool_table.jpg")
    create_test_image(output_path)
    
    print(f"Test image saved to: {os.path.abspath(output_path)}")
    print("You can now run the table detection on this image using:")
    print(f"python test_table_detection.py {output_path}")
