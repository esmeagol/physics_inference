#!/usr/bin/env python3
"""
Test script for the TableDetector class in table_detection.py

This script demonstrates how to use the TableDetector class to detect
and segment tables in images using a trained segmentation model.
"""

import os
import sys
import cv2
import numpy as np
import pytest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the TableDetector class
from src.detection.table_detection import TableDetector

def test_table_detector():
    """
    Test the TableDetector class with a sample image.
    """
    # Path to the model
    model_path = str(project_root / "trained_models" / "snkr_segm-egvem-3-roboflow-weights.pt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    # Path to test image
    test_image_path = str(project_root / "assets" / "test_images" / "230.jpg")
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        pytest.skip(f"Test image not found at {test_image_path}")
    
    print(f"Using model: {model_path}")
    print(f"Using test image: {test_image_path}")
    
    # Initialize the TableDetector
    detector = TableDetector(model_path=model_path, confidence=0.5)
    
    # Load the test image
    image = cv2.imread(test_image_path)
    assert image is not None, f"Could not load image {test_image_path}"
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Run table detection
    predictions = detector.predict(image)
    
    # Verify predictions structure
    assert 'model' in predictions, "Predictions missing 'model' key"
    assert 'image' in predictions, "Predictions missing 'image' key"
    assert 'predictions' in predictions, "Predictions missing 'predictions' key"
    
    # Verify image dimensions
    assert predictions['image']['width'] == image.shape[1], "Image width mismatch"
    assert predictions['image']['height'] == image.shape[0], "Image height mismatch"
    
    # There should be at least one detection (the table)
    assert len(predictions['predictions']) > 0, "No objects detected in the image"
    
    # Find table predictions
    table_predictions = [p for p in predictions['predictions'] if 'table' in p.get('class', '').lower()]
    assert len(table_predictions) > 0, "No table detected in the image"
    
    # Test table mask detection
    mask = detector.detect_table_mask(image)
    assert mask is not None, "Table mask detection failed"
    assert mask.shape == (image.shape[0], image.shape[1]), "Mask dimensions don't match image"
    assert np.count_nonzero(mask) > 0, "Mask is empty (all zeros)"
    
    # Test bounding box detection
    bbox = detector.get_table_bounding_box(image)
    assert bbox is not None, "Table bounding box detection failed"
    x, y, w, h = bbox
    assert w > 0 and h > 0, "Invalid bounding box dimensions"
    assert 0 <= x < image.shape[1], "Bounding box x-coordinate out of range"
    assert 0 <= y < image.shape[0], "Bounding box y-coordinate out of range"
    
    # Test table cropping
    cropped = detector.crop_to_table(image, padding=10)
    assert cropped is not None, "Table cropping failed"
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0, "Cropped image has invalid dimensions"
    
    # Test visualization
    visualization = detector.visualize_predictions(image, predictions)
    assert visualization is not None, "Visualization failed"
    assert visualization.shape == image.shape, "Visualization dimensions don't match original image"

if __name__ == "__main__":
    # When run as a script, execute the test and print detailed results
    try:
        test_table_detector()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()
