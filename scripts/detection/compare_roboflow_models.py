#!/usr/bin/env python3
"""
Script to compare object detection results between two Roboflow models.
Generates text output and annotated images for comparison.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from tqdm import tqdm

# Import the InferenceRunner implementations
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detection.roboflow_local_inference import RoboflowLocal

def parse_model_id(full_model_id: str) -> tuple[str, str, int]:
    """Parse a full model ID into workspace, model_id, and version."""
    # Format: workspace/model_id/version
    parts = full_model_id.split('/')
    if len(parts) != 3:
        raise ValueError(f"Invalid model ID format: {full_model_id}. Expected format: workspace/model_id/version")
    return parts[0], parts[1], int(parts[2])

def load_roboflow_models(api_key: str, model1_full_id: str, model2_full_id: str) -> tuple[RoboflowLocal, RoboflowLocal]:
    """
    Initialize and load two Roboflow models using the RoboflowLocal implementation.
    
    Args:
        api_key: Roboflow API key
        model1_full_id: First model full ID (e.g., 'workspace/model_id/1')
        model2_full_id: Second model full ID (e.g., 'workspace/model_id/1')
        
    Returns:
        Tuple of (model1, model2) as RoboflowLocal instances
    """
    # Parse model 1
    workspace1, model1_id, version1 = parse_model_id(model1_full_id)
    model1_id_full = f"{workspace1}/{model1_id}"
    print(f"Loading model 1: {model1_id_full} (v{version1})")
    model1 = RoboflowLocal(api_key=api_key, model_id=model1_id_full, version=version1)
    
    # Parse model 2
    workspace2, model2_id, version2 = parse_model_id(model2_full_id)
    model2_id_full = f"{workspace2}/{model2_id}"
    print(f"Loading model 2: {model2_id_full} (v{version2})")
    model2 = RoboflowLocal(api_key=api_key, model_id=model2_id_full, version=version2)
    
    return model1, model2

def process_image(model: RoboflowLocal, image_path: str, confidence: float = 0.5, overlap: int = 30) -> dict[str, Any]:
    """
    Process a single image with the given model.
    
    Args:
        model: RoboflowLocal model instance
        image_path: Path to input image
        confidence: Confidence threshold for detections
        overlap: Overlap threshold for NMS
        
    Returns:
        Detection results as a dictionary
    """
    # Process the image with the model
    result = model.predict(image_path, confidence=confidence, overlap=overlap)
    return dict(result) if result is not None else {}

def process_image_pair(model1: RoboflowLocal, model2: RoboflowLocal, image_path: str, output_dir: str, confidence: float = 0.5) -> None:
    """
    Process an image with both models and save results.
    
    Args:
        model1: First Roboflow model
        model2: Second Roboflow model
        image_path: Path to input image
        output_dir: Directory to save outputs
        confidence: Confidence threshold for detections
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    filename = Path(image_path).stem
    
    # Process with both models
    print(f"\nProcessing {filename}...")
    results1 = process_image(model1, image_path, confidence)
    results2 = process_image(model2, image_path, confidence)
    
    # Save text output
    save_text_results(results1, results2, filename, output_dir)
    
    # Generate and save annotated image
    save_annotated_comparison(image_path, results1, results2, filename, output_dir)

def save_text_results(results1: dict[str, Any], results2: dict[str, Any], filename: str, output_dir: str) -> None:
    """Save detection results to a text file."""
    output_path = os.path.join(output_dir, f"{filename}_results.txt")
    
    def format_detections(results: dict, model_name: str) -> str:
        output = [f"=== {model_name} ==="]
        predictions = results.get('predictions', [])
        
        # Count objects by class
        class_counts: dict[str, int] = {}
        for pred in predictions:
            cls = pred['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Add summary
        output.append(f"Total detections: {len(predictions)}")
        output.append("Class counts:")
        for cls, count in sorted(class_counts.items()):
            output.append(f"  {cls}: {count}")
            
        # Add detailed predictions
        output.append("\nDetailed predictions:")
        for i, pred in enumerate(predictions, 1):
            output.append(
                f"{i}. {pred['class']} - {pred['confidence']:.2f} "
                f"(x: {pred['x']:.0f}, y: {pred['y']:.0f}, "
                f"w: {pred['width']:.0f}, h: {pred['height']:.0f})"
            )
            
        return "\n".join(output)
    
    with open(output_path, 'w') as f:
        f.write(f"=== Object Detection Comparison ===\n")
        f.write(f"Image: {filename}\n")
        f.write("\n" + "-"*50 + "\n")
        f.write(format_detections(results1, "Model 1") + "\n")
        f.write("\n" + "-"*50 + "\n")
        f.write(format_detections(results2, "Model 2") + "\n")
    
    print(f"Saved text results to {output_path}")

def save_annotated_comparison(image_path: str, results1: dict[str, Any], results2: dict[str, Any], 
                            filename: str, output_dir: str) -> None:
    """Generate and save a side-by-side comparison of detections."""
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image {image_path}")
    
    # Create side-by-side comparison
    h, w = img.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = img  # Left side
    comparison[:, w:] = img  # Right side
    
    # Draw detections for model 1 (left)
    comparison = draw_detections(comparison, results1, (0, 0), "Model 1")
    
    # Draw detections for model 2 (right)
    comparison = draw_detections(comparison, results2, (w, 0), "Model 2")
    
    # Add dividing line
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
    
    # Save the result
    output_path = os.path.join(output_dir, f"{filename}_comparison.jpg")
    cv2.imwrite(output_path, comparison)
    print(f"Saved comparison image to {output_path}")

def draw_detections(img: np.ndarray, results: Dict[str, Any], offset: Tuple[int, int], 
                   model_name: str) -> np.ndarray:
    """Draw detections on the image with a summary."""
    x_offset, y_offset = offset
    h, w = img.shape[:2]
    
    # Define colors for different classes
    colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'pink': (203, 192, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'brown': (42, 42, 165)
    }
    
    # Draw each detection
    predictions = results.get('predictions', [])
    class_counts: Dict[str, int] = {}
    
    for pred in predictions:
        cls = pred['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Get color for this class
        color = colors.get(cls.lower(), (255, 255, 255))  # Default to white
        
        # Draw bounding box
        x = int(pred['x'] - pred['width']/2) + x_offset
        y = int(pred['y'] - pred['height']/2) + y_offset
        w = int(pred['width'])
        h = int(pred['height'])
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Add label with confidence
        label = f"{cls} {pred['confidence']:.2f}"
        cv2.putText(img, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add summary text
    summary = [f"{model_name}:", f"Total: {len(predictions)}"]
    for cls, count in sorted(class_counts.items()):
        summary.append(f"{cls}: {count}")
    
    # Draw semi-transparent background for text
    text_height = len(summary) * 25 + 10
    cv2.rectangle(img, (x_offset + 10, 10), 
                 (x_offset + 200, text_height), 
                 (0, 0, 0), -1)
    
    # Draw text
    for i, line in enumerate(summary):
        y = 30 + i * 25
        cv2.putText(img, line, (x_offset + 15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def process_directory(model1: RoboflowLocal, model2: RoboflowLocal, input_dir: str, output_dir: str, 
                     confidence: float = 0.5, limit: int | None = None) -> None:
    """Process all images in a directory."""
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    # Limit number of images if specified
    if limit and limit > 0:
        image_paths = image_paths[:limit]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            process_image_pair(
                model1, model2, 
                str(img_path), 
                output_dir,
                confidence=confidence
            )
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare two Roboflow models on a set of images.')
    parser.add_argument('--api-key', type=str, required=True, 
                       help='Roboflow API key')
    parser.add_argument('--model1', type=str, required=True,
                       help='First model full ID (e.g., "workspace/model_id/1")')
    parser.add_argument('--model2', type=str, required=True,
                       help='Second model full ID (e.g., "workspace/model_id/1")')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save output files')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    model1, model2 = load_roboflow_models(
        args.api_key,
        args.model1,
        args.model2
    )
    
    # Process all images in the input directory
    process_directory(
        model1, model2,
        args.input_dir,
        args.output_dir,
        confidence=args.confidence,
        limit=args.limit
    )
    
    print("\nProcessing complete!")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
# Test comment
# Test comment for pre-commit hook
