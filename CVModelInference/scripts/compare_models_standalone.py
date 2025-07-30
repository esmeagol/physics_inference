#!/usr/bin/env python3
"""
Standalone script to compare object detection results between two local PyTorch models on a directory of images.
Processes all images in a directory and generates annotated output images with side-by-side comparison.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
import numpy as np
from tqdm import tqdm
import glob
import sys

# Add parent directory to path to import from CVModelInference
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from CVModelInference.local_pt_inference import LocalPT


# LocalPT is imported from CVModelInference.local_pt_inference


def load_models(model1_path: str, model2_path: str, confidence: float = 0.5, iou: float = 0.5):
    """
    Load two local PyTorch models.
    
    Args:
        model1_path: Path to first model weights (.pt file)
        model2_path: Path to second model weights (.pt file)
        confidence: Confidence threshold for detections
        iou: IoU threshold for NMS
        
    Returns:
        Tuple of (model1, model2) as LocalPT instances
    """
    print(f"Loading model 1: {os.path.basename(model1_path)}")
    model1 = LocalPT(model_path=model1_path, confidence=confidence, iou=iou)
    
    print(f"Loading model 2: {os.path.basename(model2_path)}")
    model2 = LocalPT(model_path=model2_path, confidence=confidence, iou=iou)
    
    return model1, model2


def process_image(model, image: np.ndarray, confidence: float = 0.5) -> Dict:
    """
    Process a single image with the given model.
    
    Args:
        model: LocalPT instance
        image: Input image as numpy array
        confidence: Confidence threshold
        
    Returns:
        Dictionary containing detection results
    """
    start_time = time.time()
    result = model.predict(image, confidence=confidence)
    inference_time = time.time() - start_time
    
    # Adapt the output format to match what the rest of this script expects
    # The LocalPT class returns a different format than the original LocalPTModel
    return {
        "predictions": result.get("predictions", []),
        "model_info": model.get_model_info(),
        "inference_time": inference_time
    }


def create_side_by_side_image(image: np.ndarray, results1: Dict, results2: Dict, 
                             model1_name: str, model2_name: str) -> np.ndarray:
    """
    Create a side-by-side comparison image with detection results.
    
    Args:
        image: Original input image
        results1: Detection results from model 1
        results2: Detection results from model 2
        model1_name: Display name for model 1
        model2_name: Display name for model 2
        
    Returns:
        Side-by-side comparison image with annotations
    """
    # Create a copy of the image for each model
    image1 = image.copy()
    image2 = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Draw detections on each image
    draw_detections(image1, results1, model1_name)
    draw_detections(image2, results2, model2_name)
    
    # Create side-by-side comparison image
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    comparison[:, :width] = image1
    comparison[:, width:] = image2
    
    # Add a vertical line separating the two images
    cv2.line(comparison, (width, 0), (width, height), (255, 255, 255), 2)
    
    return comparison


def draw_detections(image: np.ndarray, results: Dict, model_name: str) -> None:
    """
    Draw detection results on an image.
    
    Args:
        image: Image to draw on
        results: Detection results
        model_name: Name of the model to display
    """
    predictions = results.get('predictions', [])
    
    # Count objects by class
    class_counts: dict[str, int] = {}
    for pred in predictions:
        cls = pred['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Draw each prediction
    for pred in predictions:
        # Get class and confidence
        cls = pred['class']
        conf = pred['confidence']
        
        # Determine color based on class (simple hash function for consistent colors)
        color_hash = hash(cls) % 0xFFFFFF
        color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
        
        # Draw bounding box
        x = int(pred['x'] - pred['width']/2)
        y = int(pred['y'] - pred['height']/2)
        w = int(pred['width'])
        h = int(pred['height'])
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label text
        label_text = f"{cls}: {conf:.2f}"
        
        # Calculate label position
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y, label_size[1])
        
        # Draw label background
        cv2.rectangle(
            image, 
            (x, label_y - label_size[1]), 
            (x + label_size[0], label_y + 5), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    # Draw model name and class counts at the top
    header_text = f"{model_name} - "
    for cls, count in class_counts.items():
        header_text += f"{cls}: {count}, "
    header_text = header_text.rstrip(", ")
    
    cv2.putText(
        image,
        header_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


def process_images(model1, model2, input_dir: str, output_dir: str,
                  model1_name: str, model2_name: str, confidence: float = 0.5,
                  image_extensions: List[str] | None = None) -> Dict:
    """
    Process all images in a directory with two models and generate comparison images.
    
    Args:
        model1: First LocalPTModel model
        model2: Second LocalPTModel model
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        model1_name: Display name for model 1
        model2_name: Display name for model 2
        confidence: Confidence threshold for detections
        image_extensions: List of image file extensions to process
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return {}
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    processed_count = 0
    start_time = time.time()
    
    # Track inference times for each model
    model1_inference_times = []
    model2_inference_times = []
    
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for image_file in image_files:
            try:
                # Read image
                image = cv2.imread(image_file)
                if image is None:
                    print(f"Failed to read image: {image_file}")
                    continue
                
                # Process with both models
                results1 = process_image(model1, image, confidence)
                results2 = process_image(model2, image, confidence)
                
                # Track inference times
                model1_inference_times.append(results1['inference_time'])
                model2_inference_times.append(results2['inference_time'])
                
                # Create side-by-side comparison
                comparison = create_side_by_side_image(
                    image, results1, results2, model1_name, model2_name
                )
                
                # Save output image
                filename = os.path.basename(image_file)
                output_path = os.path.join(output_dir, f"comparison_{filename}")
                cv2.imwrite(output_path, comparison)
                
                processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing image {image_file}: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Calculate inference statistics
    model1_avg_time = sum(model1_inference_times) / max(1, len(model1_inference_times))
    model2_avg_time = sum(model2_inference_times) / max(1, len(model2_inference_times))
    
    model1_it_per_sec = 1.0 / model1_avg_time if model1_avg_time > 0 else 0
    model2_it_per_sec = 1.0 / model2_avg_time if model2_avg_time > 0 else 0
    
    # Prepare results
    results = {
        "processed_count": processed_count,
        "total_time": total_time,
        "avg_time_per_image": total_time / max(1, processed_count),
        "model1": {
            "name": model1_name,
            "avg_inference_time": model1_avg_time,
            "it_per_sec": model1_it_per_sec
        },
        "model2": {
            "name": model2_name,
            "avg_inference_time": model2_avg_time,
            "it_per_sec": model2_it_per_sec
        }
    }
    
    # Print summary
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total images processed: {processed_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/max(1, processed_count):.2f} seconds")
    print("\nPer-model inference performance:")
    print(f"  {model1_name}:")
    print(f"    Average inference time: {model1_avg_time*1000:.2f} ms")
    print(f"    Inference it/sec: {model1_it_per_sec:.2f}")
    print(f"  {model2_name}:")
    print(f"    Average inference time: {model2_avg_time*1000:.2f} ms")
    print(f"    Inference it/sec: {model2_it_per_sec:.2f}")
    print(f"\nOutput images saved to: {os.path.abspath(output_dir)}")
    print("="*50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare two local PyTorch models on a directory of images')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model weights (.pt file)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model weights (.pt file)')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save output images')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--extensions', type=str, default='.jpg,.jpeg,.png,.bmp',
                       help='Comma-separated list of image file extensions to process')
    
    args = parser.parse_args()
    
    # Verify input files/directories exist
    if not os.path.isfile(args.model1):
        print(f"Error: Model 1 file does not exist: {args.model1}")
        return
    
    if not os.path.isfile(args.model2):
        print(f"Error: Model 2 file does not exist: {args.model2}")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse image extensions
    image_extensions = args.extensions.split(',')
    
    # Extract model names from file paths
    model1_name = os.path.splitext(os.path.basename(args.model1))[0]
    model2_name = os.path.splitext(os.path.basename(args.model2))[0]
    
    # Load models
    print(f"Loading model 1: {args.model1}")
    print(f"Loading model 2: {args.model2}")
    model1, model2 = load_models(
        args.model1,
        args.model2,
        confidence=args.confidence,
        iou=args.iou
    )
    
    # Process images
    results = process_images(
        model1, model2,
        args.input_dir,
        args.output_dir,
        model1_name,
        model2_name,
        confidence=args.confidence,
        image_extensions=image_extensions
    )
    
    # You can do additional processing with results if needed
    return results


if __name__ == "__main__":
    main()
