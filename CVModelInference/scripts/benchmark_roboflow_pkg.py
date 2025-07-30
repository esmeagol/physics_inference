#!/usr/bin/env python3
"""
Benchmark different inference approaches for Roboflow models using the Roboflow Python package.

Compares FPS between:
1. Direct inference
2. Table-cropped and resized inference (640x640)
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any
import numpy as np
import cv2
from tqdm import tqdm
from roboflow import Roboflow

# Import table detection functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PureCV.table_detection import detect_green_table, order_points

class RoboflowBenchmark:
    def __init__(self, api_key: str, model1_id: str, model2_id: str):
        # Initialize Roboflow client
        self.rf = Roboflow(api_key=api_key)
        
        # Load models
        workspace1, model1, version1 = model1_id.split('/')
        workspace2, model2, version2 = model2_id.split('/')
        
        print(f"Loading model 1: {model1_id}")
        self.model1 = self.rf.workspace(workspace1).project(model1).version(int(version1)).model
        
        print(f"Loading model 2: {model2_id}")
        self.model2 = self.rf.workspace(workspace2).project(model2).version(int(version2)).model
        
        # Results storage
        self.results: dict[str, list[float]] = {
            'model1_direct': [],
            'model1_cropped': [],
            'model2_direct': [],
            'model2_cropped': []
        }
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (640, 640)) -> Tuple[Optional[np.ndarray], bool]:
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                return None, False
                
            # Detect table
            table_contour = detect_green_table(image)
            
            if table_contour is None:
                # If no table detected, use the whole image
                return cv2.resize(image, target_size), True
            
            # Get bounding rectangle with 5% margin
            x, y, w, h = cv2.boundingRect(table_contour)
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            
            # Apply margin (ensure within image bounds)
            h_img, w_img = image.shape[:2]
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_img, x + w + margin_x)
            y2 = min(h_img, y + h + margin_y)
            
            # Crop and resize
            cropped = image[y1:y2, x1:x2]
            return cv2.resize(cropped, target_size), True
            
        except Exception as e:
            print(f"Error in preprocessing image {image_path}: {e}")
            return None, False
    
    def run_inference(self, model, image_path: str, preprocess: bool = False) -> Optional[float]:
        try:
            if preprocess:
                # Preprocess image
                preprocessed, success = self.preprocess_image(image_path)
                if not success:
                    return None
                
                # Save preprocessed image to a temporary file
                temp_path = "/tmp/temp_preprocessed.jpg"
                cv2.imwrite(temp_path, preprocessed)
                
                # Run inference on preprocessed image
                start_time = time.time()
                result = model.predict(temp_path, confidence=0.5)
                return time.time() - start_time
            else:
                # Direct inference
                start_time = time.time()
                result = model.predict(image_path, confidence=0.5)
                return time.time() - start_time
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def benchmark_image(self, image_path: str) -> Dict[str, float]:
        times = {}
        
        # Model 1 - Direct
        time_taken = self.run_inference(self.model1, image_path, preprocess=False)
        if time_taken is not None:
            times['model1_direct'] = time_taken
        
        # Model 1 - Cropped
        time_taken = self.run_inference(self.model1, image_path, preprocess=True)
        if time_taken is not None:
            times['model1_cropped'] = time_taken
        
        # Model 2 - Direct
        time_taken = self.run_inference(self.model2, image_path, preprocess=False)
        if time_taken is not None:
            times['model2_direct'] = time_taken
        
        # Model 2 - Cropped
        time_taken = self.run_inference(self.model2, image_path, preprocess=True)
        if time_taken is not None:
            times['model2_cropped'] = time_taken
        
        return times
    
    def run_benchmark(self, image_dir: str, num_images: int | None = None, warmup: int = 1) -> Dict[str, float]:
        # Get list of image files
        image_files = [
            str(f) for f in Path(image_dir).glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return {}
        
        if num_images is not None:
            image_files = image_files[:num_images]
        
        print(f"Found {len(image_files)} images for benchmarking")
        
        # Use the first image for warmup
        sample_image = image_files[0]
        
        # Warmup runs
        print("Running warmup...")
        for _ in range(warmup):
            for model in [self.model1, self.model2]:
                for preprocess in [False, True]:
                    try:
                        if preprocess:
                            preprocessed, success = self.preprocess_image(sample_image)
                            if success:
                                temp_path = "/tmp/temp_preprocessed_warmup.jpg"
                                cv2.imwrite(temp_path, preprocessed)
                                _ = model.predict(temp_path, confidence=0.5)
                        else:
                            _ = model.predict(sample_image, confidence=0.5)
                    except Exception as e:
                        print(f"Warning: Warmup failed: {e}")
        
        # Run benchmark
        print("Running benchmark...")
        for image_path in tqdm(image_files, desc="Benchmarking"):
            times = self.benchmark_image(image_path)
            
            # Store results (converting time to FPS)
            for key, time_taken in times.items():
                if time_taken is not None and time_taken > 0:
                    self.results[key].append(1.0 / time_taken)  # Convert to FPS
        
        # Calculate average FPS for each approach
        avg_fps = {}
        for key, fps_values in self.results.items():
            if fps_values:
                avg_fps[key] = sum(fps_values) / len(fps_values)
            else:
                avg_fps[key] = 0.0
        
        return avg_fps

def main():
    parser = argparse.ArgumentParser(description='Benchmark Roboflow models using the Roboflow Python package')
    parser.add_argument('--api-key', type=str, required=True,
                       help='Roboflow API key')
    parser.add_argument('--model1', type=str, required=True,
                       help='First model full ID (e.g., workspace/model_id/version)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Second model full ID (e.g., workspace/model_id/version)')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--num-images', type=int, default=None,
                       help='Number of images to process (default: all)')
    parser.add_argument('--warmup', type=int, default=1,
                       help='Number of warmup iterations')
    args = parser.parse_args()
    
    # Verify input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Run benchmark
    print("Starting benchmark with Roboflow Python package")
    benchmark = RoboflowBenchmark(
        args.api_key, 
        args.model1, 
        args.model2
    )
    
    results = benchmark.run_benchmark(
        args.input_dir,
        num_images=args.num_images,
        warmup=args.warmup
    )
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Models: {args.model1} vs {args.model2}")
    print(f"Processed {len(next(iter(benchmark.results.values())))} images\n")
    
    print("Average FPS:")
    print(f"Model 1 (Direct): {results.get('model1_direct', 0):.2f} FPS")
    print(f"Model 1 (Cropped): {results.get('model1_cropped', 0):.2f} FPS")
    print(f"Model 2 (Direct): {results.get('model2_direct', 0):.2f} FPS")
    print(f"Model 2 (Cropped): {results.get('model2_cropped', 0):.2f} FPS")
    
    # Calculate speedup
    if 'model1_direct' in results and 'model1_cropped' in results and results['model1_direct'] > 0:
        speedup1 = (results['model1_cropped'] / results['model1_direct'] - 1) * 100
        print(f"\nModel 1 Speedup: {speedup1:+.2f}%")
    
    if 'model2_direct' in results and 'model2_cropped' in results and results['model2_direct'] > 0:
        speedup2 = (results['model2_cropped'] / results['model2_direct'] - 1) * 100
        print(f"Model 2 Speedup: {speedup2:+.2f}%")

if __name__ == "__main__":
    main()
