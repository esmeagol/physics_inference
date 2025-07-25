#!/usr/bin/env python3
"""
Benchmark different inference approaches for Roboflow models.

Compares FPS between:
1. Direct inference
2. Table-cropped and resized inference (640x640)
"""

import os
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import cv2
from tqdm import tqdm
from roboflow import Roboflow

# Import table detection functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PureCV.table_detection import detect_green_table, order_points

class InferenceBenchmark:
    def __init__(self, api_key: str, model1_id: str, model2_id: str):
        """Initialize the benchmark with both models."""
        self.rf = Roboflow(api_key=api_key)
        
        # Load both models
        print("Loading models...")
        self.model1 = self._load_model(model1_id)
        self.model2 = self._load_model(model2_id)
        
        # Results storage
        self.results = {
            'model1_direct': [],
            'model1_cropped': [],
            'model2_direct': [],
            'model2_cropped': []
        }
    
    def _load_model(self, full_model_id: str):
        """Load a Roboflow model from full ID (workspace/model_id/version)."""
        workspace, model_id, version = full_model_id.split('/')
        return self.rf.workspace(workspace).project(model_id).version(int(version)).model
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, bool]:
        """
        Preprocess image by detecting table, cropping with 5% margin, and resizing.
        
        Returns:
            Tuple of (preprocessed_image, success)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, False
        
        # Detect table
        table_contour = detect_green_table(
            image,
            hsv_lower=(35, 40, 40),
            hsv_upper=(85, 255, 255),
            min_area=0.1,
            max_aspect_ratio=2.5
        )
        
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
    
    def run_inference(self, model, image_path: str, preprocess: bool = False) -> float:
        """Run inference and return time taken in seconds."""
        try:
            if preprocess:
                # Preprocess image
                preprocessed, success = self.preprocess_image(image_path)
                if not success:
                    return None
                
                # Run inference
                start_time = time.time()
                _ = model.predict(preprocessed, confidence=0.5, overlap=30).json()
                return time.time() - start_time
            else:
                # Direct inference
                start_time = time.time()
                _ = model.predict(image_path, confidence=0.5, overlap=30).json()
                return time.time() - start_time
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def benchmark_image(self, image_path: str) -> Dict[str, float]:
        """Run all inference variants on a single image."""
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
    
    def run_benchmark(self, image_dir: str, num_images: int = None, warmup: int = 5) -> Dict[str, float]:
        """
        Run benchmark on all images in the directory.
        
        Args:
            image_dir: Directory containing images
            num_images: Number of images to process (None for all)
            warmup: Number of warmup iterations per model
            
        Returns:
            Dictionary with average FPS for each approach
        """
        # Get image paths
        image_exts = {'.jpg', '.jpeg', '.png'}
        image_paths = []
        
        for ext in image_exts:
            image_paths.extend(list(Path(image_dir).glob(f'*{ext}')))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        # Limit number of images if specified
        if num_images and num_images > 0:
            image_paths = image_paths[:num_images]
        
        print(f"Found {len(image_paths)} images for benchmarking")
        
        # Warmup runs (discard first few iterations)
        print("Running warmup...")
        for _ in range(warmup):
            for model in [self.model1, self.model2]:
                try:
                    _ = model.predict(str(image_paths[0]), confidence=0.5, overlap=30).json()
                except:
                    pass
        
        # Main benchmark
        print(f"Running benchmark on {len(image_paths)} images...")
        for img_path in tqdm(image_paths, desc="Benchmarking"):
            times = self.benchmark_image(str(img_path))
            
            # Store times
            for k, v in times.items():
                self.results[k].append(v)
        
        # Calculate average FPS
        avg_fps = {}
        for k, v in self.results.items():
            if v:  # Only calculate if we have valid measurements
                avg_time = sum(v) / len(v)
                avg_fps[k] = 1.0 / avg_time if avg_time > 0 else float('inf')
        
        return avg_fps

def main():
    parser = argparse.ArgumentParser(description='Benchmark Roboflow inference approaches')
    parser.add_argument('--api-key', type=str, required=True,
                       help='Roboflow API key')
    parser.add_argument('--model1', type=str, required=True,
                       help='First model full ID (e.g., workspace/model_id/1)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Second model full ID (e.g., workspace/model_id/1)')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--num-images', type=int, default=None,
                       help='Number of images to process (default: all)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = InferenceBenchmark(args.api_key, args.model1, args.model2)
    results = benchmark.run_benchmark(
        args.input_dir,
        num_images=args.num_images,
        warmup=args.warmup
    )
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Models: {args.model1} vs {args.model2}")
    print(f"Processed {len(benchmark.results['model1_direct'])} images")
    print("\nAverage FPS:")
    print(f"Model 1 (Direct): {results.get('model1_direct', 'N/A'):.2f} FPS")
    print(f"Model 1 (Cropped): {results.get('model1_cropped', 'N/A'):.2f} FPS")
    print(f"Model 2 (Direct): {results.get('model2_direct', 'N/A'):.2f} FPS")
    print(f"Model 2 (Cropped): {results.get('model2_cropped', 'N/A'):.2f} FPS")
    
    # Calculate speedup
    if 'model1_direct' in results and 'model1_cropped' in results:
        speedup1 = (results['model1_cropped'] / results['model1_direct'] - 1) * 100
        print(f"\nModel 1 Speedup: {speedup1:+.2f}%")
    
    if 'model2_direct' in results and 'model2_cropped' in results:
        speedup2 = (results['model2_cropped'] / results['model2_direct'] - 1) * 100
        print(f"Model 2 Speedup: {speedup2:+.2f}%")

if __name__ == "__main__":
    main()
