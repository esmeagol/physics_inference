#!/usr/bin/env python3
"""
Benchmark different inference approaches for Roboflow models using local inference server.

Compares FPS between:
1. Direct inference
2. Table-cropped and resized inference (640x640)
"""

import os
import time
import json
import base64
import requests
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import cv2
from tqdm import tqdm

# Import table detection functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PureCV.table_detection import detect_green_table, order_points

class InferenceBenchmark:
    def __init__(self, api_key: str, model1_id: str, model2_id: str, server_url: str = "http://localhost:9001"):
        """
        Initialize the benchmark with both models using local inference server.
        
        Args:
            api_key: Roboflow API key
            model1_id: First model ID (format: workspace/model_id/version)
            model2_id: Second model ID (format: workspace/model_id/version)
            server_url: URL of the local inference server (default: http://localhost:9001)
        """
        # Store configuration
        self.api_key = api_key
        self.server_url = server_url.rstrip('/')
        self.model1_id = model1_id
        self.model2_id = model2_id
        
        # Results storage
        self.results: dict[str, list[float]] = {
            'model1_direct': [],
            'model1_cropped': [],
            'model2_direct': [],
            'model2_cropped': []
        }
        
        # Check server status
        self._check_server_status()
    
    def _check_server_status(self):
        """Check if the local inference server is running and accessible."""
        try:
            response = requests.get(f"{self.server_url}/info")
            if response.status_code == 200:
                print(f"Connected to local inference server: {response.json()}")
            else:
                print(f"Warning: Server returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to local inference server: {e}")
            print("Make sure the server is running with: docker run -p 9001:9001 roboflow/roboflow-inference-server-cpu")
    
    def _infer(self, image: Union[str, np.ndarray], model_id: str) -> dict:
        """
        Send inference request to local server.
        
        Args:
            image: Either a path to an image or a numpy array
            model_id: Model ID in format workspace/model_id/version
            
        Returns:
            Dictionary with inference results
        """
        # Convert image to base64 if it's a numpy array
        if isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            image_data = base64.b64encode(buffer).decode('utf-8')
            data = {
                'api_key': self.api_key,
                'model_id': model_id,
                'image': {
                    'type': 'base64',
                    'value': image_data
                },
                'confidence': 0.5
            }
        else:
            # For file paths
            data = {
                'api_key': self.api_key,
                'model_id': model_id,
                'image': {
                    'type': 'file',
                    'value': image
                },
                'confidence': 0.5
            }
        
        try:
            response = requests.post(
                f"{self.server_url}/infer",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Inference request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
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
    
    def run_inference(self, model_id: str, image_path: str, preprocess: bool = False) -> Optional[float]:
        """
        Run inference using local inference server and return time taken in seconds.
        
        Args:
            model_id: Model ID in format workspace/model_id/version
            image_path: Path to input image
            preprocess: Whether to preprocess the image (crop table and resize)
            
        Returns:
            Time taken in seconds or None if error occurs
        """
        try:
            if preprocess:
                # Preprocess image
                preprocessed, success = self.preprocess_image(image_path)
                if not success:
                    return None
                
                # Run inference on preprocessed image (as numpy array)
                start_time = time.time()
                _ = self._infer(preprocessed, model_id)
                return time.time() - start_time
            else:
                # Direct inference from file path
                start_time = time.time()
                _ = self._infer(image_path, model_id)
                return time.time() - start_time
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def benchmark_image(self, image_path: str) -> Dict[str, float]:
        """
        Run all inference variants on a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with timing results for each inference variant
        """
        times = {}
        
        # Model 1 - Direct
        time_taken = self.run_inference(self.model1_id, image_path, preprocess=False)
        if time_taken is not None:
            times['model1_direct'] = time_taken
        
        # Model 1 - Cropped
        time_taken = self.run_inference(self.model1_id, image_path, preprocess=True)
        if time_taken is not None:
            times['model1_cropped'] = time_taken
        
        # Model 2 - Direct
        time_taken = self.run_inference(self.model2_id, image_path, preprocess=False)
        if time_taken is not None:
            times['model2_direct'] = time_taken
        
        # Model 2 - Cropped
        time_taken = self.run_inference(self.model2_id, image_path, preprocess=True)
        if time_taken is not None:
            times['model2_cropped'] = time_taken
        
        return times
    
    def run_inference(self, model_id: str, image_path: str, preprocess: bool = False) -> Optional[float]:
        """
        Run inference using local inference server and return time taken in seconds.
        
        Args:
            model_id: Model ID in format workspace/model_id/version
            image_path: Path to input image
        preprocess: Whether to preprocess the image (crop table and resize)
        
    Returns:
        Time taken in seconds or None if error occurs
    """
    try:
        if preprocess:
            # Preprocess image
            preprocessed, success = self.preprocess_image(image_path)
            if not success:
                return None
            
            # Run inference on preprocessed image (as numpy array)
            start_time = time.time()
            _ = self._infer(preprocessed, model_id)
            return time.time() - start_time
        else:
            # Direct inference from file path
            start_time = time.time()
            _ = self._infer(image_path, model_id)
            return time.time() - start_time
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def benchmark_image(self, image_path: str) -> Dict[str, float]:
    """
    Run all inference variants on a single image.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Dictionary with timing results for each inference variant
    """
    times = {}
    
    # Model 1 - Direct
    time_taken = self.run_inference(self.model1_id, image_path, preprocess=False)
    if time_taken is not None:
        times['model1_direct'] = time_taken
    
    # Model 1 - Cropped
    time_taken = self.run_inference(self.model1_id, image_path, preprocess=True)
    if time_taken is not None:
        times['model1_cropped'] = time_taken
    
    # Model 2 - Direct
    time_taken = self.run_inference(self.model2_id, image_path, preprocess=False)
    if time_taken is not None:
        times['model2_direct'] = time_taken
    
    # Model 2 - Cropped
    time_taken = self.run_inference(self.model2_id, image_path, preprocess=True)
    if time_taken is not None:
        times['model2_cropped'] = time_taken
    
    return times

    def run_benchmark(self, image_dir: str, num_images: int | None = None, warmup: int = 5) -> Dict[str, float]:
        """
        Run benchmark on all images in the directory.
        
        Args:
            image_dir: Directory containing input images
            num_images: Maximum number of images to process (None for all)
            warmup: Number of warmup iterations per model and approach
                
        Returns:
            Dictionary with average FPS for each approach
    """
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
        for model_id in [self.model1_id, self.model2_id]:
            for preprocess in [False, True]:
                try:
                    if preprocess:
                        preprocessed, success = self.preprocess_image(sample_image)
                        if success:
                            _ = self._infer(preprocessed, model_id)
                    else:
                        _ = self._infer(sample_image, model_id)
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
    parser = argparse.ArgumentParser(description='Benchmark Roboflow local inference approaches')
    parser.add_argument('--api-key', type=str, required=True,
                       help='Roboflow API key')
    parser.add_argument('--model1', type=str, required=True,
                       help='First model full ID (e.g., workspace/model_id/1)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Second model full ID (e.g., workspace/model_id/1)')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--server-url', type=str, default='http://localhost:9001',
                       help='URL of the local inference server (default: http://localhost:9001)')
    parser.add_argument('--num-images', type=int, default=None,
                       help='Number of images to process (default: all)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    args = parser.parse_args()
    
    # Verify input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Run benchmark
    print(f"Starting benchmark with local inference server at {args.server_url}")
    benchmark = InferenceBenchmark(
        args.api_key, 
        args.model1, 
        args.model2,
        server_url=args.server_url
    )
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
