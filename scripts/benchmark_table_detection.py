"""
Table Detection Benchmark

This script benchmarks the table detection performance on the snooker dataset.
It processes all images in the specified directory, measures detection time,
and calculates success rate based on whether a table was detected.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

# Import the table detection functions
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from PureCV import detect_green_table, perspective_transform

class TableDetectionBenchmark:
    def __init__(self, data_dir: str, output_dir: str = "output/benchmark"):
        """
        Initialize the benchmark with data and output directories.
        
        Args:
            data_dir: Directory containing the test images
            output_dir: Directory to save benchmark results and visualizations
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.images_dir = self.output_dir / "images"
        
        # Create output directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            "start_time": datetime.now().isoformat(),
            "total_images": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "image_results": []
        }
    
    def find_images(self, sample_rate: int = 5) -> List[Path]:
        """
        Find all JPEG and PNG images in the data directory.
        
        Args:
            sample_rate: Sample every N images (e.g., 5 for every 5th image)
            
        Returns:
            List of image paths, sampled according to the sample rate
        """
        image_extensions = ['.jpg', '.jpeg', '.png']
        all_image_paths = []
        
        # Find all images
        for ext in image_extensions:
            all_image_paths.extend(sorted(self.data_dir.glob(f'*{ext}')))
        
        # Sample every N images
        sampled_paths = all_image_paths[::sample_rate]
        
        print(f"Found {len(all_image_paths)} total images, sampling {len(sampled_paths)} images (every {sample_rate}th image)")
        return sampled_paths
    
    def process_image(self, image_path: Path) -> Dict:
        """
        Process a single image and return the detection result.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing detection results and metrics
        """
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                "image_path": str(image_path),
                "success": False,
                "error": "Failed to load image",
                "processing_time": 0.0
            }
        
        # Measure detection time
        start_time = time.time()
        table_contour = detect_green_table(image)
        processing_time = time.time() - start_time
        
        result = {
            "image_path": str(image_path),
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "success": table_contour is not None,
            "processing_time": processing_time
        }
        
        # If table was detected, save the visualization
        if table_contour is not None:
            # Draw the contour on the image
            vis_image = image.copy()
            cv2.polylines(vis_image, [table_contour.astype(int)], True, (0, 255, 0), 3)
            
            # Add corner numbers
            for i, point in enumerate(table_contour):
                x, y = point.astype(int)
                cv2.circle(vis_image, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(vis_image, str(i+1), (x-5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save the visualization
            vis_path = self.images_dir / f"{image_path.stem}_detected.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            result["visualization_path"] = str(vis_path)
            
            # Try to create a warped view
            try:
                warped = perspective_transform(image, table_contour)
                warped_path = self.images_dir / f"{image_path.stem}_warped.jpg"
                cv2.imwrite(str(warped_path), warped)
                result["warped_path"] = str(warped_path)
            except Exception as e:
                result["warp_error"] = str(e)
        
        return result
    
    def run_benchmark(self, max_images: Optional[int] = None, sample_rate: int = 5):
        """
        Run the benchmark on sampled images from the data directory.
        
        Args:
            max_images: Maximum number of images to process (None for all)
            sample_rate: Sample every N images (e.g., 5 for every 5th image)
        """
        # Find sampled images in the data directory
        image_paths = self.find_images(sample_rate=sample_rate)
        
        if not image_paths:
            print(f"No images found in {self.data_dir}")
            return
        
        # Limit the number of images if specified
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        self.results["total_images"] = len(image_paths)
        print(f"Starting benchmark on {len(image_paths)} images...")
        
        # Process each image
        for i, image_path in enumerate(image_paths, 1):
            print(f"\nProcessing image {i}/{len(image_paths)}: {image_path.name}")
            
            try:
                # Process the image
                result = self.process_image(image_path)
                
                # Update statistics
                if result["success"]:
                    self.results["successful_detections"] += 1
                    print(f"  ✓ Table detected in {result['processing_time']:.2f}s")
                else:
                    self.results["failed_detections"] += 1
                    error_msg = result.get("error", "No table detected")
                    print(f"  ✗ {error_msg} in {result['processing_time']:.2f}s")
                
                self.results["total_processing_time"] += result["processing_time"]
                self.results["image_results"].append(result)
                
            except Exception as e:
                print(f"  ! Error processing {image_path.name}: {str(e)}")
                self.results["image_results"].append({
                    "image_path": str(image_path),
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                })
        
        # Calculate final statistics
        if self.results["total_images"] > 0:
            self.results["success_rate"] = (
                self.results["successful_detections"] / self.results["total_images"] * 100
            )
            self.results["average_processing_time"] = (
                self.results["total_processing_time"] / self.results["total_images"]
            )
        
        # Add end time
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_duration"] = (
            datetime.fromisoformat(self.results["end_time"]) - 
            datetime.fromisoformat(self.results["start_time"])
        ).total_seconds()
        
        # Save results to a JSON file
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save the benchmark results to a JSON file."""
        # Create a copy of results for saving (to handle datetime objects)
        results_to_save = self.results.copy()
        
        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        print("\n" + "="*60)
        print("TABLE DETECTION BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total images processed: {self.results['total_images']}")
        print(f"Successful detections: {self.results['successful_detections']} "
              f"({self.results.get('success_rate', 0):.1f}%)")
        print(f"Failed detections: {self.results['failed_detections']}")
        print(f"Total processing time: {self.results['total_processing_time']:.2f} seconds")
        print(f"Average processing time per image: {self.results.get('average_processing_time', 0):.3f} seconds")
        print(f"Total benchmark duration: {self.results.get('total_duration', 0):.2f} seconds")
        print("="*60)

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Benchmark table detection on snooker images')
    parser.add_argument('--data-dir', type=str, 
                       default='/Users/abhinavrai/Playground/snooker_data/step_2_training_data',
                       help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, 
                       default='output/benchmark',
                       help='Directory to save benchmark results')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Run the benchmark with sampling
    benchmark = TableDetectionBenchmark(args.data_dir, args.output_dir)
    benchmark.run_benchmark(max_images=args.max_images, sample_rate=5)  # Sample every 5th image

if __name__ == "__main__":
    main()
