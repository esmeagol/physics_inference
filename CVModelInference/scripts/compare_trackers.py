#!/usr/bin/env python
"""
Compare Trackers Script

This script demonstrates how to use the tracker benchmark system to compare
different tracking methodologies on a video.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from CVModelInference.local_pt_inference import LocalPT
from CVModelInference.tracker_benchmark import TrackerBenchmark
from CVModelInference.trackers.deepsort_tracker import DeepSORTTracker
from CVModelInference.trackers.sv_bytetrack_tracker import SVByteTrackTracker

# Try to import SnookerTrackerBenchmark if available
try:
    from CVModelInference.snooker_benchmark import SnookerTrackerBenchmark
    SNK_BENCHMARK_AVAILABLE = True
except ImportError:
    SNK_BENCHMARK_AVAILABLE = False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare different tracking methodologies')
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to input video (e.g., /path/to/snooker_video.mp4)')
    parser.add_argument('--model', type=str, help='Path to detection model weights (.pt file)')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output')
    parser.add_argument('--detection-interval', type=int, default=5, 
                        help='Interval (in frames) to run detection')
    parser.add_argument('--snooker', action='store_true', help='Use snooker-specific benchmark')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization frames')
    parser.add_argument('--save-frames', action='store_true', help='Save visualization frames to disk')
    parser.add_argument('--trackers', type=str, nargs='+', default=['deepsort', 'sv-bytetrack'],
                        help='List of trackers to benchmark (deepsort, sv-bytetrack)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize detection model if provided
    detection_model = None
    if args.model:
        print(f"Loading detection model from {args.model}")
        detection_model = LocalPT(model_path=args.model, confidence=0.25)
    
    # Initialize benchmark
    if args.snooker:
        if SNK_BENCHMARK_AVAILABLE:
            benchmark = SnookerTrackerBenchmark(detection_model)
            print("Using Snooker-specific benchmark")
        else:
            print("Warning: SnookerTrackerBenchmark not available. Using standard TrackerBenchmark.")
            benchmark = TrackerBenchmark(detection_model)
    else:
        benchmark = TrackerBenchmark(detection_model)
    
    # Trackers will be added later after potential initialization
    
    # Add trackers to benchmark based on user selection
    tracker_map = {
        'deepsort': ('DeepSORT', DeepSORTTracker(max_age=30, min_hits=3)),
        'sv-bytetrack': ('SV-ByteTrack', SVByteTrackTracker())
    }
    
    for tracker_name in args.trackers:
        tracker_name = tracker_name.lower()
        if tracker_name in tracker_map:
            benchmark.add_tracker(*tracker_map[tracker_name])
            print(f"Added tracker: {tracker_map[tracker_name][0]}")
        else:
            print(f"Warning: Unknown tracker '{tracker_name}'. Skipping.")
    
    # If we have a detection model, we can initialize trackers with first frame detections if needed
    if detection_model and args.model:
        # Open video to get first frame
        cap = cv2.VideoCapture(args.video)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            # Get detections for first frame
            predictions = detection_model.predict(first_frame)
            initial_detections = []
            
            # Convert predictions to detection format
            for pred in predictions.get('predictions', []):
                detection = {
                    'x': pred['x'],
                    'y': pred['y'],
                    'width': pred['width'],
                    'height': pred['height'],
                    'confidence': pred['confidence'],
                    'class': pred['class'],
                    'class_id': pred['class_id']
                }
                initial_detections.append(detection)
            
            # Note: The trackers will be initialized in the benchmark run_benchmark method
    
    # Run benchmark
    print(f"Running benchmark on {args.video}")
    results = benchmark.run_benchmark(
        args.video,
        detection_interval=args.detection_interval,
        output_dir=args.output_dir if args.visualize else None,
        visualize=args.visualize,
        save_frames=args.save_frames
    )
    
    # Print results
    benchmark.print_results()
    
    # Visualize results
    vis_path = os.path.join(args.output_dir, 'benchmark_results.png')
    benchmark.visualize_results(vis_path)
    
    # Generate tracking videos for each tracker
    if args.visualize:
        for tracker_name in results.keys():
            output_video = os.path.join(args.output_dir, f"{tracker_name}_tracking.mp4")
            benchmark.generate_tracking_video(args.video, tracker_name, output_video)
    
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
