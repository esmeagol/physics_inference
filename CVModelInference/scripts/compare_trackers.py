#!/usr/bin/env python
"""
Compare Trackers Script with Ground Truth Evaluation

This script demonstrates how to use the enhanced tracker benchmark system to compare
different tracking methodologies on a video with comprehensive ground truth evaluation.

Example Ground Truth Events Format:
[
    {"time": 0, "red": 15, "yellow": 1, "green": 1, "brown": 1, "blue": 1, "pink": 1, "black": 1, "white": 1},
    {"time": 42.0, "ball_potted": "red"},
    {"time_range": "41.5-42.5", "ball_potted": "red"},  # Time range format
    {"time": 120, "ball_potted": "black"},
    {"time": 135, "ball_placed_back": "black"},
    {"time_range": "120-125", "balls_occluded": {"red": 2, "blue": 1}},
    {"time_range": "130-135", "ignore_errors": "view_not_clear"},
    {"frame": 1500, "ball_potted": "red"}  # Frame-based event
]

Expected behavior for potting events with time ranges:
- Before time range starts: expect n balls
- During time range: expect n-1 or n balls (transition period)
- After time range ends: expect n-1 balls
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from typing import Dict, Any

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from detection.local_pt_inference import LocalPT
    from CVModelInference.tracker_benchmark import SnookerTrackerBenchmark
    from CVModelInference.trackers.deepsort_tracker import DeepSORTTracker
    from CVModelInference.trackers.sv_bytetrack_tracker import SVByteTrackTracker
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import benchmark components: {e}")
    BENCHMARK_AVAILABLE = False

# Try to import MOLT tracker if available
try:
    from PureCV.molt import MOLTTracker
    MOLT_AVAILABLE = True
except ImportError:
    MOLT_AVAILABLE = False


def load_ground_truth_events(file_path):
    """Load ground truth events from JSON file."""
    try:
        with open(file_path, 'r') as f:
            events = json.load(f)
        print(f"Loaded {len(events)} ground truth events from {file_path}")
        return events
    except Exception as e:
        print(f"Error loading ground truth events: {e}")
        return None


def generate_detailed_tracking_video(video_path: str, tracker_result: Dict[str, Any], 
                                   output_path: str, tracker_name: str) -> None:
    """
    Generate a detailed annotated tracking video showing all detection and tracking information.
    
    Args:
        video_path: Path to the original video
        tracker_result: Tracker result dictionary
        output_path: Path to save the annotated video
        tracker_name: Name of the tracker for display
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Get tracker results
    tracker_tracks = tracker_result.get('tracks', [])
    
    # Create a lookup for tracks by frame index
    tracks_by_frame = {}
    for track_data in tracker_tracks:
        frame_idx = track_data['frame_idx']
        tracks_by_frame[frame_idx] = track_data['tracks']
    
    # Define colors for different ball types
    ball_colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'brown': (42, 42, 165),
        'blue': (255, 0, 0),
        'pink': (203, 192, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'cue': (255, 255, 255),
        'unknown': (128, 128, 128)
    }
    
    print(f"  Processing {total_frames} frames...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracks for this frame
        tracks = tracks_by_frame.get(frame_idx, [])
        
        # Draw tracks on frame
        annotated_frame = frame.copy()
        
        # Count balls by type for display
        ball_counts = {}
        
        for track in tracks:
            # Get track information
            track_id = track.get('id', -1)
            x = int(track.get('x', 0))
            y = int(track.get('y', 0))
            width = int(track.get('width', 20))
            height = int(track.get('height', 20))
            confidence = track.get('confidence', 0.0)
            ball_class = track.get('class_name', 'unknown')
            
            # Normalize ball class
            ball_type = ball_class.lower()
            if ball_type in ['cue', 'cue_ball']:
                ball_type = 'white'
            elif ball_type not in ball_colors:
                ball_type = 'unknown'
            
            # Count balls
            if ball_type not in ball_counts:
                ball_counts[ball_type] = 0
            ball_counts[ball_type] += 1
            
            # Get color for this ball type
            color = ball_colors.get(ball_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x - width//2, y - height//2), 
                         (x + width//2, y + height//2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} {ball_type} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x - width//2, y - height//2 - label_size[1] - 5),
                         (x - width//2 + label_size[0], y - height//2), 
                         color, -1)
            
            # Draw label text
            text_color = (255, 255, 255) if ball_type == 'black' else (0, 0, 0)
            cv2.putText(annotated_frame, label, 
                       (x - width//2, y - height//2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Add frame information overlay
        info_y = 30
        cv2.putText(annotated_frame, f"{tracker_name} - Frame: {frame_idx}/{total_frames}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(annotated_frame, f"Time: {frame_idx/fps:.2f}s", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(annotated_frame, f"Total Tracks: {len(tracks)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add ball count information
        info_y += 30
        cv2.putText(annotated_frame, "Ball Counts:", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for ball_type, count in sorted(ball_counts.items()):
            info_y += 20
            color = ball_colors.get(ball_type, (128, 128, 128))
            cv2.putText(annotated_frame, f"  {ball_type}: {count}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add detection information if available
        if tracks:
            info_y += 30
            cv2.putText(annotated_frame, "Raw Track Data:", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show first few tracks' raw data
            for i, track in enumerate(tracks[:3]):  # Show first 3 tracks
                info_y += 15
                track_info = f"  T{i}: ID={track.get('id', 'N/A')} cls='{track.get('class_name', 'unknown')}' conf={track.get('confidence', 0):.3f}"
                cv2.putText(annotated_frame, track_info, 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Write frame
        out.write(annotated_frame)
        
        frame_idx += 1
        
        # Print progress every 100 frames
        if frame_idx % 100 == 0:
            print(f"    Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"  Detailed tracking video completed: {total_frames} frames processed")


def export_tracking_analysis(raw_results: Dict[str, Any], output_path: str) -> None:
    """
    Export detailed tracking analysis to a text file.
    
    Args:
        raw_results: Raw tracker results dictionary
        output_path: Path to save the analysis file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED TRACKING ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for tracker_name, tracker_result in raw_results.items():
            f.write(f"TRACKER: {tracker_name}\n")
            f.write("-" * 40 + "\n")
            
            tracks_data = tracker_result.get('tracks', [])
            f.write(f"Total frames with tracking data: {len(tracks_data)}\n")
            
            if not tracks_data:
                f.write("No tracking data available.\n\n")
                continue
            
            # Analyze track statistics
            all_track_ids = set()
            all_classes = set()
            class_counts = {}
            confidence_scores = []
            track_lifespans = {}
            
            for frame_data in tracks_data:
                frame_idx = frame_data.get('frame_idx', 0)
                tracks = frame_data.get('tracks', [])
                
                for track in tracks:
                    track_id = track.get('id', -1)
                    track_class = track.get('class_name', 'unknown')
                    confidence = track.get('confidence', 0.0)
                    
                    all_track_ids.add(track_id)
                    all_classes.add(track_class)
                    confidence_scores.append(confidence)
                    
                    # Count classes
                    if track_class not in class_counts:
                        class_counts[track_class] = 0
                    class_counts[track_class] += 1
                    
                    # Track lifespans
                    if track_id not in track_lifespans:
                        track_lifespans[track_id] = {'first': frame_idx, 'last': frame_idx, 'count': 0}
                    track_lifespans[track_id]['last'] = frame_idx
                    track_lifespans[track_id]['count'] += 1
            
            f.write(f"Unique track IDs: {len(all_track_ids)}\n")
            f.write(f"Detected classes: {sorted(all_classes)}\n")
            f.write(f"Total detections: {len(confidence_scores)}\n")
            
            if confidence_scores:
                f.write(f"Confidence scores - Min: {min(confidence_scores):.3f}, "
                       f"Max: {max(confidence_scores):.3f}, "
                       f"Avg: {sum(confidence_scores)/len(confidence_scores):.3f}\n")
            
            f.write("\nClass Distribution:\n")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(confidence_scores)) * 100 if confidence_scores else 0
                f.write(f"  {class_name}: {count} detections ({percentage:.1f}%)\n")
            
            f.write("\nTrack Lifespan Analysis:\n")
            lifespans = [data['last'] - data['first'] + 1 for data in track_lifespans.values()]
            detection_counts = [data['count'] for data in track_lifespans.values()]
            
            if lifespans:
                f.write(f"  Average track lifespan: {sum(lifespans)/len(lifespans):.1f} frames\n")
                f.write(f"  Longest track: {max(lifespans)} frames\n")
                f.write(f"  Shortest track: {min(lifespans)} frames\n")
                f.write(f"  Average detections per track: {sum(detection_counts)/len(detection_counts):.1f}\n")
            
            # Sample track details
            f.write("\nSample Track Details (first 10 tracks):\n")
            sample_tracks = list(track_lifespans.items())[:10]
            for track_id, data in sample_tracks:
                f.write(f"  Track ID {track_id}: frames {data['first']}-{data['last']} "
                       f"({data['count']} detections)\n")
            
            # Frame-by-frame analysis (first 10 frames)
            f.write("\nFrame-by-Frame Analysis (first 10 frames):\n")
            for i, frame_data in enumerate(tracks_data[:10]):
                frame_idx = frame_data.get('frame_idx', i)
                tracks = frame_data.get('tracks', [])
                f.write(f"  Frame {frame_idx}: {len(tracks)} tracks\n")
                
                for j, track in enumerate(tracks[:3]):  # First 3 tracks per frame
                    f.write(f"    Track {j}: ID={track.get('id', 'N/A')}, "
                           f"Class='{track.get('class_name', 'unknown')}', "
                           f"Conf={track.get('confidence', 0):.3f}, "
                           f"Pos=({track.get('x', 0):.1f},{track.get('y', 0):.1f}), "
                           f"Size=({track.get('width', 0):.1f}x{track.get('height', 0):.1f})\n")
                
                if len(tracks) > 3:
                    f.write(f"    ... and {len(tracks) - 3} more tracks\n")
            
            if len(tracks_data) > 10:
                f.write(f"  ... and {len(tracks_data) - 10} more frames\n")
            
            f.write("\n" + "=" * 80 + "\n\n")


def create_sample_ground_truth_events():
    """
    Create sample ground truth events for demonstration.
    This shows the expected format with mixed time and time range events.
    """
    return [
        # Initial state at video start
        {
            "time": 0, 
            "red": 15, "yellow": 1, "green": 1, "brown": 1, 
            "blue": 1, "pink": 1, "black": 1, "white": 1
        },
        
        # Single time point potting event
        {"time": 5.0, "ball_potted": "red"},
        
        # Time range potting event (demonstrates before/during/after logic)
        # Expect 13 reds before 10.0s, 12-13 reds during 10.0-12.0s, 12 reds after 12.0s
        {"time_range": "10.0-12.0", "ball_potted": "red"},
        
        # Another red potted
        {"time": 15.0, "ball_potted": "red"},
        
        # Yellow ball potted
        {"time": 20.0, "ball_potted": "yellow"},
        
        # Yellow ball placed back
        {"time": 22.0, "ball_placed_back": "yellow"},
        
        # Occlusion event - balls temporarily hidden but not potted
        {"time_range": "25.0-27.0", "balls_occluded": {"red": 2, "blue": 1}},
        
        # Error suppression during unclear video segment
        {"time_range": "28.0-30.0", "ignore_errors": "camera_shake"}
    ]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compare different tracking methodologies with optional ground truth evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic tracker comparison
  python compare_trackers.py --video snooker.mp4 --model weights.pt --snooker
  
  # Create sample ground truth events file
  python compare_trackers.py --create-sample-gt sample_events.json
  
  # Run with ground truth evaluation
  python compare_trackers.py --video snooker.mp4 --ground-truth events.json --moment-duration 0.5
  
  # Compare specific trackers with ground truth
  python compare_trackers.py --video snooker.mp4 --ground-truth events.json --trackers deepsort sv-bytetrack molt
  
  # Export detailed results with debug information
  python compare_trackers.py --video snooker.mp4 --ground-truth events.json --export-results results.json --debug-tracking
  
  # Generate annotated videos showing tracked objects
  python compare_trackers.py --video snooker.mp4 --visualize --debug-tracking
        """
    )
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
                        help='List of trackers to benchmark (deepsort, sv-bytetrack, molt)')
    
    # Ground truth evaluation arguments
    parser.add_argument('--ground-truth', type=str, 
                        help='Path to ground truth events JSON file')
    parser.add_argument('--moment-duration', type=float, default=0.5,
                        help='Duration of each evaluation moment in seconds (default: 0.5)')
    parser.add_argument('--export-results', type=str,
                        help='Export detailed results to JSON file')
    parser.add_argument('--create-sample-gt', type=str,
                        help='Create sample ground truth events file at specified path')
    parser.add_argument('--debug-tracking', action='store_true',
                        help='Enable detailed tracking debug output and analysis')
    
    args = parser.parse_args()
    
    # Handle sample ground truth creation
    if args.create_sample_gt:
        sample_events = create_sample_ground_truth_events()
        try:
            with open(args.create_sample_gt, 'w') as f:
                json.dump(sample_events, f, indent=2)
            print(f"Sample ground truth events created at {args.create_sample_gt}")
            print("Edit this file to match your video's actual events.")
            return
        except Exception as e:
            print(f"Error creating sample ground truth file: {e}")
            return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize detection model if provided
    detection_model = None
    if args.model:
        print(f"Loading detection model from {args.model}")
        detection_model = LocalPT(model_path=args.model, confidence=0.25)
    
    # Load ground truth events if provided
    ground_truth_events = None
    if args.ground_truth:
        ground_truth_events = load_ground_truth_events(args.ground_truth)
        if ground_truth_events is None:
            print("Failed to load ground truth events. Continuing without ground truth evaluation.")
    
    # Initialize benchmark
    use_ground_truth = ground_truth_events is not None
    
    if not BENCHMARK_AVAILABLE:
        print("Error: Benchmark components not available. Please check imports.")
        return
    
    # Always use SnookerTrackerBenchmark since it now has ground truth capabilities
    benchmark = SnookerTrackerBenchmark(detection_model)
    
    if use_ground_truth:
        try:
            benchmark.set_ground_truth_events(ground_truth_events)
            benchmark.set_moment_duration(args.moment_duration)
            print(f"Using Snooker benchmark with ground truth evaluation (moment duration: {args.moment_duration}s)")
        except Exception as e:
            print(f"Warning: Failed to set up ground truth evaluation: {e}")
            print("Continuing with standard benchmark...")
            use_ground_truth = False
    else:
        print("Using Snooker-specific benchmark")
    
    # Trackers will be added later after potential initialization
    
    # Add trackers to benchmark based on user selection
    tracker_map = {
        'deepsort': ('DeepSORT', DeepSORTTracker(max_age=30, min_hits=3)),
        'sv-bytetrack': ('SV-ByteTrack', SVByteTrackTracker())
    }
    
    # Add MOLT tracker if available
    if MOLT_AVAILABLE:
        tracker_map['molt'] = ('MOLT', MOLTTracker())
    
    for tracker_name in args.trackers:
        tracker_name = tracker_name.lower()
        if tracker_name in tracker_map:
            benchmark.add_tracker(*tracker_map[tracker_name])
            print(f"Added tracker: {tracker_map[tracker_name][0]}")
        elif tracker_name == 'molt' and not MOLT_AVAILABLE:
            print(f"Warning: MOLT tracker not available. Install PureCV.molt package. Skipping.")
        else:
            print(f"Warning: Unknown tracker '{tracker_name}'. Available: {list(tracker_map.keys())}. Skipping.")
    
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
    
    if use_ground_truth:
        # Run benchmark with ground truth evaluation
        try:
            results = benchmark.run_benchmark_with_ground_truth(
                args.video,
                detection_interval=args.detection_interval,
                output_dir=args.output_dir if args.visualize else None,
                visualize=args.visualize,
                save_frames=args.save_frames
            )
        except Exception as e:
            print(f"Error running ground truth benchmark: {e}")
            print("Falling back to standard benchmark...")
            results = benchmark.run_benchmark(
                args.video,
                detection_interval=args.detection_interval,
                output_dir=args.output_dir if args.visualize else None,
                visualize=args.visualize,
                save_frames=args.save_frames
            )
            use_ground_truth = False
    else:
        # Run standard benchmark
        results = benchmark.run_benchmark(
            args.video,
            detection_interval=args.detection_interval,
            output_dir=args.output_dir if args.visualize else None,
            visualize=args.visualize,
            save_frames=args.save_frames
        )
    
    # Print results
    benchmark.print_results()
    
    # Print ground truth specific results if available
    if use_ground_truth and hasattr(benchmark, 'ground_truth_results') and benchmark.ground_truth_results:
        print("\n" + "="*60)
        print("GROUND TRUTH EVALUATION SUMMARY")
        print("="*60)
        
        for tracker_name, gt_result in benchmark.ground_truth_results.items():
            if gt_result is not None:
                print(f"\n{tracker_name}:")
                print(f"  Overall Accuracy: {gt_result.overall_accuracy:.1f}%")
                print(f"  Moments Evaluated: {gt_result.moments_evaluated}/{gt_result.total_moments}")
                print(f"  Tracking Continuity: {gt_result.continuity_stats.get('continuity_percentage', 0):.1f}%")
                print(f"  Total Duplications: {gt_result.duplication_summary.get('total_duplication_errors', 0)}")
                
                # Show per-ball accuracy
                if gt_result.per_ball_accuracy:
                    print("  Per-ball accuracy:")
                    for ball_type, stats in gt_result.per_ball_accuracy.items():
                        print(f"    {ball_type}: {stats['accuracy']:.1f}% ({stats['correct_moments']}/{stats['total_moments']})")
            else:
                print(f"\n{tracker_name}: Ground truth evaluation failed")
    
    # Visualize results
    vis_path = os.path.join(args.output_dir, 'benchmark_results.png')
    benchmark.visualize_results(vis_path)
    
    # Export detailed results if requested
    if args.export_results:
        try:
            with open(args.export_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Detailed results exported to {args.export_results}")
        except Exception as e:
            print(f"Error exporting results: {e}")
    
    # Export raw tracker results for debugging if requested
    if args.debug_tracking:
        raw_results_path = os.path.join(args.output_dir, "raw_tracker_results.json")
        try:
            # Get the actual tracker results
            if use_ground_truth and isinstance(results, dict) and 'standard_benchmark' in results:
                raw_results = results['standard_benchmark']
            else:
                raw_results = results
                
            with open(raw_results_path, 'w') as f:
                json.dump(raw_results, f, indent=2, default=str)
            print(f"Raw tracker results exported to {raw_results_path}")
            
            # Print sample of raw results for immediate debugging
            print("\n" + "="*60)
            print("RAW TRACKER RESULTS SAMPLE")
            print("="*60)
            
            for tracker_name, tracker_result in raw_results.items():
                print(f"\n{tracker_name}:")
                tracks_data = tracker_result.get('tracks', [])
                print(f"  Total track frames: {len(tracks_data)}")
                
                # Show sample tracks from first few frames
                sample_frames = tracks_data[:3]  # First 3 frames
                for i, frame_data in enumerate(sample_frames):
                    frame_idx = frame_data.get('frame_idx', i)
                    tracks = frame_data.get('tracks', [])
                    print(f"  Frame {frame_idx}: {len(tracks)} tracks")
                    
                    # Show details of first few tracks
                    for j, track in enumerate(tracks[:2]):  # First 2 tracks per frame
                        print(f"    Track {j}: ID={track.get('id', 'N/A')}, "
                              f"Class={track.get('class_name', 'unknown')}, "
                              f"Conf={track.get('confidence', 'N/A'):.3f}, "
                              f"Pos=({track.get('x', 0):.1f},{track.get('y', 0):.1f})")
                        
                if len(tracks_data) > 3:
                    print(f"  ... and {len(tracks_data) - 3} more frames")
                    
        except Exception as e:
            print(f"Error exporting raw results: {e}")
    else:
        # Get raw results for other processing
        if use_ground_truth and isinstance(results, dict) and 'standard_benchmark' in results:
            raw_results = results['standard_benchmark']
        else:
            raw_results = results
    
    # Generate tracking videos for each tracker
    if args.visualize:
        print("\n" + "="*60)
        print("GENERATING ANNOTATED TRACKING VIDEOS")
        print("="*60)
        
        # Get the actual tracker names from the benchmark results
        if use_ground_truth and isinstance(results, dict) and 'standard_benchmark' in results:
            tracker_names = results['standard_benchmark'].keys()
            raw_results = results['standard_benchmark']
        elif hasattr(benchmark, 'results') and benchmark.results:
            tracker_names = benchmark.results.keys()
            raw_results = benchmark.results
        else:
            tracker_names = results.keys() if results else []
            raw_results = results
            
        for tracker_name in tracker_names:
            try:
                print(f"\nGenerating annotated video for {tracker_name}...")
                
                # Generate standard tracking video
                output_video = os.path.join(args.output_dir, f"{tracker_name}_tracking.mp4")
                benchmark.generate_tracking_video(args.video, tracker_name, output_video)
                print(f"  Standard tracking video: {output_video}")
                
                # Generate enhanced annotated video with detailed information
                enhanced_video = os.path.join(args.output_dir, f"{tracker_name}_detailed_tracking.mp4")
                generate_detailed_tracking_video(
                    args.video, 
                    raw_results[tracker_name], 
                    enhanced_video,
                    tracker_name
                )
                print(f"  Detailed tracking video: {enhanced_video}")
                
            except Exception as e:
                print(f"Warning: Could not generate tracking video for {tracker_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate comprehensive report if ground truth was used
    if use_ground_truth and hasattr(benchmark, 'ground_truth_results') and benchmark.ground_truth_results:
        try:
            report_dir = os.path.join(args.output_dir, "ground_truth_report")
            benchmark.generate_comprehensive_report(report_dir, include_visualizations=args.visualize)
            print(f"\nComprehensive ground truth report generated in: {report_dir}")
        except Exception as e:
            print(f"Warning: Failed to generate comprehensive report: {e}")
        
        # Export detailed tracking analysis if debug mode is enabled
        if args.debug_tracking:
            try:
                tracking_analysis_path = os.path.join(args.output_dir, "tracking_analysis.txt")
                export_tracking_analysis(raw_results, tracking_analysis_path)
                print(f"Detailed tracking analysis exported to: {tracking_analysis_path}")
            except Exception as e:
                print(f"Warning: Failed to export tracking analysis: {e}")
        
        # Print recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        for tracker_name, gt_result in benchmark.ground_truth_results.items():
            if gt_result is not None and hasattr(benchmark, 'ground_truth_evaluator'):
                try:
                    report = benchmark.ground_truth_evaluator.generate_detailed_report()
                    recommendations = report.get('recommendations', [])
                    if recommendations:
                        print(f"\n{tracker_name}:")
                        for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
                            print(f"  {i}. [{rec['priority'].upper()}] {rec['issue']}")
                            print(f"     → {rec['recommendation']}")
                    else:
                        print(f"\n{tracker_name}: No specific recommendations - performance is satisfactory")
                except Exception as e:
                    print(f"\n{tracker_name}: Could not generate recommendations: {e}")
    
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
