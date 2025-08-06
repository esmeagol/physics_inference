"""
Tracker Benchmark Module for CVModelInference

This module provides functionality to benchmark and compare different tracking algorithms
with a focus on snooker-specific metrics.
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

from .tracker import Tracker
from src.detection.inference_runner import InferenceRunner
from .ground_truth_evaluator import GroundTruthEvaluator, EvaluationSummary
from .ground_truth_visualizer import GroundTruthVisualizer, VisualizationConfig


class TrackerBenchmark:
    """
    Class for benchmarking and comparing different tracking algorithms.
    """
    
    def __init__(self, detection_model: InferenceRunner | None = None):
        """
        Initialize the tracker benchmark.
        
        Args:
            detection_model: Optional detection model for generating detections
        """
        self.detection_model = detection_model
        self.trackers: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        
    def add_tracker(self, name: str, tracker: Tracker) -> None:
        """
        Add a tracker to the benchmark.
        
        Args:
            name: Name of the tracker
            tracker: Tracker instance
        """
        self.trackers[name] = tracker
        
    def run_benchmark(self, video_path: str, detection_interval: int = 5, 
                     output_dir: Optional[str] = None, visualize: bool = True,
                     save_frames: bool = False) -> Dict:
        """
        Run the benchmark on a video.
        
        Args:
            video_path: Path to the video file
            detection_interval: Interval (in frames) to run detection
            output_dir: Optional directory to save output visualizations
            visualize: Whether to generate visualization frames
            save_frames: Whether to save individual frame images (default: False)
            
        Returns:
            Dictionary containing benchmark results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if not self.trackers:
            raise ValueError("No trackers added to benchmark")
            
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")
        
        # Initialize results
        self.results = {
            name: {
                'processing_time': 0,
                'frames_processed': 0,
                'avg_fps': 0,
                'tracks': [],
                'metrics': {
                    'track_count': 0,
                    'track_switches': 0,
                    'track_fragmentations': 0,
                    'lost_tracks': 0,
                    'new_tracks': 0,
                    'track_length': []
                }
            }
            for name in self.trackers
        }
        
        # Process video
        frame_idx = 0
        detections_cache = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection at specified intervals or first frame
            if frame_idx % detection_interval == 0 or frame_idx == 0:
                if self.detection_model:
                    # Run detection
                    try:
                        predictions = self.detection_model.predict(frame)
                        detections = []
                        
                        # Convert predictions to detection format
                        for pred in predictions.get('predictions', []):
                            detection = {
                                'x': pred['x'],
                                'y': pred['y'],
                                'width': pred['width'],
                                'height': pred['height'],
                                'confidence': pred['confidence'],
                                'class': pred['class'],
                                'class_id': pred.get('class_id', 0)
                            }
                            detections.append(detection)
                            
                        detections_cache[frame_idx] = detections
                    except Exception as e:
                        print(f"Error running detection on frame {frame_idx}: {str(e)}")
                        detections_cache[frame_idx] = []
                else:
                    # No detection model, use empty detections
                    detections_cache[frame_idx] = []
            
            # Get detections for current frame
            current_detections = detections_cache.get(frame_idx, [])
            
            # Process frame with each tracker
            for name, tracker in self.trackers.items():
                tracker_result = self.results[name]
                
                # Initialize tracker if first frame
                if frame_idx == 0:
                    start_time = time.time()
                    tracker.init(frame, current_detections)
                    tracker_result['processing_time'] += time.time() - start_time
                    tracker_result['frames_processed'] += 1
                    continue
                
                # Update tracker
                start_time = time.time()
                tracks = tracker.update(frame, current_detections if frame_idx % detection_interval == 0 else None)
                process_time = time.time() - start_time
                
                # Update timing stats
                tracker_result['processing_time'] += process_time
                tracker_result['frames_processed'] += 1
                
                # Store tracks for this frame
                frame_tracks = {
                    'frame_idx': frame_idx,
                    'tracks': tracks,
                    'process_time': process_time
                }
                tracker_result['tracks'].append(frame_tracks)
                
                # Generate visualization
                if visualize and output_dir:
                    vis_frame = tracker.visualize(frame, tracks)
                    
                    # Add performance info
                    avg_time = tracker_result['processing_time'] / tracker_result['frames_processed']
                    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                    
                    cv2.putText(
                        vis_frame, f"{name} - FPS: {avg_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    
                    # Save visualization if requested
                    if save_frames:
                        output_path = os.path.join(output_dir, f"{name}_frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(output_path, vis_frame)
            
            # Print progress
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                
            frame_idx += 1
            
        # Close video
        cap.release()
        
        # Calculate final metrics
        for name, result in self.results.items():
            # Calculate average FPS
            avg_time = result['processing_time'] / result['frames_processed'] if result['frames_processed'] > 0 else 0
            result['avg_fps'] = 1.0 / avg_time if avg_time > 0 else 0
            
            # Calculate tracking metrics
            self._calculate_tracking_metrics(result)
            
        return self.results
    
    def _calculate_tracking_metrics(self, result: Dict) -> None:
        """
        Calculate tracking metrics from tracking results.
        
        Args:
            result: Tracking result dictionary
        """
        # Initialize tracking metrics
        metrics = result['metrics']
        
        # Track all object IDs across frames
        all_ids = set()
        id_first_seen = {}
        id_last_seen = {}
        id_frames = defaultdict(list)
        
        # Analyze tracks
        for frame_data in result['tracks']:
            frame_idx = frame_data['frame_idx']
            tracks = frame_data['tracks']
            
            # Record IDs in this frame
            frame_ids = set()
            
            for track in tracks:
                track_id = track['id']
                frame_ids.add(track_id)
                all_ids.add(track_id)
                
                # Update first/last seen
                if track_id not in id_first_seen:
                    id_first_seen[track_id] = frame_idx
                id_last_seen[track_id] = frame_idx
                
                # Record frame
                id_frames[track_id].append(frame_idx)
        
        # Calculate metrics
        metrics['track_count'] = len(all_ids)
        
        # Calculate track fragmentations and switches
        fragmentations = 0
        switches = 0
        
        for track_id, frames in id_frames.items():
            # Calculate gaps in tracking
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] > 1:
                    fragmentations += 1
                    
            # Track length
            track_length = id_last_seen[track_id] - id_first_seen[track_id] + 1
            metrics['track_length'].append(track_length)
        
        # Calculate track switches by analyzing consecutive frames
        prev_frame_ids = None
        for frame_data in result['tracks']:
            frame_ids = {track['id'] for track in frame_data['tracks']}
            
            if prev_frame_ids is not None:
                # Count new tracks
                new_tracks = len(frame_ids - prev_frame_ids)
                metrics['new_tracks'] += new_tracks
                
                # Count lost tracks
                lost_tracks = len(prev_frame_ids - frame_ids)
                metrics['lost_tracks'] += lost_tracks
                
                # Estimate switches based on simultaneous appearance/disappearance
                switches += min(new_tracks, lost_tracks)
                
            prev_frame_ids = frame_ids
            
        metrics['track_switches'] = switches
        metrics['track_fragmentations'] = fragmentations
        
        # Calculate average track length
        if metrics['track_length']:
            metrics['avg_track_length'] = sum(metrics['track_length']) / len(metrics['track_length'])
        else:
            metrics['avg_track_length'] = 0
    
    def visualize_results(self, output_path: Optional[str] = None) -> None:
        """
        Visualize benchmark results.
        
        Args:
            output_path: Optional path to save visualization
        """
        if not self.results:
            print("No benchmark results to visualize")
            return
            
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Processing speed (FPS)
        fps_values = [result['avg_fps'] for result in self.results.values()]
        axs[0, 0].bar(self.results.keys(), fps_values)
        axs[0, 0].set_title('Processing Speed (FPS)')
        axs[0, 0].set_ylabel('Frames per Second')
        
        # Plot 2: Track count
        track_counts = [result['metrics']['track_count'] for result in self.results.values()]
        axs[0, 1].bar(self.results.keys(), track_counts)
        axs[0, 1].set_title('Total Tracks')
        axs[0, 1].set_ylabel('Number of Tracks')
        
        # Plot 3: Track switches and fragmentations
        tracker_names = list(self.results.keys())
        track_switches = [result['metrics']['track_switches'] for result in self.results.values()]
        track_frags = [result['metrics']['track_fragmentations'] for result in self.results.values()]
        
        x = np.arange(len(tracker_names))
        width = 0.35
        
        axs[1, 0].bar(x - width/2, track_switches, width, label='Track Switches')
        axs[1, 0].bar(x + width/2, track_frags, width, label='Track Fragmentations')
        axs[1, 0].set_title('Tracking Stability')
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(tracker_names)
        axs[1, 0].legend()
        
        # Plot 4: Lost and new tracks
        lost_tracks = [result['metrics']['lost_tracks'] for result in self.results.values()]
        new_tracks = [result['metrics']['new_tracks'] for result in self.results.values()]
        
        axs[1, 1].bar(x - width/2, lost_tracks, width, label='Lost Tracks')
        axs[1, 1].bar(x + width/2, new_tracks, width, label='New Tracks')
        axs[1, 1].set_title('Track Continuity')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(tracker_names)
        axs[1, 1].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show figure
        if output_path:
            plt.savefig(output_path)
            print(f"Results visualization saved to {output_path}")
        else:
            plt.show()
    
    def print_results(self) -> None:
        """
        Print benchmark results to console.
        """
        if not self.results:
            print("No benchmark results to print")
            return
            
        print("\n===== TRACKER BENCHMARK RESULTS =====\n")
        
        # Print header
        header = f"{'Tracker':<15} | {'FPS':<8} | {'Tracks':<8} | {'Switches':<10} | {'Frags':<8} | {'Lost':<8} | {'New':<8}"
        print(header)
        print("-" * len(header))
        
        # Print results for each tracker
        for name, result in self.results.items():
            metrics = result['metrics']
            print(f"{name:<15} | {result['avg_fps']:<8.2f} | {metrics['track_count']:<8} | "
                 f"{metrics['track_switches']:<10} | {metrics['track_fragmentations']:<8} | "
                 f"{metrics['lost_tracks']:<8} | {metrics['new_tracks']:<8}")
            
        print("\n=== Detailed Metrics ===\n")
        
        for name, result in self.results.items():
            metrics = result['metrics']
            print(f"{name}:")
            print(f"  - Processing time: {result['processing_time']:.2f} seconds")
            print(f"  - Frames processed: {result['frames_processed']}")
            print(f"  - Average FPS: {result['avg_fps']:.2f}")
            print(f"  - Total tracks: {metrics['track_count']}")
            print(f"  - Track switches: {metrics['track_switches']}")
            print(f"  - Track fragmentations: {metrics['track_fragmentations']}")
            print(f"  - Lost tracks: {metrics['lost_tracks']}")
            print(f"  - New tracks: {metrics['new_tracks']}")
            print(f"  - Average track length: {metrics.get('avg_track_length', 0):.2f} frames")
            print()
            
    def generate_tracking_video(self, video_path: str, tracker_name: str, 
                              output_path: str) -> None:
        """
        Generate a video with tracking visualization for a specific tracker.
        
        Args:
            video_path: Path to the original video
            tracker_name: Name of the tracker to visualize
            output_path: Path to save the output video
        """
        if tracker_name not in self.results:
            raise ValueError(f"No results found for tracker: {tracker_name}")
            
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
        
        # Create video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Get tracker results
        tracker = self.trackers[tracker_name]
        tracker_results = self.results[tracker_name]['tracks']
        
        # Process video
        frame_idx = 0
        result_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Find corresponding tracking result
            tracks = []
            if result_idx < len(tracker_results) and tracker_results[result_idx]['frame_idx'] == frame_idx:
                tracks = tracker_results[result_idx]['tracks']
                result_idx += 1
                
            # Visualize tracks
            vis_frame = tracker.visualize(frame, tracks)
            
            # Add frame number
            cv2.putText(
                vis_frame, f"Frame: {frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Write frame
            out.write(vis_frame)
            
            frame_idx += 1
            
        # Release resources
        cap.release()
        out.release()
        
        print(f"Tracking video saved to {output_path}")


class SnookerTrackerBenchmark(TrackerBenchmark):
    """
    Specialized tracker benchmark for snooker videos with snooker-specific metrics.
    """
    
    def __init__(self, detection_model: InferenceRunner | None = None):
        """
        Initialize the snooker tracker benchmark.
        
        Args:
            detection_model: Optional detection model for generating detections
        """
        super().__init__(detection_model)
        
        # Ground truth evaluation components
        self.ground_truth_evaluator: Optional[GroundTruthEvaluator] = None
        self.ground_truth_events: List[Dict[str, Any]] = []
        self.moment_duration: float = 1.0  # Default 1 second per moment
        self.ground_truth_results: Dict[str, Optional[EvaluationSummary]] = {}
        
        # Visualization components
        self.visualizer = GroundTruthVisualizer()
        
    def run_benchmark(self, video_path: str, detection_interval: int = 5, 
                     output_dir: Optional[str] = None, visualize: bool = True,
                     save_frames: bool = False) -> Dict:
        """
        Run the benchmark on a snooker video.
        
        Args:
            video_path: Path to the video file
            detection_interval: Interval (in frames) to run detection
            output_dir: Optional directory to save output visualizations
            visualize: Whether to generate visualization frames
            save_frames: Whether to save individual frame images (default: False)
            
        Returns:
            Dictionary containing benchmark results
        """
        # Run standard benchmark
        results = super().run_benchmark(video_path, detection_interval, output_dir, visualize, save_frames)
        
        # Calculate snooker-specific metrics
        for name, result in results.items():
            # Initialize snooker metrics
            result['snooker_metrics'] = {
                'ball_id_consistency': 0,
                'cue_ball_tracking': 0,
                'color_ball_tracking': 0,
                'red_ball_tracking': 0,
                'pocket_detection': 0,
                'collision_detection': 0
            }
            
            # Calculate snooker-specific metrics
            self._calculate_snooker_metrics(result)
            
        return results
    
    def _calculate_snooker_metrics(self, result: Dict) -> None:
        """
        Calculate snooker-specific tracking metrics.
        
        Args:
            result: Tracking result dictionary
        """
        # Initialize metrics
        snooker_metrics = result['snooker_metrics']
        
        # Track balls by class
        ball_tracks = defaultdict(list)
        ball_ids = defaultdict(set)
        
        # Analyze tracks
        for frame_data in result['tracks']:
            tracks = frame_data['tracks']
            
            for track in tracks:
                if 'class' in track:
                    ball_class = track['class']
                    ball_tracks[ball_class].append(track)
                    ball_ids[ball_class].add(track['id'])
        
        # Calculate ball ID consistency (lower is better)
        # Ideally, each ball class should have a consistent ID
        for ball_class, ids in ball_ids.items():
            if ball_class.lower() in ['cue', 'cue ball', 'white']:
                # Cue ball should have exactly one ID
                snooker_metrics['cue_ball_tracking'] = 100 if len(ids) == 1 else max(0, 100 - (len(ids) - 1) * 20)
            elif ball_class.lower() in ['red', 'red ball']:
                # Red balls can have multiple IDs but should be consistent
                snooker_metrics['red_ball_tracking'] = max(0, 100 - len(ids) * 2)
            elif ball_class.lower() in ['yellow', 'green', 'brown', 'blue', 'pink', 'black']:
                # Color balls should have one ID each
                snooker_metrics['color_ball_tracking'] += 100 if len(ids) == 1 else max(0, 100 - (len(ids) - 1) * 20)
        
        # Normalize color ball tracking
        color_count = sum(1 for c in ball_ids if c.lower() in ['yellow', 'green', 'brown', 'blue', 'pink', 'black'])
        if color_count > 0:
            snooker_metrics['color_ball_tracking'] /= color_count
        
        # Calculate overall ball ID consistency
        total_balls = len(ball_ids)
        total_ids = sum(len(ids) for ids in ball_ids.values())
        if total_balls > 0:
            # Perfect consistency would be 1 ID per ball class
            snooker_metrics['ball_id_consistency'] = max(0, 100 - (total_ids - total_balls) * 5)
        
        # Detect potential pockets (corners of the table)
        # This is a simplified heuristic - in a real implementation, 
        # you would detect actual pockets or use table coordinates
        pocket_detection = 0
        
        # Detect potential collisions
        # Simple heuristic: count frames where balls are close to each other
        collision_detection = 0
        
        # Set placeholder values for metrics that need video analysis
        snooker_metrics['pocket_detection'] = pocket_detection
        snooker_metrics['collision_detection'] = collision_detection
    


    def set_ground_truth_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Set ground truth events for evaluation.
        
        Args:
            events: List of ground truth event dictionaries
            
        Raises:
            ValueError: If events are invalid
        """
        self.ground_truth_events = events
        
        # Initialize ground truth evaluator
        self.ground_truth_evaluator = GroundTruthEvaluator(
            video_fps=30.0,  # Default FPS, can be updated when video is processed
            detection_interval=1,
            distance_threshold=50.0
        )
        
        try:
            self.ground_truth_evaluator.set_ground_truth_events(events)
            print(f"Ground truth events set successfully: {len(events)} events")
        except Exception as e:
            raise ValueError(f"Failed to set ground truth events: {e}")
    
    def set_moment_duration(self, duration: float) -> None:
        """
        Set the duration for each evaluation moment.
        
        Args:
            duration: Duration in seconds for each moment
            
        Raises:
            ValueError: If duration is not positive
        """
        if duration <= 0:
            raise ValueError("Moment duration must be positive")
        
        self.moment_duration = duration
        
        if self.ground_truth_evaluator:
            self.ground_truth_evaluator.set_moment_duration(duration)
    
    def run_benchmark_with_ground_truth(self, video_path: str, detection_interval: int = 5,
                                       output_dir: Optional[str] = None, visualize: bool = True,
                                       save_frames: bool = False) -> Dict[str, Any]:
        """
        Run benchmark with ground truth evaluation.
        
        Args:
            video_path: Path to the video file
            detection_interval: Interval (in frames) to run detection
            output_dir: Optional directory to save output visualizations
            visualize: Whether to generate visualization frames
            save_frames: Whether to save individual frame images
            
        Returns:
            Dictionary containing both standard benchmark and ground truth results
            
        Raises:
            ValueError: If ground truth events are not set
        """
        if not self.ground_truth_events or not self.ground_truth_evaluator:
            raise ValueError("Ground truth events must be set before running benchmark with ground truth")
        
        # Run standard benchmark first
        standard_results = self.run_benchmark(
            video_path, detection_interval, output_dir, visualize, save_frames
        )
        
        # Extract video FPS for ground truth evaluator
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.ground_truth_evaluator.video_fps = fps
            self.ground_truth_evaluator.event_processor.video_fps = fps
            cap.release()
        
        # Run ground truth evaluation for each tracker
        self.ground_truth_results = {}
        
        for tracker_name, tracker_result in standard_results.items():
            print(f"\nRunning ground truth evaluation for {tracker_name}...")
            
            # Convert tracker results to ground truth evaluator format
            tracker_output = self._convert_tracker_results_to_gt_format(
                tracker_result, video_path
            )
            
            if tracker_output:
                try:
                    # Run ground truth evaluation
                    gt_summary = self.ground_truth_evaluator.evaluate_tracker_output(tracker_output)
                    self.ground_truth_results[tracker_name] = gt_summary
                    
                    print(f"Ground truth evaluation completed for {tracker_name}")
                    print(f"  Overall accuracy: {gt_summary.overall_accuracy:.1f}%")
                    print(f"  Moments evaluated: {gt_summary.moments_evaluated}/{gt_summary.total_moments}")
                    
                except Exception as e:
                    print(f"Error in ground truth evaluation for {tracker_name}: {e}")
                    self.ground_truth_results[tracker_name] = None
            else:
                print(f"No valid tracker output for ground truth evaluation: {tracker_name}")
                self.ground_truth_results[tracker_name] = None
        
        # Combine results
        combined_results = {
            'standard_benchmark': standard_results,
            'ground_truth_evaluation': self.ground_truth_results,
            'ground_truth_events': self.ground_truth_events,
            'moment_duration': self.moment_duration
        }
        
        return combined_results
    
    def _convert_tracker_results_to_gt_format(self, tracker_result: Dict[str, Any], 
                                            video_path: str) -> List[Dict[str, Any]]:
        """
        Convert tracker results to ground truth evaluator format.
        
        Args:
            tracker_result: Standard tracker benchmark result
            video_path: Path to video file for timing information
            
        Returns:
            List of tracker output dictionaries in ground truth format
        """
        # Get video FPS for timestamp conversion
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        gt_format_results = []
        
        # Process each frame's tracking results
        for frame_data in tracker_result.get('tracks', []):
            frame_idx = frame_data['frame_idx']
            tracks = frame_data['tracks']
            
            # Convert frame index to timestamp
            timestamp = frame_idx / fps
            
            # Count balls by type
            ball_counts = {}
            detections = []
            
            for track in tracks:
                # Extract ball type from class_name
                ball_class = track.get('class_name', 'unknown').lower()
                
                # Normalize ball class names for snooker balls
                if ball_class in ['cue', 'cue_ball', 'white', 'cue-ball']:
                    ball_type = 'white'
                elif ball_class in ['red', 'red_ball', 'reds']:
                    ball_type = 'red'
                elif ball_class in ['yellow', 'green', 'brown', 'blue', 'pink', 'black']:
                    ball_type = ball_class
                # Skip non-ball objects (pockets, table, etc.)
                elif ball_class in ['bottom-pocket', 'top-pocket', 'corner-pocket', 'side-pocket', 'table', 'rail']:
                    continue  # Skip non-ball objects
                else:
                    continue  # Skip unknown ball types
                
                # Count balls
                if ball_type not in ball_counts:
                    ball_counts[ball_type] = 0
                ball_counts[ball_type] += 1
                
                # Create detection entry
                detection = {
                    'ball_type': ball_type,
                    'x': track.get('x', 0),
                    'y': track.get('y', 0),
                    'confidence': track.get('confidence', 1.0),
                    'track_id': track.get('id', -1)
                }
                detections.append(detection)
            
            # Create ground truth format entry
            gt_entry = {
                'timestamp': timestamp,
                'counts': ball_counts,
                'detections': detections
            }
            
            gt_format_results.append(gt_entry)
        
        return gt_format_results
    

    

    
    def _visualize_ground_truth_results(self, output_path: Optional[str] = None) -> None:
        """
        Create visualizations for ground truth evaluation results.
        
        Args:
            output_path: Optional path to save visualization
        """
        # Filter out failed evaluations
        valid_results = {name: result for name, result in self.ground_truth_results.items() 
                        if result is not None}
        
        if not valid_results:
            print("No valid ground truth results to visualize")
            return
        
        # Create figure for ground truth metrics
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        tracker_names = list(valid_results.keys())
        overall_accuracy = [result.overall_accuracy for result in valid_results.values()]
        continuity = [result.continuity_stats.get('continuity_percentage', 0) 
                     for result in valid_results.values()]
        duplications = [result.duplication_summary.get('total_duplication_errors', 0) 
                       for result in valid_results.values()]
        moments_evaluated = [result.moments_evaluated for result in valid_results.values()]
        
        # Plot 1: Overall Accuracy
        axs[0, 0].bar(tracker_names, overall_accuracy, color='skyblue')
        axs[0, 0].set_title('Ground Truth Overall Accuracy')
        axs[0, 0].set_ylabel('Accuracy (%)')
        axs[0, 0].set_ylim(0, 100)
        
        # Plot 2: Tracking Continuity
        axs[0, 1].bar(tracker_names, continuity, color='lightgreen')
        axs[0, 1].set_title('Tracking Continuity')
        axs[0, 1].set_ylabel('Continuity (%)')
        axs[0, 1].set_ylim(0, 100)
        
        # Plot 3: Duplication Errors
        axs[1, 0].bar(tracker_names, duplications, color='lightcoral')
        axs[1, 0].set_title('Duplication Errors')
        axs[1, 0].set_ylabel('Number of Duplications')
        
        # Plot 4: Moments Evaluated
        axs[1, 1].bar(tracker_names, moments_evaluated, color='lightyellow')
        axs[1, 1].set_title('Moments Evaluated')
        axs[1, 1].set_ylabel('Number of Moments')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show figure
        if output_path:
            base_name, ext = os.path.splitext(output_path)
            gt_output = f"{base_name}_ground_truth{ext}"
            plt.savefig(gt_output)
            print(f"Ground truth metrics visualization saved to {gt_output}")
        else:
            plt.show()
    
    def export_ground_truth_results(self, output_dir: str) -> None:
        """
        Export ground truth evaluation results to files.
        
        Args:
            output_dir: Directory to save the exported results
            
        Raises:
            ValueError: If no ground truth results are available
        """
        if not self.ground_truth_results:
            raise ValueError("No ground truth results to export")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for tracker_name, gt_result in self.ground_truth_results.items():
            if gt_result is None:
                continue
            
            # Export detailed report as JSON
            if self.ground_truth_evaluator is not None:
                json_path = os.path.join(output_dir, f"{tracker_name}_ground_truth_report.json")
                self.ground_truth_evaluator.export_results_to_json(json_path, include_moment_details=True)
                
                # Export summary as CSV
                csv_path = os.path.join(output_dir, f"{tracker_name}_ground_truth_summary.csv")
                self.ground_truth_evaluator.export_results_to_csv(csv_path)
            
            print(f"Ground truth results exported for {tracker_name}:")
            print(f"  - Detailed report: {json_path}")
            print(f"  - Summary CSV: {csv_path}")
    
    def print_ground_truth_summary(self, tracker_name: Optional[str] = None) -> None:
        """
        Print detailed ground truth summary for a specific tracker or all trackers.
        
        Args:
            tracker_name: Optional specific tracker name. If None, prints all trackers.
        """
        if not self.ground_truth_results:
            print("No ground truth results available")
            return
        
        if tracker_name:
            if tracker_name not in self.ground_truth_results:
                print(f"No ground truth results for tracker: {tracker_name}")
                return
            
            gt_result = self.ground_truth_results[tracker_name]
            if gt_result is None:
                print(f"Ground truth evaluation failed for {tracker_name}")
                return
            
            print(f"\n=== Ground Truth Summary for {tracker_name} ===")
            if self.ground_truth_evaluator is not None:
                self.ground_truth_evaluator.print_summary_report()
        else:
            # Print summary for all trackers
            for name, gt_result in self.ground_truth_results.items():
                if gt_result is None:
                    print(f"\n=== Ground Truth Summary for {name} ===")
                    print("Ground truth evaluation failed")
                    continue
                
                print(f"\n=== Ground Truth Summary for {name} ===")
                if self.ground_truth_evaluator is not None:
                    # Temporarily set the evaluator's results to this tracker's results
                    original_results = self.ground_truth_evaluator.evaluation_results
                    self.ground_truth_evaluator.evaluation_results = []  # Reset for clean summary
                    
                    # Note: This is a simplified approach. In a full implementation,
                    # you might want to store separate evaluator instances for each tracker
                    print(f"Overall Accuracy: {gt_result.overall_accuracy:.1f}%")
                    print(f"Moments Evaluated: {gt_result.moments_evaluated}/{gt_result.total_moments}")
                    print(f"Tracking Continuity: {gt_result.continuity_stats.get('continuity_percentage', 0):.1f}%")
                    
                    # Restore original results
                    self.ground_truth_evaluator.evaluation_results = original_results
    
    def visualize_ground_truth_timeline(self, tracker_name: str, 
                                      save_path: Optional[str] = None) -> None:
        """
        Create timeline visualization for a specific tracker.
        
        Args:
            tracker_name: Name of the tracker to visualize
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If tracker has no ground truth results
        """
        if not self.ground_truth_results or tracker_name not in self.ground_truth_results:
            raise ValueError(f"No ground truth results available for tracker: {tracker_name}")
        
        if self.ground_truth_results[tracker_name] is None:
            raise ValueError(f"Ground truth evaluation failed for tracker: {tracker_name}")
        
        if self.ground_truth_evaluator is not None:
            self.visualizer.create_ground_truth_timeline(
                self.ground_truth_evaluator, tracker_name, save_path
            )
    
    def visualize_accuracy_analysis(self, tracker_name: str, 
                                  save_path: Optional[str] = None) -> None:
        """
        Create accuracy analysis visualization for a specific tracker.
        
        Args:
            tracker_name: Name of the tracker to visualize
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If tracker has no ground truth results
        """
        if not self.ground_truth_results or tracker_name not in self.ground_truth_results:
            raise ValueError(f"No ground truth results available for tracker: {tracker_name}")
        
        gt_result = self.ground_truth_results[tracker_name]
        if gt_result is None:
            raise ValueError(f"Ground truth evaluation failed for tracker: {tracker_name}")
        
        self.visualizer.create_accuracy_analysis(gt_result, tracker_name, save_path)
    
    def visualize_error_distribution(self, tracker_name: str, 
                                   save_path: Optional[str] = None) -> None:
        """
        Create error distribution visualization for a specific tracker.
        
        Args:
            tracker_name: Name of the tracker to visualize
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If tracker has no ground truth results
        """
        if not self.ground_truth_results or tracker_name not in self.ground_truth_results:
            raise ValueError(f"No ground truth results available for tracker: {tracker_name}")
        
        gt_result = self.ground_truth_results[tracker_name]
        if gt_result is None:
            raise ValueError(f"Ground truth evaluation failed for tracker: {tracker_name}")
        
        self.visualizer.create_error_distribution_charts(gt_result, tracker_name, save_path)
    
    def visualize_comparative_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create comparative analysis visualization for all trackers.
        
        Args:
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If insufficient ground truth results are available
        """
        if not self.ground_truth_results:
            raise ValueError("No ground truth results available for comparative analysis")
        
        # Filter out failed evaluations
        valid_results = {name: result for name, result in self.ground_truth_results.items() 
                        if result is not None}
        
        if len(valid_results) < 2:
            raise ValueError("Need at least 2 successful evaluations for comparative analysis")
        
        self.visualizer.create_comparative_analysis(valid_results, save_path)
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all key metrics.
        
        Args:
            save_path: Optional path to save the visualization
            
        Raises:
            ValueError: If no ground truth results are available
        """
        if not self.ground_truth_results:
            raise ValueError("No ground truth results available for dashboard")
        
        # Filter out failed evaluations
        valid_results = {name: result for name, result in self.ground_truth_results.items() 
                        if result is not None}
        
        if not valid_results:
            raise ValueError("No successful ground truth evaluations for dashboard")
        
        self.visualizer.create_summary_dashboard(valid_results, save_path)
    
    def generate_comprehensive_report(self, output_dir: str, 
                                    include_visualizations: bool = True,
                                    include_moment_details: bool = False) -> None:
        """
        Generate a comprehensive evaluation report with all metrics and visualizations.
        
        Args:
            output_dir: Directory to save all report files
            include_visualizations: Whether to generate visualization plots
            include_moment_details: Whether to include detailed moment-by-moment results
            
        Raises:
            ValueError: If no ground truth results are available
        """
        if not self.ground_truth_results:
            raise ValueError("No ground truth results available for report generation")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Generating comprehensive report in {output_dir}...")
        
        # Generate text report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            try:
                print("=" * 80)
                print("COMPREHENSIVE GROUND TRUTH EVALUATION REPORT")
                print("=" * 80)
                print()
                
                # Print standard benchmark results
                print("STANDARD BENCHMARK RESULTS")
                print("-" * 40)
                super().print_results()
                print()
                
                # Print ground truth results
                self.print_results()
                print()
                
                # Print detailed summaries for each tracker
                for tracker_name in self.ground_truth_results.keys():
                    if self.ground_truth_results[tracker_name] is not None:
                        print(f"DETAILED ANALYSIS FOR {tracker_name.upper()}")
                        print("-" * 50)
                        self.print_ground_truth_summary(tracker_name)
                        print()
                
                # Print recommendations
                print("RECOMMENDATIONS")
                print("-" * 20)
                for tracker_name, gt_result in self.ground_truth_results.items():
                    if gt_result is None:
                        continue
                    
                    print(f"\n{tracker_name}:")
                    if self.ground_truth_evaluator is not None:
                        report = self.ground_truth_evaluator.generate_detailed_report()
                    else:
                        report = {}
                    recommendations = report.get('recommendations', [])
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
                            print(f"  {i}. [{rec['priority'].upper()}] {rec['issue']}")
                            print(f"     → {rec['recommendation']}")
                    else:
                        print("  No specific recommendations - performance is satisfactory")
                
            finally:
                sys.stdout = original_stdout
        
        print(f"Text report saved to {report_path}")
        
        # Export detailed data
        self.export_ground_truth_results(output_dir)
        
        # Generate visualizations if requested
        if include_visualizations:
            print("Generating visualizations...")
            
            valid_results = {name: result for name, result in self.ground_truth_results.items() 
                           if result is not None}
            
            # Individual tracker visualizations
            for tracker_name in valid_results.keys():
                try:
                    # Timeline
                    timeline_path = os.path.join(output_dir, f"{tracker_name}_timeline.png")
                    self.visualize_ground_truth_timeline(tracker_name, timeline_path)
                    
                    # Accuracy analysis
                    accuracy_path = os.path.join(output_dir, f"{tracker_name}_accuracy.png")
                    self.visualize_accuracy_analysis(tracker_name, accuracy_path)
                    
                    # Error distribution
                    error_path = os.path.join(output_dir, f"{tracker_name}_errors.png")
                    self.visualize_error_distribution(tracker_name, error_path)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate visualizations for {tracker_name}: {e}")
            
            # Comparative analysis
            if len(valid_results) >= 2:
                try:
                    comparative_path = os.path.join(output_dir, "comparative_analysis.png")
                    self.visualize_comparative_analysis(comparative_path)
                    
                    dashboard_path = os.path.join(output_dir, "dashboard.png")
                    self.create_comprehensive_dashboard(dashboard_path)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate comparative visualizations: {e}")
        
        # Export visualization data for external analysis
        for tracker_name, gt_result in self.ground_truth_results.items():
            if gt_result is not None:
                viz_data_path = os.path.join(output_dir, f"{tracker_name}_visualization_data.csv")
                self.visualizer.export_visualization_data(gt_result, tracker_name, viz_data_path)
        
        print(f"Comprehensive report generation completed in {output_dir}")
    
    def set_visualization_config(self, config: VisualizationConfig) -> None:
        """
        Set custom visualization configuration.
        
        Args:
            config: Visualization configuration object
        """
        self.visualizer = GroundTruthVisualizer(config)
    
    def visualize_results(self, output_path: Optional[str] = None) -> None:
        """
        Enhanced visualization including ground truth metrics.
        
        Args:
            output_path: Optional path to save visualization
        """
        # Call parent method for standard and snooker metrics
        super().visualize_results(output_path)
        
        # Add ground truth visualizations if available
        if self.ground_truth_results:
            try:
                # Create comparative dashboard if multiple trackers
                valid_results = {name: result for name, result in self.ground_truth_results.items() 
                               if result is not None}
                
                if len(valid_results) >= 2:
                    if output_path:
                        base_name, ext = os.path.splitext(output_path)
                        dashboard_path = f"{base_name}_gt_dashboard{ext}"
                    else:
                        dashboard_path = None
                    
                    self.create_comprehensive_dashboard(dashboard_path)
                elif len(valid_results) == 1:
                    # Single tracker - show detailed analysis
                    tracker_name = list(valid_results.keys())[0]
                    
                    if output_path:
                        base_name, ext = os.path.splitext(output_path)
                        analysis_path = f"{base_name}_gt_analysis{ext}"
                    else:
                        analysis_path = None
                    
                    self.visualize_accuracy_analysis(tracker_name, analysis_path)
                    
            except Exception as e:
                print(f"Warning: Failed to generate ground truth visualizations: {e}")
    
    def print_results(self) -> None:
        """
        Enhanced results printing with comprehensive ground truth metrics.
        """
        # Print standard metrics first
        super().print_results()
        
        # Print enhanced ground truth results if available
        if self.ground_truth_results:
            print("\n" + "=" * 80)
            print("ENHANCED GROUND TRUTH EVALUATION RESULTS")
            print("=" * 80)
            
            # Summary table
            print(f"\n{'Tracker':<15} | {'GT Accuracy':<12} | {'Continuity':<12} | {'Moments':<12} | {'Errors':<8} | {'Status':<10}")
            print("-" * 85)
            
            for tracker_name, gt_result in self.ground_truth_results.items():
                if gt_result is None:
                    print(f"{tracker_name:<15} | {'FAILED':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<8} | {'ERROR':<10}")
                    continue
                
                continuity = gt_result.continuity_stats.get('continuity_percentage', 0)
                
                # Calculate total errors
                total_errors = (
                    sum(stats.get('over_detections', 0) + stats.get('under_detections', 0) 
                       for stats in gt_result.per_ball_accuracy.values()) +
                    gt_result.continuity_stats.get('total_illegal_disappearances', 0) +
                    gt_result.continuity_stats.get('total_illegal_reappearances', 0) +
                    gt_result.duplication_summary.get('total_duplication_errors', 0)
                )
                
                status = "GOOD" if gt_result.overall_accuracy >= 80 else "POOR" if gt_result.overall_accuracy < 60 else "FAIR"
                
                print(f"{tracker_name:<15} | {gt_result.overall_accuracy:<12.1f} | "
                     f"{continuity:<12.1f} | {gt_result.moments_evaluated}/{gt_result.total_moments:<8} | "
                     f"{total_errors:<8} | {status:<10}")
            
            # Detailed breakdown
            print(f"\n{'='*20} DETAILED BREAKDOWN {'='*20}")
            
            for tracker_name, gt_result in self.ground_truth_results.items():
                if gt_result is None:
                    print(f"\n{tracker_name}: Ground truth evaluation failed")
                    continue
                
                print(f"\n{tracker_name}:")
                print(f"  Overall Performance:")
                print(f"    • Accuracy: {gt_result.overall_accuracy:.1f}%")
                print(f"    • Continuity: {gt_result.continuity_stats.get('continuity_percentage', 0):.1f}%")
                print(f"    • Moments: {gt_result.moments_evaluated}/{gt_result.total_moments} "
                      f"({gt_result.moments_suppressed} suppressed)")
                
                print(f"  Per-Ball Performance:")
                for ball_type, stats in gt_result.per_ball_accuracy.items():
                    print(f"    • {ball_type.capitalize()}: {stats['accuracy']:.1f}% "
                          f"({stats['correct_moments']}/{stats['total_moments']} moments)")
                
                print(f"  Error Analysis:")
                total_over = sum(stats.get('over_detections', 0) for stats in gt_result.per_ball_accuracy.values())
                total_under = sum(stats.get('under_detections', 0) for stats in gt_result.per_ball_accuracy.values())
                
                print(f"    • Over-detections: {total_over}")
                print(f"    • Under-detections: {total_under}")
                print(f"    • Illegal disappearances: {gt_result.continuity_stats.get('total_illegal_disappearances', 0)}")
                print(f"    • Illegal reappearances: {gt_result.continuity_stats.get('total_illegal_reappearances', 0)}")
                print(f"    • Duplications: {gt_result.duplication_summary.get('total_duplication_errors', 0)}")
                
                # Context performance
                if gt_result.context_accuracy:
                    print(f"  Context Performance:")
                    for context, stats in gt_result.context_accuracy.items():
                        context_name = context.replace('_', ' ').title()
                        print(f"    • {context_name}: {stats['accuracy']:.1f}%")
            
            print("\n" + "=" * 80)
