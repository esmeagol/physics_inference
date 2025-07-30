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
from .inference_runner import InferenceRunner


class TrackerBenchmark:
    """
    Class for benchmarking and comparing different tracking algorithms.
    """
    
    def __init__(self, detection_model: InferenceRunner = None):
        """
        Initialize the tracker benchmark.
        
        Args:
            detection_model: Optional detection model for generating detections
        """
        self.detection_model = detection_model
        self.trackers = {}
        self.results = {}
        
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    
    def __init__(self, detection_model: InferenceRunner = None):
        """
        Initialize the snooker tracker benchmark.
        
        Args:
            detection_model: Optional detection model for generating detections
        """
        super().__init__(detection_model)
        
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
    
    def print_results(self) -> None:
        """
        Print benchmark results including snooker-specific metrics.
        """
        # Print standard metrics
        super().print_results()
        
        if not self.results:
            return
            
        print("\n=== Snooker-Specific Metrics ===\n")
        
        # Print header
        header = f"{'Tracker':<15} | {'Ball ID':<10} | {'Cue Ball':<10} | {'Color Balls':<12} | {'Red Balls':<10}"
        print(header)
        print("-" * len(header))
        
        # Print results for each tracker
        for name, result in self.results.items():
            metrics = result.get('snooker_metrics', {})
            print(f"{name:<15} | {metrics.get('ball_id_consistency', 0):<10.1f} | "
                 f"{metrics.get('cue_ball_tracking', 0):<10.1f} | "
                 f"{metrics.get('color_ball_tracking', 0):<12.1f} | "
                 f"{metrics.get('red_ball_tracking', 0):<10.1f}")
            
    def visualize_results(self, output_path: Optional[str] = None) -> None:
        """
        Visualize benchmark results including snooker-specific metrics.
        
        Args:
            output_path: Optional path to save visualization
        """
        # Call parent method for standard metrics
        super().visualize_results(output_path)
        
        if not self.results:
            return
            
        # Create figure for snooker-specific metrics
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        tracker_names = list(self.results.keys())
        ball_id_consistency = [result.get('snooker_metrics', {}).get('ball_id_consistency', 0) 
                             for result in self.results.values()]
        cue_ball_tracking = [result.get('snooker_metrics', {}).get('cue_ball_tracking', 0) 
                           for result in self.results.values()]
        color_ball_tracking = [result.get('snooker_metrics', {}).get('color_ball_tracking', 0) 
                             for result in self.results.values()]
        red_ball_tracking = [result.get('snooker_metrics', {}).get('red_ball_tracking', 0) 
                           for result in self.results.values()]
        
        # Plot 1: Ball ID Consistency
        axs[0, 0].bar(tracker_names, ball_id_consistency)
        axs[0, 0].set_title('Ball ID Consistency')
        axs[0, 0].set_ylabel('Score (higher is better)')
        axs[0, 0].set_ylim(0, 100)
        
        # Plot 2: Cue Ball Tracking
        axs[0, 1].bar(tracker_names, cue_ball_tracking)
        axs[0, 1].set_title('Cue Ball Tracking')
        axs[0, 1].set_ylabel('Score (higher is better)')
        axs[0, 1].set_ylim(0, 100)
        
        # Plot 3: Color Ball Tracking
        axs[1, 0].bar(tracker_names, color_ball_tracking)
        axs[1, 0].set_title('Color Ball Tracking')
        axs[1, 0].set_ylabel('Score (higher is better)')
        axs[1, 0].set_ylim(0, 100)
        
        # Plot 4: Red Ball Tracking
        axs[1, 1].bar(tracker_names, red_ball_tracking)
        axs[1, 1].set_title('Red Ball Tracking')
        axs[1, 1].set_ylabel('Score (higher is better)')
        axs[1, 1].set_ylim(0, 100)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show figure
        if output_path:
            base_name, ext = os.path.splitext(output_path)
            snooker_output = f"{base_name}_snooker{ext}"
            plt.savefig(snooker_output)
            print(f"Snooker metrics visualization saved to {snooker_output}")
        else:
            plt.show()
