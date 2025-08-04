#!/usr/bin/env python3
"""
Unified MOLT Tracker Test

This test combines the best features of both comprehensive and final tests
to provide a single, complete validation suite for the MOLT tracker with
real video input and detailed reporting.

Features:
- Uses YOLOv11 model for initial detections (first N frames)
- Tracks objects through MOLT after initial detection phase
- Validates specific ball loss events at expected timestamps
- Generates annotated output video with tracking information
- Provides detailed performance metrics and validation reports
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from CVModelInference.local_pt_inference import LocalPT
from PureCV.molt import MOLTTracker, MOLTTrackerConfig


class MOLTUnifiedTest:
    """
    Unified test class for MOLT tracker with real video validation.
    """
    
    def __init__(self, 
                 video_path: str,
                 model_path: str = "CVModelInference/trained_models/ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt",
                 output_dir: str = "assets/output/molt_unified_test",
                 max_frames: int = 1500):  # ~33.3 seconds at 45 FPS
        """
        Initialize the unified MOLT test.
        
        Args:
            video_path: Path to the test video file
            model_path: Path to YOLOv11 model for initial detections
            output_dir: Directory to save test outputs
            max_frames: Maximum number of frames to process
        """
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.max_frames = max_frames
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test configuration
        self.initial_detection_frames = 10  # Use first 10 frames for detection
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        
        # Expected ball loss events (in seconds)
        self.expected_events = {
            'red_ball_loss_1': 11.0,      # Red ball potted at ~11s
            'black_ball_loss': 22.0,      # Black ball lost at ~22s
            'black_ball_respot': 24.0,    # Black ball respotted at ~24s
            'red_ball_loss_2': 28.0       # Another red ball potted at ~28s
        }
        
        # Initialize components
        self.inference_model = None
        self.molt_tracker = None
        self.video_cap = None
        
        # Test results
        self.test_results = {
            'initialization': False,
            'tracking_performance': {},
            'event_validation': {},
            'output_files': [],
            'statistics': {}
        }
        
        # Tracking data
        self.frame_tracks = []
        self.ball_events = []
        self.tracking_results = []
        self.performance_metrics = defaultdict(list)
        
    def setup(self) -> bool:
        """
        Set up the test environment and initialize components.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            print("Setting up unified MOLT test...")
            
            # Verify video file exists
            if not os.path.exists(self.video_path):
                print(f"Error: Video file not found: {self.video_path}")
                return False
            
            # Verify model file exists
            if not os.path.exists(self.model_path):
                print(f"Error: Model file not found: {self.model_path}")
                return False
            
            # Initialize inference model
            print(f"Loading inference model: {self.model_path}")
            self.inference_model = LocalPT(
                model_path=self.model_path,
                confidence=self.confidence_threshold,
                iou=self.iou_threshold
            )
            
            # Initialize MOLT tracker with snooker configuration
            print("Initializing MOLT tracker...")
            self.molt_tracker = MOLTTracker(
                config=MOLTTrackerConfig(
                    # Population sizes for different ball types
                    population_sizes={
                        'red': 300,
                        'yellow': 250,
                        'green': 250,
                        'brown': 250,
                        'blue': 250,
                        'pink': 250,
                        'black': 250,
                        'white': 400,  # Larger population for cue ball
                        'default': 300
                    },
                    # Exploration radii for different ball types
                    exploration_radii={
                        'red': 15,
                        'yellow': 15,
                        'green': 15,
                        'brown': 15,
                        'blue': 15,
                        'pink': 15,
                        'black': 15,
                        'white': 30,  # Larger radius for cue ball
                        'default': 15
                    },
                    # Expected ball counts
                    expected_ball_counts={
                        'red': 15,
                        'yellow': 1,
                        'green': 1,
                        'brown': 1,
                        'blue': 1,
                        'pink': 1,
                        'black': 1,
                        'white': 1
                    },
                    # Histogram configuration
                    histogram_bins=16,
                    color_space='hsv',
                    # Similarity weights
                    similarity_weights={
                        'histogram': 0.7,
                        'spatial': 0.3
                    },
                    # Minimum confidence for valid tracks
                    min_confidence=0.3
                )
            )
            
            # Open video capture
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                print(f"Error: Failed to open video: {self.video_path}")
                return False
            
            self.test_results['initialization'] = True
            return True
            
        except Exception as e:
            print(f"Error during setup: {e}")
            return False
    
    def collect_initial_detections(self) -> List[Dict[str, Any]]:
        """
        Collect detections from the first N frames and average them for stable initialization.
        
        Returns:
            List of averaged detection dictionaries
        """
        print(f"Collecting initial detections from first {self.initial_detection_frames} frames...")
        
        frame_detections = []
        
        # Reset video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process initial frames
        for i in range(self.initial_detection_frames):
            ret, frame = self.video_cap.read()
            if not ret:
                break
                
            # Get detections for this frame
            result = self.inference_model.predict(frame)
            
            # Extract predictions from result
            formatted_detections = []
            for det in result.get('predictions', []):
                formatted_detections.append({
                    'x': det['x'],
                    'y': det['y'],
                    'width': det['width'],
                    'height': det['height'],
                    'confidence': det['confidence'],
                    'class': det['class'],
                    'class_id': det['class_id']
                })
            
            frame_detections.append(formatted_detections)
            print(f"  Frame {i+1}: {len(formatted_detections)} detections")
        
        # Average detections across frames
        averaged_detections = self._average_detections(frame_detections)
        print(f"Averaged detections: {len(averaged_detections)}")
        
        # Reset video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return averaged_detections
    
    def _average_detections(self, frame_detections: List[List[Dict]]) -> List[Dict[str, Any]]:
        """
        Average detections across multiple frames using spatial clustering.
        
        Args:
            frame_detections: List of detection lists from each frame
            
        Returns:
            List of averaged detections
        """
        if not frame_detections:
            return []
        
        # Group detections by class and spatial proximity
        detection_groups = defaultdict(list)
        
        for detections in frame_detections:
            for det in detections:
                ball_class = det['class']
                
                # Find existing group for this detection
                assigned = False
                for group_key in detection_groups:
                    if group_key.startswith(ball_class):
                        # Check if this detection is close to existing group
                        group_detections = detection_groups[group_key]
                        if group_detections:
                            avg_x = sum(d['x'] for d in group_detections) / len(group_detections)
                            avg_y = sum(d['y'] for d in group_detections) / len(group_detections)
                            
                            # If within 50 pixels, add to existing group
                            distance = np.sqrt((det['x'] - avg_x)**2 + (det['y'] - avg_y)**2)
                            if distance < 50:
                                detection_groups[group_key].append(det)
                                assigned = True
                                break
                
                # Create new group if not assigned
                if not assigned:
                    group_key = f"{ball_class}_{len([k for k in detection_groups if k.startswith(ball_class)])}"
                    detection_groups[group_key] = [det]
        
        # Average each group
        averaged_detections = []
        for group_key, group_detections in detection_groups.items():
            if len(group_detections) >= 2:  # Require at least 2 detections for stability
                avg_detection = {
                    'x': sum(d['x'] for d in group_detections) / len(group_detections),
                    'y': sum(d['y'] for d in group_detections) / len(group_detections),
                    'width': sum(d['width'] for d in group_detections) / len(group_detections),
                    'height': sum(d['height'] for d in group_detections) / len(group_detections),
                    'confidence': sum(d['confidence'] for d in group_detections) / len(group_detections),
                    'class': group_detections[0]['class'],
                    'class_id': group_detections[0]['class_id']
                }
                averaged_detections.append(avg_detection)
        
        return averaged_detections
    
    def run_tracking_test(self) -> bool:
        """
        Run the main tracking test with MOLT tracker.
        
        Returns:
            bool: True if test completed successfully
        """
        print("Running MOLT tracking test...")
        
        # Get video properties
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output video
        output_video_path = os.path.join(self.output_dir, "molt_tracking_results.mp4")
        video_writer = cv2.VideoWriter(
            output_video_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )
        
        self.test_results['output_files'].append(output_video_path)
        
        # Collect initial detections
        averaged_detections = self.collect_initial_detections()
        if not averaged_detections:
            print("Error: No stable detections found in initial frames")
            return False
        
        # Initialize MOLT tracker
        first_frame_ret, first_frame = self.video_cap.read()
        if not first_frame_ret:
            print("Error: Failed to read first frame")
            return False
        
        print("Initializing MOLT tracker with averaged detections...")
        init_success = self.molt_tracker.init(first_frame, averaged_detections)
        if not init_success:
            print("Error: Failed to initialize MOLT tracker")
            return False
        
        # Reset video to beginning for tracking
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Run tracking loop
        print("Starting tracking loop...")
        start_time = time.time()
        frame_idx = 0
        
        while True:
            ret, frame = self.video_cap.read()
            if not ret or frame_idx >= self.max_frames:
                break
            
            # Track timing
            track_start = time.time()
            
            # Update tracker
            tracks = self.molt_tracker.update(frame)
            
            # Track timing
            track_time = time.time() - track_start
            self.performance_metrics['processing_time'].append(track_time)
            
            # Store tracking results
            current_time = frame_idx / fps
            result = {
                'frame_idx': frame_idx,
                'timestamp': current_time,
                'tracks': tracks,
                'processing_time': track_time
            }
            self.tracking_results.append(result)
            
            # Check for ball events
            self._check_ball_events(tracks, current_time, frame_idx)
            
            # Create visualization
            vis_frame = frame.copy()
            
            # Draw tracks
            for track in tracks:
                x, y = int(track['x']), int(track['y'])
                ball_class = track.get('class', 'unknown')
                track_id = track.get('id', -1)
                confidence = track.get('confidence', 0)
                
                # Draw circle for ball
                color = (42, 42, 165)  # Brown
                
                cv2.circle(vis_frame, (x, y), 15, color, 2)
                
                # Draw ID and confidence
                cv2.putText(vis_frame, f"{ball_class}:{track_id}", 
                           (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis_frame, f"{confidence:.2f}", 
                           (x - 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add frame info
            cv2.putText(vis_frame, f"Frame: {frame_idx} | Time: {current_time:.2f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Tracks: {len(tracks)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add ball counts
            ball_counts = defaultdict(int)
            for track in tracks:
                ball_counts[track.get('class', 'unknown')] += 1
            
            y_offset = 90
            for ball_class, count in ball_counts.items():
                cv2.putText(vis_frame, f"{ball_class}: {count}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Write frame
            video_writer.write(vis_frame)
            
            # Progress
            if frame_idx % 30 == 0:  # Every second
                avg_fps = 1.0 / track_time if track_time > 0 else 0
                print(f"  Frame {frame_idx}/{self.max_frames}: {len(tracks)} tracks, {avg_fps:.1f} FPS")
            
            frame_idx += 1
        
        # Clean up
        video_writer.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        
        # Store performance metrics
        self.test_results['tracking_performance'] = {
            'total_frames_processed': frame_idx,
            'total_time': total_time,
            'average_fps': avg_fps,
            'average_processing_time': np.mean(self.performance_metrics['processing_time']),
            'average_track_count': np.mean([len(r['tracks']) for r in self.tracking_results])
        }
        
        print(f"\nTracking completed!")
        print(f"  Frames processed: {frame_idx}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Output video: {output_video_path}")
        
        return len(self.tracking_results) > 0
    
    def _check_ball_events(self, tracks: List[Dict], current_time: float, frame_idx: int) -> None:
        """
        Check for expected ball loss/respot events at specific timestamps.
        
        Args:
            tracks: Current tracking results
            current_time: Current video timestamp in seconds
            frame_idx: Current frame index
        """
        # Get ball counts by class
        ball_counts = defaultdict(int)
        for track in tracks:
            ball_counts[track.get('class', 'unknown')] += 1
        
        # Check for significant changes in ball counts
        if len(self.tracking_results) > 1:
            prev_result = self.tracking_results[-2]
            prev_tracks = prev_result['tracks']
            
            prev_counts = defaultdict(int)
            for track in prev_tracks:
                prev_counts[track.get('class', 'unknown')] += 1
            
            # Check for ball losses
            for ball_class, count in prev_counts.items():
                if ball_counts[ball_class] < count:
                    # Ball loss detected
                    event = {
                        'event': f"{ball_class}_ball_loss",
                        'timestamp': current_time,
                        'frame_idx': frame_idx,
                        'previous_count': count,
                        'current_count': ball_counts[ball_class],
                        'total_tracks': len(tracks)
                    }
                    self.ball_events.append(event)
                    print(f"Ball event detected: {ball_class} ball loss at {current_time:.2f}s")
            
            # Check for ball respots
            for ball_class, count in ball_counts.items():
                if count > prev_counts[ball_class]:
                    # Ball respot detected
                    event = {
                        'event': f"{ball_class}_ball_respot",
                        'timestamp': current_time,
                        'frame_idx': frame_idx,
                        'previous_count': prev_counts[ball_class],
                        'current_count': count,
                        'total_tracks': len(tracks)
                    }
                    self.ball_events.append(event)
                    print(f"Ball event detected: {ball_class} ball respot at {current_time:.2f}s")
    
    def validate_results(self) -> Dict[str, Any]:
        """
        Validate test results against expected outcomes.
        
        Returns:
            Dictionary containing validation results
        """
        print("Validating test results...")
        
        validation_results = {
            'event_validation': {},
            'tracking_performance': self.test_results['tracking_performance'],
            'overall_success': False,
            'issues_found': []
        }
        
        # Validate event detection
        for event_name, expected_time in self.expected_events.items():
            event_type = event_name.split('_')[0]  # red, black, etc.
            event_action = '_'.join(event_name.split('_')[1:])  # ball_loss, ball_respot
            
            # Look for matching events
            matching_events = []
            for event in self.ball_events:
                if event['event'] == f"{event_type}_ball_{event_action}":
                    matching_events.append(event)
            
            # Find closest event by time
            closest_event = None
            min_time_diff = float('inf')
            
            for event in matching_events:
                time_diff = abs(event['timestamp'] - expected_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_event = event
            
            # Validate event
            if closest_event:
                time_diff = abs(closest_event['timestamp'] - expected_time)
                validation_results['event_validation'][event_name] = {
                    'detected': True,
                    'expected_time': expected_time,
                    'actual_time': closest_event['timestamp'],
                    'time_difference': time_diff,
                    'validated': time_diff <= 2.0  # Within 2 seconds is valid
                }
                
                if time_diff > 2.0:
                    validation_results['issues_found'].append(
                        f"Event {event_name} detected but with large time difference: {time_diff:.1f}s"
                    )
            else:
                validation_results['event_validation'][event_name] = {
                    'detected': False,
                    'expected_time': expected_time
                }
                validation_results['issues_found'].append(f"Event {event_name} not detected")
        
        # Validate tracking performance
        avg_track_count = validation_results['tracking_performance']['average_track_count']
        if avg_track_count < 5:
            validation_results['issues_found'].append(
                f"Low average track count: {avg_track_count:.1f}"
            )
        
        # Overall success
        validation_results['overall_success'] = (
            len(validation_results['issues_found']) == 0 and
            len(self.tracking_results) > 0
        )
        
        return validation_results
    
    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            validation_results: Results from validate_results()
            
        Returns:
            Path to generated report file
        """
        print("Generating test report...")
        
        # Generate JSON report
        report_path = os.path.join(self.output_dir, "molt_unified_report.json")
        report = {
            'test_info': {
                'video_path': self.video_path,
                'model_path': self.model_path,
                'frames_processed': len(self.tracking_results),
                'duration_seconds': len(self.tracking_results) / self.video_cap.get(cv2.CAP_PROP_FPS),
                'max_frames': self.max_frames
            },
            'tracking_performance': validation_results['tracking_performance'],
            'event_validation': validation_results['event_validation'],
            'ball_events': self.ball_events,
            'overall_success': validation_results['overall_success'],
            'issues_found': validation_results['issues_found'],
            'output_files': self.test_results['output_files']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_path = os.path.join(self.output_dir, "molt_unified_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MOLT Unified Test Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Frames processed: {len(self.tracking_results)}\n")
            f.write(f"Duration: {len(self.tracking_results) / self.video_cap.get(cv2.CAP_PROP_FPS):.2f} seconds\n")
            f.write(f"Average FPS: {validation_results['tracking_performance']['average_fps']:.2f}\n\n")
            
            f.write("Tracking Performance:\n")
            f.write("-" * 20 + "\n")
            perf = validation_results['tracking_performance']
            f.write(f"Average FPS: {perf['average_fps']:.2f}\n")
            f.write(f"Average Processing Time: {perf['average_processing_time']:.4f}s\n")
            f.write(f"Average Track Count: {perf['average_track_count']:.1f}\n")
            f.write(f"Total Frames: {perf['total_frames_processed']}\n\n")
            
            f.write("Event Validation Results:\n")
            f.write("-" * 30 + "\n")
            for event_name, result in validation_results['event_validation'].items():
                if result['detected']:
                    status = "✓ PASS" if result.get('validated', False) else "⚠ DETECTED"
                    f.write(f"{event_name}: {status}\n")
                    f.write(f"  Expected: {result['expected_time']:.1f}s\n")
                    f.write(f"  Actual: {result['actual_time']:.1f}s\n")
                    f.write(f"  Difference: {result['time_difference']:.1f}s\n")
                else:
                    f.write(f"{event_name}: ✗ NOT DETECTED\n")
                    f.write(f"  Expected: {result['expected_time']:.1f}s\n")
                f.write("\n")
            
            # Track continuity
            track_ids = set()
            for result in self.tracking_results:
                for track in result['tracks']:
                    track_ids.add(track.get('id', -1))
            
            f.write(f"Track Statistics:\n")
            f.write(f"  Unique track IDs: {len(track_ids)}\n\n")
            
            f.write("Overall Result: ")
            if validation_results['overall_success']:
                f.write("✓ PASS\n")
            else:
                f.write("✗ FAIL\n")
                f.write("Issues Found:\n")
                for issue in validation_results['issues_found']:
                    f.write(f"  - {issue}\n")
        
        print(f"  Report saved: {report_path}")
        print(f"  Summary saved: {summary_path}")
        
        self.test_results['output_files'].extend([report_path, summary_path])
        return report_path
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.video_cap:
            self.video_cap.release()
        cv2.destroyAllWindows()
    
    def run_full_test(self) -> bool:
        """
        Run the complete unified test.
        
        Returns:
            bool: True if test completed successfully
        """
        try:
            # Setup
            if not self.setup():
                return False
            
            # Run tracking test
            if not self.run_tracking_test():
                return False
            
            # Validate results
            validation_results = self.validate_results()
            
            # Generate report
            self.generate_report(validation_results)
            
            # Print summary
            print("\n" + "=" * 60)
            print("MOLT UNIFIED TEST COMPLETED")
            print("=" * 60)
            print(f"Overall Result: {'PASS' if validation_results['overall_success'] else 'FAIL'}")
            print(f"Output Directory: {self.output_dir}")
            print("Generated Files:")
            for file_path in self.test_results['output_files']:
                print(f"  - {file_path}")
            
            return validation_results['overall_success']
            
        except Exception as e:
            print(f"Error during unified test: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main function to run the unified MOLT test."""
    # Test configuration
    video_path = "assets/test_videos/video_1.mp4"
    model_path = "CVModelInference/trained_models/ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt"
    output_dir = "assets/output/molt_unified_test"
    max_frames = 1500  # Process first 1500 frames (~33.3 seconds at 45 FPS)
    
    print("MOLT Tracker Unified Test")
    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Max frames: {max_frames}")
    print(f"Output: {output_dir}")
    print()
    
    # Run test
    test = MOLTUnifiedTest(video_path, model_path, output_dir, max_frames)
    success = test.run_full_test()
    
    # Exit with appropriate code
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
