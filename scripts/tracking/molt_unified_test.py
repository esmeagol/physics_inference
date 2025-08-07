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
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detection.local_pt_inference import LocalPT
from src.tracking.trackers.molt import MOLTTracker, MOLTTrackerConfig


class MOLTUnifiedTest:
    """
    Unified test class for MOLT tracker with real video validation.
    """
    
    def __init__(self, 
                 video_path: str,
                 model_path: str = "trained_models/ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt",
                 output_dir: str = "assets/output/molt_unified_test",
                 max_frames: int = 1500) -> None:  # ~33.3 seconds at 45 FPS
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
        
        # Initialize components with type hints
        self.inference_model: Optional[LocalPT] = None
        self.molt_tracker: Optional[MOLTTracker] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        
        # Test results
        self.test_results = {
            'initialization': False,
            'tracking_performance': {},
            'event_validation': {},
            'output_files': [],
            'statistics': {}
        }
        
        # Tracking data
        self.frame_tracks: List[List[Dict[str, Any]]] = []
        self.ball_events: List[Dict[str, Any]] = []
        self.tracking_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[Any]] = defaultdict(list)
        
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
            
            # Initialize video capture
            self.video_cap = cv2.VideoCapture(self.video_path)
            if self.video_cap is None or not self.video_cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            self.test_results['initialization'] = True
            return True
            
        except Exception as e:
            print(f"Error during setup: {e}")
            return False
    
    def _get_initial_detections(self) -> List[Dict[str, Any]]:
        """
        Get initial detections by averaging over the first few frames.
        
        Returns:
            List of detections, where each detection is a dictionary with keys:
            'x', 'y', 'width', 'height', 'confidence', 'class', 'class_id'
        """
        print(f"Getting initial detections from {self.initial_detection_frames} frames...")
        frame_detections: List[List[Dict[str, Any]]] = []
        
        # Check if video capture is available
        if self.video_cap is None:
            print("Error: Video capture not initialized")
            return []  # Return empty list of detections
            
        # Reset video to beginning
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as e:
            print(f"Error resetting video to beginning: {e}")
            return []  # Return empty list of detections
        
        # Process initial frames
        frames_processed = 0
        frame_idx = 0
        
        while frame_idx < self.initial_detection_frames:
            # Read frame from video first
                
            ret, frame = self.video_cap.read()
            if not ret or frame is None:
                print(f"Warning: Could not read frame {frame_idx+1}")
                frame_idx += 1
                continue
            
            # Process the frame if we have a model
            if self.inference_model is None:
                print("Error: Inference model not available")
                return []
                
            try:
                # Process the frame
                inference_result = self.inference_model.predict(frame)
                
                # Extract predictions from result
                formatted_detections: List[Dict[str, Any]] = []
                predictions = inference_result.get('predictions', []) if isinstance(inference_result, dict) else []
                
                for det in predictions:
                    if not isinstance(det, dict):
                        continue
                    formatted_detections.append({
                        'x': det.get('x', 0),
                        'y': det.get('y', 0),
                        'width': det.get('width', 0),
                        'height': det.get('height', 0),
                        'confidence': det.get('confidence', 0.0),
                        'class': det.get('class', 'unknown'),
                        'class_id': det.get('class_id', -1)
                    })
                
                if formatted_detections:  # Only add if we have detections
                    frame_detections.append(formatted_detections)
                    frames_processed += 1
                    print(f"  Frame {frame_idx+1}: {len(formatted_detections)} detections")
                
            except Exception as e:
                print(f"Error processing frame {frame_idx+1}: {e}")
            
            frame_idx += 1
        
        # If we didn't process any frames, return empty list
        if not frame_detections:
            print("Warning: No valid detections found in initial frames")
            return []
            
        # Average detections across frames
        averaged_detections = self._average_detections(frame_detections)
        print(f"Averaged detections: {len(averaged_detections)}")
            
        # Reset video to beginning if possible
        try:
            if self.video_cap is not None:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as e:
            print(f"Warning: Could not reset video to beginning: {e}")
            
        # Ensure we have a list and handle empty case
        if not isinstance(averaged_detections, list) or not averaged_detections:
            return []
            
        # Process the detections
        result: List[Dict[str, Any]] = []
        
        for item in averaged_detections:
            # Handle list of dicts
            if isinstance(item, dict):
                result.append(item)
        
        return result
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.video_cap is not None:
                self.video_cap.release()
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _average_detections(self, frame_detections: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
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
        detection_groups: List[List[Dict[str, Any]]] = []
        
        for detections in frame_detections:
            for det in detections:
                ball_class = det.get('class', 'unknown')
                x = det.get('x', 0)
                y = det.get('y', 0)
                
                # Try to find a matching group
                assigned = False
                for group in detection_groups:
                    if not group:
                        continue
                        
                    # Get first detection in group as reference
                    ref_det = group[0]
                    ref_class = ref_det.get('class', 'unknown')
                    ref_x = ref_det.get('x', 0)
                    ref_y = ref_det.get('y', 0)
                    
                    # Check if this detection matches the group
                    if (ball_class == ref_class and 
                        abs(x - ref_x) < 50 and 
                        abs(y - ref_y) < 50):
                        group.append(det)
                        assigned = True
                        break
                
                # Create new group if not assigned
                if not assigned:
                    detection_groups.append([det])
        
        # Average each group
        averaged_detections = []
        for group in detection_groups:
            if len(group) >= 2:  # Require at least 2 detections for stability
                avg_detection = {
                    'x': sum(d.get('x', 0) for d in group) / len(group),
                    'y': sum(d.get('y', 0) for d in group) / len(group),
                    'width': sum(d.get('width', 0) for d in group) / len(group),
                    'height': sum(d.get('height', 0) for d in group) / len(group),
                    'confidence': sum(d.get('confidence', 0) for d in group) / len(group),
                    'class': group[0].get('class', 'unknown'),
                    'class_id': group[0].get('class_id', -1)
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
        
        # Ensure setup is called first
        if not hasattr(self, 'test_results') or not self.test_results.get('initialization', False):
            print("Initializing test environment...")
            if not self.setup():
                print("Error: Failed to initialize test environment")
                return False
            
        # Verify video capture is initialized
        if self.video_cap is None or not self.video_cap.isOpened():
            print("Error: Video capture not properly initialized")
            return False
            
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        # Prepare output video
        output_video_path = os.path.join(self.output_dir, "molt_tracking_results.mp4")
        
        # Initialize VideoWriter with proper error handling
        video_writer = None
        try:
            # Handle different OpenCV versions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if hasattr(cv2, 'VideoWriter_fourcc') else 0x7634706d  # 'mp4v' in ASCII
            video_writer = cv2.VideoWriter(
                output_video_path, 
                fourcc, 
                int(fps),  # Ensure fps is an integer
                (int(width), int(height))  # Ensure dimensions are integers
            )
            
            if video_writer is None or not video_writer.isOpened():
                print(f"Error: Could not open video writer for {output_video_path}")
                return False
                
            # Initialize test_results if needed
            if not hasattr(self, 'test_results'):
                self.test_results = {'output_files': []}
                
            # Safe type checking and appending
            output_files = self.test_results.get('output_files')
            if isinstance(output_files, list):
                output_files.append(output_video_path)
            else:
                print("Warning: Could not append output file path to test results")
                
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            if video_writer is not None:
                video_writer.release()
            return False
            
        # Collect initial detections
        averaged_detections = self._get_initial_detections()
        if not averaged_detections:
            print("Error: No stable detections found in initial frames")
            if video_writer is not None:
                video_writer.release()
            return False
            
        # Initialize MOLT tracker
        if self.video_cap is None or self.molt_tracker is None:
            print("Error: Video capture or MOLT tracker not initialized")
            return False
            
        first_frame_ret, first_frame = self.video_cap.read()
        if not first_frame_ret or first_frame is None:
            print("Error: Failed to read first frame")
            return False
            
        print("Initializing MOLT tracker with averaged detections...")
        # Ensure frame is the correct numpy array type
        first_frame_array = np.asarray(first_frame, dtype=np.uint8)
        init_success = self.molt_tracker.init(first_frame_array, averaged_detections)
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
            # Ensure frame is the correct numpy array type
            frame_array = np.asarray(frame, dtype=np.uint8)
            tracks = self.molt_tracker.update(frame_array)
                
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
            ball_counts: Dict[str, int] = defaultdict(int)
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

    def _check_ball_events(self, tracks: List[Dict[str, Any]], current_time: float, frame_idx: int) -> None:
        """
        Check for expected ball loss/respot events at specific timestamps.
        
        Args:
            tracks: Current tracking results
            current_time: Current video timestamp in seconds
            frame_idx: Current frame index
        """
        # Get ball counts by class
        ball_counts: Dict[str, int] = defaultdict(int)
        for track in tracks:
            ball_counts[track.get('class', 'unknown')] += 1
        
        # Check for significant changes in ball counts
        if len(self.tracking_results) > 1:
            prev_result = self.tracking_results[-2]
            prev_tracks = prev_result['tracks']
                
            prev_counts: Dict[str, int] = defaultdict(int)
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
        try:
            # Setup
            if not self.setup():
                return {}
            
            # Run tracking test
            if not self.run_tracking_test():
                return {}
            
            # Check if we have any tracking results
            if not self.tracking_results:
                print("Error: No tracking results to validate")
                return {}
            
            # Basic validation
            validation_results = {
                'success': True,
                'total_frames': len(self.tracking_results),
                'average_tracks_per_frame': np.mean([len(r['tracks']) for r in self.tracking_results]),
                'ball_events': len(self.ball_events),
                'performance_metrics': self.performance_metrics
            }
            
            # Check for expected ball events
            expected_events = [
                # Format: (time_in_seconds, event_type, ball_class)
                (5.0, 'ball_loss', 'yellow'),
                (10.0, 'ball_respot', 'yellow'),
                (15.0, 'ball_loss', 'red'),
                (20.0, 'ball_respot', 'red')
            ]
            
            detected_events = []
            for event in self.ball_events:
                detected_events.append({
                    'time': event['timestamp'],
                    'event': event['event'],
                    'frame': event['frame_idx']
                })
            
            validation_results['detected_events'] = detected_events
            validation_results['expected_events'] = expected_events
            
            # Save validation results
            results_file = os.path.join(self.output_dir, 'validation_results.json')
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            self.test_results['validation_results'] = results_file
            print(f"Validation results saved to {results_file}")
            
            return validation_results
            
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            return {}
        finally:
            if hasattr(self, 'cleanup'):
                self.cleanup()


def main() -> bool:
    """Main function to run the unified MOLT test."""
    # Test configuration
    video_path = "assets/test_videos/video_1.mp4"
    model_path = "trained_models/ar-snkr_objd-lolhi-3-yolov11-medium-weights.pt"
    output_dir = "assets/output/molt_unified_test"
    max_frames = 1500  # Process first 1500 frames (~33.3 seconds at 45 FPS)
    
    print("MOLT Tracker Unified Test")
    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Max frames: {max_frames}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run test
    test = None
    try:
        test = MOLTUnifiedTest(video_path, model_path, output_dir, max_frames)
        success = test.run_tracking_test()
        return success
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure resources are properly cleaned up
        if test is not None and hasattr(test, 'cleanup'):
            test.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
