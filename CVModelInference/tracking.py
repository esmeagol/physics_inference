"""
Tracking Module for CVModelInference

This module provides functionality to track objects across video frames
using detection models and the supervision library.
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

from .inference_runner import InferenceRunner
from .local_pt_inference import LocalPT


class Tracker:
    """
    Class for tracking objects across video frames.
    
    This class uses a detection model and supervision's ByteTrack implementation
    to track objects across video frames.
    """
    
    def __init__(self, 
                model_path: str,
                confidence: float = 0.5,
                iou: float = 0.5,
                tracker_type: str = "bytetrack",
                **kwargs):
        """
        Initialize the tracker.
        
        Args:
            model_path: Path to the local PyTorch model weights (.pt file)
            confidence: Minimum confidence threshold for predictions (0-1)
            iou: IoU threshold for NMS (0-1)
            tracker_type: Type of tracker to use (default: "bytetrack")
            **kwargs: Additional parameters for model initialization
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        
        # Initialize the detection model
        self.detector = LocalPT(model_path=model_path, confidence=confidence, iou=iou)
        
        # Initialize the tracker
        if tracker_type.lower() == "bytetrack":
            self.tracker = sv.ByteTrack()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2
        )
        
        # Store model info
        self.model_name = os.path.basename(model_path)
        
    def process_frame(self, 
                     frame: np.ndarray, 
                     **kwargs) -> Dict:
        """
        Process a single frame for detection and tracking.
        
        Args:
            frame: Input frame as numpy array
            **kwargs: Additional parameters to override the defaults
                    (confidence, iou, etc.)
                    
        Returns:
            Dictionary containing the detection and tracking results
        """
        conf = kwargs.get('confidence', self.confidence)
        
        try:
            # Run detection
            detection_results = self.detector.predict(frame, confidence=conf)
            
            # Convert to supervision Detections format
            detections = self._convert_to_sv_detections(detection_results, frame)
            
            # Update tracker
            detections = self.tracker.update_with_detections(detections=detections)
            
            # Return results
            return {
                'detections': detections,
                'frame_shape': frame.shape,
                'model_name': self.model_name
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during tracking: {str(e)}")
    
    def annotate_frame(self, 
                      frame: np.ndarray,
                      tracking_results: Dict,
                      draw_traces: bool = True,
                      draw_boxes: bool = True,
                      **kwargs) -> np.ndarray:
        """
        Annotate a frame with tracking results.
        
        Args:
            frame: Input frame as numpy array
            tracking_results: Results from process_frame()
            draw_traces: Whether to draw motion traces
            draw_boxes: Whether to draw bounding boxes
            **kwargs: Additional parameters for annotation
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        detections = tracking_results.get('detections')
        
        if detections is None or len(detections) == 0:
            return annotated_frame
        
        # Draw traces if requested
        if draw_traces:
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
        
        # Draw boxes if requested
        if draw_boxes:

            # Create labels for each detection
            labels = []
            for i in range(len(detections)):
                # Get class ID if available
                class_id = detections.class_id[i] if detections.class_id is not None else 0
                if detections.tracker_id is None:
                    print("Tracker ID for object with class ID {class_id} is None")
                    track_id = i
                else:
                    track_id = detections.tracker_id[i]

                # Create label
                labels.append(f"{class_id}_{track_id}")
            
            # Draw boxes
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            
            # Add labels manually
            for i, label in enumerate(labels):
                if i < len(detections):
                    # Get box coordinates
                    x1, y1, x2, y2 = detections.xyxy[i]
                    # Add text above the box
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.2,
                        (0, 255, 0),
                        1
                    )
        
        return annotated_frame
    
    def process_video(self,
                     input_video_path: str,
                     output_video_path: str,
                     confidence: float = None,
                     fps_limit: Optional[float] = None,
                     start_time: float = 0,
                     duration: Optional[float] = None,
                     show_preview: bool = False,
                     draw_traces: bool = True,
                     draw_boxes: bool = True) -> None:
        """
        Process a video with tracking and generate an annotated output video.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video
            confidence: Confidence threshold for detections
            fps_limit: Limit processing FPS
            start_time: Start time in seconds
            duration: Duration to process in seconds
            show_preview: Show preview window during processing
            draw_traces: Whether to draw motion traces
            draw_boxes: Whether to draw bounding boxes
        """
        import time
        from tqdm import tqdm
        
        # Use provided confidence or default
        if confidence is None:
            confidence = self.confidence
        
        # Open the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame ranges
        start_frame = int(start_time * fps)
        if duration:
            end_frame = start_frame + int(duration * fps)
        else:
            end_frame = total_frames
        
        # Set up FPS limiting
        if fps_limit:
            fps_delay = 1.0 / fps_limit
        else:
            fps_delay = 0
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_count = start_frame
        processed_count = 0
        
        # Set up progress bar
        pbar = tqdm(total=end_frame - start_frame, desc="Processing video")
        
        try:
            while frame_count < end_frame:
                start_process_time = time.time()
                
                # Read the next frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame
                tracking_results = self.process_frame(frame, confidence=confidence)
                
                # Annotate the frame
                annotated_frame = self.annotate_frame(
                    frame, 
                    tracking_results,
                    draw_traces=draw_traces,
                    draw_boxes=draw_boxes
                )
                
                # Write to output video
                out.write(annotated_frame)
                
                # Show preview if requested
                if show_preview:
                    cv2.imshow('Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Update counters
                frame_count += 1
                processed_count += 1
                pbar.update(1)
                
                # Limit FPS if specified
                if fps_limit:
                    process_time = time.time() - start_process_time
                    sleep_time = max(0, fps_delay - process_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
            pbar.close()
        
        print(f"Processed {processed_count} frames")
        print(f"Output saved to: {os.path.abspath(output_video_path)}")
    
    def _convert_to_sv_detections(self, detection_results: Dict, frame: np.ndarray) -> sv.Detections:
        """
        Convert detection results to supervision Detections format.
        
        Args:
            detection_results: Detection results from detector.predict()
            frame: Original frame
            
        Returns:
            supervision.Detections object
        """
        predictions = detection_results.get('predictions', [])
        
        if not predictions:
            # Return empty detections
            return sv.Detections.empty()
        
        # Extract bounding boxes, confidence scores, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in predictions:
            # Get box coordinates (convert from center format to xyxy format)
            x_center = pred['x']
            y_center = pred['y']
            width = pred['width']
            height = pred['height']
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(pred['confidence'])
            class_ids.append(pred['class_id'])
        
        # Convert to numpy arrays
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)
        
        # Create and return supervision Detections
        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
