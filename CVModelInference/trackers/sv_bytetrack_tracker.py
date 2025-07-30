"""
Supervision ByteTrack Tracker Implementation Module

This module implements the Tracker interface using the supervision library's ByteTrack implementation,
which provides a more robust tracking solution compared to the custom implementation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, cast
import cv2
import supervision as sv
import numpy.typing as npt

from ..tracker import Tracker

# Type aliases
NDArray = npt.NDArray[np.float32]
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Detection = Dict[str, Any]  # Dictionary representing a detection
Track = Dict[str, Any]  # Dictionary representing a track


class SVByteTrackTracker(Tracker):
    """
    Implementation of the Tracker interface using supervision's ByteTrack algorithm.
    
    This class wraps the supervision ByteTrack implementation to conform to the
    project's Tracker interface.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the supervision ByteTrack tracker.
        
        Args:
            **kwargs: Additional parameters for ByteTrack
        """
        # Store parameters for reference
        self.params: Dict[str, Any] = kwargs
        
        # Initialize the supervision ByteTrack tracker with appropriate parameters
        self.tracker = sv.ByteTrack(
            track_activation_threshold=float(kwargs.get('track_activation_threshold', 0.25)),
            lost_track_buffer=int(kwargs.get('lost_track_buffer', 30)),
            minimum_matching_threshold=float(kwargs.get('minimum_matching_threshold', 0.8)),
            frame_rate=int(kwargs.get('frame_rate', 30))  # Must be int for ByteTrack
        )
        
        # Store tracker parameters for reference
        self.lost_track_buffer: int = int(kwargs.get('lost_track_buffer', 30))
        self.track_activation_threshold: float = float(kwargs.get('track_activation_threshold', 0.25))
        self.minimum_matching_threshold: float = float(kwargs.get('minimum_matching_threshold', 0.8))
        
        # Initialize state variables
        self.frame_count: int = 0
        self.tracks: Dict[int, List[Tuple[float, float]]] = {}  # For visualization trails
        self.class_mapping: Dict[int, int] = {}  # track_id -> class_id
        self.active_tracks: Set[int] = set()
        self.lost_tracks: Set[int] = set()
        self.track_history: Dict[int, int] = {}  # track_id -> frames_since_last_update
        self.total_tracks_created: int = 0
        self.total_tracks_lost: int = 0
        self.total_fragmentations: int = 0
        self.active_tracks = set()  # To track currently active track IDs
        self.lost_tracks = set()    # To track recently lost tracks
        self.track_history = {}     # To track frame counts and other stats
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_fragmentations = 0
        
    def init(self, frame: np.ndarray, detections: List[Dict]) -> bool:
        """
        Initialize the tracker with the first frame and initial detections.
        
        Args:
            frame: First frame as numpy array
            detections: List of detection dictionaries
                        
        Returns:
            Boolean indicating if initialization was successful
        """
        if frame is None or not detections:
            return False
            
        # Reset state
        self.reset()
        
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert detections to supervision format and update tracker
        sv_detections = self._convert_to_sv_detections(detections, frame)
        self.tracker.update_with_detections(sv_detections)
        
        # Update frame count
        self.frame_count = 1
        
        # Store class information for each detection
        for det in detections:
            class_id = det.get('class_id', -1)
            class_name = det.get('class', '')
            self.class_mapping[class_id] = class_name
        
        return True
    
    def update(self, frame: np.ndarray, detections: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Update the tracker with a new frame and optional new detections.
        
        Args:
            frame: New frame as numpy array (H, W, C) in BGR format
            detections: Optional list of detection dictionaries to incorporate
                        
        Returns:
            List of tracked objects with updated positions and IDs
        """
        # Convert detections to supervision format
        sv_detections = self._convert_to_sv_detections(detections or [], frame)
        
        # Update tracker
        self.tracker.update_with_detections(sv_detections)
        
        # Get updated tracks
        tracks = []
        confidences = sv_detections.confidence if sv_detections.confidence is not None else [1.0] * len(sv_detections)
        class_ids = sv_detections.class_id if sv_detections.class_id is not None else [-1] * len(sv_detections)
        tracker_ids = sv_detections.tracker_id if sv_detections.tracker_id is not None else range(len(sv_detections))
        
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(zip(
            sv_detections.xyxy,
            confidences,
            class_ids,
            tracker_ids
        )):
            # Convert xyxy to [x, y, w, h] format
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            x = x1 + w / 2
            y = y1 + h / 2
            
            # Ensure tracker_id is an integer
            track_id = int(tracker_id) if tracker_id is not None else i
            
            # Update track history
            if track_id not in self.tracks:
                self.tracks[track_id] = []
                self.total_tracks_created += 1
                
            # Store center point for trail
            center = (float(x), float(y))
            self.tracks[track_id].append(center)
            
            # Limit trail length
            if len(self.tracks[track_id]) > 30:  # Keep last 30 points
                self.tracks[track_id] = self.tracks[track_id][-30:]
            
            # Create track dictionary
            track: Dict[str, Any] = {
                'id': track_id,
                'x': float(x),
                'y': float(y),
                'width': float(w),
                'height': float(h),
                'confidence': float(confidence),
                'class_id': int(class_id) if class_id is not None else -1,
                'trail': self.tracks[track_id].copy()
            }
            tracks.append(track)
            
            # Update class mapping
            if class_id is not None and class_id != -1:
                self.class_mapping[track_id] = int(class_id)
        
        # Update frame count
        self.frame_count += 1
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return tracks
    
    def reset(self) -> None:
        """
        Reset the tracker state.
        
        Returns:
            None
        """
        self.tracker.reset()
        self.frame_count = 0
        self.tracks = {}
        self.class_mapping = {}
        self.active_tracks = set()
        self.lost_tracks = set()
        self.track_history = {}
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_fragmentations = 0
        
    def _cleanup_old_tracks(self) -> None:
        """
        Clean up old tracks that haven't been seen in a while.
        
        This method removes tracks that haven't been updated for more than
        lost_track_buffer frames.
        """
        if not hasattr(self.tracker, 'trackers') or not hasattr(self.tracker.trackers, 'tracked_stracks'):
            return
            
        current_tracks = self.tracker.trackers.tracked_stracks
        active_ids: Set[int] = set()
        
        # Update active tracks
        for track in current_tracks:
            if hasattr(track, 'is_activated') and track.is_activated and hasattr(track, 'track_id'):
                try:
                    active_ids.add(int(track.track_id))
                except (ValueError, TypeError):
                    continue
        
        self.active_tracks = active_ids
        
        # Update lost tracks
        lost_tracks: Set[int] = set()
        for track in current_tracks:
            if (hasattr(track, 'is_activated') and not track.is_activated and 
                hasattr(track, 'track_id') and track.track_id is not None):
                try:
                    lost_tracks.add(int(track.track_id))
                except (ValueError, TypeError):
                    continue
        
        self.lost_tracks = lost_tracks
        
        # Update track history and remove old tracks
        to_remove = []
        for track_id in list(self.track_history.keys()):
            if track_id in active_ids:
                self.track_history[track_id] = 0
            else:
                self.track_history[track_id] = self.track_history.get(track_id, 0) + 1
                if self.track_history[track_id] > self.lost_track_buffer * 2:
                    to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in to_remove:
            self.track_history.pop(track_id, None)
            self.tracks.pop(track_id, None)
            self.class_mapping.pop(track_id, None)
            
    def get_tracker_info(self) -> Dict[str, Any]:
        """
        Get information about the tracker.
        
        Returns:
            Dictionary containing tracker information
        """
        return {
            'name': 'Supervision ByteTrack',
            'type': 'sv_bytetrack',
            'parameters': {
                'track_activation_threshold': self.track_activation_threshold,
                'lost_track_buffer': self.lost_track_buffer,
                'minimum_matching_threshold': self.minimum_matching_threshold,
                'frame_rate': int(self.tracker.frame_rate) if hasattr(self.tracker, 'frame_rate') else 30
            },
            'frame_count': self.frame_count,
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_lost': self.total_tracks_lost,
            'total_fragmentations': self.total_fragmentations
        }
        
    def visualize(self, frame: np.ndarray, tracks: List[Dict], 
                 output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize the tracks on the input frame.
        
        Args:
            frame: Input frame
            tracks: Tracking results from update()
            output_path: Optional path to save the visualization
            
        Returns:
            Frame with tracks visualized as a numpy array
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
            
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Create annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        
        # Convert our tracks to supervision format
        xyxy = []
        class_ids = []
        confidences = []
        tracker_ids = []
        traces: Dict[int, np.ndarray] = {}
        
        for track in tracks:
            # Skip if track is not a dictionary or doesn't have required fields
            if not isinstance(track, dict) or not all(k in track for k in ['x', 'y', 'width', 'height']):
                continue
            
            # Extract track data with type safety
            try:
                x = float(track.get('x', 0))
                y = float(track.get('y', 0))
                w = float(track.get('width', 0))
                h = float(track.get('height', 0))
                
                # Convert to xyxy format
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                xyxy.append([x1, y1, x2, y2])
                class_ids.append(int(track.get('class_id', -1)))
                confidences.append(float(track.get('confidence', 1.0)))
                tracker_ids.append(int(track.get('id', 0)))
                
                # Add trace points if available
                if 'trail' in track and isinstance(track['trail'], (list, np.ndarray)) and len(track['trail']) > 0:
                    traces[track['id']] = np.array(track['trail'], dtype=np.float32)
                    
            except (ValueError, TypeError) as e:
                continue
        
        # Create supervision detections if we have any boxes
        if xyxy:
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int) if class_ids else None,
                confidence=np.array(confidences, dtype=float) if confidences else None,
                tracker_id=np.array(tracker_ids, dtype=int) if tracker_ids else None
            )
            
            # Annotate frame with boxes
            vis_frame = box_annotator.annotate(scene=vis_frame, detections=detections)
            
            # Add labels manually
            for i in range(len(detections)):
                if i >= len(detections):
                    continue
                    
                try:
                    # Get box coordinates
                    x1, y1, x2, y2 = detections.xyxy[i]
                    
                    # Get label information
                    class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                    tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
                    confidence = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                    
                    # Create label
                    label = f"ID: {tracker_id}" if tracker_id is not None else f"Class: {class_id}"
                    if confidence < 1.0:  # Only show confidence if not 1.0
                        label += f" ({confidence:.2f})"
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_frame, (int(x1), int(y1) - 20), (int(x1) + label_w, int(y1)), (0, 0, 0), -1)
                    
                    # Draw label text
                    cv2.putText(vis_frame, label, (int(x1), int(y1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except (IndexError, ValueError, TypeError):
                    continue
        
        # Draw trails
        for track_id, points in traces.items():
            if len(points) >= 2:
                points = np.array(points, dtype=np.int32)
                color_hash = hash(track_id) % 0xFFFFFF
                color = (int(color_hash & 0xFF), int((color_hash >> 8) & 0xFF), int((color_hash >> 16) & 0xFF))
                
                for i in range(1, len(points)):
                    try:
                        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                        pt2 = (int(points[i][0]), int(points[i][1]))
                        cv2.line(vis_frame, pt1, pt2, color, 2)
                    except (IndexError, ValueError, TypeError):
                        continue
        
        # Save visualization if output path is provided
        if output_path and isinstance(output_path, str):
            try:
                cv2.imwrite(output_path, vis_frame)
            except Exception as e:
                print(f"Error saving visualization to {output_path}: {e}")
            
        return vis_frame
    
    # get_tracker_info method is already defined above
        
    def _convert_to_sv_detections(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> sv.Detections:
        """
        Convert our detection format to supervision Detections format.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame (H, W, C) in BGR format
            
        Returns:
            supervision.Detections object with xyxy, confidence, and class_id
        """
        if not detections or not isinstance(detections, list):
            return sv.Detections.empty()
            
        # Initialize lists to store detection data
        bboxes: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []
        
        for det in detections:
            # Skip if detection is not a dictionary or doesn't have required fields
            if not isinstance(det, dict) or 'bbox' not in det:
                continue
            
            # Handle different bbox formats with type safety
            bbox = det.get('bbox', [])
            if not isinstance(bbox, (list, np.ndarray)) or len(bbox) < 4:
                continue
                
            try:
                if len(bbox) == 4:  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(float, bbox[:4])
                elif len(bbox) >= 5:  # [x1, y1, x2, y2, conf, ...]
                    x1, y1, x2, y2 = map(float, bbox[:4])
                else:
                    continue  # Skip invalid bbox format
                    
                # Skip detections with low confidence
                conf = float(det.get('confidence', 0))
                if conf < self.track_activation_threshold:
                    continue
                    
                # Clip coordinates to frame boundaries
                if not isinstance(frame, np.ndarray) or len(frame.shape) < 2:
                    continue
                    
                h, w = frame.shape[:2]
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                bboxes.append([x1, y1, x2, y2])
                confidences.append(conf)
                class_ids.append(int(det.get('class_id', -1)))
                
            except (ValueError, TypeError) as e:
                continue
            
        if not bboxes:
            return sv.Detections.empty()
            
        try:
            # Convert to numpy arrays with explicit types
            bboxes_np = np.array(bboxes, dtype=np.float32)
            confidences_np = np.array(confidences, dtype=np.float32)
            class_ids_np = np.array(class_ids, dtype=int)
            
            # Create and return supervision Detections object
            return sv.Detections(
                xyxy=bboxes_np,
                confidence=confidences_np,
                class_id=class_ids_np if any(id_ != -1 for id_ in class_ids) else None
            )
        except Exception as e:
            return sv.Detections.empty()
