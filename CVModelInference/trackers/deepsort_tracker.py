"""
DeepSORT Tracker Implementation Module

This module implements the Tracker interface using DeepSORT algorithm,
which extends SORT with appearance features for more robust tracking.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, TypedDict, cast, Sequence
import cv2
from scipy.optimize import linear_sum_assignment
import numpy.typing as npt

from ..tracker import Tracker, Frame, Detection, Track


# Type aliases
BBox = Tuple[float, float, float, float]  # [x, y, width, height]
Point = Tuple[int, int]  # (x, y) coordinates
FeatureVector = npt.NDArray[np.float32]


class ClassInfo(TypedDict, total=False):
    """Type definition for class information in trackers."""
    class_name: str
    class_id: int
    confidence: float


class TrackerState(TypedDict):
    """Type definition for internal tracker state."""
    id: int
    bbox: BBox
    features: FeatureVector
    class_info: ClassInfo
    time_since_update: int
    hits: int
    hit_streak: int
    age: int
    trail: List[Point]


class FeatureExtractor:
    """
    Simple feature extractor for DeepSORT.
    
    In a real implementation, this would use a deep neural network to extract
    appearance features. For simplicity, we use a basic color histogram approach.
    """
    
    def __init__(self, num_bins: int = 16) -> None:
        """
        Initialize feature extractor.
        
        Args:
            num_bins: Number of bins for color histogram
        """
        self.num_bins = num_bins
        
    def extract(self, image: np.ndarray, bbox: BBox) -> FeatureVector:
        """
        Extract features from image patch using color histograms.
        
        Args:
            image: Input image in BGR format with shape (H, W, 3)
            bbox: Bounding box in [x, y, width, height] format
            
        Returns:
            Feature vector as a numpy array of shape (num_bins * 3,)
        """
        # Convert bbox to integers
        x, y, w, h = bbox
        x1, y1 = int(round(x - w/2)), int(round(y - h/2))
        x2, y2 = int(round(x + w/2)), int(round(y + h/2))
        
        # Clip coordinates to image boundaries
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Extract patch
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros(self.num_bins * 3, dtype=np.float32)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [self.num_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.num_bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.num_bins], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate histograms
        features = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        
        return features


class DeepSORTTracker(Tracker[Dict[str, Any]]):
    """
    Implementation of the Tracker interface using DeepSORT algorithm.
    
    This tracker combines motion and appearance information for robust
    multi-object tracking across video frames.
    """
    
    def __init__(
        self, 
        max_age: int = 30, 
        min_hits: int = 3, 
        iou_threshold: float = 0.3, 
        appearance_threshold: float = 0.5, 
        **kwargs: Any
    ) -> None:
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track alive without matching
            min_hits: Minimum number of hits needed to display a track
            iou_threshold: Intersection over Union threshold for matching (0.0 to 1.0)
            appearance_threshold: Threshold for appearance feature similarity (0.0 to 1.0)
            **kwargs: Additional parameters (ignored)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        
        # Internal state
        self.trackers: List[TrackerState] = []
        self.frame_count: int = 0
        self.next_id: int = 0
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
    def init(self, frame: Frame, detections: List[Detection]) -> bool:
        """
        Initialize the tracker with the first frame and initial detections.
        
        Args:
            frame: First frame as numpy array with shape (H, W, 3) in BGR format
            detections: List of detection dictionaries, each containing at minimum:
                       {
                           'x': float,  # center x coordinate
                           'y': float,  # center y coordinate
                           'width': float,  # bounding box width
                           'height': float,  # bounding box height
                           'class': str,  # class name
                           'confidence': float,  # detection confidence
                           'id': Optional[Union[int, str]]  # optional track ID
                       }
                        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.frame_count = 0
        self.trackers = []
        self.next_id = 0
        
        if frame is None or len(detections) == 0:
            return False
        
        # Initialize trackers for each detection
        for detection in detections:
            x = float(detection.get('x', 0))
            y = float(detection.get('y', 0))
            w = float(detection.get('width', 0))
            h = float(detection.get('height', 0))
            
            # Extract features from the detected region
            features = self.feature_extractor.extract(frame, (x, y, w, h))
            
            # Create class info dictionary with type-safe access
            class_info: ClassInfo = {}
            if 'class' in detection:
                class_info['class_name'] = str(detection['class'])
            if 'class_id' in detection:
                class_info['class_id'] = int(detection['class_id'])
            if 'confidence' in detection:
                class_info['confidence'] = float(detection['confidence'])
                    
            # Create tracker state with type-safe dictionary
            tracker: TrackerState = {
                'id': self.next_id,
                'bbox': (x, y, w, h),
                'features': features,
                'class_info': class_info,
                'time_since_update': 0,
                'hits': 1,
                'hit_streak': 1,
                'age': 0,
                'trail': [(int(round(x)), int(round(y)))]
            }
            
            self.trackers.append(tracker)
            self.next_id += 1
            
        self.frame_count += 1
        return True
    
    def update(self, frame: Frame, detections: Optional[List[Detection]] = None) -> List[Track]:
        """
        Update the tracker with a new frame and optional new detections.
        
        Args:
            frame: New frame as numpy array with shape (H, W, 3) in BGR format
            detections: Optional list of new detections to incorporate.
                       If None or empty, the tracker will only update existing tracks.
                       
                       Each detection dictionary should contain:
                       {
                           'x': float,  # center x coordinate
                           'y': float,  # center y coordinate
                           'width': float,  # bounding box width
                           'height': float,  # bounding box height
                           'class': str,  # class name
                           'confidence': float,  # detection confidence
                           'id': Optional[Union[int, str]]  # optional track ID
                       }
                        
        Returns:
            List[Track]: List of tracked objects with updated positions and IDs.
                         Each track is a dictionary containing at least:
                         {
                             'id': int,  # unique track ID
                             'x': float,  # center x coordinate
                             'y': float,  # center y coordinate
                             'width': float,  # bounding box width
                             'height': float,  # bounding box height
                             'class': str,  # class name
                             'confidence': float,  # detection confidence
                             'age': int,  # number of frames since first detection
                             'active': bool  # whether the track is currently active
                         }
        """
        self.frame_count += 1
        
        # If no detections, just update trackers with empty list
        if not detections:
            return self._update_trackers(frame, [])
            
        # Extract features from detections
        det_boxes: List[BBox] = []
        det_features: List[FeatureVector] = []
        
        for det in detections:
            x = float(det.get('x', 0))
            y = float(det.get('y', 0))
            w = float(det.get('width', 0))
            h = float(det.get('height', 0))
            
            # Extract features from the detected region
            features = self.feature_extractor.extract(frame, (x, y, w, h))
            
            det_boxes.append((x, y, w, h))
            det_features.append(features)
        
        # Update trackers with new detections
        return self._update_trackers(frame, detections, det_boxes, det_features)
            
        # Process new detections
        if detections and len(detections) > 0:
            # Convert detections to format [x, y, w, h]
            dets = []
            det_features = []
            det_info = []
            
            for detection in detections:
                x = detection['x']
                y = detection['y']
                w = detection['width']
                h = detection['height']
                dets.append([x, y, w, h])
                
                # Extract features
                features = self.feature_extractor.extract(frame, [x, y, w, h])
                det_features.append(features)
                
                # Store additional info
                info = {}
                for key in ['class', 'confidence', 'class_id']:
                    if key in detection:
                        info[key] = detection[key]
                det_info.append(info)
                
            # Associate detections to trackers
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                dets, det_features, frame)
                
            # Update matched trackers with assigned detections
            for det_idx, trk_idx in matched:
                self.trackers[trk_idx]['bbox'] = dets[det_idx]
                self.trackers[trk_idx]['time_since_update'] = 0
                self.trackers[trk_idx]['hits'] += 1
                self.trackers[trk_idx]['hit_streak'] += 1
                
                # Update features with moving average
                alpha = 0.7  # Weight for new features
                old_features = self.trackers[trk_idx]['features']
                new_features = det_features[det_idx]
                self.trackers[trk_idx]['features'] = alpha * new_features + (1 - alpha) * old_features
                
                # Update trail
                x, y = int(dets[det_idx][0]), int(dets[det_idx][1])
                self.trackers[trk_idx]['trail'].append((x, y))
                if len(self.trackers[trk_idx]['trail']) > 30:  # Limit trail length
                    self.trackers[trk_idx]['trail'] = self.trackers[trk_idx]['trail'][-30:]
                    
                # Update class info if confidence is higher
                if ('confidence' in det_info[det_idx] and 
                    ('confidence' not in self.trackers[trk_idx]['class_info'] or 
                     det_info[det_idx]['confidence'] > self.trackers[trk_idx]['class_info']['confidence'])):
                    self.trackers[trk_idx]['class_info'] = det_info[det_idx]
                
            # Create and initialize new trackers for unmatched detections
            for det_idx in unmatched_dets:
                tracker = {
                    'id': len(self.trackers),
                    'bbox': dets[det_idx],
                    'features': det_features[det_idx],
                    'class_info': det_info[det_idx],
                    'time_since_update': 0,
                    'hits': 1,
                    'hit_streak': 1,
                    'age': 0,
                    'trail': [(int(dets[det_idx][0]), int(dets[det_idx][1]))]
                }
                
                self.trackers.append(tracker)
                
        # Mark trackers for removal instead of removing while iterating
        to_remove = []
        ret = []
        
        for i, tracker in enumerate(self.trackers):
            if ((tracker['time_since_update'] < self.max_age) and 
                (tracker['hit_streak'] >= self.min_hits or self.frame_count <= self.min_hits)):
                
                x, y, w, h = tracker['bbox']
                
                track = {
                    'id': tracker['id'],
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'trail': tracker['trail'].copy(),
                    'age': tracker['age'],
                    'hits': tracker['hits'],
                    'time_since_update': tracker['time_since_update']
                }
                
                # Add class info if available
                if tracker['class_info']:
                    for key, value in tracker['class_info'].items():
                        track[key] = value
                        
                ret.append(track)
            
            # Mark dead tracks for removal
            elif tracker['time_since_update'] > self.max_age:
                to_remove.append(i)
        
        # Remove dead tracks in reverse order to avoid index shifting
        for i in sorted(to_remove, reverse=True):
            if i < len(self.trackers):  # Safety check
                self.trackers.pop(i)
                
        return ret
    
    def reset(self) -> None:
        """
        Reset the tracker state.
        
        This clears all internal state and prepares the tracker for a new sequence.
        """
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0
    
    def visualize(self, frame: Frame, tracks: Sequence[Track], 
                 output_path: Optional[str] = None) -> Frame:
        """
        Visualize the tracks on the input frame.
        
        Args:
            frame: Input frame as numpy array with shape (H, W, 3) in BGR format
            tracks: Tracking results from update()
            output_path: Optional path to save the visualization.
                        If provided, the visualization will be saved to this path.
            
        Returns:
            Frame: A copy of the input frame with tracks visualized.
                   The frame will have the same shape as the input (H, W, 3).
        """
        # Create a copy of the frame to draw on
        vis_frame = frame.copy()
        
        # Draw each track
        for track in tracks:
            try:
                x = float(track.get('x', 0))
                y = float(track.get('y', 0))
                w = float(track.get('width', 0))
                h = float(track.get('height', 0))
                track_id = int(track.get('id', -1))
                
                # Calculate bounding box coordinates
                x1 = int(round(x - w/2))
                y1 = int(round(y - h/2))
                x2 = int(round(x + w/2))
                y2 = int(round(y + h/2))
                
                # Generate consistent color based on track ID
                color = (
                    int(track_id * 50 % 255),
                    int(track_id * 100 % 255),
                    int(track_id * 150 % 255)
                )
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID
                cv2.putText(vis_frame, f'ID: {track_id}', (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw class and confidence if available
                class_name = track.get('class', '')
                confidence = track.get('confidence', 0.0)
                if class_name and confidence > 0:
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(vis_frame, label, (x1, y1 - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw trail if available
                if 'trail' in track and isinstance(track['trail'], list):
                    trail = [(int(round(p[0])), int(round(p[1]))) 
                            for p in track['trail'] 
                            if isinstance(p, (tuple, list)) and len(p) >= 2]
                    if len(trail) > 1:
                        cv2.polylines(vis_frame, [np.array(trail, np.int32)], 
                                     False, color, 2, cv2.LINE_AA)
                
            except (ValueError, KeyError, TypeError) as e:
                # Skip malformed tracks
                continue
        
        # Save visualization if output path is provided
        if output_path:
            try:
                cv2.imwrite(output_path, vis_frame)
            except Exception as e:
                print(f"Error saving visualization to {output_path}: {e}")
            
        return vis_frame
    
def visualize(self, frame: Frame, tracks: Sequence[Track], 
             output_path: Optional[str] = None) -> Frame:
    """
    Visualize the tracks on the input frame.
    
    Args:
        frame: Input frame as numpy array with shape (H, W, 3) in BGR format
        tracks: Tracking results from update()
        output_path: Optional path to save the visualization.
                    If provided, the visualization will be saved to this path.
            
    Returns:
        Frame: A copy of the input frame with tracks visualized.
               The frame will have the same shape as the input (H, W, 3).
    """
    # Create a copy of the frame to draw on
    vis_frame = frame.copy()
        
    # Draw each track
    for track in tracks:
        try:
            x = float(track.get('x', 0))
            y = float(track.get('y', 0))
            w = float(track.get('width', 0))
            h = float(track.get('height', 0))
            track_id = int(track.get('id', -1))
                
            # Calculate bounding box coordinates
            x1 = int(round(x - w/2))
            y1 = int(round(y - h/2))
            x2 = int(round(x + w/2))
            y2 = int(round(y + h/2))
                
            # Generate consistent color based on track ID
            color = (
                int(track_id * 50 % 255),
                int(track_id * 100 % 255),
                int(track_id * 150 % 255)
            )
                
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
            # Draw track ID
            cv2.putText(vis_frame, f'ID: {track_id}', (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Draw class and confidence if available
            class_name = track.get('class', '')
            confidence = track.get('confidence', 0.0)
            if class_name and confidence > 0:
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(vis_frame, label, (x1, y1 - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            # Draw trail if available
            if 'trail' in track and isinstance(track['trail'], list):
                trail = [(int(round(p[0])), int(round(p[1]))) 
                        for p in track['trail'] 
                        if isinstance(p, (tuple, list)) and len(p) >= 2]
                if len(trail) > 1:
                    cv2.polylines(vis_frame, [np.array(trail, np.int32)], 
                                 False, color, 2, cv2.LINE_AA)
                
        except (ValueError, KeyError, TypeError) as e:
            # Skip malformed tracks
            continue
        
    # Save visualization if output path is provided
    if output_path:
        try:
            cv2.imwrite(output_path, vis_frame)
        except Exception as e:
            print(f"Error saving visualization to {output_path}: {e}")
            
    return vis_frame
    
def get_tracker_info(self) -> Dict[str, Any]:
    """
    Get information about the tracker.
    
    Returns:
        Dict[str, Any]: Dictionary containing tracker information with the following
                      optional keys:
                      {
                          'name': str,  # tracker name
                          'type': str,  # tracker type/algorithm
                          'parameters': Dict[str, Any],  # tracker parameters
                          'frame_count': int,  # number of frames processed
                          'active_tracks': int,  # number of currently active tracks
                          'total_tracks_created': int,  # total tracks created so far
                          'total_tracks_lost': int,  # total tracks lost so far
                          'total_fragmentations': int  # total track fragmentations
                      }
    """
    active_tracks = len([t for t in self.trackers if t['time_since_update'] == 0])
    total_created = max([t['id'] for t in self.trackers] + [0]) + 1
    total_lost = len([t for t in self.trackers if t['time_since_update'] > 0])
        
    return {
        'name': 'DeepSORT',
        'type': 'appearance_based',
        'parameters': {
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold,
            'appearance_threshold': self.appearance_threshold
        },
        'frame_count': self.frame_count,
        'active_tracks': active_tracks,
        'total_tracks_created': total_created,
        'total_tracks_lost': total_lost,
        'total_fragmentations': 0  # Not implemented in this simple version
    }
    
def _associate_detections_to_trackers(
    self,
    detections: List[List[float]],
    det_features: List[FeatureVector],
    frame: Frame
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Assign detections to tracked objects using both IoU and appearance features.
    
    Args:
        detections: List of detections as [[x, y, width, height], ...]
        det_features: List of detection feature vectors
        frame: Current frame (unused, kept for compatibility)
            
    Returns:
        Tuple containing:
            - matches: Array of matched indices (detection_idx, tracker_idx)
            - unmatched_detections: List of unmatched detection indices
            - unmatched_trackers: List of unmatched tracker indices
    """
    if len(self.trackers) == 0:
        return np.empty((0, 2), dtype=int), list(range(len(detections))), []
            
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), [], list(range(len(self.trackers)))
        
    # Compute IoU matrix
    iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
        
    for d, det in enumerate(detections):
        for t, trk in enumerate(self.trackers):
            iou_matrix[d, t] = self._iou(det, list(trk['bbox']))
        
    # Compute appearance similarity matrix
    appearance_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
        
    for d, det_feat in enumerate(det_features):
        for t, trk in enumerate(self.trackers):
            # Compute cosine similarity between features
            trk_feat = trk['features']
            if len(trk_feat) > 0 and len(det_feat) > 0:
                # Normalize features
                trk_feat_norm = trk_feat / (np.linalg.norm(trk_feat) + 1e-6)
                det_feat_norm = det_feat / (np.linalg.norm(det_feat) + 1e-6)
                # Compute cosine similarity
                appearance_matrix[d, t] = float(np.dot(trk_feat_norm, det_feat_norm))
        
    # Combine IoU and appearance scores
    combined_matrix = iou_matrix * 0.5 + appearance_matrix * 0.5
        
    # Apply linear assignment (Hungarian algorithm)
    matched_indices = linear_sum_assignment(-combined_matrix)
    matched_indices = np.column_stack(matched_indices)
        
    # Find unmatched detections and trackers
    unmatched_detections = [d for d in range(len(detections)) 
                          if d not in matched_indices[:, 0]]
        
    unmatched_trackers = [t for t in range(len(self.trackers)) 
                        if t not in matched_indices[:, 1]]
        
    # Filter out matches with low score
    matches = []
    for m in matched_indices:
        if combined_matrix[m[0], m[1]] >= self.iou_threshold:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        
    if matches:
        matches = np.concatenate(matches, axis=0)
    else:
        matches = np.empty((0, 2), dtype=int)
        
    return matches, unmatched_detections, unmatched_trackers
    
def _iou(self, box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box as [x_center, y_center, width, height]
        box2: Second box as [x_center, y_center, width, height]
            
    Returns:
        float: IoU value between 0.0 and 1.0
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calculate intersection coordinates
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # Calculate IoU
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0.0
    
    return iou
