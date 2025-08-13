#!/usr/bin/env python3
"""
Transform snooker detection points using perspective transformation and filter false positives.

This script loads detection data from a JSONL file, transforms all points using a 
pre-computed perspective transformation matrix, and filters out points that fall
outside the valid playing area boundaries.
"""

import sys
import os
import json
import cv2
import numpy as np

from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray
from collections import defaultdict


# Playing area boundaries in the standardized coordinate system (0-2022 x 0-4056)
# These are the boundaries after transformation, not the full table boundaries
PLAY_AREA_TOP_LEFT_X = 0
PLAY_AREA_TOP_LEFT_Y = 0
PLAY_AREA_BOTTOM_RIGHT_X = 2022
PLAY_AREA_BOTTOM_RIGHT_Y = 4056
PLAY_AREA_WIDTH = 2022
PLAY_AREA_HEIGHT = 4056


def is_point_in_playing_area(point: Tuple[float, float]) -> bool:
    """
    Check if a point is within the valid playing area boundaries.
    
    Args:
        point: (x, y) coordinates to check
        
    Returns:
        True if point is within playing area, False otherwise
    """
    x, y = point
    return (PLAY_AREA_TOP_LEFT_X <= x <= PLAY_AREA_BOTTOM_RIGHT_X and 
            PLAY_AREA_TOP_LEFT_Y <= y <= PLAY_AREA_BOTTOM_RIGHT_Y)





def transform_point(point: Tuple[float, float], transformation_matrix: NDArray) -> Tuple[float, float]:
    """
    Transform a single point using a perspective transformation matrix.
    
    Args:
        point: (x, y) coordinates to transform
        transformation_matrix: 3x3 perspective transformation matrix
        
    Returns:
        Transformed (x, y) coordinates
    """
    # Convert point to homogeneous coordinates
    point_homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
    
    # Apply transformation
    transformed_homogeneous = transformation_matrix @ point_homogeneous
    
    # Convert back from homogeneous coordinates
    if transformed_homogeneous[2] != 0:
        x = transformed_homogeneous[0] / transformed_homogeneous[2]
        y = transformed_homogeneous[1] / transformed_homogeneous[2]
    else:
        # Handle edge case where z = 0
        x = transformed_homogeneous[0]
        y = transformed_homogeneous[1]
    
    return (float(x), float(y))


def load_transformation_config(config_path: str) -> Dict[str, Any]:
    """
    Load transformation configuration from JSON file.
    
    Args:
        config_path: Path to transformation config JSON file
        
    Returns:
        Dictionary containing transformation configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert transformation matrix to numpy array
    config['transformation_matrix'] = np.array(config['transformation_matrix'])
    
    return config  # type: ignore[no-any-return]


def transform_detections_with_filtering(input_jsonl_path: str, 
                                      transformation_config_path: str, 
                                      output_json_path: str,
                                      false_positives_report_path: str,
                                      filtered_predictions_report_path: str | None = None) -> None:
    """
    Transform all detection points in a JSONL file using perspective transformation
    and filter out false positives outside the playing area.
    
    Args:
        input_jsonl_path: Path to input JSONL file with detections
        transformation_config_path: Path to transformation config JSON file
        output_json_path: Path to output JSON file with transformed detections
        false_positives_report_path: Path to output JSON file with false positive report
    """
    print(f"Loading transformation config from: {transformation_config_path}")
    config = load_transformation_config(transformation_config_path)
    transformation_matrix = config['transformation_matrix']
    
    print(f"Reading detections from: {input_jsonl_path}")
    print(f"Output will be saved to: {output_json_path}")
    print(f"False positives report will be saved to: {false_positives_report_path}")
    
    transformed_detections: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []
    false_positive_stats: Dict[str, int] = defaultdict(int)
    
    # Track individual filtered predictions
    individual_filtered_predictions: List[Dict[str, Any]] = []
    
    # Read JSONL file line by line
    with open(input_jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                detection = json.loads(line.strip())
                
                # Create a copy of the detection for transformation
                transformed_detection = detection.copy()
                is_valid_detection = True
                rejection_reasons = []
                
                # Transform predictions (the main coordinate data)
                if 'predictions' in detection:
                    transformed_predictions = []
                    for pred in detection['predictions']:
                        transformed_pred = pred.copy()
                        
                        # Transform x, y coordinates
                        if 'x' in pred and 'y' in pred:
                            original_point = (pred['x'], pred['y'])
                            transformed_point = transform_point(original_point, transformation_matrix)
                            transformed_pred['x'] = transformed_point[0]
                            transformed_pred['y'] = transformed_point[1]
                        
                        transformed_predictions.append(transformed_pred)
                    
                    transformed_detection['predictions'] = transformed_predictions
                
                # Transform bounding box coordinates if they exist
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    if len(bbox) == 4:  # [x1, y1, x2, y2] format
                        # Transform the four corners of the bounding box
                        corners = [
                            (bbox[0], bbox[1]),  # top-left
                            (bbox[2], bbox[1]),  # top-right
                            (bbox[2], bbox[3]),  # bottom-right
                            (bbox[0], bbox[3])   # bottom-left
                        ]
                        
                        transformed_corners = [transform_point(corner, transformation_matrix) for corner in corners]
                        
                        # Calculate new bounding box from transformed corners
                        x_coords = [corner[0] for corner in transformed_corners]
                        y_coords = [corner[1] for corner in transformed_corners]
                        
                        new_bbox = [
                            min(x_coords),  # x1
                            min(y_coords),  # y1
                            max(x_coords),  # x2
                            max(y_coords)   # y2
                        ]
                        
                        transformed_detection['bbox'] = new_bbox
                
                # Transform center point if it exists
                if 'center' in detection:
                    center = detection['center']
                    if isinstance(center, list) and len(center) == 2:  # [x, y] format
                        center_tuple = (center[0], center[1])
                        transformed_center = transform_point(center_tuple, transformation_matrix)
                        transformed_detection['center'] = transformed_center
                
                # Transform any other point fields that might exist
                point_fields = ['position', 'location', 'point']
                for field in point_fields:
                    if field in detection:
                        point = detection[field]
                        if isinstance(point, list) and len(point) == 2:
                            point_tuple = (point[0], point[1])
                            transformed_point = transform_point(point_tuple, transformation_matrix)
                            transformed_detection[field] = transformed_point
                
                # Filter out individual predictions that are outside the playing area
                # Keep valid predictions and remove only the out-of-bounds ones
                if 'predictions' in transformed_detection and len(transformed_detection['predictions']) > 0:
                    original_predictions = transformed_detection['predictions']
                    filtered_predictions = []
                    
                    for i, pred in enumerate(original_predictions):
                        if 'x' in pred and 'y' in pred:
                            center_point = (pred['x'], pred['y'])
                            if is_point_in_playing_area(center_point):
                                # Keep this prediction - it's within bounds
                                filtered_predictions.append(pred)
                            else:
                                # This prediction is outside bounds - track it individually
                                false_positive_stats["predictions_outside_playing_area"] += 1
                                
                                # Track the filtered prediction for reporting
                                filtered_prediction_info = {
                                    'frame': detection.get('frame'),
                                    'time': detection.get('time'),
                                    'prediction_index': i,
                                    'original_prediction': pred,
                                    'transformed_coordinates': center_point,
                                    'reason': 'outside_playing_area'
                                }
                                individual_filtered_predictions.append(filtered_prediction_info)
                        else:
                            # Keep predictions without x,y coordinates (shouldn't happen but just in case)
                            filtered_predictions.append(pred)
                    
                    # Update the detection with filtered predictions
                    transformed_detection['predictions'] = filtered_predictions
                    
                    # If all predictions were filtered out, mark as having no balls
                    if len(filtered_predictions) == 0:
                        transformed_detection['no_balls_detected'] = True
                    
                    # Note: We don't add rejection reasons for individual prediction filtering
                    # because we want to keep the frame as valid if it has any valid predictions
                elif 'predictions' in transformed_detection and len(transformed_detection['predictions']) == 0:
                    # Empty predictions - no balls detected in this frame
                    # This is not a false positive, just a frame with no detections
                    # Keep the detection as valid but mark it as having no balls
                    transformed_detection['no_balls_detected'] = True
                elif 'center' in transformed_detection:
                    # Check center point if it exists
                    center_point = transformed_detection['center']
                    if not is_point_in_playing_area(center_point):
                        is_valid_detection = False
                        rejection_reasons.append("center_outside_playing_area")
                        false_positive_stats["center_outside_playing_area"] += 1
                else:
                    # If no predictions and no center point found, mark as invalid
                    is_valid_detection = False
                    rejection_reasons.append("no_center_point_found")
                    false_positive_stats["no_center_point_found"] += 1
                
                # All frames are now considered valid (we filter individual predictions instead)
                # Only add to false positives if there are actual frame-level issues
                if rejection_reasons:
                    transformed_detection['rejection_reasons'] = rejection_reasons
                    transformed_detection['line_number'] = line_num
                    false_positives.append(transformed_detection)
                else:
                    transformed_detections.append(transformed_detection)
                
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} detections...")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                false_positive_stats["invalid_json"] += 1
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                false_positive_stats["processing_error"] += 1
                continue
    
    # Calculate total predictions and false positive percentage
    total_predictions: int = 0
    valid_predictions: int = 0
    
    # Count total predictions in all frames
    for detection in transformed_detections:
        if 'predictions' in detection:
            total_predictions += len(detection['predictions'])
            valid_predictions += len(detection['predictions'])
    
    # Add predictions from false positive frames (if any)
    for detection in false_positives:
        if 'predictions' in detection:
            total_predictions += len(detection['predictions'])
    
    # Add the filtered out predictions to total
    total_predictions += false_positive_stats["predictions_outside_playing_area"]
    
    # Calculate false positive percentage
    false_positive_percentage: float = (false_positive_stats["predictions_outside_playing_area"] / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"Total frames processed: {len(transformed_detections) + len(false_positives)}")
    print(f"Valid frames: {len(transformed_detections)}")
    print(f"False positive frames: {len(false_positives)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Valid predictions: {valid_predictions}")
    print(f"Filtered out-of-bounds predictions: {false_positive_stats['predictions_outside_playing_area']}")
    print(f"False positive percentage: {false_positive_percentage:.2f}%")
    
    # Print false positive statistics
    print("\nFalse Positive Statistics:")
    for reason, count in false_positive_stats.items():
        print(f"  {reason}: {count}")
    
    # Save transformed detections to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(transformed_detections, f, indent=2)
    
    # Save false positives report
    false_positives_report = {
        "summary": {
            "total_frames": len(transformed_detections) + len(false_positives),
            "valid_frames": len(transformed_detections),
            "false_positive_frames": len(false_positives),
            "total_predictions": total_predictions,
            "valid_predictions": valid_predictions,
            "filtered_out_of_bounds_predictions": false_positive_stats["predictions_outside_playing_area"],
            "false_positive_percentage": false_positive_percentage
        },
        "false_positive_statistics": dict(false_positive_stats),
        "false_positives": false_positives
    }
    
    with open(false_positives_report_path, 'w') as f:
        json.dump(false_positives_report, f, indent=2)
    
    # Save filtered predictions report if requested
    if filtered_predictions_report_path:
        filtered_predictions_report = {
            "summary": {
                "total_filtered_predictions": len(individual_filtered_predictions),
                "filtered_predictions_by_reason": {
                    "outside_playing_area": len([p for p in individual_filtered_predictions if p.get('reason') == 'outside_playing_area'])
                }
            },
            "filtered_predictions": individual_filtered_predictions
        }
        
        with open(filtered_predictions_report_path, 'w') as f:
            json.dump(filtered_predictions_report, f, indent=2)
        
        print(f"Filtered predictions report saved to: {filtered_predictions_report_path}")
    
    print(f"\nTransformed detections saved to: {output_json_path}")
    print(f"False positives report saved to: {false_positives_report_path}")


if __name__ == "__main__":
    # Use the specific file paths provided
    input_jsonl_path = "/Users/abhinavrai/Playground/physics_inference/assets/output/snooker_detections.txt"
    transformation_config_path = "/Users/abhinavrai/Playground/physics_inference/assets/output/transformed.png/transformation_config.json"
    output_json_path = "/Users/abhinavrai/Playground/physics_inference/assets/output/transformed_detections.json"
    false_positives_report_path = "/Users/abhinavrai/Playground/physics_inference/assets/output/false_positives_report.json"
    
    # Validate input files exist
    if not os.path.exists(input_jsonl_path):
        print(f"Error: Input file not found: {input_jsonl_path}")
        sys.exit(1)
    
    if not os.path.exists(transformation_config_path):
        print(f"Error: Transformation config file not found: {transformation_config_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    transform_detections_with_filtering(
        input_jsonl_path, 
        transformation_config_path, 
        output_json_path, 
        false_positives_report_path,
        "/Users/abhinavrai/Playground/physics_inference/assets/output/filtered_predictions_report.json"
    )
