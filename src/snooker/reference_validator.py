"""
Reference validation utilities for snooker perspective transformation.

This module provides tools to validate perspective transformation results
against reference snooker table images and measurements.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging
from numpy.typing import NDArray

from .table.table_constants import (
    TABLE_WIDTH, TABLE_HEIGHT,
    TABLE_LEFT, TABLE_TOP, TABLE_RIGHT, TABLE_BOTTOM,
    BAULK_LINE_Y, MIDDLE_LINE_Y, MIDDLE_LINE_X,
    STANDARD_BALL_POSITIONS, BROWN_SPOT
)

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


class ReferenceValidator:
    """
    Validates perspective transformation results against reference measurements.
    """
    
    def __init__(self, reference_image_path: Optional[str] = None):
        """
        Initialize the reference validator.
        
        Args:
            reference_image_path: Optional path to reference snooker table image
        """
        self.reference_image: Optional[NDArray[np.uint8]] = None
        if reference_image_path:
            self.load_reference_image(reference_image_path)
    
    def load_reference_image(self, image_path: str) -> bool:
        """
        Load reference snooker table image.
        
        Args:
            image_path: Path to reference image
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            loaded_image = cv2.imread(image_path)
            if loaded_image is not None:
                self.reference_image = loaded_image.astype(np.uint8)
            if self.reference_image is None:
                logger.error(f"Could not load reference image from {image_path}")
                return False
            
            logger.info(f"Reference image loaded: {self.reference_image.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading reference image: {str(e)}")
            return False
    
    def validate_table_dimensions(self, transformed_image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        Validate that the transformed image has correct table dimensions.
        
        Args:
            transformed_image: Perspective-transformed snooker table image (full or cropped)
            
        Returns:
            Dictionary with validation metrics
        """
        height, width = transformed_image.shape[:2]
        
        # All transformed images should now be table area dimensions
        expected_width = TABLE_WIDTH
        expected_height = TABLE_HEIGHT
        
        # Check if dimensions match expected values
        width_ratio = width / expected_width
        height_ratio = height / expected_height
        
        # Calculate aspect ratio
        actual_aspect = width / height
        expected_aspect = expected_width / expected_height
        aspect_error = abs(actual_aspect - expected_aspect) / expected_aspect
        
        validation_results = {
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'aspect_ratio_error': aspect_error,
            'expected_dimensions': (expected_width, expected_height),
            'actual_dimensions': (width, height),
            'dimensions_valid': (0.95 <= width_ratio <= 1.05 and 
                               0.95 <= height_ratio <= 1.05 and 
                               aspect_error < 0.05)
        }
        
        logger.info(f"Dimension validation: {validation_results}")
        return validation_results
    
    def detect_table_lines(self, image: NDArray[np.uint8]) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect horizontal and vertical lines in the table image.
        
        Args:
            image: Input table image
            
        Returns:
            Dictionary with detected lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                # Extract coordinates from line array
                line_array = np.asarray(line)
                coords = line_array[0] if line_array.ndim > 1 else line_array
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                
                # Calculate line angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Classify as horizontal or vertical
                if abs(angle) < 10 or abs(angle) > 170:  # Horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
                elif 80 < abs(angle) < 100:  # Vertical
                    vertical_lines.append((x1, y1, x2, y2))
        
        return {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines
        }
    
    def validate_baulk_line_position(self, transformed_image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        Validate the position of the baulk line in the transformed image.
        
        Args:
            transformed_image: Perspective-transformed table image
            
        Returns:
            Dictionary with baulk line validation results
        """
        lines = self.detect_table_lines(transformed_image)
        horizontal_lines = lines['horizontal_lines']
        
        # Expected baulk line position
        expected_y = BAULK_LINE_Y
        
        # Find the horizontal line closest to expected baulk line position
        best_match = None
        min_distance = float('inf')
        
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            line_y = (y1 + y2) / 2
            
            # Check if line spans most of the table width
            line_length = abs(x2 - x1)
            if line_length > TABLE_WIDTH * 0.7:  # At least 70% of table width
                distance = abs(line_y - expected_y)
                if distance < min_distance:
                    min_distance = distance
                    best_match = line
        
        validation_results = {
            'baulk_line_found': best_match is not None,
            'position_error': min_distance if best_match else float('inf'),
            'position_valid': min_distance < 50 if best_match else False
        }
        
        if best_match:
            x1, y1, x2, y2 = best_match
            validation_results['detected_position'] = (y1 + y2) / 2
            validation_results['expected_position'] = expected_y
        
        logger.info(f"Baulk line validation: {validation_results}")
        return validation_results
    
    def validate_table_color(self, transformed_image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        Validate that the table has the expected green color.
        
        Args:
            transformed_image: Perspective-transformed table image (full or cropped)
            
        Returns:
            Dictionary with color validation results
        """
        # All transformed images are now table area only
        playing_area = transformed_image
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(playing_area, cv2.COLOR_BGR2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green pixels
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        total_pixels = playing_area.shape[0] * playing_area.shape[1]
        green_pixels = np.sum(green_mask > 0)
        green_percentage = green_pixels / total_pixels
        
        # Calculate average color in the playing area
        mean_color = np.mean(playing_area.reshape(-1, 3), axis=0)
        
        validation_results = {
            'green_percentage': float(green_percentage),
            'mean_bgr_color': [float(x) for x in mean_color],
            'color_valid': bool(green_percentage > 0.6)  # At least 60% green
        }
        
        logger.info(f"Color validation: {validation_results}")
        return validation_results
    
    def create_validation_report(self, transformed_image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        Create a comprehensive validation report for the transformed image.
        
        Args:
            transformed_image: Perspective-transformed table image
            
        Returns:
            Dictionary with complete validation results
        """
        report = {
            'timestamp': cv2.getTickCount(),
            'image_shape': transformed_image.shape,
        }
        
        # Run all validations
        report['dimensions'] = self.validate_table_dimensions(transformed_image)
        report['baulk_line'] = self.validate_baulk_line_position(transformed_image)
        report['color'] = self.validate_table_color(transformed_image)
        
        # Calculate overall validation score
        dimensions_result = report['dimensions']
        baulk_result = report['baulk_line']
        color_result = report['color']
        
        if isinstance(dimensions_result, dict) and isinstance(baulk_result, dict) and isinstance(color_result, dict):
            dim_valid = bool(dimensions_result.get('dimensions_valid', False))
            baulk_valid = bool(baulk_result.get('position_valid', False))
            color_valid = bool(color_result.get('color_valid', False))
            
            validations = [dim_valid, baulk_valid, color_valid]
            
            overall_score = float(sum(validations)) / len(validations)
            report['overall_score'] = overall_score
            report['validation_passed'] = bool(overall_score >= (2/3))  # At least 2/3 tests pass
        else:
            report['overall_score'] = 0.0
            report['validation_passed'] = False
        
        logger.info(f"Validation report: Overall score {report['overall_score']:.2f}")
        return report
    
    def visualize_validation(self, 
                           transformed_image: NDArray[np.uint8], 
                           validation_report: Dict[str, Any],
                           output_path: Optional[str] = None) -> NDArray[np.uint8]:
        """
        Create a visualization of the validation results.
        
        Args:
            transformed_image: Transformed table image
            validation_report: Results from create_validation_report
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = transformed_image.copy()
        
        # Draw expected table boundaries
        cv2.rectangle(vis_image, (TABLE_LEFT, TABLE_TOP), (TABLE_RIGHT, TABLE_BOTTOM), 
                     (0, 255, 255), 3)  # Yellow rectangle
        
        # Draw expected baulk line
        cv2.line(vis_image, (TABLE_LEFT, BAULK_LINE_Y), (TABLE_RIGHT, BAULK_LINE_Y), 
                (0, 255, 255), 2)  # Yellow line
        
        # Draw middle line
        cv2.line(vis_image, (MIDDLE_LINE_X, TABLE_TOP), (MIDDLE_LINE_X, TABLE_BOTTOM), 
                (0, 255, 255), 1)  # Yellow line
        
        # Draw ball spots
        for ball_color, position in STANDARD_BALL_POSITIONS.items():
            cv2.circle(vis_image, (position["x"], position["y"]), 10, (0, 255, 255), 2)
        
        # Add validation text
        y_offset = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0) if validation_report['validation_passed'] else (0, 0, 255)
        
        # Overall result
        status = "PASSED" if validation_report['validation_passed'] else "FAILED"
        cv2.putText(vis_image, f"Validation: {status}", (20, y_offset), 
                   font, font_scale, color, 2)
        
        # Individual results
        y_offset += 40
        cv2.putText(vis_image, f"Dimensions: {'OK' if validation_report['dimensions']['dimensions_valid'] else 'FAIL'}", 
                   (20, y_offset), font, 0.6, color, 1)
        
        y_offset += 30
        cv2.putText(vis_image, f"Baulk Line: {'OK' if validation_report['baulk_line']['position_valid'] else 'FAIL'}", 
                   (20, y_offset), font, 0.6, color, 1)
        
        y_offset += 30
        cv2.putText(vis_image, f"Color: {'OK' if validation_report['color']['color_valid'] else 'FAIL'}", 
                   (20, y_offset), font, 0.6, color, 1)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Validation visualization saved to: {output_path}")
        
        return vis_image


def validate_transformation(transformed_image: NDArray[np.uint8], 
                          reference_image_path: Optional[str] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to validate a perspective transformation result.
    
    Args:
        transformed_image: Perspective-transformed table image
        reference_image_path: Optional path to reference image
        output_dir: Optional directory to save validation outputs
        
    Returns:
        Validation report dictionary
    """
    validator = ReferenceValidator(reference_image_path)
    report = validator.create_validation_report(transformed_image)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save validation visualization
        vis_path = os.path.join(output_dir, "validation_visualization.png")
        validator.visualize_validation(transformed_image, report, vis_path)
        
        # Save validation report as JSON
        import json
        report_path = os.path.join(output_dir, "validation_report.json")
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_report = _convert_numpy_types(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Validation outputs saved to: {output_dir}")
    
    return report