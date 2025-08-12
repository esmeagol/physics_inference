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
    IMAGE_WIDTH, IMAGE_HEIGHT,
    PLAY_AREA_TOP_LEFT_X, PLAY_AREA_TOP_LEFT_Y, 
    PLAY_AREA_BOTTOM_RIGHT_X, PLAY_AREA_BOTTOM_RIGHT_Y,
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
        Validate that the transformed image has correct playing area dimensions.
        
        Args:
            transformed_image: Perspective-transformed playing area image
            
        Returns:
            Dictionary with validation metrics
        """
        height, width = transformed_image.shape[:2]
        
        # Transformed images should be playing area dimensions
        expected_width = IMAGE_WIDTH
        expected_height = IMAGE_HEIGHT
        
        validation_results = {
            'expected_dimensions': (expected_width, expected_height),
            'actual_dimensions': (width, height),
            'dimensions_valid': (width == expected_width and height == expected_height)
        }
        
        logger.info(f"Dimension validation: {validation_results}")
        return validation_results
    
    def validate_table_color(self, transformed_image: NDArray[np.uint8]) -> Dict[str, Any]:
        """
        Validate that the playing area has the expected green color.
        
        Args:
            transformed_image: Perspective-transformed playing area image
            
        Returns:
            Dictionary with color validation results
        """
        # Transformed images are now playing area only
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
        Create a comprehensive validation report for the transformed playing area image.
        
        Args:
            transformed_image: Perspective-transformed playing area image
            
        Returns:
            Dictionary with complete validation results
        """
        report = {
            'timestamp': cv2.getTickCount(),
            'image_shape': transformed_image.shape,
        }
        
        # Run all validations
        report['dimensions'] = self.validate_table_dimensions(transformed_image)
        report['color'] = self.validate_table_color(transformed_image)
        
        # Calculate overall validation score
        dimensions_result = report['dimensions']
        color_result = report['color']
        
        if isinstance(dimensions_result, dict) and isinstance(color_result, dict):
            dim_valid = bool(dimensions_result.get('dimensions_valid', False))
            color_valid = bool(color_result.get('color_valid', False))
            
            validations = [dim_valid, color_valid]
            
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
            transformed_image: Transformed playing area image
            validation_report: Results from create_validation_report
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = transformed_image.copy()
        
        # Draw expected playing area boundaries (should be full image since it's already cropped)
        cv2.rectangle(vis_image, (PLAY_AREA_TOP_LEFT_X, PLAY_AREA_TOP_LEFT_Y), (PLAY_AREA_BOTTOM_RIGHT_X, PLAY_AREA_BOTTOM_RIGHT_Y), 
                     (0, 255, 255), 3)  # Yellow rectangle
        
        # Draw expected baulk line (relative to playing area)
        cv2.line(vis_image, (PLAY_AREA_TOP_LEFT_X, BAULK_LINE_Y), (PLAY_AREA_BOTTOM_RIGHT_X, BAULK_LINE_Y), 
                (0, 255, 255), 2)  # Yellow line
        
        # Draw middle line (relative to playing area)
        cv2.line(vis_image, (MIDDLE_LINE_X, PLAY_AREA_TOP_LEFT_Y), (MIDDLE_LINE_X, PLAY_AREA_BOTTOM_RIGHT_Y), 
                (0, 255, 255), 1)  # Yellow line
        
        # Draw ball spots (relative to playing area)
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
        transformed_image: Perspective-transformed playing area image
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