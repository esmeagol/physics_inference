"""
Perspective transformation module for snooker table standardization.

This module provides functionality to transform video frames from arbitrary
camera angles to a standardized top-down view of the snooker table using
manual point selection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from numpy.typing import NDArray

from .table.table_constants import TABLE_LEFT, TABLE_TOP, TABLE_RIGHT, TABLE_BOTTOM, TABLE_WIDTH, TABLE_HEIGHT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointSelector:
    """Interactive point selector for manual table region definition."""
    
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []
        self.window_name = "Select Table Corners"
        self.image: Optional[NDArray[np.uint8]] = None
        self.original_image: Optional[NDArray[np.uint8]] = None
        self.display_image: Optional[NDArray[np.uint8]] = None
        self.scale_factor: float = 1.0
        
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Optional[object]) -> None:
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and self.display_image is not None:
            # Convert display coordinates back to original image coordinates
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)
            
            self.points.append((original_x, original_y))
            logger.info(f"Selected point {len(self.points)}: ({original_x}, {original_y}) [display: ({x}, {y}), scale: {self.scale_factor:.3f}]")
            
            # Draw point on display image (using display coordinates)
            cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                self.display_image, 
                f"{len(self.points)}", 
                (x + 15, y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 0), 
                2
            )
            
            # Draw lines between points (using display coordinates)
            if len(self.points) > 1:
                # Convert previous point to display coordinates for line drawing
                prev_display_x = int(self.points[-2][0] * self.scale_factor)
                prev_display_y = int(self.points[-2][1] * self.scale_factor)
                cv2.line(
                    self.display_image, 
                    (prev_display_x, prev_display_y), 
                    (x, y), 
                    (255, 0, 0), 
                    2
                )
            
            # Close the quadrilateral
            if len(self.points) == 4:
                # Convert first point to display coordinates for closing line
                first_display_x = int(self.points[0][0] * self.scale_factor)
                first_display_y = int(self.points[0][1] * self.scale_factor)
                cv2.line(
                    self.display_image, 
                    (x, y), 
                    (first_display_x, first_display_y), 
                    (255, 0, 0), 
                    2
                )
            
            cv2.imshow(self.window_name, self.display_image)
            
    def select_points(self, image: NDArray[np.uint8], num_points: int = 4) -> List[Tuple[int, int]]:
        """
        Select points interactively from an image.
        
        Args:
            image: Input image for point selection
            num_points: Number of points to select (default: 4 for table corners)
            
        Returns:
            List of selected points as (x, y) tuples in original image coordinates
        """
        self.points = []
        self.original_image = image.copy()
        
        # Create display image with appropriate scaling
        height, width = image.shape[:2]
        if width > 1200 or height > 800:
            self.scale_factor = min(1200/width, 800/height)
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.display_image = cv2.resize(image, (new_width, new_height)).astype(np.uint8)
        else:
            self.scale_factor = 1.0
            self.display_image = image.copy()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\nSelect {num_points} table corner points in clockwise order:")
        print("1. Top-left corner of the table")
        print("2. Top-right corner of the table") 
        print("3. Bottom-right corner of the table")
        print("4. Bottom-left corner of the table")
        print("Press 'r' to reset, 'q' to quit, ENTER to confirm selection")
        
        if self.display_image is not None:
            cv2.imshow(self.window_name, self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset points
                logger.info("Resetting point selection")
                self.points = []
                # Reset display image
                if self.scale_factor != 1.0 and self.original_image is not None:
                    self.display_image = cv2.resize(self.original_image, 
                                                  (int(width * self.scale_factor), 
                                                   int(height * self.scale_factor))).astype(np.uint8)
                elif self.original_image is not None:
                    self.display_image = self.original_image.copy()
                if self.display_image is not None:
                    cv2.imshow(self.window_name, self.display_image)
                print("Points reset. Select again.")
                
            elif key == ord('q'):  # Quit
                logger.info("Point selection cancelled")
                cv2.destroyAllWindows()
                return []
                
            elif key == 13 or key == 10:  # Enter key
                if len(self.points) == num_points:
                    logger.info(f"Point selection completed: {self.points}")
                    cv2.destroyAllWindows()
                    # Points are already in original image coordinates
                    return self.points
                else:
                    print(f"Please select exactly {num_points} points. Currently selected: {len(self.points)}")


class PerspectiveTransformer:
    """
    Handles perspective transformation of snooker table from arbitrary camera angles
    to standardized top-down view using manual point selection.
    """
    
    def __init__(self, transformation_points: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize the perspective transformer.
        
        Args:
            transformation_points: Optional pre-defined transformation points
        """
        self.transformation_matrix: Optional[NDArray[np.floating[Any]]] = None
        self.source_points: Optional[List[Tuple[int, int]]] = transformation_points
        self.target_points = self._get_target_points()
        self.point_selector = PointSelector()
        
        if transformation_points:
            self._calculate_transformation_matrix()
    
    def _get_target_points(self) -> List[Tuple[int, int]]:
        """
        Get the target points for the standardized table view.
        
        Returns:
            List of target points in clockwise order
        """
        return [
            (0, 0),                       # Top-left
            (TABLE_WIDTH, 0),             # Top-right  
            (TABLE_WIDTH, TABLE_HEIGHT),  # Bottom-right
            (0, TABLE_HEIGHT)             # Bottom-left
        ]
    
    def setup_transformation(self, frame: NDArray[np.uint8]) -> bool:
        """
        Setup transformation using manual point selection on the provided frame.
        
        Args:
            frame: Input frame for point selection
            
        Returns:
            True if transformation setup was successful, False otherwise
        """
        logger.info("Setting up perspective transformation with manual point selection")
        
        # Select source points manually
        source_points = self.point_selector.select_points(frame, num_points=4)
        
        if len(source_points) != 4:
            logger.error("Failed to select 4 corner points")
            return False
        
        self.source_points = source_points
        logger.info(f"Source points selected: {self.source_points}")
        
        # Calculate transformation matrix
        return self._calculate_transformation_matrix()
    
    def _calculate_transformation_matrix(self) -> bool:
        """
        Calculate the perspective transformation matrix from source to target points.
        
        Returns:
            True if calculation was successful, False otherwise
        """
        if not self.source_points or len(self.source_points) != 4:
            logger.error("Source points not properly defined")
            return False
        
        try:
            # Convert points to numpy arrays
            src_pts = np.array(self.source_points, dtype=np.float32)
            dst_pts = np.array(self.target_points, dtype=np.float32)
            
            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.transformation_matrix = matrix.astype(np.float64)
            
            logger.info("Perspective transformation matrix calculated successfully")
            logger.debug(f"Transformation matrix:\n{self.transformation_matrix}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to calculate transformation matrix: {str(e)}")
            return False
    
    def transform_frame(self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.uint8]]:
        """
        Transform frame to top-down table view using established transformation matrix.
        
        Args:
            frame: Input frame to transform
            
        Returns:
            Transformed frame cropped to table area or None if transformation fails
        """
        if self.transformation_matrix is None:
            logger.error("Transformation matrix not initialized")
            return None
        
        try:
            # Apply perspective transformation to table area size
            transformed = cv2.warpPerspective(
                frame, 
                self.transformation_matrix, 
                (TABLE_WIDTH, TABLE_HEIGHT)
            )
            
            # Return the transformed image (already at table dimensions)
            return transformed.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Failed to transform frame: {str(e)}")
            return None
    
    def get_transformation_matrix(self) -> Optional[NDArray[np.floating[Any]]]:
        """
        Get the current perspective transformation matrix.
        
        Returns:
            Transformation matrix or None if not initialized
        """
        return self.transformation_matrix
    
    def save_transformation_config(self, filepath: str) -> bool:
        """
        Save transformation configuration to file.
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            True if save was successful, False otherwise
        """
        if not self.source_points or self.transformation_matrix is None:
            logger.error("No transformation configuration to save")
            return False
        
        try:
            config = {
                'source_points': self.source_points,
                'target_points': self.target_points,
                'transformation_matrix': self.transformation_matrix.tolist()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Transformation configuration saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transformation configuration: {str(e)}")
            return False
    
    def load_transformation_config(self, filepath: str) -> bool:
        """
        Load transformation configuration from file.
        
        Args:
            filepath: Path to load configuration from
            
        Returns:
            True if load was successful, False otherwise
        """
        try:
            import json
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.source_points = [tuple(pt) for pt in config['source_points']]
            self.target_points = [tuple(pt) for pt in config['target_points']]
            self.transformation_matrix = np.array(config['transformation_matrix'])
            
            logger.info(f"Transformation configuration loaded from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load transformation configuration: {str(e)}")
            return False
    
    def validate_transformation(self, 
                              transformed_frame: NDArray[np.uint8],
                              reference_image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate the quality of the perspective transformation.
        
        Args:
            transformed_frame: The transformed frame to validate
            reference_image_path: Optional path to reference image
            
        Returns:
            Validation report dictionary
        """
        try:
            from .reference_validator import validate_transformation
            return validate_transformation(transformed_frame, reference_image_path)
        except ImportError:
            logger.warning("Reference validator not available")
            return {"validation_passed": False, "error": "Validator not available"}
    
    def visualize_transformation(self, 
                               original_frame: NDArray[np.uint8], 
                               output_path: Optional[str] = None) -> NDArray[np.uint8]:
        """
        Create a visualization showing the transformation result.
        
        Args:
            original_frame: Original input frame
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        if self.transformation_matrix is None:
            logger.error("No transformation matrix available for visualization")
            return original_frame
        
        # Transform the frame
        transformed = self.transform_frame(original_frame)
        if transformed is None:
            return original_frame
        
        # Create side-by-side visualization
        orig_height, orig_width = original_frame.shape[:2]
        trans_height, trans_width = transformed.shape[:2]
        
        # Resize images to same height for side-by-side display
        target_height = 600
        orig_scale = target_height / orig_height
        trans_scale = target_height / trans_height
        
        orig_resized = cv2.resize(original_frame, 
                                (int(orig_width * orig_scale), target_height))
        trans_resized = cv2.resize(transformed, 
                                 (int(trans_width * trans_scale), target_height))
        
        # Create combined image
        combined_width = orig_resized.shape[1] + trans_resized.shape[1] + 20
        combined = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
        
        # Place original image
        combined[:, :orig_resized.shape[1]] = orig_resized
        
        # Place transformed image
        start_x = orig_resized.shape[1] + 20
        combined[:, start_x:start_x + trans_resized.shape[1]] = trans_resized
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Transformed", (start_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, combined)
            logger.info(f"Transformation visualization saved to: {output_path}")
        
        return combined