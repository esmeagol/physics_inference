"""
Snooker table base frame generator.

This module creates a sophisticated 2D representation of a snooker table
that serves as the base for ball position visualization and perspective
transformation reference.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from numpy.typing import NDArray
import os
import sys

# Add the src directory to the path for imports when running as script
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from . import table_constants
except ImportError:
    # Fallback for when running as script
    import table_constants as table_constants_fallback
    table_constants = table_constants_fallback



class SnookerTableGenerator:
    """
    Generates a sophisticated 2D snooker table base frame for visualization
    and perspective transformation reference.
    """
    
    def __init__(self) -> None:
        """Initialize the table generator."""
        self.base_color = (255, 255, 255)  # White base color
        self.cushion_color = (34, 139, 34)  # Forest green in BGR
        self.table_color = (139, 69, 19)  # Saddle brown in BGR
        self.line_color = (255, 255, 255)  # White lines
        self.pocket_color = (0, 0, 0)  # Black pockets
        self.spot_color = (255, 255, 255)  # White spots
    
        
    def load_base_table_image(self) -> NDArray[np.uint8]:
        """
        Load the base snooker table image from file.
        
        Returns:
            numpy array representing the table image
        """
        import cv2
        
        # Load the base table image
        image_path = "/Users/abhinavrai/Playground/physics_inference/src/snooker/table/base_table_image.png"
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Base table image not found at: {image_path}")
        
        # Load and return the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from: {image_path}")
        
        # Ensure the image is uint8
        return img.astype(np.uint8)
    
    def create_table_with_balls(self, 
                              ball_positions: Optional[Dict[str, Dict[str, int]]] = None,
                              show_standard_positions: bool = True) -> NDArray[np.uint8]:
        """
        Create snooker table with balls positioned.
        
        Args:
            ball_positions: Dictionary mapping ball colors to positions
            show_standard_positions: Whether to show standard starting positions
            
        Returns:
            numpy array representing the table with balls
        """
        img = self.load_base_table_image()
        
        if show_standard_positions and ball_positions is None:
            # Show standard starting positions
            ball_positions = table_constants.STANDARD_BALL_POSITIONS.copy()
            
            # Add red balls in triangle formation
            red_positions = table_constants.get_red_ball_positions()
            for i, pos in enumerate(red_positions):
                ball_positions[f"red_{i+1}"] = pos
        
        if ball_positions:
            self._draw_balls(img, ball_positions)
        
        return img
    
    
    def _draw_balls(self, img: NDArray[np.uint8], ball_positions: Dict[str, Dict[str, int]]) -> None:
        """
        Draw balls on the table at specified positions.
        
        Args:
            img: Table image to draw on
            ball_positions: Dictionary mapping ball identifiers to positions
        """
        ball_radius = table_constants.BALL_SIZE // 2
        
        for ball_id, position in ball_positions.items():
            # Determine ball color
            if ball_id.startswith("red"):
                ball_color = "red"
            else:
                ball_color = ball_id
            
            if ball_color not in table_constants.BALL_COLORS:
                continue
            
            color_info = table_constants.BALL_COLORS[ball_color]
            ball_bgr = color_info["color"]
            
            
            
            # Draw ball shadow (slightly offset)
            shadow_offset = 3
            cv2.circle(
                img,
                (position["x"] + shadow_offset, position["y"] + shadow_offset),
                ball_radius,
                (50, 50, 50),  # Dark gray shadow
                -1
            )
            
            # Draw main ball
            if isinstance(ball_bgr, (list, tuple)):
                ball_color_tuple = tuple(int(c) for c in ball_bgr)
            else:
                ball_color_tuple = (255, 255, 255)  # Default white
            cv2.circle(
                img,
                (position["x"], position["y"]),
                ball_radius,
                ball_color_tuple,
                -1
            )
            
            # Add ball highlight (for 3D effect)
            highlight_offset = ball_radius // 3
            cv2.circle(
                img,
                (position["x"] - highlight_offset, position["y"] - highlight_offset),
                ball_radius // 4,
                (255, 255, 255),
                -1
            )
            
            # Add ball border
            cv2.circle(
                img,
                (position["x"], position["y"]),
                ball_radius,
                (0, 0, 0),
                2
            )
    
    def get_table_corners(self) -> List[Tuple[int, int]]:
        """
        Get the four corner coordinates of the playing area.
        
        Returns:
            List of (x, y) tuples for table corners in clockwise order (table-relative coordinates)
        """
        return [
            (table_constants.PLAY_AREA_TOP_LEFT_X, table_constants.PLAY_AREA_TOP_LEFT_Y),                       # Top-left
            (table_constants.PLAY_AREA_TOP_RIGHT_X, table_constants.PLAY_AREA_TOP_RIGHT_Y),             # Top-right
            (table_constants.PLAY_AREA_BOTTOM_RIGHT_X, table_constants.PLAY_AREA_BOTTOM_RIGHT_Y),  # Bottom-right
            (table_constants.PLAY_AREA_BOTTOM_LEFT_X, table_constants.PLAY_AREA_BOTTOM_LEFT_Y)             # Bottom-left
        ]
    
    
    def save_table_image(self, 
                        filename: str, 
                        with_balls: bool = True,
                        ball_positions: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """
        Save the table image to file.
        
        Args:
            filename: Output filename
            with_balls: Whether to include balls
            ball_positions: Custom ball positions (if None, uses standard)
        """
        if with_balls:
            img = self.create_table_with_balls(ball_positions)
        else:
            img = self.load_base_table_image()
        
        cv2.imwrite(filename, img)
        print(f"Table image saved to: {filename}")


def create_reference_table(output_path: Optional[str] = None) -> NDArray[np.uint8]:
    """
    Create a reference snooker table image for testing and visualization.
    
    Args:
        output_path: Optional path to save the image
        
    Returns:
        numpy array of the table image
    """
    generator = SnookerTableGenerator()
    table_img = generator.create_table_with_balls()
    
    if output_path:
        generator.save_table_image(output_path)
    
    return table_img


if __name__ == "__main__":
    # Create and save reference table
    generator = SnookerTableGenerator()
        
    # Create table with standard ball positions
    table_with_balls = generator.create_table_with_balls()
    generator.save_table_image("/Users/abhinavrai/Playground/physics_inference/src/snooker/table/snooker_table_standard.png", with_balls=True)
    
    print("Reference table images created successfully!")
    print(f"Table dimensions: {table_constants.TABLE_WIDTH}x{table_constants.TABLE_HEIGHT}")