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

from .table_constants import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    TABLE_WIDTH, TABLE_HEIGHT,
    TABLE_LEFT, TABLE_TOP, TABLE_RIGHT, TABLE_BOTTOM,
    BAULK_LINE_Y, MIDDLE_LINE_Y, MIDDLE_LINE_X,
    POCKETS, CORNER_POCKET_WIDTH, CORNER_POCKET_HEIGHT,
    MIDDLE_POCKET_WIDTH, MIDDLE_POCKET_HEIGHT,
    STANDARD_BALL_POSITIONS, get_red_ball_positions,
    BALL_COLORS, BALL_SIZE, BROWN_SPOT, YELLOW_SPOT
)


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
        
    def create_base_table(self) -> NDArray[np.uint8]:
        """
        Create the base snooker table image without balls.
        
        Returns:
            numpy array representing the table image
        """
        # Create base image with table dimensions
        img = np.full((TABLE_HEIGHT, TABLE_WIDTH, 3), self.base_color, dtype=np.uint8)
        

        # Fill entire image with table color (since we're creating table-sized image)
        img[:] = self.table_color

        # Draw table lines
        self._draw_table_lines(img)
        
        # Draw pockets
        self._draw_pockets(img)
        
        # Draw ball spots
        self._draw_ball_spots(img)
        
        return img
    
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
        img = self.create_base_table()
        
        if show_standard_positions and ball_positions is None:
            # Show standard starting positions
            ball_positions = STANDARD_BALL_POSITIONS.copy()
            
            # Add red balls in triangle formation
            red_positions = get_red_ball_positions()
            for i, pos in enumerate(red_positions):
                ball_positions[f"red_{i+1}"] = pos
        
        if ball_positions:
            self._draw_balls(img, ball_positions)
        
        return img
    
    def _draw_table_lines(self, img: NDArray[np.uint8]) -> None:
        """Draw the standard snooker table lines."""
        line_thickness = 3
        
        # Convert to table-relative coordinates
        baulk_y_relative = BAULK_LINE_Y - TABLE_TOP
        cv2.line(
            img,
            (0, baulk_y_relative),
            (TABLE_WIDTH, baulk_y_relative),
            self.line_color,
            line_thickness
        )
        
        # Baulk semicircle (D-shaped area)
        # Center at brown spot, radius = distance between yellow and brown (or brown and green)
        # Convert to table-relative coordinates
        brown_x_relative = BROWN_SPOT["x"] - TABLE_LEFT
        brown_y_relative = BROWN_SPOT["y"] - TABLE_TOP
        yellow_x_relative = YELLOW_SPOT["x"] - TABLE_LEFT
        baulk_radius = abs(brown_x_relative - yellow_x_relative)  # Distance between brown and yellow spots
        
        cv2.ellipse(
            img,
            (brown_x_relative, brown_y_relative),
            (baulk_radius, baulk_radius),
            0, 180, 360,  # Draw lower semicircle (towards the bottom of table)
            self.line_color,
            line_thickness
        )
        
        # Center line (optional, for reference)
        middle_x_relative = MIDDLE_LINE_X - TABLE_LEFT
        cv2.line(
            img,
            (middle_x_relative, 0),
            (middle_x_relative, TABLE_HEIGHT),
            self.line_color,
            1  # Thinner line
        )
    
    def _draw_pockets(self, img: NDArray[np.uint8]) -> None:
        """Draw the six pockets on the table."""
        for pocket_name, position in POCKETS.items():
            if "middle" in pocket_name:
                width, height = MIDDLE_POCKET_WIDTH, MIDDLE_POCKET_HEIGHT
            else:
                width, height = CORNER_POCKET_WIDTH, CORNER_POCKET_HEIGHT
            
            # Convert to table-relative coordinates
            pocket_x = position["x"] - TABLE_LEFT
            pocket_y = position["y"] - TABLE_TOP
            
            # Draw pocket as filled ellipse
            cv2.ellipse(
                img,
                (pocket_x, pocket_y),
                (width // 2, height // 2),
                0, 0, 360,
                self.pocket_color,
                -1
            )
            
            # Add pocket border
            cv2.ellipse(
                img,
                (pocket_x, pocket_y),
                (width // 2, height // 2),
                0, 0, 360,
                (100, 100, 100),  # Dark gray border
                2
            )
    
    def _draw_ball_spots(self, img: NDArray[np.uint8]) -> None:
        """Draw the standard ball spots on the table."""
        spot_radius = 8
        
        for ball_color, position in STANDARD_BALL_POSITIONS.items():
            # Convert to table-relative coordinates
            spot_x = position["x"] - TABLE_LEFT
            spot_y = position["y"] - TABLE_TOP
            
            cv2.circle(
                img,
                (spot_x, spot_y),
                spot_radius,
                self.spot_color,
                -1
            )
            
            # Add small colored indicator
            color_bgr = BALL_COLORS[ball_color]["color"]
            if isinstance(color_bgr, (list, tuple)):
                color_tuple = tuple(int(c) for c in color_bgr)
            else:
                color_tuple = (255, 255, 255)  # Default white
            cv2.circle(
                img,
                (spot_x, spot_y),
                spot_radius - 2,
                color_tuple,
                2
            )
    
    def _draw_balls(self, img: NDArray[np.uint8], ball_positions: Dict[str, Dict[str, int]]) -> None:
        """
        Draw balls on the table at specified positions.
        
        Args:
            img: Table image to draw on
            ball_positions: Dictionary mapping ball identifiers to positions
        """
        ball_radius = BALL_SIZE // 2
        
        for ball_id, position in ball_positions.items():
            # Determine ball color
            if ball_id.startswith("red"):
                ball_color = "red"
            else:
                ball_color = ball_id
            
            if ball_color not in BALL_COLORS:
                continue
            
            color_info = BALL_COLORS[ball_color]
            ball_bgr = color_info["color"]
            
            # Convert to table-relative coordinates
            ball_x = position["x"] - TABLE_LEFT if "x" in position else position["x"]
            ball_y = position["y"] - TABLE_TOP if "y" in position else position["y"]
            
            # Handle case where position might already be relative
            if ball_x < 0 or ball_y < 0:
                ball_x = position["x"]
                ball_y = position["y"]
            
            # Draw ball shadow (slightly offset)
            shadow_offset = 3
            cv2.circle(
                img,
                (ball_x + shadow_offset, ball_y + shadow_offset),
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
                (ball_x, ball_y),
                ball_radius,
                ball_color_tuple,
                -1
            )
            
            # Add ball highlight (for 3D effect)
            highlight_offset = ball_radius // 3
            cv2.circle(
                img,
                (ball_x - highlight_offset, ball_y - highlight_offset),
                ball_radius // 4,
                (255, 255, 255),
                -1
            )
            
            # Add ball border
            cv2.circle(
                img,
                (ball_x, ball_y),
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
            (0, 0),                       # Top-left
            (TABLE_WIDTH, 0),             # Top-right
            (TABLE_WIDTH, TABLE_HEIGHT),  # Bottom-right
            (0, TABLE_HEIGHT)             # Bottom-left
        ]
    
    def get_pocket_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Get pocket regions as bounding boxes for event detection.
        
        Returns:
            Dictionary mapping pocket names to (x, y, width, height) tuples (table-relative coordinates)
        """
        regions = {}
        
        for pocket_name, position in POCKETS.items():
            if "middle" in pocket_name:
                width, height = MIDDLE_POCKET_WIDTH, MIDDLE_POCKET_HEIGHT
            else:
                width, height = CORNER_POCKET_WIDTH, CORNER_POCKET_HEIGHT
            
            # Convert to table-relative coordinates and then to bounding box
            pocket_x = position["x"] - TABLE_LEFT
            pocket_y = position["y"] - TABLE_TOP
            x = pocket_x - width // 2
            y = pocket_y - height // 2
            
            regions[pocket_name] = (x, y, width, height)
        
        return regions
    
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
            img = self.create_base_table()
        
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
    
    # Create table without balls
    base_table = generator.create_base_table()
    generator.save_table_image("snooker_table_base.png", with_balls=False)
    
    # Create table with standard ball positions
    table_with_balls = generator.create_table_with_balls()
    generator.save_table_image("snooker_table_standard.png", with_balls=True)
    
    print("Reference table images created successfully!")
    print(f"Table dimensions: {TABLE_WIDTH}x{TABLE_HEIGHT}")