"""
Snooker table constants and dimensions.

This module defines the standard snooker table dimensions and ball positions
based on actual image analysis for accurate 2D reconstruction.
"""

from typing import Dict, Tuple, List

# Snooker table dimensions based on actual image analysis (in pixels)
# Origin (0, 0) is at top-left corner of the image
IMAGE_WIDTH = 1830
IMAGE_HEIGHT = 3660

# Playing area dimensions (green surface inside cushions)
TABLE_WIDTH = 1620
TABLE_HEIGHT = 3450
TABLE_LEFT = 105
TABLE_TOP = 105
TABLE_RIGHT = 1725
TABLE_BOTTOM = 3555

# Ball size for rendering
BALL_SIZE = 50

# Key line positions
BAULK_LINE_Y = 824
MIDDLE_LINE_Y = 1826
MIDDLE_LINE_X = 912

# Standard ball starting positions for snooker (based on actual image coordinates)
YELLOW_SPOT = {"x": 646, "y": BAULK_LINE_Y}
GREEN_SPOT = {"x": 1178, "y": BAULK_LINE_Y}
BROWN_SPOT = {"x": MIDDLE_LINE_X, "y": BAULK_LINE_Y}
BLUE_SPOT = {"x": MIDDLE_LINE_X, "y": MIDDLE_LINE_Y}
PINK_SPOT = {"x": MIDDLE_LINE_X, "y": 2680}
BLACK_SPOT = {"x": MIDDLE_LINE_X, "y": 3238}

# Pocket dimensions (rectangular regions)
CORNER_POCKET_WIDTH = 85
CORNER_POCKET_HEIGHT = 85
MIDDLE_POCKET_WIDTH = 75
MIDDLE_POCKET_HEIGHT = 75

# Pocket positions (center coordinates)
POCKETS = {
    "top_left": {"x": TABLE_LEFT, "y": TABLE_TOP},
    "top_right": {"x": TABLE_RIGHT, "y": TABLE_TOP},
    "middle_left": {"x": TABLE_LEFT, "y": MIDDLE_LINE_Y},
    "middle_right": {"x": TABLE_RIGHT, "y": MIDDLE_LINE_Y},
    "bottom_left": {"x": TABLE_LEFT, "y": TABLE_BOTTOM},
    "bottom_right": {"x": TABLE_RIGHT, "y": TABLE_BOTTOM},
}

# Ball colors and their point values
BALL_COLORS = {
    "red": {"value": 1, "color": (0, 0, 255)},      # Red in BGR
    "yellow": {"value": 2, "color": (0, 255, 255)}, # Yellow in BGR
    "green": {"value": 3, "color": (0, 255, 0)},    # Green in BGR
    "brown": {"value": 4, "color": (42, 42, 165)},  # Brown in BGR
    "blue": {"value": 5, "color": (255, 0, 0)},     # Blue in BGR
    "pink": {"value": 6, "color": (203, 192, 255)}, # Pink in BGR
    "black": {"value": 7, "color": (0, 0, 0)},      # Black in BGR
    "white": {"value": 0, "color": (255, 255, 255)} # White (cue ball) in BGR
}

# Standard ball positions for game start
STANDARD_BALL_POSITIONS = {
    "yellow": YELLOW_SPOT,
    "green": GREEN_SPOT,
    "brown": BROWN_SPOT,
    "blue": BLUE_SPOT,
    "pink": PINK_SPOT,
    "black": BLACK_SPOT,
}

# Red ball triangle formation (15 balls)
# Starting from the pink spot and forming a triangle towards the black spot
RED_TRIANGLE_START_Y = PINK_SPOT["y"] + 52  # Start below pink spot
RED_BALL_SPACING = 52  # Spacing between red balls

def get_red_ball_positions() -> List[Dict[str, int]]:
    """
    Generate the standard red ball triangle formation positions.
    
    Returns:
        List of dictionaries with x, y coordinates for 15 red balls
    """
    positions = []
    
    # Triangle formation: 5 rows with 1, 2, 3, 4, 5 balls respectively
    row_balls = [1,2,3,4,5]  # Bottom to top
    
    for row_idx, num_balls in enumerate(row_balls):
        y = RED_TRIANGLE_START_Y + (row_idx * RED_BALL_SPACING)
        
        # Center the balls in each row
        start_x = MIDDLE_LINE_X - ((num_balls - 1) * RED_BALL_SPACING) // 2
        
        for ball_idx in range(num_balls):
            x = start_x + (ball_idx * RED_BALL_SPACING)
            positions.append({"x": x, "y": y})
    
    return positions