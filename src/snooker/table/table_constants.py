"""
Snooker table constants and dimensions.

This module defines the standard snooker table dimensions and ball positions
based on actual image analysis for accurate 2D reconstruction.
"""

from typing import Dict, Tuple, List

# Origin (0, 0) is at top-left corner of the image
IMAGE_WIDTH = 2439
IMAGE_HEIGHT = 4500

# # Snooker table dimensions based on actual image analysis (in pixels)
TABLE_WIDTH = 2240
TABLE_HEIGHT = 4322
TABLE_TOP_LEFT_X = 90
TABLE_TOP_LEFT_Y = 78
TABLE_BOTTOM_RIGHT_X = TABLE_TOP_LEFT_X + TABLE_WIDTH  # 2330
TABLE_BOTTOM_RIGHT_Y = TABLE_TOP_LEFT_Y + TABLE_HEIGHT # 4400

# Additional table boundary constants for reference_validator.py
TABLE_LEFT = TABLE_TOP_LEFT_X
TABLE_TOP = TABLE_TOP_LEFT_Y
TABLE_RIGHT = TABLE_BOTTOM_RIGHT_X
TABLE_BOTTOM = TABLE_BOTTOM_RIGHT_Y

# Playing area dimensions (green surface inside cushions)
PLAY_AREA_WIDTH = 2022 # equivalent to 1778 mm on a standard table
PLAY_AREA_HEIGHT = 4056 # equivalent to 3569 mm on a standard table
PLAY_AREA_TOP_LEFT_X = 218
PLAY_AREA_TOP_LEFT_Y = 215
PLAY_AREA_BOTTOM_RIGHT_X = PLAY_AREA_TOP_LEFT_X + PLAY_AREA_WIDTH # 2240       
PLAY_AREA_BOTTOM_RIGHT_Y = PLAY_AREA_TOP_LEFT_Y + PLAY_AREA_HEIGHT # 4271
PLAY_AREA_TOP_RIGHT_X = PLAY_AREA_BOTTOM_RIGHT_X
PLAY_AREA_TOP_RIGHT_Y = PLAY_AREA_TOP_LEFT_Y
PLAY_AREA_BOTTOM_LEFT_X = PLAY_AREA_TOP_LEFT_X
PLAY_AREA_BOTTOM_LEFT_Y = PLAY_AREA_BOTTOM_RIGHT_Y

CONVERSION_FACTOR = PLAY_AREA_WIDTH / 1778

# Ball size for rendering
BALL_SIZE = 60 # 52.4 mm = 60 pixels

# Key line positions
BAULK_LINE_Y = 1053 # 29 inches = 736 mm = 838 pixels from the PLAY_AREA_TOP_LEFT_Y
LONG_MIDDLE_LINE_X = int(PLAY_AREA_TOP_LEFT_X + PLAY_AREA_WIDTH / 2)
SHORT_MIDDLE_LINE_Y = int(PLAY_AREA_TOP_LEFT_Y + PLAY_AREA_HEIGHT / 2) # 2243

# Additional middle line constants for reference_validator.py
MIDDLE_LINE_X = LONG_MIDDLE_LINE_X
MIDDLE_LINE_Y = SHORT_MIDDLE_LINE_Y

D_RADIUS = 330 # 290mm = 330 pixels

# Standard ball starting positions for snooker (based on actual image coordinates)
YELLOW_SPOT = {"x": int(LONG_MIDDLE_LINE_X - D_RADIUS), "y": BAULK_LINE_Y}
GREEN_SPOT = {"x": int(LONG_MIDDLE_LINE_X + D_RADIUS), "y": BAULK_LINE_Y}
BROWN_SPOT = {"x": LONG_MIDDLE_LINE_X, "y": BAULK_LINE_Y}
BLUE_SPOT = {"x": LONG_MIDDLE_LINE_X, "y": SHORT_MIDDLE_LINE_Y}
PINK_SPOT = {"x": LONG_MIDDLE_LINE_X, "y": int(SHORT_MIDDLE_LINE_Y + PLAY_AREA_HEIGHT / 4)} # 3257
BLACK_SPOT = {"x": LONG_MIDDLE_LINE_X, "y": 3904} # 324 mm = 367 pixels from PLAY_AREA_BOTTOM_RIGHT_Y

# Pocket dimensions (rectangular regions)
CORNER_POCKET_WIDTH = 95
MIDDLE_POCKET_WIDTH = 100


# Pocket positions (center coordinates)
POCKETS = {
    "top_left": {"x": PLAY_AREA_TOP_LEFT_X, "y": PLAY_AREA_TOP_LEFT_Y},
    "top_right": {"x": PLAY_AREA_TOP_RIGHT_X, "y": PLAY_AREA_TOP_RIGHT_Y},
    "middle_left": {"x": PLAY_AREA_TOP_LEFT_X, "y": SHORT_MIDDLE_LINE_Y},
    "middle_right": {"x": PLAY_AREA_TOP_RIGHT_X, "y": SHORT_MIDDLE_LINE_Y},
    "bottom_left": {"x": PLAY_AREA_BOTTOM_LEFT_X, "y": PLAY_AREA_BOTTOM_LEFT_Y},
    "bottom_right": {"x": PLAY_AREA_BOTTOM_RIGHT_X, "y": PLAY_AREA_BOTTOM_RIGHT_Y},
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
RED_TRIANGLE_START_Y = PINK_SPOT["y"] + BALL_SIZE  # Start below pink spot
RED_BALL_SPACING = BALL_SIZE+2  # Spacing between red balls

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
        y = int(RED_TRIANGLE_START_Y + (row_idx * BALL_SIZE))
        
        # Center the balls in each row
        start_x = int(LONG_MIDDLE_LINE_X - ((num_balls - 1) * RED_BALL_SPACING) / 2)
        
        for ball_idx in range(num_balls):
            x = int(start_x + (ball_idx * RED_BALL_SPACING))
            positions.append({"x": x, "y": y})
    
    return positions