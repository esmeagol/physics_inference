"""
PureCV - Traditional Computer Vision for Cue Sports Analysis

This module contains implementations of traditional computer vision techniques
for analyzing cue sports, focusing on reliability and interpretability.
"""

__version__ = "0.1.0"

from .table_detection import (
    detect_green_table,
    order_points,
    perspective_transform
)

from .molt_tracker import (
    MOLTTracker,
    MOLTTrackerConfig
)

__all__ = [
    'detect_green_table',
    'order_points',
    'perspective_transform',
    'MOLTTracker',
    'MOLTTrackerConfig'
]
