"""
PureCV - Traditional Computer Vision for Cue Sports Analysis

This module contains implementations of traditional computer vision techniques
for analyzing cue sports, focusing on reliability and interpretability.

Key Components:
- Table Detection: Functions for detecting and transforming pool/snooker tables
- MOLT Tracker: Multiple Object Local Tracker for robust ball tracking
- Configuration: Flexible configuration system for different game types

The MOLT (Multiple Object Local Tracker) is the main tracking algorithm,
designed specifically for tracking multiple small, similar objects like
balls in cue sports videos under challenging conditions.
"""

__version__ = "0.2.0"  # Updated with MOLT tracker implementation

from .table_detection import (
    detect_green_table,
    order_points,
    perspective_transform
)

from .molt import (
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
