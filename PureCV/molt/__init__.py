"""
MOLT (Multiple Object Local Tracker) package.

This package implements the MOLT algorithm for robust tracking of multiple small,
similar objects in low-quality, low-frame-rate video. MOLT uses a population-based
approach where each object is tracked by multiple local trackers that combine
appearance features with motion constraints.
"""

from .tracker import MOLTTracker
from .config import MOLTTrackerConfig
from .types import Detection, Track, Frame

__version__ = "1.0.0"
__all__ = ['MOLTTracker', 'MOLTTrackerConfig', 'Detection', 'Track', 'Frame']