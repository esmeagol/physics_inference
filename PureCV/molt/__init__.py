"""
MOLT (Multiple Object Local Tracker) Package

This package implements the MOLT algorithm for robust tracking of multiple small,
similar objects in low-quality, low-frame-rate video. MOLT is specifically designed
for challenging tracking scenarios such as cue sports (snooker, pool, billiards)
where traditional trackers struggle due to:

- Multiple similar-looking objects (balls of similar size and color)
- Low frame rates and motion blur
- Occlusions and rapid movements
- Poor lighting conditions
- Small object sizes relative to frame resolution

Key Features:
- Population-based tracking with configurable tracker counts per object
- Appearance modeling using color histograms with multiple color spaces
- Motion constraints with spatial consistency checking
- Ball counting and verification logic for game-specific rules
- Comprehensive visualization with trajectory trails and statistics
- Modular architecture for easy customization and extension

Algorithm Overview:
MOLT uses a population-based approach where each object is tracked by multiple
local trackers (typically 100-2000 per object). Each tracker maintains:
- Position hypothesis (x, y coordinates)
- Size information (width, height)
- Appearance model (color histogram)
- Similarity weights (histogram + spatial)

The algorithm works by:
1. Initializing populations around detected objects
2. Updating all trackers with new frame data
3. Computing appearance and spatial similarities
4. Ranking trackers by combined similarity scores
5. Selecting best tracker as object position
6. Regenerating populations around top performers
7. Verifying ball counts and handling violations

Quick Start:
    >>> from PureCV.molt import MOLTTracker
    >>> 
    >>> # Create tracker with default configuration
    >>> tracker = MOLTTracker()
    >>> 
    >>> # Initialize with first frame and detections
    >>> success = tracker.init(first_frame, initial_detections)
    >>> 
    >>> # Process video frames
    >>> for frame in video_frames:
    ...     tracks = tracker.update(frame)
    ...     vis_frame = tracker.visualize(frame, tracks)

Configuration:
    >>> from PureCV.molt import MOLTTrackerConfig
    >>> 
    >>> # Game-specific configurations
    >>> snooker_config = MOLTTrackerConfig.create_for_snooker()
    >>> pool_config = MOLTTrackerConfig.create_for_pool()
    >>> 
    >>> # Custom configuration
    >>> config = MOLTTrackerConfig()
    >>> config.population_sizes = {'red': 300, 'white': 1500}
    >>> config.histogram_bins = 20
    >>> tracker = MOLTTracker(config=config)

For detailed documentation, see:
- USAGE_GUIDE.md: Comprehensive usage instructions and examples
- BALL_COUNTING_GUIDE.md: Ball counting logic and snooker-specific features
- TROUBLESHOOTING_GUIDE.md: Common issues and performance optimization

Classes:
    MOLTTracker: Main tracker class implementing the MOLT algorithm
    MOLTTrackerConfig: Configuration management with validation and presets
    
Types:
    Detection: Type alias for detection dictionaries
    Track: Type alias for tracking result dictionaries
    Frame: Type alias for image frames (numpy arrays)
"""

from .tracker import MOLTTracker
from .config import MOLTTrackerConfig
from .types import Detection, Track, Frame

__version__ = "1.0.0"
__all__ = ['MOLTTracker', 'MOLTTrackerConfig', 'Detection', 'Track', 'Frame']