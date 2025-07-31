# MOLT (Multiple Object Local Tracker)

A robust tracking algorithm specifically designed for tracking multiple small, similar objects in low-quality, low-frame-rate video, with specialized features for cue sports applications.

## Overview

MOLT uses a population-based approach where each object is tracked by multiple local trackers that combine appearance features (color histograms) with motion constraints. This approach provides exceptional robustness in challenging tracking scenarios where traditional trackers fail.

### Key Features

- **Population-Based Tracking**: Each object tracked by 100-2000 local trackers for robustness
- **Appearance Modeling**: Color histogram-based appearance features with multiple color spaces
- **Motion Constraints**: Spatial consistency checking with configurable exploration radii
- **Ball Counting Logic**: Specialized counting and verification for snooker/billiards games
- **Comprehensive Visualization**: Trajectory trails, population statistics, and performance metrics
- **Modular Architecture**: Clean separation of concerns for easy customization and testing

### Designed For

- **Cue Sports**: Snooker, pool, billiards ball tracking
- **Small Objects**: Objects that are small relative to frame size
- **Similar Appearance**: Multiple objects with similar colors and shapes
- **Challenging Conditions**: Low frame rates, motion blur, poor lighting
- **Real-Time Applications**: Optimized for real-time performance

## Quick Start

### Basic Usage

```python
from PureCV.molt import MOLTTracker

# Create tracker with default configuration
tracker = MOLTTracker()

# Initialize with first frame and detections
detections = [
    {'x': 100, 'y': 150, 'width': 30, 'height': 30, 'class': 'red'},
    {'x': 200, 'y': 250, 'width': 30, 'height': 30, 'class': 'white'}
]
success = tracker.init(first_frame, detections)

# Process video frames
for frame in video_frames:
    tracks = tracker.update(frame)
    
    # Visualize results
    vis_frame = tracker.visualize(frame, tracks)
    
    # Process tracking results
    for track in tracks:
        print(f"Track {track['id']}: {track['class']} at ({track['x']}, {track['y']})")
```

### Configuration

```python
from PureCV.molt import MOLTTracker, MOLTTrackerConfig

# Use game-specific presets
snooker_config = MOLTTrackerConfig.create_for_snooker()
pool_config = MOLTTrackerConfig.create_for_pool()

# Custom configuration
config = MOLTTrackerConfig()
config.population_sizes = {'red': 300, 'white': 1500, 'yellow': 200}
config.histogram_bins = 20
config.color_space = 'HSV'

tracker = MOLTTracker(config=config)

# Parameter overrides
tracker = MOLTTracker(
    histogram_bins=16,
    color_space='RGB',
    population_sizes={'red': 200, 'white': 1000}
)
```

## Architecture

### Core Components

```
PureCV/molt/
├── __init__.py              # Package exports and documentation
├── tracker.py               # Main MOLTTracker class
├── config.py               # Configuration management
├── population.py           # TrackerPopulation management
├── local_tracker.py        # Individual tracker logic
├── histogram_extractor.py  # Appearance feature extraction
├── ball_count_manager.py   # Count verification logic
├── types.py                # Type definitions and constants
├── USAGE_GUIDE.md          # Comprehensive usage guide
├── BALL_COUNTING_GUIDE.md  # Ball counting documentation
├── TROUBLESHOOTING_GUIDE.md # Troubleshooting and optimization
└── README.md               # This file
```

### Algorithm Flow

1. **Initialization**: Create tracker populations around detected objects
2. **Population Update**: Update all trackers with current frame data
3. **Similarity Computation**: Calculate appearance and spatial similarities
4. **Tracker Ranking**: Sort trackers by combined similarity scores
5. **Best Selection**: Select highest-weighted tracker as object position
6. **Population Regeneration**: Create new populations around top performers
7. **Count Verification**: Check ball counts and handle violations
8. **Result Generation**: Format and return tracking results

## Configuration Parameters

### Population Management

```python
# Population sizes (number of trackers per object type)
population_sizes = {
    'white': 1500,    # White ball (moves frequently)
    'red': 300,       # Red balls (medium movement)
    'yellow': 200,    # Colored balls (less movement)
    # ... other ball types
}

# Exploration radii (search area for new trackers)
exploration_radii = {
    'white': 30,      # Larger radius for fast-moving white ball
    'red': 20,        # Medium radius for red balls
    'default': 15     # Default radius for other balls
}
```

### Appearance Modeling

```python
# Histogram parameters
histogram_bins = 16           # Number of histogram bins (8-32 typical)
color_space = 'HSV'          # Color space ('HSV', 'RGB', 'LAB')

# Similarity weights
similarity_weights = {
    'histogram': 0.6,         # Appearance similarity weight
    'spatial': 0.4           # Spatial consistency weight
}
```

### Ball Counting

```python
# Expected ball counts (game-specific)
expected_ball_counts = {
    'white': 1,              # Cue ball
    'red': 15,               # Red balls (snooker)
    'yellow': 1,             # Yellow ball
    'green': 1,              # Green ball
    # ... other colored balls
}
```

## Ball Counting Logic

MOLT includes sophisticated ball counting and verification specifically designed for cue sports:

### Features

- **Count Verification**: Continuous monitoring of ball counts vs. expected values
- **Violation Detection**: Automatic detection of over-count and under-count scenarios
- **Recovery Mechanisms**: Suggestions for track merging and reassignment
- **Game State Adaptation**: Dynamic count updates as balls are potted

### Violation Handling

```python
# Monitor ball count violations
violations = tracker.ball_count_manager.get_count_violations()
for ball_class, info in violations.items():
    if info['violation_type'] != 'none':
        print(f"{ball_class}: {info['violation_type']} "
              f"(expected {info['expected']}, got {info['current']})")

# Handle over-count (too many balls)
excess_tracks = tracker.ball_count_manager.handle_duplicate_ball('red', track_ids)

# Handle under-count (missing balls)
recovery_suggestion = tracker.ball_count_manager.handle_lost_ball('yellow')
```

## Performance Optimization

### Speed Optimization

```python
# Reduce population sizes
config.population_sizes = {'red': 150, 'white': 800}

# Use fewer histogram bins
config.histogram_bins = 8

# Use RGB color space (faster conversion)
config.color_space = 'RGB'

# Reduce exploration radii
config.exploration_radii = {'white': 20, 'default': 10}
```

### Accuracy Optimization

```python
# Increase population sizes
config.population_sizes = {'red': 500, 'white': 2000}

# Use more histogram bins
config.histogram_bins = 24

# Use LAB color space (better discrimination)
config.color_space = 'LAB'

# Favor appearance over spatial
config.similarity_weights = {'histogram': 0.8, 'spatial': 0.2}
```

## Visualization

MOLT provides comprehensive visualization capabilities:

### Features

- **Bounding Boxes**: Color-coded by ball type with confidence-based thickness
- **Trajectory Trails**: Fading trails showing recent object positions
- **Population Statistics**: Real-time display of tracker performance metrics
- **Ball Count Information**: Current vs. expected counts with violation indicators
- **Performance Metrics**: Frame rate, population sizes, and weight statistics

### Example

```python
# Create detailed visualization
vis_frame = tracker.visualize(frame, tracks, output_path="frame_001.jpg")

# Visualization includes:
# - Bounding boxes with track IDs and ball classes
# - Confidence scores and population information
# - Trajectory trails with fading effects
# - Frame-level statistics and ball count status
```

## API Reference

### MOLTTracker

Main tracker class implementing the MOLT algorithm.

#### Methods

- `__init__(config=None, **kwargs)`: Initialize tracker with configuration
- `init(frame, detections)`: Initialize with first frame and detections
- `update(frame, detections=None)`: Update with new frame
- `visualize(frame, tracks, output_path=None)`: Create visualization
- `get_tracker_info()`: Get comprehensive tracker information
- `reset()`: Reset tracker state for new sequence

### MOLTTrackerConfig

Configuration management with validation and presets.

#### Class Methods

- `create_default()`: Default configuration
- `create_for_snooker()`: Snooker-optimized configuration
- `create_for_pool()`: Pool-optimized configuration

#### Methods

- `validate()`: Validate all parameters
- `to_dict()`: Convert to dictionary format

## Examples

### Complete Tracking Example

```python
import cv2
from PureCV.molt import MOLTTracker, MOLTTrackerConfig

# Create and configure tracker
config = MOLTTrackerConfig.create_for_snooker()
config.histogram_bins = 20  # Higher resolution histograms
tracker = MOLTTracker(config=config)

# Load video
cap = cv2.VideoCapture('snooker_video.mp4')

# Get initial detections (your detection method)
ret, first_frame = cap.read()
initial_detections = get_initial_detections(first_frame)

# Initialize tracker
if tracker.init(first_frame, initial_detections):
    print("Tracker initialized successfully")
else:
    print("Tracker initialization failed")
    exit()

# Process video
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update tracker
    tracks = tracker.update(frame)
    
    # Create visualization
    vis_frame = tracker.visualize(frame, tracks)
    
    # Display results
    cv2.imshow('MOLT Tracking', vis_frame)
    
    # Print tracking statistics
    if frame_count % 30 == 0:  # Every 30 frames
        info = tracker.get_tracker_info()
        print(f"Frame {frame_count}: {info['active_tracks']} active tracks")
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Ball Count Monitoring

```python
def monitor_ball_counts(tracker):
    """Monitor and report ball count violations."""
    stats = tracker.ball_count_manager.get_statistics()
    
    print(f"Ball Count Status:")
    print(f"  Total violations: {stats['total_count_violations']}")
    print(f"  Lost ball recoveries: {stats['lost_ball_recoveries']}")
    print(f"  Duplicate ball merges: {stats['duplicate_ball_merges']}")
    
    # Check current violations
    violations = tracker.ball_count_manager.get_count_violations()
    for ball_class, info in violations.items():
        if info['violation_type'] != 'none':
            print(f"  ⚠️  {ball_class}: {info['violation_type']} "
                  f"(expected {info['expected']}, got {info['current']})")

# Use in tracking loop
for frame in video_frames:
    tracks = tracker.update(frame)
    monitor_ball_counts(tracker)
```

## Testing

The MOLT package includes comprehensive tests:

```bash
# Run all tests
python PureCV/run_all_tests.py

# Run specific test suites
python PureCV/tests/test_config.py
python PureCV/tests/test_histogram_extractor.py
python PureCV/tests/test_integration.py

# Run examples
python PureCV/examples/basic_usage.py
```

## Documentation

Comprehensive documentation is available:

- **USAGE_GUIDE.md**: Detailed usage instructions and API reference
- **BALL_COUNTING_GUIDE.md**: Ball counting logic and snooker-specific features
- **TROUBLESHOOTING_GUIDE.md**: Common issues and performance optimization
- **Inline Documentation**: Comprehensive docstrings for all public methods

## Requirements

- Python 3.7+
- NumPy
- OpenCV (cv2)
- SciPy (optional, for advanced histogram methods)

## License

This implementation is part of the PureCV package and follows the same licensing terms.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass (`python PureCV/run_all_tests.py`)
2. Code follows the existing style and documentation standards
3. New features include comprehensive tests and documentation
4. Performance implications are considered and documented

## Citation

If you use MOLT in your research, please cite:

```
MOLT (Multiple Object Local Tracker)
Part of the PureCV Computer Vision Package
```

For more information and advanced usage examples, see the comprehensive documentation in the `PureCV/molt/` directory.