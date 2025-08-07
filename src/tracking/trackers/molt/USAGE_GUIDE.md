# MOLT Tracker Usage Guide

## Overview

The MOLT (Multiple Object Local Tracker) is a robust tracking algorithm specifically designed for tracking multiple small, similar objects in low-quality, low-frame-rate video. It uses a population-based approach where each object is tracked by multiple local trackers that combine appearance features (color histograms) with motion constraints.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [API Reference](#api-reference)
4. [Ball Counting Logic](#ball-counting-logic)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Quick Start

### Basic Usage

```python
from tracking.trackers.molt import MOLTTracker

# Create tracker with default configuration
tracker = MOLTTracker()

# Initialize with first frame and detections
success = tracker.init(frame, initial_detections)

# Process subsequent frames
for frame in video_frames:
    tracks = tracker.update(frame)
    
    # Visualize results
    vis_frame = tracker.visualize(frame, tracks)
```

### Detection Format

Detections should be provided as a list of dictionaries with the following format:

```python
detection = {
    'x': 100.0,           # Center x coordinate
    'y': 150.0,           # Center y coordinate
    'width': 30.0,        # Bounding box width
    'height': 30.0,       # Bounding box height
    'class': 'red',       # Ball color ('red', 'white', 'yellow', etc.)
    'class_id': 0,        # Numeric class identifier (optional)
    'confidence': 0.9     # Detection confidence [0, 1] (optional)
}
```

### Tracking Results

The tracker returns a list of tracking results with the following format:

```python
track = {
    'id': 1,                    # Unique track ID
    'x': 105.0,                 # Center x coordinate
    'y': 155.0,                 # Center y coordinate
    'width': 30.0,              # Bounding box width
    'height': 30.0,             # Bounding box height
    'class': 'red',             # Ball color
    'class_id': 0,              # Numeric class identifier
    'confidence': 0.85,         # Tracking confidence [0, 1]
    'trail': [(100, 150), ...], # Position history
    'population_size': 300,     # Number of trackers for this object
    'best_weight': 0.85,        # Weight of best tracker
    'frames_tracked': 15        # Number of frames tracked
}
```

## Configuration

### Default Configuration

The MOLT tracker comes with sensible defaults optimized for snooker tracking:

```python
from tracking.trackers.molt import MOLTTrackerConfig

# Create default configuration
config = MOLTTrackerConfig.create_default()

# View configuration parameters
print(config.population_sizes)    # Population sizes per ball type
print(config.exploration_radii)   # Search radii per ball type
print(config.expected_ball_counts) # Expected ball counts
```

### Game-Specific Configurations

#### Snooker Configuration

```python
# Optimized for snooker (15 red balls + 6 colored balls + white ball)
config = MOLTTrackerConfig.create_for_snooker()
tracker = MOLTTracker(config=config)
```

#### Pool/Billiards Configuration

```python
# Optimized for pool/billiards (8-ball, 9-ball, etc.)
config = MOLTTrackerConfig.create_for_pool()
tracker = MOLTTracker(config=config)
```

### Custom Configuration

```python
# Create custom configuration
config = MOLTTrackerConfig()

# Customize population sizes (more trackers = better accuracy, slower performance)
config.population_sizes = {
    'white': 1500,    # White ball moves frequently, needs more trackers
    'red': 300,       # Red balls move less frequently
    'yellow': 200,    # Colored balls move least frequently
    'green': 200,
    'brown': 200,
    'blue': 200,
    'pink': 200,
    'black': 200
}

# Customize exploration radii (larger radius = wider search area)
config.exploration_radii = {
    'white': 30,      # White ball can move farther between frames
    'red': 20,        # Red balls have medium movement
    'default': 15     # Default for other ball types
}

# Customize histogram parameters
config.histogram_bins = 16        # Number of histogram bins (8-32 typical)
config.color_space = 'HSV'       # Color space: 'HSV', 'RGB', or 'LAB'

# Customize similarity weights
config.similarity_weights = {
    'histogram': 0.6,  # Weight for appearance similarity
    'spatial': 0.4     # Weight for spatial consistency
}

# Create tracker with custom configuration
tracker = MOLTTracker(config=config)
```

### Parameter Override with kwargs

You can override specific parameters without creating a full configuration:

```python
# Override specific parameters
tracker = MOLTTracker(
    histogram_bins=12,
    color_space='RGB',
    min_confidence=0.2,
    population_sizes={'red': 200, 'white': 1000}
)
```

## API Reference

### MOLTTracker Class

#### Constructor

```python
MOLTTracker(config=None, **kwargs)
```

**Parameters:**
- `config` (MOLTTrackerConfig, optional): Configuration object with tracker parameters
- `**kwargs`: Additional parameters that override config values

**Example:**
```python
# With configuration object
config = MOLTTrackerConfig.create_for_snooker()
tracker = MOLTTracker(config=config)

# With parameter overrides
tracker = MOLTTracker(histogram_bins=16, color_space='HSV')
```

#### init(frame, detections)

Initialize the tracker with the first frame and initial detections.

**Parameters:**
- `frame` (numpy.ndarray): First frame as (H, W, C) BGR image
- `detections` (List[Dict]): List of initial detection dictionaries

**Returns:**
- `bool`: True if initialization successful, False otherwise

**Example:**
```python
success = tracker.init(first_frame, initial_detections)
if not success:
    print("Tracker initialization failed")
```

#### update(frame, detections=None)

Update the tracker with a new frame and optional new detections.

**Parameters:**
- `frame` (numpy.ndarray): New frame as (H, W, C) BGR image
- `detections` (List[Dict], optional): New detections to incorporate

**Returns:**
- `List[Dict]`: List of tracking results

**Example:**
```python
tracks = tracker.update(current_frame)
for track in tracks:
    print(f"Track {track['id']}: {track['class']} at ({track['x']}, {track['y']})")
```

#### visualize(frame, tracks, output_path=None)

Create a visualization of tracking results on the input frame.

**Parameters:**
- `frame` (numpy.ndarray): Input frame as (H, W, C) BGR image
- `tracks` (List[Dict]): Tracking results from update()
- `output_path` (str, optional): Path to save visualization

**Returns:**
- `numpy.ndarray`: Frame with tracking visualization

**Example:**
```python
vis_frame = tracker.visualize(frame, tracks, "output/frame_001.jpg")
cv2.imshow("MOLT Tracking", vis_frame)
```

#### get_tracker_info()

Get comprehensive information about the tracker state and performance.

**Returns:**
- `Dict`: Tracker information including parameters and statistics

**Example:**
```python
info = tracker.get_tracker_info()
print(f"Tracker: {info['name']}")
print(f"Active tracks: {info['active_tracks']}")
print(f"Frames processed: {info['frame_count']}")
```

#### reset()

Reset the tracker state for a new video sequence.

**Example:**
```python
tracker.reset()  # Clear all tracking state
```

### MOLTTrackerConfig Class

#### Class Methods

```python
MOLTTrackerConfig.create_default()      # Default configuration
MOLTTrackerConfig.create_for_snooker()  # Snooker-optimized
MOLTTrackerConfig.create_for_pool()     # Pool-optimized
```

#### validate()

Validate all configuration parameters.

**Raises:**
- `ValueError`: If any parameter is invalid

**Example:**
```python
config = MOLTTrackerConfig()
config.histogram_bins = -1  # Invalid value
config.validate()  # Raises ValueError
```

#### to_dict()

Convert configuration to dictionary format.

**Returns:**
- `Dict`: Configuration as dictionary

## Ball Counting Logic

The MOLT tracker includes sophisticated ball counting logic specifically designed for snooker and billiards games.

### Expected Ball Counts

Configure expected ball counts for your game type:

```python
# Snooker (default)
expected_counts = {
    'white': 1,    # Cue ball
    'red': 15,     # Red balls (can be reduced as balls are potted)
    'yellow': 1,   # Yellow ball
    'green': 1,    # Green ball
    'brown': 1,    # Brown ball
    'blue': 1,     # Blue ball
    'pink': 1,     # Pink ball
    'black': 1     # Black ball
}

# Pool/8-ball
expected_counts = {
    'white': 1,    # Cue ball
    'yellow': 7,   # Stripes or solids
    'red': 7,      # Stripes or solids
    'black': 1     # 8-ball
}
```

### Count Violation Handling

The tracker automatically detects and handles count violations:

#### Over-count (Too Many Balls)

When more balls of a type are detected than expected:

1. **Detection**: System identifies excess balls of the same type
2. **Analysis**: Calculates spatial distances between similar balls
3. **Suggestion**: Recommends track merging for closest ball pairs
4. **Action**: Logs violation and suggests corrective measures

#### Under-count (Missing Balls)

When fewer balls are detected than expected:

1. **Detection**: System identifies missing ball types
2. **Recovery**: Attempts to reassign existing tracks
3. **Search**: Increases exploration radius for missing ball types
4. **Logging**: Records recovery attempts and success rates

### Accessing Count Information

```python
# Get current ball count statistics
info = tracker.get_tracker_info()
current_counts = info['parameters']['current_ball_counts']
violations = info['parameters']['ball_count_violations']

print(f"Current ball counts: {current_counts}")
print(f"Total violations: {violations}")

# Get detailed violation information
if hasattr(tracker, 'ball_count_manager'):
    stats = tracker.ball_count_manager.get_statistics()
    print(f"Lost ball recoveries: {stats['lost_ball_recoveries']}")
    print(f"Duplicate ball merges: {stats['duplicate_ball_merges']}")
```

## Best Practices

### 1. Initialization

- **Use Quality Detections**: Initialize with high-confidence detections from the first few frames
- **Verify Frame Quality**: Ensure the first frame has good lighting and minimal motion blur
- **Check Initialization Success**: Always verify that `init()` returns `True`

```python
# Good initialization practice
success = tracker.init(first_frame, initial_detections)
if not success:
    # Try with different detections or frame
    success = tracker.init(backup_frame, backup_detections)
```

### 2. Configuration Tuning

- **Population Sizes**: Start with defaults, increase for better accuracy or decrease for speed
- **Exploration Radii**: Adjust based on object movement speed and frame rate
- **Histogram Bins**: Use 8-16 bins for speed, 16-32 for accuracy

```python
# Performance vs. accuracy trade-offs
fast_config = MOLTTrackerConfig()
fast_config.population_sizes = {'red': 100, 'white': 500}  # Smaller populations
fast_config.histogram_bins = 8  # Fewer bins

accurate_config = MOLTTrackerConfig()
accurate_config.population_sizes = {'red': 500, 'white': 2000}  # Larger populations
accurate_config.histogram_bins = 32  # More bins
```

### 3. Frame Processing

- **Consistent Frame Rate**: Process frames at consistent intervals
- **Frame Quality**: Ensure frames are properly exposed and focused
- **Error Handling**: Handle frame processing errors gracefully

```python
# Robust frame processing
for frame in video_frames:
    if frame is not None and frame.size > 0:
        try:
            tracks = tracker.update(frame)
            # Process tracks...
        except Exception as e:
            print(f"Frame processing error: {e}")
            continue
```

### 4. Performance Optimization

- **Batch Processing**: Process multiple frames before visualization
- **Selective Visualization**: Only visualize every Nth frame
- **Memory Management**: Reset tracker periodically for long videos

```python
# Optimized processing loop
for i, frame in enumerate(video_frames):
    tracks = tracker.update(frame)
    
    # Only visualize every 10th frame
    if i % 10 == 0:
        vis_frame = tracker.visualize(frame, tracks)
        
    # Reset tracker every 1000 frames to prevent memory buildup
    if i % 1000 == 0 and i > 0:
        # Save current state if needed, then reset
        tracker.reset()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Poor Tracking Performance

**Symptoms:**
- Tracks frequently lost
- High number of false positives
- Inconsistent object identification

**Solutions:**
```python
# Increase population sizes
config.population_sizes = {'red': 500, 'white': 1500}

# Adjust similarity weights
config.similarity_weights = {'histogram': 0.7, 'spatial': 0.3}

# Use more histogram bins
config.histogram_bins = 24

# Increase exploration radius
config.exploration_radii = {'white': 40, 'default': 25}
```

#### 2. Slow Performance

**Symptoms:**
- High CPU usage
- Low frame processing rate
- Memory consumption issues

**Solutions:**
```python
# Reduce population sizes
config.population_sizes = {'red': 150, 'white': 800}

# Use fewer histogram bins
config.histogram_bins = 8

# Reduce exploration radius
config.exploration_radii = {'white': 20, 'default': 10}

# Use RGB instead of HSV (faster conversion)
config.color_space = 'RGB'
```

#### 3. Ball Count Violations

**Symptoms:**
- Frequent count violation warnings
- Incorrect ball assignments
- Missing or duplicate balls

**Solutions:**
```python
# Adjust expected counts for game state
tracker.ball_count_manager.update_expected_counts({
    'red': 10,  # Reduced from 15 as balls are potted
    'white': 1,
    # ... other balls
})

# Increase minimum confidence threshold
config.min_confidence = 0.3

# Adjust similarity weights to favor appearance
config.similarity_weights = {'histogram': 0.8, 'spatial': 0.2}
```

#### 4. Initialization Failures

**Symptoms:**
- `init()` returns `False`
- No populations created
- Empty tracking results

**Solutions:**
```python
# Validate detections before initialization
valid_detections = []
for det in initial_detections:
    if (det.get('width', 0) > 0 and det.get('height', 0) > 0 and 
        det.get('class') is not None):
        valid_detections.append(det)

# Use multiple initialization attempts
for frame_idx in range(5):  # Try first 5 frames
    frame = video_frames[frame_idx]
    if tracker.init(frame, valid_detections):
        break
```

### Debug Information

Enable detailed logging to diagnose issues:

```python
# Get comprehensive tracker information
info = tracker.get_tracker_info()
print(f"Tracker state: {info}")

# Get population statistics
if hasattr(tracker, 'populations'):
    for i, pop in enumerate(tracker.populations):
        stats = pop.get_population_statistics()
        print(f"Population {i}: {stats}")

# Get ball count information
if hasattr(tracker, 'ball_count_manager'):
    count_stats = tracker.ball_count_manager.get_statistics()
    print(f"Ball count stats: {count_stats}")
```

## Advanced Usage

### 1. Custom Histogram Extractors

```python
from tracking.trackers.molt.histogram_extractor import HistogramExtractor

# Create custom histogram extractor
hist_extractor = HistogramExtractor(num_bins=20, color_space='LAB')

# Use with tracker (requires modifying tracker initialization)
tracker.histogram_extractor = hist_extractor
```

### 2. Population Statistics Analysis

```python
# Analyze population performance
def analyze_populations(tracker):
    for pop in tracker.populations:
        stats = pop.get_population_statistics()
        print(f"Object {stats['object_id']} ({stats['object_class']}):")
        print(f"  Best weight: {stats['best_weight']:.3f}")
        print(f"  Average weight: {stats['average_weight']:.3f}")
        print(f"  Weight std: {stats['weight_std']:.3f}")
        print(f"  Frames tracked: {stats['frame_count']}")

# Call after processing several frames
analyze_populations(tracker)
```

### 3. Custom Ball Count Rules

```python
# Implement custom ball counting logic
class CustomBallCountManager:
    def __init__(self, custom_rules):
        self.rules = custom_rules
    
    def verify_counts(self, tracks):
        # Custom verification logic
        pass

# Replace default ball count manager
tracker.ball_count_manager = CustomBallCountManager(my_rules)
```

### 4. Track Persistence and Recovery

```python
# Save tracker state
def save_tracker_state(tracker, filepath):
    state = {
        'populations': [pop.get_population_statistics() for pop in tracker.populations],
        'ball_counts': tracker.ball_counts,
        'frame_count': tracker.frame_count
    }
    with open(filepath, 'w') as f:
        json.dump(state, f)

# Load tracker state
def load_tracker_state(tracker, filepath):
    with open(filepath, 'r') as f:
        state = json.load(f)
    # Restore state logic here...
```

### 5. Multi-Camera Tracking

```python
# Track across multiple camera views
class MultiCameraMOLT:
    def __init__(self, num_cameras):
        self.trackers = [MOLTTracker() for _ in range(num_cameras)]
    
    def update_all_cameras(self, frames):
        all_tracks = []
        for i, (tracker, frame) in enumerate(zip(self.trackers, frames)):
            tracks = tracker.update(frame)
            # Add camera ID to tracks
            for track in tracks:
                track['camera_id'] = i
            all_tracks.extend(tracks)
        return all_tracks
```

This comprehensive usage guide covers all aspects of using the MOLT tracker effectively. For additional help or advanced use cases, refer to the example scripts.