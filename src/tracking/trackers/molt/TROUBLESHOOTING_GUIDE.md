# MOLT Tracker Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for the MOLT (Multiple Object Local Tracker) algorithm, covering common issues, diagnostic techniques, and solutions for optimal tracking performance.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Diagnostic Tools](#diagnostic-tools)
3. [Performance Optimization](#performance-optimization)
4. [Configuration Tuning](#configuration-tuning)
5. [Error Messages](#error-messages)
6. [Best Practices](#best-practices)

## Common Issues

### 1. Poor Tracking Accuracy

#### Symptoms
- Frequent track losses
- Objects jumping between different IDs
- Inconsistent position estimates
- High number of false positives

#### Possible Causes
- Insufficient population sizes
- Inappropriate similarity weights
- Poor histogram discrimination
- Inadequate exploration radius

#### Solutions

**Increase Population Sizes:**
```python
config = MOLTTrackerConfig()
config.population_sizes = {
    'white': 2000,  # Increase from default 1500
    'red': 500,     # Increase from default 300
    'yellow': 300   # Increase from default 200
}
```

**Adjust Similarity Weights:**
```python
# Favor appearance over spatial consistency
config.similarity_weights = {
    'histogram': 0.8,  # Increase from default 0.6
    'spatial': 0.2     # Decrease from default 0.4
}
```

**Improve Histogram Discrimination:**
```python
config.histogram_bins = 24    # Increase from default 16
config.color_space = 'LAB'    # Try LAB for better color discrimination
```

**Increase Exploration Radius:**
```python
config.exploration_radii = {
    'white': 40,      # Increase from default 30
    'red': 30,        # Increase from default 20
    'default': 25     # Increase from default 15
}
```

### 2. Slow Performance

#### Symptoms
- Low frame processing rate
- High CPU usage
- Memory consumption issues
- Real-time processing difficulties

#### Possible Causes
- Excessive population sizes
- Too many histogram bins
- Inefficient color space conversion
- Large exploration radii

#### Solutions

**Reduce Population Sizes:**
```python
config = MOLTTrackerConfig()
config.population_sizes = {
    'white': 800,   # Reduce from default 1500
    'red': 150,     # Reduce from default 300
    'yellow': 100   # Reduce from default 200
}
```

**Optimize Histogram Parameters:**
```python
config.histogram_bins = 8     # Reduce from default 16
config.color_space = 'RGB'    # Faster than HSV conversion
```

**Reduce Exploration Radius:**
```python
config.exploration_radii = {
    'white': 20,      # Reduce from default 30
    'red': 15,        # Reduce from default 20
    'default': 10     # Reduce from default 15
}
```

**Optimize Processing:**
```python
# Process every Nth frame for visualization
for i, frame in enumerate(frames):
    tracks = tracker.update(frame)
    
    # Only visualize every 5th frame
    if i % 5 == 0:
        vis_frame = tracker.visualize(frame, tracks)
```

### 3. Ball Count Violations

#### Symptoms
- Frequent count violation warnings
- Incorrect ball assignments
- Missing or duplicate balls
- Unstable track counts

#### Possible Causes
- Poor detection quality
- Incorrect expected counts
- Track fragmentation
- Misclassification

#### Solutions

**Verify Expected Counts:**
```python
# Ensure expected counts match game state
current_expected = {
    'white': 1,
    'red': 12,    # Adjusted for potted balls
    'yellow': 1,
    # ... other balls
}
tracker.ball_count_manager.update_expected_counts(current_expected)
```

**Adjust Confidence Thresholds:**
```python
config.min_confidence = 0.3  # Increase from default 0.1
```

**Improve Classification:**
```python
config.histogram_bins = 20
config.color_space = 'HSV'
config.similarity_weights = {'histogram': 0.7, 'spatial': 0.3}
```

**Monitor Violations:**
```python
def monitor_violations(tracker):
    violations = tracker.ball_count_manager.get_count_violations()
    for ball_class, info in violations.items():
        if info['violation_type'] != 'none':
            print(f"⚠️  {ball_class}: {info['violation_type']} "
                  f"(expected {info['expected']}, got {info['current']})")
```

### 4. Initialization Failures

#### Symptoms
- `init()` returns `False`
- No populations created
- Empty tracking results
- Error messages during initialization

#### Possible Causes
- Invalid frame format
- Empty or malformed detections
- Insufficient detection quality
- Configuration errors

#### Solutions

**Validate Input Data:**
```python
def validate_frame(frame):
    if frame is None:
        return False, "Frame is None"
    if len(frame.shape) != 3:
        return False, f"Frame has wrong shape: {frame.shape}"
    if frame.shape[2] != 3:
        return False, f"Frame is not 3-channel: {frame.shape[2]} channels"
    return True, "Frame is valid"

def validate_detections(detections):
    if not detections:
        return False, "No detections provided"
    
    for i, det in enumerate(detections):
        if det.get('width', 0) <= 0 or det.get('height', 0) <= 0:
            return False, f"Detection {i} has invalid size"
        if not det.get('class'):
            return False, f"Detection {i} missing class"
    
    return True, "Detections are valid"

# Use validation before initialization
frame_valid, frame_msg = validate_frame(first_frame)
det_valid, det_msg = validate_detections(initial_detections)

if frame_valid and det_valid:
    success = tracker.init(first_frame, initial_detections)
else:
    print(f"Validation failed: {frame_msg}, {det_msg}")
```

**Try Multiple Initialization Attempts:**
```python
def robust_initialization(tracker, frames, detections_list):
    for i, (frame, detections) in enumerate(zip(frames[:5], detections_list[:5])):
        print(f"Attempting initialization with frame {i}")
        if tracker.init(frame, detections):
            print(f"✓ Initialization successful on frame {i}")
            return True
        else:
            print(f"✗ Initialization failed on frame {i}")
    
    print("❌ All initialization attempts failed")
    return False
```

### 5. Memory Issues

#### Symptoms
- Increasing memory usage over time
- Out of memory errors
- System slowdown during long sequences

#### Possible Causes
- Trail history accumulation
- Population statistics buildup
- Violation history growth
- Unreleased resources

#### Solutions

**Limit Trail History:**
```python
# Modify max_trail_length in tracker
tracker.max_trail_length = 20  # Reduce from default 30
```

**Periodic Reset:**
```python
# Reset tracker periodically for long sequences
for i, frame in enumerate(video_frames):
    tracks = tracker.update(frame)
    
    # Reset every 1000 frames
    if i % 1000 == 0 and i > 0:
        print(f"Resetting tracker at frame {i}")
        # Save state if needed
        tracker.reset()
        # Re-initialize with current detections
        tracker.init(frame, current_detections)
```

**Clear Statistics:**
```python
# Clear violation history periodically
if hasattr(tracker, 'ball_count_manager'):
    if len(tracker.ball_count_manager.violation_history) > 50:
        tracker.ball_count_manager.violation_history = \
            tracker.ball_count_manager.violation_history[-25:]  # Keep last 25
```

## Diagnostic Tools

### 1. Tracker Information Analysis

```python
def analyze_tracker_state(tracker):
    """Comprehensive tracker state analysis."""
    info = tracker.get_tracker_info()
    
    print("=== TRACKER STATE ANALYSIS ===")
    print(f"Tracker: {info['name']} ({info['type']})")
    print(f"Frames processed: {info['frame_count']}")
    print(f"Active tracks: {info['active_tracks']}")
    print(f"Total tracks created: {info['total_tracks_created']}")
    print(f"Total tracks lost: {info['total_tracks_lost']}")
    
    # Performance metrics
    params = info['parameters']
    print(f"\nPerformance Metrics:")
    print(f"  Average best weight: {params.get('avg_best_weight', 0):.3f}")
    print(f"  Average population weight: {params.get('avg_population_weight', 0):.3f}")
    print(f"  Total population size: {params.get('total_population_size', 0)}")
    
    # Ball count information
    print(f"\nBall Count Information:")
    print(f"  Current counts: {params.get('current_ball_counts', {})}")
    print(f"  Violations: {params.get('ball_count_violations', 0)}")
    print(f"  Recoveries: {params.get('lost_ball_recoveries', 0)}")
    print(f"  Merges: {params.get('duplicate_ball_merges', 0)}")
```

### 2. Population Performance Analysis

```python
def analyze_population_performance(tracker):
    """Analyze individual population performance."""
    if not hasattr(tracker, 'populations') or not tracker.populations:
        print("No populations to analyze")
        return
    
    print("=== POPULATION PERFORMANCE ANALYSIS ===")
    
    for i, population in enumerate(tracker.populations):
        stats = population.get_population_statistics()
        
        print(f"\nPopulation {i} (ID: {stats['object_id']}):")
        print(f"  Object class: {stats['object_class']}")
        print(f"  Population size: {stats['population_size']}")
        print(f"  Best weight: {stats['best_weight']:.3f}")
        print(f"  Average weight: {stats['average_weight']:.3f}")
        print(f"  Weight std: {stats['weight_std']:.3f}")
        print(f"  Frame count: {stats['frame_count']}")
        print(f"  Total updates: {stats['total_updates']}")
        
        # Weight history analysis
        history = stats['best_weights_history']
        if history:
            print(f"  Weight trend: {history[-5:]}")  # Last 5 weights
            print(f"  Weight stability: {np.std(history[-10:]):.3f}")  # Stability measure
```

### 3. Ball Count Diagnostics

```python
def diagnose_ball_counts(tracker):
    """Diagnose ball counting issues."""
    if not hasattr(tracker, 'ball_count_manager'):
        print("No ball count manager available")
        return
    
    print("=== BALL COUNT DIAGNOSTICS ===")
    
    # Get current statistics
    stats = tracker.ball_count_manager.get_statistics()
    violations = tracker.ball_count_manager.get_count_violations()
    
    print(f"Total violations: {stats['total_count_violations']}")
    print(f"Lost ball recoveries: {stats['lost_ball_recoveries']}")
    print(f"Duplicate ball merges: {stats['duplicate_ball_merges']}")
    print(f"Recent violations: {stats['recent_violations']}")
    
    print(f"\nCurrent vs Expected Counts:")
    for ball_class in stats['expected_counts']:
        expected = stats['expected_counts'][ball_class]
        current = stats['current_counts'].get(ball_class, 0)
        violation = violations[ball_class]['violation_type']
        
        status = "✓" if violation == 'none' else "⚠️"
        print(f"  {status} {ball_class}: {current}/{expected} ({violation})")
    
    # Track assignments
    print(f"\nTrack Assignments:")
    assignments = stats['track_assignments']
    for track_id, ball_class in assignments.items():
        print(f"  Track {track_id}: {ball_class}")
```

### 4. Performance Profiling

```python
import time
from collections import defaultdict

class PerformanceProfiler:
    def __init__(self):
        self.times = defaultdict(list)
    
    def time_operation(self, operation_name):
        """Context manager for timing operations."""
        class Timer:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                duration = time.time() - self.start_time
                self.profiler.times[self.name].append(duration)
        
        return Timer(self, operation_name)
    
    def report(self):
        """Generate performance report."""
        print("=== PERFORMANCE PROFILE ===")
        for operation, times in self.times.items():
            avg_time = np.mean(times)
            total_time = sum(times)
            count = len(times)
            
            print(f"{operation}:")
            print(f"  Count: {count}")
            print(f"  Average: {avg_time*1000:.2f}ms")
            print(f"  Total: {total_time:.2f}s")
            print(f"  Min: {min(times)*1000:.2f}ms")
            print(f"  Max: {max(times)*1000:.2f}ms")

# Usage example
profiler = PerformanceProfiler()

for frame in video_frames:
    with profiler.time_operation("total_update"):
        with profiler.time_operation("tracker_update"):
            tracks = tracker.update(frame)
        
        with profiler.time_operation("visualization"):
            vis_frame = tracker.visualize(frame, tracks)

profiler.report()
```

## Performance Optimization

### 1. Configuration Optimization

```python
def optimize_for_speed(config):
    """Optimize configuration for maximum speed."""
    # Reduce population sizes
    config.population_sizes = {k: max(50, v // 3) for k, v in config.population_sizes.items()}
    
    # Use fewer histogram bins
    config.histogram_bins = 8
    
    # Use RGB color space (faster conversion)
    config.color_space = 'RGB'
    
    # Reduce exploration radii
    config.exploration_radii = {k: max(5, v // 2) for k, v in config.exploration_radii.items()}
    
    return config

def optimize_for_accuracy(config):
    """Optimize configuration for maximum accuracy."""
    # Increase population sizes
    config.population_sizes = {k: v * 2 for k, v in config.population_sizes.items()}
    
    # Use more histogram bins
    config.histogram_bins = 24
    
    # Use LAB color space (better discrimination)
    config.color_space = 'LAB'
    
    # Increase exploration radii
    config.exploration_radii = {k: v + 10 for k, v in config.exploration_radii.items()}
    
    # Favor appearance over spatial
    config.similarity_weights = {'histogram': 0.8, 'spatial': 0.2}
    
    return config
```

### 2. Adaptive Configuration

```python
class AdaptiveTracker:
    def __init__(self, base_config):
        self.tracker = MOLTTracker(config=base_config)
        self.performance_history = []
        self.adaptation_interval = 50  # Adapt every 50 frames
        
    def update(self, frame):
        start_time = time.time()
        tracks = self.tracker.update(frame)
        duration = time.time() - start_time
        
        self.performance_history.append(duration)
        
        # Adapt configuration if needed
        if len(self.performance_history) % self.adaptation_interval == 0:
            self._adapt_configuration()
        
        return tracks
    
    def _adapt_configuration(self):
        recent_times = self.performance_history[-self.adaptation_interval:]
        avg_time = np.mean(recent_times)
        
        if avg_time > 0.1:  # Too slow (>100ms per frame)
            print("Adapting for speed...")
            self._reduce_complexity()
        elif avg_time < 0.02:  # Very fast (<20ms per frame)
            print("Adapting for accuracy...")
            self._increase_complexity()
    
    def _reduce_complexity(self):
        # Reduce population sizes by 20%
        for ball_type in self.tracker.population_sizes:
            self.tracker.population_sizes[ball_type] = int(
                self.tracker.population_sizes[ball_type] * 0.8
            )
    
    def _increase_complexity(self):
        # Increase population sizes by 20%
        for ball_type in self.tracker.population_sizes:
            self.tracker.population_sizes[ball_type] = int(
                self.tracker.population_sizes[ball_type] * 1.2
            )
```

## Configuration Tuning

### 1. Parameter Sensitivity Analysis

```python
def analyze_parameter_sensitivity(base_config, test_frames, test_detections):
    """Analyze sensitivity to different parameters."""
    
    # Test population sizes
    population_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    results = {}
    
    for multiplier in population_multipliers:
        config = copy.deepcopy(base_config)
        config.population_sizes = {
            k: int(v * multiplier) for k, v in config.population_sizes.items()
        }
        
        tracker = MOLTTracker(config=config)
        tracker.init(test_frames[0], test_detections[0])
        
        # Measure performance
        start_time = time.time()
        for frame in test_frames[1:10]:  # Test on 10 frames
            tracks = tracker.update(frame)
        duration = time.time() - start_time
        
        # Measure accuracy (simplified)
        final_tracks = tracker.update(test_frames[10])
        accuracy = len(final_tracks) / len(test_detections[0])  # Simplified metric
        
        results[multiplier] = {
            'duration': duration,
            'accuracy': accuracy,
            'tracks': len(final_tracks)
        }
    
    # Report results
    print("Population Size Sensitivity Analysis:")
    for multiplier, result in results.items():
        print(f"  {multiplier}x: {result['duration']:.3f}s, "
              f"accuracy: {result['accuracy']:.2f}, "
              f"tracks: {result['tracks']}")
```

### 2. Automatic Parameter Tuning

```python
def auto_tune_parameters(video_frames, initial_detections, target_fps=30):
    """Automatically tune parameters for target performance."""
    
    target_frame_time = 1.0 / target_fps
    
    # Start with default configuration
    config = MOLTTrackerConfig.create_default()
    
    # Test current performance
    tracker = MOLTTracker(config=config)
    tracker.init(video_frames[0], initial_detections)
    
    # Measure performance on sample frames
    test_frames = video_frames[1:21]  # Test on 20 frames
    start_time = time.time()
    
    for frame in test_frames:
        tracks = tracker.update(frame)
    
    avg_frame_time = (time.time() - start_time) / len(test_frames)
    
    print(f"Initial performance: {avg_frame_time:.3f}s per frame")
    print(f"Target: {target_frame_time:.3f}s per frame")
    
    # Adjust parameters if needed
    if avg_frame_time > target_frame_time:
        # Too slow - reduce complexity
        reduction_factor = target_frame_time / avg_frame_time
        
        config.population_sizes = {
            k: max(50, int(v * reduction_factor)) 
            for k, v in config.population_sizes.items()
        }
        
        if reduction_factor < 0.5:
            config.histogram_bins = max(8, int(config.histogram_bins * 0.7))
        
        print(f"Reduced complexity by factor {reduction_factor:.2f}")
    
    return config
```

## Error Messages

### Common Error Messages and Solutions

#### "Tracker initialization failed"
**Cause**: Invalid frame or detections
**Solution**: Validate input data format and content

#### "Frame must be a 3-channel image"
**Cause**: Incorrect frame format
**Solution**: Ensure frame is BGR format with shape (H, W, 3)

#### "Histogram cannot be None or empty"
**Cause**: Histogram extraction failure
**Solution**: Check patch size and image quality

#### "Population size must be positive"
**Cause**: Invalid configuration
**Solution**: Ensure all population sizes are > 0

#### "Color space must be HSV, RGB, or LAB"
**Cause**: Unsupported color space
**Solution**: Use one of the supported color spaces

#### "Similarity weights must contain 'histogram' and 'spatial' keys"
**Cause**: Missing weight configuration
**Solution**: Provide both histogram and spatial weights

## Best Practices

### 1. Initialization Best Practices

```python
# Use high-quality detections from multiple frames
def robust_initialization(tracker, video_frames):
    # Try first 5 frames for initialization
    for i in range(min(5, len(video_frames))):
        frame = video_frames[i]
        detections = get_detections(frame)  # Your detection method
        
        # Filter high-confidence detections
        good_detections = [
            det for det in detections 
            if det.get('confidence', 0) > 0.7 and 
               det.get('width', 0) > 10 and 
               det.get('height', 0) > 10
        ]
        
        if len(good_detections) >= 3:  # Need minimum number of detections
            if tracker.init(frame, good_detections):
                print(f"Successfully initialized with frame {i}")
                return True
    
    return False
```

### 2. Configuration Best Practices

```python
def create_optimized_config(video_properties):
    """Create configuration based on video properties."""
    config = MOLTTrackerConfig.create_default()
    
    # Adjust based on frame rate
    fps = video_properties.get('fps', 30)
    if fps < 15:
        # Low frame rate - increase exploration radius
        config.exploration_radii = {k: v * 1.5 for k, v in config.exploration_radii.items()}
    elif fps > 60:
        # High frame rate - can use smaller radius
        config.exploration_radii = {k: v * 0.8 for k, v in config.exploration_radii.items()}
    
    # Adjust based on resolution
    width = video_properties.get('width', 640)
    height = video_properties.get('height', 480)
    resolution_factor = (width * height) / (640 * 480)
    
    if resolution_factor > 2:
        # High resolution - can use more bins
        config.histogram_bins = 20
    elif resolution_factor < 0.5:
        # Low resolution - use fewer bins
        config.histogram_bins = 12
    
    return config
```

### 3. Monitoring Best Practices

```python
def setup_monitoring(tracker):
    """Set up comprehensive monitoring."""
    
    class TrackerMonitor:
        def __init__(self, tracker):
            self.tracker = tracker
            self.frame_count = 0
            self.performance_log = []
            self.violation_log = []
        
        def update(self, frame):
            start_time = time.time()
            tracks = self.tracker.update(frame)
            duration = time.time() - start_time
            
            self.frame_count += 1
            self.performance_log.append(duration)
            
            # Check for violations
            if hasattr(self.tracker, 'ball_count_manager'):
                if not self.tracker.ball_count_manager.verify_counts():
                    violations = self.tracker.ball_count_manager.get_count_violations()
                    self.violation_log.append((self.frame_count, violations))
            
            # Report every 100 frames
            if self.frame_count % 100 == 0:
                self.report()
            
            return tracks
        
        def report(self):
            recent_performance = self.performance_log[-100:]
            avg_time = np.mean(recent_performance)
            
            print(f"Frame {self.frame_count}: {avg_time*1000:.1f}ms avg")
            
            if self.violation_log:
                recent_violations = len([v for v in self.violation_log if v[0] > self.frame_count - 100])
                print(f"  Recent violations: {recent_violations}")
    
    return TrackerMonitor(tracker)
```

This comprehensive troubleshooting guide provides detailed solutions for common MOLT tracker issues and best practices for optimal performance.