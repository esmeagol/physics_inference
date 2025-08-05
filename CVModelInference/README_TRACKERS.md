# Tracker Benchmark System

This directory contains a comprehensive tracking benchmark system for comparing different object tracking methodologies, with a focus on snooker ball tracking.

## Overview

The system consists of:

1. **Tracker Interface** (`tracker.py`): An abstract base class defining a common interface for all trackers
2. **Tracker Implementations**:
   - **OpenCV Tracker** (`trackers/opencv_tracker.py`): Uses OpenCV's built-in tracking algorithms
   - **SORT Tracker** (`trackers/sort_tracker.py`): Implements Simple Online and Realtime Tracking with Kalman filtering
   - **DeepSORT Tracker** (`trackers/deepsort_tracker.py`): Extends SORT with appearance features for more robust tracking
3. **Benchmarking System** (`tracker_benchmark.py`):
   - `TrackerBenchmark`: General-purpose benchmark for comparing trackers
   - `SnookerTrackerBenchmark`: Specialized benchmark with snooker-specific metrics

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- SciPy
- Matplotlib

Note: The OpenCV tracker implementation depends on the available tracking algorithms in your OpenCV installation. The current implementation supports:
- MIL tracker (cv2.TrackerMIL_create)

## Usage

### Basic Usage

```python
from CVModelInference.trackers.opencv_tracker import OpenCVTracker
from CVModelInference.trackers.sort_tracker import SORTTracker
from CVModelInference.trackers.deepsort_tracker import DeepSORTTracker
from CVModelInference.tracker_benchmark import TrackerBenchmark

# Initialize benchmark
benchmark = TrackerBenchmark()

# Add trackers to benchmark
benchmark.add_tracker('OpenCV-MIL', OpenCVTracker(tracker_type='mil'))
benchmark.add_tracker('SORT', SORTTracker(max_age=30, min_hits=3))
benchmark.add_tracker('DeepSORT', DeepSORTTracker(max_age=30, min_hits=3))

# Run benchmark
results = benchmark.run_benchmark('path/to/video.mp4', detection_interval=5)

# Print results
benchmark.print_results()

# Visualize results
benchmark.visualize_results('benchmark_results.png')
```

### Using with a Detection Model

For best results, provide a detection model that implements the `InferenceRunner` interface:

```python
from detection.local_pt_inference import LocalPT
from CVModelInference.tracker_benchmark import SnookerTrackerBenchmark

# Initialize detection model
detection_model = LocalPT(model_path='path/to/model.pt', confidence=0.25)

# Initialize benchmark with detection model
benchmark = SnookerTrackerBenchmark(detection_model)

# Add trackers and run benchmark as before
```

### Command-Line Interface

The system includes a command-line script for easy benchmarking:

```bash
python -m CVModelInference.scripts.compare_trackers \
    --video path/to/video.mp4 \
    --model path/to/model.pt \
    --output-dir output \
    --detection-interval 5 \
    --snooker \
    --visualize
```

## Snooker-Specific Metrics

The `SnookerTrackerBenchmark` class provides specialized metrics for snooker ball tracking:

1. **Ball ID Consistency**: Measures how consistently each ball maintains its tracking ID
2. **Cue Ball Tracking**: Evaluates tracking quality for the cue ball
3. **Color Ball Tracking**: Evaluates tracking quality for colored balls
4. **Red Ball Tracking**: Evaluates tracking quality for red balls

## Extending the System

### Adding a New Tracker

To add a new tracker, create a class that implements the `Tracker` interface:

```python
from CVModelInference.tracker import Tracker

class MyNewTracker(Tracker):
    def __init__(self, **kwargs):
        # Initialize your tracker
        pass
        
    def init(self, frame, detections):
        # Initialize tracking with first frame and detections
        pass
        
    def update(self, frame, detections=None):
        # Update tracking with new frame and optional detections
        pass
        
    def reset(self):
        # Reset tracker state
        pass
        
    def visualize(self, frame, tracks, output_path=None):
        # Visualize tracking results
        pass
        
    def get_tracker_info(self):
        # Return tracker metadata
        pass
```

### Adding New Metrics

To add new metrics, extend the `TrackerBenchmark` class:

```python
from CVModelInference.tracker_benchmark import TrackerBenchmark

class MyBenchmark(TrackerBenchmark):
    def __init__(self, detection_model=None):
        super().__init__(detection_model)
        self.my_metrics = {}
        
    def calculate_metrics(self):
        super().calculate_metrics()
        # Calculate your custom metrics
        for tracker_name, tracker_data in self.results.items():
            self.my_metrics[tracker_name] = calculate_my_metric(tracker_data)
```

## Limitations

1. The OpenCV tracker implementation is limited by the available tracking algorithms in your OpenCV installation.
2. For optimal tracking performance, a good detection model is essential.
3. The snooker-specific metrics are based on simple heuristics and may need refinement for production use.
