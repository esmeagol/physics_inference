# API Reference

## Detection

### LocalPT
```python
from detection.local_pt_inference import LocalPT

# Constructor
LocalPT(model_path: str, confidence_threshold: float = 0.5, device: Optional[str] = None)

# Methods
run_inference(image: np.ndarray) -> List[Dict[str, Any]]
run_batch_inference(images: List[np.ndarray]) -> List[List[Dict[str, Any]]]
get_model_info() -> Dict[str, Any]
```

### RoboflowLocalInference
```python
from detection.roboflow_local_inference import RoboflowLocalInference

# Constructor
RoboflowLocalInference(api_key: str, workspace: str, project: str, version: int)

# Methods
run_inference(image: np.ndarray) -> List[Dict[str, Any]]
```

## Tracking

### MOLTTracker
```python
from tracking.trackers.molt import MOLTTracker, MOLTTrackerConfig

# Constructor
MOLTTracker(config: Optional[MOLTTrackerConfig] = None, **kwargs)

# Methods
initialize(frame: np.ndarray, detections: List[Dict[str, Any]]) -> None
update(frame: np.ndarray) -> List[Dict[str, Any]]
get_tracks() -> Dict[int, List[Dict[str, Any]]]
```

### MOLTTrackerConfig
```python
# Factory methods
MOLTTrackerConfig.create_for_snooker() -> MOLTTrackerConfig
MOLTTrackerConfig.create_for_pool() -> MOLTTrackerConfig

# Key parameters: histogram_bins, color_space, population_sizes, spatial_threshold
```

### TrackerBenchmark
```python
from tracking.tracker_benchmark import TrackerBenchmark

# Constructor
TrackerBenchmark(metrics: Optional[List[str]] = None, output_dir: Optional[str] = None)

# Methods
add_tracker(name: str, tracker: Any) -> None
run_benchmark(video_path: str, **kwargs) -> Dict[str, Any]
```

## Utilities

### Ground Truth Evaluation
```python
from tracking.ground_truth_evaluator import GroundTruthEvaluator
evaluate_tracking_accuracy(ground_truth: List, predictions: List) -> Dict[str, float]
```

### Visualization
```python
from tracking.ground_truth_visualizer import GroundTruthVisualizer
visualize_tracks(video_path: str, tracks: Dict, output_path: str) -> None
```

## Data Formats

**Detection**: `{x, y, width, height, class, confidence}`
**Track**: `{track_id, detections, timestamps}`

**Exceptions**: `FileNotFoundError`, `ValueError`, `RuntimeError`, `ImportError`

## Example

```python
import cv2
from detection.local_pt_inference import LocalPT
from tracking.trackers.molt import MOLTTracker

detector = LocalPT("models/snooker.pt")
tracker = MOLTTracker()

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()
detections = detector.run_inference(frame)
tracker.initialize(frame, detections)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    tracks = tracker.update(frame)
    # Process tracks...
```
