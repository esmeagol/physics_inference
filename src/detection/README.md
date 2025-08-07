# Detection Module

Object detection for cue sports analysis with local PyTorch models and Roboflow integration.

## Components

- **`LocalPT`**: Local PyTorch/YOLO inference with GPU support
- **`RoboflowLocalInference`**: Cloud-based Roboflow API integration  
- **`TableDetection`**: Table boundary detection and perspective correction
- **`InferenceRunner`**: Abstract base class for all inference runners

## Quick Start

```python
# Local PyTorch inference
from detection.local_pt_inference import LocalPT
detector = LocalPT("models/snooker.pt")
detections = detector.run_inference(image)

# Roboflow integration
from detection.roboflow_local_inference import RoboflowLocalInference
detector = RoboflowLocalInference(api_key="key", workspace="ws", project="proj", version=1)
detections = detector.run_inference(image)

# Table detection
from detection.table_detection import TableDetection
detector = TableDetection()
table_corners = detector.detect_table(image)
```

## Configuration

Set up `.env` file:
```bash
ROBOFLOW_API_KEY=your_key
ROBOFLOW_WORKSPACE=workspace
ROBOFLOW_PROJECT=project
DEFAULT_YOLO_MODEL_PATH=trained_models/best.pt
```

**Output Format**: All detectors return standardized dictionaries:
- `x`, `y`, `width`, `height`: Bounding box
- `class`: Object class name  
- `confidence`: Detection confidence

**Testing**: `python -m pytest tests/test_detection/ -v`
