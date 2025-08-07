# Physics Inference for Cue Sports

A comprehensive research project focused on analyzing cue sports (like snooker) using computer vision and machine learning techniques. The project provides tools for object detection, multi-object tracking, physics analysis, and game state understanding.

## 🎯 Key Features

- **Advanced Multi-Object Tracking**: MOLT (Multiple Object Local Tracker) for robust ball tracking
- **Deep Learning Integration**: Support for PyTorch models and Roboflow inference
- **Traditional CV Methods**: OpenCV-based detection and tracking algorithms
- **Comprehensive Testing**: Full test coverage with quality assurance tools
- **Type Safety**: Complete MyPy compliance with comprehensive type hints
- **Modular Architecture**: Clean, maintainable code with focused components

## 📁 Project Structure

### `src/` - Core Library
- **`detection/`**: Object detection models and inference runners
  - Local PyTorch model inference (`local_pt_inference.py`)
  - Roboflow integration (`roboflow_local_inference.py`)
  - Table detection utilities (`table_detection.py`)
- **`tracking/`**: Multi-object tracking systems
  - MOLT tracker implementation (`trackers/molt/`)
  - Benchmark system (`tracker_benchmark.py`)
  - Ground truth evaluation tools
  - Multiple tracking algorithm implementations

### `scripts/` - Utilities and Tools
- **Quality Assurance**: Comprehensive testing and type checking tools
- **Detection Scripts**: Model comparison and evaluation utilities
- **Tracking Scripts**: Tracker benchmarking and comparison tools

### `tests/` - Test Suite
- Unit tests, integration tests, and script functionality tests

## 🚀 Getting Started

### Prerequisites
- Python 3.7+ (recommended: 3.8+)
- OpenCV 4.5+
- PyTorch 1.9+ (for deep learning models)
- NumPy, SciPy, Matplotlib
- Additional dependencies listed in `requirements.txt`

### Installation

#### Option 1: Development Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/physics_inference.git
cd physics_inference

# Create and activate virtual environment
python -m venv venv_3.12
source venv_3.12/bin/activate  # On Windows: venv_3.12\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Option 2: Package Installation
```bash
pip install -e git+https://github.com/yourusername/physics_inference.git#egg=physics_inference
```

### Quick Start Example

```python
from tracking.trackers.molt import MOLTTracker
from detection.local_pt_inference import LocalPT
import cv2

# Initialize detector and tracker
detector = LocalPT(model_path="path/to/your/model.pt")
tracker = MOLTTracker()

# Process video
video = cv2.VideoCapture("snooker_video.mp4")
ret, frame = video.read()

# Get initial detections
detections = detector.run_inference(frame)

# Initialize tracker
tracker.initialize(frame, detections)

# Track objects in subsequent frames
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    tracked_objects = tracker.update(frame)
    # Process tracked objects...
```

## 📖 Documentation

- **[Tracking System Guide](src/tracking/README_TRACKERS.md)**: Comprehensive tracker documentation
- **[MOLT Tracker Guide](src/tracking/trackers/molt/README.md)**: Detailed MOLT usage and configuration
- **[Scripts Documentation](scripts/README.md)**: Quality assurance and utility scripts
- **[API Reference](docs/api/)**: Detailed API documentation (coming soon)

## 🧪 Testing and Quality Assurance

Run the complete test suite:
```bash
# Run all tests and checks
python scripts/run_all_checks.py

# Run only unit tests
python scripts/run_tests.py

# Run only type checking
python scripts/run_mypy.py
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- Related research papers and open-source projects
