# Computer Vision Model Inference

This directory contains deep learning models and inference pipelines for analyzing cue sports using modern computer vision techniques.

## Features

- **Object Detection**: Pre-trained and custom models for detecting balls, cues, and table
- **Instance Segmentation**: Precise boundary detection for game objects
- **Pose Estimation**: Player pose and cue stick angle prediction
- **Action Recognition**: Classify different shots and player actions
- **Trajectory Prediction**: Predict ball paths using physics-informed models

## Structure

```
CVModelInference/
├── models/          # Model architectures
├── inference/       # Inference pipelines
├── training/        # Training scripts and utilities
├── utils/           # Helper functions
└── configs/         # Configuration files for models
```

## Usage

```python
from CVModelInference.models import build_detector
from CVModelInference.inference import VideoAnalyzer

# Initialize model and analyzer
model = build_detector('faster_rcnn', pretrained=True)
analyzer = VideoAnalyzer(model)

# Process video
results = analyzer.analyze_video('game_recording.mp4')
```

## Dependencies
- PyTorch or TensorFlow
- torchvision or tf.keras
- OpenCV
- NumPy
