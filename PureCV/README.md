# Pure Computer Vision for Cue Sports

This directory contains traditional computer vision techniques for analyzing cue sports. The focus is on reliable, interpretable methods that don't require deep learning.

## Features

- **Ball Detection**: Identify and track balls using color segmentation and contour detection
- **Table Detection**: Locate the playing surface and correct perspective
- **Cue Detection**: Detect and track the cue stick
- **Physics Analysis**: Calculate ball velocities and predict trajectories
- **Collision Detection**: Detect and analyze ball-ball and ball-cushion collisions

## Structure

```
PureCV/
├── detection/       # Object detection modules
├── tracking/        # Object tracking implementations
├── utils/           # Helper functions and utilities
├── visualization/   # Visualization tools
└── physics/         # Physics modeling and analysis
```

## Usage

```python
import cv2
from PureCV.detection import BallDetector
from PureCV.tracking import BallTracker

# Initialize detector and tracker
detector = BallDetector()
tracker = BallTracker()

# Process video frame
frame = cv2.imread('snooker_frame.jpg')
balls = detector.detect(frame)
tracks = tracker.update(balls)
```

## Dependencies
- OpenCV
- NumPy
- SciPy
