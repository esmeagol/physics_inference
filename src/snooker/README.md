# Snooker Scoring Module

This module provides automated snooker match analysis through video processing, including perspective transformation, ball detection, tracking, and scoring.

## Overview

The snooker scoring module implements a pipeline architecture that processes video input through sequential stages:

```
Video Input → PerspectiveTransformer → ColorBasedBallDetector → SnookerBallDetector → BallTracker → EventAnalyzer → ScoringEngine → Score Output
```

## Components

### 1. Table Generation (`table_generator.py`)

Creates sophisticated 2D snooker table representations with accurate dimensions and markings.

**Features:**
- Standard snooker table dimensions (1830x3660 pixels)
- Accurate ball positions and spots
- Correctly positioned D-shape (baulk semicircle) centered at brown spot
- Baulk line, middle line, and pocket positions
- Realistic colors and 3D ball effects

**Usage:**
```python
from src.snooker.table_generator import SnookerTableGenerator

generator = SnookerTableGenerator()
table = generator.create_table_with_balls()
generator.save_table_image("table.png")
```

### 2. Perspective Transformation (`perspective_transformer.py`)

Transforms video frames from arbitrary camera angles to standardized top-down view using manual point selection.

**Features:**
- Interactive point selection using OpenCV
- Consistent transformation across video frames
- Validation against reference measurements
- Configuration save/load functionality

**Usage:**
```python
from src.snooker.perspective_transformer import PerspectiveTransformer

transformer = PerspectiveTransformer()
success = transformer.setup_transformation(first_frame)
if success:
    transformed = transformer.transform_frame(frame)
```

### 3. Reference Validation (`reference_validator.py`)

Validates perspective transformation quality against expected snooker table measurements.

**Features:**
- Dimension validation
- Baulk line position checking
- Table color analysis
- Comprehensive validation reports

## Table Constants (`table_constants.py`)

Defines standard snooker table dimensions and positions:

- **Image Dimensions**: 1830x3660 pixels
- **Playing Area**: 1620x3450 pixels (green surface)
- **Ball Positions**: Standard spots for all colored balls
- **Pocket Regions**: Corner and middle pocket dimensions
- **Line Positions**: Baulk line, middle line coordinates

## Testing Scripts



### Test with Real Images
```bash
# List available test images
python scripts/snooker/test_with_image.py --list

# Test with specific image
python scripts/snooker/test_with_image.py --image assets/test_images/230.jpg
```

### Test with Real Videos
```bash
# List available test videos
python scripts/snooker/test_with_real_video.py --list

# Test with specific video
python scripts/snooker/test_with_real_video.py --video assets/test_videos/hires_video_65s.mp4
```

### Demo with Reference Image
```bash
python scripts/snooker/setup_with_reference.py your_video.mp4 --reference path/to/reference.png
```

## Output Structure

All outputs are saved to `assets/output/snooker/` with the following structure:

```
assets/output/snooker/
├── base_table.png                    # Generated table without balls
├── table_with_balls.png             # Table with standard ball positions
├── demo/                            # Demo outputs
├── reference/                       # Reference-guided setup outputs
├── video_test/                      # Video test outputs
│   ├── original_frame.png
│   ├── transformed_frame.png
│   ├── transformation_config.json
│   ├── transformation_visualization.png
│   └── validation/
│       ├── validation_visualization.png
│       └── validation_report.json
└── image_test/                      # Image test outputs
    ├── original_image.png
    ├── transformed_image.png
    ├── comparison_with_reference.png
    └── validation/
```

## Manual Point Selection Guide

When setting up perspective transformation, you'll be asked to select 4 corner points:

1. **Top-left corner**: Click on the top-left corner of the green playing surface
2. **Top-right corner**: Click on the top-right corner of the green playing surface
3. **Bottom-right corner**: Click on the bottom-right corner of the green playing surface
4. **Bottom-left corner**: Click on the bottom-left corner of the green playing surface

**Important Notes:**
- Select corners of the **green playing area**, not the outer cushions
- Click points in **clockwise order** starting from top-left
- Use the reference table image as a guide for accurate selection
- Press 'r' to reset points, 'q' to quit, ENTER to confirm

## Validation Metrics

The system validates transformation quality using:

- **Dimension Ratio**: Checks if transformed dimensions match expected ratios
- **Aspect Ratio Error**: Validates table proportions
- **Baulk Line Position**: Verifies correct line placement
- **Color Analysis**: Confirms green table surface detection

A transformation passes validation if at least 2/3 of these metrics are satisfied.

## Dependencies

- OpenCV (cv2)
- NumPy
- Pathlib
- JSON (for configuration)
- Logging

## Future Components

The following components are planned for implementation:

- **ColorBasedBallDetector**: Ball detection using computer vision
- **SnookerBallDetector**: Rule-based detection sanitization
- **BallTracker**: Multi-object tracking using ByteTrack
- **EventAnalyzer**: Ball event detection (potting, collisions)
- **ScoringEngine**: Snooker rules implementation

## Contributing

When adding new components:

1. Follow the existing interface patterns
2. Add comprehensive logging
3. Include validation and testing
4. Output debug artifacts for troubleshooting
5. Update this README with new functionality

## D-Shape Implementation

The D-shape (baulk semicircle) has been correctly implemented:

- **Center**: Brown spot position (912, 824)
- **Radius**: Distance between brown and yellow spots (266 pixels)  
- **Arc**: Lower semicircle (180° to 360°)