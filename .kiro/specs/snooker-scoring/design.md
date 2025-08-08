# Design Document

## Overview

The snooker scoring module implements a pipeline architecture that processes video input through sequential stages: perspective transformation, ball detection, ball tracking, event analysis, and scoring. The design leverages existing detection infrastructure while adding specialized components for snooker-specific processing.

The system follows a modular approach where each component has a clear interface and can be tested independently. Components are designed to be interchangeable, allowing for different detection methods (ML-based vs color-based) to be used together or separately.

## Architecture

```
Video Input → PerspectiveTransformer → ColorBasedBallDetector → SnookerBallDetector → BallTracker → EventAnalyzer → ScoringEngine → Score Output
```

### Core Pipeline Components

1. **PerspectiveTransformer**: Converts table view to standardized top-down perspective
2. **ColorBasedBallDetector**: Detects balls and classifies colors using computer vision techniques
3. **SnookerBallDetector**: Sanitizes detection results using snooker rules and temporal smoothing
4. **BallTracker**: Maintains ball identities across frames
5. **EventAnalyzer**: Analyzes tracking data to detect potting, collisions, and respotting
6. **ScoringEngine**: Applies snooker rules to calculate scores from events

### Data Flow

- Raw video frames flow through the perspective transformer
- Transformed frames are processed by the ball detector
- Detection results feed into the ball tracker
- Tracking data is analyzed for events
- Events are processed by the scoring engine to produce match scores

## Components and Interfaces

### PerspectiveTransformer

```python
class PerspectiveTransformer:
    def __init__(self, transformation_points: Optional[List[Tuple[int, int]]] = None):
        """Initialize with optional pre-defined transformation points"""
        
    def setup_transformation(self, frame: np.ndarray) -> bool:
        """Setup transformation using manual point selection on first frame"""
        
    def transform_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Transform frame to top-down table view using established matrix"""
        
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Get current perspective transformation matrix"""
```

**Responsibilities:**
- Use manual point selection (similar to trim_video_to_table.py) for table boundary detection
- Allow user to click 4 corner points on the first frame to define table boundaries
- Calculate perspective transformation matrix from selected points
- Apply consistent transformation to all subsequent frames
- Provide fallback to automatic detection if manual selection fails

### BallDetector

```python
class ColorBasedBallDetector(InferenceRunner):
    def __init__(self, confidence: float = 0.5):
        """Initialize color-based ball detector"""
        
    def predict(self, image: Union[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Detect balls using color-based computer vision"""
        
    def _detect_balls_by_color(self, image: np.ndarray) -> List[Dict]:
        """Core color-based detection logic from notebook"""
```

**Responsibilities:**
- Implement color-based ball detection using logic from TrackingSnookerBalls.ipynb
- Maintain consistent interface with existing ML-based detectors
- Filter contours by size and shape to identify balls
- Classify ball colors using HSV color space analysis
- Return standardized detection format

### BallTracker

The system will leverage the existing `src/tracking/tracking.py` module which already implements ByteTrack-based tracking. The existing `Tracker` class provides:

```python
# Existing Tracker class from src/tracking/tracking.py
tracker = Tracker(model_path="dummy", confidence=0.5)  # Will be adapted for ball detection
tracking_results = tracker.process_frame(frame)
```

**Responsibilities:**
- Use existing supervision's ByteTrack implementation from tracking module
- Adapt existing tracker to work with ball detection results
- Maintain ball identities across frames using established tracking infrastructure
- Convert between detection formats as needed
- Provide track IDs and trajectories using existing annotation system

### EventAnalyzer

```python
class EventAnalyzer:
    def __init__(self, pocket_regions: List[Tuple[int, int, int, int]]):
        """Initialize with pocket region definitions"""
        
    def analyze_frame(self, tracks: List[Dict]) -> List[Dict]:
        """Analyze tracking data for events"""
        
    def detect_potting_events(self, tracks: List[Dict]) -> List[Dict]:
        """Detect when balls are potted"""
```

**Responsibilities:**
- Detect ball potting events based on proximity to pockets and disappearance
- Identify ball collisions using trajectory analysis
- Detect respotting events when balls reappear
- Classify event types and confidence levels

### ScoringEngine

```python
class ScoringEngine:
    def __init__(self):
        """Initialize scoring engine with snooker rules"""
        
    def process_events(self, events: List[Dict]) -> Dict[str, Any]:
        """Process events and update scores"""
        
    def get_current_score(self) -> Dict[str, int]:
        """Get current match score"""
```

**Responsibilities:**
- Apply snooker scoring rules to events
- Track game state (which balls are on table, current player, etc.)
- Calculate points for potted balls
- Handle fouls and penalty points
- Determine frame and match winners

### SnookerScoringPipeline

```python
class SnookerScoringPipeline:
    def __init__(self, ball_detector: BallDetector):
        """Initialize complete pipeline"""
        
    def setup_table_transformation(self, first_frame: np.ndarray) -> bool:
        """Setup table transformation using manual point selection"""
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process entire video and return scoring results"""
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame through pipeline"""
```

**Responsibilities:**
- Orchestrate the complete processing pipeline
- Handle manual table setup on first frame
- Handle video input/output
- Coordinate between components
- Provide unified interface for video processing

## Data Models

### Detection Format
```python
{
    'x': float,           # Center x coordinate
    'y': float,           # Center y coordinate  
    'width': float,       # Bounding box width
    'height': float,      # Bounding box height
    'confidence': float,  # Detection confidence
    'class': str,         # Ball color (red, yellow, green, brown, blue, pink, black, white)
    'class_id': int       # Numeric class identifier
}
```

### Track Format
```python
{
    'track_id': int,      # Unique track identifier
    'x': float,           # Current x position
    'y': float,           # Current y position
    'velocity_x': float,  # X velocity
    'velocity_y': float,  # Y velocity
    'class': str,         # Ball color
    'confidence': float,  # Tracking confidence
    'age': int           # Frames since track started
}
```

### Event Format
```python
{
    'type': str,          # 'potted', 'collision', 'respotted'
    'timestamp': float,   # Frame timestamp
    'ball_color': str,    # Affected ball color
    'position': Tuple[float, float],  # Event position
    'confidence': float   # Event confidence
}
```

### Score Format
```python
{
    'player1_score': int,
    'player2_score': int,
    'current_player': int,
    'frame_score': Dict[str, int],
    'events': List[Dict],
    'game_state': str     # 'reds', 'colors', 'finished'
}
```

## Error Handling

### Graceful Degradation
- When table detection fails, skip frame and continue processing
- When ball detection quality is poor, use previous frame data
- When tracking loses balls, attempt re-identification in subsequent frames
- When events are ambiguous, log uncertainty and make conservative decisions

### Error Recovery
- Maintain transformation matrix from previous successful frames
- Use temporal smoothing for ball positions when detection is noisy
- Implement track re-identification when balls temporarily disappear
- Provide fallback scoring logic for unclear situations

## Testing Strategy

### Unit Tests
- Test each component independently with known inputs
- Verify color-based detection accuracy with synthetic images
- Test tracking algorithm with simulated ball movements
- Validate scoring logic with known game sequences

### Integration Tests
- Test complete pipeline with sample video clips
- Verify data flow between components
- Test error handling and recovery mechanisms
- Validate output format consistency

### Performance Tests
- Measure processing speed on different video qualities
- Test memory usage with long videos
- Verify real-time processing capabilities
- Benchmark against existing detection methods