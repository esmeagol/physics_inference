# Implementation Plan

- [x] 1. Create PerspectiveTransformer component with manual point selection

  - Implement interactive point selection using OpenCV mouse callbacks
  - Calculate perspective transformation matrix from 4 corner points
  - Apply consistent transformation to video frames
  - Output transformed video for debugging and verification
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 2. Implement ColorBasedBallDetector following existing detector interface

  - Create ColorBasedBallDetector class inheriting from InferenceRunner
  - Port ball detection logic from TrackingSnookerBalls.ipynb
  - Implement color-based ball classification using HSV analysis
  - Return standardized detection format compatible with existing infrastructure
  - Output annotated frames showing detected balls with bounding boxes and color labels
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 3. Create SnookerBallDetector sanitization layer

  - Implement stateful ball detection sanitizer that tracks game state
  - Enforce snooker ball count constraints (max 15 reds, 1 of each color)
  - Track expected ball counts based on game progression
  - Implement temporal smoothing across multiple frames (5-frame window)
  - Handle missing ball detection by checking for clustering or misclassification
  - Reject outlier detections that violate snooker rules
  - Add comprehensive logging for all sanitization decisions and rule applications
  - Log confidence adjustments, ball count violations, and temporal smoothing actions
  - Output sanitized detection results with confidence adjustments
  - Output debugging information showing rule violations and corrections
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 4. Create BallTracker adapter for existing tracking infrastructure

  - Adapt existing Tracker class to work with sanitized ball detection results
  - Implement conversion between detection formats
  - Integrate with supervision's ByteTrack through existing tracking module
  - Output tracking visualization with ball trajectories and track IDs
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 5. Implement EventAnalyzer for detecting ball events

  - Create EventAnalyzer class to process tracking data
  - Implement potting event detection based on ball disappearance near pockets
  - Detect collision events using ball proximity analysis
  - Identify respotting events when balls reappear
  - Output event timeline with timestamps and event types for debugging
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [ ] 6. Create ScoringEngine with snooker rules implementation

  - Implement basic snooker scoring rules for ball potting
  - Track game state (reds vs colors phase)
  - Calculate points for different colored balls
  - Handle basic foul detection and penalty scoring
  - Output detailed scoring log with frame-by-frame score changes
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 7. Build SnookerScoringPipeline orchestrator

  - Create main pipeline class coordinating all components
  - Implement video processing workflow with sanitization layer
  - Handle manual table setup on first frame
  - Integrate all components with proper data flow
  - Output intermediate results from each pipeline stage for debugging
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 8. Create command-line interface script

  - Implement main script for video processing
  - Add argument parsing for input/output paths and parameters
  - Add debug mode flag to enable intermediate artifact outputs
  - Provide user feedback during processing
  - Output scoring results in structured format (JSON/CSV)
  - _Requirements: 1.1, 1.5_

- [ ] 9. Write comprehensive tests for core functionality
  - Test PerspectiveTransformer with known transformation matrices
  - Test ColorBasedBallDetector with synthetic ball images
  - Write extensive tests for SnookerBallDetector covering all sanitization scenarios:
    - Test ball count constraint enforcement (too many/few reds, multiple colored balls)
    - Test temporal smoothing with missing balls in single frames
    - Test game state progression (reds phase to colors phase)
    - Test clustering detection and ball separation logic
    - Test misclassification correction (red detected as color, vice versa)
    - Test confidence adjustment algorithms
    - Test edge cases (all balls potted, respotting scenarios)
  - Test EventAnalyzer with simulated tracking data
  - Test ScoringEngine with known game sequences
  - Test complete pipeline with sample video clips
  - _Requirements: All requirements validation_
