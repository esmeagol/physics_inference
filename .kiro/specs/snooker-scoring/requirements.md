# Requirements Document

## Introduction

This document outlines the requirements for a snooker scoring module that processes video input to automatically track and score snooker matches. The system will build a pipeline of iterative processing components that transform raw video into match scoring data through perspective transformation, ball detection, ball tracking, and scoring logic.

The module will leverage existing detection and tracking infrastructure while adding new components for perspective transformation, ball position analysis, and scoring calculation based on snooker rules.

## Requirements

### Requirement 1

**User Story:** As a snooker analyst, I want to input a video file and receive automated scoring data, so that I can analyze match performance without manual tracking.

#### Acceptance Criteria

1. WHEN a video file is provided as input THEN the system SHALL process the entire video and output scoring data
2. WHEN the video contains a snooker table THEN the system SHALL automatically detect and transform the table to a standardized top-down view
3. WHEN balls are visible on the table THEN the system SHALL detect and track their positions throughout the video
4. WHEN ball movements indicate scoring events THEN the system SHALL calculate and record the appropriate scores
5. WHEN processing is complete THEN the system SHALL output a comprehensive scoring report

### Requirement 2

**User Story:** As a developer, I want a modular perspective transformation component, so that I can standardize snooker table views for consistent ball detection.

#### Acceptance Criteria

1. WHEN a video frame contains a snooker table THEN the perspective transformer SHALL detect the table boundaries
2. WHEN table boundaries are detected THEN the system SHALL calculate the perspective transformation matrix
3. WHEN the transformation matrix is applied THEN the system SHALL produce a rectangular top-down view of the table
4. WHEN multiple frames are processed THEN the transformation SHALL remain consistent across frames
5. WHEN the table is partially occluded THEN the system SHALL maintain transformation stability using previous frame data
6. WHEN no table is detected THEN the system SHALL return an error status and skip the frame

### Requirement 3

**User Story:** As a developer, I want a ball detection component that works on transformed table views, so that I can accurately identify ball positions and colors.

#### Acceptance Criteria

1. WHEN a transformed table image is provided THEN the ball detector SHALL identify all visible balls
2. WHEN balls are detected THEN the system SHALL determine their precise center coordinates
3. WHEN balls are detected THEN the system SHALL classify their colors (red, yellow, green, brown, blue, pink, black, white)
4. WHEN ball detection confidence is below threshold THEN the system SHALL mark the detection as uncertain
5. WHEN multiple balls overlap THEN the system SHALL attempt to separate and identify individual balls
6. WHEN lighting conditions vary THEN the system SHALL adapt detection parameters automatically

### Requirement 4

**User Story:** As a developer, I want a ball tracking component, so that I can follow ball movements and identify when balls are potted.

#### Acceptance Criteria

1. WHEN balls are detected in consecutive frames THEN the tracker SHALL maintain consistent ball identities
2. WHEN a ball disappears from the table THEN the system SHALL determine if it was potted or temporarily occluded
3. WHEN a ball is potted THEN the system SHALL record the potting event with timestamp and ball color
4. WHEN balls collide THEN the system SHALL maintain tracking through the collision
5. WHEN the cue ball moves THEN the system SHALL track its path and final position
6. WHEN tracking confidence drops THEN the system SHALL flag uncertain tracking periods

### Requirement 5

**User Story:** As a developer, I want a ball event analyzer component, so that I can detect potting, collisions, and respotting events from tracking data.

#### Acceptance Criteria

1. WHEN a ball disappears from tracking THEN the event analyzer SHALL determine if it was potted or temporarily occluded
2. WHEN a ball disappears near a pocket THEN the system SHALL classify it as a potting event
3. WHEN a ball reappears on the table THEN the system SHALL classify it as a respotting event
4. WHEN two or more balls come into close proximity THEN the system SHALL detect and classify collision events
5. WHEN the cue ball stops moving THEN the system SHALL mark the end of a shot sequence
6. WHEN ball movements are erratic or inconsistent THEN the system SHALL flag uncertain event classifications
7. WHEN multiple events occur simultaneously THEN the system SHALL sequence them chronologically

### Requirement 6

**User Story:** As a snooker enthusiast, I want automatic scoring based on snooker rules, so that I can get accurate match scores without manual calculation.

#### Acceptance Criteria

1. WHEN a red ball is potted THEN the system SHALL award 1 point to the current player
2. WHEN a colored ball is potted in sequence THEN the system SHALL award the appropriate points (yellow=2, green=3, brown=4, blue=5, pink=6, black=7)
3. WHEN a foul occurs THEN the system SHALL deduct points and award penalty points to the opponent
4. WHEN all red balls are potted THEN the system SHALL enforce colored ball sequence (yellow, green, brown, blue, pink, black)
5. WHEN a frame ends THEN the system SHALL calculate the final frame score and determine the winner