# Requirements Document

## Introduction

This feature enhances the existing SnookerTrackerBenchmark with event-based ground truth evaluation capabilities. The enhancement allows for sparse, event-based ground truth annotations that describe game events (ball potting, placement, occlusion) at specific time points, enabling more accurate and meaningful evaluation of tracking performance in snooker videos.

The system introduces a "moment" concept for temporal granularity, reconstructs expected ball states from sparse events, and provides detailed evaluation metrics comparing tracker output against expected ground truth states. This approach is particularly valuable for snooker tracking where traditional dense ground truth annotation would be impractical due to video length and the complexity of ball interactions.

## Requirements

### Requirement 1

**User Story:** As a computer vision researcher, I want to provide sparse event-based ground truth annotations for snooker videos, so that I can evaluate tracker performance without requiring dense frame-by-frame annotations.

#### Acceptance Criteria

1. WHEN ground truth events are provided THEN the system SHALL accept a list of dictionaries representing game events
2. WHEN each event dictionary is processed THEN the system SHALL support time-based events with "time" key (seconds into video)
3. WHEN each event dictionary is processed THEN the system SHALL support frame-based events with "frame" key (frame number)
4. WHEN time range events are processed THEN the system SHALL support time ranges with "time_range" key (e.g., "120-125" for 5-second duration)
5. WHEN frame range events are processed THEN the system SHALL support frame ranges with "frame_range" key (e.g., "1000-1150" for frame range)
6. WHEN ball count events are processed THEN the system SHALL support initial ball counts with keys like "reds", "yellow", "green", "brown", "blue", "pink", "black"
7. WHEN ball action events are processed THEN the system SHALL support events like "ball_potted": "red", "ball_potted": "black", "ball_placed_back": "black"
8. WHEN occlusion events are processed THEN the system SHALL support "balls_occluded" events with specific ball counts that are temporarily hidden
9. WHEN error suppression is needed THEN the system SHALL support "ignore_errors" events to suppress evaluation during unclear video segments

### Requirement 2

**User Story:** As a computer vision researcher, I want to define temporal granularity for evaluation, so that I can group consecutive frames together and ignore outlier detection failures.

#### Acceptance Criteria

1. WHEN moment duration is configured THEN the system SHALL accept a moment parameter (e.g., 0.5 seconds)
2. WHEN video properties are known THEN the system SHALL calculate frames per moment based on video FPS and detection interval
3. WHEN tracker output is processed THEN the system SHALL group consecutive tracker outputs into moments
4. WHEN moments are created THEN the system SHALL aggregate detection counts across all frames within each moment
5. WHEN moment aggregation is performed THEN the system SHALL use majority voting or averaging to handle outlier frames

### Requirement 3

**User Story:** As a computer vision researcher, I want to reconstruct expected ball states from sparse events, so that I can determine what balls should be visible at any given time.

#### Acceptance Criteria

1. WHEN ground truth events are processed THEN the system SHALL create a function to reconstruct expected state for any moment
2. WHEN initial state is established THEN the system SHALL use the first event (time=0) to set initial ball counts
3. WHEN potting events are processed THEN the system SHALL decrease ball counts for "ball_potted" events
4. WHEN placement events are processed THEN the system SHALL increase ball counts for "ball_placed_back" events
5. WHEN occlusion events are processed THEN the system SHALL temporarily adjust visible ball counts without changing total counts
6. WHEN error suppression events are processed THEN the system SHALL mark time ranges where evaluation errors should be ignored
7. WHEN time range events are processed THEN the system SHALL apply event effects for the entire duration of the specified range
8. WHEN state interpolation is needed THEN the system SHALL maintain consistent state between events until the next event occurs

### Requirement 4

**User Story:** As a computer vision researcher, I want to compare detected object counts against expected counts, so that I can identify tracking errors and performance issues.

#### Acceptance Criteria

1. WHEN each moment is evaluated THEN the system SHALL compare detected counts of each ball type against expected counts
2. WHEN count mismatches are found THEN the system SHALL report moments with too many balls detected ("over-detection")
3. WHEN count mismatches are found THEN the system SHALL report moments with too few balls detected ("under-detection")
4. WHEN count errors are reported THEN the system SHALL specify the ball type, expected count, detected count, and error magnitude
5. WHEN ignore_errors events are active THEN the system SHALL suppress error reporting for the specified time ranges
6. WHEN evaluation results are generated THEN the system SHALL provide frame-level and moment-level error statistics

### Requirement 5

**User Story:** As a computer vision researcher, I want to detect illegal ball disappearances and reappearances, so that I can identify tracking failures that don't correspond to actual game events.

#### Acceptance Criteria

1. WHEN ball disappearances are detected THEN the system SHALL check if they correspond to annotated events
2. WHEN ball reappearances are detected THEN the system SHALL check if they correspond to annotated events
3. WHEN illegal disappearances are found THEN the system SHALL report moments where balls vanish without corresponding potting events
4. WHEN illegal reappearances are found THEN the system SHALL report moments where balls appear without corresponding placement events
5. WHEN tracking continuity is evaluated THEN the system SHALL distinguish between expected game events and tracking failures

### Requirement 6

**User Story:** As a computer vision researcher, I want to detect object duplication and missing objects within ball types, so that I can identify tracking ID management issues.

#### Acceptance Criteria

1. WHEN position data is available THEN the system SHALL analyze spatial distribution of detected objects
2. WHEN duplicate objects are suspected THEN the system SHALL identify moments where multiple objects of the same type are too close together
3. WHEN missing objects are suspected THEN the system SHALL identify moments where expected objects are not detected in their expected regions
4. WHEN spatial analysis is performed THEN the system SHALL use configurable distance thresholds for duplication detection
5. WHEN duplication reports are generated THEN the system SHALL provide object positions and confidence scores for analysis

### Requirement 7

**User Story:** As a computer vision researcher, I want comprehensive evaluation reports, so that I can analyze tracker performance and identify specific failure modes.

#### Acceptance Criteria

1. WHEN evaluation is complete THEN the system SHALL generate detailed reports with moment-by-moment analysis
2. WHEN error statistics are calculated THEN the system SHALL provide per-ball-type accuracy metrics
3. WHEN temporal analysis is performed THEN the system SHALL identify time periods with high error rates
4. WHEN failure modes are analyzed THEN the system SHALL categorize errors by type (over-detection, under-detection, illegal changes)
5. WHEN visualization is requested THEN the system SHALL generate plots showing error trends over time
6. WHEN summary statistics are calculated THEN the system SHALL provide overall accuracy scores and error distributions

### Requirement 8

**User Story:** As a computer vision researcher, I want the enhanced benchmark to integrate seamlessly with existing SnookerTrackerBenchmark, so that I can use ground truth evaluation alongside existing metrics.

#### Acceptance Criteria

1. WHEN ground truth is provided THEN the system SHALL extend SnookerTrackerBenchmark without breaking existing functionality
2. WHEN ground truth evaluation is enabled THEN the system SHALL add ground truth metrics to existing benchmark results
3. WHEN no ground truth is provided THEN the system SHALL function exactly as the original SnookerTrackerBenchmark
4. WHEN results are displayed THEN the system SHALL include ground truth metrics in print_results() and visualize_results()
5. WHEN backward compatibility is maintained THEN the system SHALL not require changes to existing benchmark usage patterns