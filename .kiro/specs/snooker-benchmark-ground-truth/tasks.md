# Implementation Plan

- [x] 1. Create EventProcessor class for ground truth event handling

  - Implement flexible event parsing supporting both single time and time range formats
  - Add support for time-based and frame-based events with automatic conversion
  - Create time range parsing for all event types (potting, placement, occlusion, error suppression)
  - Add validation that potting/placement events can use either single time or time range
  - Add chronological event sorting and filtering methods
  - Write unit tests for event processing functionality with mixed time formats
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement StateReconstructor class for ball state management

  - [x] 2.1 Create core state reconstruction functionality

    - Implement initial state setup from first event
    - Add flexible event parsing supporting both single time and time range formats
    - Create state interpolation between events with before/during/after logic
    - Add validation for impossible ball states
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.8_

  - [x] 2.2 Add time range event processing for potting and placement

    - Implement potting event logic with before/during/after state expectations
    - Add placement event logic with time range support
    - Create transition state handling during event time ranges
    - Add validation that n balls exist before potting, n-1 or n during, n-1 after
    - _Requirements: 3.3, 3.4, 3.7_

  - [x] 2.3 Add occlusion and error suppression handling
    - Implement occlusion event processing to temporarily hide balls
    - Add error suppression event handling for ignore_errors ranges
    - Create time range application for duration-based events
    - Write unit tests for state reconstruction functionality
    - _Requirements: 3.5, 3.6, 3.7_

- [x] 3. Implement MomentEvaluator class for tracker output evaluation

  - [x] 3.1 Create sophisticated count comparison evaluation

    - Implement expected vs detected ball count comparison with time range awareness
    - Add before/during/after evaluation logic for potting and placement events
    - Create flexible count validation allowing n-1 or n balls during transition periods
    - Add over-detection and under-detection error identification with context
    - Create per-ball-type accuracy calculation with time range considerations
    - Add error magnitude calculation and reporting with event context
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 3.2 Implement illegal change detection

    - Add ball disappearance detection without corresponding events
    - Implement ball reappearance detection without corresponding events
    - Create tracking continuity analysis between moments
    - Add distinction between expected game events and tracking failures
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 3.3 Add duplication and spatial analysis
    - Implement spatial distribution analysis for detected objects
    - Add duplicate object detection using distance thresholds
    - Create missing object detection in expected regions
    - Write unit tests for moment evaluation functionality
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 4. Create GroundTruthEvaluator orchestration class

  - [x] 4.1 Implement core evaluator functionality

    - Create evaluator initialization with events and video parameters
    - Add moment-based evaluation coordination with time range awareness
    - Implement state reconstruction for specific moments with before/during/after logic
    - Add tracker output processing and aggregation with transition period handling
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 4.2 Add comprehensive evaluation reporting with time range context
    - Create detailed evaluation report generation including transition period analysis
    - Add per-ball-type accuracy metrics calculation with event context
    - Implement temporal error analysis and trend detection across event ranges
    - Add error categorization distinguishing between stable periods and transitions
    - Write integration tests for complete evaluation pipeline with mixed event formats
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

- [x] 5. Enhance SnookerTrackerBenchmark with ground truth capabilities

  - [x] 5.1 Extend benchmark class with ground truth methods

    - Add set_ground_truth_events method for event configuration
    - Implement set_moment_duration method for temporal granularity
    - Create run_benchmark_with_ground_truth method
    - Add backward compatibility preservation for existing functionality
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 5.2 Integrate ground truth evaluation into benchmark pipeline
    - Add ground truth evaluation to existing benchmark workflow
    - Implement moment grouping from tracker output frames
    - Create ground truth metrics integration with existing results
    - Add error suppression application during evaluation
    - _Requirements: 4.5, 8.4_

- [x] 6. Implement enhanced visualization and reporting

  - [x] 6.1 Create ground truth visualization components

    - Add ground truth timeline visualization with event markers
    - Implement accuracy over time plotting functionality
    - Create error distribution visualization charts
    - Add comparative analysis plots for multiple trackers
    - _Requirements: 7.5_

  - [x] 6.2 Enhance result reporting methods
    - Update print_results method to include ground truth metrics
    - Enhance visualize_results method with ground truth plots
    - Add comprehensive report generation with recommendations
    - Create detailed error analysis and breakdown reporting
    - Write integration tests for visualization and reporting
    - _Requirements: 8.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 7. Create comprehensive test suite and validation

  - [ ] 7.1 Write unit tests for all components

    - Create EventProcessor unit tests for parsing and validation
    - Add StateReconstructor unit tests for state management
    - Implement MomentEvaluator unit tests for evaluation logic
    - Add GroundTruthEvaluator unit tests for orchestration
    - _Requirements: All component requirements_

  - [ ] 7.2 Create integration tests and examples
    - Write end-to-end benchmark tests with sample ground truth
    - Create example usage scripts demonstrating all features
    - Add performance tests for scalability validation
    - Implement edge case handling tests for robustness
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 8. Define and implement comprehensive evaluator output format

  - [ ] 8.1 Design detailed evaluation output structure

    - Create moment-by-moment evaluation results with timestamps
    - Define count error reporting with expected vs detected counts
    - Add illegal change detection results with event context
    - Design duplication error reporting with spatial information
    - Create summary statistics and accuracy metrics structure
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

  - [ ] 8.2 Implement evaluation result generation and formatting
    - Create structured output with nested dictionaries for different error types
    - Add human-readable error descriptions and recommendations
    - Implement CSV and JSON export formats for analysis
    - Add visualization-ready data structures for plotting
    - Write comprehensive examples showing expected output format
    - _Requirements: 7.5, 7.6_

- [ ] 9. Add error handling and performance optimization

  - [ ] 9.1 Implement comprehensive error handling

    - Add event validation with detailed error messages
    - Implement graceful handling of malformed events
    - Create fallback mechanisms for missing data
    - Add logging and debugging support throughout system
    - _Requirements: All error handling aspects_

  - [ ] 9.2 Optimize performance for large datasets
    - Implement lazy state reconstruction for efficiency
    - Add streaming evaluation for memory optimization
    - Create batch processing for moment evaluations
    - Add progress reporting for long-running evaluations
    - _Requirements: Performance and scalability aspects_

- [ ] 10. Create documentation and update existing scripts

  - [ ] 10.1 Write comprehensive documentation

    - Create detailed API documentation for all classes
    - Add usage guide with ground truth event format examples showing both single time and time range formats
    - Document before/during/after evaluation logic for potting and placement events
    - Write troubleshooting guide for common issues
    - Create performance tuning recommendations
    - _Requirements: All requirements for usability_

  - [x] 10.2 Update compare_trackers.py script with ground truth evaluation
    - Add command line arguments for ground truth events file input (--ground-truth, --moment-duration)
    - Integrate ground truth evaluation into existing benchmark workflow
    - Add example ground truth events with mixed time formats in script comments
    - Update output to include ground truth metrics alongside existing metrics
    - Add example showing potting event with time range: expect n balls before, n-1 or n during, n-1 after
    - Implement comprehensive ground truth evaluation demonstration using real snooker video
    - _Requirements: All requirements for practical usage_
