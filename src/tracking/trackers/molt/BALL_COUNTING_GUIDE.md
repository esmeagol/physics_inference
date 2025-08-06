# MOLT Ball Counting Logic Documentation

## Overview

The MOLT tracker includes sophisticated ball counting and verification logic specifically designed for snooker and billiards games. This system continuously monitors the number of tracked balls of each type and compares them against expected counts, automatically detecting and handling violations.

## Table of Contents

1. [Ball Counting Fundamentals](#ball-counting-fundamentals)
2. [Expected Ball Counts](#expected-ball-counts)
3. [Count Verification Process](#count-verification-process)
4. [Violation Types and Handling](#violation-types-and-handling)
5. [API Reference](#api-reference)
6. [Configuration Examples](#configuration-examples)
7. [Troubleshooting](#troubleshooting)

## Ball Counting Fundamentals

### Purpose

Ball counting serves several critical functions in cue sports tracking:

1. **Accuracy Validation**: Ensures the tracker maintains the correct number of balls
2. **Error Detection**: Identifies when balls are lost or incorrectly duplicated
3. **Track Management**: Suggests track merging or reassignment to resolve violations
4. **Game State Awareness**: Adapts to changing ball counts as balls are potted

### Core Principles

- **Expected vs. Current**: Continuously compares current tracked balls against expected counts
- **Violation Detection**: Identifies over-count and under-count scenarios
- **Automatic Recovery**: Attempts to resolve violations through track management
- **Statistical Tracking**: Maintains history of violations and recovery attempts

## Expected Ball Counts

### Snooker Configuration

Standard snooker uses 22 balls total:

```python
expected_counts = {
    'white': 1,    # Cue ball (always present)
    'red': 15,     # Red balls (reduced as potted)
    'yellow': 1,   # Yellow ball (2 points)
    'green': 1,    # Green ball (3 points)
    'brown': 1,    # Brown ball (4 points)
    'blue': 1,     # Blue ball (5 points)
    'pink': 1,     # Pink ball (6 points)
    'black': 1     # Black ball (7 points)
}
```

### Pool/8-Ball Configuration

Standard 8-ball pool uses 16 balls total:

```python
expected_counts = {
    'white': 1,    # Cue ball
    'yellow': 7,   # Solid balls (1-7) or stripes (9-15)
    'red': 7,      # Solid balls (1-7) or stripes (9-15)
    'black': 1     # 8-ball
}
```

### Dynamic Count Updates

Ball counts can be updated during play to reflect game state:

```python
# Initial snooker setup
tracker.ball_count_manager.update_expected_counts({
    'white': 1, 'red': 15, 'yellow': 1, 'green': 1,
    'brown': 1, 'blue': 1, 'pink': 1, 'black': 1
})

# After 5 red balls are potted
tracker.ball_count_manager.update_expected_counts({
    'white': 1, 'red': 10, 'yellow': 1, 'green': 1,
    'brown': 1, 'blue': 1, 'pink': 1, 'black': 1
})
```

## Count Verification Process

### Verification Workflow

The ball count verification process runs automatically during each tracking update:

1. **Count Collection**: Extract current ball counts from active tracks
2. **Comparison**: Compare current counts against expected counts
3. **Violation Detection**: Identify discrepancies (over-count or under-count)
4. **Violation Logging**: Record violations with timestamps and details
5. **Recovery Attempts**: Suggest or perform corrective actions

### Verification Triggers

Count verification is triggered:
- After each tracking update
- When new detections are incorporated
- When tracks are created or destroyed
- When expected counts are updated

### Verification Results

```python
# Check if counts are valid
counts_valid = tracker.ball_count_manager.verify_counts()

if not counts_valid:
    # Get detailed violation information
    violations = tracker.ball_count_manager.get_count_violations()
    for ball_class, violation_info in violations.items():
        print(f"{ball_class}: {violation_info['violation_type']}")
        print(f"  Expected: {violation_info['expected']}")
        print(f"  Current: {violation_info['current']}")
        print(f"  Difference: {violation_info['difference']}")
```

## Violation Types and Handling

### Over-Count Violations

**Definition**: More balls of a type are tracked than expected.

**Common Causes**:
- False positive detections
- Track fragmentation (one ball tracked as multiple)
- Misclassification of ball colors
- Reflection or shadow artifacts

**Handling Strategy**:
1. **Spatial Analysis**: Calculate distances between balls of the same type
2. **Merge Suggestions**: Identify closest ball pairs for potential merging
3. **Confidence Ranking**: Prioritize tracks with higher confidence scores
4. **Automatic Merging**: Merge tracks that are spatially close and have similar appearance

**Example**:
```python
# Handle over-count violation
ball_class = 'red'
track_ids = [1, 2, 3, 4]  # 4 red balls detected, but only 3 expected

excess_tracks = tracker.ball_count_manager.handle_duplicate_ball(ball_class, track_ids)
print(f"Excess tracks to merge: {excess_tracks}")

# Get merge suggestions
merge_suggestions = tracker.ball_count_manager.suggest_track_merges(current_tracks)
for track_id1, track_id2 in merge_suggestions:
    print(f"Suggest merging tracks {track_id1} and {track_id2}")
```

### Under-Count Violations

**Definition**: Fewer balls of a type are tracked than expected.

**Common Causes**:
- Balls temporarily occluded or out of frame
- Poor lighting or contrast
- Track loss due to rapid movement
- Ball potting (legitimate count reduction)

**Handling Strategy**:
1. **Recovery Attempts**: Increase exploration radius for missing ball types
2. **Track Reassignment**: Check if other tracks might be misclassified
3. **Confidence Adjustment**: Lower confidence thresholds for missing ball types
4. **Historical Analysis**: Review recent track history for recovery clues

**Example**:
```python
# Handle under-count violation
ball_class = 'yellow'
suggested_track = tracker.ball_count_manager.handle_lost_ball(ball_class)

if suggested_track is not None:
    print(f"Suggested track for recovery: {suggested_track}")
else:
    print(f"No recovery suggestion for {ball_class} ball")
```

### Violation Statistics

The system maintains comprehensive statistics about violations:

```python
stats = tracker.ball_count_manager.get_statistics()

print(f"Total violations: {stats['total_count_violations']}")
print(f"Lost ball recoveries: {stats['lost_ball_recoveries']}")
print(f"Duplicate ball merges: {stats['duplicate_ball_merges']}")
print(f"Recent violations: {stats['recent_violations']}")
```

## API Reference

### BallCountManager Class

#### Constructor

```python
BallCountManager(expected_counts: Dict[str, int])
```

**Parameters**:
- `expected_counts`: Dictionary mapping ball colors to expected counts

#### Key Methods

##### update_counts_from_tracks(tracks)

Update current ball counts from tracking results.

```python
tracks = tracker.update(frame)
tracker.ball_count_manager.update_counts_from_tracks(tracks)
```

##### verify_counts()

Check if current counts match expected counts.

```python
is_valid = tracker.ball_count_manager.verify_counts()
```

##### get_count_violations()

Get detailed information about current violations.

```python
violations = tracker.ball_count_manager.get_count_violations()
```

##### handle_lost_ball(ball_class)

Handle under-count violations for a specific ball type.

```python
suggestion = tracker.ball_count_manager.handle_lost_ball('red')
```

##### handle_duplicate_ball(ball_class, track_ids)

Handle over-count violations for a specific ball type.

```python
excess_tracks = tracker.ball_count_manager.handle_duplicate_ball('red', [1, 2, 3])
```

##### suggest_track_merges(tracks)

Suggest track pairs for merging to resolve violations.

```python
merge_suggestions = tracker.ball_count_manager.suggest_track_merges(tracks)
```

##### get_statistics()

Get comprehensive statistics about ball count management.

```python
stats = tracker.ball_count_manager.get_statistics()
```

##### update_expected_counts(new_counts)

Update expected ball counts (e.g., when balls are potted).

```python
tracker.ball_count_manager.update_expected_counts({'red': 10, 'white': 1})
```

##### reset()

Reset all counts and statistics while preserving configuration.

```python
tracker.ball_count_manager.reset()
```

## Configuration Examples

### Basic Snooker Setup

```python
from tracking.trackers.molt import MOLTTracker, MOLTTrackerConfig

# Create snooker configuration
config = MOLTTrackerConfig.create_for_snooker()
tracker = MOLTTracker(config=config)

# Verify expected counts
print("Expected ball counts:", config.expected_ball_counts)
```

### Custom Game Setup

```python
# Custom game with specific ball counts
config = MOLTTrackerConfig()
config.expected_ball_counts = {
    'white': 1,
    'red': 10,     # Custom red ball count
    'blue': 2,     # Custom blue ball count
    'yellow': 1
}

tracker = MOLTTracker(config=config)
```

### Dynamic Count Management

```python
# Start with full snooker setup
tracker = MOLTTracker()
success = tracker.init(frame, initial_detections)

# Process several frames
for frame in frames[:50]:
    tracks = tracker.update(frame)

# Simulate potting 3 red balls
new_counts = tracker.expected_ball_counts.copy()
new_counts['red'] = 12  # Reduced from 15
tracker.ball_count_manager.update_expected_counts(new_counts)

# Continue tracking with updated counts
for frame in frames[50:]:
    tracks = tracker.update(frame)
```

### Violation Monitoring

```python
# Set up violation monitoring
def monitor_violations(tracker):
    stats = tracker.ball_count_manager.get_statistics()
    
    if stats['total_count_violations'] > 0:
        print(f"⚠️  Total violations: {stats['total_count_violations']}")
        
        violations = tracker.ball_count_manager.get_count_violations()
        for ball_class, info in violations.items():
            if info['violation_type'] != 'none':
                print(f"  {ball_class}: {info['violation_type']} "
                      f"(expected {info['expected']}, got {info['current']})")

# Use in tracking loop
for frame in video_frames:
    tracks = tracker.update(frame)
    monitor_violations(tracker)
```

## Troubleshooting

### Common Issues

#### High Violation Rates

**Symptoms**: Frequent count violations, unstable tracking

**Causes**:
- Poor detection quality
- Incorrect expected counts
- Suboptimal tracker parameters

**Solutions**:
```python
# Adjust confidence thresholds
config.min_confidence = 0.2  # Lower threshold

# Increase population sizes for better accuracy
config.population_sizes = {'red': 400, 'white': 1800}

# Adjust similarity weights to favor appearance
config.similarity_weights = {'histogram': 0.8, 'spatial': 0.2}
```

#### False Positive Merges

**Symptoms**: Legitimate separate balls being merged

**Causes**:
- Overly aggressive merge suggestions
- Similar ball appearances
- Close spatial proximity

**Solutions**:
```python
# Increase minimum merge distance
# (This would require custom implementation)

# Improve ball classification
config.histogram_bins = 24  # More detailed histograms
config.color_space = 'LAB'  # Better color discrimination
```

#### Missed Ball Recovery

**Symptoms**: Lost balls not being recovered

**Causes**:
- Insufficient exploration radius
- Poor track history
- Incorrect ball classification

**Solutions**:
```python
# Increase exploration radii
config.exploration_radii = {'white': 40, 'red': 30, 'default': 25}

# Adjust population sizes
config.population_sizes = {'red': 500, 'white': 2000}
```

### Debug Information

#### Detailed Violation Analysis

```python
def analyze_violations(tracker):
    violations = tracker.ball_count_manager.get_count_violations()
    
    for ball_class, info in violations.items():
        print(f"\n{ball_class.upper()} BALL ANALYSIS:")
        print(f"  Expected: {info['expected']}")
        print(f"  Current: {info['current']}")
        print(f"  Difference: {info['difference']}")
        print(f"  Violation Type: {info['violation_type']}")
        
        if info['violation_type'] == 'over_count':
            # Analyze spatial distribution
            class_tracks = [t for t in current_tracks if t.get('class') == ball_class]
            if len(class_tracks) > 1:
                distances = []
                for i in range(len(class_tracks)):
                    for j in range(i+1, len(class_tracks)):
                        t1, t2 = class_tracks[i], class_tracks[j]
                        dist = np.sqrt((t1['x'] - t2['x'])**2 + (t1['y'] - t2['y'])**2)
                        distances.append(dist)
                
                print(f"  Minimum distance between {ball_class} balls: {min(distances):.1f}")
                print(f"  Average distance: {np.mean(distances):.1f}")
```

#### Track Assignment Analysis

```python
def analyze_track_assignments(tracker):
    assignments = tracker.ball_count_manager.track_assignments
    
    print("CURRENT TRACK ASSIGNMENTS:")
    for track_id, ball_class in assignments.items():
        print(f"  Track {track_id}: {ball_class}")
    
    # Group by ball class
    by_class = {}
    for track_id, ball_class in assignments.items():
        if ball_class not in by_class:
            by_class[ball_class] = []
        by_class[ball_class].append(track_id)
    
    print("\nTRACKS BY BALL CLASS:")
    for ball_class, track_ids in by_class.items():
        print(f"  {ball_class}: {track_ids} (count: {len(track_ids)})")
```

This comprehensive ball counting documentation covers all aspects of the MOLT tracker's ball counting and verification system, providing both theoretical understanding and practical implementation guidance.