# Design Document

## Overview

This design standardizes class field naming across the tracking system by using `'class'` everywhere and removing all `'class_name'` references. The fix involves updating the DeepSORT tracker to return `'class'` instead of `'class_name'` and removing fallback logic in analysis tools.

## Architecture

The tracking pipeline has these key components that handle class information:

1. **DeepSORT Tracker** - Returns track dictionaries with class info
2. **Analysis Tools** - Process tracking results and generate reports  
3. **Visualization Tools** - Display class labels in videos and plots

Currently, the DeepSORT tracker stores class info as `'class_name'` internally but should return `'class'` in track dictionaries. Analysis tools have fallback logic that tries both `'class'` and `'class_name'` fields.

## Components and Interfaces

### DeepSORT Tracker Changes

The `_update_trackers` method currently copies class info from internal `class_info` dict to track results. We need to ensure it uses `'class'` as the key name.

**Current behavior:**
```python
# Copies class_info keys directly, including 'class_name'
for key, value in tracker['class_info'].items():
    track[key] = value
```

**New behavior:**
```python
# Map class_name to class for consistency
if 'class_name' in tracker['class_info']:
    track['class'] = tracker['class_info']['class_name']
# Copy other fields as-is
for key, value in tracker['class_info'].items():
    if key != 'class_name':
        track[key] = value
```

### Analysis Tools Changes

Remove all fallback logic that tries `'class_name'` after `'class'`. Update to only use `'class'` field.

**Current pattern:**
```python
track.get('class', track.get('class_name', 'N/A'))
```

**New pattern:**
```python
track.get('class', 'unknown')
```

## Data Models

### Track Dictionary Format

**Before:**
```python
{
    'id': int,
    'x': float, 'y': float, 'width': float, 'height': float,
    'class_name': str,  # Inconsistent field name
    'class_id': int,
    'confidence': float
}
```

**After:**
```python
{
    'id': int,
    'x': float, 'y': float, 'width': float, 'height': float,
    'class': str,       # Standardized field name
    'class_id': int,
    'confidence': float
}
```

## Error Handling

No special error handling needed. If `'class'` field is missing, analysis tools will use default value 'unknown' instead of 'N/A'.

## Testing Strategy

1. **Unit Test**: Verify DeepSORT tracker returns `'class'` field in track dictionaries
2. **Integration Test**: Run tracking analysis and verify class names appear correctly (not 'N/A')
3. **Regression Test**: Ensure existing tracking functionality still works after changes