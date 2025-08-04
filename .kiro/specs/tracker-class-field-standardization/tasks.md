# Implementation Plan

- [x] 1. Fix DeepSORT tracker to return 'class' field instead of 'class_name'

  - Modify the `_update_trackers` method in `CVModelInference/trackers/deepsort_tracker.py`
  - Change class info copying logic to map 'class_name' to 'class' in returned track dictionaries
  - _Requirements: 1.1, 2.2_

- [x] 2. Remove fallback logic from analysis tools

  - Update `CVModelInference/scripts/compare_trackers.py` to only use 'class' field
  - Update `CVModelInference/tracker_benchmark.py` to only use 'class' field
  - Replace all instances of `track.get('class', track.get('class_name', 'N/A'))` with `track.get('class', 'unknown')`
  - _Requirements: 1.2, 2.1, 2.3_

- [x] 3. Test the fix with existing tracking data
  - Run the tracker benchmark with the snooker video to verify class information appears correctly
  - Verify tracking analysis shows actual class names instead of 'N/A'
  - _Requirements: 1.3_
