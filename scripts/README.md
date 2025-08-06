# Scripts Directory

This directory contains utility scripts for the physics_inference project.

## Quality Assurance Scripts

### `run_all_checks.py` - Master Quality Check Script
Runs the complete quality assurance suite including tests and type checking.

```bash
# Run all checks
python scripts/run_all_checks.py

# Run only tests
python scripts/run_all_checks.py --tests-only

# Run only MyPy type checking
python scripts/run_all_checks.py --mypy-only
```

### `run_tests.py` - Comprehensive Test Suite
Runs all tests including unit tests, integration tests, and script functionality tests.

```bash
python scripts/run_tests.py
```

**Test Categories:**
- Import validation tests
- Unit tests (unittest discovery)
- Script functionality tests
- Integration tests

### `run_mypy.py` - MyPy Type Checking
Runs comprehensive MyPy type checking with different strictness levels for different components.

```bash
python scripts/run_mypy.py
```

**Check Categories:**
- Core tracking modules (strict checking)
- Detection modules (standard checking)
- Scripts (relaxed checking)
- Test files (minimal checking)
- Full project check

## Tracking Scripts

### `compare_trackers.py` - Tracker Benchmarking
Compare different tracking algorithms with optional ground truth evaluation.

```bash
# Basic comparison
python scripts/tracking/compare_trackers.py --video video.mp4 --model weights.pt --snooker

# With ground truth evaluation
python scripts/tracking/compare_trackers.py --video video.mp4 --ground-truth events.json --trackers deepsort sv-bytetrack
```

### `track_objects.py` - Object Tracking
Track objects in a video using YOLOv11.

```bash
python scripts/tracking/track_objects.py --video input.mp4 --model weights.pt --output tracked.mp4
```

### `molt_basic_usage.py` - MOLT Tracker Examples
Demonstrates basic usage of the MOLT tracker with examples.

```bash
python scripts/tracking/molt_basic_usage.py
```

### `test_molt_unified.py` - MOLT Integration Test
Comprehensive test of MOLT tracker integration.

```bash
python scripts/tracking/test_molt_unified.py
```

## Detection Scripts

### `compare_local_models.py` - Model Comparison
Compare different local PyTorch models.

```bash
python scripts/detection/compare_local_models.py --help
```

### `compare_roboflow_models.py` - Roboflow Model Comparison
Compare different Roboflow models.

```bash
python scripts/detection/compare_roboflow_models.py --help
```

### `test_roboflow_video.py` - Roboflow Video Testing
Test Roboflow inference on video files.

```bash
python scripts/detection/test_roboflow_video.py --help
```

## Usage Notes

- All scripts are designed to be run from the project root directory
- Scripts automatically handle Python path setup to find project modules
- Use `--help` flag with any script to see detailed usage information
- Quality assurance scripts provide comprehensive validation of the entire project

## Quality Metrics

The quality assurance scripts validate:

✅ **Import Safety**: All critical modules can be imported without errors  
✅ **Type Safety**: Core modules pass strict MyPy type checking  
✅ **Functionality**: All scripts execute without errors  
✅ **Test Coverage**: Unit and integration tests pass  
✅ **Code Quality**: Consistent import paths and module structure  

Run `python scripts/run_all_checks.py` to validate the entire project!