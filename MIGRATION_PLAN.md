# Repository Reorganization Migration Plan

## Target Directory Structure

```
physics_inference/
├── src/                           # All source code
│   ├── detection/                 # Object detection modules
│   │   ├── local_pt_inference.py
│   │   ├── roboflow_local_inference.py
│   │   └── ...
│   ├── tracking/                  # Object tracking and related modules
│   │   ├── tracker.py
│   │   ├── tracking.py
│   │   ├── trackers/              # Different tracker implementations
│   │   │   ├── deepsort_tracker.py
│   │   │   ├── sv_bytetrack_tracker.py
│   │   │   ├── molt/              # MOLT tracker implementation
│   │   │   │   ├── tracker.py
│   │   │   │   ├── ball_count_manager.py
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── evaluation/            # Tracking evaluation
│   │   │   ├── ground_truth_evaluator.py
│   │   │   ├── moment_evaluator.py
│   │   │   └── ...
│   │   ├── state_reconstructor.py # State reconstruction
│   │   ├── event_processor.py     # Event processing
│   │   └── ...
│   ├── training/                  # Training-related modules
│   │   └── ...
│   └── common/                    # Shared utilities and interfaces
│       ├── inference_runner.py
│       ├── ground_truth_visualizer.py
│       └── ...
├── tests/                         # All test files
│   ├── detection/
│   ├── tracking/
│   │   ├── evaluation/
│   │   ├── trackers/
│   │   │   ├── molt/              # Tests for MOLT tracker
│   │   │   └── ...
│   │   └── ...
│   ├── training/
│   ├── common/
│   └── integration/               # Integration tests
├── scripts/                       # Utility scripts
├── examples/                      # Example usage
├── assets/                        # Images, models, etc.
├── requirements.txt
└── setup.py
```

## Phase 1: Preparation

1. **Create the new directory structure**
   - Create `src` directory and subdirectories
   - Create test subdirectories in the existing `tests` directory
   - Keep original files in place until migration

2. **Identify dependencies between modules**
   - Map out import relationships between files
   - Identify circular dependencies that might need to be resolved

3. **Set up testing infrastructure**
   - Create a script to run all tests and check for failures
   - Create a script to run mypy for type checking

## Phase 2: Migration of Common Modules

1. **Move common utilities and interfaces**
   - Move `inference_runner.py` to `src/common/`
   - Move `ground_truth_visualizer.py` to `src/common/`
   - Update imports in these files
   - Run tests and type checking to ensure everything works

## Phase 3: Migration of Detection Modules

1. **Move detection-related modules**
   - Move `local_pt_inference.py` to `src/detection/`
   - Move `roboflow_local_inference.py` to `src/detection/`
   - Update imports in these files
   - Run tests and type checking

## Phase 4: Migration of Tracking Modules

1. **Move base tracking modules**
   - Move `tracker.py` and `tracking.py` to `src/tracking/`
   - Update imports
   - Run tests and type checking

2. **Move tracker implementations**
   - Create `src/tracking/trackers/` directory
   - Move `deepsort_tracker.py` and `sv_bytetrack_tracker.py` to `src/tracking/trackers/`
   - Update imports
   - Run tests and type checking

3. **Move MOLT tracker**
   - Create `src/tracking/trackers/molt/` directory
   - Move MOLT-related files from `PureCV/molt/` to `src/tracking/trackers/molt/`
   - Update imports
   - Run tests and type checking

4. **Move evaluation modules**
   - Create `src/tracking/evaluation/` directory
   - Move `ground_truth_evaluator.py` and `moment_evaluator.py` to `src/tracking/evaluation/`
   - Update imports
   - Run tests and type checking

5. **Move remaining tracking-related modules**
   - Move `state_reconstructor.py` to `src/tracking/`
   - Move `event_processor.py` to `src/tracking/`
   - Update imports
   - Run tests and type checking

## Phase 5: Migration of Training Modules

1. **Identify and move training-related modules**
   - Move any training-related modules to `src/training/`
   - Update imports
   - Run tests and type checking

## Phase 6: Migration of Tests

1. **Move detection tests**
   - Move detection-related test files to `tests/detection/`
   - Update imports
   - Run tests

2. **Move tracking tests**
   - Move tracking-related test files to `tests/tracking/`
   - Move evaluation test files to `tests/tracking/evaluation/`
   - Move tracker implementation tests to `tests/tracking/trackers/`
   - Move MOLT tests to `tests/tracking/trackers/molt/`
   - Update imports
   - Run tests

3. **Move common module tests**
   - Move tests for common modules to `tests/common/`
   - Update imports
   - Run tests

4. **Move integration tests**
   - Move integration test files to `tests/integration/`
   - Update imports
   - Run tests

## Phase 7: Update Package Structure

1. **Update `setup.py`**
   - Modify to reflect the new directory structure
   - Ensure packages are properly discovered

2. **Update import statements in examples and scripts**
   - Update any import statements in example files
   - Update any import statements in script files

3. **Create or update `__init__.py` files**
   - Ensure each directory has an appropriate `__init__.py`
   - Set up proper exports to maintain backward compatibility

## Phase 8: Validation and Cleanup

1. **Run comprehensive test suite**
   - Ensure all tests pass
   - Fix any remaining issues

2. **Run type checking**
   - Ensure mypy passes without errors
   - Fix any type-related issues

3. **Remove old directories**
   - Once everything is working, remove the original directories

## Implementation Details for Each Step

For each file move:

1. **Create the target directory if it doesn't exist**
2. **Copy the file to the new location** (don't move yet to maintain working state)
3. **Update imports in the copied file**
4. **Update imports in files that import the moved file**
5. **Run tests to ensure everything still works**
6. **Run mypy to ensure type checking passes**
7. **Only after verification, remove the original file**

This incremental approach ensures that at each step, the codebase remains in a working state with passing tests and proper type checking.

## Testing Commands

Run tests:
```bash
# Run all tests
python -m unittest discover

# Run specific test file
python -m unittest path/to/test_file.py
```

Run type checking:
```bash
# Run mypy on the entire project
mypy .

# Run mypy on a specific file or directory
mypy path/to/file_or_directory
```

## Handling Import Issues

When updating imports, follow these guidelines:

1. **Relative imports**: Use relative imports within packages when appropriate
   ```python
   # Example: In src/tracking/trackers/deepsort_tracker.py importing from src/tracking/tracker.py
   from .. import tracker
   ```

2. **Absolute imports**: Use absolute imports for cross-package references
   ```python
   # Example: In src/tracking/trackers/deepsort_tracker.py importing from src/detection/local_pt_inference.py
   from src.detection.local_pt_inference import LocalPT
   ```

3. **Circular dependencies**: Resolve by:
   - Moving the shared functionality to a common module
   - Using import statements inside functions rather than at the module level
   - Refactoring to eliminate the circular dependency

## Backward Compatibility

To maintain backward compatibility during the transition:

1. **Create proxy modules** in the original locations that import and re-export from the new locations
2. **Use deprecation warnings** to inform users of the new import locations
3. **Update documentation** to reflect the new structure

## Rollback Plan

If issues arise during migration:

1. **Keep a backup** of the original structure
2. **Document each change** made during migration
3. **Have a script ready** to restore the original structure if needed
