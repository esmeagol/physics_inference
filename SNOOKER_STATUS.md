# Snooker Module Status

## Current State

The snooker module has been reorganized with table-related code moved to `src/snooker/table/` directory.

## ✅ Completed Features

1. **Perspective Transformation** (`src/snooker/perspective_transformer.py`)
   - Manual point selection with proper coordinate scaling
   - Perspective matrix calculation and transformation
   - Cropped output showing only table area
   - Configuration save/load functionality

2. **Reference Validation** (`src/snooker/reference_validator.py`)
   - Dimension validation for transformed images
   - Color analysis for table surface detection
   - Baulk line position validation
   - Comprehensive validation reports

3. **Table Generation** (`src/snooker/table/`)
   - Moved to separate subdirectory for better organization
   - Contains table-specific constants and generator classes
   - **Note**: Has coordinate system bugs that need fixing

## ⚠️ Known Issues

### Critical Bugs After IMAGE_WIDTH/IMAGE_HEIGHT Removal

The removal of `IMAGE_WIDTH` and `IMAGE_HEIGHT` constants has introduced coordinate system bugs:

1. **Table Generator Issues**:
   - Mixed coordinate systems (full image vs table-relative)
   - Inconsistent coordinate transformations
   - D-shape positioning may be incorrect
   - Ball positioning calculations need fixing

2. **Import Path Updates**:
   - Some imports may still reference old paths
   - Test scripts need verification after reorganization

3. **Coordinate System Confusion**:
   - Original constants assumed full image canvas
   - New system should use table-relative coordinates (0,0 at table top-left)
   - Need to standardize coordinate system throughout

## 🔧 Required Fixes

1. **Fix Table Generator**:
   - Standardize coordinate system to table-relative (0,0 = table top-left)
   - Fix D-shape positioning and radius calculation
   - Correct ball positioning logic
   - Update pocket and line drawing coordinates

2. **Update Constants**:
   - Remove IMAGE_WIDTH/IMAGE_HEIGHT references
   - Ensure all coordinates are table-relative
   - Update documentation to reflect new coordinate system

3. **Test and Verify**:
   - Test table generation after fixes
   - Verify perspective transformation still works
   - Check all import paths are correct

## 📁 File Structure

```
src/snooker/
├── __init__.py                   ✅ Main module
├── perspective_transformer.py   ✅ Working (general processing)
├── reference_validator.py       ✅ Working (general validation)
├── README.md                     ✅ Documentation
└── table/                        ⚠️ Has bugs (table-specific code)
    ├── __init__.py
    ├── table_constants.py        ⚠️ Coordinate system issues
    └── table_generator.py        ⚠️ Mixed coordinate systems

scripts/snooker/
├── test_with_image.py            ✅ Working
├── test_with_real_video.py       ✅ Working
└── setup_with_reference.py      ✅ Working
```

## Next Steps

1. Fix coordinate system bugs in table module
2. Test table generation functionality
3. Proceed with Task 2: ColorBasedBallDetector implementation

---

**Note**: This reorganization was done to prepare for the next phase of development. The table-related bugs need to be resolved before proceeding with ball detection implementation.