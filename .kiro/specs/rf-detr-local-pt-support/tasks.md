# Implementation Plan

- [x] 1. Add RF-DETR model detection to LocalPT

  - Implement `_is_rfdetr_model()` method that inspects checkpoint for RF-DETR specific keys
  - Add model type detection logic in `__init__()` method
  - _Requirements: 1.1_

- [x] 2. Implement RF-DETR model loading

  - Add `_load_rfdetr_model()` method using `RFDETRNano(pretrain_weights=model_path)`
  - Extract underlying PyTorch model from RF-DETR wrapper
  - Handle import errors gracefully with fallback to YOLO loading
  - _Requirements: 1.2, 1.3, 1.4_

- [x] 3. Add RF-DETR inference support

  - Implement `_predict_rfdetr()` method for RF-DETR specific inference
  - Add RF-DETR branch in existing `predict()` method
  - Apply confidence threshold filtering to RF-DETR outputs
  - _Requirements: 2.1, 2.4_

- [x] 4. Implement RF-DETR output format conversion

  - Add `_convert_rfdetr_to_standard_format()` method
  - Convert RF-DETR logits and boxes to LocalPT standard format
  - Handle coordinate transformation from cxcywh to standard format
  - _Requirements: 2.2, 2.3_

- [x] 5. Update model info method for RF-DETR

  - Modify `get_model_info()` to return correct type for RF-DETR models
  - Ensure consistent metadata format across model types
  - _Requirements: 3.1, 3.2_

- [x] 6. Write integration tests

  - Create test with real RF-DETR model file for end-to-end inference
  - Test comparison script with mixed YOLO and RF-DETR models
  - Verify output format consistency between model types
  - _Requirements: 3.3_

- [x] 7. Test the comparison script
  - Run comparison script with provided RF-DETR model and YOLO model
  - Verify side-by-side comparison images are generated correctly
  - Confirm RF-DETR detections display properly in output
  - _Requirements: 3.1, 3.2, 3.3_
