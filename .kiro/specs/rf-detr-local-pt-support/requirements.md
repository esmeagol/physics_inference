# Requirements Document

## Introduction

This feature enhances the LocalPT inference module to support RF-DETR models alongside existing YOLO support. The goal is minimal changes to enable the comparison script to work with RF-DETR models.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to load RF-DETR models through LocalPT, so that I can run inference with the same interface as YOLO models.

#### Acceptance Criteria

1. WHEN LocalPT receives an RF-DETR model path THEN it SHALL detect the model type from the filename
2. WHEN an RF-DETR model is detected THEN it SHALL load using RFDETRNano.from_pretrained()
3. WHEN RF-DETR loading fails THEN it SHALL fall back to manual checkpoint loading
4. WHEN model loading succeeds THEN it SHALL be ready for inference

### Requirement 2

**User Story:** As a researcher, I want RF-DETR inference to return the same format as YOLO, so that existing code works without changes.

#### Acceptance Criteria

1. WHEN predict() is called with RF-DETR THEN it SHALL run model inference
2. WHEN RF-DETR inference completes THEN it SHALL convert outputs to LocalPT standard format
3. WHEN converting results THEN it SHALL map boxes, classes, and confidence scores correctly
4. WHEN confidence threshold is specified THEN it SHALL filter detections accordingly

### Requirement 3

**User Story:** As a researcher, I want the comparison script to work with RF-DETR models, so that I can compare different model types.

#### Acceptance Criteria

1. WHEN comparison script loads RF-DETR models THEN it SHALL use the enhanced LocalPT module
2. WHEN processing images THEN both YOLO and RF-DETR models SHALL work through the same interface
3. WHEN generating outputs THEN RF-DETR results SHALL display correctly in comparison images