# Design Document

## Overview

Enhance LocalPT to support RF-DETR models with minimal code changes. The design focuses on reusing existing infrastructure while adding RF-DETR-specific loading and inference logic.

## Architecture

### Model Detection Strategy
- Inspect model checkpoint contents to determine model type
- Look for RF-DETR specific keys in state_dict (e.g., transformer layers, decoder components)
- Keep existing YOLO detection as default fallback

### Loading Strategy
- For Roboflow RF-DETR models: Use `RFDETRNano(pretrain_weights=model_path)`
- Extract the underlying PyTorch model from the RF-DETR wrapper
- Reuse existing error handling patterns

### Inference Strategy  
- Add RF-DETR inference path in `predict()` method
- Convert RF-DETR outputs to existing standard format
- Reuse existing visualization and batch processing code

## Components and Interfaces

### Enhanced LocalPT Class
```python
class LocalPT(InferenceRunner):
    def __init__(self, model_path, confidence=0.5, iou=0.5, **kwargs):
        # Detect model type and load accordingly
        
    def _is_rfdetr_model(self, model_path: str) -> bool:
        # Inspect checkpoint to detect RF-DETR architecture
        
    def _load_rfdetr_model(self, model_path: str):
        # Load using RFDETRNano and extract PyTorch model
        
    def _predict_rfdetr(self, image, **kwargs):
        # RF-DETR specific inference
        
    def _convert_rfdetr_to_standard_format(self, outputs, image_shape):
        # Convert RF-DETR outputs to standard format
```

### Key Changes
1. Add model type detection in `__init__`
2. Add RF-DETR loading method
3. Add RF-DETR inference branch in `predict()`
4. Add RF-DETR output conversion method
5. Update `get_model_info()` to return correct type

## Data Models

### RF-DETR Output Format
RF-DETR returns different output structure than YOLO:
- Logits: `[batch, num_queries, num_classes]`  
- Boxes: `[batch, num_queries, 4]` (normalized cxcywh format)

### Standard Format (unchanged)
```python
{
    'predictions': [
        {
            'x': float,      # center x (normalized)
            'y': float,      # center y (normalized) 
            'width': float,  # width (normalized)
            'height': float, # height (normalized)
            'confidence': float,
            'class': str,
            'class_id': int
        }
    ],
    'image': {'width': int, 'height': int},
    'model': str
}
```

## Error Handling

### RF-DETR Import Errors
- Catch `ImportError` for missing `rfdetr` package
- Provide clear error message with installation instructions
- Fall back to YOLO loading if RF-DETR package unavailable

### Model Loading Errors
- Try `from_pretrained()` first
- Fall back to manual checkpoint loading
- Provide descriptive error messages for both paths

### Inference Errors
- Handle RF-DETR specific inference failures
- Maintain existing error handling patterns

## Testing Strategy

### Unit Tests
- Test RF-DETR model detection logic
- Test output format conversion
- Test error handling paths

### Integration Tests  
- Test end-to-end inference with real RF-DETR model
- Test comparison script with mixed model types
- Verify output format consistency

### Test Data Requirements
- Small RF-DETR model for testing
- Sample images for inference testing
- Expected output format examples