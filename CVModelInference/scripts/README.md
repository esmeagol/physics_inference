# Scripts Directory

This directory contains various utility scripts for model analysis, benchmarking, conversion, and testing in the physics inference project.

## Model Conversion Scripts

### `convert_pt_to_tf.py`

Converts Roboflow 3.0 PyTorch models (weights.pt) to TensorFlow SavedModel and TensorFlow Lite formats.

**Usage:**
```bash
python convert_pt_to_tf.py --model /path/to/weights.pt [--tflite] [--float16] [--roboflow] [--ultralytics] [--safe_loading]
```

**Arguments:**
- `--model`: Path to the PyTorch model file (weights.pt)
- `--tflite`: (Optional) Convert to TFLite format
- `--float16`: (Optional) Use float16 precision for TFLite model
- `--roboflow`: (Optional) Use Roboflow SDK for conversion
- `--ultralytics`: (Optional) Use Ultralytics export
- `--safe_loading`: (Optional) Use safe loading with torch.serialization.add_safe_globals

**Supported Models:**
- Ultralytics YOLOv8 models
- YOLOv11 models
- Other Roboflow 3.0 compatible models

### `convert_rfdetr_to_tf.py`

Custom conversion script for RF-DETR PyTorch models to TensorFlow and TensorFlow Lite formats.

**Usage:**
```bash
python convert_rfdetr_to_tf.py --model /path/to/rf-detr-weights.pt [--tflite] [--float16] [--input_size 640]
```

**Arguments:**
- `--model`: Path to the RF-DETR PyTorch model file
- `--tflite`: (Optional) Convert to TFLite format
- `--float16`: (Optional) Use float16 precision for TFLite model
- `--input_size`: (Optional) Input size for the model, default is 640
- `--cleanup`: (Optional) Remove intermediate files after conversion

**Notes:**
- Currently creates a simplified placeholder model for RF-DETR architectures
- Future versions will implement true RF-DETR architecture in TensorFlow

### `convert_to_onnx.py`

Converts PyTorch models to ONNX format.

**Usage:**
```bash
python convert_to_onnx.py --model /path/to/model.pt --output /path/to/output.onnx
```

### `convert_to_tflite.py`

Converts TensorFlow SavedModel to TensorFlow Lite format.

**Usage:**
```bash
python convert_to_tflite.py --model /path/to/tf_model --output /path/to/output.tflite [--float16]
```

## Model Analysis Scripts

### `analyze_model.py`

Analyzes model architecture, parameters, and structure.

**Usage:**
```bash
python analyze_model.py --model /path/to/model
```

### `analyze_state_dict.py`

Analyzes PyTorch state dictionary structure and parameters.

**Usage:**
```bash
python analyze_state_dict.py --model /path/to/weights.pt
```

### `inspect_rfdetr.py` and `inspect_rfdetr_model.py`

Inspect RF-DETR model structure and parameters.

**Usage:**
```bash
python inspect_rfdetr.py --model /path/to/rf-detr-weights.pt
```

### `explore_rfdetr_export.py` and `export_rfdetr_nano.py`

Experimental scripts for exploring RF-DETR model export options.

## Benchmarking Scripts

### `benchmark_inference.py`

Benchmarks inference performance for various model formats.

**Usage:**
```bash
python benchmark_inference.py --model /path/to/model --input /path/to/input_data [--iterations 100]
```

### `benchmark_roboflow_pkg.py`

Benchmarks Roboflow package inference performance.

**Usage:**
```bash
python benchmark_roboflow_pkg.py --api_key YOUR_API_KEY --model_id YOUR_MODEL_ID
```

### `benchmark_table_detection.py`

Benchmarks table detection performance.

**Usage:**
```bash
python benchmark_table_detection.py --model /path/to/model --dataset /path/to/dataset
```

## Comparison and Testing Scripts

### `compare_roboflow_models.py`

Compares different Roboflow models for performance and accuracy.

**Usage:**
```bash
python compare_roboflow_models.py --models model1,model2 --dataset /path/to/dataset
```

### `test_roboflow_video.py`

Tests Roboflow models on video input.

**Usage:**
```bash
python test_roboflow_video.py --model /path/to/model --video /path/to/video.mp4
```

### `test_table_detection.py`

Tests table detection models on images.

**Usage:**
```bash
python test_table_detection.py --model /path/to/model --image /path/to/image.jpg
```

### `process_test_images.py`

Processes test images with detection models.

**Usage:**
```bash
python process_test_images.py --model /path/to/model --input /path/to/images --output /path/to/output
```

### `generate_test_image.py`

Generates synthetic test images for model evaluation.

**Usage:**
```bash
python generate_test_image.py --output /path/to/output.npy --shape 20,128,128,3
```

## Utility Scripts

### `load_rfdetr_nano.py`

Utility script for loading RF-DETR nano models.

**Usage:**
```bash
python load_rfdetr_nano.py --model /path/to/rf-detr-nano-weights.pt
```

## Notes

- Most scripts support the `--help` flag for detailed usage information
- For model conversion scripts, ensure all required dependencies are installed
- When using TFLite conversion with float16 precision, verify model accuracy after conversion
- RF-DETR model conversion is still experimental and may not preserve all model capabilities
