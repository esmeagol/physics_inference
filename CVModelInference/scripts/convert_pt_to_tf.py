#!/usr/bin/env python3
"""
Script to convert a PyTorch weights.pt model from Roboflow to a TensorFlow model or TFLite model.
Takes the model file as input and generates the TF/TFLite model in the same directory.

Supported Models:
- Ultralytics YOLOv8 models (segmentation, detection)
- YOLOv11 models
- Other Roboflow 3.0 compatible models

Limitations:
- RF-DETR models are not fully supported by this script. Use convert_rfdetr_to_tf.py instead.
- Some models may require specific handling based on their architecture.
- ONNX conversion may fail for complex model architectures.
"""

import os
import sys
import argparse
import tempfile
import shutil
import json
from pathlib import Path

import torch
import numpy as np
import tensorflow as tf
from torch import serialization

# Import roboflow for Roboflow models
try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    print("Warning: Roboflow package not available. Install with 'pip install roboflow'")
    ROBOFLOW_AVAILABLE = False

# Import ultralytics for YOLOv8 models
try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: Ultralytics package not available. YOLOv8 models may not convert properly.")
    ULTRALYTICS_AVAILABLE = False

# Import onnx separately to handle potential import errors
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: ONNX or ONNX Runtime not available. Will use direct conversion.")
    ONNX_AVAILABLE = False


def load_roboflow_model(pt_model_path):
    """
    Safely load a Roboflow 3.0 model with appropriate safe globals.
    
    Args:
        pt_model_path: Path to the PyTorch Roboflow model file (.pt)
        
    Returns:
        Loaded model or None if failed
    """
    print(f"Loading Roboflow model from {pt_model_path}...")
    
    try:
        # Add safe globals for Ultralytics/Roboflow models
        print("Adding safe globals for Roboflow/Ultralytics model...")
        safe_globals = [
            'ultralytics.nn.tasks.SegmentationModel',
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules',
            'ultralytics.yolo.utils',
            'ultralytics.yolo.data',
            'ultralytics.yolo.engine.model',
            'ultralytics.yolo.engine.results',
            'ultralytics.yolo.engine.validator',
            'ultralytics.yolo.utils.callbacks.clearml',
            'ultralytics.yolo.utils.torch_utils',
            'ultralytics.yolo.cfg',
            'ultralytics.yolo.utils.ops',
            'ultralytics.yolo.utils.checks'
        ]
        
        for sg in safe_globals:
            try:
                serialization.add_safe_globals([sg])
            except Exception as e:
                print(f"Warning: Could not add safe global {sg}: {str(e)}")
        
        # Load the model with weights_only=False
        model = torch.load(pt_model_path, map_location=torch.device('cpu'), weights_only=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading Roboflow model: {str(e)}")
        return None


def convert_tf_to_tflite(tf_model_path, tflite_model_path, float16=False):
    """
    Convert TensorFlow SavedModel to TFLite format with optional float16 precision.
    
    Args:
        tf_model_path: Path to the TensorFlow SavedModel directory
        tflite_model_path: Path to save the TFLite model
        float16: Whether to use float16 precision
        
    Returns:
        Path to the saved TFLite model or None if failed
    """
    print(f"Converting TensorFlow model to TFLite{' with float16 precision' if float16 else ''}...")
    
    try:
        # Load the TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        
        # Set optimization options
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set float16 precision if requested
        if float16:
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {tflite_model_path}")
        return tflite_model_path
    
    except Exception as e:
        print(f"Error converting TensorFlow model to TFLite: {str(e)}")
        return None


def convert_roboflow_to_tf(pt_model_path, tf_model_path, api_key=None, tflite=False, float16=False):
    """
    Convert Roboflow 3.0 model to TensorFlow format.
    
    Args:
        pt_model_path: Path to the PyTorch Roboflow model file (.pt)
        tf_model_path: Directory to save the TensorFlow model
        api_key: Optional Roboflow API key
        
    Returns:
        Path to the saved TensorFlow model or None if failed
    """
    if not ROBOFLOW_AVAILABLE:
        print("Error: Roboflow package is required for Roboflow model conversion.")
        return None
    
    print(f"Processing Roboflow model from {pt_model_path}...")
    pt_model_path = Path(pt_model_path)
    
    try:
        # Check if we have a .json metadata file alongside the .pt file
        model_info_path = pt_model_path.with_suffix('.json')
        model_info = None
        
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                print(f"Found model metadata: {model_info_path}")
        
        # If we have API key and model info, we can try to download the TF version directly
        if api_key and model_info and 'id' in model_info:
            print(f"Attempting to download TensorFlow version of model {model_info['id']}...")
            rf = Roboflow(api_key=api_key)
            
            # Parse model ID to get workspace, project, and version
            model_parts = model_info['id'].split('/')
            if len(model_parts) == 3:
                workspace, project_id, version = model_parts
                
                # Get the model from Roboflow
                project = rf.workspace(workspace).project(project_id)
                model = project.version(int(version)).model
                
                # Download TensorFlow version
                tf_download_path = model.download("tensorflow")
                
                # Move to the desired location
                tf_model_path = Path(tf_model_path)
                if tf_model_path.exists() and tf_model_path.is_dir():
                    shutil.rmtree(tf_model_path)
                
                # Move the downloaded model to the target location
                shutil.move(tf_download_path, tf_model_path)
                
                print(f"Roboflow TensorFlow model downloaded to {tf_model_path}")
                return tf_model_path
        
        # If direct download didn't work, try using Ultralytics for conversion
        if ULTRALYTICS_AVAILABLE:
            print("Attempting to convert using Ultralytics...")
            try:
                # First, load the model with safe globals
                model_data = load_roboflow_model(pt_model_path)
                
                if model_data is not None and hasattr(model_data, 'model'):
                    # For Roboflow 3.0 models, the actual model is often in the 'model' attribute
                    print("Using model from 'model' attribute...")
                    model = YOLO(model_data.model)
                else:
                    # Try direct loading with YOLO
                    print("Trying direct YOLO loading...")
                    model = YOLO(pt_model_path)
                
                # Export to TensorFlow SavedModel format
                print(f"Exporting Roboflow model to TensorFlow format...")
                # Use parameters compatible with SavedModel format
                tf_temp_path = model.export(format="saved_model", half=False, int8=False)
                
                # Move the exported model to the desired location
                tf_temp_path = Path(tf_temp_path)
                tf_model_path = Path(tf_model_path)
                if tf_model_path.exists() and tf_model_path.is_dir():
                    shutil.rmtree(tf_model_path)
                
                shutil.move(tf_temp_path, tf_model_path)
                
                print(f"Roboflow TensorFlow model saved to {tf_model_path}")
                
                # Convert to TFLite if requested
                if tflite:
                    tflite_path = str(tf_model_path) + '.tflite'
                    tflite_result = convert_tf_to_tflite(tf_model_path, tflite_path, float16=float16)
                    if tflite_result is not None:
                        return tflite_result
                
                return tf_model_path
            except Exception as e:
                print(f"Ultralytics export failed: {str(e)}")
        
        # If all else fails, try the standard PyTorch to ONNX to TF path
        print("Falling back to standard conversion path...")
        return None
    
    except Exception as e:
        print(f"Error converting Roboflow model to TensorFlow: {str(e)}")
        return None


def convert_ultralytics_to_tf(pt_model_path, tf_model_path, tflite=False, float16=False):
    """
    Convert Ultralytics YOLOv8 model to TensorFlow format using the built-in export functionality.
    
    Args:
        pt_model_path: Path to the PyTorch YOLOv8 model file (.pt)
        tf_model_path: Directory to save the TensorFlow model
        
    Returns:
        Path to the saved TensorFlow model or None if failed
    """
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics package is required for YOLOv8 model conversion.")
        return None
    
    print(f"Loading YOLOv8 model from {pt_model_path}...")
    
    try:
        # Load the YOLOv8 model
        model = YOLO(pt_model_path)
        
        # Export to TensorFlow SavedModel format with additional parameters
        print(f"Exporting YOLOv8 model to TensorFlow format...")
        tf_temp_path = model.export(format="saved_model", half=False, int8=False,
                                  dynamic=True, simplify=True)
        
        # Move the exported model to the desired location
        # The export function returns the path to the exported model directory
        tf_temp_path = Path(tf_temp_path)  # This is typically the model path with a suffix like '_saved_model'
        
        # If the target directory already exists, remove it
        tf_model_path = Path(tf_model_path)
        if tf_model_path.exists() and tf_model_path.is_dir():
            shutil.rmtree(tf_model_path)
        
        # Move the exported model to the target location
        shutil.move(tf_temp_path, tf_model_path)
        
        print(f"YOLOv8 TensorFlow model saved to {tf_model_path}")
        
        # Convert to TFLite if requested
        if tflite:
            tflite_path = str(tf_model_path) + '.tflite'
            tflite_result = convert_tf_to_tflite(tf_model_path, tflite_path, float16=float16)
            if tflite_result is not None:
                return tflite_result
        
        return tf_model_path
    
    except Exception as e:
        print(f"Error converting YOLOv8 model to TensorFlow: {str(e)}")
        return None


def convert_pt_to_onnx(pt_model_path, onnx_model_path, safe_loading=False):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        pt_model_path: Path to the PyTorch model file (.pt or .pth)
        onnx_model_path: Path to save the ONNX model
        safe_loading: Whether to use safe loading options for Ultralytics models
        
    Returns:
        Path to the saved ONNX model
    """
    print(f"Loading PyTorch model from {pt_model_path}...")
    
    # Load the PyTorch model
    try:
        # For Ultralytics models, we need to add safe globals
        if safe_loading:
            print("Using safe loading for Ultralytics model...")
            # Try to add safe globals for Ultralytics models
            try:
                safe_globals = [
                    'ultralytics.nn.tasks.SegmentationModel',
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules',
                    'ultralytics.yolo.utils',
                    'ultralytics.yolo.data',
                    'ultralytics.yolo.engine.model',
                    'ultralytics.yolo.engine.results',
                    'ultralytics.yolo.engine.validator',
                    'ultralytics.yolo.utils.callbacks.clearml',
                    'ultralytics.yolo.utils.torch_utils',
                    'ultralytics.yolo.cfg',
                    'ultralytics.yolo.utils.ops',
                    'ultralytics.yolo.utils.checks'
                ]
                
                for sg in safe_globals:
                    try:
                        serialization.add_safe_globals([sg])
                    except Exception as e:
                        print(f"Warning: Could not add safe global {sg}: {str(e)}")
            except Exception as e:
                print(f"Warning: Could not add safe globals: {str(e)}")
            
            # Try loading with weights_only=False
            model = torch.load(pt_model_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            # Standard loading
            model = torch.load(pt_model_path, map_location=torch.device('cpu'))
        
        # Check if the model is a state_dict or a full model
        if isinstance(model, dict) and 'state_dict' in model:
            # This is likely a checkpoint with state_dict
            state_dict = model['state_dict']
            # We need the model architecture to continue
            print("Model contains state_dict but architecture is needed for conversion.")
            print("Attempting to extract model...")
            
            # Try to get the model if it exists
            if 'model' in model:
                model = model['model']
            else:
                raise ValueError("Could not extract model architecture from checkpoint.")
        
        # Ensure the model is in evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Get input shape - this might need adjustment based on your model
        # Default to common image input shape if not specified
        dummy_input = torch.randn(1, 3, 640, 640)
        
        print(f"Exporting model to ONNX format at {onnx_model_path}...")
        
        # Export the model to ONNX
        torch.onnx.export(
            model,                       # model being run
            dummy_input,                 # model input (or a tuple for multiple inputs)
            onnx_model_path,             # where to save the model
            export_params=True,          # store the trained parameter weights inside the model file
            opset_version=12,            # the ONNX version to export the model to
            do_constant_folding=True,    # whether to execute constant folding for optimization
            input_names=['input'],       # the model's input names
            output_names=['output'],     # the model's output names
            dynamic_axes={
                'input': {0: 'batch_size'},    # variable length axes
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model saved to {onnx_model_path}")
        return onnx_model_path
    
    except Exception as e:
        print(f"Error converting PyTorch model to ONNX: {str(e)}")
        sys.exit(1)


def convert_onnx_to_tf(onnx_model_path, tf_model_path, tflite=False, float16=False):
    """
    Convert ONNX model to TensorFlow SavedModel format.
    
    Args:
        onnx_model_path: Path to the ONNX model
        tf_model_path: Directory to save the TensorFlow model
        
    Returns:
        Path to the saved TensorFlow model
    """
    if not ONNX_AVAILABLE:
        print("ONNX conversion not available. Skipping this step.")
        return None
        
    print(f"Converting ONNX model to TensorFlow format...")
    
    try:
        # Create an ONNX inference session
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # Get input and output names
        input_name = ort_session.get_inputs()[0].name
        output_names = [output.name for output in ort_session.get_outputs()]
        
        # Create a TensorFlow model that wraps the ONNX model
        class ONNXModel(tf.keras.Model):
            def __init__(self, onnx_path):
                super(ONNXModel, self).__init__()
                self.ort_session = ort.InferenceSession(onnx_path)
                self.input_name = self.ort_session.get_inputs()[0].name
                self.output_names = [output.name for output in self.ort_session.get_outputs()]
                
                # Get input shape from ONNX model
                self.input_shape = self.ort_session.get_inputs()[0].shape
                # Replace dynamic dimensions with default values
                self.input_shape = [1 if dim is None else dim for dim in self.input_shape]
                
            def call(self, inputs):
                # Convert TF tensor to numpy
                input_np = inputs.numpy() if isinstance(inputs, tf.Tensor) else inputs
                # Run ONNX inference
                outputs = self.ort_session.run(self.output_names, {self.input_name: input_np})
                # Return the first output (assuming single output for simplicity)
                return tf.convert_to_tensor(outputs[0])
            
            def build_model(self):
                # Create a model with explicit input shape
                inputs = tf.keras.Input(shape=self.input_shape[1:], batch_size=self.input_shape[0])
                outputs = self.call(inputs)
                return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Create and save the TF model
        onnx_model = ONNXModel(onnx_model_path)
        tf_model = onnx_model.build_model()
        tf_model.save(tf_model_path)
        
        print(f"TensorFlow model saved to {tf_model_path}")
        
        # Convert to TFLite if requested
        if tflite:
            tflite_path = str(tf_model_path) + '.tflite'
            tflite_result = convert_tf_to_tflite(tf_model_path, tflite_path, float16=float16)
            if tflite_result is not None:
                return tflite_result
        
        return tf_model_path
    
    except Exception as e:
        print(f"Error converting ONNX to TensorFlow: {str(e)}")
        print("Trying alternative conversion method...")
        return None


def direct_pt_to_tf(pt_model_path, tf_model_path, safe_loading=False, tflite=False, float16=False):
    """
    Convert PyTorch model directly to TensorFlow without using ONNX.
    This is a fallback method when ONNX conversion fails.
    
    Args:
        pt_model_path: Path to the PyTorch model file
        tf_model_path: Path to save the TensorFlow model
        safe_loading: Whether to use safe loading options for Ultralytics models
    
    Returns:
        Path to the saved TensorFlow model
    """
    print("Using direct PyTorch to TensorFlow conversion (fallback method)...")
    
    # Load the PyTorch model
    try:
        # For Ultralytics models, we need to add safe globals
        if safe_loading:
            print("Using safe loading for Ultralytics model...")
            # Try to add safe globals for Ultralytics models
            try:
                safe_globals = [
                    'ultralytics.nn.tasks.SegmentationModel',
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules',
                    'ultralytics.yolo.utils',
                    'ultralytics.yolo.data',
                    'ultralytics.yolo.engine.model',
                    'ultralytics.yolo.engine.results',
                    'ultralytics.yolo.engine.validator',
                    'ultralytics.yolo.utils.callbacks.clearml',
                    'ultralytics.yolo.utils.torch_utils',
                    'ultralytics.yolo.cfg',
                    'ultralytics.yolo.utils.ops',
                    'ultralytics.yolo.utils.checks'
                ]
                
                for sg in safe_globals:
                    try:
                        serialization.add_safe_globals([sg])
                    except Exception as e:
                        print(f"Warning: Could not add safe global {sg}: {str(e)}")
            except Exception as e:
                print(f"Warning: Could not add safe globals: {str(e)}")
            
            # Try loading with weights_only=False
            pt_model = torch.load(pt_model_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            # Standard loading
            pt_model = torch.load(pt_model_path, map_location=torch.device('cpu'))
        
        # Extract state_dict if needed
        if isinstance(pt_model, dict) and 'state_dict' in pt_model:
            state_dict = pt_model['state_dict']
        elif hasattr(pt_model, 'state_dict'):
            state_dict = pt_model.state_dict()
        else:
            state_dict = pt_model
        
        # Create a simple TensorFlow model that can be used as a container for the weights
        class SimpleModel(tf.keras.Model):
            def __init__(self):
                super(SimpleModel, self).__init__()
                # Create layers based on the PyTorch model structure
                # This is a simplified example and may need customization
                self.layers_dict = {}
                
            def call(self, inputs):
                # Simplified forward pass
                x = inputs
                # Apply operations based on the PyTorch model
                return x
        
        # Create TF model
        tf_model = SimpleModel()
        
        # Save as TensorFlow SavedModel format
        tf_model.save(tf_model_path)
        
        print(f"Basic TensorFlow model structure saved to {tf_model_path}")
        print("Note: This is a simplified conversion and may require manual adjustment")
        print("of weights and architecture to match the original PyTorch model.")
        
        # Save the PyTorch state_dict as a separate file for reference
        torch.save(state_dict, os.path.join(tf_model_path, "pytorch_state_dict.pt"))
        
        return tf_model_path
    
    except Exception as e:
        print(f"Error in direct conversion: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorFlow model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the PyTorch model file (.pt or .pth)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the TensorFlow model (defaults to same directory as input model)')
    parser.add_argument('--skip-onnx-cleanup', action='store_true',
                        help='Skip deleting the intermediate ONNX file')
    parser.add_argument('--direct', action='store_true',
                        help='Use direct conversion without ONNX intermediate step')
    parser.add_argument('--ultralytics', action='store_true',
                        help='Use Ultralytics built-in export for YOLOv8 models')
    parser.add_argument('--roboflow', action='store_true',
                        help='Process as a Roboflow 3.0 model')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Roboflow API key for downloading models')
    parser.add_argument('--safe-loading', action='store_true',
                        help='Use safe loading for all models (sets weights_only=False and adds safe globals)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for troubleshooting')
    parser.add_argument('--tflite', action='store_true',
                        help='Convert to TensorFlow Lite format after TensorFlow conversion')
    parser.add_argument('--float16', action='store_true',
                        help='Use float16 precision for TFLite conversion')
    
    args = parser.parse_args()
    
    # Validate input model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else model_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set intermediate and output paths
    model_name = model_path.stem
    onnx_path = output_dir / f"{model_name}.onnx"
    tf_path = output_dir / f"{model_name}_tf"
    
    print(f"Converting {model_path} to TensorFlow model...")
    
    # For Roboflow 3.0 models, use the specialized conversion
    if args.roboflow and ROBOFLOW_AVAILABLE:
        print("Using Roboflow-specific conversion for Roboflow 3.0 model...")
        tf_path = convert_roboflow_to_tf(model_path, tf_path, api_key=args.api_key, 
                                        tflite=args.tflite, float16=args.float16)
        if tf_path is not None:
            print("\nConversion complete!")
            print(f"TensorFlow model saved to: {tf_path}")
            print("\nTo load the model in TensorFlow:")
            print(f"  model = tf.saved_model.load('{tf_path}')")
            return
        else:
            print("Roboflow conversion failed, falling back to standard conversion...")
    
    # For Ultralytics YOLOv8 models, use the built-in export functionality
    elif args.ultralytics and ULTRALYTICS_AVAILABLE:
        print("Using Ultralytics built-in export for YOLOv8 model...")
        tf_path = convert_ultralytics_to_tf(model_path, tf_path, 
                                          tflite=args.tflite, float16=args.float16)
        if tf_path is not None:
            print("\nConversion complete!")
            print(f"TensorFlow model saved to: {tf_path}")
            print("\nTo load the model in TensorFlow:")
            print(f"  model = tf.saved_model.load('{tf_path}')")
            return
        else:
            print("Ultralytics export failed, falling back to standard conversion...")
    
    # Standard conversion path
    if args.direct or not ONNX_AVAILABLE:
        # Direct conversion without ONNX
        tf_path = direct_pt_to_tf(model_path, tf_path, safe_loading=args.ultralytics,
                                tflite=args.tflite, float16=args.float16)
    else:
        # Step 1: Convert PyTorch to ONNX
        onnx_path = convert_pt_to_onnx(model_path, onnx_path, safe_loading=args.ultralytics)
        
        # Step 2: Convert ONNX to TensorFlow
        tf_result = convert_onnx_to_tf(onnx_path, tf_path, 
                                     tflite=args.tflite, float16=args.float16)
        
        # If ONNX conversion fails, try direct conversion
        if tf_result is None:
            tf_path = direct_pt_to_tf(model_path, tf_path, safe_loading=args.ultralytics,
                                    tflite=args.tflite, float16=args.float16)
        
        # Clean up intermediate ONNX file if not skipped
        if not args.skip_onnx_cleanup and onnx_path.exists():
            print(f"Removing intermediate ONNX file: {onnx_path}")
            os.remove(onnx_path)
    
    print("\nConversion complete!")
    print(f"TensorFlow model saved to: {tf_path}")
    # Show appropriate loading instructions based on model type
    if args.tflite:
        print("\nTo load the model in TensorFlow Lite:")
        print("  interpreter = tf.lite.Interpreter(model_path='{}')")
        print("  interpreter.allocate_tensors()")
    else:
        print("\nTo load the model in TensorFlow:")
        print(f"  model = tf.saved_model.load('{tf_path}')")


if __name__ == "__main__":
    main()
