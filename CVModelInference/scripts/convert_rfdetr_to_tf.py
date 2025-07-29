#!/usr/bin/env python3
"""
Script to convert RF-DETR PyTorch models to TensorFlow/TFLite format.
This script handles the specific architecture of RF-DETR models.

NOTE: This script currently creates a simplified placeholder TensorFlow model
rather than a true RF-DETR model. The generated model will not have the same
functionality as the original PyTorch model. It is intended for demonstration
and development purposes only.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for ONNX availability
try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Install with: pip install onnx onnxruntime")

# Check for TensorFlow availability
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

# Check for tf2onnx availability
try:
    # Apply patch for NumPy compatibility issue before importing tf2onnx
    import numpy as np
    import sys
    
    # Patch NumPy to handle deprecated np.object
    if not hasattr(np, 'object'):
        np.object = object
        print("Applied NumPy patch for tf2onnx compatibility")
    
    import tf2onnx
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False
    print("Warning: tf2onnx not available. Install with: pip install tf2onnx")


class RFDETRWrapper(nn.Module):
    """
    Wrapper for RF-DETR model to make it exportable to ONNX/TensorFlow.
    
    IMPORTANT: This is currently a simplified placeholder implementation that does not
    replicate the actual RF-DETR architecture or functionality. It creates a model with
    the same input/output structure but different internal implementation.
    
    Future development should replace this with a proper implementation that
    maps the actual RF-DETR architecture and weights correctly.
    """
    def __init__(self, model_weights, input_size=640):
        super().__init__()
        self.input_size = input_size
        self.model_weights = model_weights
        
        # Extract model configuration
        if isinstance(model_weights, dict) and 'args' in model_weights:
            self.args = model_weights['args']
            self.num_classes = getattr(self.args, 'num_classes', 13)  # Default to 13 if not specified
        else:
            self.num_classes = 13  # Default value
            
        # Create a simplified model structure for export
        # This is a placeholder - actual implementation would depend on RF-DETR architecture
        self.backbone = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.neck = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(256, self.num_classes, kernel_size=1)
        
        # Initialize with dummy weights for now
        # In a real implementation, we would map the weights from model_weights to these layers
    
    def forward(self, x):
        """
        Forward pass for export purposes.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Dictionary with detection outputs
        """
        # Simple forward pass for demonstration
        # Real implementation would follow RF-DETR architecture
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        
        # Return in a format suitable for export
        return {
            'pred_logits': x.flatten(2).permute(0, 2, 1),  # (batch_size, num_queries, num_classes)
            'pred_boxes': torch.zeros(x.shape[0], 100, 4)  # (batch_size, num_queries, 4)
        }


def load_rfdetr_model(model_path):
    """
    Load RF-DETR model with safe globals.
    
    Args:
        model_path: Path to the RF-DETR model file
        
    Returns:
        Loaded model dictionary or None if failed
    """
    try:
        # Add necessary safe globals for RF-DETR models
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
        
        # Load the model
        model_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"RF-DETR model loaded successfully from {model_path}")
        return model_dict
    
    except Exception as e:
        print(f"Error loading RF-DETR model: {str(e)}")
        return None


def convert_rfdetr_to_onnx(model_dict, onnx_path, input_size=640):
    """
    Convert RF-DETR model to ONNX format.
    
    Args:
        model_dict: Loaded RF-DETR model dictionary
        onnx_path: Path to save the ONNX model
        input_size: Input size for the model
        
    Returns:
        Path to the saved ONNX model or None if failed
    """
    try:
        # Create wrapper model for export
        wrapper_model = RFDETRWrapper(model_dict, input_size=input_size)
        wrapper_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # Export to ONNX
        torch.onnx.export(
            wrapper_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['pred_logits', 'pred_boxes'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'pred_logits': {0: 'batch_size'},
                'pred_boxes': {0: 'batch_size'}
            }
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"RF-DETR model exported to ONNX: {onnx_path}")
        return onnx_path
    
    except Exception as e:
        print(f"Error converting RF-DETR model to ONNX: {str(e)}")
        return None


def convert_onnx_to_tf(onnx_path, tf_path):
    """
    Convert ONNX model to TensorFlow SavedModel format.
    
    Args:
        onnx_path: Path to the ONNX model
        tf_path: Path to save the TensorFlow model
        
    Returns:
        Path to the saved TensorFlow model or None if failed
    """
    try:
        # Check if tf2onnx is available
        if not TF2ONNX_AVAILABLE:
            print("tf2onnx is not available. Cannot convert ONNX to TensorFlow.")
            return None
        
        # Convert ONNX to TensorFlow using subprocess
        # This avoids API compatibility issues with tf2onnx
        with tempfile.TemporaryDirectory() as temp_dir:
            tf_temp_path = os.path.join(temp_dir, "model")
            
            # Use tf2onnx command line tool
            import subprocess
            cmd = [
                sys.executable, "-m", "tf2onnx.convert",
                "--input", str(onnx_path),
                "--output", str(tf_temp_path),
                "--opset", "12",
                "--target", "tf_saved_model"
            ]
            
            print(f"Running conversion command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"tf2onnx.convert command failed with exit code {result.returncode}")
            
            # Move to final location
            if os.path.exists(tf_path):
                shutil.rmtree(tf_path)
            shutil.move(tf_temp_path, tf_path)
        
        print(f"TensorFlow model saved to {tf_path}")
        return tf_path
    
    except Exception as e:
        print(f"Error converting ONNX to TensorFlow: {str(e)}")
        
        # Try direct TensorFlow approach
        try:
            print("Trying direct TensorFlow approach...")
            
            # Create a simple TensorFlow model that can be exported
            class SimpleModel(tf.Module):
                def __init__(self):
                    super().__init__()
                    # Define some sample weights for demonstration
                    self.conv1 = tf.Variable(tf.random.normal([3, 3, 3, 64]), name="conv1/kernel")
                    self.conv2 = tf.Variable(tf.random.normal([3, 3, 64, 128]), name="conv2/kernel")
                    self.dense = tf.Variable(tf.random.normal([128, 13]), name="dense/kernel")
                
                @tf.function(input_signature=[tf.TensorSpec(shape=[None, 640, 640, 3], dtype=tf.float32)])
                def __call__(self, x):
                    # Simple forward pass
                    x = tf.nn.conv2d(x, self.conv1, strides=[1, 2, 2, 1], padding="SAME")
                    x = tf.nn.relu(x)
                    x = tf.nn.conv2d(x, self.conv2, strides=[1, 2, 2, 1], padding="SAME")
                    x = tf.nn.relu(x)
                    x = tf.reduce_mean(x, axis=[1, 2])  # Global average pooling
                    logits = tf.matmul(x, self.dense)
                    
                    # Return in format similar to RF-DETR output
                    return {
                        "pred_logits": logits,
                        "pred_boxes": tf.zeros([tf.shape(x)[0], 100, 4])
                    }
            
            # Create and save the TensorFlow model
            print("Creating simplified TensorFlow model...")
            tf_model = SimpleModel()
            tf.saved_model.save(tf_model, tf_path)
            
            print(f"Simplified TensorFlow model saved to {tf_path}")
            print("WARNING: This is a simplified placeholder model, not the actual RF-DETR model")
            return tf_path
        
        except Exception as inner_e:
            print(f"Error with direct TensorFlow approach: {str(inner_e)}")
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


def convert_rfdetr_to_tf(pt_model_path, tf_model_path, tflite=False, float16=False, input_size=640):
    """
    Convert RF-DETR PyTorch model to TensorFlow/TFLite format.
    
    Args:
        pt_model_path: Path to the PyTorch model
        tf_model_path: Path to save the TensorFlow model
        tflite: Whether to convert to TFLite format
        float16: Whether to use float16 precision for TFLite
        input_size: Input size for the model
        
    Returns:
        Path to the saved model or None if failed
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(tf_model_path), exist_ok=True)
    
    # Step 1: Load the RF-DETR model
    model_dict = load_rfdetr_model(pt_model_path)
    if model_dict is None:
        return None
    
    # Step 2: Convert to ONNX
    onnx_path = str(Path(pt_model_path).with_suffix('.onnx'))
    onnx_result = convert_rfdetr_to_onnx(model_dict, onnx_path, input_size=input_size)
    if onnx_result is None:
        return None
    
    # Step 3: Convert ONNX to TensorFlow
    tf_result = convert_onnx_to_tf(onnx_path, tf_model_path)
    if tf_result is None:
        return None
    
    # Step 4: Convert to TFLite if requested
    if tflite:
        tflite_path = str(Path(tf_model_path)) + '.tflite'
        tflite_result = convert_tf_to_tflite(tf_model_path, tflite_path, float16=float16)
        if tflite_result is not None:
            return tflite_result
    
    return tf_model_path


def main():
    """Main function to parse arguments and run conversion."""
    print("\n" + "*"*80)
    print("WARNING: This script creates a simplified placeholder TensorFlow model")
    print("rather than a true RF-DETR model. The generated model will NOT have the same")
    print("functionality as the original PyTorch model and is for demonstration purposes only.")
    print("*"*80 + "\n")
    parser = argparse.ArgumentParser(description='Convert RF-DETR PyTorch model to TensorFlow/TFLite')
    parser.add_argument('--model', required=True, help='Path to RF-DETR PyTorch model file (.pt)')
    parser.add_argument('--output', help='Path to save the TensorFlow model (default: same directory as input with _tf suffix)')
    parser.add_argument('--input-size', type=int, default=640, help='Input size for the model (default: 640)')
    parser.add_argument('--tflite', action='store_true', help='Convert to TensorFlow Lite format')
    parser.add_argument('--float16', action='store_true', help='Use float16 precision for TFLite conversion')
    parser.add_argument('--skip-onnx-cleanup', action='store_true', help='Skip cleanup of intermediate ONNX file')
    
    args = parser.parse_args()
    
    # Check if TensorFlow is available
    if not TF_AVAILABLE:
        print("Error: TensorFlow is not available. Install with: pip install tensorflow")
        sys.exit(1)
    
    # Check if ONNX is available
    if not ONNX_AVAILABLE:
        print("Error: ONNX is not available. Install with: pip install onnx onnxruntime")
        sys.exit(1)
    
    # Set up paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Set output path if not specified
    if args.output:
        tf_path = Path(args.output)
    else:
        tf_path = model_path.parent / f"{model_path.stem}_tf"
    
    print(f"Converting {model_path} to TensorFlow model...")
    
    # Run the conversion
    result = convert_rfdetr_to_tf(
        str(model_path),
        str(tf_path),
        tflite=args.tflite,
        float16=args.float16,
        input_size=args.input_size
    )
    
    if result is not None:
        print("\nConversion complete!")
        print(f"Model saved to: {result}")
        
        # Show appropriate loading instructions
        if args.tflite:
            print("\nTo load the model in TensorFlow Lite:")
            print(f"  interpreter = tf.lite.Interpreter(model_path='{result}')")
            print("  interpreter.allocate_tensors()")
        else:
            print("\nTo load the model in TensorFlow:")
            print(f"  model = tf.saved_model.load('{result}')")
    else:
        print("\nConversion failed.")
        sys.exit(1)
    
    # Clean up intermediate ONNX file if not skipped
    onnx_path = model_path.with_suffix('.onnx')
    if not args.skip_onnx_cleanup and onnx_path.exists():
        os.remove(onnx_path)
        print(f"Removed intermediate ONNX file: {onnx_path}")


if __name__ == "__main__":
    main()
