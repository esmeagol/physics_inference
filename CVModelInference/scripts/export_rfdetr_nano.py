#!/usr/bin/env python3
"""
RF-DETR Nano Model Export Utility

This is an EXPERIMENTAL utility script for exporting RF-DETR Nano models to ONNX format.
It is intended for development and debugging purposes only.

NOTE: This script has hardcoded paths and may require modifications to work in different environments.
It should be considered a development tool rather than a production-ready script.
"""

import os
import sys
import torch
import numpy as np

try:
    from rfdetr import RFDETRNano
except ImportError:
    print("Error: rfdetr package not available. This script requires the RF-DETR package.")
    print("Please install it or ensure it's in your Python path.")
    sys.exit(1)

# Import utility functions from the correct module path
try:
    from rfdetr.util.misc import nested_tensor_from_tensor_list
except ImportError:
    # Try alternative import path if the above fails
    from rfdetr.models.util.misc import nested_tensor_from_tensor_list

def export_rfdetr_nano(model_path, output_dir):
    """
    Export RF-DETR Nano model to ONNX format using PyTorch's built-in export.
    
    Args:
        model_path: Path to the RF-DETR Nano model weights (.pt file)
        output_dir: Directory to save the exported ONNX model
    """
    print(f"Exporting RF-DETR Nano model from {model_path} to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rfdetr_nano.onnx")
    
    try:
        # Initialize the model with pretrained weights
        print("Initializing RFDETRNano with pretrained weights...")
        model = RFDETRNano(pretrain_weights=model_path)
        print("Model initialized successfully!")
        
        # Get the underlying PyTorch model
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            torch_model = model.model
            print("Found underlying PyTorch model")
        else:
            print("Warning: Could not find underlying PyTorch model, using model as is")
            torch_model = model
        
        # Set model to evaluation mode
        torch_model.eval()
        
        # Create dummy input (adjust shape as needed for the model)
        # RF-DETR typically expects a batch of images with shape [batch, 3, height, width]
        batch_size = 1
        height, width = 640, 640  # Default input size for RF-DETR Nano
        dummy_input = torch.randn(batch_size, 3, height, width, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a nested tensor for the input (if required by the model)
        # Some DETR variants expect a nested tensor with masks
        dummy_input_nested = nested_tensor_from_tensor_list([dummy_input[0]])
        
        # Define input and output names
        input_names = ["input"]
        output_names = ["boxes", "scores", "labels"]
        
        # Define dynamic axes for variable batch size
        dynamic_axes = {
            'input': {0: 'batch_size'},  # variable length axes
            'boxes': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'labels': {0: 'batch_size'}
        }
        
        # Export the model to ONNX
        print(f"Exporting model to ONNX format at {output_path}...")
        torch.onnx.export(
            torch_model,                  # model being run
            dummy_input_nested,           # model input (or a tuple for multiple inputs)
            output_path,                  # where to save the model
            export_params=True,           # store the trained parameter weights inside the model file
            opset_version=12,             # the ONNX version to export the model to
            do_constant_folding=True,     # whether to execute constant folding for optimization
            input_names=input_names,      # the model's input names
            output_names=output_names,    # the model's output names
            dynamic_axes=dynamic_axes     # variable length axes
        )
        
        print(f"Successfully exported ONNX model to {output_path}")
        
        print(f"Model exported successfully to {output_dir}")
        
        # Check if the ONNX file was created
        onnx_files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
        if onnx_files:
            print(f"Found exported ONNX file: {os.path.join(output_dir, onnx_files[0])}")
            return os.path.join(output_dir, onnx_files[0])
        else:
            print("Warning: No ONNX file found in the output directory")
            return None
            
    except Exception as e:
        print(f"Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Set paths
    model_path = "/Users/abhinavrai/Playground/snooker_data/rf-detr-nano-trained-model/weights.pt"
    output_dir = "/Users/abhinavrai/Playground/snooker_data/converted_models"
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Export the model
    onnx_path = export_rfdetr_nano(model_path, output_dir)
    
    if onnx_path:
        print(f"\nExport successful! ONNX model saved to: {onnx_path}")
    else:
        print("\nExport failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
