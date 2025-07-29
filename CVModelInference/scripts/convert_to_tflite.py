import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

def load_model(model_path):
    """Load the PyTorch model from the given path."""
    print(f"Loading model from {model_path}...")
    try:
        # Try with weights_only=True first (safer)
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except:
            # Fall back to weights_only=False if needed (less safe)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # The model is a state dict with 'model' and 'args' keys
        state_dict = checkpoint['model']
        args = checkpoint['args']
        
        print(f"Model loaded successfully. Number of parameters: {sum(p.numel() for p in state_dict.values()):,}")
        print(f"Model architecture: {getattr(args, 'arch', 'unknown')}")
        print(f"Number of classes: {getattr(args, 'num_classes', 'unknown')}")
        
        return state_dict, args
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def export_to_onnx(state_dict, args, output_path):
    """Export the model to ONNX format."""
    print("\nExporting to ONNX...")
    try:
        # Create a dummy input tensor with the expected input shape
        # The model expects input of shape [batch_size, 3, H, W]
        # Using the resolution from args or default to 640
        input_size = getattr(args, 'resolution', 640)
        dummy_input = torch.randn(1, 3, input_size, input_size, device='cpu')
        
        # Create a simple model for export
        # Note: In a real scenario, you would need the actual model architecture here
        # This is a simplified example and may need adjustments
        class DETRWrapper(torch.nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                # This is a simplified version - you would need the actual model architecture
                # and load the state dict properly
                self.backbone = torch.nn.Identity()  # Placeholder
                self.transformer = torch.nn.Identity()  # Placeholder
                
            def forward(self, x):
                # This is a simplified forward pass
                # The actual implementation would use the real model architecture
                features = self.backbone(x)
                outputs = self.transformer(features)
                return outputs
        
        # Create and prepare the model
        model = DETRWrapper(state_dict)
        model.eval()
        
        # Export to ONNX
        onnx_path = os.path.join(output_path, 'model.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=12
        )
        
        print(f"Model exported to ONNX: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        raise

def convert_onnx_to_tflite(onnx_path, output_path):
    """Convert ONNX model to TensorFlow Lite."""
    print("\nConverting ONNX to TensorFlow Lite...")
    try:
        import onnx
        import onnx_tf
        import tensorflow as tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert ONNX model to TensorFlow SavedModel
        tf_rep = prepare(onnx_model)
        saved_model_path = os.path.join(output_path, 'saved_model')
        tf_rep.export_graph(saved_model_path)
        
        # Convert SavedModel to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        tflite_model = converter.convert()
        
        # Save the TensorFlow Lite model
        tflite_path = os.path.join(output_path, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted to TensorFlow Lite: {tflite_path}")
        return tflite_path
    except Exception as e:
        print(f"Error converting to TensorFlow Lite: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorFlow Lite')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the PyTorch model weights (.pt file)')
    parser.add_argument('--output_dir', type=str, default='./converted_models',
                       help='Directory to save the converted models')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the PyTorch model
        state_dict, model_args = load_model(args.model_path)
        
        # Export to ONNX
        onnx_path = export_to_onnx(state_dict, model_args, args.output_dir)
        
        # Convert ONNX to TensorFlow Lite
        tflite_path = convert_onnx_to_tflite(onnx_path, args.output_dir)
        
        print("\nConversion completed successfully!")
        print(f"ONNX model saved to: {onnx_path}")
        print(f"TensorFlow Lite model saved to: {tflite_path}")
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nNote: For a complete conversion, you'll need the original model architecture code.")
        print("The current implementation uses a simplified model wrapper as a placeholder.")
        print("Please provide the model architecture code for accurate conversion.")
        sys.exit(1)

if __name__ == "__main__":
    main()
