import os
import sys
import pickle
import torch
import argparse
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# Add parent directory to path to import utility modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_rfdetr_nano(model_path):
    """Load the RF-DETR Nano model from the given checkpoint."""
    print(f"Loading RF-DETR Nano model from {model_path}...")
    
    # First, try to load the model directly using the RFDETRNano class
    try:
        print("Attempting to load model using RFDETRNano.from_pretrained...")
        model = RFDETRNano.from_pretrained(model_path)
        print("Model loaded successfully using from_pretrained!")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except (AttributeError, RuntimeError) as e:
        print(f"Could not load using from_pretrained: {e}")
        print("Falling back to manual loading...")
    
    # If direct loading fails, try manual loading
    try:
        # Create the model with the correct architecture
        # The model will be initialized with random weights
        model = RFDETRNano(num_classes=90)  # COCO has 90 classes
        
        # Load the checkpoint
        print("Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel or DistributedDataParallel wrapper
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Manually assign the state dict to the model's parameters
        print("Assigning state dict to model parameters...")
        model_state_dict = model.state_dict()
        
        # Update the model's state dict with the loaded weights
        for name, param in state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Warning: Shape mismatch for {name}: expected {model_state_dict[name].shape}, got {param.shape}")
            else:
                print(f"Warning: Parameter {name} not found in model")
        
        # Load the updated state dict back to the model
        model.load_state_dict(model_state_dict, strict=False)
        
        print("Model loaded successfully!")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nAvailable methods in RFDETRNano:")
        print(dir(RFDETRNano))
        raise

def export_to_onnx(model, output_path, input_size=640):
    """
    Export the RF-DETR model to ONNX format using the model's export method.
    
    Args:
        model: The RF-DETR model instance
        output_path: Path to save the ONNX model
        input_size: Input image size (assumed square)
    """
    print(f"\nExporting RF-DETR model to ONNX with input size {input_size}x{input_size}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        # Check if the model has an export method
        if hasattr(model, 'export') and callable(getattr(model, 'export')):
            print("Using model's export method...")
            # The export method might have its own parameters
            model.export(output_path, format='onnx', imgsz=(input_size, input_size))
        else:
            # Fallback to standard PyTorch export (unlikely to work but trying anyway)
            print("Warning: Using standard PyTorch ONNX export (might not work)")
            model.eval()
            dummy_input = torch.randn(1, 3, input_size, input_size, device='cpu')
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['pred_logits', 'pred_boxes'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'pred_logits': {0: 'batch_size'},
                    'pred_boxes': {0: 'batch_size'}
                },
                opset_version=12,
                do_constant_folding=True,
                export_params=True,
                verbose=True
            )
        
        # Verify the ONNX model was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Successfully exported ONNX model to {output_path}")
            print(f"ONNX model size: {file_size:.2f} MB")
            
            # Try to load the ONNX model to verify it's valid
            try:
                import onnx
                model_onnx = onnx.load(output_path)
                onnx.checker.check_model(model_onnx)
                print("ONNX model check passed!")
            except Exception as e:
                print(f"Warning: ONNX model verification failed: {e}")
        else:
            raise RuntimeError("Failed to create ONNX model file")
            
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        print("\nAvailable methods in the model:")
        print([m for m in dir(model) if not m.startswith('_')])
        raise
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Load RF-DETR Nano model and export to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the RF-DETR Nano model weights (.pt file)')
    parser.add_argument('--output_path', type=str, default='rfdetr_nano.onnx',
                       help='Path to save the ONNX model')
    parser.add_argument('--input_size', type=int, default=640,
                       help='Input image size (default: 640)')
    
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_rfdetr_nano(args.model_path)
        
        # Export to ONNX
        export_to_onnx(model, args.output_path, args.input_size)
        
        print("\nConversion completed successfully!")
        print(f"ONNX model saved to: {os.path.abspath(args.output_path)}")
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
