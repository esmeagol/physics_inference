import os
import sys
import torch
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

def inspect_rfdetr_model(model_path):
    """Inspect the RF-DETR model and its export capabilities."""
    print("Inspecting RF-DETR model...")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Try to create an instance of RFDETRNano
    print("\nCreating RFDETRNano instance...")
    try:
        # Create the model with default parameters
        model = RFDETRNano()
        print("Successfully created RFDETRNano instance!")
        
        # Print model information
        print("\nModel information:")
        print(f"Class names: {model.class_names}")
        print(f"Input size: {model.size}")
        print(f"Means: {model.means}")
        print(f"Stds: {model.stds}")
        
        # Check if the model has a predict method
        if hasattr(model, 'predict') and callable(model.predict):
            print("\nModel has a predict method")
        
        # Check if the model has an export method
        if hasattr(model, 'export') and callable(model.export):
            print("\nModel has an export method")
            # Print the export method signature
            import inspect
            sig = inspect.signature(model.export)
            print(f"Export method signature: {sig}")
            
            # Try to export the model
            print("\nAttempting to export the model...")
            output_path = "/Users/abhinavrai/Playground/snooker_data/converted_models/rfdetr_nano_exported"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                # Try exporting with different formats
                for fmt in ['onnx', 'torchscript']:
                    try:
                        print(f"\nExporting to {fmt.upper()}...")
                        model.export(output_path, format=fmt, imgsz=(640, 640))
                        print(f"Successfully exported to {fmt.upper()}!")
                        
                        # Check if the file was created
                        ext = '.onnx' if fmt == 'onnx' else '.pt'
                        if os.path.exists(output_path + ext):
                            size_mb = os.path.getsize(output_path + ext) / (1024 * 1024)
                            print(f"Exported {fmt.upper()} size: {size_mb:.2f} MB")
                    except Exception as e:
                        print(f"Error exporting to {fmt.upper()}: {e}")
                        
            except Exception as e:
                print(f"Error during export: {e}")
                
    except Exception as e:
        print(f"Error creating RFDETRNano instance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_path = "/Users/abhinavrai/Playground/snooker_data/rf-detr-nano-trained-model/weights.pt"
    inspect_rfdetr_model(model_path)
