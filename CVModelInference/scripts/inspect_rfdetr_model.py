import os
import torch
import inspect
import pprint
from rfdetr import RFDETRNano

def inspect_rfdetr_model(model_path):
    """Inspect the RFDETRNano model to understand its structure."""
    print("Inspecting RFDETRNano model...")
    
    # Initialize the model with pretrained weights
    print("\nInitializing RFDETRNano with pretrained weights...")
    model = RFDETRNano(pretrain_weights=model_path)
    print("Model initialized successfully!")
    
    # Print model type and attributes
    print("\nModel type:", type(model))
    print("\nModel attributes:", dir(model))
    
    # Check if model has a model attribute that might be the PyTorch model
    if hasattr(model, 'model'):
        print("\nModel has 'model' attribute:", type(model.model))
        print("Model.model attributes:", dir(model.model))
    
    # Check for common PyTorch model attributes
    for attr in ['state_dict', 'parameters', 'modules', 'children']:
        has_attr = hasattr(model, attr)
        print(f"\nModel has '{attr}': {has_attr}")
        if has_attr:
            try:
                result = getattr(model, attr)()
                print(f"{attr} type: {type(result)}")
                if attr == 'state_dict':
                    print(f"State dict keys: {list(result.keys())[:5]}...")
            except Exception as e:
                print(f"Error calling {attr}: {e}")
    
    # Inspect the model attribute in detail
    if hasattr(model, 'model'):
        print("\nInspecting model.model in detail...")
        model_instance = model.model
        
        # Print model instance attributes
        print("\nModel instance attributes:")
        for attr in dir(model_instance):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(model_instance, attr)
                    print(f"- {attr}: {type(attr_value).__name__}")
                except:
                    print(f"- {attr}: <error accessing>")
        
        # Check if model has a model attribute that might be the PyTorch model
        if hasattr(model_instance, 'model'):
            print("\nmodel_instance has 'model' attribute:", type(model_instance.model))
            print("model_instance.model attributes:", dir(model_instance.model))
            
            # If it's a PyTorch model, print its structure
            if isinstance(model_instance.model, torch.nn.Module):
                print("\nPyTorch model structure:")
                print(model_instance.model)
                
                # Print model's state dict keys
                try:
                    state_dict = model_instance.model.state_dict()
                    print("\nModel state dict keys:")
                    for key in list(state_dict.keys())[:10]:  # Print first 10 keys
                        print(f"- {key}: {state_dict[key].shape}")
                    if len(state_dict) > 10:
                        print(f"... and {len(state_dict) - 10} more")
                except Exception as e:
                    print(f"Error accessing model state dict: {e}")
        
        # Check for inference_model attribute which might be the actual PyTorch model
        if hasattr(model_instance, 'inference_model'):
            print("\nmodel_instance has 'inference_model' attribute:", 
                  type(model_instance.inference_model))
            if isinstance(model_instance.inference_model, torch.nn.Module):
                print("\nInference model structure:")
                print(model_instance.inference_model)
        
        # Check for any methods that might help with export
        print("\nChecking for export-related methods...")
        for name, method in inspect.getmembers(model_instance, inspect.ismethod):
            if 'export' in name.lower() or 'onnx' in name.lower():
                print(f"Found method: {name}")
                print(f"  Signature: {inspect.signature(method)}")
    
    # Check for any other attributes that might be PyTorch modules
    print("\nSearching for PyTorch modules in the RFDETRNano instance...")
    for name, module in model.__dict__.items():
        if isinstance(module, torch.nn.Module):
            print(f"Found PyTorch module: {name} - {module.__class__.__name__}")
    
    # Check for any methods that might return a model
    print("\nChecking for methods that might return a model...")
    for name, method in inspect.getmembers(model, inspect.ismethod):
        if any(keyword in name.lower() for keyword in ['model', 'export', 'onnx', 'pytorch']):
            try:
                sig = inspect.signature(method)
                print(f"{name}{sig}")
                if len(sig.parameters) == 1:  # Only try methods with no required args
                    result = method()
                    print(f"  Returned: {type(result).__name__}")
                    if isinstance(result, torch.nn.Module):
                        print("  Found PyTorch model!")
            except Exception as e:
                print(f"  Error calling {name}(): {e}")

if __name__ == "__main__":
    model_path = "/Users/abhinavrai/Playground/snooker_data/rf-detr-nano-trained-model/weights.pt"
    inspect_rfdetr_model(model_path)
