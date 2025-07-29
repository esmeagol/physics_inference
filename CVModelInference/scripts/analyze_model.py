import torch
import sys
import pickle
from collections import OrderedDict
import torch.serialization

def analyze_model(model_path):
    try:
        # First try with weights_only=True (safer)
        try:
            model = torch.load(model_path, 
                             map_location=torch.device('cpu'),
                             weights_only=True)
            print("Model loaded successfully with weights_only=True")
        except Exception as e:
            print(f"Loading with weights_only=True failed: {e}")
            print("Trying with weights_only=False...")
            # If that fails, try with weights_only=False (less safe)
            model = torch.load(model_path, 
                             map_location=torch.device('cpu'),
                             weights_only=False)
            print("Model loaded successfully with weights_only=False")
        
        print("\nModel type:", type(model))
        
        # Print model structure
        if hasattr(model, '__dict__'):
            print("\nModel attributes:")
            print("-" * 50)
            for key, value in model.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    print(f"{key}: {type(value)}")
        
        # Print model state dict keys if available
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            print("\nModel state dict keys (first 20):")
            print("-" * 50)
            for i, key in enumerate(state_dict.keys()):
                if i < 20:  # Only show first 20 keys to avoid too much output
                    print(f"{key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'no shape'}")
                elif i == 20:
                    print(f"... and {len(state_dict) - 20} more keys")
        
        # Check if it's a state dict or a full model
        if isinstance(model, dict):
            print("\nModel appears to be a state dictionary.")
            print(f"Number of top-level keys: {len(model)}")
            
            # Analyze the 'model' key (should be the state dict)
            if 'model' in model and isinstance(model['model'], (dict, OrderedDict)):
                state_dict = model['model']
                print(f"\nState dictionary contains {len(state_dict)} parameters:")
                
                # Print parameter shapes and sizes
                total_params = 0
                print("\nParameter shapes:")
                print("-" * 50)
                for i, (name, param) in enumerate(state_dict.items()):
                    if i < 20:  # Only show first 20 parameters to avoid too much output
                        param_size = param.numel() if hasattr(param, 'numel') else 'unknown size'
                        if hasattr(param, 'shape'):
                            print(f"{name}: {param.shape} (size: {param_size:,})")
                        else:
                            print(f"{name}: {type(param)} (size: {param_size:,})")
                        if hasattr(param, 'numel'):
                            total_params += param.numel()
                    elif i == 20:
                        print(f"... and {len(state_dict) - 20} more parameters")
                
                print(f"\nTotal parameters: {total_params:,}")
            
            # Analyze the 'args' key if it exists
            if 'args' in model and hasattr(model['args'], '__dict__'):
                print("\nModel arguments:")
                print("-" * 50)
                args = model['args']
                for key, value in vars(args).items():
                    # Skip large objects that might not be helpful to print
                    if not key.startswith('_') and not isinstance(value, (list, dict, tuple)) and not (isinstance(value, str) and len(value) > 50):
                        print(f"{key}: {value}")
                
                # Print model architecture if available in args
                if hasattr(args, 'arch'):
                    print(f"\nModel architecture: {args.arch}")
                if hasattr(args, 'num_classes'):
                    print(f"Number of classes: {args.num_classes}")
                if hasattr(args, 'input_size'):
                    print(f"Input size: {args.input_size}" if hasattr(args.input_size, '__len__') else f"Input size: {args.input_size}x{args.input_size}")
        
        # Try to get model architecture information
        if hasattr(model, 'config') and hasattr(model.config, 'to_dict'):
            print("\nModel config:")
            print("-" * 50)
            print(model.config.to_dict())
            
    except Exception as e:
        print(f"Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_model.py <path_to_model.pt>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    print(f"Analyzing model at: {model_path}")
    analyze_model(model_path)
