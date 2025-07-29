import torch
import argparse
from collections import OrderedDict

def analyze_state_dict(state_dict):
    print(f"Total parameters: {sum(p.numel() for p in state_dict.values()):,}")
    print(f"Number of parameter tensors: {len(state_dict)}")
    
    # Group parameters by layer type
    param_groups = {}
    for name, param in state_dict.items():
        # Get the layer type (first part of the name)
        layer_type = name.split('.')[0]
        if layer_type not in param_groups:
            param_groups[layer_type] = []
        param_groups[layer_type].append((name, param.shape, param.numel()))
    
    # Print summary by layer type
    print("\nParameters by layer type:")
    print("-" * 80)
    for layer_type, params in param_groups.items():
        num_params = sum(p[2] for p in params)
        print(f"{layer_type}: {len(params)} tensors, {num_params:,} parameters")
    
    # Print detailed parameter shapes for the first few layers of each type
    print("\nDetailed parameter shapes (first 5 of each type):")
    print("-" * 80)
    for layer_type, params in param_groups.items():
        print(f"\n{layer_type}:")
        for name, shape, numel in params[:5]:  # Show first 5 of each type
            print(f"  {name}: {tuple(shape)} ({numel:,} parameters)")
        if len(params) > 5:
            print(f"  ... and {len(params) - 5} more {layer_type} parameters")

def main():
    parser = argparse.ArgumentParser(description='Analyze PyTorch model state dictionary')
    parser.add_argument('model_path', type=str, help='Path to the PyTorch model weights (.pt file)')
    args = parser.parse_args()
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=False)
        
        if 'model' in checkpoint and isinstance(checkpoint['model'], (dict, OrderedDict)):
            print(f"Analyzing model state dictionary from {args.model_path}")
            analyze_state_dict(checkpoint['model'])
        else:
            print("No model state dictionary found in the checkpoint.")
            print("Available keys in checkpoint:", checkpoint.keys())
            
    except Exception as e:
        print(f"Error analyzing model: {e}")
        raise

if __name__ == "__main__":
    main()
