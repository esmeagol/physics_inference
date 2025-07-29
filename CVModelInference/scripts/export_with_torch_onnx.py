import os
import torch
import numpy as np
from rfdetr import RFDETRNano

def export_with_torch_onnx(model_path, output_dir):
    """
    Export RF-DETR Nano model to ONNX format using torch.onnx.export
    
    Args:
        model_path: Path to the RF-DETR Nano model weights (.pt file)
        output_dir: Directory to save the exported ONNX model
    """
    print(f"Exporting RF-DETR Nano model from {model_path} to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rfdetr_nano_torch_export.onnx")
    
    try:
        # Initialize the model with pretrained weights
        print("Initializing RFDETRNano with pretrained weights...")
        model = RFDETRNano(pretrain_weights=model_path)
        print("Model initialized successfully!")
        
        # Get the underlying PyTorch model from the RFDETRNano wrapper
        # From our inspection, the model has a 'model' attribute that contains the actual PyTorch model
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            torch_model = model.model.model
            print("Found underlying PyTorch model for export")
            
            # Set the model to evaluation mode
            torch_model.eval()
            
            # For ONNX export, it's more reliable to use CPU
            # Move model to CPU for export (MPS has limited support for ONNX export)
            if hasattr(torch_model, 'to'):
                print("Model moved to CPU for ONNX export")
                torch_model = torch_model.to('cpu')
            
            # Import NestedTensor class
            from rfdetr.util.misc import NestedTensor
            
            # Patch the Linear layer to handle NestedTensor inputs
            original_linear = torch.nn.functional.linear
            
            def patched_linear(input, weight, bias=None):
                if isinstance(input, NestedTensor):
                    # If input is NestedTensor, use its tensors
                    return original_linear(input.tensors, weight, bias)
                return original_linear(input, weight, bias)
            
            # Apply the patch
            torch.nn.functional.linear = patched_linear
            
            # Patch the model's forward method to handle NestedTensor inputs
            original_forward = torch_model.forward
            
            def patched_forward(self, samples):
                # If input is a regular tensor, convert to NestedTensor
                if isinstance(samples, torch.Tensor):
                    # Create a dummy mask (assuming no padding for simplicity)
                    mask = torch.ones(
                        (samples.shape[0], samples.shape[2], samples.shape[3]),
                        dtype=torch.bool,
                        device=samples.device
                    )
                    samples = NestedTensor(samples, mask)
                
                # Call the original forward method
                result = original_forward(samples)
                
                # Convert NestedTensor outputs to regular tensors
                def convert_output(output):
                    if isinstance(output, NestedTensor):
                        return output.tensors
                    elif isinstance(output, dict):
                        return {k: convert_output(v) for k, v in output.items()}
                    elif isinstance(output, (list, tuple)):
                        return type(output)(convert_output(v) for v in output)
                    return output
                
                return convert_output(result)
            
            # Apply the patch to the model
            torch_model.forward = patched_forward.__get__(torch_model, type(torch_model))
            
            # Patch the model to replace the unsupported operator
            def patch_unsupported_ops(module):
                for name, child in module.named_children():
                    # Recursively apply to all submodules
                    patch_unsupported_ops(child)
                    
                    # Check if this is a module that might contain the unsupported op
                    if hasattr(child, 'forward'):
                        original_forward = child.forward
                        
                        def patched_forward(*args, **kwargs):
                            try:
                                return original_forward(*args, **kwargs)
                            except Exception as e:
                                if "_upsample_bicubic2d_aa" in str(e):
                                    print(f"Detected _upsample_bicubic2d_aa in {name}, patching...")
                                    # Replace with a compatible upsampling operation
                                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                                        input_tensor = args[0]
                                        # Use interpolate with bicubic mode as a fallback
                                        return torch.nn.functional.interpolate(
                                            input_tensor, 
                                            scale_factor=2.0,  # Default scale factor, adjust if needed
                                            mode='bicubic',
                                            align_corners=False
                                        )
                                raise e
                        
                        # Replace the forward method with our patched version
                        child.forward = patched_forward
            
            print("Patching model to replace unsupported operations...")
            patch_unsupported_ops(torch_model)
            
            # Create a NestedTensor for the input with the correct resolution for RFDETRNano
            # RFDETRNano expects 384x384 input resolution according to its config
            batch_size = 1
            height, width = 384, 384  # RFDETRNano's expected input resolution
            
            print(f"Creating input with shape: (batch={batch_size}, channels=3, height={height}, width={width})")
            
            try:
                # Create a dummy image tensor with values in [0, 1] range and normalize
                dummy_tensor = torch.rand(batch_size, 3, height, width, device=torch.device('cpu'))
                
                # Normalize the input (assuming ImageNet mean and std)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                dummy_tensor = (dummy_tensor - mean) / std
                
                # Create a mask (1 for valid pixels, 0 for padding)
                # For a single image with no padding, this is all ones
                dummy_mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=torch.device('cpu'))
                
                # Create the NestedTensor
                from rfdetr.util.misc import NestedTensor
                nested_tensor = NestedTensor(dummy_tensor, dummy_mask)
                
            except Exception as e:
                print(f"Error creating input tensor: {str(e)}")
                print("Troubleshooting steps:")
                print("1. Check if the input dimensions are correct (should be 384x384 for RFDETRNano)")
                print("2. Verify tensor values are in the expected range (normalized with ImageNet stats)")
                print("3. Ensure the device (CPU/GPU) matches between model and input")
                raise
            
            # Try using the forward_export method which is specifically designed for export
            print("Attempting to use forward_export method for ONNX export...")
            try:
                # The forward_export method expects a tensor directly, not a NestedTensor
                dummy_tensor = nested_tensor.tensors
                
                # Test forward_export with the tensor
                with torch.no_grad():
                    output = torch_model.forward_export(dummy_tensor)
                
                # Print output information
                print("Forward export successful with direct tensor input")
                if isinstance(output, (list, tuple)):
                    print(f"Output is a sequence of length {len(output)}")
                    for i, out in enumerate(output):
                        print(f"  Output {i} type: {type(out)}")
                        if hasattr(out, 'shape'):
                            print(f"    shape: {out.shape}")
                elif hasattr(output, 'shape'):
                    print(f"Output shape: {output.shape}")
                else:
                    print(f"Output type: {type(output)}")
                
                # Use the tensor directly for ONNX export
                dummy_input = dummy_tensor
                
            except Exception as e:
                print(f"Forward export failed with error: {str(e)}")
                print("Falling back to standard forward pass with NestedTensor...")
                
                # Fall back to standard forward pass
                with torch.no_grad():
                    output = torch_model(nested_tensor)
                
                # Print output information
                print("Standard forward pass successful with NestedTensor input")
                if isinstance(output, dict):
                    print("Output is a dictionary with keys:", output.keys())
                    for k, v in output.items():
                        if hasattr(v, 'shape'):
                            print(f"  {k}: shape {v.shape}")
                        else:
                            print(f"  {k}: {type(v)}")
                elif isinstance(output, (list, tuple)):
                    print(f"Output is a sequence of length {len(output)}")
                    for i, out in enumerate(output):
                        print(f"  Output {i}: {type(out)}")
                        if hasattr(out, 'shape'):
                            print(f"    shape: {out.shape}")
                else:
                    print(f"Output type: {type(output)}")
                    if hasattr(output, 'shape'):
                        print(f"  shape: {output.shape}")
                
                # Use NestedTensor for ONNX export
                dummy_input = nested_tensor
            
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
            
            # Export the model to ONNX with a newer opset version
            print(f"Exporting model to ONNX format at {output_path}...")
            
            # Try with a newer opset version that might support the bicubic upsampling operator
            # Opset 16 is known to have better support for vision operators
            opset_version = 16
            print(f"Using ONNX opset version: {opset_version}")
            
            # Add a custom symbolic function for the unsupported operator
            def register_custom_symbolic():
                from torch.onnx.symbolic_helper import parse_args
                
                # Try to import the standard upsample_bilinear2d as a fallback
                try:
                    from torch.onnx.symbolic_opset11 import upsample_bilinear2d
                    
                    # Define a custom symbolic function for _upsample_bicubic2d_aa
                    @parse_args('v', 'v', 'is', 'b', 'f')
                    def _upsample_bicubic2d_aa_symbolic(g, input, output_size, align_corners, scale_factors, scale_f):
                        # Fall back to bilinear upsampling as an approximation
                        # This is a simpler alternative to bicubic
                        return upsample_bilinear2d(g, input, output_size, align_corners, scale_factors)
                    
                    # Register the custom symbolic function
                    from torch.onnx import register_custom_op_symbolic
                    register_custom_op_symbolic('aten::_upsample_bicubic2d_aa', _upsample_bicubic2d_aa_symbolic, opset_version)
                    print("Registered custom symbolic function for _upsample_bicubic2d_aa")
                    
                except ImportError as e:
                    print(f"Warning: Could not import required symbolic functions: {e}")
                    print("Trying to use a different approach...")
                    
                    # Try to use a different approach for older PyTorch versions
                    try:
                        from torch.onnx.symbolic_helper import _interpolate_helper
                        
                        @parse_args('v', 'v', 'is')
                        def _upsample_bicubic2d_aa_symbolic_legacy(g, input, output_size, align_corners, scale_factors):
                            # Use the helper function with mode='bicubic' and align_corners=False
                            return _interpolate_helper(g, input, None, None, output_size, None, 'bicubic', align_corners, None, scale_factors)
                        
                        from torch.onnx import register_custom_op_symbolic
                        register_custom_op_symbolic('aten::_upsample_bicubic2d_aa', _upsample_bicubic2d_aa_symbolic_legacy, opset_version)
                        print("Registered legacy custom symbolic function for _upsample_bicubic2d_aa")
                        
                    except Exception as e2:
                        print(f"Warning: Could not register legacy symbolic function: {e2}")
                        print("Proceeding without custom symbolic function - this may fail")
            
            # Register the custom symbolic function
            register_custom_symbolic()
            
            # Export the model
            torch.onnx.export(
                torch_model,            # model being run
                dummy_input,            # model input (or a tuple for multiple inputs)
                output_path,            # where to save the model
                export_params=True,     # store the trained parameter weights inside the model file
                opset_version=opset_version,  # use the newer opset version
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=input_names,    # the model's input names
                output_names=output_names,  # the model's output names
                dynamic_axes=dynamic_axes,  # variable length axes
                training=torch.onnx.TrainingMode.EVAL,  # export the model in inference mode
                verbose=True,           # print verbose output
                custom_opsets=None,     # use default opsets
                export_modules_as_functions=False
            )
        else:
            raise RuntimeError("Could not find underlying PyTorch model in the RFDETRNano wrapper")
        
        print(f"Successfully exported ONNX model to {output_path}")
        return output_path
        
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
    onnx_path = export_with_torch_onnx(model_path, output_dir)
    
    if onnx_path:
        print(f"\nExport successful! ONNX model saved to: {onnx_path}")
    else:
        print("\nExport failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
