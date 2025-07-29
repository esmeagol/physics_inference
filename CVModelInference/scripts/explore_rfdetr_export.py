#!/usr/bin/env python3
"""
RF-DETR Export Method Explorer

This is an EXPERIMENTAL utility script for exploring the export method of RF-DETR models.
It is intended for development and debugging purposes only, not for production use.

The script attempts to introspect the RF-DETR model's export method to understand
how it might be adapted for TensorFlow conversion.
"""

import os
import sys
import inspect
import torch

try:
    from rfdetr import RFDETRNano
except ImportError:
    print("Error: rfdetr package not available. This script requires the RF-DETR package.")
    print("Please install it or ensure it's in your Python path.")
    sys.exit(1)

def get_method_source(method):
    """Get the source code of a method."""
    try:
        return inspect.getsource(method)
    except (TypeError, OSError, IOError):
        return "Source not available"

def explore_export_method():
    """Explore the export method of RFDETRNano."""
    print("Exploring RFDETRNano export method...")
    
    # Create an instance of RFDETRNano
    model = RFDETRNano()
    
    # Get the export method
    export_method = getattr(model, 'export', None)
    if not export_method or not callable(export_method):
        print("Error: RFDETRNano does not have a callable export method")
        return
    
    # Print method signature
    print("\nExport method signature:")
    sig = inspect.signature(export_method)
    print(sig)
    
    # Get method source code
    print("\nExport method source code:")
    print(get_method_source(export_method))
    
    # Get method documentation
    print("\nExport method docstring:")
    print(export_method.__doc__ or "No docstring available")
    
    # Check if the method is from a parent class
    for base in model.__class__.__bases__:
        if hasattr(base, 'export'):
            print(f"\nExport method is defined in parent class: {base.__name__}")
            print("Parent class export method source:")
            print(get_method_source(base.export))
    
    # Try to get the module where the method is defined
    try:
        module = inspect.getmodule(export_method)
        if module:
            print(f"\nExport method is defined in module: {module.__name__}")
    except Exception as e:
        print(f"\nCould not determine module for export method: {e}")
    
    # Try to find the implementation file
    try:
        file_path = inspect.getsourcefile(export_method)
        if file_path:
            print(f"\nExport method is defined in file: {file_path}")
    except (TypeError, OSError, IOError) as e:
        print(f"\nCould not locate source file for export method: {e}")

if __name__ == "__main__":
    print("RF-DETR Export Method Explorer")
    print("=============================")
    explore_export_method()
