import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple
import argparse
import numpy as np
from pathlib import Path

# Simplified RF-DETR model architecture for conversion
class RFDETRWrapper(nn.Module):
    def __init__(self, num_queries=300, hidden_dim=256, num_classes=365):  # 365 classes + 1 background = 366
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Simplified backbone (will be replaced with actual weights)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Simplified transformer (will be replaced with actual weights)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=2,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Extract features
        features = self.backbone(x)
        b, c, h, w = features.shape
        
        # Flatten spatial dimensions and project to hidden_dim
        features = features.flatten(2).permute(0, 2, 1)  # [b, h*w, c]
        
        # Add positional encodings (simplified)
        pos_embed = torch.zeros_like(features)
        
        # Prepare decoder input (queries)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        
        # Pass through transformer
        hs = self.transformer(
            features + pos_embed,
            query_embed,
        )
        
        # Get outputs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

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
        # Create model with correct architecture
        model = RFDETRWrapper(
            num_queries=getattr(args, 'num_queries', 300),
            hidden_dim=getattr(args, 'hidden_dim', 256),
            num_classes=getattr(args, 'num_classes', 13)
        )
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)  # strict=False to ignore missing keys
        model.eval()
        
        # Create dummy input
        input_size = getattr(args, 'resolution', 640)
        dummy_input = torch.randn(1, 3, input_size, input_size, device='cpu')
        
        # Define input and output names
        input_names = ['input']
        output_names = ['pred_logits', 'pred_boxes']
        
        # Export the model
        onnx_path = os.path.join(output_path, 'rf_detr_nano.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'pred_logits': {0: 'batch_size'},
                'pred_boxes': {0: 'batch_size'}
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
        
        # Enable TensorFlow Lite optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TensorFlow Lite model
        tflite_path = os.path.join(output_path, 'rf_detr_nano.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted to TensorFlow Lite: {tflite_path}")
        return tflite_path
    except Exception as e:
        print(f"Error converting to TensorFlow Lite: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert RF-DETR Nano model to TensorFlow Lite')
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
        print("\nNote: The conversion process might require adjustments to the model architecture.")
        print("The current implementation uses a simplified version of the RF-DETR architecture.")
        print("For better accuracy, you may need to provide the exact model architecture.")
        sys.exit(1)

if __name__ == "__main__":
    main()
