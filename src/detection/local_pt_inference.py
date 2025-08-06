"""
LocalPT Inference Module for CVModelInference

This module provides functionality to run inference using locally stored
PyTorch model weights through Ultralytics.
"""

import os
from typing import Dict, List, Optional, Union, Any
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from .inference_runner import InferenceRunner

try:
    from rfdetr import RFDETRNano
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False


class LocalPT(InferenceRunner):
    """
    Class for running inference with local PyTorch models using Ultralytics.
    
    This class implements the InferenceRunner interface for local PyTorch models.
    """
    
    def __init__(self, 
                model_path: str,
                confidence: float = 0.5,
                iou: float = 0.5,
                **kwargs: Any) -> None:
        """
        Initialize the local PyTorch inference runner.
        
        Args:
            model_path: Path to the local PyTorch model weights (.pt file)
            confidence: Minimum confidence threshold for predictions (0-1)
            iou: IoU threshold for NMS (0-1)
            **kwargs: Additional parameters for model initialization
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.model_name = os.path.basename(model_path)
        
        # Detect model type and load accordingly
        if self._is_rfdetr_model(model_path):
            self.model_type = 'rf-detr'
            self.model = self._load_rfdetr_model(model_path)
        else:
            self.model_type = 'yolo'
            try:
                self.model = YOLO(model_path)
                print(f"Loaded YOLO model from {model_path}")
            except Exception as e:
                raise RuntimeError(f"Error loading YOLO model: {str(e)}")
        
    def predict(self, 
               image: Union[str, np.ndarray], 
               **kwargs: Any) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Either a file path to an image or a numpy array
            **kwargs: Additional parameters to override the defaults
                    (confidence, iou, etc.)
                    
        Returns:
            Dictionary containing the prediction results in a standardized format
        """
        conf = kwargs.get('confidence', self.confidence)
        iou = kwargs.get('iou', self.iou)
        
        try:
            if self.model_type == 'rf-detr':
                return self._predict_rfdetr(image, conf=conf)
            else:
                # Run YOLO inference
                results = self.model(image, conf=conf, iou=iou, verbose=False)
                return self._convert_to_standard_format(results[0])
            
        except Exception as e:
            raise RuntimeError(f"Error during inference: {str(e)}")
    
    def predict_batch(self, 
                     images: List[Union[str, np.ndarray]],
                     **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of file paths or numpy arrays
            **kwargs: Additional parameters to pass to predict()
            
        Returns:
            List of prediction results for each image
        """
        return [self.predict(img, **kwargs) for img in images]
    
    def visualize_predictions(self,
                             image: Union[str, np.ndarray],
                             predictions: Dict,
                             output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize the predictions on the input image.
        
        Args:
            image: Input image (file path or numpy array)
            predictions: Prediction results from predict()
            output_path: Optional path to save the visualization
            
        Returns:
            Image with predictions visualized
        """
        # Load the image if it's a file path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Draw each prediction
        for pred in predictions.get('predictions', []):
            # Extract bounding box coordinates
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            w = int(pred['width'])
            h = int(pred['height'])
            
            # Determine color based on class (simple hash function for consistent colors)
            class_name = pred['class']
            color_hash = hash(class_name) % 0xFFFFFF
            color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Add label and confidence
            label = f"{class_name} {pred['confidence']:.2f}"
            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Save the visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, img)
            
        return img
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'name': self.model_name,
            'type': 'pytorch',
            'framework': 'rf-detr' if self.model_type == 'rf-detr' else 'ultralytics',
            'path': self.model_path,
            'confidence': self.confidence,
            'iou': self.iou
        }
    
    def _convert_to_standard_format(self, result: Any) -> Dict[str, Any]:
        """
        Convert Ultralytics result to standardized format.
        
        Args:
            result: Ultralytics detection result
            
        Returns:
            Dictionary with standardized prediction format
        """
        # Extract boxes, classes, and confidences
        boxes = result.boxes
        
        predictions = []
        for i in range(len(boxes)):
            # Get box coordinates (xywh format, normalized)
            x, y, w, h = boxes.xywh[i].tolist()
            
            # Get class and confidence
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names[cls_id]
            conf = float(boxes.conf[i].item())
            
            # Create prediction entry
            pred = {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'confidence': conf,
                'class': cls_name,
                'class_id': cls_id
            }
            predictions.append(pred)
        
        # Create standardized output format
        output = {
            'predictions': predictions,
            'image': {
                'width': result.orig_shape[1],
                'height': result.orig_shape[0]
            },
            'model': self.model_name
        }
        
        return output

    def _is_rfdetr_model(self, model_path: str) -> bool:
        """
        Check if the model is an RF-DETR model by inspecting checkpoint contents.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if RF-DETR model, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Look for RF-DETR specific keys in state_dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model', checkpoint)
                if isinstance(state_dict, dict):
                    keys = list(state_dict.keys())
                    # Check for transformer/decoder components typical of RF-DETR
                    rfdetr_indicators = ['transformer', 'decoder', 'query_embed', 'class_embed', 'bbox_embed']
                    return any(indicator in str(keys) for indicator in rfdetr_indicators)
            
            return False
        except Exception:
            return False

    def _load_rfdetr_model(self, model_path: str) -> Any:
        """
        Load RF-DETR model using RFDETRNano.
        
        Args:
            model_path: Path to the RF-DETR model file
            
        Returns:
            Loaded RF-DETR model
        """
        if not RFDETR_AVAILABLE:
            raise ImportError("RF-DETR package not available. Install with: pip install rfdetr")
        
        try:
            # First try with pretrain_weights
            model = RFDETRNano(pretrain_weights=model_path)
            print(f"Loaded RF-DETR model from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load RF-DETR with pretrain_weights: {e}")
            # Try alternative loading approach
            try:
                # Load checkpoint manually and create model without pretrained weights
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Determine number of classes from checkpoint
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    # Look for class embedding to determine num_classes
                    for key, tensor in state_dict.items():
                        if 'class_embed' in key and tensor.dim() == 2:
                            num_classes = tensor.shape[0]
                            break
                    else:
                        num_classes = 90  # Default COCO classes
                else:
                    num_classes = 90
                
                # Create model with correct number of classes
                model = RFDETRNano(num_classes=num_classes)
                print(f"Created RF-DETR model with {num_classes} classes")
                
                # Try to load state dict with strict=False
                if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
                    model.model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded RF-DETR model from {model_path} (manual loading)")
                
                return model
            except Exception as e2:
                raise RuntimeError(f"Error loading RF-DETR model: {str(e2)}")

    def _predict_rfdetr(self, image: Union[str, np.ndarray], conf: float = 0.5) -> Dict[str, Any]:
        """
        Run inference using RF-DETR model.
        
        Args:
            image: Input image (file path or numpy array)
            conf: Confidence threshold
            
        Returns:
            Dictionary with standardized prediction format
        """
        # Load image if it's a file path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Run RF-DETR inference - use predict method if available
        if hasattr(self.model, 'predict'):
            results = self.model.predict(img)
        else:
            # Fallback to direct call
            results = self.model(img)
        
        # Convert to standard format
        return self._convert_rfdetr_to_standard_format(results, img_width, img_height, conf)

    def _convert_rfdetr_to_standard_format(self, results: Any, img_width: int, img_height: int, conf_threshold: float) -> Dict[str, Any]:
        """
        Convert RF-DETR results to standardized format.
        
        Args:
            results: RF-DETR detection results (supervision.Detections format)
            img_width: Original image width
            img_height: Original image height
            conf_threshold: Confidence threshold for filtering
            
        Returns:
            Dictionary with standardized prediction format
        """
        predictions = []
        
        # Handle supervision.Detections format (RF-DETR output)
        if hasattr(results, 'xyxy') and hasattr(results, 'confidence') and hasattr(results, 'class_id'):
            boxes = results.xyxy
            confidences = results.confidence
            class_ids = results.class_id
            
            # Define class names mapping (based on snooker dataset)
            class_names = {
                1: 'white-ball',
                37: 'red-ball', 
                62: 'green-ball',
                15: 'blue-ball',
                72: 'yellow-ball',
            }
            
            for i in range(len(boxes)):
                score = float(confidences[i])
                if score < conf_threshold:
                    continue
                    
                # Convert box coordinates from xyxy to center format
                x1, y1, x2, y2 = boxes[i].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                class_id = int(class_ids[i])
                class_name = class_names.get(class_id, f'class_{class_id}')
                
                pred = {
                    'x': cx,
                    'y': cy,
                    'width': w,
                    'height': h,
                    'confidence': score,
                    'class': class_name,
                    'class_id': class_id
                }
                predictions.append(pred)
        
        # Create standardized output format
        output = {
            'predictions': predictions,
            'image': {
                'width': img_width,
                'height': img_height
            },
            'model': self.model_name
        }
        
        return output
