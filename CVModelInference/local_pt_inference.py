"""
LocalPT Inference Module for CVModelInference

This module provides functionality to run inference using locally stored
PyTorch model weights through Ultralytics.
"""

import os
from typing import Dict, List, Optional, Union, Any
import numpy as np
import cv2
from ultralytics import YOLO

from .inference_runner import InferenceRunner


class LocalPT(InferenceRunner):
    """
    Class for running inference with local PyTorch models using Ultralytics.
    
    This class implements the InferenceRunner interface for local PyTorch models.
    """
    
    def __init__(self, 
                model_path: str,
                confidence: float = 0.5,
                iou: float = 0.5,
                **kwargs):
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
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
            
        # Store model info
        self.model_name = os.path.basename(model_path)
        
    def predict(self, 
               image: Union[str, np.ndarray], 
               **kwargs) -> Dict:
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
            # Run inference
            results = self.model(image, conf=conf, iou=iou, verbose=False)
            
            # Convert to standardized format
            return self._convert_to_standard_format(results[0])
            
        except Exception as e:
            raise RuntimeError(f"Error during inference: {str(e)}")
    
    def predict_batch(self, 
                     images: List[Union[str, np.ndarray]],
                     **kwargs) -> List[Dict]:
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
            'framework': 'ultralytics',
            'path': self.model_path,
            'confidence': self.confidence,
            'iou': self.iou
        }
    
    def _convert_to_standard_format(self, result) -> Dict:
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
