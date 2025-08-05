"""Roboflow Local Inference Module.

This module provides functionality to run inference using Roboflow models
through the Roboflow Python SDK.
"""

import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from roboflow import Roboflow


class RoboflowLocal:
    """Roboflow inference model using the Roboflow Python SDK."""
    
    def __init__(
        self,
        api_key: str,
        model_id: str,
        version: Optional[int] = None,
        confidence: float = 0.5,
        overlap: float = 0.5
    ) -> None:
        """Initialize the Roboflow inference model.
        
        Args:
            api_key: Roboflow API key
            model_id: Model ID (e.g., 'workspace/model_id' or 'model_id')
            version: Model version number (default: latest)
            confidence: Confidence threshold (0-1)
            overlap: Overlap threshold for NMS (0-1)
        """
        if not 0 <= confidence <= 1 or not 0 <= overlap <= 1:
            raise ValueError("Confidence and overlap must be between 0 and 1")
            
        # Initialize Roboflow and load model
        rf = Roboflow(api_key=api_key)
        
        # Parse model_id to get workspace if present
        if '/' in model_id:
            workspace, model_id = model_id.split('/')
            project = rf.workspace(workspace).project(model_id)
        else:
            project = rf.project(model_id)
        
        self.model = project.version(version).model if version else project.versions()[0].model
        
        # Store config
        self.model_id = model_id
        self.version = str(version) if version else str(project.versions()[0].version)
        self.confidence = confidence
        self.overlap = overlap
        self.classes = getattr(project, 'classes', [])

    def predict(
        self,
        image: Union[str, np.ndarray],
        confidence: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run prediction on an image.
        
        Args:
            image: Image path or numpy array (BGR format)
            confidence: Optional confidence threshold override
            
        Returns:
            Dictionary containing predictions and image info
        """
        conf = confidence if confidence is not None else self.confidence
        
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = self.model.predict(image, confidence=conf * 100).json()
            else:
                if not os.path.isfile(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                result = self.model.predict(image, confidence=conf * 100).json()
            
            # Format predictions
            predictions = [{
                'x': p['x'],
                'y': p['y'],
                'width': p['width'],
                'height': p['height'],
                'confidence': p['confidence'],
                'class': p['class'],
                'class_id': p.get('class_id', 0)
            } for p in result.get('predictions', [])]
            
            return {
                'predictions': predictions,
                'image': {
                    'width': result.get('image', {}).get('width', 0),
                    'height': result.get('image', {}).get('height', 0)
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Run prediction on multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of prediction results
        """
        return [self.predict(img, **kwargs) for img in images]

    def visualize(
        self,
        image: Union[str, np.ndarray],
        predictions: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Draw predictions on an image.
        
        Args:
            image: Input image (path or numpy array)
            predictions: Results from predict()
            output_path: Optional path to save the result
            
        Returns:
            Image with drawn predictions (BGR format)
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image.copy()
        
        # Draw predictions
        for pred in predictions.get('predictions', []):
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            w, h = int(pred['width']), int(pred['height'])
            
            # Draw box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{pred['class']} {pred['confidence']:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
            
        return img
