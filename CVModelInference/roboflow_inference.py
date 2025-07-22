"""
Roboflow Inference Module for CVModelInference

This module provides functionality to run inference using Roboflow models
through a local inference server.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Union
import numpy as np
import cv2

class RoboflowInference:
    """
    A class to handle Roboflow model inference using a local inference server.
    
    This class provides methods to perform inference on images using a Roboflow
    model running on a local inference server.
    """
    
    def __init__(
        self,
        api_key: str,
        model_id: str,
        version: int,
        server_url: str = "http://localhost:9001",
        confidence: float = 0.5,
        overlap: float = 0.5
    ) -> None:
        """
        Initialize the Roboflow inference client.
        
        Args:
            api_key: Roboflow API key
            model_id: Roboflow model ID
            version: Model version number
            server_url: URL of the local Roboflow inference server 
                       (default: http://localhost:9001)
            confidence: Minimum confidence threshold for predictions (0-1)
            overlap: Maximum overlap between predictions (0-1)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.version = version
        self.server_url = server_url.rstrip('/')
        self.confidence = confidence
        self.overlap = overlap
        
        # Construct the inference URL
        self.inference_url = f"{self.server_url}/{self.model_id}/{self.version}"
        
        # Set up the headers for the API requests
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def predict(
        self, 
        image: Union[str, np.ndarray], 
        **kwargs
    ) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: Either a file path to an image or a numpy array
            **kwargs: Additional parameters to override the defaults
                    (confidence, overlap, etc.)
                    
        Returns:
            Dictionary containing the prediction results
        """
        # Handle numpy array input
        if isinstance(image, np.ndarray):
            # Convert numpy array to bytes
            _, img_encoded = cv2.imencode('.jpg', image)
            image_data = img_encoded.tobytes()
            files = {'file': ('image.jpg', image_data, 'image/jpeg')}
        # Handle file path input
        elif os.path.isfile(image):
            files = {'file': open(image, 'rb')}
        else:
            raise ValueError("Input must be a valid file path or numpy array")
        
        # Prepare parameters
        params = {
            'api_key': self.api_key,
            'confidence': kwargs.get('confidence', self.confidence),
            'overlap': kwargs.get('overlap', self.overlap),
        }
        
        try:
            # Make the prediction request
            response = requests.post(
                self.inference_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=params
            )
            response.raise_for_status()
            
            # Parse and return the JSON response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error making prediction request: {str(e)}")
        finally:
            # Close the file if we opened it
            if isinstance(image, str) and 'files' in locals():
                files['file'].close()
    
    def predict_batch(
        self, 
        images: List[Union[str, np.ndarray]],
        **kwargs
    ) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of file paths or numpy arrays
            **kwargs: Additional parameters to pass to predict()
            
        Returns:
            List of prediction results for each image
        """
        return [self.predict(img, **kwargs) for img in images]
    
    def visualize_predictions(
        self,
        image: Union[str, np.ndarray],
        predictions: Dict,
        output_path: Optional[str] = None
    ) -> np.ndarray:
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
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label and confidence
            label = f"{pred['class']} {pred['confidence']:.2f}"
            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        # Save the visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, img)
            
        return img
