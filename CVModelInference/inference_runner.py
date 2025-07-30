"""
InferenceRunner Interface Module for CVModelInference

This module defines the InferenceRunner interface for classes that implement
running image inference models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np


class InferenceRunner(ABC):
    """
    Abstract base class defining the interface for image inference model runners.
    
    This interface provides a consistent API for running inference with different
    model backends (PyTorch, TensorFlow, Roboflow, etc.).
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the inference runner.
        
        Args:
            **kwargs: Implementation-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def predict(self, 
                image: Union[str, np.ndarray], 
                **kwargs) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: Either a file path to an image or a numpy array
            **kwargs: Additional parameters for inference
                    
        Returns:
            Dictionary containing the prediction results
        """
        pass
    
    @abstractmethod
    def predict_batch(self, 
                     images: List[Union[str, np.ndarray]],
                     **kwargs) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of file paths or numpy arrays
            **kwargs: Additional parameters for inference
            
        Returns:
            List of prediction results for each image
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information (name, type, version, etc.)
        """
        pass
