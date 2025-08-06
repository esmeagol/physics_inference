"""
Table Detection Module using Segmentation Model

This module provides functionality to detect and segment table objects in images
and video frames using a trained segmentation model.
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from .inference_runner import InferenceRunner


class TableDetector(InferenceRunner):
    """
    Table detection class using segmentation model for precise table boundary detection.
    
    This class uses a trained segmentation model to detect table objects and extract
    their precise boundaries for video processing applications.
    """
    
    def __init__(self, 
                 model_path: str = "snkr_segm-egvem-3-roboflow-weights.pt",
                 confidence: float = 0.5,
                 **kwargs: Any) -> None:
        """
        Initialize the table detector.
        
        Args:
            model_path: Path to the segmentation model weights
            confidence: Minimum confidence threshold for detections
            **kwargs: Additional parameters
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_path = model_path
        self.confidence = confidence
        self.model_name = os.path.basename(model_path)
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded table segmentation model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading segmentation model: {str(e)}")
    
    def predict(self, 
                image: Union[str, np.ndarray], 
                **kwargs: Any) -> Dict[str, Any]:
        """
        Run table detection on a single image.
        
        Args:
            image: Either a file path to an image or a numpy array
            **kwargs: Additional parameters (confidence, etc.)
                    
        Returns:
            Dictionary containing detection results and segmentation masks
        """
        conf = kwargs.get('confidence', self.confidence)
        
        try:
            # Run segmentation inference
            results = self.model(image, conf=conf, verbose=False)
            return self._convert_to_standard_format(results[0])
            
        except Exception as e:
            raise RuntimeError(f"Error during table detection: {str(e)}")
    
    def detect_table_mask(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract table segmentation mask from image.
        
        Args:
            image: Input image (file path or numpy array)
            
        Returns:
            Binary mask where table pixels are 255, background is 0
        """
        # Load image if it's a file path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Get predictions
        predictions = self.predict(img)
        
        # Find table predictions
        table_masks = []
        for pred in predictions.get('predictions', []):
            if 'table' in pred.get('class', '').lower():
                if 'mask' in pred:
                    table_masks.append(pred['mask'])
        
        if not table_masks:
            return None
        
        # Create an empty mask with the same dimensions as the image
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # Combine all table masks, ensuring they match the image dimensions
        for mask in table_masks:
            # Resize mask if needed to match the image dimensions
            if mask.shape[:2] != (img.shape[0], img.shape[1]):
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Ensure mask is binary (0 or 255)
            binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            
            # Combine with the existing mask
            # Create a new mask from the bitwise_or operation to avoid type issues
            result_mask = cv2.bitwise_or(combined_mask, binary_mask)
            combined_mask = result_mask.astype(np.uint8)
        
        return combined_mask
    
    def get_table_bounding_box(self, image: Union[str, np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the bounding box of the detected table.
        
        Args:
            image: Input image (file path or numpy array)
            
        Returns:
            Tuple of (x, y, width, height) or None if no table detected
        """
        mask = self.detect_table_mask(image)
        if mask is None:
            return None
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour (assumed to be the table)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def crop_to_table(self, image: Union[str, np.ndarray], padding: int = 10) -> Optional[np.ndarray]:
        """
        Crop image to table region with optional padding.
        
        Args:
            image: Input image (file path or numpy array)
            padding: Padding around the table in pixels
            
        Returns:
            Cropped image or None if no table detected
        """
        # Load image if it's a file path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        bbox = self.get_table_bounding_box(img)
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        
        # Add padding and ensure we don't go out of bounds
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Crop the image
        cropped = img[y_start:y_end, x_start:x_end]
        
        return cropped
    
    def visualize_predictions(self,
                             image: Union[str, np.ndarray],
                             predictions: Dict,
                             output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize the table detection results on the input image.
        
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
        
        # Create overlay for masks
        overlay = img.copy()
        
        # Draw each prediction
        for pred in predictions.get('predictions', []):
            # Draw segmentation mask if available
            if 'mask' in pred:
                mask = pred['mask']
                
                # Resize mask if needed to match the overlay dimensions
                if mask.shape[:2] != overlay.shape[:2]:
                    mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                color = (0, 255, 0)  # Green for table
                overlay[mask > 0] = color
            
            # Draw bounding box
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            w = int(pred['width'])
            h = int(pred['height'])
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label and confidence
            label = f"{pred['class']} {pred['confidence']:.2f}"
            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Blend the overlay with the original image
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Save the visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, result)
            
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the table detection model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'name': self.model_name,
            'type': 'segmentation',
            'framework': 'ultralytics',
            'path': self.model_path,
            'confidence': self.confidence,
            'task': 'table_detection'
        }
        
    def predict_batch(self, 
                     images: List[Union[str, np.ndarray]],
                     **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Run table detection on a batch of images.
        
        Args:
            images: List of file paths to images or numpy arrays
            **kwargs: Additional parameters (confidence, etc.)
                    
        Returns:
            List of dictionaries containing detection results and segmentation masks
        """
        results = []
        conf = kwargs.get('confidence', self.confidence)
        
        try:
            # Process each image in the batch
            for image in images:
                # Run prediction on single image
                result = self.predict(image, confidence=conf, **kwargs)
                results.append(result)
                
            return results
            
        except Exception as e:
            raise RuntimeError(f"Error during batch table detection: {str(e)}")
    
    def _convert_to_standard_format(self, result: Any) -> Dict[str, Any]:
        """
        Convert Ultralytics segmentation result to standardized format.
        
        Args:
            result: Ultralytics segmentation result
            
        Returns:
            Dictionary with standardized prediction format including masks
        """
        predictions = []
        
        # Check if we have segmentation results
        if hasattr(result, 'masks') and result.masks is not None:
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(boxes)):
                # Get box coordinates (xywh format)
                x, y, w, h = boxes.xywh[i].tolist()
                
                # Get class and confidence
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id]
                conf = float(boxes.conf[i].item())
                
                # Get segmentation mask
                mask = masks.data[i].cpu().numpy().astype(np.uint8) * 255
                
                # Create prediction entry
                pred = {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': conf,
                    'class': cls_name,
                    'class_id': cls_id,
                    'mask': mask
                }
                predictions.append(pred)
        else:
            # Fallback to bounding box only
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xywh format)
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