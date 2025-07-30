"""
Table Detection Module for Cue Sports Analysis

This module implements traditional computer vision techniques to detect
the playing surface (typically green baize) in cue sports images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union


def detect_green_table(
    image: np.ndarray,
    hsv_lower: Tuple[int, int, int] = (35, 40, 40),
    hsv_upper: Tuple[int, int, int] = (85, 255, 255),
    min_area: float = 0.1,
    max_aspect_ratio: float = 2.5,
) -> Optional[np.ndarray]:
    """
    Detect the green table in an image using HSV color segmentation.

    Args:
        image: Input BGR image
        hsv_lower: Lower bound for HSV color range (H, S, V)
        hsv_upper: Upper bound for HSV color range (H, S, V)
        min_area: Minimum area as a fraction of the image area (0-1)
        max_aspect_ratio: Maximum aspect ratio for the table contour

    Returns:
        Contour points of the detected table (4 points) or None if no table is detected
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for green color range
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for finer details
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Add a border to help with edge detection
    border_size = 5
    mask = cv2.copyMakeBorder(mask, border_size, border_size, border_size, border_size, 
                             cv2.BORDER_CONSTANT, value=0)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return None
    if len(contours) == 0:
        return None
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    best_contour = None
    
    for contour in contours:
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < min_area * image.shape[0] * image.shape[1]:
            continue
            
        # Calculate the convex hull of the contour
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Skip if the hull area is too small relative to the contour area
        if hull_area < min_area * image.shape[0] * image.shape[1] or area / hull_area < 0.7:
            continue
            
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(hull, True)  # Slightly larger epsilon for smoother shape
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # We're looking for a 4-sided polygon
        if len(approx) == 4:
            # Calculate the aspect ratio of the bounding rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            # Ensure width is always the larger dimension
            if width < height:
                width, height = height, width
                
            aspect_ratio = width / (height + 1e-6)
            
            # Only consider contours with reasonable aspect ratio
            if aspect_ratio <= max_aspect_ratio:
                best_contour = approx.reshape(4, 2)
    
    # If we have a best contour, return it
    if best_contour is not None:
        return order_points(best_contour)
    
    # If no good contour found, try to find the best fitting rectangle from the largest contour
    if len(contours) > 0 and len(contours[0]) >= 4:
        largest_contour = contours[0]  # Already sorted by area
        try:
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)  # Convert to integer coordinates
            return order_points(box.reshape(4, 2))
        except Exception as e:
            # Fall back to convex hull approximation
            hull = cv2.convexHull(largest_contour)
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) >= 4:
                # Take the largest 4-point approximation
                approx = cv2.approxPolyDP(hull, 0.1 * epsilon, True)
                if len(approx) >= 4:
                    return order_points(approx.reshape(-1, 2)[:4])
    
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order starting from top-left.
    
    Args:
        pts: Array of 4 points (4, 2)
        
    Returns:
        Ordered points in clockwise order: top-left, top-right, bottom-right, bottom-left
    """
    # Initialize the ordered coordinates
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def perspective_transform(
    image: np.ndarray,
    pts: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> np.ndarray:
    """
    Apply perspective transform to get a top-down view of the table.
    
    Args:
        image: Input image
        pts: Array of 4 points defining the table corners
        width: Width of the output image (defaults to max width of the table)
        height: Height of the output image (defaults to max height of the table)
        
    Returns:
        Warped image with top-down view of the table
    """
    # Order the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # If width and height are not specified, use the maximum width and height
    # of the table
    if width is None or height is None:
        # Compute the width of the new image
        widthA = np.linalg.norm(br - bl)  # Bottom edge
        widthB = np.linalg.norm(tr - tl)  # Top edge
        max_width = max(int(widthA), int(widthB))
        
        # Compute the height of the new image
        heightA = np.linalg.norm(tr - br)  # Right edge
        heightB = np.linalg.norm(tl - bl)  # Left edge
        max_height = max(int(heightA), int(heightB))
        
        # Ensure the aspect ratio is reasonable for a pool table (typically 2:1)
        aspect_ratio = max_width / max_height
        if aspect_ratio > 2.5:  # Too wide
            max_width = int(max_height * 2.0)  # Cap at 2:1
        elif aspect_ratio < 1.5:  # Too tall
            max_height = int(max_width / 2.0)  # Cap at 2:1
        
        width = max_width if width is None else width
        height = max_height if height is None else height
    
    # Define the destination points for the perspective transform
    # Ensure the points are in the correct order (top-left, top-right, bottom-right, bottom-left)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped
