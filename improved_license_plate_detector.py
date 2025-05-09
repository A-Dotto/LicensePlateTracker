"""
Improved License Plate Detector using Contour-based approach
This module implements license plate detection using OpenCV's contour detection
and filtering techniques, inspired by 'detect plate' approach from Kaggle.

This approach has several advantages:
1. Works well with CPU-only environments (no GPU required)
2. Fast detection speed for real-time applications
3. Good accuracy for clearly visible license plates
"""

import cv2
import numpy as np
import imutils
import logging
import time
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedLicensePlateDetector:
    """
    License plate detector using contour-based approaches
    Inspired by: https://www.kaggle.com/code/namthanh189/detect-plate
    """
    
    def __init__(self):
        """Initialize the license plate detector with default parameters"""
        # Parameters for filtering
        self.min_contour_area = 500
        self.max_contour_area = 50000
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 7.0
        self.min_plate_width = 60
        self.min_plate_height = 20
        
        # Parameters for Canny edge detection
        self.canny_low = 30
        self.canny_high = 200
        
        # Parameters for bilateral filter
        self.bilateral_d = 11
        self.bilateral_sigma_color = 17
        self.bilateral_sigma_space = 17
        
        # Last detection results
        self.last_plate_box = None
        self.last_plate_img = None
        self.processing_time = 0
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image to enhance license plate detection
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Tuple of (gray image, edge image)
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Apply bilateral filter for noise reduction while preserving edges
        bfilter = cv2.bilateralFilter(
            gray, 
            self.bilateral_d, 
            self.bilateral_sigma_color, 
            self.bilateral_sigma_space
        )
        
        # Apply Canny edge detection
        edged = cv2.Canny(bfilter, self.canny_low, self.canny_high)
        
        return gray, edged
    
    def find_plate_candidates(self, edged: np.ndarray) -> List[np.ndarray]:
        """
        Find potential license plate contours from edges
        
        Args:
            edged: Edge image from Canny
            
        Returns:
            List of contours (numpy arrays) that might be license plates
        """
        # Find contours from edge image
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        
        # Sort contours by area (largest first) and take top 10
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Filter contours by shape, area, and aspect ratio
        plate_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = w / float(h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
                
            # Filter by minimum width and height
            if w < self.min_plate_width or h < self.min_plate_height:
                continue
                
            # Approximate contour shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            
            # License plates typically have 4 corners (rectangle)
            # Sometimes the approximation gives 4-8 points
            if 4 <= len(approx) <= 8:
                plate_candidates.append(approx)
        
        return plate_candidates
    
    def extract_plate_region(self, img: np.ndarray, gray: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract the plate region from the image based on contour
        
        Args:
            img: Original image
            gray: Grayscale image
            contour: License plate contour
            
        Returns:
            Tuple of (plate image, plate coordinates (x, y, w, h))
        """
        # Create mask for the contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)  # Use proper color format (BGR)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the plate region
        plate_img = gray[y:y+h, x:x+w]
        
        return plate_img, (x, y, w, h)
    
    def detect_license_plate(self, image_data: bytes) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect license plate in an image
        
        Args:
            image_data: Binary image data
            
        Returns:
            Tuple of (license plate image, license plate coordinates (x, y, w, h))
        """
        start_time = time.time()
        
        try:
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image data")
                
            # Preprocess image
            gray, edged = self.preprocess_image(img)
            
            # Find license plate candidates
            plate_candidates = self.find_plate_candidates(edged)
            
            if not plate_candidates:
                logger.info("No license plate candidates found")
                self.processing_time = time.time() - start_time
                return None, None
                
            # Take the first candidate (already sorted by size)
            plate_contour = plate_candidates[0]
            
            # Extract the plate region
            plate_img, plate_box = self.extract_plate_region(img, gray, plate_contour)
            
            # Store results
            self.last_plate_box = plate_box
            self.last_plate_img = plate_img
            
            # Calculate processing time
            self.processing_time = time.time() - start_time
            
            return plate_img, plate_box
            
        except Exception as e:
            logger.error(f"Error in license plate detection: {str(e)}")
            self.processing_time = time.time() - start_time
            return None, None
    
    def get_annotated_image(self, image_data: bytes, license_plate_text: str, confidence: float) -> np.ndarray:
        """
        Create an annotated image with license plate highlighted and text displayed
        
        Args:
            image_data: Binary image data
            license_plate_text: Detected license plate text
            confidence: Confidence score (0-1)
            
        Returns:
            Annotated image
        """
        try:
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image data")
                
            # If no plate was detected, just return the original image with message
            if self.last_plate_box is None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "No license plate detected", (30, 30),
                           font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return img
                
            # Get plate box
            x, y, w, h = self.last_plate_box
            
            # Draw the rectangle around the plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Check if license_plate_text contains "No license plate detected"
            if license_plate_text == "No license plate detected" or license_plate_text == "No plate":
                text = "No license plate detected"
            else:
                text = f"{license_plate_text} ({confidence:.2f})"
            
            # Position the text below the plate
            text_x = x
            text_y = y + h + 30
            
            # Draw text with background
            cv2.putText(img, text, (text_x, text_y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            return img
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {str(e)}")
            # Return the original image on error with an error message
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "Error processing image", (30, 30),
                           font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return img
            return np.zeros((300, 400, 3), dtype=np.uint8)

# Create a singleton instance for global use
detector = ImprovedLicensePlateDetector()

def detect_plate(image_data: bytes) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Global function to detect license plate
    
    Args:
        image_data: Binary image data
        
    Returns:
        Tuple of (license plate image, license plate coordinates)
    """
    return detector.detect_license_plate(image_data)

def get_annotated_image(image_data: bytes, license_plate_text: str, confidence: float) -> np.ndarray:
    """
    Global function to get annotated image
    
    Args:
        image_data: Binary image data
        license_plate_text: Detected license plate text
        confidence: Confidence score (0-1)
        
    Returns:
        Annotated image
    """
    return detector.get_annotated_image(image_data, license_plate_text, confidence)