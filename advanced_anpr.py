"""
Advanced Automatic Number Plate Recognition (ANPR) system
Using a combination of computer vision techniques and specialized OCR
to achieve high accuracy number plate recognition
"""

import os
import cv2
import numpy as np
import logging
import urllib.request
import json
from pathlib import Path
import base64
import time
import pytesseract
from PIL import Image
from skimage import transform, filters, morphology, measure, segmentation

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants for the ANPR model
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# Country-specific license plate formats
PLATE_FORMATS = {
    "US": r"[A-Z0-9]{5,8}",  # US general format
    "EU": r"[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}",  # European format
    "UK": r"[A-Z]{2}[0-9]{2}[A-Z]{3}",  # UK format
    "AU": r"[A-Z0-9]{6,7}",  # Australian format
}

# Visual processing modes
VISUAL_MODES = {
    "threshold": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    "adaptive": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    "canny": None,  # Special case for Canny edge detection
}

# Character confidence scores
CHAR_CONFIDENCE = {
    "0": 0.95, "1": 0.95, "2": 0.93, "3": 0.91, "4": 0.92,
    "5": 0.90, "6": 0.91, "7": 0.94, "8": 0.89, "9": 0.90,
    "A": 0.93, "B": 0.90, "C": 0.92, "D": 0.91, "E": 0.93,
    "F": 0.92, "G": 0.88, "H": 0.91, "I": 0.95, "J": 0.89,
    "K": 0.89, "L": 0.93, "M": 0.91, "N": 0.92, "O": 0.93,
    "P": 0.91, "Q": 0.87, "R": 0.90, "S": 0.91, "T": 0.92,
    "U": 0.90, "V": 0.90, "W": 0.89, "X": 0.90, "Y": 0.90,
    "Z": 0.90
}

# Tesseract OCR configurations for different scenarios
TESSERACT_CONFIGS = [
    "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",  # License plate mode
    "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",  # Single line mode
    "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",  # Single char mode
    "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",  # Block of text mode
]


class AdvancedPlateDetector:
    """
    Advanced license plate detector using modern computer vision techniques
    """
    def __init__(self):
        # Attempt to load cascades if available
        try:
            # Access cascade files in a compatible way
            cascade_path = os.path.join(cv2.data.haarcascades 
                                      if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades') 
                                      else '', 'haarcascade_russian_plate_number.xml')
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                logger.warning("Haar cascade for license plate detection not found or failed to load")
                self.cascade = None
            else:
                logger.info("Successfully loaded license plate Haar cascade")
        except Exception as e:
            logger.error(f"Error loading Haar cascade: {e}")
            self.cascade = None
            
        # Use additional detection methods
        self.use_contour_method = True
        self.use_morphology_method = True
        self.enable_visual_indicators = True
        
        # Create the MSER detector for text region detection
        try:
            # Use MSER if available in this OpenCV version
            if hasattr(cv2, 'MSER_create'):
                self.mser = cv2.MSER_create()
                self.mser.setMinArea(100)
                self.mser.setMaxArea(5000)
                logger.info("MSER detector initialized for text region detection")
            else:
                logger.warning("MSER_create not available in this OpenCV version")
                self.mser = None
        except Exception as e:
            logger.warning(f"Could not create MSER detector: {e}")
            self.mser = None
            
        # Processing parameters
        self.min_plate_width = 60
        self.min_plate_height = 20
        self.min_plate_ratio = 1.5
        self.max_plate_ratio = 7.0

    def preprocess_image(self, image):
        """Apply preprocessing to enhance the image for plate detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Create multiple versions with different processing
        processed_images = []
        
        # 1. Basic grayscale
        processed_images.append(gray)
        
        # 2. Bilateral filter (noise reduction while preserving edges)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        processed_images.append(bilateral)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        processed_images.append(clahe_img)
        
        # 4. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph_img)
        
        return processed_images

    def detect_plates(self, image):
        """
        Detect license plates in an image using multiple methods
        Returns list of (x, y, w, h, confidence) tuples
        """
        height, width = image.shape[:2]
        all_plates = []
        
        # Use multiple preprocessing approaches for better detection
        processed_images = self.preprocess_image(image)
        
        # 1. Haar Cascade method if available
        if self.cascade is not None:
            for processed in processed_images:
                try:
                    plates = self.cascade.detectMultiScale(
                        processed, 
                        scaleFactor=1.1, 
                        minNeighbors=5,
                        minSize=(self.min_plate_width, self.min_plate_height)
                    )
                    
                    for (x, y, w, h) in plates:
                        # Calculate confidence based on detection strength
                        # (simplified for this implementation)
                        confidence = 0.75
                        all_plates.append((x, y, w, h, confidence, "cascade"))
                except Exception as e:
                    logger.debug(f"Cascade detection error: {e}")
        
        # 2. Contour-based method
        if self.use_contour_method:
            for processed in processed_images:
                try:
                    # Find edges
                    edges = cv2.Canny(processed, 50, 150)
                    
                    # Find contours
                    contours, _ = cv2.findContours(
                        edges.copy(), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Filter by aspect ratio and minimum size
                        if w > self.min_plate_width and h > self.min_plate_height:
                            aspect_ratio = w / float(h)
                            if self.min_plate_ratio <= aspect_ratio <= self.max_plate_ratio:
                                # Calculate area and extent (ratio of contour area to bounding rectangle area)
                                rect_area = w * h
                                contour_area = cv2.contourArea(contour)
                                extent = float(contour_area) / rect_area if rect_area > 0 else 0
                                
                                # Calculate approximate perimeter length
                                peri = cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                                
                                # More confident if the contour has 4 corners (like a license plate)
                                corner_bonus = 0.15 if len(approx) == 4 else 0
                                
                                # Confidence based on several factors
                                confidence = 0.5 + (extent * 0.3) + corner_bonus
                                
                                all_plates.append((x, y, w, h, confidence, "contour"))
                except Exception as e:
                    logger.debug(f"Contour detection error: {e}")
        
        # 3. Morphology-based method (good for plates with consistent text height)
        if self.use_morphology_method:
            for processed in processed_images:
                try:
                    # Apply threshold
                    _, threshold = cv2.threshold(
                        processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    
                    # Create a rectangular kernel for license plate structure
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
                    
                    # Apply morphological operations
                    morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
                    
                    # Find contours in the morphed image
                    contours, _ = cv2.findContours(
                        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Apply the same filtering as before
                        if w > self.min_plate_width and h > self.min_plate_height:
                            aspect_ratio = w / float(h)
                            if self.min_plate_ratio <= aspect_ratio <= self.max_plate_ratio:
                                # Slightly higher base confidence for morphology method
                                # since it often works well for license plates
                                confidence = 0.65
                                all_plates.append((x, y, w, h, confidence, "morphology"))
                except Exception as e:
                    logger.debug(f"Morphology detection error: {e}")
        
        # 4. MSER text regions (good for detecting text regions that could be license plates)
        if self.mser is not None:
            for processed in processed_images[:1]:  # Only use grayscale for MSER
                try:
                    # Detect regions
                    regions, _ = self.mser.detectRegions(processed)
                    
                    if regions:
                        # Convert regions to boxes
                        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
                        
                        # Filter and merge text regions that could be license plates
                        mask = np.zeros((height, width), dtype=np.uint8)
                        for hull in hulls:
                            # Use proper color format for drawContours (BGR tuple)
                            cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                        
                        # Dilate the mask to connect nearby text regions
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                        mask = cv2.dilate(mask, kernel, iterations=3)
                        
                        # Find contours in the mask
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Apply similar filtering
                            if w > self.min_plate_width and h > self.min_plate_height:
                                aspect_ratio = w / float(h)
                                if self.min_plate_ratio <= aspect_ratio <= self.max_plate_ratio:
                                    # MSER is good for text detection, so give it reasonable confidence
                                    confidence = 0.6
                                    all_plates.append((x, y, w, h, confidence, "mser"))
                except Exception as e:
                    logger.debug(f"MSER detection error: {e}")
        
        # If no plates detected, try a more aggressive approach
        if not all_plates:
            try:
                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                # Apply adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Find contours in the thresholded image
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Less strict filtering for fallback approach
                    if w > 40 and h > 15:
                        aspect_ratio = w / float(h)
                        if 1.0 <= aspect_ratio <= 8.0:
                            confidence = 0.4  # Lower confidence for fallback method
                            all_plates.append((x, y, w, h, confidence, "fallback"))
            except Exception as e:
                logger.debug(f"Fallback detection error: {e}")
        
        # Non-maximum suppression to eliminate overlapping boxes
        return self._non_max_suppression(all_plates, overlap_thresh=0.4)

    def _non_max_suppression(self, boxes, overlap_thresh=0.4):
        """
        Apply non-maximum suppression to eliminate redundant
        overlapping bounding boxes.
        """
        if len(boxes) == 0:
            return []
        
        # Initialize the list of picked indexes
        pick = []
        
        # Extract coordinates and confidence
        x1 = np.array([box[0] for box in boxes])
        y1 = np.array([box[1] for box in boxes])
        x2 = np.array([box[0] + box[2] for box in boxes])
        y2 = np.array([box[1] + box[3] for box in boxes])
        confidence = np.array([box[4] for box in boxes])
        method = [box[5] for box in boxes]
        
        # Compute area of each box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by confidence
        idxs = np.argsort(confidence)[::-1]
        
        # Keep looping while indexes remain in the list
        while len(idxs) > 0:
            # Grab the last index and add to picked list
            last = len(idxs) - 1
            i = idxs[0]
            pick.append(i)
            
            # Find the coordinates of the intersection
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            # Compute width and height of the intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[1:]]
            
            # Delete indexes with overlap greater than threshold
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
        
        # Return the boxes that were picked
        result = []
        for i in pick:
            # Format: (x, y, w, h, confidence, method)
            result.append((
                int(x1[i]), int(y1[i]), 
                int(x2[i] - x1[i] + 1), int(y2[i] - y1[i] + 1),
                float(confidence[i]), method[i]
            ))
        
        return result

    def extract_plate_region(self, image, box):
        """Extract the license plate region from the image"""
        x, y, w, h = box[:4]
        
        # Get the license plate region
        plate_img = image[y:y+h, x:x+w]
        
        return plate_img
    
    def visually_annotate_image(self, image, boxes, detected_text=None):
        """Draw bounding boxes and text on the image for visualization"""
        if not self.enable_visual_indicators:
            return image
            
        result = image.copy()
        
        for i, box in enumerate(boxes):
            x, y, w, h, confidence, method = box
            
            # Color based on method
            color_map = {
                "cascade": (0, 255, 0),    # Green for cascade
                "contour": (0, 0, 255),    # Red for contour
                "morphology": (255, 0, 0), # Blue for morphology
                "mser": (255, 255, 0),     # Cyan for MSER
                "fallback": (128, 0, 128)  # Purple for fallback
            }
            color = color_map.get(method, (255, 255, 255))
            
            # Draw the bounding box
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Add the confidence score and method
            label = f"{method}: {confidence:.2f}"
            cv2.putText(
                result, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Add detected text if available
            if detected_text and i < len(detected_text):
                text = detected_text[i][0]
                cv2.putText(
                    result, text, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
        
        return result


class AdvancedPlateRecognizer:
    """
    Advanced OCR for license plate recognition
    """
    def __init__(self):
        # OCR processing parameters
        self.char_height_min = 10  # Minimum character height in pixels
        self.char_width_min = 5    # Minimum character width in pixels
        self.char_aspect_ratio_min = 0.2  # Minimum character aspect ratio (width/height)
        self.char_aspect_ratio_max = 0.9  # Maximum character aspect ratio
        self.min_confidence = 40  # Minimum confidence for character recognition
        
        # Common plate patterns by country/region
        self.plate_patterns = PLATE_FORMATS
        
        # Visual processing modes
        self.visual_modes = VISUAL_MODES
        
        # Tesseract configurations to try
        self.tesseract_configs = TESSERACT_CONFIGS
        
        # Character confidence scores
        self.char_confidence = CHAR_CONFIDENCE
        
        # Processing flags
        self.apply_perspective_correction = True
        self.apply_character_segmentation = True
        self.enable_multi_pass = True

    def _enhance_plate_image(self, img):
        """Apply various image enhancement techniques for OCR"""
        # Make a copy to avoid modifying the original
        enhanced = img.copy()
        
        # Resize if the image is too small
        if enhanced.shape[0] < 40 or enhanced.shape[1] < 100:
            # Calculate scaling factor to make height at least 40px
            scale = max(40 / enhanced.shape[0], 100 / enhanced.shape[1])
            enhanced = cv2.resize(
                enhanced, 
                None, 
                fx=scale, 
                fy=scale, 
                interpolation=cv2.INTER_CUBIC
            )
        
        # Convert to grayscale if needed
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced.copy()
        
        # Apply bilateral filter to remove noise but preserve edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        return enhanced

    def _create_processing_variations(self, img):
        """Create multiple processing variations to improve OCR success rate"""
        variations = []
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 1. Original grayscale
        variations.append(("original", gray))
        
        # 2. Bilateral filter
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        variations.append(("bilateral", bilateral))
        
        # 3. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variations.append(("clahe", clahe_img))
        
        # 4. Otsu's thresholding
        _, otsu = cv2.threshold(
            bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variations.append(("otsu", otsu))
        
        # 5. Inverted Otsu's
        _, inv_otsu = cv2.threshold(
            bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        variations.append(("inv_otsu", inv_otsu))
        
        # 6. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        variations.append(("adaptive", adaptive))
        
        # 7. Inverted adaptive
        inv_adaptive = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        variations.append(("inv_adaptive", inv_adaptive))
        
        # 8. Morphological operations (closing) on adaptive threshold
        # to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        variations.append(("morph_adaptive", morph_adaptive))
        
        # 9. Canny edges
        edges = cv2.Canny(bilateral, 30, 200)
        variations.append(("canny", edges))
        
        # 10. Dilated edges to connect broken characters
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        variations.append(("dilated_edges", dilated_edges))
        
        return variations

    def _correct_perspective(self, img):
        """Correct perspective distortion in license plate image"""
        if not self.apply_perspective_correction:
            return img
            
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return img
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the rotated rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            
            # Get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Ensure the dimensions are valid
            if width <= 0 or height <= 0:
                return img
            
            # If the rectangle is too rotated, swap width and height
            angle = rect[2]
            if angle < -45:
                width, height = height, width
            
            # Create the perspective transform matrix
            src_pts = box.astype("float32")
            
            # Arrange the points in top-left, top-right, bottom-right, bottom-left order
            s = src_pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            # Top-left point has the smallest sum
            rect[0] = src_pts[np.argmin(s)]
            # Bottom-right point has the largest sum
            rect[2] = src_pts[np.argmax(s)]
            
            # Calculate the difference and compute the other two points
            diff = np.diff(src_pts, axis=1)
            # Top-right point has the smallest difference
            rect[1] = src_pts[np.argmin(diff)]
            # Bottom-left point has the largest difference
            rect[3] = src_pts[np.argmax(diff)]
            
            # Define the destination points
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # Calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst_pts)
            
            # Apply the perspective transformation
            warped = cv2.warpPerspective(img, M, (width, height))
            
            return warped
        except Exception as e:
            logger.debug(f"Perspective correction error: {e}")
            return img

    def _segment_characters(self, img):
        """Segment individual characters in the license plate"""
        if not self.apply_character_segmentation:
            return img, None
            
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply connected component analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                thresh, connectivity=8
            )
            
            # Filter components based on size and aspect ratio to remove noise
            # Stats format: [x, y, width, height, area]
            char_contours = []
            for i in range(1, num_labels):  # Skip the background (label 0)
                x, y, w, h, area = stats[i]
                
                # Filter based on size and aspect ratio
                if h > self.char_height_min and w > self.char_width_min:
                    aspect_ratio = w / float(h)
                    if self.char_aspect_ratio_min <= aspect_ratio <= self.char_aspect_ratio_max:
                        # Store as (x, y, w, h, centroid_x)
                        centroid_x = centroids[i][0]
                        char_contours.append((x, y, w, h, centroid_x))
            
            # Sort characters from left to right based on x-coordinate
            char_contours.sort(key=lambda x: x[0])
            
            # Extract individual characters
            char_images = []
            for x, y, w, h, _ in char_contours:
                char_img = gray[y:y+h, x:x+w]
                # Resize to a standard size for better OCR
                char_img = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_CUBIC)
                char_images.append(char_img)
            
            return img, char_images
        except Exception as e:
            logger.debug(f"Character segmentation error: {e}")
            return img, None

    def _recognize_characters(self, char_images):
        """Recognize individual segmented characters"""
        if not char_images:
            return None
            
        recognized_chars = []
        confidences = []
        
        for char_img in char_images:
            best_char = None
            best_conf = 0
            
            # Try different preprocessing variations
            variations = self._create_processing_variations(char_img)
            
            for name, processed in variations:
                # Try different Tesseract configurations
                for config in self.tesseract_configs:
                    try:
                        # Use Tesseract in single character mode
                        result = pytesseract.image_to_data(
                            processed, 
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Check if any text was recognized
                        if result["text"] and any(result["text"]):
                            # Get the first recognized character and its confidence
                            for i, text in enumerate(result["text"]):
                                if text.strip():
                                    conf = int(result["conf"][i])
                                    if conf > best_conf:
                                        best_char = text[0]  # Just the first character
                                        best_conf = conf
                                        if best_conf > 90:  # High confidence, no need to try more
                                            break
                    except Exception as e:
                        logger.debug(f"Character recognition error: {e}")
            
            # If no character was recognized with confidence
            if not best_char or best_conf < self.min_confidence:
                continue
                
            # Add the character to the result
            recognized_chars.append(best_char)
            confidences.append(best_conf / 100.0)  # Normalize to [0, 1]
        
        if not recognized_chars:
            return None
            
        plate_text = ''.join(recognized_chars)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return (plate_text, avg_confidence)

    def _ocr_on_variations(self, variations):
        """Apply OCR on different image variations and select the best result"""
        results = []
        
        # Limit the number of variations to process (to avoid timeouts)
        variations_to_process = variations[:2] if len(variations) > 2 else variations
        
        # Process each variation with limited Tesseract configurations
        for name, img in variations_to_process:
            # Only use the first two configs for better performance and to avoid timeouts
            limited_configs = self.tesseract_configs[:2] if len(self.tesseract_configs) > 2 else self.tesseract_configs
            
            for config in limited_configs:
                try:
                    # Apply OCR with timeout
                    import signal
                    import time
                    
                    # Set a short timeout to avoid hanging (2 seconds)
                    # This is a simplified solution - the proper timeout is handled by try/except
                    start_time = time.time()
                    max_time = 2  # 2 seconds max per OCR operation
                    
                    # Apply OCR
                    try:
                        data = pytesseract.image_to_string(
                            img,
                            config=config,
                            timeout=1  # 1 second timeout for tesseract
                        )
                        
                        # If we got a result, create a simple dict to mimic the image_to_data structure
                        if data:
                            # Create a simplified result structure
                            text = data.strip()
                            if text:
                                # Store the result (text, confidence, preprocessing, config)
                                # Use a default confidence of 0.5 since we don't have detailed confidence data
                                results.append((text, 0.5, name, config))
                                
                        # Check if we're taking too long
                        if time.time() - start_time > max_time:
                            logger.warning(f"OCR operation taking too long, skipping remaining operations")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Tesseract timeout or error: {str(e)}")
                        continue
                    
                except Exception as e:
                    logger.debug(f"OCR error on {name} with {config}: {e}")
        
        # Add a default result if no results were obtained
        if not results:
            results.append(("UNKNOWN", 0.1, "default", ""))
            
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def score_plate_text(self, text):
        """Score the recognized text based on plate patterns and character confidence"""
        # Calculate base score from character confidence
        char_score = 0
        char_count = 0
        
        for char in text:
            if char in self.char_confidence:
                char_score += self.char_confidence[char]
                char_count += 1
        
        # Average character confidence
        if char_count > 0:
            char_score = char_score / char_count
        else:
            char_score = 0
        
        # Basic length score (plates usually have 5-8 characters)
        length_score = 0
        if 5 <= len(text) <= 8:
            length_score = 0.9
        elif 3 <= len(text) <= 10:
            length_score = 0.7
        else:
            length_score = 0.3
        
        # Pattern match score
        import re
        pattern_score = 0
        for pattern in self.plate_patterns.values():
            if re.match(pattern, text):
                pattern_score = 0.9
                break
        
        # Combine scores with weights
        final_score = (char_score * 0.4) + (length_score * 0.3) + (pattern_score * 0.3)
        
        return final_score

    def filter_and_select_best_plate(self, results):
        """Filter OCR results and select the best license plate text"""
        # If no results, return None
        if not results:
            return None, 0
            
        # Filter out non-alphanumeric or too short results
        filtered = []
        for text, conf, preproc, config in results:
            # Keep only alphanumeric
            cleaned = ''.join(c for c in text if c.isalnum())
            
            # Skip if too short
            if len(cleaned) < 3:
                continue
                
            # Score the plate text
            score = self.score_plate_text(cleaned) * conf
            
            # Add to filtered results
            filtered.append((cleaned, score, preproc, config))
        
        # Sort by score
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best result, or None if no valid results
        if filtered:
            return filtered[0][0], filtered[0][1]
        else:
            return None, 0

    def recognize_plate(self, img):
        """
        Recognize text in a license plate image using multiple techniques
        Returns: (text, confidence)
        """
        # If image is empty or None
        if img is None or img.size == 0:
            logger.warning("Empty image received for plate recognition")
            return "UNKNOWN", 0.1
        
        try:
            # Enhance the plate image
            enhanced = self._enhance_plate_image(img)
            
            # Apply perspective correction
            if self.apply_perspective_correction:
                corrected = self._correct_perspective(enhanced)
            else:
                corrected = enhanced
            
            # Create multiple processing variations
            variations = self._create_processing_variations(corrected)
            
            try:
                # Apply OCR on all variations with timeout handling
                ocr_results = self._ocr_on_variations(variations)
            except Exception as e:
                logger.error(f"OCR error in recognize_plate: {str(e)}")
                # Return a default value to prevent processing failures
                return "UNKNOWN", 0.1
                
            # Apply character segmentation and recognition if enabled
            if self.apply_character_segmentation:
                try:
                    _, char_images = self._segment_characters(corrected)
                    if char_images:
                        segment_result = self._recognize_characters(char_images)
                        if segment_result:
                            # Add the segmentation result to the OCR results
                            text, conf = segment_result
                            ocr_results.append((text, conf, "segmentation", "char_by_char"))
                except Exception as seg_error:
                    logger.error(f"Character segmentation error: {str(seg_error)}")
            
            # Filter and select the best plate
            best_text, confidence = self.filter_and_select_best_plate(ocr_results)
            
            # If confidence is too low or text is None, return a default value
            if confidence < 0.1 or best_text is None:
                return "UNKNOWN", 0.1
                
            return best_text, confidence
            
        except Exception as e:
            logger.error(f"Error in recognize_plate: {str(e)}")
            # Return a default value to prevent the application from crashing
            return "UNKNOWN", 0.1


class AdvancedANPR:
    """
    Advanced ANPR system that combines detection and recognition
    """
    def __init__(self):
        self.detector = AdvancedPlateDetector()
        self.recognizer = AdvancedPlateRecognizer()
        
        # Processing flags
        self.enable_visual_indicators = True
        self.enable_detail_logging = True
        
        # Cache for results
        self.last_processed_image = None
        self.last_detection_boxes = None
        self.last_recognition_results = None
        
        # Timing information
        self.last_processing_time = 0

    def process_image(self, image_data):
        """
        Process an image to detect and recognize license plates
        Returns: (license_plate_text, confidence)
        """
        start_time = time.time()
        
        try:
            # Convert image data to numpy array if needed
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image_data
                
            if img is None:
                logger.error("Failed to convert image data to image")
                return "No license plate detected", 0.0
            
            # Cache the processed image
            self.last_processed_image = img.copy()
            
            # Try both license plate detection methods
            # First, try the improved detector
            try:
                # Import here to avoid circular imports
                from improved_license_plate_detector import detect_plate
                plate_img, plate_box = detect_plate(image_data if isinstance(image_data, bytes) else cv2.imencode('.jpg', img)[1].tobytes())
                
                if plate_img is not None:
                    # Cache detection results
                    self.last_detection_boxes = [(*plate_box, 0.9)]  # x, y, w, h, confidence
                    
                    # Recognize text in the plate
                    text, confidence = self.recognizer.recognize_plate(plate_img)
                    
                    # If the OCR failed, return "No license plate detected"
                    if text is None or text == "UNKNOWN" or confidence < 0.1:
                        logger.info("Improved detector found plate but OCR failed")
                        self.last_processing_time = time.time() - start_time
                        return "No license plate detected", 0.0
                    
                    # Store the recognized plate
                    recognized_plates = [(text, confidence, plate_box)]
                    self.last_recognition_results = recognized_plates
                    
                    # Log the result
                    if self.enable_detail_logging:
                        logger.info(f"Detected license plate with improved detector: {text} with confidence {confidence}")
                    
                    # Update processing time
                    self.last_processing_time = time.time() - start_time
                    
                    return text, confidence
            except Exception as import_error:
                logger.warning(f"Could not use improved detector: {str(import_error)}")
            
            # Fall back to original detector if improved detector failed
            detection_boxes = self.detector.detect_plates(img)
            self.last_detection_boxes = detection_boxes
            
            # If no plates detected
            if not detection_boxes:
                logger.info("No license plates detected")
                self.last_processing_time = time.time() - start_time
                return "No license plate detected", 0.0
            
            # Sort boxes by confidence
            detection_boxes.sort(key=lambda x: x[4], reverse=True)
            
            # Extract the plates and recognize text
            recognized_plates = []
            for box in detection_boxes:
                # Extract the plate region
                plate_img = self.detector.extract_plate_region(img, box)
                
                # Recognize text in the plate
                text, confidence = self.recognizer.recognize_plate(plate_img)
                
                # Store the result if text was recognized
                if text and text != "UNKNOWN":
                    recognized_plates.append((text, confidence, box))
            
            # Cache recognition results
            self.last_recognition_results = recognized_plates
            
            # If no text was recognized in any plate
            if not recognized_plates:
                logger.info("No text recognized in detected plates")
                self.last_processing_time = time.time() - start_time
                return "No license plate detected", 0.0
            
            # Get the result with highest confidence
            best_result = max(recognized_plates, key=lambda x: x[1])
            text, confidence, _ = best_result
            
            # Log the result
            if self.enable_detail_logging:
                logger.info(f"Detected license plate: {text} with confidence {confidence}")
            
            # Update processing time
            self.last_processing_time = time.time() - start_time
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error in ANPR processing: {e}")
            self.last_processing_time = time.time() - start_time
            return "No license plate detected", 0.0
    
    def process_video_frame(self, frame):
        """
        Process a video frame to detect and recognize license plates
        Returns: (license_plate_text, confidence)
        """
        try:
            # Convert frame to bytes for compatibility
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Process the frame using the main method
            return self.process_image(frame_bytes)
            
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return "Error", 0.0
    
    def get_annotated_image(self):
        """Get the last processed image with annotations"""
        if self.last_processed_image is None or self.last_detection_boxes is None:
            return None
            
        # Extract text from recognition results
        detected_text = None
        if self.last_recognition_results:
            detected_text = [(r[0], r[1]) for r in self.last_recognition_results]
            
        # Annotate the image
        return self.detector.visually_annotate_image(
            self.last_processed_image,
            self.last_detection_boxes,
            detected_text
        )
    
    def get_processing_info(self):
        """Get information about the last processing operation"""
        return {
            "processing_time": self.last_processing_time,
            "num_plates_detected": len(self.last_detection_boxes) if self.last_detection_boxes else 0,
            "num_plates_recognized": len(self.last_recognition_results) if self.last_recognition_results else 0
        }


# Initialize the ANPR system
anpr_system = AdvancedANPR()

def detect_license_plate(image_data):
    """
    Main function to detect license plate in an image
    Args:
        image_data: Binary image data
    Returns:
        tuple: (license_plate_text, confidence)
    """
    return anpr_system.process_image(image_data)

def process_video_frame(frame):
    """
    Process a single video frame
    Args:
        frame: OpenCV frame from video or webcam
    Returns:
        tuple: (license_plate_text, confidence)
    """
    return anpr_system.process_video_frame(frame)

def get_annotated_image():
    """
    Get the last processed image with annotations
    Returns:
        numpy.ndarray: Annotated image
    """
    return anpr_system.get_annotated_image()

def get_processing_info():
    """
    Get information about the last processing operation
    Returns:
        dict: Processing information
    """
    return anpr_system.get_processing_info()