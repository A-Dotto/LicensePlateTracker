import cv2
import numpy as np
import pytesseract
import logging
import math
from skimage.transform import resize, warp, AffineTransform
from skimage import exposure

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def image_to_numpy(image_data):
    """Convert image data to numpy array"""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error converting image to numpy array: {str(e)}")
        return None

def deskew_image(image):
    """
    Deskew an image using Hough Line Transform to identify the dominant angle
    and correct for perspective distortion.
    """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle to be robust against outliers
        median_angle = float(np.median(angles))
        
        # Adjust angle to be between -45 and 45 degrees
        if median_angle < -45:
            median_angle = 90 + median_angle
        elif median_angle > 45:
            median_angle = median_angle - 90
        
        # Rotate image to deskew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return deskewed
    except Exception as e:
        logger.error(f"Error deskewing image: {str(e)}")
        return image

def enhance_image(image):
    """Apply advanced image enhancement techniques for better OCR"""
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Reduce noise with a Gaussian blur
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        return closing
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image

def correct_perspective(image, contour):
    """
    Correct perspective distortion of a license plate using the detected contour
    """
    try:
        # Get rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Use np.int32 instead of np.int0
        
        # Get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        # Set destination points for perspective transform
        # Order points: top-left, top-right, bottom-right, bottom-left
        src_pts = box.astype("float32")
        
        # Sort points by their y-coordinates
        src_pts = src_pts[np.argsort(src_pts[:, 1])]
        
        # Now sort the top and bottom points by their x-coordinates
        top = src_pts[:2]
        top = top[np.argsort(top[:, 0])]
        bottom = src_pts[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        # Combine points in order: top-left, top-right, bottom-right, bottom-left
        src_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
        
        # Destination points
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    except Exception as e:
        logger.error(f"Error correcting perspective: {str(e)}")
        return image

def find_best_plate_candidates(img_region):
    """
    Find multiple plate candidates in an image using different techniques
    and return a list of potential license plate images
    """
    try:
        # Convert to grayscale if not already
        if len(img_region.shape) == 3:
            gray_region = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = img_region.copy()
            
        # Create a copy for drawing contours (for debugging)
        color_img = cv2.cvtColor(gray_region, cv2.COLOR_GRAY2BGR) if len(img_region.shape) != 3 else img_region.copy()
        
        # Apply bilateral filter for preserving edges while reducing noise
        filtered = cv2.bilateralFilter(gray_region, 11, 17, 17)
        
        # Apply different edge detection techniques
        canny_edges = cv2.Canny(filtered, 30, 200)
        
        # Use Sobel for additional edge detection
        sobel_x = cv2.Sobel(filtered, cv2.CV_8U, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(filtered, cv2.CV_8U, 0, 1, ksize=3)
        sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        
        # Find contours in both edge images
        canny_contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sobel_contours, _ = cv2.findContours(sobel.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine and remove duplicates
        all_contours = list(canny_contours) + list(sobel_contours)
        
        # Filter contours by area
        min_area = 1000  # Minimum area to be considered (adjust as needed)
        filtered_contours = [c for c in all_contours if cv2.contourArea(c) > min_area]
        
        # Sort contours by area and keep the largest ones
        contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:15]
        
        candidates = []
        
        for contour in contours:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filter by aspect ratio typical for license plates (based on a range normally seen in license plates)
            if 1.5 <= aspect_ratio <= 5.0:
                # Extract region
                plate_candidate = img_region[y:y+h, x:x+w]
                candidates.append(plate_candidate)
        
        return candidates
    
    except Exception as e:
        logger.error(f"Error finding plate candidates: {str(e)}")
        return []

def process_region_for_plate(img_region):
    """
    Process a specific region of an image to extract license plate text
    with enhanced algorithms for handling oblique angles and perspective distortion.
    Optimized for Indian license plates in format MH20DV2366.
    
    Args:
        img_region: Region of image (numpy array) to process for license plate
        
    Returns:
        tuple: (license_plate_text, confidence)
    """
    # Initialize best result variables
    best_text = None
    best_confidence = 0.0
    
    try:
        # Convert to grayscale if not already
        if len(img_region.shape) == 3:
            gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_region.copy()
        
        # Apply bilateral filter for noise reduction while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and keep the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Process each potential license plate contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Check if the aspect ratio is reasonable for a license plate
            if 1.5 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                # Extract the region
                roi = gray[y:y+h, x:x+w]
                
                # Create multiple processing variations for better OCR results
                variations = []
                
                # Original image
                variations.append(roi)
                
                # Standard threshold
                _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variations.append(roi_thresh)
                
                # Inverted threshold
                variations.append(cv2.bitwise_not(roi_thresh))
                
                # Enhanced contrast
                enhanced = exposure.equalize_hist(roi) * 255
                enhanced = enhanced.astype(np.uint8)
                _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variations.append(enhanced_thresh)
                
                # Apply morphological operations
                kernel = np.ones((1, 1), np.uint8)
                dilated = cv2.dilate(roi_thresh, kernel, iterations=1)
                variations.append(dilated)
                
                # Edge enhancement
                edges = cv2.Canny(roi, 100, 200)
                variations.append(edges)
                
                # Different tesseract configs for optimal recognition
                configs = [
                    '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ]
                
                # Try all combinations of image variations and configs
                all_results = []
                for img_var in variations:
                    for config in configs:
                        try:
                            text = pytesseract.image_to_string(img_var, config=config)
                            text = ''.join(c for c in text if c.isalnum())
                            
                            if 4 <= len(text) <= 12:  # Increased max length for Indian plates
                                all_results.append(text)
                        except Exception as ocr_error:
                            logger.debug(f"OCR error with config {config}: {str(ocr_error)}")
                            continue
                
                # Process the results
                if all_results:
                    # Score results based on Indian plate patterns
                    scored_results = []
                    for result in all_results:
                        score = 0
                        
                        # Basic length check (longer is usually better)
                        score += min(len(result), 10) * 0.1
                        
                        # Check for Indian state code pattern (two letters followed by two digits)
                        if len(result) >= 4:
                            # Check if first 2 chars are letters (state code)
                            if result[:2].isalpha():
                                score += 1.0
                                # Common state codes get higher scores
                                if result[:2] in ['MH', 'DL', 'KA', 'TN', 'AP', 'UP', 'MP']:
                                    score += 2.0
                                
                                # Check if next 2 chars are digits (district code)
                                if result[2:4].isdigit():
                                    score += 1.5
                                    
                                    # Check full pattern (state, district, series, number)
                                    if len(result) >= 8 and result[4:6].isalpha():
                                        score += 2.0
                        
                        scored_results.append((result, score))
                    
                    # Sort by score (highest first)
                    scored_results.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take the highest scoring result
                    if scored_results:
                        best_result = scored_results[0][0]
                        result_score = scored_results[0][1]
                        
                        # Calculate confidence based on pattern match strength
                        confidence = min(0.99, 0.6 + (result_score / 10.0))
                        
                        # Update best result if better than current
                        if confidence > best_confidence:
                            best_text = best_result
                            best_confidence = confidence
                            
                            # If confidence is high enough, return immediately
                            if best_confidence > 0.85:
                                logger.info(f"High confidence match: {best_text} ({best_confidence:.2f})")
                                return best_text, best_confidence
        
        # If we found a good result, return it
        if best_text:
            logger.info(f"Found plate text: {best_text} ({best_confidence:.2f})")
            return best_text, best_confidence
        
        # Fallback: try processing the whole region directly
        try:
            # Try various thresholding methods on the whole image
            if len(img_region.shape) == 3:
                gray_all = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_all = img_region.copy()
            
            _, thresh_all = cv2.threshold(gray_all, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply OCR with Indian plate specific config
            text = pytesseract.image_to_string(
                thresh_all, 
                config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            text = ''.join(c for c in text if c.isalnum())
            
            if 4 <= len(text) <= 12:
                return text, 0.6
            
        except Exception as fallback_error:
            logger.error(f"Fallback OCR error: {str(fallback_error)}")
        
        # No valid result found
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Error processing region for license plate: {str(e)}")
        return None, 0.0

def detect_license_plate(image_data):
    """
    Detect license plate in an image and extract the text with improved algorithms
    for handling perspective distortion and oblique viewing angles.
    Optimized for performance and reliability.
    
    Args:
        image_data: Binary image data
        
    Returns:
        tuple: (license_plate_text, confidence)
    """
    try:
        # Convert image data to numpy array
        img = image_to_numpy(image_data)
        if img is None:
            logger.error("Failed to convert image to numpy array")
            return None, 0.0

        # Create a copy for processing
        original_img = img.copy()
        
        # Process the original image directly
        best_text, best_confidence = process_region_for_plate(original_img)
        
        # If we found a license plate with reasonable confidence, return it immediately
        if best_text and best_confidence >= 0.65:
            logger.info(f"Detected license plate: {best_text} with confidence {best_confidence}")
            return best_text, best_confidence
        
        # If no reasonable result was found, return whatever we have
        if best_text:
            logger.info(f"Detected license plate with low confidence: {best_text} ({best_confidence})")
            return best_text, best_confidence
        else:
            logger.info("No license plate text detected in the image")
            # Return a default value so the app doesn't crash
            return "No plate", 0.1
            
    except Exception as e:
        logger.error(f"Error in license plate detection: {str(e)}")
        # Return a default value to prevent app crashes
        return "ERROR", 0.1