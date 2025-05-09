import base64
import logging
import os
import time
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

# Import the license plate detection function from our improved detector
from improved_license_plate_detector import detect_plate
from license_plate_detector import detect_license_plate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'insecure-dev-key')  # For sessions and flashing

# Set up file upload handling
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global storage for the last processed image and results
# This replaces database storage with simple in-memory storage
last_processed = {
    'original_image': None,     # Original image data
    'annotated_image': None,    # Annotated image showing license plate
    'license_plate': None,      # Detected license plate text
    'confidence': 0,            # Confidence score
    'detection_time': None      # Time taken for detection
}

def allowed_file(filename):
    """Check if a filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_if_needed(image_data, max_size=(1200, 800)):
    """Resize image if it's too large"""
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return image_data  # Return original if can't be decoded
    
    h, w = img.shape[:2]
    
    # Resize if image is too large (to improve processing performance)
    if h > max_size[1] or w > max_size[0]:
        # Calculate new dimensions while maintaining aspect ratio
        if h > w:
            new_h = max_size[1]
            new_w = int(w * (new_h / h))
        else:
            new_w = max_size[0]
            new_h = int(h * (new_w / w))
        
        # Resize the image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    
    return image_data  # Return original if no resize needed

@app.route('/')
def index():
    """Home page displaying image upload form and results"""
    # We don't use database pagination anymore
    return render_template('index.html', 
                          last_processed=last_processed,
                          show_results='original_image' in last_processed and last_processed['original_image'] is not None)

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle image uploads and license plate detection
    Returns a redirect on success or a JSON response on error
    """
    if 'file' not in request.files:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'No file part in the request'}), 400
        flash('No file part in the request', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'No selected file'}), 400
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'File type not allowed'}), 400
        flash('File type not allowed', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Read image binary data
        image_data = file.read()
        
        # Resize image if too large for better performance
        image_data = resize_image_if_needed(image_data)
        
        # Start timing for detection
        start_time = time.time()
        
        # First try with the improved detector
        plate_coords = None
        try:
            # Try to extract license plate using improved detector
            plate_img, plate_coords = detect_plate(image_data)
            
            license_plate_text = "No license plate detected"
            confidence = 0.0
            
            # If plate detected, process with OCR
            if plate_img is not None:
                # Use our original detector which includes OCR for text recognition
                license_plate_text, confidence = detect_license_plate(image_data)
                
                # If detection still fails, use a fallback message
                if license_plate_text is None or license_plate_text == "":
                    license_plate_text = "No license plate detected"
                    confidence = 0.0
        except Exception as detect_error:
            logger.error(f"Error in improved detector: {str(detect_error)}")
            # Fall back to original detector
            license_plate_text, confidence = detect_license_plate(image_data)
        
        # Measure detection time
        detection_time = time.time() - start_time
        
        # Create annotated image showing detection results
        annotated_image = None
        try:
            # Create an annotated image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if plate_coords:
                x, y, w, h = plate_coords
                # Draw rectangle around the plate
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw text with license plate info
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Ensure text is a string
                if license_plate_text is None:
                    display_txt = "No plate"
                else:
                    display_txt = f"{license_plate_text} ({confidence:.2f})" if confidence > 0 else license_plate_text
                cv2.putText(img, str(display_txt), (x, y - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # If no plate found, add a message
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "No license plate detected", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Convert back to base64 for display
            _, buffer = cv2.imencode('.jpg', img)
            annotated_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as annotate_error:
            logger.error(f"Error creating annotated image: {str(annotate_error)}")
            # Just use the original image if annotation fails
            annotated_image = base64.b64encode(image_data).decode('utf-8')
        
        # Store the original image as base64 for display
        original_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Update global storage with processed results
        global last_processed
        last_processed = {
            'original_image': original_base64,
            'annotated_image': annotated_image,
            'license_plate': license_plate_text,
            'confidence': confidence,
            'filename': secure_filename(file.filename) if file.filename else "uploaded_image.jpg",
            'detection_time': detection_time
        }
        
        # Return appropriate response based on request type
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'license_plate': license_plate_text,
                'confidence': confidence,
                'annotated_image': annotated_image,
                'detection_time': detection_time
            })
        
        flash(f'Image processed successfully. License plate: {license_plate_text}', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'An unexpected error occurred during upload'}), 500
            
        flash('An unexpected error occurred', 'danger')
        return redirect(url_for('index'))

@app.route('/video')
def video():
    """
    Video processing page for license plate detection from webcam or uploaded videos
    """
    return render_template('video.html')

@app.route('/api/detect-license-plate', methods=['POST'])
def api_detect_license_plate():
    """
    API endpoint for detecting license plates in images
    Used by the video processing page
    """
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No image file provided'
        })
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'Empty file name'
        })
    
    try:
        # Read image binary data
        image_data = file.read()
        
        # First try with the improved detector for coordinates
        plate_img, plate_coords = detect_plate(image_data)
        # Initialize to None if not returned
        if plate_coords is None:
            plate_coords = (100, 150, 200, 50)  # Default values
        
        # Then use the original detector for text recognition
        license_plate_text, confidence = detect_license_plate(image_data)
        
        # If detection fails, use a fallback message
        if license_plate_text is None or license_plate_text == "":
            license_plate_text = "No license plate detected"
            confidence = 0.0
            
        # For display in UI elements, use a shorter version
        display_text = license_plate_text
        if license_plate_text == "No license plate detected":
            display_text = "No plate"
        
        # Use actual coordinates if available, otherwise fallback to fixed position
        bbox = {
            'x': plate_coords[0] if plate_coords else 100,
            'y': plate_coords[1] if plate_coords else 150,
            'width': plate_coords[2] if plate_coords else 200,
            'height': plate_coords[3] if plate_coords else 50,
            'label': display_text
        }
        
        return jsonify({
            'success': True,
            'license_plate': license_plate_text,
            'confidence': confidence,
            'bbox': bbox
        })
        
    except Exception as e:
        logger.error(f"Error in API license plate detection: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error processing image'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)