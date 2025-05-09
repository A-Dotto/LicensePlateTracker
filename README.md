# Indian License Plate Recognition System

A web app that uses advanced computer vision techniques to identify and extract license plate information from vehicle images with high accuracy, specifically optimized for Indian license plates. This system focuses on CPU-based performance, avoiding the need for GPU acceleration.

## Features

- **Contour-Based Detection**: Uses OpenCV's contour detection algorithms to identify license plate regions efficiently
- **Indian License Plate Optimization**: Specifically designed to recognize Indian state and district codes (e.g., MH20, KA03)
- **Advanced OCR Processing**: Multiple image processing variations and OCR configurations for optimal text recognition
- **User-friendly Interface**: Sleek, modern interface for uploading and processing license plate images
- **Responsive Design**: Built with Bootstrap for a great experience on all devices
- **Video Detection**: Process video streams to detect license plates in real-time

## Project Structure

```
.
├── app.py                # Main Flask application file with routes and view functions
├── license_plate_detector.py  # License plate detection algorithm using OpenCV and PyTesseract
├── improved_license_plate_detector.py  # Enhanced contour-based detector
├── advanced_anpr.py      # Advanced ANPR implementation with multiple detection approaches
├── main.py               # Entry point for the application
├── static/               # Static files (CSS, JS)
│   ├── css/
│   │   └── style.css     # Custom CSS styles
│   └── js/
│       └── main.js       # Frontend JavaScript for image handling
├── templates/            # HTML templates
│   ├── index.html        # Main page template with upload functionality
│   ├── video.html        # Video detection page
│   └── layout.html       # Base layout template with common elements
├── setup.sh              # Setup script for Linux/macOS
├── setup.bat             # Setup script for Windows
├── dependencies.md       # List of project dependencies with installation instructions
└── README.md             # Project documentation
```

## Key Components and Their Purpose

### Core Application Files

- **main.py**: Application entry point that imports the Flask app.
- **app.py**: Contains all Flask routes and view functions for handling HTTP requests and responses.

### License Plate Detection Modules

- **license_plate_detector.py**: The primary detector with special optimizations for Indian license plates:
  - Advanced image preprocessing for improved text recognition
  - Pattern recognition for Indian state codes (e.g., MH, KA, TN, DL)
  - Multiple OCR configurations for optimal text extraction
  - Scoring system that prioritizes Indian license plate patterns

- **improved_license_plate_detector.py**: Contour-based detector using modern CV techniques:
  - Uses CPU-optimized routines for better compatibility
  - Implements bilateral filtering and Canny edge detection
  - Uses contour approximation for license plate identification
  - Handles "No plate" cases correctly
  
- **advanced_anpr.py**: Advanced ANPR implementation with fallback mechanisms:
  - Includes timeout handling for challenging image processing
  - Integrates multiple detection methods with graceful failure handling
  - Provides detailed logging and processing information
  - Includes visual feedback for debugging and user understanding

### Frontend Files

- **templates/index.html**: Main interface for uploading and processing images.
- **templates/video.html**: Interface for video-based license plate detection.
- **templates/layout.html**: Base template with navigation and common structure.
- **static/js/main.js**: JavaScript for handling image uploads and UI interactions.
- **static/css/style.css**: Custom styling for the application.

## Technologies Used

- **Backend**: Python 3.11, Flask 2.3
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5.3
- **Computer Vision**: OpenCV 4.8, PyTesseract 0.3
- **Deployment**: Gunicorn 23.0

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Tesseract OCR engine (version 4.1+ recommended)

## Installation

### Windows

1. Clone the repository:
   ```
   git clone https://github.com/A-Dotto/LicensePlateTracker.git
   cd LicensePlateTracker
   ```

2. Install Tesseract OCR:
   - Download the installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install and add it to your PATH

3. Create a virtual environment and activate it:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```
   pip install flask gunicorn numpy opencv-python pytesseract pillow scikit-image imutils
   ```

### macOS

1. Clone the repository:
   ```
   git clone https://github.com/A-Dotto/LicensePlateTracker.git
   cd LicensePlateTracker
   ```

2. Install Tesseract OCR using Homebrew:
   ```
   brew install tesseract
   ```

3. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:
   ```
   pip install flask gunicorn numpy opencv-python pytesseract pillow scikit-image imutils
   ```

### Linux (Ubuntu/Debian)

1. Clone the repository:
   ```
   git clone https://github.com/A-Dotto/LicensePlateTracker.git
   cd LicensePlateTracker
   ```

2. Install Tesseract OCR:
   ```
   sudo apt update
   sudo apt install tesseract-ocr
   ```

3. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:
   ```
   pip install flask gunicorn numpy opencv-python pytesseract pillow scikit-image imutils
   ```

## Running the Application

1. Start the application:
   ```
   python main.py
   ```
   or with Gunicorn:
   ```
   gunicorn --bind 0.0.0.0:5000 main:app
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload Images**:
   - Click the "Upload" card and select an image containing a vehicle with an Indian license plate
   - The system will automatically detect and extract the license plate text

2. **Video Detection**:
   - Navigate to the Video Detection page to process video streams or webcam input
   - License plates will be detected and recognized in real-time

## Algorithm Explanation

The license plate detection and recognition process follows these steps:

1. **Image Preprocessing**:
   - Conversion to grayscale
   - Noise reduction with bilateral filtering
   - Edge detection using Canny algorithm
   - Adaptive thresholding

2. **License Plate Localization**:
   - Contour detection and filtering based on geometric properties
   - Aspect ratio analysis to identify potential license plate regions
   - Selection of highest probability candidates

3. **Text Extraction Optimization**:
   - Multiple image processing variations (thresholding, dilation, edge enhancement)
   - Various OCR configurations to handle different fonts and conditions
   - Specific pattern recognition for Indian license plates

4. **Result Verification**:
   - Validation based on Indian license plate patterns (state code, district code)
   - Confidence scoring based on pattern match strength
   - Selection of highest probability result

## Troubleshooting

### Common Issues

1. **Tesseract OCR not found**:
   Make sure Tesseract is installed and the path is correctly set in your system environment variables.

2. **Image detection issues**:
   For better results, use clear images where the license plate is visible and well-lit.

3. **OCR accuracy problems**:
   The system is optimized for Indian license plates. If recognition is poor, ensure the image quality is good and the license plate is clearly visible.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgements

- [OpenCV](https://opencv.org/) for computer vision processing
- [PyTesseract](https://github.com/madmaze/pytesseract) for OCR capabilities
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the frontend interface
