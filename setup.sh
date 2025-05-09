#!/bin/bash
# Setup script for License Plate Recognition System

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python version must be at least 3.8.0"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install flask flask-sqlalchemy gunicorn numpy opencv-python-headless psycopg2-binary pytesseract requests SQLAlchemy Werkzeug Pillow

# Check for Tesseract OCR
if ! command -v tesseract &> /dev/null; then
    echo "Warning: Tesseract OCR not found in PATH"
    echo "Please install Tesseract OCR manually:"
    echo "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    echo "  - macOS: brew install tesseract"
    echo "  - Linux: sudo apt install tesseract-ocr"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env with your database credentials"
fi

echo "Setup complete! You can now run the application with:"
echo "  source venv/bin/activate  # if not already activated"
echo "  python main.py"