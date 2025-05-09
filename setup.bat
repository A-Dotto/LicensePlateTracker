@echo off
REM Setup script for License Plate Recognition System on Windows

REM Check Python version
python --version | findstr /R "3\.[89]\.[0-9]*" > nul
if errorlevel 1 (
    echo Error: Python version must be at least 3.8.0
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install flask flask-sqlalchemy gunicorn numpy opencv-python-headless psycopg2-binary pytesseract requests SQLAlchemy Werkzeug Pillow

REM Check for Tesseract OCR
tesseract --version > nul 2>&1
if errorlevel 1 (
    echo Warning: Tesseract OCR not found in PATH
    echo Please install Tesseract OCR manually from:
    echo   https://github.com/UB-Mannheim/tesseract/wiki
    echo And make sure it's added to your PATH
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from example...
    copy .env.example .env
    echo Please edit .env with your database credentials
)

echo Setup complete! You can now run the application with:
echo   venv\Scripts\activate.bat  # if not already activated
echo   python main.py