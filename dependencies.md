# Project Dependencies

This document lists all dependencies required for the License Plate Recognition system.

## Python Dependencies

```
flask==2.3.3
flask-sqlalchemy==3.1.1
gunicorn==23.0.0
numpy==2.2.5
opencv-python-headless==4.11.0.86
psycopg2-binary==2.9.9
pytesseract==0.3.13
requests==2.32.3
SQLAlchemy==2.0.29
Werkzeug==3.0.2
Pillow==11.2.1
```

## System Dependencies

- Tesseract OCR (for text recognition)
- PostgreSQL (for database)

## Installation Instructions

### Install Python Dependencies

```bash
pip install flask flask-sqlalchemy gunicorn numpy opencv-python-headless psycopg2-binary pytesseract requests SQLAlchemy Werkzeug Pillow
```

### Install Tesseract OCR

#### Windows
- Download the installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Install and add it to your PATH

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Install PostgreSQL

#### Windows
- Download and install from [postgresql.org](https://www.postgresql.org/download/windows/)

#### macOS
```bash
brew install postgresql
brew services start postgresql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres createuser --interactive
sudo -u postgres createdb licenseplatedb
```