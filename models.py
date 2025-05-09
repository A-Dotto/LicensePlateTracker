from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import BYTEA, TEXT

db = SQLAlchemy()

class Car(db.Model):
    """Model for car images and their license plate information"""
    __tablename__ = 'cars'
    
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(TEXT, nullable=False)  # Base64 encoded image as TEXT for PostgreSQL
    license_plate = db.Column(db.String(20), nullable=True, index=True)  # Add index for faster searches
    confidence = db.Column(db.Float, nullable=True)
    filename = db.Column(db.String(255), nullable=True)
    manually_edited = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)  # Add index for sorting
    
    def __init__(self, **kwargs):
        """Initialize a Car instance with the provided attributes"""
        # Allow explicit setting of attributes during initialization
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        return f'<Car {self.id} - {self.license_plate}>'
