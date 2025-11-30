"""
Routers package for AgriSmart API
Contains API route handlers
"""

from . import crop_prediction, profitable_crops

__all__ = ["crop_prediction", "profitable_crops"]