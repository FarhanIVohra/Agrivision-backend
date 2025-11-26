"""
Models package for AgriSmart API
Contains Pydantic models for request/response validation
"""

from .schemas import (
    CropPredictionRequest,
    CropRecommendation,
    CropPredictionResponse,
    ErrorResponse,
    HealthResponse
)

__all__ = [
    "CropPredictionRequest",
    "CropRecommendation", 
    "CropPredictionResponse",
    "ErrorResponse",
    "HealthResponse"
]