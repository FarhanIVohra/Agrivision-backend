"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class CropPredictionRequest(BaseModel):
    """Request model for crop prediction with soil and environmental parameters"""
    
    nitrogen: float = Field(
        ..., 
        ge=0, 
        le=200, 
        description="Nitrogen content in soil (0-200 kg/ha)"
    )
    phosphorus: float = Field(
        ..., 
        ge=0, 
        le=150, 
        description="Phosphorus content in soil (0-150 kg/ha)"
    )
    potassium: float = Field(
        ..., 
        ge=0, 
        le=300, 
        description="Potassium content in soil (0-300 kg/ha)"
    )
    temperature: float = Field(
        ..., 
        ge=-10, 
        le=50, 
        description="Average temperature in Celsius (-10 to 50°C)"
    )
    humidity: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Relative humidity percentage (0-100%)"
    )
    ph: float = Field(
        ..., 
        ge=3.0, 
        le=10.0, 
        description="Soil pH level (3.0-10.0)"
    )
    rainfall: float = Field(
        ..., 
        ge=0, 
        le=3000, 
        description="Annual rainfall in mm (0-3000mm)"
    )
    farm_size: float = Field(
        default=1.0,
        ge=0.1,
        le=1000,
        description="Farm size in hectares (0.1-1000 ha)"
    )

    @validator('nitrogen', 'phosphorus', 'potassium')
    def validate_nutrients(cls, v):
        """Validate nutrient values are reasonable"""
        if v < 0:
            raise ValueError('Nutrient values cannot be negative')
        return round(v, 5)

    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is within reasonable agricultural range"""
        if v < -10 or v > 50:
            raise ValueError('Temperature must be between -10°C and 50°C')
        return round(v, 2)

    @validator('humidity')
    def validate_humidity(cls, v):
        """Validate humidity percentage"""
        if v < 0 or v > 100:
            raise ValueError('Humidity must be between 0% and 100%')
        return round(v, 2)

    @validator('ph')
    def validate_ph(cls, v):
        """Validate pH level"""
        if v < 3.0 or v > 10.0:
            raise ValueError('pH must be between 3.0 and 10.0')
        return round(v, 2)

    @validator('rainfall')
    def validate_rainfall(cls, v):
        """Validate rainfall amount"""
        if v < 0:
            raise ValueError('Rainfall cannot be negative')
        return round(v, 2)

    class Config:
        schema_extra = {
            "example": {
                "nitrogen": 90.0,
                "phosphorus": 42.0,
                "potassium": 43.0,
                "temperature": 20.87,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.93
            }
        }

class CropRecommendation(BaseModel):
    """Individual crop recommendation model"""
    
    crop_id: int = Field(..., description="Crop identifier")
    crop_name: str = Field(..., description="Name of the recommended crop")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    suitability_score: float = Field(..., ge=0, le=100, description="Suitability score (0-100)")
    
    class Config:
        schema_extra = {
            "example": {
                "crop_id": 1,
                "crop_name": "Rice",
                "confidence": 0.85,
                "suitability_score": 85.2
            }
        }

class CropPredictionResponse(BaseModel):
    """Response model for crop prediction results"""
    
    success: bool = Field(True, description="Whether the prediction was successful")
    message: str = Field("Prediction completed successfully", description="Response message")
    input_parameters: CropPredictionRequest = Field(..., description="Input parameters used for prediction")
    recommendations: List[CropRecommendation] = Field(..., description="List of crop recommendations")
    total_crops: int = Field(..., description="Total number of crop recommendations")
    prediction_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata about the prediction"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Prediction completed successfully",
                "input_parameters": {
                    "nitrogen": 90.0,
                    "phosphorus": 42.0,
                    "potassium": 43.0,
                    "temperature": 20.87,
                    "humidity": 82.0,
                    "ph": 6.5,
                    "rainfall": 202.93
                },
                "recommendations": [
                    {
                        "crop_id": 1,
                        "crop_name": "Rice",
                        "confidence": 0.85,
                        "suitability_score": 85.2
                    }
                ],
                "total_crops": 1,
                "prediction_metadata": {
                    "model_version": "1.0",
                    "prediction_time": "2024-01-01T12:00:00Z"
                }
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: bool = Field(True, description="Indicates an error occurred")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": True,
                "message": "Invalid input parameters",
                "status_code": 400,
                "details": {
                    "field": "nitrogen",
                    "issue": "Value must be between 0 and 200"
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., description="API health status")
    model_status: str = Field(..., description="ML model status")
    api_version: str = Field(..., description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_status": "loaded",
                "api_version": "1.0.0"
            }
        }