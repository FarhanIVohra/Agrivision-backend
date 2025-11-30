"""
Crop Prediction Router
API endpoints for crop recommendation and prediction
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ..models.schemas import (
    CropPredictionRequest,
    CropPredictionResponse,
    CropRecommendation,
    ErrorResponse
)
from ..services.model_service import ModelService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize model service singleton
model_service = ModelService()

@router.post(
    "/recommend",
    response_model=CropPredictionResponse,
    summary="Get crop recommendations",
    description="Predict the most suitable crops based on soil and environmental parameters"
)
async def recommend_crops(
    request: Dict[str, Any],  # Changed from CropPredictionRequest to handle nested data
) -> CropPredictionResponse:
    """
    Recommend crops based on input parameters
    
    Args:
        request: Crop prediction request with soil and environmental data
        
    Returns:
        CropPredictionResponse with recommendations
    """
    try:
        logger.info(f"Received crop prediction request: {request}")
        
        # Prepare features from request (handles both nested and flat structures)
        features = model_service.prepare_features_from_request(request)
        
        # Get multiple crop recommendations
        recommendations = model_service.predict_multiple_crops(features, top_k=5)
        
        # Convert to response format
        crop_recommendations = [
            CropRecommendation(**rec) for rec in recommendations
        ]
        
        # Get prediction metadata
        metadata = model_service.get_prediction_metadata()
        
        # Create a mock CropPredictionRequest for response
        mock_request = CropPredictionRequest(
            nitrogen=features[0],
            phosphorus=features[1],
            potassium=features[2],
            temperature=features[3],
            humidity=features[4],
            ph=features[5],
            rainfall=features[6]
        )
        
        response = CropPredictionResponse(
            success=True,
            message="Crop recommendations generated successfully",
            input_parameters=mock_request,
            recommendations=crop_recommendations,
            total_crops=len(crop_recommendations),
            prediction_metadata=metadata
        )
        
        logger.info(f"Successfully generated {len(crop_recommendations)} recommendations")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in crop prediction: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input parameters: {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
    except Exception as e:
        logger.error(f"Unexpected error in crop prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@router.post(
    "/predict-single",
    response_model=Dict[str, Any],
    summary="Get single best crop recommendation",
    description="Predict the single most suitable crop based on input parameters"
)
async def predict_single_crop(
    request: Dict[str, Any],  # Changed from CropPredictionRequest to handle nested data
) -> Dict[str, Any]:
    """
    Predict the single best crop recommendation
    
    Args:
        request: Crop prediction request with soil and environmental data
        
    Returns:
        Dictionary with single crop recommendation
    """
    try:
        logger.info(f"Received single crop prediction request: {request}")
        
        # Prepare features from request (handles both nested and flat structures)
        features = model_service.prepare_features_from_request(request)
        
        # Get single crop prediction
        crop_id, confidence = model_service.predict_single_crop(features)
        
        # Get crop name from mapping
        crop_name = model_service.crop_mapping.get(crop_id, f"Unknown Crop {crop_id}")
        
        # Get prediction metadata
        metadata = model_service.get_prediction_metadata()
        
        response = {
            "success": True,
            "message": "Single crop prediction completed successfully",
            "input_parameters": request,
            "prediction": {
                "crop_id": crop_id,
                "crop_name": crop_name,
                "confidence": confidence,
                "suitability_score": round(confidence * 100, 2)
            },
            "prediction_metadata": metadata
        }
        
        logger.info(f"Successfully predicted crop: {crop_name} (confidence: {confidence:.3f})")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in single crop prediction: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input parameters: {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
    except Exception as e:
        logger.error(f"Unexpected error in single crop prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@router.get(
    "/crops",
    response_model=Dict[str, Any],
    summary="Get available crops",
    description="Get list of all crops that can be predicted by the model"
)
async def get_available_crops() -> Dict[str, Any]:
    """
    Get list of all available crops
    
    Args:
        model_service: Injected model service
        
    Returns:
        Dictionary with available crops information
    """
    try:
        model_info = model_service.get_model_info()
        
        return {
            "success": True,
            "message": "Available crops retrieved successfully",
            "total_crops": len(model_service.crop_mapping),
            "crops": [
                {"crop_id": crop_id, "crop_name": crop_name}
                for crop_id, crop_name in model_service.crop_mapping.items()
            ],
            "model_info": {
                "n_features": model_info.get("n_features"),
                "feature_names": model_info.get("feature_names"),
                "model_type": model_info.get("model_type")
            }
        }
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
    except Exception as e:
        logger.error(f"Error retrieving available crops: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving crop information"
        )

@router.get(
    "/model-info",
    response_model=Dict[str, Any],
    summary="Get model information",
    description="Get detailed information about the loaded ML model"
)
async def get_model_info() -> Dict[str, Any]:
    """
    Get detailed model information
    
    Args:
        model_service: Injected model service
        
    Returns:
        Dictionary with model information
    """
    try:
        model_info = model_service.get_model_info()
        
        return {
            "success": True,
            "message": "Model information retrieved successfully",
            "model_info": model_info
        }
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving model information"
        )