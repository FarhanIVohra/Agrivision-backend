"""
Pest Detection Router
API endpoints for crop disease detection and pest identification
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import base64
import io
from PIL import Image

from ..services.pest_detection_service import PestDetectionService
from ..services.disease_management_service import DiseaseManagementService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global service instances
pest_detection_service = None
disease_management_service = None

def get_pest_detection_service() -> PestDetectionService:
    """Dependency to get pest detection service instance"""
    global pest_detection_service
    if pest_detection_service is None:
        pest_detection_service = PestDetectionService()
    
    if not pest_detection_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Pest detection model is not available. Please try again later."
        )
    return pest_detection_service

def get_disease_management_service() -> DiseaseManagementService:
    """Dependency to get disease management service instance"""
    global disease_management_service
    if disease_management_service is None:
        disease_management_service = DiseaseManagementService()
    return disease_management_service

@router.post(
    "/detect-disease",
    response_model=Dict[str, Any],
    summary="Detect crop disease from image",
    description="Upload an image to detect crop diseases and get treatment recommendations"
)
async def detect_disease_from_upload(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, JPG)"),
    service: PestDetectionService = Depends(get_pest_detection_service),
    management_service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Detect crop disease from uploaded image file
    
    Args:
        file: Uploaded image file
        service: Injected pest detection service
        
    Returns:
        Dictionary with simplified disease prediction results (prediction and confidence only)
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file (JPEG, PNG, JPG)."
            )
        
        # Check file size (limit to 10MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="File size too large. Please upload an image smaller than 10MB."
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded. Please upload a valid image."
            )
        
        logger.info(f"Processing uploaded image: {file.filename} ({file_size} bytes)")
        
        # Get prediction from service
        result = service.predict_from_image_bytes(content)
        
        # Check if there was an error in the service
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Disease detection failed: {result['error']}"
            )
        
        # TODO: Treatment recommendations - To be implemented in future update
        # Get comprehensive treatment recommendations if disease detected
        # if result.get('prediction') and result.get('prediction') != 'healthy':
        #     disease_name = result.get('prediction')
        #     treatment_recommendations = management_service.get_treatment_recommendations(
        #         disease_name, 
        #         severity="moderate"
        #     )
        #     result['treatment_recommendations'] = treatment_recommendations
        
        logger.info(f"Disease detection completed: {result.get('prediction', 'Unknown')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )

@router.post(
    "/detect-disease-base64",
    response_model=Dict[str, Any],
    summary="Detect crop disease from base64 image",
    description="Send base64 encoded image to detect crop diseases and get treatment recommendations"
)
async def detect_disease_from_base64(
    request: Dict[str, Any],
    service: PestDetectionService = Depends(get_pest_detection_service)
) -> Dict[str, Any]:
    """
    Detect crop disease from base64 encoded image
    
    Args:
        request: Dictionary containing base64_image field
        service: Injected pest detection service
        
    Returns:
        Dictionary with simplified disease prediction results (prediction and confidence only)
    """
    try:
        # Validate request
        if "base64_image" not in request:
            raise HTTPException(
                status_code=400,
                detail="Missing 'base64_image' field in request body."
            )
        
        base64_image = request["base64_image"]
        if not base64_image or not isinstance(base64_image, str):
            raise HTTPException(
                status_code=400,
                detail="Invalid base64_image. Please provide a valid base64 encoded image string."
            )
        
        logger.info("Processing base64 encoded image")
        
        # Get prediction from service
        result = service.predict_from_base64(base64_image)
        
        # Check if there was an error in the service
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Disease detection failed: {result['error']}"
            )
        
        logger.info(f"Disease detection completed: {result.get('prediction', 'Unknown')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )

@router.get(
    "/supported-diseases",
    response_model=Dict[str, Any],
    summary="Get supported diseases",
    description="Get list of all diseases that can be detected by the model"
)
async def get_supported_diseases(
    service: PestDetectionService = Depends(get_pest_detection_service)
) -> Dict[str, Any]:
    """
    Get list of all supported diseases
    
    Args:
        service: Injected pest detection service
        
    Returns:
        Dictionary with supported diseases information
    """
    try:
        diseases = service.get_supported_diseases()
        
        return {
            "success": True,
            "message": "Supported diseases retrieved successfully",
            "total_diseases": len(diseases),
            "diseases": diseases
        }
        
    except Exception as e:
        logger.error(f"Error retrieving supported diseases: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving disease information"
        )

@router.get(
    "/model-info",
    response_model=Dict[str, Any],
    summary="Get pest detection model information",
    description="Get detailed information about the loaded pest detection model"
)
async def get_model_info(
    service: PestDetectionService = Depends(get_pest_detection_service)
) -> Dict[str, Any]:
    """
    Get detailed pest detection model information
    
    Args:
        service: Injected pest detection service
        
    Returns:
        Dictionary with model information
    """
    try:
        model_info = service.get_model_info()
        
        return {
            "success": True,
            "message": "Model information retrieved successfully",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving model information"
        )

@router.post(
    "/batch-detect",
    response_model=Dict[str, Any],
    summary="Batch disease detection",
    description="Detect diseases in multiple images at once"
)
async def batch_detect_diseases(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    service: PestDetectionService = Depends(get_pest_detection_service)
) -> Dict[str, Any]:
    """
    Detect crop diseases in multiple images
    
    Args:
        files: List of uploaded image files
        service: Injected pest detection service
        
    Returns:
        Dictionary with batch prediction results
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 10 images allowed per batch."
            )
        
        results = []
        total_size = 0
        
        for i, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": "Invalid file type"
                    })
                    continue
                
                # Read file content
                content = await file.read()
                file_size = len(content)
                total_size += file_size
                
                # Check individual file size
                if file_size > 10 * 1024 * 1024:  # 10MB limit per file
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": "File size too large (>10MB)"
                    })
                    continue
                
                # Check total batch size
                if total_size > 50 * 1024 * 1024:  # 50MB total limit
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": "Total batch size too large"
                    })
                    continue
                
                # Get prediction
                result = service.predict_from_image_bytes(content)
                result["file_index"] = i
                result["filename"] = file.filename
                result["file_size"] = file_size
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        successful_predictions = [r for r in results if r.get("success", False)]
        failed_predictions = [r for r in results if not r.get("success", False)]
        
        return {
            "success": True,
            "message": "Batch disease detection completed",
            "total_files": len(files),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(failed_predictions),
            "total_size_bytes": total_size,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch disease detection: {str(e)}")
@router.post(
    "/predict",
    response_model=Dict[str, Any],
    summary="Predict pest risk based on environmental data",
    description="Predict pest risk based on environmental conditions like temperature, humidity, rainfall"
)
async def predict_pest_risk(
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Predict pest risk based on environmental data
    
    Args:
        request: Dictionary containing environmental data
        
    Returns:
        Dictionary with pest risk prediction
    """
    try:
        logger.info(f"Predicting pest risk for data: {request}")
        
        # Extract environmental parameters
        temperature = request.get('temperature', 25.0)
        humidity = request.get('humidity', 60.0)
        rainfall = request.get('rainfall', 100.0)
        season = request.get('season', 'summer')
        crop_type = request.get('crop_type', 'rice')
        
        # Calculate pest risk based on environmental conditions
        risk_score = calculate_pest_risk(temperature, humidity, rainfall, season)
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "Very High"
            color = "red"
        elif risk_score >= 60:
            risk_level = "High"
            color = "orange"
        elif risk_score >= 40:
            risk_level = "Medium"
            color = "yellow"
        elif risk_score >= 20:
            risk_level = "Low"
            color = "lightgreen"
        else:
            risk_level = "Very Low"
            color = "green"
        
        # TODO: Pest management recommendations - To be implemented in future update
        # Generate recommendations
        # recommendations = generate_pest_recommendations(risk_score, temperature, humidity, rainfall)
        recommendations = []  # Temporarily empty until future implementation
        
        # Common pests for the conditions
        common_pests = get_common_pests(temperature, humidity, season, crop_type)
        
        response = {
            "success": True,
            "pest_risk": {
                "risk_score": round(risk_score, 1),
                "risk_level": risk_level,
                "color": color,
                "confidence": min(95, max(70, 100 - abs(risk_score - 50) * 0.5))
            },
            "environmental_factors": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "season": season,
                "crop_type": crop_type
            },
            "common_pests": common_pests,
            "recommendations": recommendations,
            "prediction_date": "2024-01-20T10:00:00Z"
        }
        
        logger.info(f"Pest risk prediction completed: {risk_level} ({risk_score}%)")
        return response
        
    except Exception as e:
        logger.error(f"Error predicting pest risk: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while predicting pest risk: {str(e)}"
        )


def calculate_pest_risk(temperature: float, humidity: float, rainfall: float, season: str) -> float:
    """Calculate pest risk score based on environmental conditions"""
    risk_score = 0
    
    # Temperature factor (optimal pest conditions: 25-35°C)
    if 25 <= temperature <= 35:
        risk_score += 30
    elif 20 <= temperature < 25 or 35 < temperature <= 40:
        risk_score += 20
    elif 15 <= temperature < 20 or 40 < temperature <= 45:
        risk_score += 10
    
    # Humidity factor (high humidity increases pest risk)
    if humidity >= 80:
        risk_score += 35
    elif humidity >= 60:
        risk_score += 25
    elif humidity >= 40:
        risk_score += 15
    else:
        risk_score += 5
    
    # Rainfall factor (moderate rainfall increases pest risk)
    if 50 <= rainfall <= 200:
        risk_score += 25
    elif 200 < rainfall <= 400:
        risk_score += 15
    elif rainfall > 400:
        risk_score += 10
    else:
        risk_score += 5
    
    # Season factor
    season_multiplier = {
        'summer': 1.2,
        'monsoon': 1.3,
        'winter': 0.7,
        'spring': 1.0
    }
    
    risk_score *= season_multiplier.get(season.lower(), 1.0)
    
    return min(100, max(0, risk_score))

# TODO: Pest management recommendations function - To be implemented in future update
# def generate_pest_recommendations(risk_score: float, temperature: float, humidity: float, rainfall: float) -> List[str]:
#     """Generate pest management recommendations"""
#     recommendations = []
#     
#     if risk_score >= 70:
#         recommendations.extend([
#             "Implement immediate pest monitoring",
#             "Consider preventive pesticide application",
#             "Increase field inspection frequency to daily",
#             "Set up pest traps in the field"
#         ])
#     elif risk_score >= 50:
#         recommendations.extend([
#             "Monitor crops regularly for pest signs",
#             "Prepare pest management tools",
#             "Check crops every 2-3 days",
#             "Consider biological pest control methods"
#         ])
#     else:
#         recommendations.extend([
#             "Continue regular monitoring",
#             "Maintain good field hygiene",
#             "Weekly crop inspection is sufficient"
#         ])
#     
#     if humidity > 80:
#         recommendations.append("Improve field drainage to reduce humidity")
#     
#     if temperature > 35:
#         recommendations.append("Provide shade or cooling measures if possible")
#     
#     if rainfall > 300:
#         recommendations.append("Ensure proper water drainage to prevent waterlogging")
#     
#     return recommendations

def get_common_pests(temperature: float, humidity: float, season: str, crop_type: str) -> List[Dict[str, Any]]:
    """Get common pests based on conditions"""
    pests = []
    
    # Temperature and humidity based pests
    if temperature >= 25 and humidity >= 60:
        pests.extend([
            {"name": "Aphids", "severity": "High", "description": "Small insects that suck plant sap"},
            {"name": "Whiteflies", "severity": "Medium", "description": "Small white flying insects"}
        ])
    
    if humidity >= 80:
        pests.extend([
            {"name": "Fungal diseases", "severity": "High", "description": "Various fungal infections"},
            {"name": "Slugs and snails", "severity": "Medium", "description": "Mollusks that eat plant leaves"}
        ])
    
    # Crop-specific pests
    crop_pests = {
        'rice': [
            {"name": "Brown planthopper", "severity": "High", "description": "Major rice pest"},
            {"name": "Rice stem borer", "severity": "Medium", "description": "Bores into rice stems"}
        ],
        'wheat': [
            {"name": "Wheat aphid", "severity": "Medium", "description": "Aphids specific to wheat"},
            {"name": "Armyworm", "severity": "High", "description": "Caterpillars that eat wheat leaves"}
        ],
        'maize': [
            {"name": "Fall armyworm", "severity": "High", "description": "Major maize pest"},
            {"name": "Corn borer", "severity": "Medium", "description": "Bores into corn stalks"}
        ]
    }
    
    if crop_type.lower() in crop_pests:
        pests.extend(crop_pests[crop_type.lower()])
    
    return pests[:5]  # Return top 5 pests


def suggest_basic_treatment(disease_name: str) -> str:
    name = (disease_name or "").lower()
    if "late blight" in name:
        return "Apply copper-based fungicide; remove infected leaves; improve air circulation."
    if "early blight" in name:
        return "Use fungicide (chlorothalonil/mancozeb); remove lower leaves; mulch to reduce soil splash."
    if "bacterial" in name:
        return "Remove infected tissue; apply copper sprays; avoid overhead irrigation."
    if "powdery mildew" in name:
        return "Use potassium bicarbonate or sulfur sprays; improve ventilation; avoid high humidity."
    if "leaf mold" in name:
        return "Increase airflow; reduce humidity; consider chlorothalonil-based treatment."
    if "rust" in name:
        return "Apply appropriate fungicide (strobilurin/triazole); remove heavily infected leaves."
    if "mite" in name:
        return "Use miticide or insecticidal soap; monitor underside of leaves; improve plant vigor."
    if "spot" in name:
        return "Copper-based fungicide; remove infected leaves; avoid wetting foliage."
    return "Use integrated pest management: monitor, remove infected parts, consider organic or copper-based treatments."

@router.post(
    "/",
    response_model=dict,
    summary="Detect crop disease",
    description="POST an image file to get disease, confidence (0-1), severity, treatment, and notes"
)
async def predict_disease_root(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, JPG)"),
    service: PestDetectionService = Depends(get_pest_detection_service)
) -> dict:
    # Small disease->treatment mapping per requirements
    treatment_map = {
        "leaf spot": "Use Copper-based fungicide and ensure proper sunlight",
        "powdery mildew": "Apply sulfur-based spray and avoid overwatering",
        "healthy": "No action needed",
    }

    try:
        if not image or not image.content_type or not image.content_type.startswith('image/'):
            return JSONResponse(status_code=400, content={
                "status": "failed",
                "message": "Invalid file type or missing image. Upload a JPEG/PNG image.",
                "error": "image_invalid"
            })

        content = await image.read()
        if not content:
            return JSONResponse(status_code=400, content={
                "status": "failed",
                "message": "Empty file uploaded. Please upload a valid image.",
                "error": "image_empty"
            })
        if len(content) > 10 * 1024 * 1024:
            return JSONResponse(status_code=400, content={
                "status": "failed",
                "message": "File too large (>10MB). Please upload a smaller image.",
                "error": "image_too_large"
            })

        raw = service.predict_from_image_bytes(content)
        if "error" in raw:
            return JSONResponse(status_code=500, content={
                "status": "failed",
                "message": f"Model error: {raw['error']}",
                "error": "inference_error"
            })

        predicted_label = (raw.get("prediction") or "Unknown").strip()
        confidence = float(raw.get("confidence", 0.0))  # 0-1

        # Fallback rule: under 0.45 -> Unknown
        if confidence < 0.45:
            return {
                "status": "success",
                "disease": "Unknown",
                "confidence": confidence,
                "severity": "low",
                "treatment": "Unable to identify. Please upload clearer image.",
                "notes": "Upload clearer image if confidence is low."
            }

        # Severity mapping from 0-1 confidence
        if confidence >= 0.75:
            severity = "high"
        elif confidence >= 0.60:
            severity = "medium"
        else:
            severity = "low"

        # Treatment guidance: prefer explicit mapping, fallback to heuristic
        key = predicted_label.lower()
        treatment = treatment_map.get(key)
        if not treatment:
            treatment = suggest_basic_treatment(predicted_label)

        return {
            "status": "success",
            "disease": predicted_label,
            "confidence": round(confidence, 4),
            "severity": severity,
            "treatment": treatment,
            "notes": "Upload clearer image if confidence is low."
        }

    except Exception as e:
        logger.error(f"Error in /api/pest-detection: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": "failed",
            "message": "Unexpected error during inference.",
            "error": str(e)
        })