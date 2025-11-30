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

# Initialize services
pest_detection_service = PestDetectionService()
disease_management_service = DiseaseManagementService()

@router.post(
    "/detect-disease",
    response_model=Dict[str, Any],
    summary="Detect crop disease from image",
    description="Upload an image to detect crop diseases and get treatment recommendations"
)
async def detect_disease_from_upload(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, JPG)")
) -> Dict[str, Any]:
    """
    Detect crop disease from uploaded image file
    
    Args:
        file: Uploaded image file
        
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
        result = pest_detection_service.predict_from_image_bytes(content)
        
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
        #     treatment_recommendations = disease_management_service.get_treatment_recommendations(
        #         disease_name, 
        #         severity="moderate"
        #     )
        #     result['treatment_recommendations'] = treatment_recommendations
        
        logger.info(f"Disease detection completed: {result.get('prediction', 'Unknown')}")
        return result
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
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
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect crop disease from base64 encoded image
    
    Args:
        request: Dictionary containing base64_image field
        
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
        result = pest_detection_service.predict_from_base64(base64_image)
        
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
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
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
async def get_supported_diseases() -> Dict[str, Any]:
    """
    Get list of all supported diseases
    
    Args:
        service: Injected pest detection service
        
    Returns:
        Dictionary with supported diseases information
    """
    try:
        diseases = pest_detection_service.get_supported_diseases()
        
        return {
            "success": True,
            "message": "Supported diseases retrieved successfully",
            "total_diseases": len(diseases),
            "diseases": diseases
        }
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
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
async def get_model_info() -> Dict[str, Any]:
    """
    Get detailed pest detection model information
    
    Args:
        service: Injected pest detection service
        
    Returns:
        Dictionary with model information
    """
    try:
        model_info = pest_detection_service.get_model_info()
        
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

@router.post(
    "/batch-detect",
    response_model=Dict[str, Any],
    summary="Batch disease detection",
    description="Detect diseases in multiple images at once"
)
async def batch_detect_diseases(
    files: List[UploadFile] = File(..., description="Multiple image files")
) -> Dict[str, Any]:
    """
    Detect crop diseases in multiple images
    
    Args:
        files: List of uploaded image files
        
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
                result = pest_detection_service.predict_from_image_bytes(content)
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
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the images: {str(e)}"
        )

@router.post(
    "/",
    response_model=dict,
    summary="Detect crop disease",
    description="POST an image file to get disease, confidence (0-1), severity, treatment, and notes"
)
async def predict_disease_root(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, JPG)")
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

        raw = pest_detection_service.predict_from_image_bytes(content)
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