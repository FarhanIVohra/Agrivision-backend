"""
Disease Database Router
API endpoints for accessing comprehensive disease information database
Supports the AI-Driven Crop Disease Prediction and Management System
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
import logging
from pydantic import BaseModel

from ..services.disease_database_service import DiseaseInformationService

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global service instance
disease_db_service = None

def get_disease_db_service() -> DiseaseInformationService:
    """Dependency to get disease database service instance"""
    global disease_db_service
    if disease_db_service is None:
        disease_db_service = DiseaseInformationService()
    return disease_db_service

# Pydantic models for request/response
class SymptomSearchRequest(BaseModel):
    symptoms: List[str]
    crop_type: Optional[str] = None

class TreatmentRequest(BaseModel):
    disease_name: str
    severity: str = "moderate"
    crop_type: Optional[str] = None

class PreventionRequest(BaseModel):
    crop_type: str
    region: Optional[str] = None

@router.get("/disease/{disease_name}", 
           summary="Get comprehensive disease information",
           description="Retrieve detailed information about a specific crop disease including symptoms, treatment, and prevention")
async def get_disease_info(
    disease_name: str,
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Get comprehensive information about a specific disease
    
    Args:
        disease_name: Name of the disease to look up
        
    Returns:
        Comprehensive disease information including symptoms, treatment, and prevention
    """
    try:
        logger.info(f"Retrieving disease information for: {disease_name}")
        
        disease_info = db_service.get_disease_information(disease_name)
        
        if "error" in disease_info:
            raise HTTPException(status_code=404, detail=disease_info["error"])
        
        return {
            "status": "success",
            "disease_info": disease_info,
            "message": f"Retrieved comprehensive information for {disease_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving disease information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/search-by-symptoms",
            summary="Search diseases by symptoms",
            description="Find possible diseases based on observed symptoms")
async def search_diseases_by_symptoms(
    request: SymptomSearchRequest,
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Search for diseases based on observed symptoms
    
    Args:
        request: Search request with symptoms and optional crop type
        
    Returns:
        List of possible diseases with match scores
    """
    try:
        logger.info(f"Searching diseases by symptoms: {request.symptoms}")
        
        if not request.symptoms:
            raise HTTPException(status_code=400, detail="At least one symptom must be provided")
        
        results = db_service.search_diseases_by_symptoms(
            symptoms=request.symptoms,
            crop_type=request.crop_type
        )
        
        return {
            "status": "success",
            "search_results": results,
            "total_matches": len(results),
            "search_criteria": {
                "symptoms": request.symptoms,
                "crop_type": request.crop_type
            },
            "message": f"Found {len(results)} possible disease matches"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching diseases by symptoms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/treatment-recommendations",
            summary="Get treatment recommendations",
            description="Get detailed treatment recommendations for a specific disease")
async def get_treatment_recommendations(
    request: TreatmentRequest,
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Get detailed treatment recommendations for a specific disease
    
    Args:
        request: Treatment request with disease name, severity, and optional crop type
        
    Returns:
        Detailed treatment recommendations
    """
    try:
        logger.info(f"Getting treatment recommendations for: {request.disease_name}")
        
        treatment_info = db_service.get_treatment_recommendations(
            disease_name=request.disease_name,
            severity=request.severity,
            crop_type=request.crop_type
        )
        
        if "error" in treatment_info:
            raise HTTPException(status_code=404, detail=treatment_info["error"])
        
        return {
            "status": "success",
            "treatment_recommendations": treatment_info,
            "disease_name": request.disease_name,
            "severity": request.severity,
            "message": f"Retrieved treatment recommendations for {request.disease_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting treatment recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/prevention-strategies",
            summary="Get prevention strategies",
            description="Get comprehensive prevention strategies for a specific crop")
async def get_prevention_strategies(
    request: PreventionRequest,
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Get comprehensive prevention strategies for a specific crop
    
    Args:
        request: Prevention request with crop type and optional region
        
    Returns:
        Comprehensive prevention strategies
    """
    try:
        logger.info(f"Getting prevention strategies for crop: {request.crop_type}")
        
        prevention_info = db_service.get_prevention_strategies(
            crop_type=request.crop_type,
            region=request.region
        )
        
        if "error" in prevention_info:
            raise HTTPException(status_code=404, detail=prevention_info["error"])
        
        return {
            "status": "success",
            "prevention_strategies": prevention_info,
            "crop_type": request.crop_type,
            "region": request.region,
            "message": f"Retrieved prevention strategies for {request.crop_type}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prevention strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/database-stats",
           summary="Get database statistics",
           description="Get statistics about the disease database coverage and content")
async def get_database_stats(
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Get statistics about the disease database
    
    Returns:
        Database statistics and coverage information
    """
    try:
        logger.info("Retrieving database statistics")
        
        stats = db_service.get_disease_database_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return {
            "status": "success",
            "database_stats": stats,
            "message": "Retrieved database statistics successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/diseases/list",
           summary="List all diseases",
           description="Get a list of all diseases in the database with basic information")
async def list_all_diseases(
    crop_type: Optional[str] = Query(None, description="Filter by crop type"),
    pathogen_type: Optional[str] = Query(None, description="Filter by pathogen type"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Get a list of all diseases in the database
    
    Args:
        crop_type: Optional filter by crop type
        pathogen_type: Optional filter by pathogen type
        severity: Optional filter by severity level
        
    Returns:
        List of diseases with basic information
    """
    try:
        logger.info("Retrieving list of all diseases")
        
        # Get all diseases from the service
        all_diseases = []
        
        for disease_key, disease_info in db_service.disease_database.items():
            # Apply filters
            if crop_type and crop_type.lower() not in [crop.lower() for crop in disease_info.get("crop_types", [])]:
                continue
            if pathogen_type and disease_info.get("pathogen_type", "").lower() != pathogen_type.lower():
                continue
            if severity and disease_info.get("severity", "").lower() != severity.lower():
                continue
            
            all_diseases.append({
                "disease_key": disease_key,
                "name": disease_info.get("name", ""),
                "scientific_name": disease_info.get("scientific_name", ""),
                "crop_types": disease_info.get("crop_types", []),
                "pathogen_type": disease_info.get("pathogen_type", ""),
                "severity": disease_info.get("severity", ""),
                "economic_impact": disease_info.get("economic_impact", ""),
                "description": disease_info.get("description", "")[:200] + "..." if len(disease_info.get("description", "")) > 200 else disease_info.get("description", "")
            })
        
        return {
            "status": "success",
            "diseases": all_diseases,
            "total_count": len(all_diseases),
            "filters_applied": {
                "crop_type": crop_type,
                "pathogen_type": pathogen_type,
                "severity": severity
            },
            "message": f"Retrieved {len(all_diseases)} diseases from database"
        }
        
    except Exception as e:
        logger.error(f"Error listing diseases: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/crops/{crop_type}/diseases",
           summary="Get diseases for specific crop",
           description="Get all diseases that affect a specific crop type")
async def get_crop_diseases(
    crop_type: str,
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Get all diseases that affect a specific crop type
    
    Args:
        crop_type: Type of crop to get diseases for
        
    Returns:
        List of diseases affecting the specified crop
    """
    try:
        logger.info(f"Retrieving diseases for crop: {crop_type}")
        
        crop_diseases = []
        
        for disease_key, disease_info in db_service.disease_database.items():
            if crop_type.lower() in [crop.lower() for crop in disease_info.get("crop_types", [])]:
                # Get additional information
                symptoms = db_service.symptom_database.get(disease_key, {})
                has_treatment = disease_key in db_service.treatment_database
                
                crop_diseases.append({
                    "disease_key": disease_key,
                    "name": disease_info.get("name", ""),
                    "scientific_name": disease_info.get("scientific_name", ""),
                    "severity": disease_info.get("severity", ""),
                    "pathogen_type": disease_info.get("pathogen_type", ""),
                    "economic_impact": disease_info.get("economic_impact", ""),
                    "description": disease_info.get("description", ""),
                    "key_symptoms": symptoms.get("diagnostic_features", [])[:3],
                    "has_treatment_info": has_treatment,
                    "favorable_conditions": disease_info.get("favorable_conditions", {})
                })
        
        # Sort by severity (very_high, high, moderate, low)
        severity_order = {"very_high": 4, "high": 3, "moderate": 2, "low": 1, "unknown": 0}
        crop_diseases.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
        
        return {
            "status": "success",
            "crop_type": crop_type,
            "diseases": crop_diseases,
            "total_count": len(crop_diseases),
            "message": f"Found {len(crop_diseases)} diseases affecting {crop_type}"
        }
        
    except Exception as e:
        logger.error(f"Error getting crop diseases: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health",
           summary="Health check for disease database service",
           description="Check if the disease database service is running properly")
async def health_check(
    db_service: DiseaseInformationService = Depends(get_disease_db_service)
) -> Dict[str, Any]:
    """
    Health check endpoint for disease database service
    
    Returns:
        Service health status and basic statistics
    """
    try:
        stats = db_service.get_disease_database_stats()
        
        return {
            "status": "healthy",
            "service": "Disease Database Service",
            "database_loaded": True,
            "total_diseases": stats.get("total_diseases", 0),
            "timestamp": "2024-01-01T00:00:00Z",
            "message": "Disease database service is running properly"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "Disease Database Service",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z",
            "message": "Disease database service is experiencing issues"
        }