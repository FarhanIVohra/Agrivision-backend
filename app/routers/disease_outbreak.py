"""
Disease Outbreak Prediction Router
API endpoints for predicting disease outbreaks based on environmental data
Aligns with AI-Driven Crop Disease Prediction and Management System requirements
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

from ..services.disease_management_service import DiseaseManagementService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global service instance
disease_management_service = None

def get_disease_management_service() -> DiseaseManagementService:
    """Dependency to get disease management service instance"""
    global disease_management_service
    if disease_management_service is None:
        disease_management_service = DiseaseManagementService()
    return disease_management_service

# Pydantic models for request/response
class EnvironmentalData(BaseModel):
    """Environmental data for disease outbreak prediction"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=60)
    humidity: float = Field(..., description="Relative humidity percentage", ge=0, le=100)
    rainfall: float = Field(..., description="Rainfall in mm", ge=0)
    wind_speed: float = Field(default=5.0, description="Wind speed in km/h", ge=0)
    crop_type: str = Field(default="general", description="Type of crop")
    field_location: Optional[str] = Field(default=None, description="Field location/region")
    soil_moisture: Optional[float] = Field(default=None, description="Soil moisture percentage", ge=0, le=100)
    previous_diseases: Optional[List[str]] = Field(default=[], description="Previously detected diseases in the field")

class OutbreakAlert(BaseModel):
    """Disease outbreak alert information"""
    alert_id: str
    disease_name: str
    risk_level: str
    probability: float
    affected_area: str
    recommended_actions: List[str]
    created_at: datetime

@router.post(
    "/predict-outbreak",
    response_model=Dict[str, Any],
    summary="Predict disease outbreak risk",
    description="Analyze environmental conditions to predict potential disease outbreaks"
)
async def predict_disease_outbreak(
    environmental_data: EnvironmentalData,
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Predict disease outbreak risk based on environmental conditions
    
    Args:
        environmental_data: Environmental parameters for analysis
        service: Disease management service instance
        
    Returns:
        Dictionary with outbreak risk assessment and recommendations
    """
    try:
        logger.info(f"Predicting disease outbreak for crop: {environmental_data.crop_type}")
        
        # Convert Pydantic model to dictionary
        env_dict = environmental_data.dict()
        
        # Get outbreak prediction
        prediction_result = service.predict_disease_outbreak_risk(env_dict)
        
        # Add additional context
        prediction_result.update({
            "analysis_timestamp": datetime.now().isoformat(),
            "location": environmental_data.field_location or "Not specified",
            "crop_type": environmental_data.crop_type,
            "data_quality": "Good" if all([
                environmental_data.temperature,
                environmental_data.humidity,
                environmental_data.rainfall
            ]) else "Partial"
        })
        
        logger.info(f"Outbreak prediction completed: {prediction_result.get('risk_level', 'Unknown')}")
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error predicting disease outbreak: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict disease outbreak: {str(e)}"
        )

@router.get(
    "/risk-assessment",
    response_model=Dict[str, Any],
    summary="Get current disease risk assessment",
    description="Get current disease risk assessment based on weather parameters"
)
async def get_risk_assessment(
    temperature: float = Query(..., description="Current temperature in Celsius"),
    humidity: float = Query(..., description="Current humidity percentage"),
    rainfall: float = Query(default=0, description="Recent rainfall in mm"),
    crop_type: str = Query(default="general", description="Type of crop"),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Get current disease risk assessment based on weather parameters
    
    Args:
        temperature: Current temperature
        humidity: Current humidity
        rainfall: Recent rainfall
        crop_type: Type of crop
        service: Disease management service instance
        
    Returns:
        Dictionary with current risk assessment
    """
    try:
        environmental_data = {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "crop_type": crop_type,
            "wind_speed": 5.0  # Default value
        }
        
        risk_assessment = service.predict_disease_outbreak_risk(environmental_data)
        
        # Add quick recommendations
        risk_assessment["quick_actions"] = _get_quick_actions(risk_assessment.get("risk_level", "Low"))
        
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Error getting risk assessment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk assessment: {str(e)}"
        )

@router.post(
    "/early-warning",
    response_model=Dict[str, Any],
    summary="Generate early warning alerts",
    description="Generate early warning alerts for potential disease outbreaks"
)
async def generate_early_warning(
    environmental_data: EnvironmentalData,
    alert_threshold: float = Query(default=0.6, description="Risk threshold for alerts", ge=0, le=1),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Generate early warning alerts for potential disease outbreaks
    
    Args:
        environmental_data: Environmental conditions
        alert_threshold: Risk threshold for generating alerts
        service: Disease management service instance
        
    Returns:
        Dictionary with early warning information
    """
    try:
        # Get outbreak prediction
        prediction = service.predict_disease_outbreak_risk(environmental_data.dict())
        
        alerts = []
        overall_risk = prediction.get("overall_risk", 0)
        
        if overall_risk >= alert_threshold:
            # Generate main alert
            main_alert = {
                "alert_id": f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "alert_type": "Disease Outbreak Warning",
                "risk_level": prediction.get("risk_level", "Unknown"),
                "probability": overall_risk,
                "message": f"High disease outbreak risk detected for {environmental_data.crop_type}",
                "recommended_actions": prediction.get("recommendations", []),
                "valid_until": (datetime.now() + timedelta(days=7)).isoformat(),
                "priority": "High" if overall_risk >= 0.8 else "Medium"
            }
            alerts.append(main_alert)
            
            # Generate specific disease alerts
            for disease_risk in prediction.get("specific_disease_risks", []):
                if disease_risk.get("probability", 0) >= alert_threshold:
                    disease_alert = {
                        "alert_id": f"DISEASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{disease_risk['disease'].replace(' ', '_')}",
                        "alert_type": "Specific Disease Warning",
                        "disease_name": disease_risk["disease"],
                        "risk_level": disease_risk["risk_level"],
                        "probability": disease_risk["probability"],
                        "message": f"High risk of {disease_risk['disease']} outbreak",
                        "conditions": disease_risk.get("conditions", ""),
                        "priority": "High" if disease_risk.get("probability", 0) >= 0.8 else "Medium"
                    }
                    alerts.append(disease_alert)
        
        return {
            "alerts_generated": len(alerts),
            "alerts": alerts,
            "overall_risk": overall_risk,
            "risk_level": prediction.get("risk_level", "Unknown"),
            "monitoring_recommendations": prediction.get("monitoring_priority", ""),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating early warning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate early warning: {str(e)}"
        )

@router.get(
    "/prevention-strategies",
    response_model=Dict[str, Any],
    summary="Get disease prevention strategies",
    description="Get comprehensive disease prevention strategies for specific crops"
)
async def get_prevention_strategies(
    crop_type: str = Query(..., description="Type of crop"),
    region: str = Query(default="general", description="Geographic region"),
    season: str = Query(default="current", description="Growing season"),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Get comprehensive disease prevention strategies for specific crops
    
    Args:
        crop_type: Type of crop
        region: Geographic region
        season: Growing season
        service: Disease management service instance
        
    Returns:
        Dictionary with prevention strategies and best practices
    """
    try:
        # Get preventive measures from service
        preventive_measures = service._get_preventive_measures(crop_type.lower())
        
        # Generate comprehensive prevention strategy
        prevention_strategy = {
            "crop_type": crop_type,
            "region": region,
            "season": season,
            "general_practices": preventive_measures,
            "seasonal_recommendations": _get_seasonal_recommendations(season),
            "monitoring_schedule": service._generate_monitoring_schedule(),
            "emergency_preparedness": [
                "Maintain emergency treatment supplies",
                "Have contact information for agricultural extension",
                "Keep weather monitoring tools ready",
                "Prepare isolation areas for infected plants"
            ],
            "integrated_management": [
                "Combine cultural, biological, and chemical controls",
                "Use resistant varieties when available",
                "Implement crop rotation strategies",
                "Maintain beneficial insect populations"
            ],
            "record_keeping": [
                "Document all treatments and applications",
                "Track weather conditions and disease incidents",
                "Monitor treatment effectiveness",
                "Maintain field history records"
            ]
        }
        
        return prevention_strategy
        
    except Exception as e:
        logger.error(f"Error getting prevention strategies: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get prevention strategies: {str(e)}"
        )

@router.get(
    "/disease-forecast",
    response_model=Dict[str, Any],
    summary="Get disease forecast",
    description="Get 7-day disease outbreak forecast based on weather predictions"
)
async def get_disease_forecast(
    crop_type: str = Query(..., description="Type of crop"),
    location: str = Query(default="general", description="Location for forecast"),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Get 7-day disease outbreak forecast based on weather predictions
    
    Args:
        crop_type: Type of crop
        location: Location for forecast
        service: Disease management service instance
        
    Returns:
        Dictionary with disease forecast information
    """
    try:
        # Generate mock 7-day forecast (in real implementation, this would use weather API)
        forecast_days = []
        
        for day in range(7):
            date = datetime.now() + timedelta(days=day)
            
            # Mock weather data (replace with actual weather API)
            mock_weather = {
                "temperature": 22 + (day * 2),  # Varying temperature
                "humidity": 70 + (day * 3),     # Varying humidity
                "rainfall": max(0, 10 - day * 2),  # Decreasing rainfall
                "crop_type": crop_type
            }
            
            # Get risk assessment for the day
            risk_assessment = service.predict_disease_outbreak_risk(mock_weather)
            
            forecast_day = {
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "weather": {
                    "temperature": mock_weather["temperature"],
                    "humidity": mock_weather["humidity"],
                    "rainfall": mock_weather["rainfall"]
                },
                "disease_risk": {
                    "overall_risk": risk_assessment.get("overall_risk", 0),
                    "risk_level": risk_assessment.get("risk_level", "Low"),
                    "specific_risks": risk_assessment.get("specific_disease_risks", [])
                },
                "recommendations": risk_assessment.get("recommendations", [])[:3]  # Top 3 recommendations
            }
            
            forecast_days.append(forecast_day)
        
        return {
            "crop_type": crop_type,
            "location": location,
            "forecast_period": "7 days",
            "generated_at": datetime.now().isoformat(),
            "daily_forecast": forecast_days,
            "summary": {
                "highest_risk_day": max(forecast_days, key=lambda x: x["disease_risk"]["overall_risk"])["date"],
                "average_risk": sum(day["disease_risk"]["overall_risk"] for day in forecast_days) / 7,
                "critical_days": len([day for day in forecast_days if day["disease_risk"]["overall_risk"] >= 0.7])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting disease forecast: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get disease forecast: {str(e)}"
        )

def _get_quick_actions(risk_level: str) -> List[str]:
    """Get quick actions based on risk level"""
    actions_map = {
        "Very High": [
            "Apply emergency fungicide treatment immediately",
            "Increase monitoring to twice daily",
            "Prepare for potential crop loss mitigation"
        ],
        "High": [
            "Apply preventive fungicide treatment",
            "Monitor fields daily",
            "Prepare treatment equipment"
        ],
        "Moderate": [
            "Continue regular monitoring",
            "Check weather forecasts",
            "Ensure treatment supplies are available"
        ],
        "Low": [
            "Maintain routine field inspections",
            "Continue preventive practices",
            "Monitor weather conditions"
        ],
        "Very Low": [
            "Continue normal operations",
            "Routine monitoring sufficient",
            "Focus on general crop health"
        ]
    }
    
    return actions_map.get(risk_level, actions_map["Low"])

def _get_seasonal_recommendations(season: str) -> List[str]:
    """Get seasonal disease prevention recommendations"""
    seasonal_map = {
        "spring": [
            "Apply dormant season treatments",
            "Prepare fields with proper drainage",
            "Select resistant varieties for planting",
            "Clean and disinfect equipment"
        ],
        "summer": [
            "Monitor for heat stress and disease interaction",
            "Ensure adequate irrigation without overwatering",
            "Maintain good air circulation",
            "Apply preventive treatments during cool periods"
        ],
        "monsoon": [
            "Improve field drainage systems",
            "Increase monitoring frequency",
            "Apply protective fungicides before rain",
            "Remove infected plant material promptly"
        ],
        "winter": [
            "Clean up crop residues",
            "Apply dormant season treatments",
            "Plan crop rotation for next season",
            "Maintain equipment and storage facilities"
        ],
        "current": [
            "Follow season-appropriate practices",
            "Monitor weather conditions closely",
            "Adjust treatments based on current conditions",
            "Maintain regular field inspections"
        ]
    }
    
    return seasonal_map.get(season.lower(), seasonal_map["current"])