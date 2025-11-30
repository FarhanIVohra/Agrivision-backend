"""
Predictions Router
API endpoints for various prediction services (pest, rainfall, etc.)
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import random

from ..services.pest_detection_service import PestDetectionService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

pest_detection_service = PestDetectionService()

@router.post("/pest")
async def predict_pest_risk(request: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Use the PestDetectionService to calculate pest risk
        pest_risk = pest_detection_service.calculate_pest_risk(temperature, humidity, rainfall, season, crop_type)
        
        # Common pests for the conditions
        common_pests = get_common_pests(temperature, humidity, season, crop_type)
        
        response = {
            "success": True,
            "pest_risk": pest_risk,
            "environmental_factors": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "season": season,
                "crop_type": crop_type
            },
            "common_pests": common_pests,
            "recommendations": [],
            "prediction_date": datetime.now().isoformat()
        }
        
        logger.info(f"Pest risk prediction completed: {pest_risk['risk_level']} ({pest_risk['risk_score']}%)")
        return response
        
    except Exception as e:
        logger.error(f"Error predicting pest risk: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while predicting pest risk: {str(e)}"
        )

@router.post("/rainfall")
async def predict_rainfall(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict rainfall based on weather data
    
    Args:
        request: Dictionary containing weather parameters
        
    Returns:
        Dictionary with rainfall prediction
    """
    try:
        logger.info(f"Predicting rainfall for data: {request}")
        
        # Extract weather parameters
        temperature = request.get('temperature', 25.0)
        humidity = request.get('humidity', 60.0)
        pressure = request.get('pressure', 1013.25)
        wind_speed = request.get('wind_speed', 10.0)
        cloud_cover = request.get('cloud_cover', 50.0)
        season = request.get('season', 'summer')
        
        # Calculate rainfall probability
        rainfall_probability = calculate_rainfall_probability(
            temperature, humidity, pressure, wind_speed, cloud_cover, season
        )
        
        # Predict rainfall amount
        if rainfall_probability >= 80:
            rainfall_amount = random.uniform(20, 50)
            intensity = "Heavy"
        elif rainfall_probability >= 60:
            rainfall_amount = random.uniform(10, 25)
            intensity = "Moderate"
        elif rainfall_probability >= 40:
            rainfall_amount = random.uniform(2, 10)
            intensity = "Light"
        else:
            rainfall_amount = random.uniform(0, 2)
            intensity = "Very Light"
        
        # Generate forecast for next 7 days
        forecast = generate_rainfall_forecast(temperature, humidity, pressure)
        
        response = {
            "success": True,
            "rainfall_prediction": {
                "probability": round(rainfall_probability, 1),
                "expected_amount_mm": round(rainfall_amount, 1),
                "intensity": intensity,
                "confidence": min(90, max(60, rainfall_probability * 0.9))
            },
            "weather_factors": {
                "temperature": temperature,
                "humidity": humidity,
                "pressure": pressure,
                "wind_speed": wind_speed,
                "cloud_cover": cloud_cover,
                "season": season
            },
            "forecast_7_days": forecast,
            "recommendations": generate_rainfall_recommendations(rainfall_probability, rainfall_amount),
            "prediction_date": datetime.now().isoformat()
        }
        
        logger.info(f"Rainfall prediction completed: {rainfall_probability}% probability, {rainfall_amount}mm expected")
        return response
        
    except Exception as e:
        logger.error(f"Error predicting rainfall: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while predicting rainfall: {str(e)}"
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

def calculate_rainfall_probability(temperature: float, humidity: float, pressure: float, 
                                 wind_speed: float, cloud_cover: float, season: str) -> float:
    """Calculate rainfall probability based on weather conditions"""
    probability = 0
    
    # Humidity factor (most important)
    if humidity >= 90:
        probability += 40
    elif humidity >= 80:
        probability += 30
    elif humidity >= 70:
        probability += 20
    elif humidity >= 60:
        probability += 10
    
    # Cloud cover factor
    if cloud_cover >= 80:
        probability += 25
    elif cloud_cover >= 60:
        probability += 15
    elif cloud_cover >= 40:
        probability += 10
    
    # Pressure factor (low pressure increases rain chance)
    if pressure < 1000:
        probability += 20
    elif pressure < 1010:
        probability += 10
    elif pressure < 1020:
        probability += 5
    
    # Temperature factor
    if 20 <= temperature <= 30:
        probability += 10
    elif 15 <= temperature < 20 or 30 < temperature <= 35:
        probability += 5
    
    # Wind speed factor
    if 5 <= wind_speed <= 15:
        probability += 5
    
    # Season factor
    season_multiplier = {
        'monsoon': 1.5,
        'spring': 1.2,
        'summer': 0.8,
        'winter': 0.9
    }
    
    probability *= season_multiplier.get(season.lower(), 1.0)
    
    return min(100, max(0, probability))

def generate_rainfall_forecast(temperature: float, humidity: float, pressure: float) -> List[Dict[str, Any]]:
    """Generate 7-day rainfall forecast"""
    forecast = []
    base_date = datetime.now()
    
    for i in range(7):
        date = base_date + timedelta(days=i)
        
        # Simulate varying conditions
        temp_variation = random.uniform(-3, 3)
        humidity_variation = random.uniform(-10, 10)
        pressure_variation = random.uniform(-5, 5)
        
        day_temp = temperature + temp_variation
        day_humidity = max(0, min(100, humidity + humidity_variation))
        day_pressure = pressure + pressure_variation
        day_cloud_cover = random.uniform(20, 90)
        
        rain_prob = calculate_rainfall_probability(
            day_temp, day_humidity, day_pressure, 10, day_cloud_cover, 'summer'
        )
        
        if rain_prob >= 70:
            rain_amount = random.uniform(10, 30)
        elif rain_prob >= 50:
            rain_amount = random.uniform(5, 15)
        elif rain_prob >= 30:
            rain_amount = random.uniform(1, 8)
        else:
            rain_amount = 0
        
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "temperature": round(day_temp, 1),
            "humidity": round(day_humidity, 1),
            "rainfall_probability": round(rain_prob, 1),
            "expected_rainfall_mm": round(rain_amount, 1),
            "cloud_cover": round(day_cloud_cover, 1)
        })
    
    return forecast

def generate_rainfall_recommendations(probability: float, amount: float) -> List[str]:
    """Generate rainfall-based recommendations"""
    recommendations = []
    
    if probability >= 80:
        recommendations.extend([
            "Prepare for heavy rainfall - ensure proper drainage",
            "Postpone field activities if possible",
            "Protect crops from waterlogging",
            "Check irrigation systems and turn off if needed"
        ])
    elif probability >= 60:
        recommendations.extend([
            "Moderate rain expected - plan field activities accordingly",
            "Ensure drainage systems are clear",
            "Consider covering sensitive crops"
        ])
    elif probability >= 40:
        recommendations.extend([
            "Light rain possible - continue normal activities",
            "Good time for planting if soil moisture is adequate"
        ])
    else:
        recommendations.extend([
            "Low rain probability - continue irrigation as needed",
            "Monitor soil moisture levels",
            "Consider water conservation measures"
        ])
    
    if amount > 20:
        recommendations.append("Heavy rainfall expected - take flood precautions")
    elif amount > 10:
        recommendations.append("Moderate rainfall - ensure good field drainage")
    
    return recommendations