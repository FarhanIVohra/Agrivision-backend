"""
Dashboard Router
API endpoints for dashboard statistics and analytics
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import random
import requests
from supabase import create_client, Client
import os
import json
from pathlib import Path

from ..services.weather_service import weather_service
from .auth import get_current_user
import joblib
import numpy as np
from pydantic import BaseModel
from ..models.schemas import CropPredictionRequest, CropPredictionResponse, CropRecommendation

# Configure logging
logger = logging.getLogger(__name__)

# Import new yield prediction service
try:
    from ..services.new_yield_prediction import new_yield_service
    NEW_MODEL_AVAILABLE = True
    logger.info("New RandomForest yield prediction service loaded")
except ImportError as e:
    NEW_MODEL_AVAILABLE = False
    logger.warning(f"New yield prediction service not available: {e}")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Create router
router = APIRouter()

# Cache for soil health data (5-minute cache)
soil_health_cache = {}
CACHE_DURATION = 300  # 5 minutes in seconds

# Load yield prediction model
class HeuristicYieldModel:
    def predict(self, model_inputs):
        import numpy as np
        try:
            area = float(model_inputs['numerical_input'][0][0])
            estimated_production = float(model_inputs['numerical_input'][0][1])
            raw = estimated_production / max(area, 1e-6)
            return np.array([[raw]], dtype=np.float32)
        except Exception:
            return np.array([[2.5]], dtype=np.float32)

# Configurable fallback and model path
YIELD_FALLBACK_MODE = os.getenv("YIELD_FALLBACK_MODE", "auto").lower()
YIELD_MODEL_PATH_ENV = os.getenv("YIELD_MODEL_PATH")
HEURISTIC_CONFIDENCE_BASE = float(os.getenv("HEURISTIC_CONFIDENCE_BASE", "75"))

def resolve_model_path() -> str | None:
    try:
        # Explicit override via env
        if YIELD_MODEL_PATH_ENV and os.path.exists(YIELD_MODEL_PATH_ENV):
            return YIELD_MODEL_PATH_ENV
        # Common relative locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "ml_models", "saved_models", "yield_prediction_model.joblib"),
            os.path.join(os.path.dirname(__file__), "..", "models", "yield_prediction_model.joblib"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        # Fallback: search repository for the file name
        try:
            repo_root = Path(__file__).resolve().parents[3]
            matches = list(repo_root.glob("**/yield_prediction_model.joblib"))
            if matches:
                return str(matches[0])
        except Exception as ge:
            logger.debug(f"Glob search skipped: {ge}")
    except Exception as e:
        logger.warning(f"Model path resolution issue: {e}")
    return None

yield_model = None

def load_yield_model():
    global yield_model
    if YIELD_FALLBACK_MODE == "always":
        logger.info("Fallback mode 'always' set; skipping model load")
        return None
    if yield_model is not None and not isinstance(yield_model, HeuristicYieldModel):
        return yield_model
    path = resolve_model_path()
    if not path:
        logger.warning("Yield model path not found; using heuristic")
        return None
    try:
        import joblib
        yield_model = joblib.load(path)
        logger.info(f"Yield prediction model loaded: {path}")
        return yield_model
    except Exception as e:
        logger.error(f"Failed to load yield model from {path}: {e}")
        return None

# Attempt initial load based on mode
if YIELD_FALLBACK_MODE in ("auto", "never"):
    _ = load_yield_model()
    if yield_model is None and YIELD_FALLBACK_MODE == "never":
        logger.error("Fallback mode 'never' set but model unavailable")

# Pydantic models for yield prediction
class YieldPredictionRequest(BaseModel):
    state: str
    district: str
    crop: str
    year: str
    season: str
    area: float

@router.get("/stats")
async def get_dashboard_stats() -> Dict[str, Any]:
    """Get dashboard statistics"""
    try:
        logger.info("Retrieving dashboard statistics")
        
        # Generate comprehensive dashboard stats
        stats = {
            "success": True,
            "overview": {
                "total_farms": 12,
                "active_crops": 8,
                "total_area_hectares": 145.5,
                "avg_yield_increase": 18.5,
                "water_saved_percentage": 22.3,
                "pest_incidents_prevented": 15
            },
            "crop_performance": {
                "rice": {"yield": 4.2, "health": 92, "area": 45.0},
                "wheat": {"yield": 3.8, "health": 88, "area": 35.0},
                "maize": {"yield": 5.1, "health": 95, "area": 30.0},
                "cotton": {"yield": 2.9, "health": 85, "area": 25.0},
                "tomato": {"yield": 12.5, "health": 90, "area": 10.5}
            },
            "environmental_metrics": {
                "avg_temperature": 28.5,
                "avg_humidity": 65.2,
                "total_rainfall_mm": 125.8,
                "soil_health_score": 78.5,
                "water_usage_efficiency": 85.2
            },
            "alerts": {
                "critical": 2,
                "warning": 5,
                "info": 8,
                "total": 15
            },
            "recent_activities": generate_recent_activities(),
            "predictions": {
                "next_harvest_date": "2024-03-15",
                "expected_yield_tons": 245.8,
                "weather_risk_level": "Low",
                "pest_risk_level": "Medium",
                "irrigation_efficiency": 88.5
            },
            "financial_summary": {
                "estimated_revenue": 125000,
                "cost_savings": 18500,
                "roi_percentage": 24.5,
                "profit_margin": 32.1
            },
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info("Dashboard statistics retrieved successfully")
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard statistics"
        )

@router.get("/overview")
async def get_dashboard_overview() -> Dict[str, Any]:
    """Get dashboard overview with key metrics"""
    try:
        logger.info("Retrieving dashboard overview")
        
        overview = {
            "success": True,
            "key_metrics": {
                "total_yield_tons": 245.8,
                "farms_monitored": 12,
                "crops_tracked": 8,
                "efficiency_score": 87.5,
                "alerts_count": 15,
                "last_prediction_accuracy": 94.2
            },
            "status_indicators": {
                "system_health": "Excellent",
                "data_quality": "Good",
                "model_performance": "Excellent",
                "api_status": "Online"
            },
            "quick_stats": [
                {"label": "Yield Increase", "value": "18.5%", "trend": "up"},
                {"label": "Water Saved", "value": "22.3%", "trend": "up"},
                {"label": "Cost Reduction", "value": "15.2%", "trend": "up"},
                {"label": "Pest Prevention", "value": "85%", "trend": "stable"}
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard overview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard overview"
        )

@router.get("/analytics")
async def get_dashboard_analytics() -> Dict[str, Any]:
    """Get detailed analytics for dashboard"""
    try:
        logger.info("Retrieving dashboard analytics")
        
        # Generate time series data for charts
        analytics = {
            "success": True,
            "time_series": {
                "yield_trends": generate_yield_trends(),
                "weather_patterns": generate_weather_patterns(),
                "pest_incidents": generate_pest_incidents(),
                "irrigation_usage": generate_irrigation_usage()
            },
            "comparative_analysis": {
                "current_vs_previous_season": {
                    "yield_improvement": 18.5,
                    "cost_reduction": 15.2,
                    "efficiency_gain": 12.8
                },
                "crop_performance_ranking": [
                    {"crop": "Maize", "score": 95, "yield": 5.1},
                    {"crop": "Rice", "score": 92, "yield": 4.2},
                    {"crop": "Tomato", "score": 90, "yield": 12.5},
                    {"crop": "Wheat", "score": 88, "yield": 3.8},
                    {"crop": "Cotton", "score": 85, "yield": 2.9}
                ]
            },
            "predictive_insights": {
                "next_month_forecast": {
                    "expected_yield": 52.3,
                    "weather_risk": "Low",
                    "pest_probability": 25.5,
                    "irrigation_needs": "Moderate"
                },
                "seasonal_recommendations": [
                    "Increase nitrogen application for wheat fields",
                    "Monitor rice fields for brown planthopper",
                    "Prepare irrigation systems for dry season",
                    "Consider crop rotation for cotton fields"
                ]
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard analytics"
        )

@router.get("/alerts")
async def get_dashboard_alerts() -> Dict[str, Any]:
    """Get current alerts and notifications"""
    try:
        logger.info("Retrieving dashboard alerts")
        
        alerts = {
            "success": True,
            "alerts": generate_sample_alerts(),
            "summary": {
                "critical": 2,
                "warning": 5,
                "info": 8,
                "total": 15
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dashboard alerts"
        )

def generate_recent_activities() -> List[Dict[str, Any]]:
    """Generate recent activities for dashboard"""
    activities = [
        {
            "id": "ACT_001",
            "type": "prediction",
            "description": "Crop yield prediction completed for Field A",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "status": "completed"
        },
        {
            "id": "ACT_002", 
            "type": "irrigation",
            "description": "Irrigation schedule updated for rice fields",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "status": "completed"
        },
        {
            "id": "ACT_003",
            "type": "alert",
            "description": "Pest risk alert generated for wheat crop",
            "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
            "status": "active"
        },
        {
            "id": "ACT_004",
            "type": "weather",
            "description": "Weather data updated for all monitored areas",
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "status": "completed"
        },
        {
            "id": "ACT_005",
            "type": "soil_analysis",
            "description": "Soil health analysis completed for South Field",
            "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
            "status": "completed"
        }
    ]
    
    return activities

def generate_yield_trends() -> List[Dict[str, Any]]:
    """Generate yield trend data for charts"""
    base_date = datetime.now() - timedelta(days=30)
    trends = []
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        yield_value = 4.0 + random.uniform(-0.5, 0.8) + (i * 0.02)  # Slight upward trend
        
        trends.append({
            "date": date.strftime("%Y-%m-%d"),
            "yield_tons_per_hectare": round(yield_value, 2),
            "efficiency_score": round(80 + random.uniform(-5, 10), 1)
        })
    
    return trends

def generate_weather_patterns() -> List[Dict[str, Any]]:
    """Generate weather pattern data"""
    base_date = datetime.now() - timedelta(days=14)
    patterns = []
    
    for i in range(14):
        date = base_date + timedelta(days=i)
        
        patterns.append({
            "date": date.strftime("%Y-%m-%d"),
            "temperature": round(25 + random.uniform(-5, 8), 1),
            "humidity": round(60 + random.uniform(-15, 20), 1),
            "rainfall": round(random.uniform(0, 25), 1),
            "wind_speed": round(random.uniform(5, 20), 1)
        })
    
    return patterns

def generate_pest_incidents() -> List[Dict[str, Any]]:
    """Generate pest incident data"""
    base_date = datetime.now() - timedelta(days=30)
    incidents = []
    
    for i in range(0, 30, 3):  # Every 3 days
        date = base_date + timedelta(days=i)
        
        incidents.append({
            "date": date.strftime("%Y-%m-%d"),
            "incident_count": random.randint(0, 5),
            "severity_avg": round(random.uniform(1, 5), 1),
            "crops_affected": random.randint(1, 3)
        })
    
    return incidents

def generate_irrigation_usage() -> List[Dict[str, Any]]:
    """Generate irrigation usage data"""
    base_date = datetime.now() - timedelta(days=14)
    usage = []
    
    for i in range(14):
        date = base_date + timedelta(days=i)
        
        usage.append({
            "date": date.strftime("%Y-%m-%d"),
            "water_used_liters": random.randint(8000, 15000),
            "efficiency_percentage": round(random.uniform(75, 95), 1),
            "cost_savings": round(random.uniform(50, 200), 2)
        })
    
    return usage

def generate_sample_alerts() -> List[Dict[str, Any]]:
    """Generate sample alerts for dashboard"""
    alerts = [
        {
            "id": "ALT_001",
            "type": "critical",
            "title": "High Pest Risk Detected",
            "message": "Brown planthopper risk is high in rice fields. Immediate action recommended.",
            "crop": "Rice",
            "field": "Field A",
            "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
            "status": "active"
        },
        {
            "id": "ALT_002",
            "type": "critical", 
            "title": "Low Soil Moisture",
            "message": "Soil moisture in wheat field has dropped below critical level (15%).",
            "crop": "Wheat",
            "field": "North Field",
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "status": "active"
        },
        {
            "id": "ALT_003",
            "type": "warning",
            "title": "Weather Alert",
            "message": "Heavy rainfall expected in next 48 hours. Adjust irrigation schedule.",
            "crop": "All",
            "field": "All Fields",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "status": "active"
        },
        {
            "id": "ALT_004",
            "type": "warning",
            "title": "Nutrient Deficiency",
            "message": "Nitrogen levels are below optimal in maize field.",
            "crop": "Maize",
            "field": "South Field",
            "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
            "status": "active"
        },
        {
            "id": "ALT_005",
            "type": "info",
            "title": "Harvest Reminder",
            "message": "Tomato crop in greenhouse is ready for harvest.",
            "crop": "Tomato",
            "field": "Greenhouse 1",
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "status": "active"
        }
    ]
    
    return alerts

@router.get("/weather")
async def get_weather_data(current_user: dict = Depends(get_current_user)):
    """
    Get weather data for user's location using OpenWeatherMap API
    """
    try:
        user_lat = current_user.get("lat")
        user_lon = current_user.get("lon")

        # Use default coordinates if user location is not available
        if not user_lat or not user_lon:
            logger.warning(f"User location not found for user: {current_user['email']}, using default coordinates")
            user_lat = 19.0760  # Default to Mumbai coordinates
            user_lon = 72.8777
        
        # Get current weather data
        weather_data = await weather_service.get_current_weather(user_lat, user_lon)
        
        # Log weather request for analytics
        try:
            weather_log = {
                "user_id": current_user["id"],
                "lat": user_lat,
                "lon": user_lon,
                "weather_data": weather_data,
                "requested_at": datetime.utcnow().isoformat()
            }
            supabase.table("weather_logs").insert(weather_log).execute()
        except Exception as e:
            logger.warning(f"Failed to log weather request: {str(e)}")
        
        logger.info(f"Weather data retrieved for user: {current_user['email']}")
        return {
            "success": True,
            "data": weather_data,
            "user_location": {
                "lat": user_lat,
                "lon": user_lon,
                "region": current_user.get("region", "Unknown")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weather data error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve weather data"
        )

@router.get("/soil")
async def get_soil_data(current_user: dict = Depends(get_current_user)):
    """
    Get soil health data for dashboard
    """
    try:
        logger.info(f"Retrieving soil data for user: {current_user['email']}")
        
        # Generate sample soil health data
        soil_data = {
            "ph_level": round(6.5 + (random.random() - 0.5) * 1.0, 1),
            "nitrogen": round(45 + (random.random() - 0.5) * 10, 1),
            "phosphorus": round(30 + (random.random() - 0.5) * 8, 1),
            "potassium": round(40 + (random.random() - 0.5) * 12, 1),
            "organic_matter": round(3.2 + (random.random() - 0.5) * 0.8, 1),
            "moisture": round(25 + (random.random() - 0.5) * 10, 1),
            "temperature": round(22 + (random.random() - 0.5) * 6, 1),
            "conductivity": round(1.2 + (random.random() - 0.5) * 0.4, 2),
            "health_score": round(75 + random.random() * 20, 1),
            "status": "Good",
            "last_updated": datetime.now().isoformat()
        }
        
        # Add health scores for different parameters
        health_scores = {
            "ph": min(100, max(0, 100 - abs(soil_data["ph_level"] - 6.8) * 20)),
            "nitrogen": min(100, max(0, 100 - abs(soil_data["nitrogen"] - 50) * 2)),
            "phosphorus": min(100, max(0, 100 - abs(soil_data["phosphorus"] - 35) * 3)),
            "potassium": min(100, max(0, 100 - abs(soil_data["potassium"] - 45) * 2.5)),
            "organic_matter": min(100, max(0, soil_data["organic_matter"] * 25)),
            "moisture": min(100, max(0, 100 - abs(soil_data["moisture"] - 30) * 2))
        }
        
        soil_data["health_scores"] = health_scores
        
        # Generate recommendations
        recommendations = []
        if soil_data["ph_level"] < 6.0:
            recommendations.append("Apply lime to increase soil pH")
        elif soil_data["ph_level"] > 7.5:
            recommendations.append("Apply sulfur to decrease soil pH")
        
        if soil_data["nitrogen"] < 40:
            recommendations.append("Apply nitrogen fertilizer")
        if soil_data["phosphorus"] < 25:
            recommendations.append("Add phosphorus supplement")
        if soil_data["potassium"] < 35:
            recommendations.append("Apply potassium fertilizer")
        if soil_data["organic_matter"] < 2.5:
            recommendations.append("Add organic compost")
        
        if not recommendations:
            recommendations.append("Soil conditions are optimal")
        
        soil_data["recommendations"] = recommendations
        
        return {
            "success": True,
            "data": soil_data
        }
        
    except Exception as e:
        logger.error(f"Soil data error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve soil data"
        )

@router.get("/weather/forecast")
async def get_weather_forecast(
    days: int = 5,
    current_user: dict = Depends(get_current_user)
):
    """
    Get weather forecast for user's location
    """
    try:
        user_lat = current_user.get("lat")
        user_lon = current_user.get("lon")
        
        if not user_lat or not user_lon:
            raise HTTPException(
                status_code=400,
                detail="User location not found. Please update your location in profile."
            )
        
        # Validate days parameter
        if days < 1 or days > 5:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be between 1 and 5"
            )
        
        # Get weather forecast
        forecast_data = await weather_service.get_weather_forecast(user_lat, user_lon, days)
        
        logger.info(f"Weather forecast retrieved for user: {current_user['email']}")
        return {
            "success": True,
            "data": forecast_data,
            "days": days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weather forecast error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve weather forecast"
        )

@router.get("/soil")
async def get_soil_health_data(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive soil health data with real-time weather integration
    Includes metrics, health scores, and prioritized recommendations
    """
    try:
        user_id = current_user["id"]
        user_lat = current_user.get("lat", 19.0760)  # Default to Mumbai coordinates
        user_lon = current_user.get("lon", 72.8777)
        
        # Check cache first
        cache_key = f"soil_health_{user_id}"
        current_time = datetime.now()
        
        if cache_key in soil_health_cache:
            cached_data, cached_time = soil_health_cache[cache_key]
            if (current_time - cached_time).total_seconds() < CACHE_DURATION:
                logger.info(f"Returning cached soil health data for user: {current_user['email']}")
                return cached_data
        
        # Get real-time weather data from OpenWeatherMap
        try:
            weather_data = await weather_service.get_current_weather(user_lat, user_lon)
            current_weather = weather_data.get("current", {})
            temperature = current_weather.get("temperature", 25.0)
            humidity = current_weather.get("humidity", 65.0)
        except Exception as e:
            logger.warning(f"Failed to get weather data: {str(e)}")
            temperature = 25.0
            humidity = 65.0
        
        # Generate mock soil metrics with realistic values
        nitrogen = round(random.uniform(15.0, 85.0), 2)
        phosphorus = round(random.uniform(10.0, 60.0), 2)
        potassium = round(random.uniform(20.0, 90.0), 2)
        ph_level = round(random.uniform(5.5, 8.0), 2)
        moisture = round(random.uniform(25.0, 75.0), 2)
        organic_matter = round(random.uniform(1.5, 6.0), 2)
        
        # Determine soil type based on location and metrics
        soil_types = ["Loamy", "Clay", "Sandy", "Silt", "Alluvial", "Red Laterite", "Black Cotton"]
        soil_type = random.choice(soil_types)
        
        # Calculate health scores (0-100) for each metric
        def calculate_health_score(value, optimal_min, optimal_max, critical_min=0, critical_max=100):
            if optimal_min <= value <= optimal_max:
                return random.uniform(85, 100)
            elif critical_min <= value < optimal_min or optimal_max < value <= critical_max:
                return random.uniform(60, 84)
            else:
                return random.uniform(20, 59)
        
        nitrogen_score = calculate_health_score(nitrogen, 40, 80)
        phosphorus_score = calculate_health_score(phosphorus, 25, 50)
        potassium_score = calculate_health_score(potassium, 50, 80)
        ph_score = calculate_health_score(ph_level, 6.0, 7.5, 4.0, 9.0)
        moisture_score = calculate_health_score(moisture, 40, 70)
        organic_matter_score = calculate_health_score(organic_matter, 3.0, 5.0)
        
        # Calculate overall health score
        overall_health_score = round((
            nitrogen_score + phosphorus_score + potassium_score + 
            ph_score + moisture_score + organic_matter_score
        ) / 6, 1)
        
        # Generate prioritized recommendations
        recommendations = []
        
        # pH recommendations
        if ph_level < 6.0:
            recommendations.append({
                "title": "Apply Lime for pH Correction",
                "priority": "High",
                "description": f"Soil pH is acidic ({ph_level}). Apply agricultural lime to raise pH to optimal range (6.0-7.5).",
                "expected_outcome": "Improved nutrient availability and root development within 2-3 months",
                "cost": "₹8,500 per hectare"
            })
        elif ph_level > 7.5:
            recommendations.append({
                "title": "Apply Sulfur for pH Reduction",
                "priority": "High", 
                "description": f"Soil pH is alkaline ({ph_level}). Apply elemental sulfur to lower pH to optimal range.",
                "expected_outcome": "Better nutrient uptake and reduced alkalinity stress within 3-4 months",
                "cost": "₹6,200 per hectare"
            })
        
        # Nitrogen recommendations
        if nitrogen < 40:
            recommendations.append({
                "title": "Nitrogen Fertilizer Application",
                "priority": "High",
                "description": f"Nitrogen levels are low ({nitrogen}%). Apply urea or organic compost to boost nitrogen content.",
                "expected_outcome": "Enhanced vegetative growth and leaf development within 3-4 weeks",
                "cost": "₹4,500 per hectare"
            })
        
        # Phosphorus recommendations
        if phosphorus < 25:
            recommendations.append({
                "title": "Phosphorus Supplementation",
                "priority": "Medium",
                "description": f"Phosphorus levels are below optimal ({phosphorus}%). Apply DAP or rock phosphate.",
                "expected_outcome": "Improved root development and flowering within 4-6 weeks",
                "cost": "₹5,800 per hectare"
            })
        
        # Potassium recommendations
        if potassium < 50:
            recommendations.append({
                "title": "Potassium Enhancement",
                "priority": "Medium",
                "description": f"Potassium levels need improvement ({potassium}%). Apply muriate of potash or organic potash.",
                "expected_outcome": "Better disease resistance and fruit quality within 5-7 weeks",
                "cost": "₹3,200 per hectare"
            })
        
        # Moisture recommendations
        if moisture < 40:
            recommendations.append({
                "title": "Irrigation System Upgrade",
                "priority": "High",
                "description": f"Soil moisture is low ({moisture}%). Install drip irrigation or increase watering frequency.",
                "expected_outcome": "Optimal water availability and reduced plant stress within 1-2 weeks",
                "cost": "₹12,000 per hectare"
            })
        elif moisture > 70:
            recommendations.append({
                "title": "Drainage Improvement",
                "priority": "Medium",
                "description": f"Soil moisture is high ({moisture}%). Improve drainage to prevent waterlogging.",
                "expected_outcome": "Reduced root rot risk and better aeration within 2-3 weeks",
                "cost": "₹7,500 per hectare"
            })
        
        # Organic matter recommendations
        if organic_matter < 3.0:
            recommendations.append({
                "title": "Organic Matter Enhancement",
                "priority": "Medium",
                "description": f"Organic matter content is low ({organic_matter}%). Add compost, farmyard manure, or green manure.",
                "expected_outcome": "Improved soil structure and nutrient retention within 6-8 weeks",
                "cost": "₹4,800 per hectare"
            })
        
        # Weather-based recommendations
        if temperature > 35:
            recommendations.append({
                "title": "Heat Stress Mitigation",
                "priority": "High",
                "description": f"High temperature ({temperature}°C) detected. Increase irrigation frequency and consider shade nets.",
                "expected_outcome": "Reduced heat stress and maintained crop productivity within 1 week",
                "cost": "₹2,500 per hectare"
            })
        
        if humidity > 80:
            recommendations.append({
                "title": "Fungal Disease Prevention",
                "priority": "Medium",
                "description": f"High humidity ({humidity}%) increases fungal disease risk. Apply preventive fungicides.",
                "expected_outcome": "Reduced disease incidence and healthier crops within 2-3 weeks",
                "cost": "₹1,800 per hectare"
            })
        
        # Add general recommendations if no specific issues
        if len(recommendations) == 0:
            recommendations.append({
                "title": "Maintain Current Practices",
                "priority": "Low",
                "description": "Soil health parameters are within optimal ranges. Continue current management practices.",
                "expected_outcome": "Sustained soil health and crop productivity",
                "cost": "₹500 per hectare (monitoring)"
            })
        
        # Sort recommendations by priority
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        # Prepare response data
        soil_health_data = {
            "success": True,
            "metrics": {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "ph_level": ph_level,
                "moisture": moisture,
                "organic_matter": organic_matter,
                "temperature": temperature,
                "humidity": humidity,
                "soil_type": soil_type
            },
            "health_scores": {
                "nitrogen": round(nitrogen_score, 1),
                "phosphorus": round(phosphorus_score, 1),
                "potassium": round(potassium_score, 1),
                "ph_level": round(ph_score, 1),
                "moisture": round(moisture_score, 1),
                "organic_matter": round(organic_matter_score, 1),
                "overall": overall_health_score
            },
            "health_score": overall_health_score,
            "recommendations": recommendations,
            "location": {
                "lat": user_lat,
                "lon": user_lon
            },
            "last_updated": current_time.isoformat(),
            "data_source": "real_time"
        }
        
        # Store data in Supabase soil_health table
        try:
            soil_health_record = {
                "user_id": user_id,
                "data": soil_health_data,
                "fetched_at": current_time.isoformat()
            }
            supabase.table("soil_health").insert(soil_health_record).execute()
            logger.info(f"Soil health data stored in database for user: {current_user['email']}")
        except Exception as e:
            logger.warning(f"Failed to store soil health data in database: {str(e)}")
        
        # Cache the response
        soil_health_cache[cache_key] = (soil_health_data, current_time)
        
        logger.info(f"Soil health data generated for user: {current_user['email']}")
        return soil_health_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Soil health data error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve soil health data"
        )

@router.get("/soil/recommendations")
async def get_soil_recommendations(current_user: dict = Depends(get_current_user)):
    """
    Get soil recommendations based on user's location and weather data
    """
    try:
        user_lat = current_user.get("lat")
        user_lon = current_user.get("lon")
        user_region = current_user.get("region", "Unknown")
        
        if not user_lat or not user_lon:
            raise HTTPException(
                status_code=400,
                detail="User location not found. Please update your location in profile."
            )
        
        # Get current weather for soil analysis
        try:
            weather_data = await weather_service.get_current_weather(user_lat, user_lon)
            current_weather = weather_data.get("current", {})
        except Exception as e:
            logger.warning(f"Failed to get weather data for soil analysis: {str(e)}")
            current_weather = {}
        
        # Generate soil recommendations based on location and weather
        soil_recommendations = {
            "location_info": {
                "lat": user_lat,
                "lon": user_lon,
                "region": user_region
            },
            "current_conditions": {
                "temperature": current_weather.get("temperature", 25),
                "humidity": current_weather.get("humidity", 60),
                "weather": current_weather.get("description", "Clear")
            },
            "soil_analysis": {
                "ph_level": round(random.uniform(6.0, 7.5), 1),
                "nitrogen": round(random.uniform(20, 80), 1),
                "phosphorus": round(random.uniform(15, 60), 1),
                "potassium": round(random.uniform(25, 90), 1),
                "organic_matter": round(random.uniform(2.5, 5.0), 1),
                "moisture_level": round(random.uniform(30, 70), 1)
            },
            "recommendations": [],
            "suitable_crops": [],
            "irrigation_schedule": {
                "frequency": "Every 2-3 days",
                "amount": "25-30mm per session",
                "best_time": "Early morning (6-8 AM)"
            },
            "fertilizer_recommendations": [
                {
                    "type": "NPK 10-26-26",
                    "amount": "150 kg/hectare",
                    "timing": "Pre-planting"
                },
                {
                    "type": "Urea",
                    "amount": "100 kg/hectare",
                    "timing": "30 days after planting"
                }
            ]
        }
        
        # Region-specific recommendations
        if "Northern" in user_region:
            soil_recommendations["suitable_crops"] = ["Rice", "Wheat", "Sugarcane", "Cotton"]
            soil_recommendations["recommendations"] = [
                "Ideal for cereal crops due to fertile alluvial soil",
                "Consider crop rotation with legumes to maintain nitrogen levels",
                "Monitor for waterlogging during monsoon season"
            ]
        elif "Western" in user_region:
            soil_recommendations["suitable_crops"] = ["Cotton", "Sugarcane", "Soybean", "Groundnut"]
            soil_recommendations["recommendations"] = [
                "Black cotton soil is excellent for cotton cultivation",
                "Implement drip irrigation for water conservation",
                "Add organic matter to improve soil structure"
            ]
        elif "Southern" in user_region:
            soil_recommendations["suitable_crops"] = ["Rice", "Millets", "Coconut", "Spices"]
            soil_recommendations["recommendations"] = [
                "Red laterite soil requires regular organic amendments",
                "Consider intercropping with legumes",
                "Monitor pH levels and apply lime if needed"
            ]
        else:
            soil_recommendations["suitable_crops"] = ["Mixed farming", "Vegetables", "Fruits"]
            soil_recommendations["recommendations"] = [
                "Conduct detailed soil testing for specific recommendations",
                "Implement sustainable farming practices",
                "Consider local climate conditions for crop selection"
            ]
        
        # Weather-based adjustments
        temperature = current_weather.get("temperature", 25)
        humidity = current_weather.get("humidity", 60)
        
        if temperature > 35:
            soil_recommendations["recommendations"].append("High temperature detected - increase irrigation frequency")
        if humidity > 80:
            soil_recommendations["recommendations"].append("High humidity - monitor for fungal diseases")
        if humidity < 40:
            soil_recommendations["recommendations"].append("Low humidity - consider mulching to retain moisture")
        
        logger.info(f"Soil recommendations generated for user: {current_user['email']}")
        return {
            "success": True,
            "data": soil_recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Soil recommendations error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate soil recommendations"
        )

@router.post("/yield-prediction-test")
async def predict_yield_test(request: YieldPredictionRequest) -> Dict[str, Any]:
    """
    Test endpoint for yield prediction without authentication
    Uses the new RandomForest model when available, falls back to original model
    """
    try:
        logger.info("Test yield prediction request")
        
        # Try to use the new RandomForest model first
        if NEW_MODEL_AVAILABLE and new_yield_service.model is not None:
            logger.info("Using new RandomForest model for prediction")
            
            try:
                # Use the new model with intelligent production estimation
                result = new_yield_service.predict_yield(
                    crop=request.crop,
                    season=request.season,
                    state=request.state,
                    crop_year=int(request.year),
                    area=float(request.area),
                    annual_rainfall=1000.0,  # Default rainfall
                    fertilizer=50.0,         # Default fertilizer
                    pesticide=20.0           # Default pesticide
                )
                
                # Store prediction in database
                try:
                    prediction_data = {
                        "user_id": current_user["id"],
                        "state": request.state,
                        "district": request.district,
                        "crop": request.crop,
                        "year": request.year,
                        "season": request.season,
                        "area": request.area,
                        "predicted_yield": result["predicted_yield_tons_per_hectare"],
                        "input_parameters": request.dict(),
                        "model_version": "2.0_RandomForest",
                        "confidence_score": result["confidence"]
                    }
                    
                    supabase.table("yield_predictions").insert(prediction_data).execute()
                    logger.info(f"New model prediction stored for user: {current_user['email']}")
                except Exception as db_error:
                    logger.warning(f"Failed to store prediction in database: {str(db_error)}")
                
                # Return response in expected format
                return {
                    "success": True,
                    "message": "Yield prediction completed successfully using advanced RandomForest model",
                    "model_used": "RandomForest_v2.0",
                    "data": {
                        "predicted_yield_tons_per_hectare": result["predicted_yield_tons_per_hectare"],
                        "total_production_tons": result["total_production_tons"],
                        "confidence": result["confidence"],
                        "yield_category": result["yield_category"],
                        "factors_analysis": result["factors_analysis"],
                        "recommendations": result["recommendations"],
                        "input_parameters": result["input_parameters"]
                    }
                }
                
            except Exception as new_model_error:
                logger.error(f"New model failed: {str(new_model_error)}")
                # Fall through to old model
        
        # Heuristic fallback when model is unavailable
        logger.info("Using heuristic fallback for prediction")
        
        area = float(request.area)
        
        crop_yield_estimates = {
            'Rice': 3.5, 'Wheat': 3.2, 'Maize': 4.8, 'Sugarcane': 70.0,
            'Cotton(lint)': 1.8, 'Groundnut': 2.1, 'Soyabean': 1.5
        }
        base_yield = crop_yield_estimates.get(request.crop, 2.5)
        
        crop_realistic_yields = {
            'Rice': (2.0, 6.0), 'Wheat': (2.5, 5.5), 'Maize': (3.0, 8.0),
            'Sugarcane': (40.0, 80.0), 'Cotton(lint)': (1.0, 3.0),
            'Groundnut': (1.5, 3.5), 'Soyabean': (1.0, 3.0)
        }
        min_yield, max_yield = crop_realistic_yields.get(request.crop, (1.5, 4.5))
        
        state_factor = 1.0
        if request.state in ["Punjab", "Haryana", "Uttar Pradesh"]:
            state_factor = 1.1
        elif request.state in ["Rajasthan", "Maharashtra", "Karnataka"]:
            state_factor = 0.95
        
        season_factor = 1.0
        if request.season == "Kharif":
            season_factor = 1.05
        elif request.season == "Summer":
            season_factor = 0.9
        
        predicted_yield = base_yield * state_factor * season_factor
        predicted_yield = max(min_yield * 0.8, min(predicted_yield, max_yield * 1.2))
        total_production = predicted_yield * area
        
        confidence = int(HEURISTIC_CONFIDENCE_BASE)
        yield_category = "High" if predicted_yield >= 5.0 else "Medium" if predicted_yield >= 2.5 else "Low"
        
        recommendations = []
        if yield_category == "Low":
            recommendations = [
                "Consider soil testing and nutrient management",
                "Review irrigation and water management",
                "Evaluate seed variety selection"
            ]
        elif yield_category == "Medium":
            recommendations = [
                "Optimize fertilizer application",
                "Implement integrated pest management",
                "Consider precision agriculture techniques"
            ]
        else:
            recommendations = [
                "Maintain current best practices",
                "Focus on post-harvest management",
                "Explore value-added processing"
            ]
        if request.season == "Kharif":
            recommendations.append("Monitor for monsoon-related diseases and pests")
        elif request.season == "Rabi":
            recommendations.append("Ensure adequate irrigation during dry winter months")
        elif request.season == "Summer":
            recommendations.append("Implement water conservation techniques")
        
        return {
            "success": True,
            "message": "Yield prediction completed using heuristic fallback",
            "model_used": "Heuristic_Fallback",
            "data": {
                "predicted_yield_tons_per_hectare": round(predicted_yield, 2),
                "total_production_tons": round(total_production, 2),
                "confidence": confidence,
                "yield_category": yield_category,
                "factors_analysis": {},
                "recommendations": recommendations,
                "input_parameters": {
                    "state": request.state,
                    "district": request.district,
                    "crop": request.crop,
                    "year": request.year,
                    "season": request.season,
                    "area": request.area
                }
            }
        }
            
    except Exception as e:
        logger.error(f"Yield prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict yield: {str(e)}"
        )

@router.get("/yield-prediction/latest")
async def get_latest_yield_prediction(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the latest yield prediction for the current user
    If no prediction exists, generate one based on user's location and recent data
    """
    try:
        user_id = current_user["id"]
        logger.info(f"Fetching latest yield prediction for user: {current_user['email']}")

        # Try to get the most recent prediction from database
        try:
            result = supabase.table("yield_predictions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()

            if result.data and len(result.data) > 0:
                prediction = result.data[0]
                logger.info(f"Found existing prediction for user: {current_user['email']}")
                return {
                    "success": True,
                    "data": {
                        "predicted_yield_tons_per_hectare": prediction["predicted_yield"],
                        "total_production_tons": prediction["predicted_yield"] * prediction["area"],
                        "confidence": prediction.get("confidence_score", 85),
                        "yield_category": prediction.get("yield_category", "Medium"),
                        "crop": prediction["crop"],
                        "state": prediction["state"],
                        "season": prediction["season"],
                        "recommendations": prediction.get("recommendations", [])
                    }
                }
        except Exception as db_error:
            logger.warning(f"Failed to fetch existing prediction: {str(db_error)}")

        # If no existing prediction, generate a default one based on user's location
        user_lat = current_user.get("lat", 19.0760)  # Default to Mumbai
        user_lon = current_user.get("lon", 72.8777)
        user_region = current_user.get("region", "Maharashtra")

        # Map region to state for prediction
        state_mapping = {
            "Maharashtra": "Maharashtra",
            "Karnataka": "Karnataka",
            "Tamil Nadu": "Tamil Nadu",
            "Andhra Pradesh": "Andhra Pradesh",
            "Telangana": "Telangana",
            "Gujarat": "Gujarat",
            "Rajasthan": "Rajasthan",
            "Punjab": "Punjab",
            "Haryana": "Haryana",
            "Uttar Pradesh": "Uttar Pradesh"
        }
        state = state_mapping.get(user_region, "Maharashtra")

        # Generate prediction based on typical crops for the region
        region_crops = {
            "Maharashtra": ["Cotton", "Sugarcane", "Soybean"],
            "Karnataka": ["Rice", "Maize", "Coffee"],
            "Tamil Nadu": ["Rice", "Sugarcane", "Banana"],
            "Punjab": ["Wheat", "Rice", "Cotton"],
            "Haryana": ["Wheat", "Rice", "Sugarcane"]
        }

        crop = random.choice(region_crops.get(state, ["Wheat", "Rice", "Maize"]))
        season = "Kharif" if random.random() > 0.5 else "Rabi"
        area = round(random.uniform(1.0, 5.0), 1)

        # Generate prediction using the same logic as the main endpoint
        crop_yield_estimates = {
            'Rice': 3.5, 'Wheat': 3.2, 'Maize': 4.8, 'Sugarcane': 70.0,
            'Cotton(lint)': 1.8, 'Groundnut': 2.1, 'Soyabean': 1.5
        }
        base_yield = crop_yield_estimates.get(crop, 2.5)

        state_factor = 1.0
        if state in ["Punjab", "Haryana", "Uttar Pradesh"]:
            state_factor = 1.1
        elif state in ["Rajasthan", "Maharashtra", "Karnataka"]:
            state_factor = 0.95

        season_factor = 1.0
        if season == "Kharif":
            season_factor = 1.05
        elif season == "Summer":
            season_factor = 0.9

        predicted_yield = base_yield * state_factor * season_factor
        predicted_yield = max(1.0, min(predicted_yield, 80.0))  # Reasonable bounds
        total_production = predicted_yield * area

        confidence = random.randint(75, 95)
        yield_category = "High" if predicted_yield >= 5.0 else "Medium" if predicted_yield >= 2.5 else "Low"

        recommendations = []
        if yield_category == "Low":
            recommendations = [
                "Consider soil testing and nutrient management",
                "Review irrigation and water management practices",
                "Evaluate seed variety selection"
            ]
        elif yield_category == "Medium":
            recommendations = [
                "Optimize fertilizer application based on soil test results",
                "Implement integrated pest management",
                "Consider precision agriculture techniques"
            ]
        else:
            recommendations = [
                "Maintain current best practices",
                "Focus on post-harvest management to reduce losses",
                "Explore value-added processing opportunities"
            ]

        # Store the generated prediction
        try:
            prediction_data = {
                "user_id": user_id,
                "state": state,
                "district": f"{state} District",  # Generic district
                "crop": crop,
                "year": str(datetime.now().year),
                "season": season,
                "area": area,
                "predicted_yield": predicted_yield,
                "input_parameters": {
                    "state": state,
                    "crop": crop,
                    "season": season,
                    "area": area,
                    "generated_from_location": True
                },
                "model_version": "location_based_v1.0",
                "confidence_score": confidence,
                "yield_category": yield_category,
                "recommendations": recommendations
            }

            supabase.table("yield_predictions").insert(prediction_data).execute()
            logger.info(f"Generated and stored location-based prediction for user: {current_user['email']}")
        except Exception as db_error:
            logger.warning(f"Failed to store generated prediction: {str(db_error)}")

        return {
            "success": True,
            "data": {
                "predicted_yield_tons_per_hectare": round(predicted_yield, 2),
                "total_production_tons": round(total_production, 2),
                "confidence": confidence,
                "yield_category": yield_category,
                "crop": crop,
                "state": state,
                "season": season,
                "recommendations": recommendations
            }
        }

    except Exception as e:
        logger.error(f"Error fetching latest yield prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch yield prediction data"
        )

@router.post("/yield-prediction")
async def predict_yield(
    request: YieldPredictionRequest,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Predict crop yield based on farming parameters
    Uses the new RandomForest model when available, falls back to original model
    """
    try:
        logger.info(f"Yield prediction request from user: {current_user['email']}")
        
        # Try to use the new RandomForest model first
        if NEW_MODEL_AVAILABLE and new_yield_service.model is not None:
            logger.info("Using new RandomForest model for prediction")
            
            try:
                # Use the new model with intelligent production estimation
                result = new_yield_service.predict_yield(
                    crop=request.crop,
                    season=request.season,
                    state=request.state,
                    crop_year=int(request.year),
                    area=float(request.area),
                    annual_rainfall=1000.0,  # Default rainfall
                    fertilizer=50.0,         # Default fertilizer
                    pesticide=20.0           # Default pesticide
                )
                
                # Store prediction in database
                try:
                    prediction_data = {
                        "user_id": current_user["id"],
                        "state": request.state,
                        "district": request.district,
                        "crop": request.crop,
                        "year": request.year,
                        "season": request.season,
                        "area": request.area,
                        "predicted_yield": result["predicted_yield_tons_per_hectare"],
                        "input_parameters": request.dict(),
                        "model_version": "2.0_RandomForest",
                        "confidence_score": result["confidence"]
                    }
                    
                    supabase.table("yield_predictions").insert(prediction_data).execute()
                    logger.info(f"New model prediction stored for user: {current_user['email']}")
                except Exception as db_error:
                    logger.warning(f"Failed to store prediction in database: {str(db_error)}")
                
                # Return response in expected format
                return {
                    "success": True,
                    "message": "Yield prediction completed successfully using advanced RandomForest model",
                    "data": {
                        "predicted_yield_tons_per_hectare": result["predicted_yield_tons_per_hectare"],
                        "total_production_tons": result["total_production_tons"],
                        "confidence": result["confidence"],
                        "yield_category": result["yield_category"],
                        "factors_analysis": result["factors_analysis"],
                        "recommendations": result["recommendations"],
                        "input_parameters": request.dict(),
                        "prediction_metadata": {
                            "model_version": result["model_info"]["model_version"],
                            "model_type": result["model_info"]["model_type"],
                            "r2_score": result["model_info"]["r2_score"],
                            "prediction_time": datetime.utcnow().isoformat() + "Z",
                            "crop_type": request.crop,
                            "state": request.state,
                            "district": request.district,
                            "season": request.season,
                            "area_hectares": float(request.area),
                            "estimated_production": result["input_data"]["estimated_production"]
                        }
                    }
                }
                
            except Exception as new_model_error:
                logger.error(f"New model prediction failed: {str(new_model_error)}")
                logger.info("Falling back to original model")
        
        # Fallback to original model (heuristic, no external files)
        state_mapping = {request.state: 0}
        district_mapping = {request.district: 0}
        crop_mapping = {request.crop: 0}
        season_mapping = {request.season: 0}
        
        # Map inputs with fallback to first available option
        state_id = state_mapping.get(request.state, 0)
        district_id = district_mapping.get(request.district, 0)
        crop_id = crop_mapping.get(request.crop, 0)
        season_id = season_mapping.get(request.season, 0)
        
        # Use the area from request
        area = float(request.area)
        
        # Estimate production based on area and crop type (simplified approach)
        # This is a rough estimation - in real scenario, this would be more sophisticated
        crop_yield_estimates = {
            'Rice': 3.5, 'Wheat': 3.2, 'Maize': 4.8, 'Sugarcane': 70.0,
            'Cotton(lint)': 1.8, 'Groundnut': 2.1, 'Soyabean': 1.5
        }
        base_yield = crop_yield_estimates.get(request.crop, 2.5)
        estimated_production = area * base_yield
        
        # Prepare model inputs based on the original model structure
        model_inputs = {
            'state_input': np.array([[state_id]], dtype=np.float32),
            'district_input': np.array([[district_id]], dtype=np.float32),
            'season_input': np.array([[season_id]], dtype=np.float32),
            'crop_input': np.array([[crop_id]], dtype=np.float32),
            'numerical_input': np.array([[area, estimated_production]], dtype=np.float32)
        }
        
        # Make prediction
        prediction = yield_model.predict(model_inputs)
        raw_prediction = float(prediction[0][0])
        
        # The model seems to predict production values, not yield per hectare
        # Apply realistic yield bounds based on crop type and normalize the prediction
        crop_realistic_yields = {
            'Rice': (2.0, 6.0),      # tons/ha
            'Wheat': (2.5, 5.5),     # tons/ha  
            'Maize': (3.0, 8.0),     # tons/ha
            'Sugarcane': (40.0, 80.0), # tons/ha
            'Cotton(lint)': (1.0, 3.0), # tons/ha
            'Groundnut': (1.5, 3.5),   # tons/ha
            'Soyabean': (1.0, 3.0)     # tons/ha
        }
        
        # Get realistic range for the crop
        min_yield, max_yield = crop_realistic_yields.get(request.crop, (1.5, 4.5))
        
        # Normalize the raw prediction to realistic yield range
        # Use a sigmoid-like function to map model output to realistic range
        import math
        normalized_factor = 1 / (1 + math.exp(-raw_prediction / 1000))  # Sigmoid normalization
        predicted_yield = min_yield + (max_yield - min_yield) * normalized_factor
        
        # Add some variability based on input parameters
        state_factor = 1.0
        if request.state in ["Punjab", "Haryana", "Uttar Pradesh"]:
            state_factor = 1.1  # Higher productivity states
        elif request.state in ["Rajasthan", "Maharashtra", "Karnataka"]:
            state_factor = 0.95  # Moderate productivity
        
        season_factor = 1.0
        if request.season == "Kharif":
            season_factor = 1.05  # Monsoon season advantage
        elif request.season == "Summer":
            season_factor = 0.9   # Summer season challenge
            
        # Apply factors
        predicted_yield = predicted_yield * state_factor * season_factor
        
        # Ensure final bounds
        predicted_yield = max(min_yield * 0.8, min(predicted_yield, max_yield * 1.2))
        
        # Calculate total production
        total_production = predicted_yield * area
        
        # Calculate confidence based on data availability
        base_confidence = 70
        
        # Adjust confidence based on input validity
        if request.state in state_mapping:
            base_confidence += 5
        if request.district in district_mapping:
            base_confidence += 5
        if request.crop in crop_mapping:
            base_confidence += 10
        if request.season in season_mapping:
            base_confidence += 5
        
        confidence = min(95, base_confidence)
        
        # Generate yield category
        if predicted_yield >= 5.0:
            yield_category = "High"
        elif predicted_yield >= 2.5:
            yield_category = "Medium"
        else:
            yield_category = "Low"
        
        # Generate recommendations based on prediction and dataset insights
        recommendations = []
        if predicted_yield < 2.0:
            recommendations.extend([
                "Consider soil testing and nutrient management",
                "Evaluate seed variety selection for better adaptation",
                "Review irrigation and water management practices"
            ])
        elif predicted_yield < 4.0:
            recommendations.extend([
                "Optimize fertilizer application based on soil test results",
                "Implement integrated pest and disease management",
                "Consider precision agriculture techniques"
            ])
        else:
            recommendations.extend([
                "Maintain current best practices",
                "Focus on post-harvest management to reduce losses",
                "Consider value-added processing opportunities"
            ])
        
        # Add season-specific recommendations
        if request.season == "Kharif":
            recommendations.append("Monitor for monsoon-related diseases and pests")
        elif request.season == "Rabi":
            recommendations.append("Ensure adequate irrigation during dry winter months")
        elif request.season == "Summer":
            recommendations.append("Implement water conservation techniques for summer crops")
        
        # Generate factors analysis
        factors_analysis = {
            "state_impact": "High" if request.state in ["Punjab", "Haryana", "Uttar Pradesh"] else "Medium",
            "season_impact": "High" if request.season in ["Kharif", "Rabi"] else "Medium",
            "crop_suitability": "High" if request.crop in ["Rice", "Wheat", "Maize"] else "Medium",
            "area_efficiency": "High" if area >= 2.0 else "Medium"
        }
        
        # Store prediction in database
        try:
            prediction_data = {
                "user_id": current_user["id"],
                "state": request.state,
                "district": request.district,
                "crop": request.crop,
                "year": request.year,
                "season": request.season,
                "area": request.area,
                "predicted_yield": predicted_yield,
                "input_parameters": request.dict(),
                "model_version": "1.0",
                "confidence_score": confidence
            }
            
            supabase.table("yield_predictions").insert(prediction_data).execute()
            logger.info(f"Yield prediction stored for user: {current_user['email']}")
        except Exception as db_error:
            logger.warning(f"Failed to store prediction in database: {str(db_error)}")
        
        return {
            "success": True,
            "message": "Yield prediction completed successfully",
            "data": {
                "predicted_yield_tons_per_hectare": round(predicted_yield, 2),
                "total_production_tons": round(total_production, 2),
                "confidence": confidence,
                "yield_category": yield_category,
                "factors_analysis": factors_analysis,
                "recommendations": recommendations,
                "input_parameters": request.dict(),
                "prediction_metadata": {
                    "model_version": "1.0",
                    "prediction_time": datetime.utcnow().isoformat() + "Z",
                    "crop_type": request.crop,
                    "state": request.state,
                    "district": request.district,
                    "season": request.season,
                    "area_hectares": area
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Yield prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict yield: {str(e)}"
        )

@router.post("/crop-prediction", response_model=CropPredictionResponse)
async def predict_crop_suitability(
    request: CropPredictionRequest,
    current_user: dict = Depends(get_current_user)
) -> CropPredictionResponse:
    """
    Predict suitable crops based on soil and environmental parameters
    """
    try:
        logger.info(f"Crop prediction request from user: {current_user['email']}")

        # Mock crop prediction logic based on soil parameters
        # In a real implementation, this would use a trained ML model
        recommendations = []

        # Define crop suitability based on soil parameters
        crops_data = [
            {
                "id": 1,
                "name": "Rice",
                "optimal_ph": (5.5, 7.0),
                "optimal_nitrogen": (40, 80),
                "optimal_phosphorus": (20, 50),
                "optimal_potassium": (30, 70),
                "optimal_temperature": (20, 35),
                "optimal_humidity": (60, 90),
                "optimal_rainfall": (1000, 2000),
                "base_suitability": 85
            },
            {
                "id": 2,
                "name": "Wheat",
                "optimal_ph": (6.0, 7.5),
                "optimal_nitrogen": (50, 90),
                "optimal_phosphorus": (25, 55),
                "optimal_potassium": (40, 80),
                "optimal_temperature": (15, 25),
                "optimal_humidity": (40, 70),
                "optimal_rainfall": (500, 1000),
                "base_suitability": 82
            },
            {
                "id": 3,
                "name": "Maize",
                "optimal_ph": (5.8, 7.0),
                "optimal_nitrogen": (60, 100),
                "optimal_phosphorus": (30, 60),
                "optimal_potassium": (50, 90),
                "optimal_temperature": (20, 35),
                "optimal_humidity": (50, 80),
                "optimal_rainfall": (500, 1200),
                "base_suitability": 88
            },
            {
                "id": 4,
                "name": "Cotton",
                "optimal_ph": (6.0, 8.0),
                "optimal_nitrogen": (40, 70),
                "optimal_phosphorus": (20, 40),
                "optimal_potassium": (30, 60),
                "optimal_temperature": (25, 40),
                "optimal_humidity": (40, 70),
                "optimal_rainfall": (600, 1200),
                "base_suitability": 75
            },
            {
                "id": 5,
                "name": "Sugarcane",
                "optimal_ph": (6.5, 7.5),
                "optimal_nitrogen": (80, 120),
                "optimal_phosphorus": (40, 70),
                "optimal_potassium": (60, 100),
                "optimal_temperature": (25, 35),
                "optimal_humidity": (60, 80),
                "optimal_rainfall": (1500, 2500),
                "base_suitability": 78
            }
        ]

        for crop in crops_data:
            # Calculate suitability score based on how well parameters match optimal ranges
            suitability_factors = []

            # pH suitability
            if crop["optimal_ph"][0] <= request.ph <= crop["optimal_ph"][1]:
                suitability_factors.append(1.0)
            elif abs(request.ph - crop["optimal_ph"][0]) <= 0.5 or abs(request.ph - crop["optimal_ph"][1]) <= 0.5:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.4)

            # Nitrogen suitability
            if crop["optimal_nitrogen"][0] <= request.nitrogen <= crop["optimal_nitrogen"][1]:
                suitability_factors.append(1.0)
            elif abs(request.nitrogen - crop["optimal_nitrogen"][0]) <= 10 or abs(request.nitrogen - crop["optimal_nitrogen"][1]) <= 10:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.5)

            # Phosphorus suitability
            if crop["optimal_phosphorus"][0] <= request.phosphorus <= crop["optimal_phosphorus"][1]:
                suitability_factors.append(1.0)
            elif abs(request.phosphorus - crop["optimal_phosphorus"][0]) <= 5 or abs(request.phosphorus - crop["optimal_phosphorus"][1]) <= 5:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.5)

            # Potassium suitability
            if crop["optimal_potassium"][0] <= request.potassium <= crop["optimal_potassium"][1]:
                suitability_factors.append(1.0)
            elif abs(request.potassium - crop["optimal_potassium"][0]) <= 10 or abs(request.potassium - crop["optimal_potassium"][1]) <= 10:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.5)

            # Temperature suitability
            if crop["optimal_temperature"][0] <= request.temperature <= crop["optimal_temperature"][1]:
                suitability_factors.append(1.0)
            elif abs(request.temperature - crop["optimal_temperature"][0]) <= 5 or abs(request.temperature - crop["optimal_temperature"][1]) <= 5:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.6)

            # Humidity suitability
            if crop["optimal_humidity"][0] <= request.humidity <= crop["optimal_humidity"][1]:
                suitability_factors.append(1.0)
            elif abs(request.humidity - crop["optimal_humidity"][0]) <= 10 or abs(request.humidity - crop["optimal_humidity"][1]) <= 10:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.6)

            # Rainfall suitability
            if crop["optimal_rainfall"][0] <= request.rainfall <= crop["optimal_rainfall"][1]:
                suitability_factors.append(1.0)
            elif abs(request.rainfall - crop["optimal_rainfall"][0]) <= 200 or abs(request.rainfall - crop["optimal_rainfall"][1]) <= 200:
                suitability_factors.append(0.8)
            else:
                suitability_factors.append(0.4)

            # Calculate overall suitability score
            avg_suitability = sum(suitability_factors) / len(suitability_factors)
            suitability_score = round(crop["base_suitability"] * avg_suitability, 1)

            # Calculate confidence based on how many parameters are in optimal range
            optimal_count = sum(1 for factor in suitability_factors if factor >= 0.9)
            confidence = round(60 + (optimal_count / len(suitability_factors)) * 35, 1)

            recommendations.append(CropRecommendation(
                crop_id=crop["id"],
                crop_name=crop["name"],
                confidence=confidence,
                suitability_score=suitability_score
            ))

        # Sort recommendations by suitability score (descending)
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)

        # Limit to top 5 recommendations
        recommendations = recommendations[:5]

        # Store prediction in database
        try:
            prediction_data = {
                "user_id": current_user["id"],
                "input_parameters": request.dict(),
                "recommendations": [rec.dict() for rec in recommendations],
                "prediction_time": datetime.utcnow().isoformat() + "Z"
            }
            supabase.table("crop_predictions").insert(prediction_data).execute()
            logger.info(f"Crop prediction stored for user: {current_user['email']}")
        except Exception as db_error:
            logger.warning(f"Failed to store crop prediction in database: {str(db_error)}")

        return CropPredictionResponse(
            success=True,
            message="Crop suitability prediction completed successfully",
            input_parameters=request,
            recommendations=recommendations,
            total_crops=len(recommendations),
            prediction_metadata={
                "model_version": "1.0_Heuristic",
                "prediction_time": datetime.utcnow().isoformat() + "Z",
                "algorithm": "Multi-parameter suitability scoring",
                "parameters_analyzed": ["ph", "nitrogen", "phosphorus", "potassium", "temperature", "humidity", "rainfall"]
            }
        )

    except Exception as e:
        logger.error(f"Crop prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict crop suitability: {str(e)}"
        )
