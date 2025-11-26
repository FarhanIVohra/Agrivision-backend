"""
Weather Router
API endpoints for weather data retrieval
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging
import requests
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Weather API configuration
OPENWEATHER_API_KEY = os.getenv("REACT_APP_WEATHER_API_KEY", "4d16be1669a3625caa757c4df10d2f60")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@router.get("/current")
async def get_current_weather(
    location: Optional[str] = Query(None, description="Location for weather data (e.g., 'Nadiad, IN')"),
    lat: Optional[float] = Query(None, description="Latitude coordinate"),
    lon: Optional[float] = Query(None, description="Longitude coordinate")
):
    """
    Get current weather data for a specific location or coordinates
    
    Args:
        location: Location string (e.g., "Nadiad, IN") - optional if lat/lon provided
        lat: Latitude coordinate - optional if location provided
        lon: Longitude coordinate - optional if location provided
        
    Returns:
        Current weather data including temperature, humidity, condition, wind speed, and rainfall
    """
    try:
        # Validate input parameters
        if not location and (lat is None or lon is None):
            raise HTTPException(
                status_code=400, 
                detail="Either 'location' or both 'lat' and 'lon' parameters must be provided"
            )
        
        # Prepare API parameters based on input
        if lat is not None and lon is not None:
            logger.info(f"Fetching weather data for coordinates: {lat}, {lon}")
            params = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }
            location_info = f"{lat}, {lon}"
        else:
            logger.info(f"Fetching weather data for location: {location}")
            params = {
                "q": location,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }
            location_info = location
        
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            weather_data = response.json()
            
            # Extract relevant data including rainfall
            current_weather = {
                "location": location_info,
                "current": {
                    "temperature": round(weather_data["main"]["temp"], 1),
                    "humidity": weather_data["main"]["humidity"],
                    "condition": weather_data["weather"][0]["description"].title(),
                    "wind_speed": round(weather_data["wind"]["speed"] * 3.6, 1),  # Convert m/s to km/h
                    "pressure": weather_data["main"]["pressure"],
                    "feels_like": round(weather_data["main"]["feels_like"], 1),
                    "visibility": weather_data.get("visibility", 10000) / 1000  # Convert to km
                },
                # Add rainfall data from the API response
                "rainfall": weather_data.get("rain", {}).get("1h", 0),  # mm in last hour
                "rain": weather_data.get("rain", {}),  # Full rain object
                "weather": weather_data.get("weather", []),  # Weather conditions array
                "timestamp": datetime.now().isoformat(),
                "source": "OpenWeatherMap"
            }
            
            logger.info(f"Successfully fetched weather data: {current_weather}")
            return current_weather
            
        else:
            logger.error(f"OpenWeatherMap API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch weather data from external API")
            
    except requests.exceptions.Timeout:
        logger.error("Weather API request timed out")
        # Return fallback data
        return get_fallback_weather_data(location_info)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {str(e)}")
        # Return fallback data
        return get_fallback_weather_data(location_info)
        
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {str(e)}")
        # Return fallback data
        return get_fallback_weather_data(location_info)

def get_fallback_weather_data(location: str) -> Dict[str, Any]:
    """
    Return fallback weather data when external API is unavailable
    
    Args:
        location: Location string
        
    Returns:
        Mock weather data with rainfall information
    """
    return {
        "location": location,
        "current": {
            "temperature": 28.0,
            "humidity": 75,
            "condition": "Partly Cloudy",
            "wind_speed": 8.0,
            "pressure": 1013,
            "feels_like": 30.0,
            "visibility": 10.0
        },
        # Add fallback rainfall data
        "rainfall": 1.2,  # mm in last hour
        "rain": {"1h": 1.2},  # Full rain object
        "weather": [{"description": "partly cloudy", "main": "Clouds"}],
        "timestamp": datetime.now().isoformat(),
        "source": "Fallback Data"
    }

@router.get("/health")
async def weather_service_health():
    """
    Check weather service health
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "weather",
        "api_key_configured": bool(OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "your_api_key_here"),
        "timestamp": datetime.now().isoformat()
    }