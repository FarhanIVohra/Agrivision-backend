"""
Weather Service for AgriSmart
Handles OpenWeatherMap API integration with caching and error handling
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import httpx
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_duration = timedelta(minutes=10)  # Cache for 10 minutes
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not found in environment variables")
    
    def _get_cache_key(self, lat: float, lon: float, endpoint: str) -> str:
        """Generate cache key for weather data"""
        return f"{endpoint}_{lat}_{lon}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry.get("timestamp", ""))
        return datetime.now() - cached_time < self.cache_duration
    
    async def get_current_weather(self, lat: float, lon: float) -> Dict:
        """
        Get current weather data for given coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict containing weather data
            
        Raises:
            HTTPException: If API request fails
        """
        if not self.api_key:
            raise HTTPException(
                status_code=500, 
                detail="Weather service not configured. Please add OPENWEATHERMAP_API_KEY to environment variables."
            )
        
        cache_key = self._get_cache_key(lat, lon, "current")
        
        # Check cache first
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info(f"Returning cached weather data for {lat}, {lon}")
            return self.cache[cache_key]["data"]
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/weather"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric"
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                
                if response.status_code == 429:
                    raise HTTPException(
                        status_code=429,
                        detail="Weather API rate limit exceeded. Please try again later."
                    )
                
                response.raise_for_status()
                weather_data = response.json()
                
                # Process and format the weather data
                formatted_data = {
                    "location": {
                        "name": weather_data.get("name", "Unknown"),
                        "country": weather_data.get("sys", {}).get("country", ""),
                        "lat": lat,
                        "lon": lon
                    },
                    "current": {
                        "temperature": weather_data["main"]["temp"],
                        "feels_like": weather_data["main"]["feels_like"],
                        "humidity": weather_data["main"]["humidity"],
                        "pressure": weather_data["main"]["pressure"],
                        "description": weather_data["weather"][0]["description"],
                        "main": weather_data["weather"][0]["main"],
                        "icon": weather_data["weather"][0]["icon"],
                        "wind_speed": weather_data.get("wind", {}).get("speed", 0),
                        "wind_direction": weather_data.get("wind", {}).get("deg", 0),
                        "visibility": weather_data.get("visibility", 0) / 1000,  # Convert to km
                        "uv_index": 0  # Current weather doesn't include UV, would need separate call
                    },
                    "timestamp": datetime.now().isoformat(),
                    "sunrise": datetime.fromtimestamp(weather_data["sys"]["sunrise"]).isoformat(),
                    "sunset": datetime.fromtimestamp(weather_data["sys"]["sunset"]).isoformat()
                }
                
                # Cache the result
                self.cache[cache_key] = {
                    "data": formatted_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Successfully fetched weather data for {lat}, {lon}")
                return formatted_data
                
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=408,
                detail="Weather service timeout. Please try again."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid weather API key. Please check your configuration."
                )
            elif e.response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail="Location not found. Please check coordinates."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Weather service error: {e.response.status_code}"
                )
        except Exception as e:
            logger.error(f"Unexpected error in weather service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Internal weather service error"
            )
    
    async def get_weather_forecast(self, lat: float, lon: float, days: int = 5) -> Dict:
        """
        Get weather forecast for given coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days for forecast (max 5 for free tier)
            
        Returns:
            Dict containing forecast data
        """
        if not self.api_key:
            raise HTTPException(
                status_code=500,
                detail="Weather service not configured"
            )
        
        cache_key = self._get_cache_key(lat, lon, f"forecast_{days}")
        
        # Check cache first
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info(f"Returning cached forecast data for {lat}, {lon}")
            return self.cache[cache_key]["data"]
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/forecast"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": days * 8  # 8 forecasts per day (every 3 hours)
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                forecast_data = response.json()
                
                # Process forecast data
                formatted_forecast = {
                    "location": {
                        "name": forecast_data["city"]["name"],
                        "country": forecast_data["city"]["country"],
                        "lat": lat,
                        "lon": lon
                    },
                    "forecast": []
                }
                
                for item in forecast_data["list"]:
                    formatted_forecast["forecast"].append({
                        "datetime": datetime.fromtimestamp(item["dt"]).isoformat(),
                        "temperature": item["main"]["temp"],
                        "feels_like": item["main"]["feels_like"],
                        "humidity": item["main"]["humidity"],
                        "pressure": item["main"]["pressure"],
                        "description": item["weather"][0]["description"],
                        "main": item["weather"][0]["main"],
                        "icon": item["weather"][0]["icon"],
                        "wind_speed": item.get("wind", {}).get("speed", 0),
                        "precipitation": item.get("rain", {}).get("3h", 0) + item.get("snow", {}).get("3h", 0)
                    })
                
                # Cache the result
                self.cache[cache_key] = {
                    "data": formatted_forecast,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Successfully fetched forecast data for {lat}, {lon}")
                return formatted_forecast
                
        except Exception as e:
            logger.error(f"Error fetching forecast: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch weather forecast"
            )
    
    def clear_cache(self):
        """Clear all cached weather data"""
        self.cache.clear()
        logger.info("Weather cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        valid_entries = sum(1 for entry in self.cache.values() if self._is_cache_valid(entry))
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries
        }

# Global weather service instance
weather_service = WeatherService()