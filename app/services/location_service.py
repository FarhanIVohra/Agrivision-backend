"""
Location Service for AgriSmart
Handles Geoapify API integration for reverse geocoding and location services
"""

import os
import asyncio
from typing import Dict, Optional, Tuple
import httpx
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationService:
    def __init__(self):
        self.api_key = os.getenv("GEOAPIFY_API_KEY")
        self.base_url = "https://api.geoapify.com/v1"
        
        if not self.api_key:
            logger.warning("Geoapify API key not found in environment variables")
    
    async def reverse_geocode(self, lat: float, lon: float) -> Dict:
        """
        Get location details from coordinates using reverse geocoding
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict containing location information
            
        Raises:
            HTTPException: If API request fails
        """
        if not self.api_key:
            raise HTTPException(
                status_code=500,
                detail="Location service not configured. Please add GEOAPIFY_API_KEY to environment variables."
            )
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/geocode/reverse"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "apiKey": self.api_key,
                    "format": "json"
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                
                if response.status_code == 429:
                    raise HTTPException(
                        status_code=429,
                        detail="Location API rate limit exceeded. Please try again later."
                    )
                
                response.raise_for_status()
                location_data = response.json()
                
                if not location_data.get("results"):
                    raise HTTPException(
                        status_code=404,
                        detail="No location found for the given coordinates"
                    )
                
                result = location_data["results"][0]
                
                # Extract location information
                formatted_location = {
                    "lat": lat,
                    "lon": lon,
                    "country": result.get("country", ""),
                    "country_code": result.get("country_code", ""),
                    "state": result.get("state", ""),
                    "state_code": result.get("state_code", ""),
                    "city": result.get("city", ""),
                    "district": result.get("district", ""),
                    "suburb": result.get("suburb", ""),
                    "postcode": result.get("postcode", ""),
                    "address_line1": result.get("address_line1", ""),
                    "address_line2": result.get("address_line2", ""),
                    "formatted": result.get("formatted", ""),
                    "region": self._determine_region(result),
                    "timezone": result.get("timezone", {}).get("name", ""),
                    "place_id": result.get("place_id", "")
                }
                
                logger.info(f"Successfully reverse geocoded coordinates {lat}, {lon}")
                return formatted_location
                
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=408,
                detail="Location service timeout. Please try again."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid location API key. Please check your configuration."
                )
            elif e.response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail="Location not found for the given coordinates."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Location service error: {e.response.status_code}"
                )
        except Exception as e:
            logger.error(f"Unexpected error in location service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Internal location service error"
            )
    
    async def geocode_address(self, address: str) -> Dict:
        """
        Get coordinates from address using geocoding
        
        Args:
            address: Address string to geocode
            
        Returns:
            Dict containing coordinates and location information
        """
        if not self.api_key:
            raise HTTPException(
                status_code=500,
                detail="Location service not configured"
            )
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/geocode/search"
                params = {
                    "text": address,
                    "apiKey": self.api_key,
                    "format": "json",
                    "limit": 1
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                location_data = response.json()
                
                if not location_data.get("results"):
                    raise HTTPException(
                        status_code=404,
                        detail="No location found for the given address"
                    )
                
                result = location_data["results"][0]
                
                formatted_location = {
                    "lat": result.get("lat"),
                    "lon": result.get("lon"),
                    "country": result.get("country", ""),
                    "state": result.get("state", ""),
                    "city": result.get("city", ""),
                    "formatted": result.get("formatted", ""),
                    "region": self._determine_region(result),
                    "confidence": result.get("rank", {}).get("confidence", 0)
                }
                
                logger.info(f"Successfully geocoded address: {address}")
                return formatted_location
                
        except Exception as e:
            logger.error(f"Error geocoding address: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to geocode address"
            )
    
    def _determine_region(self, location_data: Dict) -> str:
        """
        Determine agricultural region based on location data
        
        Args:
            location_data: Location data from API
            
        Returns:
            String representing the agricultural region
        """
        country = location_data.get("country", "").lower()
        state = location_data.get("state", "").lower()
        
        # India-specific regions
        if country == "india":
            # Northern Plains
            if any(s in state for s in ["punjab", "haryana", "uttar pradesh", "bihar"]):
                return "Northern Plains"
            # Western Region
            elif any(s in state for s in ["maharashtra", "gujarat", "rajasthan"]):
                return "Western Region"
            # Southern Region
            elif any(s in state for s in ["karnataka", "tamil nadu", "andhra pradesh", "telangana", "kerala"]):
                return "Southern Region"
            # Eastern Region
            elif any(s in state for s in ["west bengal", "odisha", "jharkhand"]):
                return "Eastern Region"
            # Central Region
            elif any(s in state for s in ["madhya pradesh", "chhattisgarh"]):
                return "Central Region"
            # Northeastern Region
            elif any(s in state for s in ["assam", "meghalaya", "manipur", "tripura", "nagaland", "mizoram", "arunachal pradesh", "sikkim"]):
                return "Northeastern Region"
            else:
                return "Other Indian Region"
        
        # Default regions for other countries
        elif country in ["united states", "usa"]:
            return "North American Region"
        elif country in ["china"]:
            return "East Asian Region"
        elif country in ["brazil", "argentina"]:
            return "South American Region"
        elif country in ["australia"]:
            return "Australian Region"
        else:
            return "International Region"
    
    async def get_location_suggestions(self, query: str, limit: int = 5) -> list:
        """
        Get location suggestions for autocomplete
        
        Args:
            query: Search query
            limit: Maximum number of suggestions
            
        Returns:
            List of location suggestions
        """
        if not self.api_key:
            raise HTTPException(
                status_code=500,
                detail="Location service not configured"
            )
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/geocode/autocomplete"
                params = {
                    "text": query,
                    "apiKey": self.api_key,
                    "limit": limit,
                    "format": "json"
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                suggestions_data = response.json()
                
                suggestions = []
                for result in suggestions_data.get("results", []):
                    suggestions.append({
                        "formatted": result.get("formatted", ""),
                        "city": result.get("city", ""),
                        "state": result.get("state", ""),
                        "country": result.get("country", ""),
                        "lat": result.get("lat"),
                        "lon": result.get("lon")
                    })
                
                return suggestions
                
        except Exception as e:
            logger.error(f"Error getting location suggestions: {str(e)}")
            return []

# Global location service instance
location_service = LocationService()