"""
Environmental Monitoring Router
API endpoints for real-time environmental monitoring and disease risk assessment
Supports the AI-Driven Crop Disease Prediction and Management System
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from ..services.environmental_monitoring_service import EnvironmentalMonitoringService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global service instance (in production, use dependency injection)
monitoring_service = EnvironmentalMonitoringService()

# Dependency to get monitoring service
async def get_monitoring_service() -> EnvironmentalMonitoringService:
    """Dependency to get environmental monitoring service instance"""
    return monitoring_service

@router.get("/health")
async def health_check():
    """Health check endpoint for environmental monitoring service"""
    return {
        "status": "healthy",
        "service": "Environmental Monitoring",
        "timestamp": datetime.now().isoformat(),
        "message": "Environmental monitoring service is operational"
    }

@router.post("/stations/register")
async def register_monitoring_station(
    station_data: Dict[str, Any] = Body(...),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Register a new environmental monitoring station
    
    Request body should contain:
    - station_id: Unique identifier for the station
    - location: Location name
    - latitude: Station latitude
    - longitude: Station longitude
    """
    try:
        required_fields = ["station_id", "location", "latitude", "longitude"]
        for field in required_fields:
            if field not in station_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        coordinates = (station_data["latitude"], station_data["longitude"])
        
        result = await service.register_monitoring_station(
            station_id=station_data["station_id"],
            location=station_data["location"],
            coordinates=coordinates
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register_monitoring_station: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/stations/{station_id}/readings")
async def record_environmental_reading(
    station_id: str,
    reading_data: Dict[str, Any] = Body(...),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Record a new environmental reading from a monitoring station
    
    Request body should contain environmental measurements:
    - temperature: Air temperature in Celsius
    - humidity: Relative humidity percentage
    - rainfall: Rainfall in mm
    - wind_speed: Wind speed in m/s
    - wind_direction: Wind direction in degrees
    - atmospheric_pressure: Atmospheric pressure in hPa
    - soil_temperature: Soil temperature in Celsius
    - soil_moisture: Soil moisture percentage
    - soil_ph: Soil pH value
    - leaf_wetness: Leaf wetness duration in hours
    - solar_radiation: Solar radiation in W/m²
    - uv_index: UV index value
    """
    try:
        result = await service.record_environmental_reading(station_id, reading_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in record_environmental_reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stations/{station_id}/risk-assessment")
async def get_current_risk_assessment(
    station_id: str,
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Get current disease risk assessment for a monitoring station
    """
    try:
        # Get the latest reading and perform risk assessment
        if station_id not in service.historical_data or not service.historical_data[station_id]:
            raise HTTPException(
                status_code=404,
                detail=f"No environmental data found for station {station_id}"
            )
        
        latest_reading = service.historical_data[station_id][-1]
        risk_assessment = await service.assess_disease_risk(station_id, latest_reading)
        
        if "error" in risk_assessment:
            raise HTTPException(status_code=400, detail=risk_assessment["error"])
        
        return risk_assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_risk_assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stations")
async def get_monitoring_stations(
    station_id: Optional[str] = Query(None, description="Specific station ID to retrieve"),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Get information about monitoring stations
    """
    try:
        result = await service.get_station_status(station_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_monitoring_stations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/alerts")
async def get_active_alerts(
    station_id: Optional[str] = Query(None, description="Filter alerts by station ID"),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Get active disease risk alerts
    """
    try:
        result = await service.get_active_alerts(station_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_active_alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stations/{station_id}/summary")
async def get_environmental_summary(
    station_id: str,
    hours_back: int = Query(24, description="Number of hours to look back for summary", ge=1, le=168),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Get environmental conditions summary for a station over a specified time period
    """
    try:
        result = await service.get_environmental_summary(station_id, hours_back)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_environmental_summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/simulate-reading")
async def simulate_environmental_reading(
    simulation_data: Dict[str, Any] = Body(...),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Simulate environmental reading for testing purposes
    
    Request body should contain:
    - station_id: Station ID (will be auto-registered if not exists)
    - location: Location name
    - environmental_data: Dictionary with environmental measurements
    """
    try:
        station_id = simulation_data.get("station_id", "test_station_001")
        location = simulation_data.get("location", "Test Location")
        environmental_data = simulation_data.get("environmental_data", {})
        
        # Auto-register station if it doesn't exist
        if station_id not in service.monitoring_stations:
            await service.register_monitoring_station(
                station_id=station_id,
                location=location,
                coordinates=(12.9716, 77.5946)  # Default coordinates (Bangalore)
            )
        
        # Set default values for missing environmental data
        default_data = {
            "temperature": 28.0,
            "humidity": 75.0,
            "rainfall": 5.0,
            "wind_speed": 2.5,
            "wind_direction": 180.0,
            "atmospheric_pressure": 1013.25,
            "soil_temperature": 26.0,
            "soil_moisture": 60.0,
            "soil_ph": 6.8,
            "leaf_wetness": 4.0,
            "solar_radiation": 800.0,
            "uv_index": 6.0
        }
        
        # Merge with provided data
        for key, default_value in default_data.items():
            if key not in environmental_data:
                environmental_data[key] = default_value
        
        # Record the reading
        result = await service.record_environmental_reading(station_id, environmental_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "message": "Environmental reading simulated successfully",
            "simulation_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simulate_environmental_reading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/disease-models")
async def get_disease_risk_models(
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Get information about disease risk prediction models
    """
    try:
        models_info = {}
        
        for disease_name, model in service.disease_risk_models.items():
            models_info[disease_name] = {
                "temperature_range": model["temperature_range"],
                "humidity_threshold": model["humidity_threshold"],
                "rainfall_threshold": model["rainfall_threshold"],
                "key_factors": list(model["weights"].keys()),
                "description": f"Risk model for {disease_name.replace('_', ' ').title()}"
            }
        
        return {
            "status": "success",
            "total_models": len(models_info),
            "disease_models": models_info,
            "risk_levels": list(service.alert_thresholds.keys())
        }
        
    except Exception as e:
        logger.error(f"Error in get_disease_risk_models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/bulk-simulate")
async def bulk_simulate_readings(
    bulk_data: Dict[str, Any] = Body(...),
    service: EnvironmentalMonitoringService = Depends(get_monitoring_service)
):
    """
    Simulate multiple environmental readings for testing and demonstration
    
    Request body should contain:
    - station_count: Number of stations to simulate (default: 3)
    - readings_per_station: Number of readings per station (default: 5)
    - base_location: Base location name (default: "Test Region")
    """
    try:
        station_count = bulk_data.get("station_count", 3)
        readings_per_station = bulk_data.get("readings_per_station", 5)
        base_location = bulk_data.get("base_location", "Test Region")
        
        results = []
        
        # Create multiple stations and readings
        for station_num in range(1, station_count + 1):
            station_id = f"station_{station_num:03d}"
            location = f"{base_location} - Station {station_num}"
            
            # Register station
            await service.register_monitoring_station(
                station_id=station_id,
                location=location,
                coordinates=(12.9716 + station_num * 0.1, 77.5946 + station_num * 0.1)
            )
            
            station_results = []
            
            # Generate readings with varying conditions
            for reading_num in range(readings_per_station):
                # Vary conditions to simulate different risk scenarios
                temp_variation = (reading_num - 2) * 2  # -4 to +4 degrees
                humidity_variation = (reading_num - 2) * 5  # -10 to +10%
                rainfall_variation = reading_num * 2  # 0 to 8mm
                
                environmental_data = {
                    "temperature": 28.0 + temp_variation,
                    "humidity": 75.0 + humidity_variation,
                    "rainfall": rainfall_variation,
                    "wind_speed": 2.5 + reading_num * 0.5,
                    "wind_direction": 180.0 + reading_num * 30,
                    "atmospheric_pressure": 1013.25,
                    "soil_temperature": 26.0 + temp_variation * 0.8,
                    "soil_moisture": 60.0 + humidity_variation * 0.5,
                    "soil_ph": 6.8,
                    "leaf_wetness": 4.0 + reading_num,
                    "solar_radiation": 800.0 - reading_num * 50,
                    "uv_index": 6.0 - reading_num * 0.5
                }
                
                reading_result = await service.record_environmental_reading(
                    station_id, environmental_data
                )
                station_results.append(reading_result)
            
            results.append({
                "station_id": station_id,
                "location": location,
                "readings_generated": len(station_results),
                "latest_risk_assessment": station_results[-1].get("risk_assessment", {})
            })
        
        return {
            "status": "success",
            "message": f"Bulk simulation completed: {station_count} stations, {readings_per_station} readings each",
            "stations_created": station_count,
            "total_readings": station_count * readings_per_station,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in bulk_simulate_readings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")