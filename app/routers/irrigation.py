"""
Irrigation Router
API endpoints for irrigation scheduling and management with weather integration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from ..services.irrigation_service import irrigation_service, IrrigationRecommendation

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class IrrigationScheduleItem(BaseModel):
    """Model for irrigation schedule item"""
    id: str
    crop_type: str
    field_name: str
    scheduled_time: str
    duration_minutes: int
    water_amount_liters: float
    status: str
    priority: str

class IrrigationRequest(BaseModel):
    """Request model for creating irrigation schedule"""
    crop_type: str = Field(..., description="Type of crop")
    field_name: str = Field(..., description="Name of the field")
    field_size_hectares: float = Field(..., ge=0.1, le=1000, description="Field size in hectares")
    soil_moisture: float = Field(default=30, ge=0, le=100, description="Current soil moisture percentage")
    crop_stage: Optional[str] = Field(default=None, description="Current growth stage of crop")
    soil_type: str = Field(default="loamy", description="Type of soil")
    latitude: float = Field(default=28.6139, description="Latitude for weather data")
    longitude: float = Field(default=77.2090, description="Longitude for weather data")

class IrrigationLogRequest(BaseModel):
    """Request model for logging irrigation schedule"""
    user_id: str = Field(..., description="User ID")
    crop_type: str = Field(..., description="Type of crop")
    field_name: str = Field(..., description="Name of the field")
    recommendation_data: Dict[str, Any] = Field(..., description="Recommendation data to log")

@router.get("/recommendation")
async def get_irrigation_recommendation(
    crop_type: str,
    soil_moisture: float,
    field_size: float = 1.0,
    crop_stage: Optional[str] = None,
    soil_type: str = "loamy",
    latitude: float = 28.6139,
    longitude: float = 77.2090
) -> Dict[str, Any]:
    """Get irrigation recommendation based on current conditions"""
    try:
        logger.info(f"Getting irrigation recommendation for {crop_type}")
        
        # Fetch weather data
        weather_data = await irrigation_service.get_weather_data(latitude, longitude)
        
        # Get irrigation recommendation
        recommendation = irrigation_service.get_irrigation_recommendation(
            crop_type=crop_type,
            soil_moisture=soil_moisture,
            weather_data=weather_data,
            crop_stage=crop_stage,
            soil_type=soil_type,
            field_size=field_size
        )
        
        response = {
            "success": True,
            "message": "Irrigation recommendation generated successfully",
            "recommendation": {
                "should_irrigate": recommendation.should_irrigate,
                "irrigation_time": recommendation.irrigation_time,
                "irrigation_amount_mm": recommendation.irrigation_amount,
                "next_check_interval_days": recommendation.next_check_interval,
                "reason": recommendation.reason,
                "soil_moisture_status": recommendation.soil_moisture_status,
                "crop_stage": recommendation.crop_stage
            },
            "weather_data": recommendation.weather_data,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated recommendation: irrigate={recommendation.should_irrigate}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating irrigation recommendation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate irrigation recommendation: {str(e)}"
        )

@router.post("/log-schedule")
async def log_irrigation_schedule(request: IrrigationLogRequest) -> Dict[str, Any]:
    """Log irrigation schedule acceptance"""
    try:
        logger.info(f"Logging irrigation schedule for user {request.user_id}")
        
        # Create recommendation object from request data
        recommendation_dict = request.recommendation_data
        recommendation = IrrigationRecommendation(
            should_irrigate=recommendation_dict.get("should_irrigate", True),
            irrigation_time=recommendation_dict.get("irrigation_time", "6:00 AM"),
            irrigation_amount=recommendation_dict.get("irrigation_amount", 15.0),
            next_check_interval=recommendation_dict.get("next_check_interval", 2),
            reason=recommendation_dict.get("reason", "User accepted recommendation"),
            weather_data=recommendation_dict.get("weather_data", {}),
            soil_moisture_status=recommendation_dict.get("soil_moisture_status", "Unknown")
        )
        
        # Log the schedule
        schedule_log = await irrigation_service.log_irrigation_schedule(
            user_id=request.user_id,
            crop_type=request.crop_type,
            recommendation=recommendation,
            field_name=request.field_name
        )
        
        response = {
            "success": True,
            "message": "Irrigation schedule logged successfully",
            "schedule_log": schedule_log,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully logged schedule: {schedule_log['id']}")
        return response
        
    except Exception as e:
        logger.error(f"Error logging irrigation schedule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log irrigation schedule: {str(e)}"
        )

@router.get("/schedule")
async def get_irrigation_schedule() -> Dict[str, Any]:
    """Get current irrigation schedule"""
    try:
        logger.info("Retrieving irrigation schedule")
        
        # Generate sample irrigation schedule
        schedule_items = generate_sample_schedule()
        
        # Calculate summary statistics
        total_items = len(schedule_items)
        pending_items = len([item for item in schedule_items if item["status"] == "pending"])
        completed_items = len([item for item in schedule_items if item["status"] == "completed"])
        
        response = {
            "success": True,
            "message": "Irrigation schedule retrieved successfully",
            "schedule": schedule_items,
            "summary": {
                "total_items": total_items,
                "pending": pending_items,
                "completed": completed_items,
                "in_progress": total_items - pending_items - completed_items
            },
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Retrieved {total_items} irrigation schedule items")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving irrigation schedule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve irrigation schedule"
        )

@router.post("/schedule")
async def create_irrigation_schedule(request: IrrigationRequest) -> Dict[str, Any]:
    """Create new irrigation schedule based on crop and field parameters"""
    try:
        logger.info(f"Creating irrigation schedule for: {request.dict()}")
        
        # Calculate irrigation requirements
        irrigation_plan = calculate_irrigation_requirements(
            request.crop_type,
            request.field_size_hectares,
            request.soil_moisture,
            request.weather_forecast,
            request.irrigation_method
        )
        
        # Generate schedule items
        schedule_items = generate_irrigation_schedule(
            request.crop_type,
            request.field_name,
            irrigation_plan
        )
        
        response = {
            "success": True,
            "message": "Irrigation schedule created successfully",
            "schedule_id": f"SCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "crop_type": request.crop_type,
            "field_name": request.field_name,
            "irrigation_plan": irrigation_plan,
            "schedule_items": schedule_items,
            "recommendations": generate_irrigation_recommendations(request.soil_moisture, request.weather_forecast),
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Created irrigation schedule with {len(schedule_items)} items")
        return response
        
    except Exception as e:
        logger.error(f"Error creating irrigation schedule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log irrigation schedule: {str(e)}"
        )

@router.get("/schedule/{schedule_id}")
async def get_schedule_by_id(schedule_id: str) -> Dict[str, Any]:
    """Get specific irrigation schedule by ID"""
    try:
        logger.info(f"Retrieving irrigation schedule: {schedule_id}")
        
        # Generate sample schedule for the ID
        schedule_items = generate_sample_schedule()
        
        response = {
            "success": True,
            "schedule_id": schedule_id,
            "schedule": schedule_items,
            "status": "active",
            "created_at": "2024-01-20T08:00:00Z",
            "last_updated": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving schedule {schedule_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve irrigation schedule"
        )

@router.put("/schedule/{schedule_id}/item/{item_id}")
async def update_schedule_item(schedule_id: str, item_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update specific irrigation schedule item"""
    try:
        logger.info(f"Updating schedule item {item_id} in schedule {schedule_id}")
        
        response = {
            "success": True,
            "message": "Schedule item updated successfully",
            "schedule_id": schedule_id,
            "item_id": item_id,
            "updated_fields": list(update_data.keys()),
            "updated_at": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error updating schedule item: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update schedule item"
        )

def generate_sample_schedule() -> List[Dict[str, Any]]:
    """Generate sample irrigation schedule"""
    base_time = datetime.now()
    schedule_items = []
    
    crops = ["Rice", "Wheat", "Maize", "Cotton", "Tomato"]
    fields = ["Field A", "Field B", "Field C", "North Field", "South Field"]
    statuses = ["pending", "in_progress", "completed", "scheduled"]
    priorities = ["high", "medium", "low"]
    
    for i in range(8):
        scheduled_time = base_time + timedelta(hours=i*3)
        
        item = {
            "id": f"IRR_{i+1:03d}",
            "crop_type": crops[i % len(crops)],
            "field_name": fields[i % len(fields)],
            "scheduled_time": scheduled_time.strftime("%Y-%m-%d %H:%M"),
            "duration_minutes": 30 + (i * 15) % 120,
            "water_amount_liters": 500 + (i * 200) % 2000,
            "status": statuses[i % len(statuses)],
            "priority": priorities[i % len(priorities)],
            "irrigation_method": "sprinkler" if i % 2 == 0 else "drip",
            "estimated_completion": (scheduled_time + timedelta(minutes=30 + (i * 15) % 120)).strftime("%Y-%m-%d %H:%M")
        }
        
        schedule_items.append(item)
    
    return schedule_items

def calculate_irrigation_requirements(crop_type: str, field_size: float, soil_moisture: float, 
                                    weather: str, method: str) -> Dict[str, Any]:
    """Calculate irrigation requirements based on parameters"""
    
    # Base water requirements per hectare per day (liters)
    crop_water_needs = {
        "rice": 15000,
        "wheat": 8000,
        "maize": 10000,
        "cotton": 12000,
        "tomato": 6000,
        "potato": 7000
    }
    
    base_requirement = crop_water_needs.get(crop_type.lower(), 9000)
    
    # Adjust for soil moisture
    if soil_moisture < 20:
        moisture_multiplier = 1.5
    elif soil_moisture < 40:
        moisture_multiplier = 1.2
    elif soil_moisture < 60:
        moisture_multiplier = 1.0
    else:
        moisture_multiplier = 0.7
    
    # Adjust for weather
    weather_multiplier = {
        "sunny": 1.3,
        "cloudy": 1.0,
        "rainy": 0.5,
        "hot": 1.5,
        "cool": 0.8
    }
    
    # Adjust for irrigation method efficiency
    method_efficiency = {
        "drip": 0.9,
        "sprinkler": 0.75,
        "flood": 0.6,
        "furrow": 0.65
    }
    
    total_requirement = (base_requirement * field_size * 
                        moisture_multiplier * 
                        weather_multiplier.get(weather.lower(), 1.0) / 
                        method_efficiency.get(method.lower(), 0.75))
    
    return {
        "daily_requirement_liters": round(total_requirement),
        "sessions_per_day": 2 if total_requirement > 10000 else 1,
        "duration_per_session_minutes": min(120, max(30, total_requirement / 200)),
        "water_per_session_liters": round(total_requirement / (2 if total_requirement > 10000 else 1)),
        "efficiency_rating": method_efficiency.get(method.lower(), 0.75) * 100,
        "estimated_cost_per_day": round(total_requirement * 0.001, 2)  # Assuming 0.001 per liter
    }

def generate_irrigation_schedule(crop_type: str, field_name: str, irrigation_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate irrigation schedule items based on plan"""
    schedule_items = []
    base_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    
    sessions_per_day = irrigation_plan["sessions_per_day"]
    duration = irrigation_plan["duration_per_session_minutes"]
    water_amount = irrigation_plan["water_per_session_liters"]
    
    # Generate schedule for next 7 days
    for day in range(7):
        day_start = base_time + timedelta(days=day)
        
        for session in range(sessions_per_day):
            # Schedule sessions at 6 AM and 6 PM if 2 sessions per day
            hour_offset = 0 if session == 0 else 12
            session_time = day_start + timedelta(hours=hour_offset)
            
            item = {
                "id": f"IRR_{day+1}_{session+1}",
                "crop_type": crop_type,
                "field_name": field_name,
                "scheduled_time": session_time.strftime("%Y-%m-%d %H:%M"),
                "duration_minutes": duration,
                "water_amount_liters": water_amount,
                "status": "scheduled" if day > 0 else "pending",
                "priority": "high" if session == 0 else "medium",
                "irrigation_method": "sprinkler",
                "estimated_completion": (session_time + timedelta(minutes=duration)).strftime("%Y-%m-%d %H:%M")
            }
            
            schedule_items.append(item)
    
    return schedule_items

def generate_irrigation_recommendations(soil_moisture: float, weather: str) -> List[str]:
    """Generate irrigation recommendations"""
    recommendations = []
    
    if soil_moisture < 20:
        recommendations.extend([
            "Immediate irrigation required - soil moisture critically low",
            "Consider increasing irrigation frequency",
            "Monitor plants for stress signs"
        ])
    elif soil_moisture < 40:
        recommendations.extend([
            "Schedule irrigation within 24 hours",
            "Check soil moisture daily",
            "Ensure proper water distribution"
        ])
    elif soil_moisture > 80:
        recommendations.extend([
            "Reduce irrigation frequency - soil moisture high",
            "Check drainage to prevent waterlogging",
            "Monitor for fungal diseases"
        ])
    
    if weather.lower() == "rainy":
        recommendations.append("Postpone irrigation due to expected rainfall")
    elif weather.lower() == "hot":
        recommendations.append("Increase irrigation frequency due to high temperatures")
    
    recommendations.extend([
        "Use mulching to retain soil moisture",
        "Consider drip irrigation for water efficiency",
        "Monitor weather forecasts for irrigation planning"
    ])
    
    return recommendations