"""
Irrigation Service
Handles irrigation scheduling based on weather data, soil moisture, and crop requirements
"""

import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class IrrigationRecommendation(BaseModel):
    """Model for irrigation recommendation"""
    should_irrigate: bool
    irrigation_time: str
    irrigation_amount: float
    next_check_interval: int
    reason: str
    weather_data: Dict[str, Any]
    soil_moisture_status: str
    crop_stage: Optional[str] = None

class IrrigationService:
    """Service for irrigation scheduling and recommendations"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.rules_file = os.path.join(os.path.dirname(__file__), "..", "irrigation_rule.json")
        self.irrigation_rules = self._load_irrigation_rules()
        
    def _load_irrigation_rules(self) -> Dict[str, Any]:
        """Load irrigation rules from JSON file"""
        try:
            with open(self.rules_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Irrigation rules file not found: {self.rules_file}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in irrigation rules file: {self.rules_file}")
            return {}
    
    async def get_weather_data(self, latitude: float = 28.6139, longitude: float = 77.2090) -> Dict[str, Any]:
        """Fetch current weather data from OpenWeatherMap API"""
        if not self.api_key:
            logger.error("OpenWeatherMap API key not found in environment variables")
            return self._get_mock_weather_data()
        
        try:
            # Current weather
            current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={self.api_key}&units=metric"
            current_response = requests.get(current_url, timeout=10)
            current_response.raise_for_status()
            current_data = current_response.json()
            
            # Hourly forecast for rainfall prediction
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={self.api_key}&units=metric"
            forecast_response = requests.get(forecast_url, timeout=10)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            # Extract relevant weather information
            weather_info = {
                "temperature": current_data["main"]["temp"],
                "humidity": current_data["main"]["humidity"],
                "pressure": current_data["main"]["pressure"],
                "wind_speed": current_data["wind"]["speed"],
                "weather_condition": current_data["weather"][0]["main"].lower(),
                "description": current_data["weather"][0]["description"],
                "rainfall_1h": current_data.get("rain", {}).get("1h", 0),
                "rainfall_3h": current_data.get("rain", {}).get("3h", 0),
                "hourly_forecast": []
            }
            
            # Process hourly forecast for next 24 hours
            for item in forecast_data["list"][:8]:  # Next 24 hours (3-hour intervals)
                weather_info["hourly_forecast"].append({
                    "datetime": item["dt_txt"],
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "rainfall": item.get("rain", {}).get("3h", 0),
                    "weather": item["weather"][0]["main"].lower()
                })
            
            return weather_info
            
        except requests.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_mock_weather_data()
        except Exception as e:
            logger.error(f"Unexpected error in weather data fetch: {e}")
            return self._get_mock_weather_data()
    
    def _get_mock_weather_data(self) -> Dict[str, Any]:
        """Return mock weather data when API is unavailable"""
        return {
            "temperature": 25.0,
            "humidity": 65.0,
            "pressure": 1013.25,
            "wind_speed": 3.5,
            "weather_condition": "clear",
            "description": "clear sky",
            "rainfall_1h": 0,
            "rainfall_3h": 0,
            "hourly_forecast": [
                {
                    "datetime": (datetime.now() + timedelta(hours=i*3)).strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": 25.0 + (i * 0.5),
                    "humidity": 65.0,
                    "rainfall": 0,
                    "weather": "clear"
                } for i in range(8)
            ]
        }
    
    def calculate_expected_rainfall(self, weather_data: Dict[str, Any]) -> float:
        """Calculate expected rainfall in next 24 hours"""
        total_rainfall = 0
        for forecast in weather_data.get("hourly_forecast", []):
            total_rainfall += forecast.get("rainfall", 0)
        return total_rainfall
    
    def get_irrigation_recommendation(
        self, 
        crop_type: str, 
        soil_moisture: float, 
        weather_data: Dict[str, Any],
        crop_stage: Optional[str] = None,
        soil_type: str = "loamy",
        field_size: float = 1.0
    ) -> IrrigationRecommendation:
        """Generate irrigation recommendation based on all parameters"""
        
        # Get crop-specific rules
        crop_rules = self.irrigation_rules.get("irrigation_rules", {}).get(crop_type.lower())
        if not crop_rules:
            crop_rules = self._get_default_crop_rules()
        
        general_rules = self.irrigation_rules.get("general_rules", {})
        weather_conditions = self.irrigation_rules.get("weather_conditions", {})
        soil_types = self.irrigation_rules.get("soil_types", {})
        
        # Determine moisture threshold based on crop stage
        if crop_stage and crop_stage in crop_rules.get("growth_stages", {}):
            stage_rules = crop_rules["growth_stages"][crop_stage]
            moisture_threshold = stage_rules["moisture_threshold"]
            base_irrigation_amount = stage_rules["irrigation_amount"]
        else:
            moisture_threshold = crop_rules["moisture_threshold"]
            base_irrigation_amount = crop_rules["irrigation_amount"]
        
        # Calculate expected rainfall
        expected_rainfall = self.calculate_expected_rainfall(weather_data)
        rainfall_threshold = general_rules.get("rainfall_threshold", 5)
        
        # Check weather condition multiplier
        weather_condition = weather_data.get("weather_condition", "clear")
        weather_multiplier = weather_conditions.get(weather_condition, {}).get("irrigation_multiplier", 1.0)
        
        # Check soil type modifier
        soil_modifier = soil_types.get(soil_type, {}).get("irrigation_frequency_modifier", 1.0)
        
        # Determine if irrigation is needed
        should_irrigate = False
        reason_parts = []
        
        # Check soil moisture
        if soil_moisture < moisture_threshold:
            should_irrigate = True
            reason_parts.append(f"Soil moisture ({soil_moisture}%) below threshold ({moisture_threshold}%)")
        
        # Check rainfall
        if expected_rainfall >= rainfall_threshold:
            should_irrigate = False
            reason_parts.append(f"Expected rainfall ({expected_rainfall:.1f}mm) above threshold ({rainfall_threshold}mm)")
        elif expected_rainfall > 0:
            reason_parts.append(f"Low expected rainfall ({expected_rainfall:.1f}mm)")
        
        # Emergency irrigation check
        emergency_threshold = general_rules.get("emergency_irrigation_threshold", 15)
        if soil_moisture < emergency_threshold:
            should_irrigate = True
            reason_parts.append(f"Emergency irrigation needed - critically low moisture ({soil_moisture}%)")
        
        # Calculate irrigation amount
        irrigation_amount = base_irrigation_amount * weather_multiplier * soil_modifier
        
        # Adjust for field size
        total_irrigation_amount = irrigation_amount * field_size
        
        # Water conservation mode
        conservation_mode = general_rules.get("water_conservation_mode", {})
        if conservation_mode.get("enabled", False) and not soil_moisture < emergency_threshold:
            reduction = conservation_mode.get("reduction_percentage", 10) / 100
            irrigation_amount *= (1 - reduction)
            total_irrigation_amount *= (1 - reduction)
            reason_parts.append(f"Water conservation mode active ({reduction*100}% reduction)")
        
        # Determine optimal irrigation time
        optimal_time = crop_rules.get("optimal_time", general_rules.get("default_irrigation_time", "6:00 AM"))
        
        # Determine next check interval
        check_interval = crop_rules.get("check_interval", 2)
        
        # Determine soil moisture status
        if soil_moisture < emergency_threshold:
            moisture_status = "Critical"
        elif soil_moisture < moisture_threshold:
            moisture_status = "Low"
        elif soil_moisture < moisture_threshold + 10:
            moisture_status = "Moderate"
        else:
            moisture_status = "Good"
        
        # Compile reason
        if should_irrigate:
            reason = "Irrigation recommended: " + "; ".join(reason_parts)
        else:
            reason = "No irrigation needed: " + "; ".join(reason_parts) if reason_parts else "Soil moisture and weather conditions are adequate"
        
        return IrrigationRecommendation(
            should_irrigate=should_irrigate,
            irrigation_time=optimal_time,
            irrigation_amount=round(total_irrigation_amount, 2),
            next_check_interval=check_interval,
            reason=reason,
            weather_data=weather_data,
            soil_moisture_status=moisture_status,
            crop_stage=crop_stage
        )
    
    def _get_default_crop_rules(self) -> Dict[str, Any]:
        """Return default crop rules when specific crop is not found"""
        return {
            "moisture_threshold": 30,
            "irrigation_amount": 15,
            "check_interval": 2,
            "optimal_time": "6:00 AM",
            "season": "all_season"
        }
    
    async def log_irrigation_schedule(
        self, 
        user_id: str, 
        crop_type: str, 
        recommendation: IrrigationRecommendation,
        field_name: str = "Default Field"
    ) -> Dict[str, Any]:
        """Log irrigation schedule acceptance to database"""
        # This would typically interact with a database
        # For now, we'll return a mock response
        schedule_log = {
            "id": f"irrigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": user_id,
            "crop_type": crop_type,
            "field_name": field_name,
            "scheduled_time": recommendation.irrigation_time,
            "irrigation_amount": recommendation.irrigation_amount,
            "next_check_date": (datetime.now() + timedelta(days=recommendation.next_check_interval)).isoformat(),
            "reason": recommendation.reason,
            "weather_conditions": recommendation.weather_data,
            "soil_moisture_status": recommendation.soil_moisture_status,
            "timestamp": datetime.now().isoformat(),
            "status": "scheduled"
        }
        
        logger.info(f"Irrigation schedule logged for user {user_id}: {schedule_log['id']}")
        return schedule_log

# Create service instance
irrigation_service = IrrigationService()