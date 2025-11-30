"""
Environmental Monitoring Service
Real-time environmental data integration for enhanced disease prediction
Supports the AI-Driven Crop Disease Prediction and Management System
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import math
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentalReading:
    """Data class for environmental readings"""
    timestamp: datetime
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    wind_direction: float
    atmospheric_pressure: float
    soil_temperature: float
    soil_moisture: float
    soil_ph: float
    leaf_wetness: float
    solar_radiation: float
    uv_index: float
    location: str

@dataclass
class DiseaseRiskAlert:
    """Data class for disease risk alerts"""
    disease_name: str
    risk_level: str
    risk_score: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    alert_timestamp: datetime
    validity_period: int  # hours

class EnvironmentalMonitoringService:
    """Service for real-time environmental monitoring and disease risk assessment"""
    
    def __init__(self):
        """Initialize the environmental monitoring service"""
        self.monitoring_stations = {}
        self.historical_data = {}
        self.disease_risk_models = self._initialize_disease_risk_models()
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.active_alerts = []
        logger.info("Environmental Monitoring Service initialized successfully")
    
    def _initialize_disease_risk_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize disease risk prediction models based on environmental conditions"""
        return {
            "bacterial_blight": {
                "temperature_range": [25, 35],
                "humidity_threshold": 80,
                "rainfall_threshold": 10,  # mm per day
                "wind_speed_factor": 0.3,
                "leaf_wetness_threshold": 6,  # hours
                "risk_formula": "temperature_factor * humidity_factor * rainfall_factor * wind_factor * leaf_wetness_factor",
                "weights": {
                    "temperature": 0.25,
                    "humidity": 0.30,
                    "rainfall": 0.20,
                    "wind_speed": 0.15,
                    "leaf_wetness": 0.10
                }
            },
            "blast_disease": {
                "temperature_range": [20, 30],
                "humidity_threshold": 85,
                "rainfall_threshold": 5,
                "wind_speed_factor": 0.2,
                "leaf_wetness_threshold": 8,
                "risk_formula": "temperature_factor * humidity_factor * rainfall_factor * leaf_wetness_factor",
                "weights": {
                    "temperature": 0.30,
                    "humidity": 0.35,
                    "rainfall": 0.15,
                    "wind_speed": 0.10,
                    "leaf_wetness": 0.10
                }
            },
            "brown_spot": {
                "temperature_range": [25, 30],
                "humidity_threshold": 75,
                "rainfall_threshold": 8,
                "wind_speed_factor": 0.1,
                "leaf_wetness_threshold": 4,
                "risk_formula": "temperature_factor * humidity_factor * rainfall_factor",
                "weights": {
                    "temperature": 0.35,
                    "humidity": 0.30,
                    "rainfall": 0.25,
                    "wind_speed": 0.05,
                    "leaf_wetness": 0.05
                }
            },
            "sheath_blight": {
                "temperature_range": [28, 32],
                "humidity_threshold": 80,
                "rainfall_threshold": 15,
                "wind_speed_factor": 0.1,
                "leaf_wetness_threshold": 10,
                "risk_formula": "temperature_factor * humidity_factor * rainfall_factor * soil_moisture_factor",
                "weights": {
                    "temperature": 0.25,
                    "humidity": 0.30,
                    "rainfall": 0.20,
                    "soil_moisture": 0.20,
                    "wind_speed": 0.05
                }
            },
            "late_blight": {
                "temperature_range": [15, 20],
                "humidity_threshold": 85,
                "rainfall_threshold": 12,
                "wind_speed_factor": 0.3,
                "leaf_wetness_threshold": 12,
                "risk_formula": "temperature_factor * humidity_factor * rainfall_factor * wind_factor",
                "weights": {
                    "temperature": 0.30,
                    "humidity": 0.35,
                    "rainfall": 0.20,
                    "wind_speed": 0.15
                }
            }
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for different risk levels"""
        return {
            "low": {"min_score": 0.0, "max_score": 0.3},
            "moderate": {"min_score": 0.3, "max_score": 0.6},
            "high": {"min_score": 0.6, "max_score": 0.8},
            "critical": {"min_score": 0.8, "max_score": 1.0}
        }
    
    async def register_monitoring_station(self, station_id: str, location: str, 
                                        coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """
        Register a new environmental monitoring station
        
        Args:
            station_id: Unique identifier for the station
            location: Location name
            coordinates: (latitude, longitude) tuple
            
        Returns:
            Registration confirmation
        """
        try:
            self.monitoring_stations[station_id] = {
                "location": location,
                "coordinates": coordinates,
                "registered_at": datetime.now(),
                "status": "active",
                "last_reading": None,
                "total_readings": 0
            }
            
            # Initialize historical data storage
            self.historical_data[station_id] = []
            
            logger.info(f"Monitoring station {station_id} registered at {location}")
            
            return {
                "status": "success",
                "station_id": station_id,
                "location": location,
                "coordinates": coordinates,
                "message": f"Monitoring station registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Error registering monitoring station: {str(e)}")
            return {"error": f"Failed to register monitoring station: {str(e)}"}
    
    async def record_environmental_reading(self, station_id: str, 
                                         reading_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a new environmental reading from a monitoring station
        
        Args:
            station_id: ID of the monitoring station
            reading_data: Environmental data readings
            
        Returns:
            Recording confirmation and risk assessment
        """
        try:
            if station_id not in self.monitoring_stations:
                return {"error": f"Monitoring station {station_id} not found"}
            
            # Create environmental reading object
            reading = EnvironmentalReading(
                timestamp=datetime.now(),
                temperature=reading_data.get("temperature", 0.0),
                humidity=reading_data.get("humidity", 0.0),
                rainfall=reading_data.get("rainfall", 0.0),
                wind_speed=reading_data.get("wind_speed", 0.0),
                wind_direction=reading_data.get("wind_direction", 0.0),
                atmospheric_pressure=reading_data.get("atmospheric_pressure", 1013.25),
                soil_temperature=reading_data.get("soil_temperature", 0.0),
                soil_moisture=reading_data.get("soil_moisture", 0.0),
                soil_ph=reading_data.get("soil_ph", 7.0),
                leaf_wetness=reading_data.get("leaf_wetness", 0.0),
                solar_radiation=reading_data.get("solar_radiation", 0.0),
                uv_index=reading_data.get("uv_index", 0.0),
                location=self.monitoring_stations[station_id]["location"]
            )
            
            # Store the reading
            self.historical_data[station_id].append(reading)
            
            # Update station info
            self.monitoring_stations[station_id]["last_reading"] = reading.timestamp
            self.monitoring_stations[station_id]["total_readings"] += 1
            
            # Keep only last 1000 readings per station
            if len(self.historical_data[station_id]) > 1000:
                self.historical_data[station_id] = self.historical_data[station_id][-1000:]
            
            # Perform real-time disease risk assessment
            risk_assessment = await self.assess_disease_risk(station_id, reading)
            
            # Generate alerts if necessary
            alerts = await self.generate_risk_alerts(station_id, risk_assessment)
            
            logger.info(f"Environmental reading recorded for station {station_id}")
            
            return {
                "status": "success",
                "station_id": station_id,
                "reading_timestamp": reading.timestamp.isoformat(),
                "risk_assessment": risk_assessment,
                "alerts_generated": len(alerts),
                "active_alerts": alerts,
                "message": "Environmental reading recorded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error recording environmental reading: {str(e)}")
            return {"error": f"Failed to record environmental reading: {str(e)}"}
    
    async def assess_disease_risk(self, station_id: str, 
                                current_reading: EnvironmentalReading) -> Dict[str, Any]:
        """
        Assess disease risk based on current and historical environmental conditions
        
        Args:
            station_id: ID of the monitoring station
            current_reading: Current environmental reading
            
        Returns:
            Disease risk assessment
        """
        try:
            risk_scores = {}
            
            for disease_name, model in self.disease_risk_models.items():
                risk_score = self._calculate_disease_risk_score(disease_name, current_reading, model)
                risk_level = self._determine_risk_level(risk_score)
                
                risk_scores[disease_name] = {
                    "risk_score": round(risk_score, 3),
                    "risk_level": risk_level,
                    "contributing_factors": self._identify_contributing_factors(disease_name, current_reading, model),
                    "trend": await self._analyze_risk_trend(station_id, disease_name)
                }
            
            # Calculate overall risk
            overall_risk = sum(score["risk_score"] for score in risk_scores.values()) / len(risk_scores)
            overall_level = self._determine_risk_level(overall_risk)
            
            return {
                "station_id": station_id,
                "assessment_timestamp": current_reading.timestamp.isoformat(),
                "overall_risk": {
                    "score": round(overall_risk, 3),
                    "level": overall_level
                },
                "disease_risks": risk_scores,
                "environmental_conditions": {
                    "temperature": current_reading.temperature,
                    "humidity": current_reading.humidity,
                    "rainfall": current_reading.rainfall,
                    "wind_speed": current_reading.wind_speed,
                    "leaf_wetness": current_reading.leaf_wetness,
                    "soil_moisture": current_reading.soil_moisture
                },
                "recommendations": self._generate_risk_recommendations(risk_scores)
            }
            
        except Exception as e:
            logger.error(f"Error assessing disease risk: {str(e)}")
            return {"error": f"Failed to assess disease risk: {str(e)}"}
    
    def _calculate_disease_risk_score(self, disease_name: str, reading: EnvironmentalReading, 
                                    model: Dict[str, Any]) -> float:
        """Calculate disease risk score based on environmental conditions"""
        try:
            # Temperature factor
            temp_range = model["temperature_range"]
            if temp_range[0] <= reading.temperature <= temp_range[1]:
                temp_factor = 1.0
            else:
                # Decrease factor as temperature moves away from optimal range
                distance = min(abs(reading.temperature - temp_range[0]), 
                             abs(reading.temperature - temp_range[1]))
                temp_factor = max(0.0, 1.0 - (distance / 10.0))
            
            # Humidity factor
            humidity_threshold = model["humidity_threshold"]
            if reading.humidity >= humidity_threshold:
                humidity_factor = 1.0
            else:
                humidity_factor = reading.humidity / humidity_threshold
            
            # Rainfall factor
            rainfall_threshold = model["rainfall_threshold"]
            if reading.rainfall >= rainfall_threshold:
                rainfall_factor = 1.0
            else:
                rainfall_factor = reading.rainfall / rainfall_threshold
            
            # Wind speed factor
            wind_factor = min(1.0, reading.wind_speed * model["wind_speed_factor"])
            
            # Leaf wetness factor
            leaf_wetness_threshold = model.get("leaf_wetness_threshold", 6)
            if reading.leaf_wetness >= leaf_wetness_threshold:
                leaf_wetness_factor = 1.0
            else:
                leaf_wetness_factor = reading.leaf_wetness / leaf_wetness_threshold
            
            # Soil moisture factor (for soil-borne diseases)
            soil_moisture_factor = min(1.0, reading.soil_moisture / 80.0)  # Assuming 80% is high moisture
            
            # Calculate weighted risk score
            weights = model["weights"]
            risk_score = (
                temp_factor * weights.get("temperature", 0.25) +
                humidity_factor * weights.get("humidity", 0.30) +
                rainfall_factor * weights.get("rainfall", 0.20) +
                wind_factor * weights.get("wind_speed", 0.15) +
                leaf_wetness_factor * weights.get("leaf_wetness", 0.10) +
                soil_moisture_factor * weights.get("soil_moisture", 0.0)
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating disease risk score: {str(e)}")
            return 0.0
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        for level, thresholds in self.alert_thresholds.items():
            if thresholds["min_score"] <= risk_score <= thresholds["max_score"]:
                return level
        return "unknown"
    
    def _identify_contributing_factors(self, disease_name: str, reading: EnvironmentalReading, 
                                     model: Dict[str, Any]) -> List[str]:
        """Identify environmental factors contributing to disease risk"""
        factors = []
        
        # Check temperature
        temp_range = model["temperature_range"]
        if temp_range[0] <= reading.temperature <= temp_range[1]:
            factors.append(f"Optimal temperature ({reading.temperature}°C)")
        
        # Check humidity
        if reading.humidity >= model["humidity_threshold"]:
            factors.append(f"High humidity ({reading.humidity}%)")
        
        # Check rainfall
        if reading.rainfall >= model["rainfall_threshold"]:
            factors.append(f"Adequate rainfall ({reading.rainfall}mm)")
        
        # Check leaf wetness
        leaf_wetness_threshold = model.get("leaf_wetness_threshold", 6)
        if reading.leaf_wetness >= leaf_wetness_threshold:
            factors.append(f"Extended leaf wetness ({reading.leaf_wetness} hours)")
        
        # Check wind conditions
        if reading.wind_speed > 5.0:
            factors.append(f"Moderate to high wind speed ({reading.wind_speed} m/s)")
        
        return factors
    
    async def _analyze_risk_trend(self, station_id: str, disease_name: str) -> str:
        """Analyze risk trend over the past few readings"""
        try:
            if station_id not in self.historical_data or len(self.historical_data[station_id]) < 3:
                return "insufficient_data"
            
            # Get last 5 readings
            recent_readings = self.historical_data[station_id][-5:]
            model = self.disease_risk_models[disease_name]
            
            # Calculate risk scores for recent readings
            risk_scores = []
            for reading in recent_readings:
                score = self._calculate_disease_risk_score(disease_name, reading, model)
                risk_scores.append(score)
            
            # Analyze trend
            if len(risk_scores) >= 3:
                recent_avg = sum(risk_scores[-3:]) / 3
                earlier_avg = sum(risk_scores[:-3]) / max(1, len(risk_scores) - 3)
                
                if recent_avg > earlier_avg + 0.1:
                    return "increasing"
                elif recent_avg < earlier_avg - 0.1:
                    return "decreasing"
                else:
                    return "stable"
            
            return "stable"
            
        except Exception as e:
            logger.error(f"Error analyzing risk trend: {str(e)}")
            return "unknown"
    
    def _generate_risk_recommendations(self, risk_scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        # Check for high-risk diseases
        high_risk_diseases = [disease for disease, data in risk_scores.items() 
                            if data["risk_level"] in ["high", "critical"]]
        
        if high_risk_diseases:
            recommendations.append("Implement immediate disease monitoring and prevention measures")
            recommendations.append("Consider applying preventive treatments for high-risk diseases")
            recommendations.append("Increase field scouting frequency")
        
        # Check for moderate risk
        moderate_risk_diseases = [disease for disease, data in risk_scores.items() 
                                if data["risk_level"] == "moderate"]
        
        if moderate_risk_diseases:
            recommendations.append("Monitor field conditions closely")
            recommendations.append("Prepare preventive treatments if conditions worsen")
        
        # General recommendations
        recommendations.extend([
            "Maintain proper field sanitation",
            "Ensure adequate drainage if rainfall is high",
            "Monitor weather forecasts for changing conditions",
            "Document any disease symptoms observed in the field"
        ])
        
        return recommendations
    
    async def generate_risk_alerts(self, station_id: str, 
                                 risk_assessment: Dict[str, Any]) -> List[DiseaseRiskAlert]:
        """Generate disease risk alerts based on assessment"""
        try:
            alerts = []
            
            for disease_name, risk_data in risk_assessment.get("disease_risks", {}).items():
                if risk_data["risk_level"] in ["high", "critical"]:
                    alert = DiseaseRiskAlert(
                        disease_name=disease_name,
                        risk_level=risk_data["risk_level"],
                        risk_score=risk_data["risk_score"],
                        contributing_factors=risk_data["contributing_factors"],
                        recommended_actions=self._get_disease_specific_actions(disease_name, risk_data["risk_level"]),
                        alert_timestamp=datetime.now(),
                        validity_period=24  # 24 hours
                    )
                    alerts.append(alert)
            
            # Add alerts to active alerts list
            self.active_alerts.extend(alerts)
            
            # Clean up expired alerts
            await self._cleanup_expired_alerts()
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {str(e)}")
            return []
    
    def _get_disease_specific_actions(self, disease_name: str, risk_level: str) -> List[str]:
        """Get disease-specific recommended actions"""
        actions_map = {
            "bacterial_blight": [
                "Apply copper-based bactericides",
                "Avoid overhead irrigation",
                "Remove infected plant debris",
                "Increase plant spacing for better air circulation"
            ],
            "blast_disease": [
                "Apply systemic fungicides (tricyclazole)",
                "Reduce nitrogen fertilization",
                "Improve field drainage",
                "Monitor for neck blast symptoms"
            ],
            "brown_spot": [
                "Apply protective fungicides",
                "Ensure balanced nutrition",
                "Avoid water stress",
                "Remove infected leaves"
            ],
            "sheath_blight": [
                "Apply validamycin or hexaconazole",
                "Drain fields periodically",
                "Avoid excessive nitrogen",
                "Monitor water line area closely"
            ],
            "late_blight": [
                "Apply protective fungicides immediately",
                "Improve air circulation",
                "Avoid overhead irrigation",
                "Remove infected plants"
            ]
        }
        
        base_actions = actions_map.get(disease_name, ["Monitor field conditions closely"])
        
        if risk_level == "critical":
            base_actions.insert(0, "URGENT: Implement immediate control measures")
        
        return base_actions
    
    async def _cleanup_expired_alerts(self):
        """Remove expired alerts from active alerts list"""
        try:
            current_time = datetime.now()
            self.active_alerts = [
                alert for alert in self.active_alerts
                if (current_time - alert.alert_timestamp).total_seconds() < (alert.validity_period * 3600)
            ]
        except Exception as e:
            logger.error(f"Error cleaning up expired alerts: {str(e)}")
    
    async def get_station_status(self, station_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of monitoring stations"""
        try:
            if station_id:
                if station_id not in self.monitoring_stations:
                    return {"error": f"Station {station_id} not found"}
                
                station_info = self.monitoring_stations[station_id].copy()
                station_info["recent_readings_count"] = len(self.historical_data.get(station_id, []))
                
                return {
                    "status": "success",
                    "station": station_info
                }
            else:
                # Return all stations
                all_stations = {}
                for sid, info in self.monitoring_stations.items():
                    station_info = info.copy()
                    station_info["recent_readings_count"] = len(self.historical_data.get(sid, []))
                    all_stations[sid] = station_info
                
                return {
                    "status": "success",
                    "total_stations": len(all_stations),
                    "stations": all_stations
                }
                
        except Exception as e:
            logger.error(f"Error getting station status: {str(e)}")
            return {"error": f"Failed to get station status: {str(e)}"}
    
    async def get_active_alerts(self, station_id: Optional[str] = None) -> Dict[str, Any]:
        """Get active disease risk alerts"""
        try:
            await self._cleanup_expired_alerts()
            
            if station_id:
                # Filter alerts for specific station
                station_location = self.monitoring_stations.get(station_id, {}).get("location", "")
                filtered_alerts = [
                    {
                        "disease_name": alert.disease_name,
                        "risk_level": alert.risk_level,
                        "risk_score": alert.risk_score,
                        "contributing_factors": alert.contributing_factors,
                        "recommended_actions": alert.recommended_actions,
                        "alert_timestamp": alert.alert_timestamp.isoformat(),
                        "validity_period": alert.validity_period
                    }
                    for alert in self.active_alerts
                    # For now, return all alerts (could be filtered by location in future)
                ]
            else:
                filtered_alerts = [
                    {
                        "disease_name": alert.disease_name,
                        "risk_level": alert.risk_level,
                        "risk_score": alert.risk_score,
                        "contributing_factors": alert.contributing_factors,
                        "recommended_actions": alert.recommended_actions,
                        "alert_timestamp": alert.alert_timestamp.isoformat(),
                        "validity_period": alert.validity_period
                    }
                    for alert in self.active_alerts
                ]
            
            return {
                "status": "success",
                "total_alerts": len(filtered_alerts),
                "alerts": filtered_alerts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            return {"error": f"Failed to get active alerts: {str(e)}"}
    
    async def get_environmental_summary(self, station_id: str, 
                                      hours_back: int = 24) -> Dict[str, Any]:
        """Get environmental conditions summary for the past period"""
        try:
            if station_id not in self.historical_data:
                return {"error": f"No data found for station {station_id}"}
            
            # Get readings from the specified time period
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_readings = [
                reading for reading in self.historical_data[station_id]
                if reading.timestamp >= cutoff_time
            ]
            
            if not recent_readings:
                return {"error": f"No recent data found for station {station_id}"}
            
            # Calculate summary statistics
            temperatures = [r.temperature for r in recent_readings]
            humidities = [r.humidity for r in recent_readings]
            rainfalls = [r.rainfall for r in recent_readings]
            wind_speeds = [r.wind_speed for r in recent_readings]
            
            summary = {
                "station_id": station_id,
                "period_hours": hours_back,
                "readings_count": len(recent_readings),
                "temperature": {
                    "min": min(temperatures),
                    "max": max(temperatures),
                    "avg": sum(temperatures) / len(temperatures),
                    "current": recent_readings[-1].temperature
                },
                "humidity": {
                    "min": min(humidities),
                    "max": max(humidities),
                    "avg": sum(humidities) / len(humidities),
                    "current": recent_readings[-1].humidity
                },
                "rainfall": {
                    "total": sum(rainfalls),
                    "max_hourly": max(rainfalls),
                    "current": recent_readings[-1].rainfall
                },
                "wind_speed": {
                    "min": min(wind_speeds),
                    "max": max(wind_speeds),
                    "avg": sum(wind_speeds) / len(wind_speeds),
                    "current": recent_readings[-1].wind_speed
                },
                "last_reading": recent_readings[-1].timestamp.isoformat()
            }
            
            return {
                "status": "success",
                "environmental_summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error getting environmental summary: {str(e)}")
            return {"error": f"Failed to get environmental summary: {str(e)}"}