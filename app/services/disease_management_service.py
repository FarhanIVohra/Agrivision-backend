"""
Disease Management Service
Provides comprehensive treatment recommendations and preventive measures for crop diseases
Aligns with the AI-Driven Crop Disease Prediction and Management System requirements
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DiseaseManagementService:
    """Service for managing crop disease treatment recommendations and preventive measures"""
    
    def __init__(self):
        self.disease_database = self._initialize_disease_database()
        self.treatment_protocols = self._initialize_treatment_protocols()
        self.preventive_measures = self._initialize_preventive_measures()
    
    def _initialize_disease_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive disease information database"""
        return {
            "Apple___Apple_scab": {
                "scientific_name": "Venturia inaequalis",
                "severity": "moderate",
                "symptoms": [
                    "Dark, scabby lesions on leaves and fruit",
                    "Olive-green to black spots on upper leaf surface",
                    "Premature leaf drop",
                    "Fruit cracking and deformation"
                ],
                "causes": ["High humidity", "Cool temperatures", "Poor air circulation"],
                "affected_parts": ["leaves", "fruit", "twigs"],
                "economic_impact": "Can cause 70% yield loss if untreated"
            },
            "Apple___Black_rot": {
                "scientific_name": "Botryosphaeria obtusa",
                "severity": "high",
                "symptoms": [
                    "Brown to black cankers on branches",
                    "Circular brown spots on leaves",
                    "Black rot on fruit",
                    "Wilting and dieback of branches"
                ],
                "causes": ["Warm, humid weather", "Wounds in bark", "Stress conditions"],
                "affected_parts": ["branches", "leaves", "fruit"],
                "economic_impact": "Can destroy entire orchards if not managed"
            },
            "Tomato___Early_blight": {
                "scientific_name": "Alternaria solani",
                "severity": "moderate",
                "symptoms": [
                    "Dark brown spots with concentric rings on leaves",
                    "Yellowing and dropping of lower leaves",
                    "Stem lesions with dark centers",
                    "Fruit rot near stem end"
                ],
                "causes": ["High humidity", "Warm temperatures", "Poor nutrition"],
                "affected_parts": ["leaves", "stems", "fruit"],
                "economic_impact": "20-30% yield reduction common"
            },
            "Tomato___Late_blight": {
                "scientific_name": "Phytophthora infestans",
                "severity": "very_high",
                "symptoms": [
                    "Water-soaked lesions on leaves",
                    "White fuzzy growth on leaf undersides",
                    "Brown to black lesions on stems",
                    "Firm brown rot on fruit"
                ],
                "causes": ["Cool, wet weather", "High humidity", "Poor air circulation"],
                "affected_parts": ["leaves", "stems", "fruit"],
                "economic_impact": "Can destroy entire crop within days"
            },
            "Corn_(maize)___Northern_Leaf_Blight": {
                "scientific_name": "Exserohilum turcicum",
                "severity": "moderate",
                "symptoms": [
                    "Elliptical gray-green lesions on leaves",
                    "Lesions turn tan to brown with age",
                    "Reduced photosynthesis",
                    "Premature leaf death"
                ],
                "causes": ["High humidity", "Moderate temperatures", "Dense planting"],
                "affected_parts": ["leaves"],
                "economic_impact": "10-40% yield loss possible"
            },
            "Potato___Late_blight": {
                "scientific_name": "Phytophthora infestans",
                "severity": "very_high",
                "symptoms": [
                    "Dark water-soaked lesions on leaves",
                    "White sporulation on leaf undersides",
                    "Brown rot in tubers",
                    "Rapid plant collapse"
                ],
                "causes": ["Cool, wet conditions", "High humidity", "Poor drainage"],
                "affected_parts": ["leaves", "stems", "tubers"],
                "economic_impact": "Historic cause of Irish Potato Famine"
            }
        }
    
    def _initialize_treatment_protocols(self) -> Dict[str, Dict[str, Any]]:
        """Initialize treatment protocols for different diseases"""
        return {
            "Apple___Apple_scab": {
                "immediate_actions": [
                    "Remove and destroy infected leaves and fruit",
                    "Improve air circulation by pruning",
                    "Apply fungicide spray (Captan or Myclobutanil)"
                ],
                "chemical_treatments": [
                    {
                        "name": "Captan 50WP",
                        "dosage": "2-3 grams per liter",
                        "frequency": "Every 7-14 days",
                        "timing": "Start at bud break"
                    },
                    {
                        "name": "Myclobutanil",
                        "dosage": "0.5-1 ml per liter",
                        "frequency": "Every 14 days",
                        "timing": "Preventive application"
                    }
                ],
                "organic_treatments": [
                    "Neem oil spray (5ml per liter)",
                    "Baking soda solution (5g per liter)",
                    "Copper sulfate spray"
                ],
                "cultural_practices": [
                    "Plant resistant varieties",
                    "Ensure proper spacing",
                    "Remove fallen leaves",
                    "Avoid overhead irrigation"
                ]
            },
            "Tomato___Late_blight": {
                "immediate_actions": [
                    "Remove infected plants immediately",
                    "Improve drainage and air circulation",
                    "Apply copper-based fungicide"
                ],
                "chemical_treatments": [
                    {
                        "name": "Copper Oxychloride",
                        "dosage": "2-3 grams per liter",
                        "frequency": "Every 5-7 days",
                        "timing": "At first sign of disease"
                    },
                    {
                        "name": "Metalaxyl + Mancozeb",
                        "dosage": "2 grams per liter",
                        "frequency": "Every 7-10 days",
                        "timing": "Preventive application"
                    }
                ],
                "organic_treatments": [
                    "Bordeaux mixture spray",
                    "Potassium bicarbonate solution",
                    "Milk spray (1:10 ratio with water)"
                ],
                "cultural_practices": [
                    "Use certified disease-free seeds",
                    "Rotate crops (3-4 year cycle)",
                    "Avoid overhead watering",
                    "Stake plants for better air circulation"
                ]
            }
        }
    
    def _initialize_preventive_measures(self) -> Dict[str, List[str]]:
        """Initialize preventive measures for different crop types"""
        return {
            "general": [
                "Use certified disease-free seeds and planting material",
                "Implement proper crop rotation (3-4 years)",
                "Maintain optimal plant spacing for air circulation",
                "Practice field sanitation and remove crop debris",
                "Monitor weather conditions regularly",
                "Use drip irrigation instead of overhead watering",
                "Apply balanced fertilization to maintain plant health",
                "Scout fields regularly for early disease detection"
            ],
            "apple": [
                "Plant resistant varieties when available",
                "Prune trees for good air circulation",
                "Remove mummified fruit and fallen leaves",
                "Apply dormant season copper sprays",
                "Avoid working in wet orchards"
            ],
            "tomato": [
                "Use resistant varieties when available",
                "Stake or cage plants for support",
                "Mulch around plants to prevent soil splash",
                "Remove lower leaves touching the ground",
                "Avoid working with wet plants"
            ],
            "corn": [
                "Plant resistant hybrids",
                "Avoid dense planting",
                "Control weeds to improve air circulation",
                "Apply balanced nitrogen fertilization",
                "Rotate with non-host crops"
            ]
        }
    
    # TODO: Treatment recommendations - To be implemented in future update
    # def get_treatment_recommendations(self, disease_name: str, severity: str = "moderate") -> Dict[str, Any]:
    #     """
    #     Get comprehensive treatment recommendations for a detected disease
    #     
    #     Args:
    #         disease_name: Name of the detected disease
    #         severity: Severity level of the infection
    #         
    #     Returns:
    #         Dictionary with detailed treatment recommendations
    #     """
    #     try:
    #         # Clean disease name
    #         clean_disease = disease_name.replace("___", "_").strip()
    #         
    #         # Get disease information
    #         disease_info = self.disease_database.get(disease_name, {})
    #         treatment_info = self.treatment_protocols.get(disease_name, {})
    #         
    #         # Determine crop type
    #         crop_type = self._extract_crop_type(disease_name)
    #         
    #         recommendations = {
    #             "disease_info": {
    #                 "name": clean_disease,
    #                 "scientific_name": disease_info.get("scientific_name", "Unknown"),
    #                 "severity_level": disease_info.get("severity", "moderate"),
    #                 "symptoms": disease_info.get("symptoms", []),
    #                 "causes": disease_info.get("causes", []),
    #                 "economic_impact": disease_info.get("economic_impact", "Variable impact")
    #             },
    #             "immediate_actions": treatment_info.get("immediate_actions", [
    #                 "Remove infected plant parts",
    #                 "Improve air circulation",
    #                 "Apply appropriate fungicide"
    #             ]),
    #             "treatment_options": {
    #                 "chemical": treatment_info.get("chemical_treatments", []),
    #                 "organic": treatment_info.get("organic_treatments", [
    #                     "Neem oil spray",
    #                     "Copper-based fungicide",
    #                     "Baking soda solution"
    #                 ]),
    #                 "cultural": treatment_info.get("cultural_practices", [])
    #             },
    #             "preventive_measures": self._get_preventive_measures(crop_type),
    #             "monitoring_schedule": self._generate_monitoring_schedule(),
    #             "follow_up_actions": [
    #                 "Monitor treated areas daily for 1 week",
    #                 "Reapply treatments as recommended",
    #                 "Document treatment effectiveness",
    #                 "Adjust treatment plan if needed"
    #             ],
    #             "emergency_contacts": {
    #                 "agricultural_extension": "Contact local agricultural extension office",
    #                 "plant_pathologist": "Consult certified plant pathologist if severe",
    #                 "pesticide_dealer": "Contact licensed pesticide dealer for products"
    #             }
    #         }
    #         
    #         return recommendations
    #         
    #     except Exception as e:
    #         logger.error(f"Error generating treatment recommendations: {str(e)}")
    #         return self._get_default_recommendations()
    
    def _extract_crop_type(self, disease_name: str) -> str:
        """Extract crop type from disease name"""
        if "Apple" in disease_name:
            return "apple"
        elif "Tomato" in disease_name:
            return "tomato"
        elif "Corn" in disease_name or "maize" in disease_name:
            return "corn"
        elif "Potato" in disease_name:
            return "potato"
        elif "Grape" in disease_name:
            return "grape"
        else:
            return "general"
    
    def _get_preventive_measures(self, crop_type: str) -> List[str]:
        """Get preventive measures for specific crop type"""
        general_measures = self.preventive_measures.get("general", [])
        crop_specific = self.preventive_measures.get(crop_type, [])
        return general_measures + crop_specific
    
    def _generate_monitoring_schedule(self) -> Dict[str, str]:
        """Generate monitoring schedule for disease management"""
        return {
            "daily": "Visual inspection of treated areas",
            "weekly": "Comprehensive field scouting",
            "bi_weekly": "Weather monitoring and disease forecasting",
            "monthly": "Treatment effectiveness evaluation"
        }
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Get default recommendations when specific disease info is not available"""
        return {
            "disease_info": {
                "name": "Unknown Disease",
                "severity_level": "moderate",
                "symptoms": ["Consult plant pathologist for proper identification"],
                "causes": ["Various factors including weather, pathogens, and stress"],
                "economic_impact": "Variable depending on disease type and severity"
            },
            "immediate_actions": [
                "Isolate affected plants",
                "Take samples for professional diagnosis",
                "Improve growing conditions",
                "Contact agricultural extension service"
            ],
            "treatment_options": {
                "chemical": ["Consult with agricultural extension for appropriate fungicides"],
                "organic": ["Neem oil", "Copper sulfate", "Baking soda solution"],
                "cultural": ["Improve air circulation", "Adjust watering practices", "Remove infected material"]
            },
            "preventive_measures": self.preventive_measures["general"],
            "monitoring_schedule": self._generate_monitoring_schedule(),
            "follow_up_actions": [
                "Get professional disease diagnosis",
                "Implement recommended treatments",
                "Monitor treatment effectiveness",
                "Adjust management practices"
            ]
        }
    
    def predict_disease_outbreak_risk(self, environmental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict disease outbreak risk based on environmental conditions
        
        Args:
            environmental_data: Dictionary with weather and environmental parameters
            
        Returns:
            Dictionary with outbreak risk assessment and recommendations
        """
        try:
            temperature = environmental_data.get("temperature", 25)
            humidity = environmental_data.get("humidity", 60)
            rainfall = environmental_data.get("rainfall", 0)
            wind_speed = environmental_data.get("wind_speed", 5)
            crop_type = environmental_data.get("crop_type", "general")
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(temperature, humidity, rainfall, wind_speed)
            
            # Determine overall risk level
            overall_risk = self._determine_risk_level(risk_factors)
            
            # Get specific disease risks
            disease_risks = self._assess_specific_disease_risks(
                temperature, humidity, rainfall, crop_type
            )
            
            return {
                "overall_risk": overall_risk,
                "risk_level": self._get_risk_category(overall_risk),
                "environmental_factors": {
                    "temperature_risk": risk_factors["temperature"],
                    "humidity_risk": risk_factors["humidity"],
                    "rainfall_risk": risk_factors["rainfall"],
                    "wind_risk": risk_factors["wind"]
                },
                "specific_disease_risks": disease_risks,
                "recommendations": self._get_outbreak_recommendations(overall_risk, disease_risks),
                "monitoring_priority": self._get_monitoring_priority(overall_risk),
                "forecast_period": "Next 7-14 days"
            }
            
        except Exception as e:
            logger.error(f"Error predicting disease outbreak risk: {str(e)}")
            return {"error": "Unable to assess disease outbreak risk"}
    
    def _calculate_risk_factors(self, temp: float, humidity: float, rainfall: float, wind: float) -> Dict[str, float]:
        """Calculate individual risk factors"""
        # Temperature risk (optimal disease range 15-25°C)
        temp_risk = 0.0
        if 15 <= temp <= 25:
            temp_risk = 0.8
        elif 10 <= temp < 15 or 25 < temp <= 30:
            temp_risk = 0.5
        else:
            temp_risk = 0.2
        
        # Humidity risk (high risk above 80%)
        humidity_risk = min(humidity / 100, 1.0)
        if humidity > 80:
            humidity_risk = 0.9
        elif humidity > 60:
            humidity_risk = 0.6
        else:
            humidity_risk = 0.3
        
        # Rainfall risk (moderate rain increases risk)
        rainfall_risk = 0.0
        if rainfall > 50:
            rainfall_risk = 0.9
        elif rainfall > 20:
            rainfall_risk = 0.7
        elif rainfall > 5:
            rainfall_risk = 0.4
        else:
            rainfall_risk = 0.1
        
        # Wind risk (low wind increases risk)
        wind_risk = 0.0
        if wind < 2:
            wind_risk = 0.8
        elif wind < 5:
            wind_risk = 0.5
        else:
            wind_risk = 0.2
        
        return {
            "temperature": temp_risk,
            "humidity": humidity_risk,
            "rainfall": rainfall_risk,
            "wind": wind_risk
        }
    
    def _determine_risk_level(self, risk_factors: Dict[str, float]) -> float:
        """Determine overall risk level from individual factors"""
        weights = {
            "temperature": 0.2,
            "humidity": 0.4,
            "rainfall": 0.3,
            "wind": 0.1
        }
        
        weighted_risk = sum(
            risk_factors[factor] * weights[factor]
            for factor in weights
        )
        
        return min(weighted_risk, 1.0)
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to category"""
        if risk_score >= 0.8:
            return "Very High"
        elif risk_score >= 0.6:
            return "High"
        elif risk_score >= 0.4:
            return "Moderate"
        elif risk_score >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _assess_specific_disease_risks(self, temp: float, humidity: float, rainfall: float, crop_type: str) -> List[Dict[str, Any]]:
        """Assess risks for specific diseases based on conditions"""
        disease_risks = []
        
        # Late blight risk (cool, wet conditions)
        if temp < 25 and humidity > 80 and rainfall > 10:
            disease_risks.append({
                "disease": "Late Blight",
                "risk_level": "High",
                "probability": 0.85,
                "conditions": "Cool, wet weather favors development"
            })
        
        # Early blight risk (warm, humid conditions)
        if 20 <= temp <= 30 and humidity > 70:
            disease_risks.append({
                "disease": "Early Blight",
                "risk_level": "Moderate",
                "probability": 0.65,
                "conditions": "Warm, humid conditions present"
            })
        
        # Powdery mildew risk (moderate temp, high humidity, low rainfall)
        if 15 <= temp <= 25 and humidity > 75 and rainfall < 5:
            disease_risks.append({
                "disease": "Powdery Mildew",
                "risk_level": "Moderate",
                "probability": 0.60,
                "conditions": "High humidity with dry conditions"
            })
        
        return disease_risks
    
    def _get_outbreak_recommendations(self, risk_level: float, disease_risks: List[Dict]) -> List[str]:
        """Get recommendations based on outbreak risk"""
        recommendations = []
        
        if risk_level >= 0.8:
            recommendations.extend([
                "Implement immediate preventive fungicide applications",
                "Increase field monitoring to daily inspections",
                "Prepare emergency treatment protocols",
                "Consider harvesting early if crop is mature"
            ])
        elif risk_level >= 0.6:
            recommendations.extend([
                "Apply preventive treatments as scheduled",
                "Monitor fields every 2-3 days",
                "Prepare treatment materials",
                "Improve field drainage if possible"
            ])
        elif risk_level >= 0.4:
            recommendations.extend([
                "Continue regular monitoring schedule",
                "Ensure treatment materials are available",
                "Monitor weather forecasts closely",
                "Maintain good cultural practices"
            ])
        else:
            recommendations.extend([
                "Continue routine field inspections",
                "Maintain preventive cultural practices",
                "Monitor weather conditions",
                "Keep treatment materials ready"
            ])
        
        return recommendations
    
    def _get_monitoring_priority(self, risk_level: float) -> str:
        """Determine monitoring priority based on risk level"""
        if risk_level >= 0.8:
            return "Critical - Daily monitoring required"
        elif risk_level >= 0.6:
            return "High - Monitor every 2-3 days"
        elif risk_level >= 0.4:
            return "Moderate - Weekly monitoring"
        else:
            return "Low - Routine monitoring sufficient"