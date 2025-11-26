"""
Soil Health Router
API endpoints for soil health analysis and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
import numpy as np
from .auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class SoilAnalysisRequest(BaseModel):
    """Request model for soil analysis"""
    nitrogen: float = Field(..., ge=0, le=200, description="Nitrogen content (kg/ha)")
    phosphorus: float = Field(..., ge=0, le=150, description="Phosphorus content (kg/ha)")
    potassium: float = Field(..., ge=0, le=300, description="Potassium content (kg/ha)")
    ph: float = Field(..., ge=3.0, le=10.0, description="Soil pH level")
    organic_matter: float = Field(default=3.0, ge=0, le=20, description="Organic matter percentage")
    moisture: float = Field(default=30, ge=0, le=100, description="Soil moisture percentage")
    temperature: float = Field(default=25, ge=-10, le=50, description="Soil temperature (°C)")

class SoilHealthResponse(BaseModel):
    """Response model for soil health analysis"""
    health_status: str
    overall_score: float
    recommendations: List[str]
    nutrient_analysis: Dict[str, Any]
    deficiencies: List[str]
    improvements: List[str]

class FertilizerPredictionRequest(BaseModel):
    """Request model for fertilizer prediction"""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    moisture: float = Field(..., description="Soil moisture percentage")
    soil_type: str = Field(..., description="Soil type (Sandy, Loamy, Black, Red, Clayey)")
    crop_type: str = Field(..., description="Crop type (Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, Ground Nuts)")
    nitrogen: float = Field(..., description="Nitrogen content (kg/ha)")
    potassium: float = Field(..., description="Potassium content (kg/ha)")
    phosphorous: float = Field(..., description="Phosphorous content (kg/ha)")

# Fertilizer name mapping from model prediction integers to actual names
FERTILIZER_MAPPING = {
    0: "10-26-26",
    1: "14-35-14", 
    2: "17-17-17",
    3: "20-20",
    4: "28-28",
    5: "DAP",
    6: "Urea"
}

class FertilizerPredictionResponse(BaseModel):
    """Response model for fertilizer prediction"""
    recommended_fertilizer: str
    confidence: float
    soil_analysis: Dict[str, Any]
    cost_estimate: Dict[str, Any]
    application_guidelines: Dict[str, Any]

@router.get("/current")
async def get_current_soil_data():
    """Get current soil health data"""
    try:
        # Return mock soil data for now
        current_data = {
            "nitrogen": 45.0,
            "phosphorus": 28.0,
            "potassium": 35.0,
            "ph": 6.8,
            "organic_matter": 3.2,
            "moisture": 32.0,
            "temperature": 24.0,
            "health_status": "Good",
            "last_updated": datetime.now().isoformat(),
            "location": "Farm Location"
        }
        
        logger.info("Successfully retrieved current soil data")
        return current_data
        
    except Exception as e:
        logger.error(f"Error fetching current soil data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch soil data")

@router.post("/analyze", response_model=SoilHealthResponse)
async def analyze_soil_health(soil_data: SoilAnalysisRequest):
    """Analyze soil health based on provided parameters"""
    try:
        logger.info(f"Analyzing soil health for data: {soil_data.dict()}")
        
        # Calculate overall health score
        ph_score = calculate_ph_score(soil_data.ph)
        nutrient_score = calculate_nutrient_score(soil_data.nitrogen, soil_data.phosphorus, soil_data.potassium)
        organic_score = min(soil_data.organic_matter / 5.0 * 100, 100)
        moisture_score = calculate_moisture_score(soil_data.moisture)
        
        overall_score = (ph_score + nutrient_score + organic_score + moisture_score) / 4
        
        # Determine health status
        if overall_score >= 80:
            health_status = "Excellent"
        elif overall_score >= 65:
            health_status = "Good"
        elif overall_score >= 50:
            health_status = "Fair"
        else:
            health_status = "Poor"
        
        # Generate recommendations
        recommendations = generate_recommendations(soil_data, overall_score)
        
        # Identify deficiencies
        deficiencies = identify_deficiencies(soil_data)
        
        # Suggest improvements
        improvements = suggest_improvements(soil_data, overall_score)
        
        # Nutrient analysis
        nutrient_analysis = {
            "nitrogen": {
                "value": soil_data.nitrogen,
                "status": "optimal" if 40 <= soil_data.nitrogen <= 60 else "needs_attention",
                "recommendation": "Maintain current levels" if 40 <= soil_data.nitrogen <= 60 else "Adjust nitrogen levels"
            },
            "phosphorus": {
                "value": soil_data.phosphorus,
                "status": "optimal" if 25 <= soil_data.phosphorus <= 45 else "needs_attention",
                "recommendation": "Maintain current levels" if 25 <= soil_data.phosphorus <= 45 else "Adjust phosphorus levels"
            },
            "potassium": {
                "value": soil_data.potassium,
                "status": "optimal" if 30 <= soil_data.potassium <= 50 else "needs_attention",
                "recommendation": "Maintain current levels" if 30 <= soil_data.potassium <= 50 else "Adjust potassium levels"
            },
            "ph": {
                "value": soil_data.ph,
                "status": "optimal" if 6.0 <= soil_data.ph <= 7.5 else "needs_attention",
                "recommendation": "Maintain current pH" if 6.0 <= soil_data.ph <= 7.5 else "Adjust soil pH"
            }
        }
        
        response = SoilHealthResponse(
            health_status=health_status,
            overall_score=round(overall_score, 1),
            recommendations=recommendations,
            nutrient_analysis=nutrient_analysis,
            deficiencies=deficiencies,
            improvements=improvements
        )
        
        logger.info(f"Soil analysis completed with status: {health_status}, score: {overall_score}")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing soil health: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze soil health")

def calculate_ph_score(ph: float) -> float:
    """Calculate pH score (0-100)"""
    if 6.0 <= ph <= 7.5:
        return 100
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        return 80
    elif 5.0 <= ph < 5.5 or 8.0 < ph <= 8.5:
        return 60
    else:
        return 40

def calculate_nutrient_score(nitrogen: float, phosphorus: float, potassium: float) -> float:
    """Calculate nutrient score (0-100)"""
    n_score = 100 if 40 <= nitrogen <= 60 else max(0, 100 - abs(nitrogen - 50) * 2)
    p_score = 100 if 25 <= phosphorus <= 45 else max(0, 100 - abs(phosphorus - 35) * 3)
    k_score = 100 if 30 <= potassium <= 50 else max(0, 100 - abs(potassium - 40) * 2.5)
    
    return (n_score + p_score + k_score) / 3

def calculate_moisture_score(moisture: float) -> float:
    """Calculate moisture score (0-100)"""
    if 25 <= moisture <= 35:
        return 100
    elif 20 <= moisture < 25 or 35 < moisture <= 40:
        return 80
    elif 15 <= moisture < 20 or 40 < moisture <= 45:
        return 60
    else:
        return 40

def generate_recommendations(soil_data: SoilAnalysisRequest, score: float) -> List[str]:
    """Generate soil health recommendations"""
    recommendations = []
    
    if soil_data.ph < 6.0:
        recommendations.append("Apply lime to increase soil pH")
    elif soil_data.ph > 7.5:
        recommendations.append("Apply sulfur or organic matter to lower soil pH")
    
    if soil_data.nitrogen < 40:
        recommendations.append("Apply nitrogen-rich fertilizer or compost")
    elif soil_data.nitrogen > 60:
        recommendations.append("Reduce nitrogen application to prevent leaching")
    
    if soil_data.phosphorus < 25:
        recommendations.append("Apply phosphorus fertilizer")
    
    if soil_data.potassium < 30:
        recommendations.append("Apply potassium fertilizer")
    
    if soil_data.organic_matter < 2.5:
        recommendations.append("Increase organic matter through compost or cover crops")
    
    if score < 65:
        recommendations.append("Consider soil testing every 6 months")
        recommendations.append("Implement crop rotation to improve soil health")
    
    return recommendations

def identify_deficiencies(soil_data: SoilAnalysisRequest) -> List[str]:
    """Identify nutrient deficiencies"""
    deficiencies = []
    
    if soil_data.nitrogen < 40:
        deficiencies.append("Nitrogen deficiency detected")
    
    if soil_data.phosphorus < 25:
        deficiencies.append("Phosphorus deficiency detected")
    
    if soil_data.potassium < 30:
        deficiencies.append("Potassium deficiency detected")
    
    if soil_data.organic_matter < 2.0:
        deficiencies.append("Low organic matter content")
    
    return deficiencies

def suggest_improvements(soil_data: SoilAnalysisRequest, score: float) -> List[str]:
    """Suggest soil improvements"""
    improvements = []
    
    if score < 80:
        improvements.append("Regular soil testing and monitoring")
        improvements.append("Balanced fertilization program")
        improvements.append("Organic matter enhancement")
    
    if soil_data.moisture < 25:
        improvements.append("Improve water retention with organic matter")
        improvements.append("Consider mulching to retain moisture")
    
    return improvements

# Fertilizer Recommendation Models
class FertilizerRecommendationRequest(BaseModel):
    """Request model for fertilizer recommendation"""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    moisture: float = Field(..., description="Soil moisture percentage")
    soil_type: str = Field(..., description="Soil type (Sandy, Loamy, Black, Red, Clayey)")
    crop_type: str = Field(..., description="Crop type (Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, Ground Nuts)")
    nitrogen: float = Field(..., description="Nitrogen content (kg/ha)")
    potassium: float = Field(..., description="Potassium content (kg/ha)")
    phosphorous: float = Field(..., description="Phosphorous content (kg/ha)")

class FertilizerRecommendationResponse(BaseModel):
    """Response model for fertilizer recommendation"""
    recommended_fertilizer: str
    confidence: float
    soil_analysis: Dict[str, Any]
    application_guidelines: List[str]
    cost_estimate: Dict[str, Any]

# Load fertilizer model (global variable for efficiency)
fertilizer_model = None

def load_fertilizer_model():
    """Load the pre-trained fertilizer model"""
    global fertilizer_model
    if fertilizer_model is None:
        try:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fertilizer_model.joblib'))
            if os.path.exists(model_path):
                fertilizer_model = joblib.load(model_path)
                logger.info("Fertilizer model loaded successfully")
            else:
                logger.error(f"Fertilizer model not found at {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading fertilizer model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load fertilizer model")
    return fertilizer_model

@router.post("/fertilizer-recommendation", response_model=FertilizerRecommendationResponse)
async def get_fertilizer_recommendation(
    request: FertilizerRecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Get fertilizer recommendation based on soil and crop parameters
    Requires JWT authentication
    """
    try:
        # Load model
        model = load_fertilizer_model()
        
        # Encode categorical variables
        soil_type_mapping = {
            "Sandy": 0,
            "Loamy": 1, 
            "Black": 2,
            "Red": 3,
            "Clayey": 4
        }
        
        crop_type_mapping = {
            "Maize": 0,
            "Sugarcane": 1,
            "Cotton": 2,
            "Tobacco": 3,
            "Paddy": 4,
            "Barley": 5,
            "Wheat": 6,
            "Millets": 7,
            "Oil seeds": 8,
            "Pulses": 9,
            "Ground Nuts": 10
        }
        
        # Encode categorical variables
        soil_type_encoded = soil_type_mapping.get(request.soil_type, 1)  # Default to Loamy
        crop_type_encoded = crop_type_mapping.get(request.crop_type, 0)  # Default to Maize
        
        # Prepare input data for prediction
        # Based on the dataset structure: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
        input_data = pd.DataFrame({
            'Temparature': [request.temperature],  # Note: keeping original spelling from dataset
            'Humidity ': [request.humidity],       # Note: keeping original spacing from dataset
            'Moisture': [request.moisture],
            'Soil Type': [soil_type_encoded],
            'Crop Type': [crop_type_encoded],
            'Nitrogen': [request.nitrogen],
            'Potassium': [request.potassium],
            'Phosphorous': [request.phosphorous]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Map integer prediction to fertilizer name using the mapping
        fertilizer_name = FERTILIZER_MAPPING.get(prediction, f"Unknown Fertilizer ({prediction})")
        
        # Calculate confidence (mock implementation - in real scenario, use model's predict_proba if available)
        confidence = 0.85  # Default confidence
        
        # Generate application guidelines based on fertilizer type
        guidelines = generate_application_guidelines(fertilizer_name, request)
        
        # Generate cost estimate
        cost_estimate = generate_cost_estimate(fertilizer_name)
        
        # Soil analysis summary
        soil_analysis = {
            "nutrient_status": {
                "nitrogen": "adequate" if request.nitrogen > 20 else "deficient",
                "phosphorous": "adequate" if request.phosphorous > 15 else "deficient", 
                "potassium": "adequate" if request.potassium > 10 else "deficient"
            },
            "soil_condition": request.soil_type,
            "moisture_level": "optimal" if 30 <= request.moisture <= 70 else "needs_attention"
        }
        
        logger.info(f"Fertilizer recommendation generated for user {current_user.get('email', 'unknown')}: {prediction} -> {fertilizer_name}")
        
        return FertilizerRecommendationResponse(
            recommended_fertilizer=fertilizer_name,
            confidence=confidence,
            soil_analysis=soil_analysis,
            application_guidelines=guidelines,
            cost_estimate=cost_estimate
        )
        
    except Exception as e:
        logger.error(f"Error generating fertilizer recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation: {str(e)}")

def generate_application_guidelines_old(fertilizer_type: str) -> List[str]:
    """Generate application guidelines based on fertilizer type"""
    guidelines_map = {
        "Urea": [
            "Apply in split doses - 50% at sowing, 50% at tillering stage",
            "Apply during cool hours (early morning or evening)",
            "Ensure adequate soil moisture before application",
            "Mix with soil immediately after application to prevent nitrogen loss"
        ],
        "DAP": [
            "Apply as basal dose at the time of sowing",
            "Place 2-3 cm below and beside the seed",
            "Ensure good soil contact for better nutrient uptake",
            "Suitable for all crops, especially at early growth stages"
        ],
        "28-28": [
            "Apply as basal fertilizer before sowing",
            "Suitable for crops requiring balanced NPK nutrition",
            "Apply in furrows and mix with soil",
            "Follow up with nitrogen top-dressing if needed"
        ],
        "17-17-17": [
            "Complete balanced fertilizer for all growth stages",
            "Apply 1/3 at sowing, 1/3 at vegetative stage, 1/3 at reproductive stage",
            "Ensure uniform distribution across the field",
            "Water immediately after application for better dissolution"
        ],
        "20-20": [
            "Balanced phosphorus and potassium fertilizer",
            "Apply as basal dose before sowing",
            "Particularly beneficial for root development",
            "Combine with nitrogen source for complete nutrition"
        ],
        "14-35-14": [
            "High phosphorus fertilizer for root development",
            "Apply at sowing time for maximum benefit",
            "Excellent for crops in early growth stages",
            "Ensure proper placement near root zone"
        ],
        "10-26-26": [
            "High potassium fertilizer for fruit/grain development",
            "Apply during reproductive stage for better yield",
            "Particularly beneficial for fruit crops and cereals",
            "Apply in split doses for better utilization"
        ]
    }
    
    return guidelines_map.get(fertilizer_type, [
        "Follow manufacturer's recommended application rates",
        "Apply during appropriate growth stages",
        "Ensure proper soil moisture conditions",
        "Monitor crop response and adjust as needed"
    ])

def generate_cost_estimate(fertilizer_type: str) -> Dict[str, Any]:
    """Generate cost estimate for fertilizer application"""
    # Mock cost data - in real implementation, fetch from market API
    cost_per_kg = {
        "Urea": 25.0,
        "DAP": 45.0,
        "28-28": 35.0,
        "17-17-17": 40.0,
        "20-20": 38.0,
        "14-35-14": 42.0,
        "10-26-26": 44.0
    }
    
    base_cost = cost_per_kg.get(fertilizer_type, 35.0)
    recommended_quantity = 50  # kg per acre (mock value)
    
    return {
        "fertilizer_type": fertilizer_type,
        "cost_per_kg": base_cost,
        "recommended_quantity_per_acre": recommended_quantity,
        "total_cost_per_acre": base_cost * recommended_quantity,
        "currency": "INR",
        "last_updated": datetime.now().isoformat()
    }

@router.post("/fertilizer-recommendation", response_model=FertilizerPredictionResponse)
async def predict_fertilizer(request: FertilizerPredictionRequest):
    """Predict fertilizer recommendation based on soil and environmental parameters"""
    try:
        logger.info(f"Processing fertilizer prediction request: {request.dict()}")
        
        # Load the fertilizer prediction model
        model_path = "D:\\new-folder1\\AI-Crop-Yield-Prediction\\agrismart\\app\\ml_models\\saved_models\\fertilizer_model.joblib"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}, using mock prediction")
            return get_mock_fertilizer_prediction(request)
        
        try:
            model = joblib.load(model_path)
            logger.info("Fertilizer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return get_mock_fertilizer_prediction(request)
        
        # Prepare input data for the model
        # Based on the dataset columns: Temparature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
        input_data = prepare_fertilizer_input(request)
        
        try:
            # Make prediction
            prediction = model.predict([input_data])[0]
            confidence = 0.85  # Mock confidence for now
            
            # Map integer prediction to fertilizer name
            fertilizer_name = FERTILIZER_MAPPING.get(prediction, f"Unknown Fertilizer ({prediction})")
            
            logger.info(f"Fertilizer prediction successful: {prediction} -> {fertilizer_name}")
            
            # Generate comprehensive response
            response = generate_fertilizer_response(fertilizer_name, request, confidence)
            
            # Ensure the response contains the mapped fertilizer name, not the integer
            response.recommended_fertilizer = fertilizer_name
            
            return response
            
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            return get_mock_fertilizer_prediction(request)
            
    except Exception as e:
        logger.error(f"Error in fertilizer prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to predict fertilizer recommendation")

def prepare_fertilizer_input(request: FertilizerPredictionRequest) -> List[float]:
    """Prepare input data for the fertilizer prediction model"""
    
    # Soil type encoding (based on common values in dataset)
    soil_type_mapping = {
        "Sandy": 0,
        "Loamy": 1, 
        "Black": 2,
        "Red": 3,
        "Clayey": 4
    }
    
    # Crop type encoding (based on common values in dataset)
    crop_type_mapping = {
        "Maize": 0,
        "Sugarcane": 1,
        "Cotton": 2,
        "Tobacco": 3,
        "Paddy": 4,
        "Barley": 5,
        "Wheat": 6,
        "Millets": 7,
        "Oil seeds": 8,
        "Pulses": 9,
        "Ground Nuts": 10
    }
    
    # Encode categorical variables
    soil_type_encoded = soil_type_mapping.get(request.soil_type, 1)  # Default to Loamy
    crop_type_encoded = crop_type_mapping.get(request.crop_type, 0)  # Default to Maize
    
    # Return input array in the order expected by the model
    # [Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous]
    return [
        request.temperature,
        request.humidity,
        request.moisture,
        soil_type_encoded,
        crop_type_encoded,
        request.nitrogen,
        request.potassium,
        request.phosphorous
    ]

def generate_fertilizer_response(prediction: str, request: FertilizerPredictionRequest, confidence: float) -> FertilizerPredictionResponse:
    """Generate comprehensive fertilizer recommendation response"""
    
    # Analyze soil nutrient status
    soil_analysis = analyze_soil_nutrients(request)
    
    # Calculate cost estimate
    cost_estimate = calculate_fertilizer_cost(prediction)
    
    # Generate application guidelines
    application_guidelines = generate_application_guidelines(prediction, request)
    
    return FertilizerPredictionResponse(
        recommended_fertilizer=prediction,
        confidence=confidence,
        soil_analysis=soil_analysis,
        cost_estimate=cost_estimate,
        application_guidelines=application_guidelines
    )

def analyze_soil_nutrients(request: FertilizerPredictionRequest) -> Dict[str, Any]:
    """Analyze soil nutrient status"""
    
    # Define optimal ranges for nutrients
    optimal_ranges = {
        "nitrogen": {"low": 40, "high": 80},
        "phosphorous": {"low": 30, "high": 60},
        "potassium": {"low": 40, "high": 80}
    }
    
    def get_nutrient_status(value: float, nutrient: str) -> str:
        ranges = optimal_ranges[nutrient]
        if value < ranges["low"]:
            return "Low"
        elif value > ranges["high"]:
            return "High"
        else:
            return "Optimal"
    
    return {
        "nitrogen_status": get_nutrient_status(request.nitrogen, "nitrogen"),
        "phosphorous_status": get_nutrient_status(request.phosphorous, "phosphorous"),
        "potassium_status": get_nutrient_status(request.potassium, "potassium"),
        "soil_type": request.soil_type,
        "moisture_level": "Optimal" if 30 <= request.moisture <= 70 else ("Low" if request.moisture < 30 else "High"),
        "temperature_condition": "Optimal" if 15 <= request.temperature <= 35 else ("Low" if request.temperature < 15 else "High")
    }

def calculate_fertilizer_cost(fertilizer_name: str) -> Dict[str, Any]:
    """Calculate estimated cost for the recommended fertilizer"""
    
    # Mock cost data (in INR per kg)
    fertilizer_costs = {
        "Urea": 6.5,
        "DAP": 24.0,
        "14-35-14": 22.0,
        "28-28": 18.0,
        "17-17-17": 20.0,
        "20-20": 16.0,
        "10-26-26": 19.0
    }
    
    base_cost = fertilizer_costs.get(fertilizer_name, 15.0)
    recommended_quantity = 50  # kg per acre
    
    return {
        "fertilizer_name": fertilizer_name,
        "cost_per_kg": base_cost,
        "recommended_quantity_per_acre": recommended_quantity,
        "total_cost_per_acre": base_cost * recommended_quantity,
        "currency": "INR",
        "cost_breakdown": {
            "material_cost": base_cost * recommended_quantity * 0.8,
            "application_cost": base_cost * recommended_quantity * 0.2
        }
    }

def generate_application_guidelines(fertilizer_name: str, request: FertilizerPredictionRequest) -> List[str]:
    """Generate application guidelines for the recommended fertilizer"""
    
    guidelines = [
        "Apply during land preparation or at planting",
        "Use broadcast application followed by incorporation",
        "Apply when soil moisture is adequate",
        "Avoid application during heavy rain",
        f"Split application recommended for {request.crop_type}",
        "Apply in evening hours to reduce volatilization",
        "Ensure uniform distribution across the field",
        "Store in dry place away from moisture"
    ]
    
    return guidelines

def get_mock_fertilizer_prediction(request: FertilizerPredictionRequest) -> FertilizerPredictionResponse:
    """Generate mock fertilizer prediction when model is not available"""
    
    # Simple rule-based mock prediction
    if request.nitrogen < 40:
        prediction = "Urea"
    elif request.phosphorous < 30:
        prediction = "DAP"
    elif request.potassium < 40:
        prediction = "10-26-26"
    else:
        prediction = "17-17-17"
    
    return generate_fertilizer_response(prediction, request, 0.75)