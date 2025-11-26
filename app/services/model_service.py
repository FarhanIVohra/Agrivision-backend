"""
Model Service for ML Model Management and Predictions
Handles loading, prediction, and crop mapping for the recommendation system
"""

import joblib
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
from .profit_prediction_service import ProfitPredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """Service class for managing ML model operations"""
    
    def __init__(self):
        self.model = None
        # Define the models directory relative to this file's location
        MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        self.model_path = os.path.join(MODELS_DIR, "crop_recommender.joblib")
        self.is_loaded = False
        
        # Initialize profit prediction service
        self.profit_service = ProfitPredictionService()
        
        # Crop mapping for the crop_recommender.joblib (22 classes)
        # This model returns integer predictions (0-21), so we map integers to crop names
        self.crop_mapping = {
            0: "Apple",
            1: "Banana", 
            2: "Blackgram",
            3: "Chickpea",
            4: "Coconut",
            5: "Coffee",
            6: "Cotton",
            7: "Grapes",
            8: "Jute",
            9: "Kidneybeans",
            10: "Lentil",
            11: "Maize",
            12: "Mango",
            13: "Mothbeans",
            14: "Mungbean",
            15: "Muskmelon",
            16: "Orange",
            17: "Papaya",
            18: "Pigeonpeas",
            19: "Pomegranate",
            20: "Rice",
            21: "Watermelon"
        }
        
        # Expected feature order: [N, P, K, temperature, humidity, ph, rainfall]
        self.feature_names = [
            "nitrogen", "phosphorus", "potassium", 
            "temperature", "humidity", "ph", "rainfall"
        ]
    
    def load_model(self) -> bool:
        """
        Load the ML model from the specified path
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Suppress sklearn version warnings
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            
            # Construct full path
            full_path = self.model_path
            
            if not os.path.exists(full_path):
                logger.error(f"Model file not found at: {full_path}")
                return False
            
            # Load the model
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully from {full_path}")
            logger.info(f"Model type: {type(self.model)}")
            logger.info(f"Expected features: {self.model.n_features_in_}")
            logger.info(f"Available crop classes: {len(self.model.classes_)}")
            logger.info(f"Crop classes: {list(self.model.classes_)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions"""
        return self.is_loaded and self.model is not None
    
    def validate_input(self, features: List[float]) -> Tuple[bool, str]:
        """
        Validate input features
        
        Args:
            features: List of feature values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.is_model_loaded():
            return False, "Model is not loaded"
        
        if len(features) != self.model.n_features_in_:
            return False, f"Expected {self.model.n_features_in_} features, got {len(features)}"
        
        # Check for invalid values
        for i, value in enumerate(features):
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                return False, f"Invalid value for feature {self.feature_names[i]}: {value}"
        
        return True, ""
    
    def predict_single_crop(self, features: List[float]) -> Tuple[str, float]:
        """
        Predict the most suitable crop for given features
        
        Args:
            features: List of feature values [N, P, K, temp, humidity, ph, rainfall]
            
        Returns:
            Tuple of (predicted_crop_name, confidence)
        """
        if not self.is_model_loaded():
            raise ValueError("Model is not loaded")
        
        # Validate input
        is_valid, error_msg = self.validate_input(features)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Suppress warnings for prediction
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        
        # Prepare input array
        input_array = np.array(features).reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.model.predict(input_array)[0]  # This returns an integer (0-21)
        probabilities = self.model.predict_proba(input_array)[0]
        
        # Find the index of the predicted class to get confidence
        predicted_class_index = np.where(self.model.classes_ == prediction)[0][0]
        confidence = probabilities[predicted_class_index]
        
        # Get display name using integer mapping
        display_name = self.crop_mapping.get(int(prediction), f"Unknown Crop {prediction}")
        
        return display_name, float(confidence)
    
    def predict_multiple_crops(self, features: List[float], top_k: int = 5, farm_size: float = 1.0) -> List[Dict[str, Any]]:
        """
        Predict multiple suitable crops with profit-focused recommendations
        
        Args:
            features: List of feature values [N, P, K, temperature, humidity, ph, rainfall]
            top_k: Number of top recommendations to return
            farm_size: Farm size in hectares for profit calculations
            
        Returns:
            List of crop recommendations ranked by profitability
        """
        if not self.is_model_loaded():
            raise ValueError("Model is not loaded")
        
        # Validate input
        is_valid, error_msg = self.validate_input(features)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Suppress warnings for prediction
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        
        # Prepare input array
        input_array = np.array(features).reshape(1, -1)
        
        # Get probabilities for all classes
        probabilities = self.model.predict_proba(input_array)[0]
        
        # Get top k predictions based on ML model suitability
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        # Create basic recommendations from ML model
        ml_recommendations = []
        for i, class_idx in enumerate(top_indices):
            # Get the actual class integer from the model
            actual_crop_id = int(self.model.classes_[class_idx])
            # Use the crop mapping for display names
            crop_name = self.crop_mapping.get(actual_crop_id, f"Unknown Crop {actual_crop_id}")
            
            confidence = float(probabilities[class_idx])
            suitability_score = confidence * 100  # Convert to percentage
            
            ml_recommendations.append({
                "crop_id": actual_crop_id,  # Use the actual integer ID from the model
                "crop_name": crop_name,
                "confidence": confidence,
                "suitability_score": round(suitability_score, 2)
            })
        
        # Prepare environmental factors for profit analysis
        environmental_factors = {
            "nitrogen": features[0],
            "phosphorus": features[1], 
            "potassium": features[2],
            "temperature": features[3],
            "humidity": features[4],
            "ph": features[5],
            "rainfall": features[6]
        }
        
        # Enhance recommendations with profit analysis
        profit_focused_recommendations = self.profit_service.rank_crops_by_profitability(
            ml_recommendations, environmental_factors, farm_size
        )
        
        return profit_focused_recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_model_loaded():
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": str(type(self.model)),
            "n_features": self.model.n_features_in_,
            "n_classes": len(self.model.classes_),
            "feature_names": self.feature_names,
            "available_crops": list(self.crop_mapping.values()),
            "model_path": self.model_path
        }
    
    def prepare_features_from_request(self, request_data: Dict[str, Any]) -> List[float]:
        """
        Prepare feature array from request data
        Handles both flat structure and nested structure from frontend
        
        Args:
            request_data: Dictionary with feature values (flat or nested)
            
        Returns:
            List of features in correct order
        """
        # Check if it's a nested structure from frontend
        if "soilData" in request_data and "weatherData" in request_data:
            # Extract from nested structure
            soil_data = request_data["soilData"]
            weather_data = request_data["weatherData"]
            
            return [
                soil_data.get("nitrogen", 50),  # Default to 50 instead of 0
                soil_data.get("phosphorus", 30),  # Default to 30 instead of 0
                soil_data.get("potassium", 40),  # Default to 40 instead of 0
                # Use temperature from soilData first, then weatherData
                soil_data.get("temperature", weather_data.get("temperature", 25)),  # Default to 25
                # Use humidity from soilData first, then weatherData  
                soil_data.get("humidity", weather_data.get("humidity", 65)),  # Default to 65
                soil_data.get("ph", 6.5),  # Default to 6.5 instead of 0 to avoid validation error
                weather_data.get("rainfall", 100)  # Default to 100 instead of 0
            ]
        else:
            # Handle flat structure (direct API calls)
            return [
                request_data.get("nitrogen", 50),  # Default to 50 instead of 0
                request_data.get("phosphorus", 30),  # Default to 30 instead of 0
                request_data.get("potassium", 40),  # Default to 40 instead of 0
                request_data.get("temperature", 25),  # Default to 25 instead of 0
                request_data.get("humidity", 65),  # Default to 65 instead of 0
                request_data.get("ph", 6.5),  # Default to 6.5 instead of 0 to avoid validation error
                request_data.get("rainfall", 100)  # Default to 100 instead of 0
            ]
    
    def get_prediction_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for predictions
        
        Returns:
            Dictionary with prediction metadata
        """
        return {
            "model_version": "1.0",
            "prediction_time": datetime.utcnow().isoformat() + "Z",
            "feature_count": len(self.feature_names),
            "available_crops": len(self.crop_mapping)
        }