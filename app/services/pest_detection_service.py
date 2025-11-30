"""
Pest Detection Service
Handles crop disease detection using the trained TensorFlow model
Provides image preprocessing, model inference, and prediction results

NOTE: Uses pest_disease_model.h5 as the SINGLE model for all pest disease detection
Updated to use the new lightweight 22-class model for improved performance
"""

import tensorflow as tf
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging
from pathlib import Path
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestDetectionService:
    _instance: Optional['PestDetectionService'] = None
    model: Optional[tf.keras.Model] = None

    def __new__(cls) -> 'PestDetectionService':
        if cls._instance is None:
            print("Initializing PestDetectionService singleton...")
            cls._instance = super(PestDetectionService, cls).__new__(cls)
            MODELS_DIR = Path(__file__).parent.parent.parent / 'models'
            preferred = MODELS_DIR / "pest_detection_model.h5"
            legacy = MODELS_DIR / "pest_disease_model.h5"
            cls.model_path = preferred if preferred.exists() else legacy
            cls.input_shape = (128, 128, 3)
            cls.num_classes = 22
            cls.disease_mapping = {
                0: "Healthy_Leaf", 1: "Apple_Scab", 2: "Apple_Black_Rot",
                3: "Apple_Cedar_Rust", 4: "Cherry_Powdery_Mildew", 5: "Corn_Gray_Leaf_Spot",
                6: "Corn_Common_Rust", 7: "Corn_Northern_Blight", 8: "Grape_Black_Rot",
                9: "Grape_Esca", 10: "Grape_Leaf_Blight", 11: "Peach_Bacterial_Spot",
                12: "Pepper_Bacterial_Spot", 13: "Potato_Early_Blight", 14: "Potato_Late_Blight",
                15: "Strawberry_Leaf_Scorch", 16: "Tomato_Bacterial_Spot", 17: "Tomato_Early_Blight",
                18: "Tomato_Late_Blight", 19: "Tomato_Leaf_Mold", 20: "Tomato_Septoria_Leaf_Spot",
                21: "Tomato_Spider_Mites"
            }
            try:
                tf.config.set_visible_devices([], 'GPU')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            except Exception:
                pass
        return cls._instance
    
    def get_model(self) -> tf.keras.Model:
        """Lazily loads the TensorFlow model and returns it."""
        if self.model is None:
            print(f"⏳ Loading pest detection model from: {self.model_path}")
            if not self.model_path.exists():
                logger.error(f"Model file not found at: {self.model_path}")
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            try:
                self.model = tf.keras.models.load_model(str(self.model_path))
                print("✅ Pest detection model loaded.")
                logger.info(f"Model loaded: {self.model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load pest detection model: {e}")
                raise e
        return self.model
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image data for model inference
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            np.ndarray: Preprocessed image array ready for model input
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.input_shape[0], self.input_shape[1]))
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def preprocess_base64_image(self, base64_image: str) -> np.ndarray:
        """
        Preprocess base64 encoded image for model inference
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            np.ndarray: Preprocessed image array ready for model input
        """
        try:
            # Remove data URL prefix if present
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_image)
            
            return self.preprocess_image(image_bytes)
            
        except Exception as e:
            logger.error(f"Error preprocessing base64 image: {str(e)}")
            raise ValueError(f"Failed to preprocess base64 image: {str(e)}")
    
    def predict_disease(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Predict disease from preprocessed image array with bias mitigation
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            Dict containing simplified prediction results (prediction and confidence only)
        """
        model = self.get_model()
        
        try:
            # Get model predictions
            predictions = model.predict(image_array)
            
            # Apply bias detection and mitigation
            result = self._apply_bias_mitigation(predictions[0])
            
            logger.info(f"Disease prediction completed: {result['prediction']} with {result['confidence']:.2%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error during disease prediction: {str(e)}")
            raise ValueError(f"Failed to predict disease: {str(e)}")
    
    def _apply_bias_mitigation(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Apply bias mitigation strategies to improve prediction reliability
        
        Args:
            predictions: Raw model predictions array
            
        Returns:
            Dict containing mitigated prediction results
        """
        # Get top predictions for analysis
        top_indices = np.argsort(predictions)[-5:][::-1]  # Top 5 predictions
        top_confidences = predictions[top_indices]
        
        predicted_class_id = top_indices[0]
        confidence = float(top_confidences[0])
        
        # Get disease name
        raw_disease_name = self.disease_mapping.get(predicted_class_id, f"Unknown Disease {predicted_class_id}")
        disease_name = self._clean_disease_name(raw_disease_name)
        
        # Bias mitigation strategies
        
        # 1. Confidence threshold check
        if confidence < 0.6:
            return {
                "prediction": "Uncertain - Low Confidence Detection",
                "confidence": round(confidence, 4),
                "status": "low_confidence",
                "alternative_predictions": self._get_alternative_predictions(top_indices, top_confidences)
            }
        
        # 2. Check for known bias patterns
        bias_check = self._detect_bias_patterns(predictions, disease_name, confidence)
        if bias_check["is_biased"]:
            return {
                "prediction": bias_check["adjusted_prediction"],
                "confidence": round(bias_check["adjusted_confidence"], 4),
                "status": "bias_adjusted",
                "original_prediction": disease_name,
                "note": bias_check["reason"]
            }
        
        # 3. Diversity check - if prediction is too certain, provide alternatives
        if confidence > 0.95:
            return {
                "prediction": disease_name,
                "confidence": round(confidence, 4),
                "status": "high_confidence_with_alternatives",
                "alternative_predictions": self._get_alternative_predictions(top_indices, top_confidences),
                "note": "Very high confidence - consider alternative possibilities"
            }
        
        # Normal prediction
        return {
            "prediction": disease_name,
            "confidence": round(confidence, 4),
            "status": "normal"
        }
    
    def _detect_bias_patterns(self, predictions: np.ndarray, disease_name: str, confidence: float) -> Dict[str, Any]:
        """
        Detect known bias patterns and suggest adjustments
        
        Args:
            predictions: Full prediction array
            disease_name: Predicted disease name
            confidence: Prediction confidence
            
        Returns:
            Dict containing bias detection results
        """
        # Known problematic patterns - expanded list
        problematic_predictions = [
            "Cherry - Powdery Mildew",
            "Potato - Healthy",
            "Corn (maize) - Northern Leaf Blight"  # Added based on test results
        ]
        
        # Check if this is a known biased prediction
        if any(prob in disease_name for prob in problematic_predictions):
            # Get top 3 predictions for better alternatives
            sorted_indices = np.argsort(predictions)[::-1]
            
            # More aggressive bias detection - lower threshold
            if confidence > 0.7:  # Lowered from 0.9
                second_best_id = sorted_indices[1]
                third_best_id = sorted_indices[2]
                
                second_confidence = predictions[second_best_id]
                third_confidence = predictions[third_best_id]
                
                second_disease = self._clean_disease_name(
                    self.disease_mapping.get(second_best_id, f"Unknown Disease {second_best_id}")
                )
                third_disease = self._clean_disease_name(
                    self.disease_mapping.get(third_best_id, f"Unknown Disease {third_best_id}")
                )
                
                # If the model is very confident about a biased prediction, provide uncertainty
                if confidence > 0.9:
                    return {
                        "is_biased": True,
                        "adjusted_prediction": f"Uncertain Detection - Possibly {second_disease} or {third_disease}",
                        "adjusted_confidence": float(max(second_confidence, third_confidence)),
                        "reason": f"Model shows strong bias towards {disease_name}. Suggesting alternatives based on secondary predictions."
                    }
                else:
                    return {
                        "is_biased": True,
                        "adjusted_prediction": f"Multiple Possibilities: {disease_name}, {second_disease}, or {third_disease}",
                        "adjusted_confidence": float((confidence + second_confidence + third_confidence) / 3),
                        "reason": f"Detected potential bias pattern. Providing multiple possibilities."
                    }
        
        # Additional check: if any single prediction is too dominant
        max_confidence = np.max(predictions)
        second_max = np.partition(predictions, -2)[-2]
        
        # If there's a huge gap between first and second prediction, it might be bias
        if max_confidence > 0.8 and (max_confidence - second_max) > 0.6:
            sorted_indices = np.argsort(predictions)[::-1]
            second_best_id = sorted_indices[1]
            second_disease = self._clean_disease_name(
                self.disease_mapping.get(second_best_id, f"Unknown Disease {second_best_id}")
            )
            
            return {
                "is_biased": True,
                "adjusted_prediction": f"High Confidence with Uncertainty: Likely {disease_name} but consider {second_disease}",
                "adjusted_confidence": float((max_confidence + second_max) / 2),
                "reason": "Detected unusually high confidence gap - adding uncertainty to prevent overconfidence."
            }
        
        return {"is_biased": False}
    
    def _get_alternative_predictions(self, top_indices: np.ndarray, top_confidences: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get alternative predictions for better decision making
        
        Args:
            top_indices: Top prediction class indices
            top_confidences: Top prediction confidences
            
        Returns:
            List of alternative predictions
        """
        alternatives = []
        for i in range(1, min(4, len(top_indices))):  # Get top 3 alternatives
            class_id = top_indices[i]
            confidence = float(top_confidences[i])
            disease_name = self._clean_disease_name(
                self.disease_mapping.get(class_id, f"Unknown Disease {class_id}")
            )
            
            alternatives.append({
                "prediction": disease_name,
                "confidence": round(confidence, 4)
            })
        
        return alternatives
    
    def _clean_disease_name(self, raw_name: str) -> str:
        """
        Clean up disease name for better presentation
        
        Args:
            raw_name: Raw disease name from model mapping
            
        Returns:
            str: Cleaned disease name
        """
        # Remove underscores and improve formatting
        cleaned = raw_name.replace("___", " - ").replace("_", " ")
        
        # Handle special cases for better readability
        cleaned = cleaned.replace("(including sour)", "")
        cleaned = cleaned.replace("Two-spotted spider mite", "Two-spotted Spider Mite")
        
        # Capitalize properly
        words = cleaned.split()
        formatted_words = []
        
        for word in words:
            if word.lower() in ["and", "or", "the", "of", "in", "on", "at", "to", "for"]:
                formatted_words.append(word.lower())
            elif "(" in word or ")" in word:
                formatted_words.append(word)
            else:
                formatted_words.append(word.capitalize())
        
        return " ".join(formatted_words).strip()

    def predict_from_image_bytes(self, image_data: bytes) -> Dict[str, Any]:
        """
        Complete pipeline: preprocess image and predict disease
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict containing simplified prediction results or error
        """
        try:
            # Preprocess image
            image_array = self.preprocess_image(image_data)
            
            # Predict disease
            return self.predict_disease(image_array)
            
        except Exception as e:
            logger.error(f"Error in complete prediction pipeline: {str(e)}")
            return {
                "error": str(e)
            }
    
    def predict_from_base64(self, base64_image: str) -> Dict[str, Any]:
        """
        Complete pipeline: preprocess base64 image and predict disease
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            Dict containing simplified prediction results or error
        """
        try:
            # Preprocess image
            image_array = self.preprocess_base64_image(base64_image)
            
            # Predict disease
            return self.predict_disease(image_array)
            
        except Exception as e:
            logger.error(f"Error in base64 prediction pipeline: {str(e)}")
            return {
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model information
        """
        try:
            self.get_model()
        except Exception as e:
            return {"error": f"Model not loaded: {e}"}
        
        
        return {
            "model_path": str(self.model_path),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "total_diseases": len(self.disease_mapping),
            "model_loaded": self.model is not None,
            "supported_formats": ["JPEG", "PNG", "JPG"],
            "max_image_size": "10MB"
        }
    
    def get_supported_diseases(self) -> List[Dict[str, Any]]:
        """
        Get list of all supported diseases
        
        Returns:
            List of disease information
        """
        return [
            {
                "disease_id": disease_id,
                "disease_name": disease_name,
                "is_healthy": "healthy" in disease_name.lower()
            }
            for disease_id, disease_name in self.disease_mapping.items()
        ]