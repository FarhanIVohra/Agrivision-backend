"""
Integration Test for New Pest Disease Model
Tests how the new pest_disease_model.h5 would integrate with existing service
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewModelIntegrationTest:
    """Test integration of new model with existing service structure"""
    
    def __init__(self):
        self.old_model_path = Path("ml_models/saved_models/crop_disease_model.h5")
        self.new_model_path = Path("ml_models/saved_models/pest_disease_model.h5")
        
        # Current service configuration (from pest_detection_service.py)
        self.current_config = {
            'model_path': str(self.old_model_path),
            'input_shape': (224, 224, 3),
            'num_classes': 42,
            'disease_mapping': {
                0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust",
                3: "Apple___healthy", 4: "Blueberry___healthy", 5: "Cherry_(including_sour)___Powdery_mildew",
                6: "Cherry_(including_sour)___healthy", 7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                8: "Corn_(maize)___Common_rust_", 9: "Corn_(maize)___Northern_Leaf_Blight",
                10: "Corn_(maize)___healthy", 11: "Grape___Black_rot", 12: "Grape___Esca_(Black_Measles)",
                13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 14: "Grape___healthy",
                15: "Orange___Haunglongbing_(Citrus_greening)", 16: "Peach___Bacterial_spot",
                17: "Peach___healthy", 18: "Pepper,_bell___Bacterial_spot", 19: "Pepper,_bell___healthy",
                20: "Potato___Early_blight", 21: "Potato___Late_blight", 22: "Potato___healthy",
                23: "Raspberry___healthy", 24: "Soybean___healthy", 25: "Squash___Powdery_mildew",
                26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy", 28: "Tomato___Bacterial_spot",
                29: "Tomato___Early_blight", 30: "Tomato___Late_blight", 31: "Tomato___Leaf_Mold",
                32: "Tomato___Septoria_leaf_spot", 33: "Tomato___Spider_mites Two-spotted_spider_mite",
                34: "Tomato___Target_Spot", 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                36: "Tomato___Tomato_mosaic_virus", 37: "Tomato___healthy", 38: "Background_without_leaves",
                39: "Invalid", 40: "Diseased", 41: "Healthy"
            }
        }
        
        # Proposed new configuration
        self.new_config = {
            'model_path': str(self.new_model_path),
            'input_shape': (128, 128, 3),
            'num_classes': 22,
            'disease_mapping': {
                0: "Healthy_Leaf", 1: "Apple_Scab", 2: "Apple_Black_Rot", 3: "Apple_Cedar_Rust",
                4: "Cherry_Powdery_Mildew", 5: "Corn_Gray_Leaf_Spot", 6: "Corn_Common_Rust",
                7: "Corn_Northern_Blight", 8: "Grape_Black_Rot", 9: "Grape_Esca",
                10: "Grape_Leaf_Blight", 11: "Peach_Bacterial_Spot", 12: "Pepper_Bacterial_Spot",
                13: "Potato_Early_Blight", 14: "Potato_Late_Blight", 15: "Strawberry_Leaf_Scorch",
                16: "Tomato_Bacterial_Spot", 17: "Tomato_Early_Blight", 18: "Tomato_Late_Blight",
                19: "Tomato_Leaf_Mold", 20: "Tomato_Septoria_Leaf_Spot", 21: "Tomato_Spider_Mites"
            }
        }
    
    def load_models(self):
        """Load both models for comparison"""
        models = {}
        
        try:
            # Load old model
            if self.old_model_path.exists():
                logger.info(f"Loading old model: {self.old_model_path}")
                models['old'] = tf.keras.models.load_model(str(self.old_model_path))
                logger.info(f"Old model loaded - Input: {models['old'].input_shape}, Output: {models['old'].output_shape}")
            else:
                logger.warning(f"Old model not found: {self.old_model_path}")
            
            # Load new model
            if self.new_model_path.exists():
                logger.info(f"Loading new model: {self.new_model_path}")
                models['new'] = tf.keras.models.load_model(str(self.new_model_path))
                logger.info(f"New model loaded - Input: {models['new'].input_shape}, Output: {models['new'].output_shape}")
            else:
                logger.warning(f"New model not found: {self.new_model_path}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return {}
    
    def preprocess_image_old(self, image):
        """Preprocess image for old model (224x224)"""
        image = image.resize((224, 224))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)
    
    def preprocess_image_new(self, image):
        """Preprocess image for new model (128x128)"""
        image = image.resize((128, 128))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)
    
    def compare_predictions(self, test_image):
        """Compare predictions from both models"""
        models = self.load_models()
        
        if not models:
            logger.error("No models loaded for comparison")
            return
        
        results = {}
        
        # Test old model
        if 'old' in models:
            try:
                processed_old = self.preprocess_image_old(test_image)
                pred_old = models['old'].predict(processed_old, verbose=0)
                class_old = np.argmax(pred_old[0])
                conf_old = float(pred_old[0][class_old])
                disease_old = self.current_config['disease_mapping'].get(class_old, f"Unknown_{class_old}")
                
                results['old'] = {
                    'class': class_old,
                    'confidence': conf_old,
                    'disease': disease_old,
                    'model_size': '41.75 MB',
                    'input_shape': '224x224'
                }
            except Exception as e:
                logger.error(f"Error with old model: {str(e)}")
        
        # Test new model
        if 'new' in models:
            try:
                processed_new = self.preprocess_image_new(test_image)
                pred_new = models['new'].predict(processed_new, verbose=0)
                class_new = np.argmax(pred_new[0])
                conf_new = float(pred_new[0][class_new])
                disease_new = self.new_config['disease_mapping'].get(class_new, f"Unknown_{class_new}")
                
                results['new'] = {
                    'class': class_new,
                    'confidence': conf_new,
                    'disease': disease_new,
                    'model_size': '1.56 MB',
                    'input_shape': '128x128'
                }
            except Exception as e:
                logger.error(f"Error with new model: {str(e)}")
        
        return results
    
    def run_integration_test(self):
        """Run integration test comparing both models"""
        logger.info("=== Model Integration Comparison Test ===")
        
        # Create test images
        test_images = [
            ("Random RGB", np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)),
            ("Green leaf-like", np.full((256, 256, 3), [50, 150, 50], dtype=np.uint8)),
            ("Brown diseased-like", np.full((256, 256, 3), [139, 69, 19], dtype=np.uint8))
        ]
        
        for test_name, image_array in test_images:
            logger.info(f"\n--- Testing: {test_name} ---")
            test_image = Image.fromarray(image_array)
            
            results = self.compare_predictions(test_image)
            
            if 'old' in results:
                old = results['old']
                logger.info(f"OLD MODEL ({old['model_size']}, {old['input_shape']}):")
                logger.info(f"  Prediction: {old['disease']}")
                logger.info(f"  Confidence: {old['confidence']:.3f}")
            
            if 'new' in results:
                new = results['new']
                logger.info(f"NEW MODEL ({new['model_size']}, {new['input_shape']}):")
                logger.info(f"  Prediction: {new['disease']}")
                logger.info(f"  Confidence: {new['confidence']:.3f}")
    
    def analyze_service_changes_needed(self):
        """Analyze what changes would be needed in pest_detection_service.py"""
        logger.info("\n=== Required Service Changes Analysis ===")
        
        changes = [
            "1. MODEL PATH: Update model_path to 'pest_disease_model.h5'",
            "2. INPUT SHAPE: Change from (224, 224, 3) to (128, 128, 3)",
            "3. NUM CLASSES: Reduce from 42 to 22 classes",
            "4. DISEASE MAPPING: Update to new 22-class mapping",
            "5. PREPROCESSING: Adjust image resize from 224x224 to 128x128",
            "6. API RESPONSES: Update supported diseases list",
            "7. TESTING: Update test cases for new class count"
        ]
        
        for change in changes:
            logger.info(f"  {change}")
        
        logger.info("\n=== Compatibility Analysis ===")
        logger.info("✓ COMPATIBLE: Same TensorFlow/Keras format")
        logger.info("✓ COMPATIBLE: Same RGB input format")
        logger.info("✓ COMPATIBLE: Same prediction workflow")
        logger.info("⚠ BREAKING: Different input dimensions (224→128)")
        logger.info("⚠ BREAKING: Different number of classes (42→22)")
        logger.info("⚠ BREAKING: Different disease names/mapping")

def main():
    """Main function to run integration tests"""
    tester = NewModelIntegrationTest()
    
    # Run comparison test
    tester.run_integration_test()
    
    # Analyze required changes
    tester.analyze_service_changes_needed()

if __name__ == "__main__":
    main()