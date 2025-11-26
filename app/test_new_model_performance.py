"""
Performance Test for New Pest Disease Model
Tests pest_disease_model.h5 with sample images and evaluates performance
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime
import time
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewModelTester:
    """Test the new pest disease model"""
    
    def __init__(self):
        self.model_path = Path("ml_models/saved_models/pest_disease_model.h5")
        self.model = None
        self.input_shape = (128, 128, 3)  # Based on model analysis
        self.num_classes = 22  # Based on model analysis
        
        # Simplified disease mapping for 22 classes (needs to be updated based on actual training)
        self.disease_mapping = {
            0: "Healthy_Leaf",
            1: "Apple_Scab",
            2: "Apple_Black_Rot",
            3: "Apple_Cedar_Rust",
            4: "Cherry_Powdery_Mildew",
            5: "Corn_Gray_Leaf_Spot",
            6: "Corn_Common_Rust",
            7: "Corn_Northern_Blight",
            8: "Grape_Black_Rot",
            9: "Grape_Esca",
            10: "Grape_Leaf_Blight",
            11: "Peach_Bacterial_Spot",
            12: "Pepper_Bacterial_Spot",
            13: "Potato_Early_Blight",
            14: "Potato_Late_Blight",
            15: "Strawberry_Leaf_Scorch",
            16: "Tomato_Bacterial_Spot",
            17: "Tomato_Early_Blight",
            18: "Tomato_Late_Blight",
            19: "Tomato_Leaf_Mold",
            20: "Tomato_Septoria_Leaf_Spot",
            21: "Tomato_Spider_Mites"
        }
        
    def load_model(self):
        """Load the new pest disease model"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
                
            logger.info(f"Loading new pest disease model from: {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Input shape: {self.model.input_shape}")
            logger.info(f"Output shape: {self.model.output_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def create_test_image(self, size=(128, 128)):
        """Create a synthetic test image"""
        # Create a random RGB image
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        try:
            # Resize to model input size
            image = image.resize((128, 128))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict_image(self, image):
        """Make prediction on a single image"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None
            
            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(processed_image, verbose=0)
            inference_time = time.time() - start_time
            
            # Get top prediction
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get disease name
            disease_name = self.disease_mapping.get(predicted_class, f"Unknown_Class_{predicted_class}")
            
            return {
                'predicted_class': int(predicted_class),
                'disease_name': disease_name,
                'confidence': confidence,
                'inference_time': inference_time,
                'all_predictions': predictions[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def run_performance_test(self, num_tests=10):
        """Run performance tests with multiple images"""
        logger.info(f"\n=== Running Performance Test with {num_tests} images ===")
        
        if not self.load_model():
            return
        
        results = []
        total_time = 0
        
        for i in range(num_tests):
            logger.info(f"Testing image {i+1}/{num_tests}")
            
            # Create test image
            test_image = self.create_test_image()
            
            # Make prediction
            result = self.predict_image(test_image)
            
            if result:
                results.append(result)
                total_time += result['inference_time']
                
                logger.info(f"  Prediction: {result['disease_name']}")
                logger.info(f"  Confidence: {result['confidence']:.3f}")
                logger.info(f"  Inference time: {result['inference_time']:.3f}s")
            else:
                logger.error(f"  Failed to process image {i+1}")
        
        # Performance summary
        if results:
            avg_time = total_time / len(results)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            logger.info(f"\n=== Performance Summary ===")
            logger.info(f"Total tests: {len(results)}/{num_tests}")
            logger.info(f"Average inference time: {avg_time:.3f}s")
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            logger.info(f"Throughput: {1/avg_time:.1f} images/second")
            
            # Class distribution
            class_counts = {}
            for result in results:
                disease = result['disease_name']
                class_counts[disease] = class_counts.get(disease, 0) + 1
            
            logger.info(f"\nPredicted classes distribution:")
            for disease, count in sorted(class_counts.items()):
                logger.info(f"  {disease}: {count} times")
        
        return results
    
    def test_model_robustness(self):
        """Test model with various input conditions"""
        logger.info(f"\n=== Testing Model Robustness ===")
        
        if not self.load_model():
            return
        
        test_cases = [
            ("Normal RGB image", self.create_test_image((128, 128))),
            ("Large image (will be resized)", self.create_test_image((512, 512))),
            ("Small image (will be resized)", self.create_test_image((64, 64))),
            ("Grayscale image", self.create_test_image((128, 128)).convert('L')),
        ]
        
        for test_name, test_image in test_cases:
            logger.info(f"\nTesting: {test_name}")
            result = self.predict_image(test_image)
            
            if result:
                logger.info(f"  ✓ Success: {result['disease_name']} ({result['confidence']:.3f})")
            else:
                logger.info(f"  ✗ Failed")

def main():
    """Main function to run new model tests"""
    tester = NewModelTester()
    
    # Run performance test
    tester.run_performance_test(num_tests=5)
    
    # Run robustness test
    tester.test_model_robustness()

if __name__ == "__main__":
    main()