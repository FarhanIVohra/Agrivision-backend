"""
Test script for the updated Pest Detection Service
Tests the service with the new pest_disease_model.h5
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import base64
import io

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from services.pest_detection_service import PestDetectionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(size=(128, 128)):
    """Create a test image for testing"""
    # Create a random RGB image
    image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string

def test_service_initialization():
    """Test service initialization"""
    logger.info("=== Testing Service Initialization ===")
    
    try:
        service = PestDetectionService()
        logger.info(f"✓ Service initialized successfully")
        logger.info(f"✓ Model loaded: {service.is_model_loaded()}")
        logger.info(f"✓ Input shape: {service.input_shape}")
        logger.info(f"✓ Number of classes: {service.num_classes}")
        return service
    except Exception as e:
        logger.error(f"✗ Service initialization failed: {str(e)}")
        return None

def test_model_info(service):
    """Test model info retrieval"""
    logger.info("\n=== Testing Model Info ===")
    
    try:
        model_info = service.get_model_info()
        logger.info(f"✓ Model info retrieved successfully:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        return True
    except Exception as e:
        logger.error(f"✗ Model info test failed: {str(e)}")
        return False

def test_supported_diseases(service):
    """Test supported diseases retrieval"""
    logger.info("\n=== Testing Supported Diseases ===")
    
    try:
        diseases = service.get_supported_diseases()
        logger.info(f"✓ Retrieved {len(diseases)} supported diseases:")
        
        # Show first 5 diseases
        for i, disease in enumerate(diseases[:5]):
            logger.info(f"  {disease['disease_id']}: {disease['disease_name']} (Healthy: {disease['is_healthy']})")
        
        if len(diseases) > 5:
            logger.info(f"  ... and {len(diseases) - 5} more diseases")
        
        return True
    except Exception as e:
        logger.error(f"✗ Supported diseases test failed: {str(e)}")
        return False

def test_image_prediction(service):
    """Test image prediction"""
    logger.info("\n=== Testing Image Prediction ===")
    
    try:
        # Create test image
        test_base64 = create_test_image()
        logger.info("✓ Test image created")
        
        # Make prediction
        result = service.predict_from_base64(test_base64)
        
        if "error" in result:
            logger.error(f"✗ Prediction failed: {result['error']}")
            return False
        
        logger.info(f"✓ Prediction successful:")
        logger.info(f"  Prediction: {result['prediction']}")
        logger.info(f"  Confidence: {result['confidence']}")
        logger.info(f"  Status: {result['status']}")
        
        if 'alternative_predictions' in result:
            logger.info(f"  Alternative predictions available: {len(result['alternative_predictions'])}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Image prediction test failed: {str(e)}")
        return False

def test_multiple_predictions(service, num_tests=3):
    """Test multiple predictions for consistency"""
    logger.info(f"\n=== Testing Multiple Predictions ({num_tests} tests) ===")
    
    results = []
    for i in range(num_tests):
        try:
            test_base64 = create_test_image()
            result = service.predict_from_base64(test_base64)
            
            if "error" not in result:
                results.append(result)
                logger.info(f"Test {i+1}: {result['prediction']} ({result['confidence']:.3f})")
            else:
                logger.error(f"Test {i+1} failed: {result['error']}")
        except Exception as e:
            logger.error(f"Test {i+1} error: {str(e)}")
    
    if results:
        # Calculate statistics
        confidences = [r['confidence'] for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        logger.info(f"✓ Completed {len(results)}/{num_tests} tests successfully")
        logger.info(f"✓ Average confidence: {avg_confidence:.3f}")
        
        # Check prediction diversity
        predictions = [r['prediction'] for r in results]
        unique_predictions = set(predictions)
        logger.info(f"✓ Unique predictions: {len(unique_predictions)}")
        
        return True
    else:
        logger.error("✗ All prediction tests failed")
        return False

def test_edge_cases(service):
    """Test edge cases"""
    logger.info("\n=== Testing Edge Cases ===")
    
    test_cases = [
        ("Very small image", (32, 32)),
        ("Large image", (512, 512)),
        ("Square image", (256, 256)),
    ]
    
    passed = 0
    for test_name, size in test_cases:
        try:
            # Create test image with specific size
            image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            result = service.predict_from_base64(base64_string)
            
            if "error" not in result:
                logger.info(f"✓ {test_name}: {result['prediction']} ({result['confidence']:.3f})")
                passed += 1
            else:
                logger.error(f"✗ {test_name}: {result['error']}")
        except Exception as e:
            logger.error(f"✗ {test_name}: {str(e)}")
    
    logger.info(f"Edge case tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def main():
    """Main test function"""
    logger.info("Starting Pest Detection Service Tests with New Model")
    logger.info("=" * 60)
    
    # Initialize service
    service = test_service_initialization()
    if not service:
        logger.error("Cannot proceed with tests - service initialization failed")
        return
    
    # Run tests
    tests = [
        ("Model Info", lambda: test_model_info(service)),
        ("Supported Diseases", lambda: test_supported_diseases(service)),
        ("Image Prediction", lambda: test_image_prediction(service)),
        ("Multiple Predictions", lambda: test_multiple_predictions(service)),
        ("Edge Cases", lambda: test_edge_cases(service))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {str(e)}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The updated service is working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please review the issues above.")

if __name__ == "__main__":
    main()