"""
Integration Test for Pest Detection Service
Tests that crop_disease_model.h5 is properly configured as the single model for pest disease detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.pest_detection_service import PestDetectionService
import numpy as np
from PIL import Image
import io

def test_service_initialization():
    """Test that the service initializes correctly with crop_disease_model.h5"""
    print("=" * 60)
    print("TESTING PEST DETECTION SERVICE INITIALIZATION")
    print("=" * 60)
    
    try:
        service = PestDetectionService()
        
        print(f"✓ Service initialized successfully")
        print(f"✓ Model loaded: {service.is_model_loaded()}")
        print(f"✓ Model path: {service.model_path}")
        print(f"✓ Input shape: {service.input_shape}")
        print(f"✓ Number of classes: {service.num_classes}")
        
        # Verify it's using crop_disease_model.h5
        if "crop_disease_model.h5" in service.model_path:
            print("✓ CONFIRMED: Using crop_disease_model.h5 as the single model")
        else:
            print("✗ ERROR: Not using crop_disease_model.h5")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False

def test_model_info():
    """Test getting model information"""
    print("\n" + "=" * 60)
    print("TESTING MODEL INFORMATION")
    print("=" * 60)
    
    try:
        service = PestDetectionService()
        model_info = service.get_model_info()
        
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Verify key information
        if model_info.get("num_classes") == 42:
            print("✓ CONFIRMED: Model supports 42 disease classes")
        else:
            print("✗ ERROR: Unexpected number of classes")
            return False
            
        if model_info.get("input_shape") == (224, 224, 3):
            print("✓ CONFIRMED: Model expects 224x224x3 input")
        else:
            print("✗ ERROR: Unexpected input shape")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Model info test failed: {e}")
        return False

def test_supported_diseases():
    """Test getting supported diseases list"""
    print("\n" + "=" * 60)
    print("TESTING SUPPORTED DISEASES")
    print("=" * 60)
    
    try:
        service = PestDetectionService()
        diseases = service.get_supported_diseases()
        
        print(f"✓ Total supported diseases: {len(diseases)}")
        
        # Show first 10 diseases
        print("\nFirst 10 supported diseases:")
        for i, disease in enumerate(diseases[:10]):
            print(f"  {disease['disease_id']}: {disease['disease_name']} (Healthy: {disease['is_healthy']})")
        
        # Verify we have 42 diseases
        if len(diseases) == 42:
            print("✓ CONFIRMED: Model supports exactly 42 disease classes")
        else:
            print(f"✗ ERROR: Expected 42 diseases, got {len(diseases)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Supported diseases test failed: {e}")
        return False

def create_test_image(color=(0, 255, 0), size=(224, 224)):
    """Create a simple test image"""
    image = Image.new('RGB', size, color)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION PIPELINE")
    print("=" * 60)
    
    try:
        service = PestDetectionService()
        
        # Test with a green image (simulating healthy leaf)
        green_image = create_test_image(color=(0, 255, 0))
        result = service.predict_from_image_bytes(green_image)
        
        print("Green image prediction:")
        print(f"  Prediction: {result.get('prediction', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        
        if 'error' not in result:
            print("✓ CONFIRMED: Prediction pipeline working correctly")
        else:
            print(f"✗ ERROR: Prediction failed - {result.get('error')}")
            return False
            
        # Test with a brown image (simulating diseased leaf)
        brown_image = create_test_image(color=(139, 69, 19))
        result2 = service.predict_from_image_bytes(brown_image)
        
        print("\nBrown image prediction:")
        print(f"  Prediction: {result2.get('prediction', 'N/A')}")
        print(f"  Confidence: {result2.get('confidence', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction pipeline test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("CROP DISEASE MODEL INTEGRATION TEST")
    print("Testing that crop_disease_model.h5 is properly configured as the single model")
    print("=" * 80)
    
    tests = [
        ("Service Initialization", test_service_initialization),
        ("Model Information", test_model_info),
        ("Supported Diseases", test_supported_diseases),
        ("Prediction Pipeline", test_prediction_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name}: PASSED")
            else:
                print(f"\n✗ {test_name}: FAILED")
        except Exception as e:
            print(f"\n✗ {test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 80)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - crop_disease_model.h5 is properly configured!")
        print("✓ CONFIRMED: Single model setup is working correctly")
    else:
        print("✗ SOME TESTS FAILED - Please check the configuration")
    
    print("=" * 80)

if __name__ == "__main__":
    main()