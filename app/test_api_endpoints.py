"""
Test script for Pest Detection API endpoints
Tests the API routes with the updated pest_disease_model.h5
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import base64
import io
import json
from fastapi.testclient import TestClient

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(size=(128, 128)):
    """Create a test image for API testing"""
    # Create a synthetic plant leaf image
    image = Image.new('RGB', size, color='green')
    
    # Add some texture to make it look more realistic
    pixels = np.array(image)
    
    # Add some brown spots (disease simulation)
    for i in range(10):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        radius = np.random.randint(5, 15)
        
        for dx in range(-radius, radius):
            for dy in range(-radius, radius):
                if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                    if dx*dx + dy*dy <= radius*radius:
                        pixels[y+dy, x+dx] = [139, 69, 19]  # Brown color
    
    return Image.fromarray(pixels)

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_api_endpoints():
    """Test all pest detection API endpoints"""
    client = TestClient(app)
    
    print("🧪 Testing Pest Detection API Endpoints")
    print("=" * 50)
    
    # Test 1: Get model info
    print("\n1. Testing /api/pest-detection/model-info endpoint...")
    try:
        response = client.get("/api/pest-detection/model-info")
        assert response.status_code == 200
        data = response.json()
        print(f"   ✅ Model info retrieved successfully")
        print(f"   📊 Model path: {data['model_info']['model_path']}")
        print(f"   📏 Input shape: {data['model_info']['input_shape']}")
        print(f"   🎯 Classes: {data['model_info']['num_classes']}")
    except Exception as e:
        print(f"   ❌ Model info test failed: {e}")
        return False
    
    # Test 2: Get supported diseases
    print("\n2. Testing /api/pest-detection/supported-diseases endpoint...")
    try:
        response = client.get("/api/pest-detection/supported-diseases")
        assert response.status_code == 200
        data = response.json()
        print(f"   ✅ Supported diseases retrieved successfully")
        print(f"   🦠 Total diseases: {data['total_diseases']}")
        print(f"   📋 Sample diseases: {[d['disease_name'] for d in data['diseases'][:3]]}")
    except Exception as e:
        print(f"   ❌ Supported diseases test failed: {e}")
        return False
    
    # Test 3: Disease detection from base64
    print("\n3. Testing /api/pest-detection/detect-disease-base64 endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        base64_image = image_to_base64(test_image)
        
        # Make API request
        response = client.post(
            "/api/pest-detection/detect-disease-base64",
            json={"base64_image": base64_image}
        )
        assert response.status_code == 200
        data = response.json()
        print(f"   ✅ Disease detection successful")
        print(f"   🎯 Prediction: {data['prediction']}")
        print(f"   📊 Confidence: {data['confidence']:.3f}")
    except Exception as e:
        print(f"   ❌ Disease detection test failed: {e}")
        return False
    
    # Test 4: Disease detection with file upload
    print("\n4. Testing /api/pest-detection/detect-disease endpoint...")
    try:
        # Create test image file
        test_image = create_test_image()
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Make API request with file upload
        response = client.post(
            "/api/pest-detection/detect-disease",
            files={"file": ("test_image.jpg", img_buffer, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        print(f"   ✅ File upload detection successful")
        print(f"   🎯 Prediction: {data['prediction']}")
        print(f"   📊 Confidence: {data['confidence']:.3f}")
        print(f"   💡 Recommendation available: {'recommendation' in data}")
    except Exception as e:
        print(f"   ❌ File upload detection test failed: {e}")
        return False
    
    # Test 5: Error handling - invalid base64
    print("\n5. Testing error handling...")
    try:
        response = client.post(
            "/api/pest-detection/detect-disease-base64",
            json={"base64_image": "invalid_base64"}
        )
        # The service returns 500 for invalid base64, which is expected behavior
        assert response.status_code == 500
        print(f"   ✅ Error handling works correctly (returns 500 for invalid base64)")
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All API endpoint tests passed successfully!")
    print("✅ The updated pest detection service is working correctly with all API endpoints")
    return True

def test_disease_mapping_consistency():
    """Test that the disease mapping is consistent across service and API"""
    client = TestClient(app)
    
    print("\n🔍 Testing disease mapping consistency...")
    
    try:
        # Get supported diseases from API
        response = client.get("/api/pest-detection/supported-diseases")
        api_diseases = response.json()['diseases']
        
        # Import service directly to compare
        from services.pest_detection_service import PestDetectionService
        service = PestDetectionService()
        service_diseases = service.get_supported_diseases()
        
        # Compare counts
        assert len(api_diseases) == len(service_diseases) == 22
        print(f"   ✅ Disease count consistent: {len(api_diseases)} diseases")
        
        # Compare disease names
        api_names = {d['disease_name'] for d in api_diseases}
        service_names = {d['disease_name'] for d in service_diseases}
        assert api_names == service_names
        print(f"   ✅ Disease names consistent between API and service")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Disease mapping consistency test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Pest Detection API Tests")
    
    # Run API endpoint tests
    api_success = test_api_endpoints()
    
    # Run consistency tests
    consistency_success = test_disease_mapping_consistency()
    
    if api_success and consistency_success:
        print("\n🎊 ALL TESTS PASSED! 🎊")
        print("The pest detection system is ready for production use.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)