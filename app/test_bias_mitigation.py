#!/usr/bin/env python3
"""
Test script to verify bias mitigation strategies are working correctly
"""

import sys
import os
import numpy as np
from PIL import Image
import json

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.pest_detection_service import PestDetectionService

def create_test_image(color=(128, 128, 128), size=(224, 224)):
    """Create a test image with specified color"""
    image = Image.new('RGB', size, color)
    return image

def test_bias_mitigation():
    """Test the bias mitigation functionality"""
    print("🧪 Testing Bias Mitigation Strategies")
    print("=" * 50)
    
    # Initialize service
    service = PestDetectionService()
    
    if not service.is_model_loaded():
        print("❌ Model failed to load")
        return False
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model supports {len(service.disease_mapping)} disease classes")
    print()
    
    # Test cases with different images
    test_cases = [
        {"name": "Gray Image", "color": (128, 128, 128)},
        {"name": "Green Image", "color": (0, 255, 0)},
        {"name": "Brown Image", "color": (139, 69, 19)},
        {"name": "Red Image", "color": (255, 0, 0)},
        {"name": "Blue Image", "color": (0, 0, 255)},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test {i}: {test_case['name']}")
        
        # Create test image
        test_image = create_test_image(test_case['color'])
        
        try:
            # Convert PIL image to bytes for preprocessing
            import io
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Process image
            processed_image = service.preprocess_image(img_byte_arr)
            
            # Get prediction with bias mitigation
            result = service.predict_disease(processed_image)
            
            # Display results
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Status: {result.get('status', 'normal')}")
            
            if 'note' in result:
                print(f"   Note: {result['note']}")
            
            if 'original_prediction' in result:
                print(f"   Original: {result['original_prediction']}")
            
            if 'alternative_predictions' in result:
                print("   Alternatives:")
                for alt in result['alternative_predictions'][:2]:  # Show top 2
                    print(f"     - {alt['prediction']}: {alt['confidence']:.4f}")
            
            results.append({
                "test": test_case['name'],
                "result": result
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                "test": test_case['name'],
                "error": str(e)
            })
        
        print()
    
    # Analysis
    print("📈 Bias Mitigation Analysis")
    print("=" * 30)
    
    # Check for diversity in predictions
    predictions = [r['result']['prediction'] for r in results if 'result' in r]
    unique_predictions = set(predictions)
    
    print(f"Total tests: {len(test_cases)}")
    print(f"Successful predictions: {len(predictions)}")
    print(f"Unique predictions: {len(unique_predictions)}")
    
    if len(predictions) > 0:
        print(f"Prediction diversity: {len(unique_predictions)/len(predictions)*100:.1f}%")
        
        # Check bias mitigation triggers
        bias_adjusted = sum(1 for r in results if 'result' in r and r['result'].get('status') == 'bias_adjusted')
        low_confidence = sum(1 for r in results if 'result' in r and r['result'].get('status') == 'low_confidence')
        high_confidence_alt = sum(1 for r in results if 'result' in r and r['result'].get('status') == 'high_confidence_with_alternatives')
        
        print(f"Bias adjustments: {bias_adjusted}")
        print(f"Low confidence detections: {low_confidence}")
        print(f"High confidence with alternatives: {high_confidence_alt}")
        
        # Check if known biased predictions are being caught
        biased_predictions = ["Cherry - Powdery Mildew", "Potato - Healthy"]
        caught_bias = 0
        for r in results:
            if 'result' in r:
                pred = r['result']['prediction']
                if any(bias in pred for bias in biased_predictions) and r['result'].get('status') == 'bias_adjusted':
                    caught_bias += 1
        
        print(f"Known bias patterns caught: {caught_bias}")
        
        print("\n🎯 Bias Mitigation Assessment:")
        if len(unique_predictions) > 1:
            print("✅ Model shows prediction diversity")
        else:
            print("⚠️  Model still shows bias (same prediction for all inputs)")
        
        if bias_adjusted > 0 or low_confidence > 0:
            print("✅ Bias mitigation strategies are active")
        else:
            print("⚠️  Bias mitigation strategies not triggered")
    else:
        print("❌ No successful predictions to analyze")
    
    return True

if __name__ == "__main__":
    test_bias_mitigation()