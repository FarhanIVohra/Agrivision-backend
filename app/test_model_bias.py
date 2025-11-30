"""
Model Bias Analysis Test
Tests crop_disease_model.h5 with various inputs to identify bias patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.pest_detection_service import PestDetectionService
import numpy as np
from PIL import Image
import io

def create_test_image(color, size=(224, 224), pattern=None):
    """Create test images with different colors and patterns"""
    image = Image.new('RGB', size, color)
    
    if pattern == "spots":
        # Add some spots to simulate disease
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for i in range(10):
            x = np.random.randint(0, size[0]-20)
            y = np.random.randint(0, size[1]-20)
            draw.ellipse([x, y, x+20, y+20], fill=(139, 69, 19))  # Brown spots
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_model_predictions():
    """Test model with various inputs to identify bias"""
    print("=" * 80)
    print("COMPREHENSIVE MODEL BIAS ANALYSIS")
    print("=" * 80)
    
    service = PestDetectionService()
    
    # Test cases with different colors and patterns
    test_cases = [
        ("Healthy Green Leaf", (0, 255, 0), None),
        ("Yellow Leaf (Disease)", (255, 255, 0), None),
        ("Brown Leaf (Disease)", (139, 69, 19), None),
        ("Red Leaf (Disease)", (255, 0, 0), None),
        ("Black Spots", (0, 0, 0), None),
        ("White Background", (255, 255, 255), None),
        ("Green with Brown Spots", (0, 255, 0), "spots"),
        ("Pure Blue (Unnatural)", (0, 0, 255), None),
        ("Pure Purple (Unnatural)", (128, 0, 128), None),
        ("Gray (Dead Plant)", (128, 128, 128), None)
    ]
    
    print(f"Testing {len(test_cases)} different image types...\n")
    
    results = {}
    
    for i, (description, color, pattern) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print("-" * 40)
        
        try:
            # Create test image
            image_data = create_test_image(color, pattern=pattern)
            
            # Get prediction
            result = service.predict_from_image_bytes(image_data)
            
            prediction = result.get('prediction', 'ERROR')
            confidence = result.get('confidence', 0.0)
            
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Track results
            if prediction not in results:
                results[prediction] = 0
            results[prediction] += 1
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()
    
    # Analysis
    print("=" * 80)
    print("BIAS ANALYSIS RESULTS")
    print("=" * 80)
    
    print("Prediction Distribution:")
    for prediction, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_cases)) * 100
        print(f"  {prediction}: {count}/{len(test_cases)} ({percentage:.1f}%)")
    
    print("\nBias Assessment:")
    if len(results) == 1:
        single_prediction = list(results.keys())[0]
        print(f"🚨 SEVERE BIAS DETECTED: Model ALWAYS predicts '{single_prediction}'")
        print("   This indicates the model is not functioning properly.")
    elif len(results) <= 3:
        print(f"⚠️  MODERATE BIAS: Model shows limited diversity ({len(results)} unique predictions)")
        print("   Model may have training issues or overfitting.")
    else:
        print(f"✅ GOOD DIVERSITY: Model shows {len(results)} different predictions")
        print("   Model appears to be working correctly.")
    
    return results

def test_random_noise():
    """Test model with random noise to check for static behavior"""
    print("\n" + "=" * 80)
    print("RANDOM NOISE TEST")
    print("=" * 80)
    
    service = PestDetectionService()
    
    print("Testing with 10 random noise images...")
    
    predictions = []
    
    for i in range(10):
        # Create random noise image
        random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(random_array)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_data = img_byte_arr.getvalue()
        
        result = service.predict_from_image_bytes(image_data)
        prediction = result.get('prediction', 'ERROR')
        confidence = result.get('confidence', 0.0)
        
        predictions.append(prediction)
        print(f"  Random {i+1}: {prediction} ({confidence:.4f})")
    
    # Check for diversity
    unique_predictions = set(predictions)
    print(f"\nRandom Noise Results:")
    print(f"  Unique predictions: {len(unique_predictions)}")
    print(f"  Most common: {max(set(predictions), key=predictions.count)}")
    
    if len(unique_predictions) == 1:
        print("🚨 CRITICAL: Model gives identical predictions for random noise!")
        print("   This confirms severe model bias/malfunction.")
    else:
        print(f"✅ Model shows some diversity with random inputs.")

def main():
    """Run comprehensive bias analysis"""
    print("CROP DISEASE MODEL BIAS ANALYSIS")
    print("Investigating why model consistently predicts 'Potato - Healthy'")
    print("=" * 80)
    
    try:
        # Test with various synthetic images
        synthetic_results = test_model_predictions()
        
        # Test with random noise
        test_random_noise()
        
        print("\n" + "=" * 80)
        print("FINAL ASSESSMENT")
        print("=" * 80)
        
        if len(synthetic_results) == 1 and "Potato - Healthy" in synthetic_results:
            print("🚨 CONFIRMED ISSUE: Model is biased towards 'Potato - Healthy'")
            print("\nPossible causes:")
            print("  1. Model training data was heavily skewed towards potato images")
            print("  2. Model overfitted to potato class during training")
            print("  3. Model weights are corrupted or improperly saved")
            print("  4. Preprocessing pipeline has issues")
            
            print("\nRecommended solutions:")
            print("  1. Retrain model with balanced dataset")
            print("  2. Implement confidence thresholding")
            print("  3. Add prediction diversity checks")
            print("  4. Use ensemble of models")
        else:
            print("✅ Model shows acceptable diversity in predictions")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()