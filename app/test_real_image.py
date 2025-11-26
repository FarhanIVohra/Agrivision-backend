import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests

# Load the model
model_path = r'D:\new-folder1\AI-Crop-Yield-Prediction\agrismart\app\ml_models\saved_models\crop_disease_model.h5'
model = tf.keras.models.load_model(model_path)

# Disease mapping
disease_mapping = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot", 
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy",
    38: "Background_without_leaves",
    39: "Invalid",
    40: "Diseased_leaf",
    41: "Healthy_leaf"
}

def preprocess_image_like_service(image_data: bytes) -> np.ndarray:
    """Replicate the exact preprocessing from the service"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

# Create test images with different patterns
def create_test_image(pattern_type):
    """Create different test images"""
    if pattern_type == "green_leaf":
        # Create a green leaf-like pattern
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 1] = 100  # Green channel
        return Image.fromarray(img)
    elif pattern_type == "brown_spots":
        # Create brown spots (disease-like)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 0] = 139  # Red
        img[:, :, 1] = 69   # Green  
        img[:, :, 2] = 19   # Blue (brown color)
        return Image.fromarray(img)
    elif pattern_type == "yellow_leaf":
        # Create yellow leaf pattern
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Red
        img[:, :, 1] = 255  # Green (yellow)
        return Image.fromarray(img)

print("Testing with synthetic images that mimic real plant conditions:")
print("=" * 60)

test_patterns = ["green_leaf", "brown_spots", "yellow_leaf"]

for pattern in test_patterns:
    # Create test image
    test_img = create_test_image(pattern)
    
    # Convert to bytes (like uploaded file)
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Preprocess using service method
    processed_img = preprocess_image_like_service(img_bytes)
    
    # Get prediction
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_id = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_id])
    
    disease_name = disease_mapping.get(predicted_class_id, f"Unknown Disease {predicted_class_id}")
    
    print(f"{pattern.upper()}:")
    print(f"  Predicted: {disease_name}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Class ID: {predicted_class_id}")
    
    # Show top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    print("  Top 3 predictions:")
    for i, idx in enumerate(top3_indices):
        print(f"    {i+1}. {disease_mapping.get(idx, f'Unknown {idx}')} ({predictions[0][idx]:.4f})")
    print()

# Test with actual preprocessing values to see distribution
print("=" * 60)
print("Checking if model has extreme bias...")

# Test multiple random inputs
results = {}
for i in range(10):
    test_random = np.random.random((1, 224, 224, 3))
    pred = model.predict(test_random, verbose=0)
    predicted_class = np.argmax(pred[0])
    
    if predicted_class not in results:
        results[predicted_class] = 0
    results[predicted_class] += 1

print("Random input predictions (10 tests):")
for class_id, count in sorted(results.items()):
    disease_name = disease_mapping.get(class_id, f"Unknown {class_id}")
    print(f"  Class {class_id} ({disease_name}): {count} times")