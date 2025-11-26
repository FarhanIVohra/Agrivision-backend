import tensorflow as tf
import numpy as np

# Load the model
model_path = r'D:\new-folder1\AI-Crop-Yield-Prediction\agrismart\app\ml_models\saved_models\crop_disease_model.h5'
model = tf.keras.models.load_model(model_path)

# Disease mapping (key classes)
disease_mapping = {
    20: "Potato___Early_blight",
    21: "Potato___Late_blight", 
    22: "Potato___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    3: "Apple___healthy",
    10: "Corn_(maize)___healthy",
    37: "Tomato___healthy"
}

print('Testing model with different inputs:')
print('=' * 50)

# Test 1: All zeros (black image)
test_zeros = np.zeros((1, 224, 224, 3))
pred_zeros = model.predict(test_zeros, verbose=0)
max_idx_zeros = np.argmax(pred_zeros[0])
print(f'Black image -> Class {max_idx_zeros}: {disease_mapping.get(max_idx_zeros, "Unknown")} (confidence: {pred_zeros[0][max_idx_zeros]:.4f})')

# Test 2: All ones (white image)  
test_ones = np.ones((1, 224, 224, 3))
pred_ones = model.predict(test_ones, verbose=0)
max_idx_ones = np.argmax(pred_ones[0])
print(f'White image -> Class {max_idx_ones}: {disease_mapping.get(max_idx_ones, "Unknown")} (confidence: {pred_ones[0][max_idx_ones]:.4f})')

# Test 3: Random noise
test_random = np.random.random((1, 224, 224, 3))
pred_random = model.predict(test_random, verbose=0)
max_idx_random = np.argmax(pred_random[0])
print(f'Random noise -> Class {max_idx_random}: {disease_mapping.get(max_idx_random, "Unknown")} (confidence: {pred_random[0][max_idx_random]:.4f})')

# Test 4: Normalized image (0-1 range like preprocessing)
test_normalized = np.random.random((1, 224, 224, 3))
pred_normalized = model.predict(test_normalized, verbose=0)
max_idx_normalized = np.argmax(pred_normalized[0])
print(f'Normalized random -> Class {max_idx_normalized}: {disease_mapping.get(max_idx_normalized, "Unknown")} (confidence: {pred_normalized[0][max_idx_normalized]:.4f})')

print('\n' + '=' * 50)
print('Class 22 (Potato healthy) bias analysis:')
print(f'Black image: {pred_zeros[0][22]:.4f}')
print(f'White image: {pred_ones[0][22]:.4f}')
print(f'Random noise: {pred_random[0][22]:.4f}')
print(f'Normalized: {pred_normalized[0][22]:.4f}')

print('\n' + '=' * 50)
print('Top 5 predictions for each test:')

def print_top5(predictions, test_name):
    top5_indices = np.argsort(predictions[0])[-5:][::-1]
    print(f'\n{test_name}:')
    for i, idx in enumerate(top5_indices):
        print(f'  {i+1}. Class {idx}: {disease_mapping.get(idx, "Unknown")} ({predictions[0][idx]:.4f})')

print_top5(pred_zeros, "Black image")
print_top5(pred_ones, "White image") 
print_top5(pred_random, "Random noise")
print_top5(pred_normalized, "Normalized random")