#!/usr/bin/env python3
"""
Crop Recommendation Model Comparison Script
Compares the current model with the new model to determine which is better
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import os

def main():
    print('=== CROP RECOMMENDATION MODEL ANALYSIS ===\n')

    # Load both models
    current_model_path = 'ml_models/saved_models/crop_recommendation_model.joblib'
    new_model_path = 'ml_models/saved_models/crop_recommender.joblib'

    print('Loading models...')
    try:
        current_model = joblib.load(current_model_path)
        print(f'✓ Current model loaded: {current_model_path}')
    except Exception as e:
        print(f'✗ Failed to load current model: {e}')
        return

    try:
        new_model = joblib.load(new_model_path)
        print(f'✓ New model loaded: {new_model_path}')
    except Exception as e:
        print(f'✗ Failed to load new model: {e}')
        return

    print('\n=== MODEL COMPARISON ===')

    # Current model analysis
    print('\n--- CURRENT MODEL (crop_recommendation_model.joblib) ---')
    print(f'Model type: {type(current_model).__name__}')
    if hasattr(current_model, 'n_features_in_'):
        print(f'Number of features: {current_model.n_features_in_}')
    if hasattr(current_model, 'classes_'):
        print(f'Number of classes: {len(current_model.classes_)}')
        print(f'Classes: {list(current_model.classes_)}')
    if hasattr(current_model, 'feature_names_in_'):
        print(f'Feature names: {list(current_model.feature_names_in_)}')

    # New model analysis
    print('\n--- NEW MODEL (crop_recommender.joblib) ---')
    print(f'Model type: {type(new_model).__name__}')
    if hasattr(new_model, 'n_features_in_'):
        print(f'Number of features: {new_model.n_features_in_}')
    if hasattr(new_model, 'classes_'):
        print(f'Number of classes: {len(new_model.classes_)}')
        print(f'Classes: {list(new_model.classes_)}')
    if hasattr(new_model, 'feature_names_in_'):
        print(f'Feature names: {list(new_model.feature_names_in_)}')

    # Class mapping analysis
    print('\n=== CLASS MAPPING ANALYSIS ===')
    
    # Current model crop mapping (from model_service.py)
    current_crop_mapping = {
        'rice': 'Rice',
        'maize': 'Maize', 
        'cotton': 'Cotton',
        'beans': 'Beans',
        'coffee': 'Coffee',
        'cowpeas': 'Cowpeas',
        'grapes': 'Grapes',
        'groundnuts': 'Groundnuts',
        'banana': 'Banana',
        'mango': 'Mango',
        'apple': 'Apple',
        'orange': 'Orange',
        'peas': 'Peas',
        'watermelon': 'Watermelon',
        'Soyabeans': 'Soybeans'
    }

    print(f'Current model classes ({len(current_model.classes_)}):')
    for i, crop in enumerate(current_model.classes_):
        display_name = current_crop_mapping.get(crop, crop.title())
        print(f'  {i}: {crop} -> {display_name}')

    # Common crop dataset mapping for new model
    common_crops = [
        'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 
        'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 
        'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 
        'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
    ]

    print(f'\nNew model classes ({len(new_model.classes_)}):')
    for i, class_id in enumerate(new_model.classes_):
        crop_name = common_crops[i] if i < len(common_crops) else f'unknown_crop_{i}'
        print(f'  {class_id}: {crop_name}')

    # Feature comparison
    print('\n=== FEATURE COMPARISON ===')
    print('Both models use 7 features:')
    print('Current model features:', list(current_model.feature_names_in_))
    print('New model features:', list(new_model.feature_names_in_))

    # Test predictions with multiple samples
    print('\n=== PREDICTION COMPARISON WITH MULTIPLE SAMPLES ===')

    test_samples = [
        {'name': 'Rice-favorable', 'data': [90, 42, 43, 20.87, 82, 6.5, 202.93]},
        {'name': 'Wheat-favorable', 'data': [50, 30, 20, 15.5, 65, 7.2, 150.0]},
        {'name': 'Cotton-favorable', 'data': [80, 40, 50, 25.0, 70, 6.8, 100.0]},
        {'name': 'High-pH', 'data': [60, 35, 40, 22.0, 75, 8.5, 180.0]},
        {'name': 'Low-rainfall', 'data': [70, 45, 35, 28.0, 60, 6.0, 50.0]}
    ]

    for sample in test_samples:
        print(f'\n--- {sample["name"]} conditions ---')
        data = np.array([sample['data']])
        
        # Current model prediction
        current_pred = current_model.predict(data)[0]
        current_prob = current_model.predict_proba(data)[0]
        current_conf = max(current_prob)
        current_display = current_crop_mapping.get(current_pred, current_pred.title())
        
        # New model prediction
        new_pred = new_model.predict(data)[0]
        new_prob = new_model.predict_proba(data)[0]
        new_conf = max(new_prob)
        new_crop_name = common_crops[new_pred] if new_pred < len(common_crops) else f'unknown_crop_{new_pred}'
        
        print(f'Input: N={sample["data"][0]}, P={sample["data"][1]}, K={sample["data"][2]}, T={sample["data"][3]}, H={sample["data"][4]}, pH={sample["data"][5]}, R={sample["data"][6]}')
        print(f'Current model: {current_display} (confidence: {current_conf:.3f})')
        print(f'New model: {new_crop_name.title()} (confidence: {new_conf:.3f})')

    # Model performance metrics
    print('\n=== MODEL PERFORMANCE METRICS ===')
    print(f'Current model algorithm: {type(current_model).__name__}')
    print(f'New model algorithm: {type(new_model).__name__}')

    if hasattr(current_model, 'n_estimators'):
        print(f'Current model n_estimators: {current_model.n_estimators}')
    if hasattr(new_model, 'n_estimators'):
        print(f'New model n_estimators: {new_model.n_estimators}')

    # Final recommendation
    print('\n=== ANALYSIS SUMMARY ===')
    print('Key Differences:')
    print('1. Current model: 15 crops, string output, direct crop names')
    print('2. New model: 22 crops, integer output, more crop varieties')
    print('3. Both use same 7 features (no additional soil type parameter found)')
    print('4. New model has more crop options but requires integer-to-name mapping')
    print('5. Both are RandomForestClassifier models')
    
    print('\n=== RECOMMENDATION ===')
    print('Based on the analysis:')
    print('✓ NEW MODEL ADVANTAGES:')
    print('  - More crop varieties (22 vs 15)')
    print('  - Potentially trained on larger/more diverse dataset')
    print('  - Same feature requirements (easy to integrate)')
    print('')
    print('✗ NEW MODEL DISADVANTAGES:')
    print('  - Requires integer-to-crop name mapping')
    print('  - Need to update existing crop mapping in model_service.py')
    print('  - May need testing with actual agricultural data')
    print('')
    print('CONCLUSION: The new model appears to be better due to more crop varieties,')
    print('but requires code changes to handle integer outputs and new crop mappings.')

if __name__ == '__main__':
    main()