import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def analyze_dataset():
    """Analyze the new crop yield dataset"""
    print("📊 Dataset Analysis:")
    print("=" * 50)
    
    # Load and analyze the new dataset
    df = pd.read_csv('ml_models/saved_models/crop_yield.csv')
    print(f'Shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')
    print()

    print('📈 Data Types:')
    print(df.dtypes)
    print()

    print('📋 Sample Data:')
    print(df.head())
    print()

    print('🔍 Unique Values:')
    print(f'Crops: {df["Crop"].nunique()} unique crops')
    print(f'States: {df["State"].nunique()} unique states') 
    print(f'Seasons: {df["Season"].nunique()} unique seasons')
    print()

    print('🎯 Target Variable (Yield) Analysis:')
    print(f'Min Yield: {df["Yield"].min():.4f}')
    print(f'Max Yield: {df["Yield"].max():.4f}')
    print(f'Mean Yield: {df["Yield"].mean():.4f}')
    print(f'Median Yield: {df["Yield"].median():.4f}')
    print()
    
    return df

def test_models():
    """Test both models with sample data"""
    print("\n🔬 Model Testing:")
    print("=" * 50)
    
    # Load models
    try:
        old_model = joblib.load('ml_models/saved_models/yield_prediction_model.joblib')
        print('✅ Old model loaded successfully')
        print(f'Old model type: {type(old_model)}')
    except Exception as e:
        print(f'❌ Failed to load old model: {e}')
        return

    try:
        new_model = joblib.load('ml_models/saved_models/crop_yield_model.joblib')
        print('✅ New model loaded successfully')
        print(f'New model type: {type(new_model)}')
    except Exception as e:
        print(f'❌ Failed to load new model: {e}')
        return
    
    # Load new dataset for testing
    df = pd.read_csv('ml_models/saved_models/crop_yield.csv')
    
    # Prepare sample data for new model (assuming it uses the same features as dataset)
    # Features: Crop_Year, Area, Annual_Rainfall, Fertilizer, Pesticide
    # We'll need to encode categorical variables
    
    # Get unique values for encoding
    crops = df['Crop'].unique()
    states = df['State'].unique()
    seasons = df['Season'].unique()
    
    print(f'\n📝 Available crops: {crops[:10]}...')  # Show first 10
    print(f'📝 Available states: {states[:10]}...')  # Show first 10
    print(f'📝 Available seasons: {seasons}')
    
    # Test with a few sample predictions
    test_samples = [
        {'Crop': 'Rice', 'State': 'Punjab', 'Season': 'Kharif', 'Crop_Year': 2020, 'Area': 100, 'Annual_Rainfall': 1200, 'Fertilizer': 50000, 'Pesticide': 200},
        {'Crop': 'Wheat', 'State': 'Haryana', 'Season': 'Rabi', 'Crop_Year': 2020, 'Area': 50, 'Annual_Rainfall': 800, 'Fertilizer': 30000, 'Pesticide': 150},
        {'Crop': 'Maize', 'State': 'Maharashtra', 'Season': 'Kharif', 'Crop_Year': 2020, 'Area': 75, 'Annual_Rainfall': 1000, 'Fertilizer': 40000, 'Pesticide': 180}
    ]
    
    print(f'\n🧪 Testing with sample data:')
    for i, sample in enumerate(test_samples):
        print(f'\nSample {i+1}: {sample["Crop"]} in {sample["State"]} ({sample["Season"]})')
        
        # For new model, we need to prepare features properly
        # This depends on how the model was trained
        try:
            # Create feature vector (this is a guess - we'll need to check the actual training code)
            crop_idx = list(crops).index(sample['Crop']) if sample['Crop'] in crops else 0
            state_idx = list(states).index(sample['State']) if sample['State'] in states else 0
            season_idx = list(seasons).index(sample['Season']) if sample['Season'] in seasons else 0
            
            # Prepare features for new model
            features = np.array([[
                crop_idx, sample['Crop_Year'], season_idx, state_idx,
                sample['Area'], sample['Annual_Rainfall'], 
                sample['Fertilizer'], sample['Pesticide']
            ]])
            
            new_prediction = new_model.predict(features)[0]
            print(f'  New model prediction: {new_prediction:.4f} tons/hectare')
            
        except Exception as e:
            print(f'  ❌ New model prediction failed: {e}')
    
    return new_model, old_model

if __name__ == "__main__":
    df = analyze_dataset()
    test_models()