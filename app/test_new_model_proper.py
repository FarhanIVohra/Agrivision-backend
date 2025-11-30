import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def test_new_model_properly():
    """Test the new model with proper feature engineering"""
    print("🔬 Testing New Crop Yield Model")
    print("=" * 50)
    
    # Load the new model
    try:
        new_model = joblib.load('ml_models/saved_models/crop_yield_model.joblib')
        print('✅ New model loaded successfully')
        print(f'Model type: {type(new_model)}')
        print(f'Expected features: {new_model.n_features_in_}')
    except Exception as e:
        print(f'❌ Failed to load new model: {e}')
        return
    
    # Load the dataset to understand feature engineering
    df = pd.read_csv('ml_models/saved_models/crop_yield.csv')
    print(f'Dataset shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')
    
    # Check if the model expects one-hot encoded features
    print(f'\nModel expects {new_model.n_features_in_} features')
    print(f'Dataset has {len(df.columns)-1} raw features (excluding target)')
    
    # The model likely uses one-hot encoding for categorical variables
    # Let's recreate the feature engineering process
    
    # Prepare the data as it was likely done during training
    df_processed = df.copy()
    
    # One-hot encode categorical variables
    categorical_cols = ['Crop', 'Season', 'State']
    
    print(f'\n📊 Categorical Variables:')
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f'{col}: {len(unique_vals)} unique values')
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
    
    # Remove target variable to get features
    feature_cols = [col for col in df_encoded.columns if col != 'Yield']
    X = df_encoded[feature_cols]
    y = df_encoded['Yield']
    
    print(f'\nAfter encoding: {X.shape[1]} features')
    print(f'Feature columns: {X.columns.tolist()[:10]}...')  # Show first 10
    
    if X.shape[1] == new_model.n_features_in_:
        print('✅ Feature count matches!')
        
        # Test with a few samples
        print(f'\n🧪 Testing predictions:')
        
        # Test with first 5 samples
        for i in range(min(5, len(X))):
            sample = X.iloc[i:i+1]
            actual = y.iloc[i]
            
            try:
                prediction = new_model.predict(sample)[0]
                print(f'Sample {i+1}: Predicted={prediction:.4f}, Actual={actual:.4f}, Error={abs(prediction-actual):.4f}')
            except Exception as e:
                print(f'Sample {i+1}: Prediction failed - {e}')
        
        # Calculate overall performance on a subset
        print(f'\n📈 Model Performance (first 100 samples):')
        subset_size = min(100, len(X))
        X_test = X.iloc[:subset_size]
        y_test = y.iloc[:subset_size]
        
        try:
            predictions = new_model.predict(X_test)
            
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f'MSE: {mse:.4f}')
            print(f'RMSE: {rmse:.4f}')
            print(f'MAE: {mae:.4f}')
            print(f'R²: {r2:.4f}')
            
            # Show prediction range
            print(f'\nPrediction Range:')
            print(f'Min: {predictions.min():.4f}')
            print(f'Max: {predictions.max():.4f}')
            print(f'Mean: {predictions.mean():.4f}')
            
            print(f'\nActual Range:')
            print(f'Min: {y_test.min():.4f}')
            print(f'Max: {y_test.max():.4f}')
            print(f'Mean: {y_test.mean():.4f}')
            
        except Exception as e:
            print(f'❌ Performance calculation failed: {e}')
    
    else:
        print(f'❌ Feature count mismatch: Expected {new_model.n_features_in_}, got {X.shape[1]}')
    
    return new_model, X, y

def create_prediction_function():
    """Create a function that can be used for new predictions"""
    print(f'\n🔧 Creating Prediction Function')
    print("=" * 50)
    
    # Load model and dataset
    new_model = joblib.load('ml_models/saved_models/crop_yield_model.joblib')
    df = pd.read_csv('ml_models/saved_models/crop_yield.csv')
    
    # Get the feature structure from training data
    categorical_cols = ['Crop', 'Season', 'State']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    feature_cols = [col for col in df_encoded.columns if col != 'Yield']
    
    print(f'Feature template created with {len(feature_cols)} features')
    
    def predict_yield(crop, season, state, crop_year, area, annual_rainfall, fertilizer, pesticide):
        """Predict yield for given parameters"""
        
        # Create a sample with the input parameters
        sample_data = {
            'Crop': crop,
            'Crop_Year': crop_year,
            'Season': season,
            'State': state,
            'Area': area,
            'Production': 0,  # This will be ignored
            'Annual_Rainfall': annual_rainfall,
            'Fertilizer': fertilizer,
            'Pesticide': pesticide,
            'Yield': 0  # This will be ignored
        }
        
        # Create DataFrame
        sample_df = pd.DataFrame([sample_data])
        
        # Apply same encoding
        sample_encoded = pd.get_dummies(sample_df, columns=categorical_cols, drop_first=False)
        
        # Ensure all columns are present
        for col in feature_cols:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        
        # Select only the feature columns in the right order
        X_sample = sample_encoded[feature_cols]
        
        # Make prediction
        prediction = new_model.predict(X_sample)[0]
        return prediction
    
    # Test the function
    print(f'\n🧪 Testing prediction function:')
    
    test_cases = [
        ('Rice', 'Kharif     ', 'Punjab', 2020, 100, 1200, 50000, 200),
        ('Wheat', 'Rabi       ', 'Haryana', 2020, 50, 800, 30000, 150),
        ('Maize', 'Kharif     ', 'Maharashtra', 2020, 75, 1000, 40000, 180)
    ]
    
    for i, (crop, season, state, year, area, rainfall, fertilizer, pesticide) in enumerate(test_cases):
        try:
            prediction = predict_yield(crop, season, state, year, area, rainfall, fertilizer, pesticide)
            print(f'Test {i+1}: {crop} in {state} ({season}) -> {prediction:.4f} tons/hectare')
        except Exception as e:
            print(f'Test {i+1}: Failed - {e}')
    
    return predict_yield

if __name__ == "__main__":
    model, X, y = test_new_model_properly()
    predict_func = create_prediction_function()