import joblib
import pandas as pd
import numpy as np

def inspect_model_features():
    """Inspect the exact features the model was trained with"""
    print("🔍 Inspecting Model Features")
    print("=" * 50)
    
    # Load the model
    model = joblib.load('ml_models/saved_models/crop_yield_model.joblib')
    
    print(f"Model type: {type(model)}")
    print(f"Expected features: {model.n_features_in_}")
    
    # Try to get feature names if available
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names available: {len(model.feature_names_in_)}")
        print("\n📋 Model Feature Names:")
        for i, feature in enumerate(model.feature_names_in_):
            print(f"{i+1:2d}. {feature}")
    else:
        print("❌ Feature names not available in model")
    
    # Load dataset and try different encoding strategies
    df = pd.read_csv('ml_models/saved_models/crop_yield.csv')
    
    print(f"\n📊 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Try different encoding approaches
    print(f"\n🧪 Testing Different Encoding Approaches:")
    
    # Approach 1: Standard get_dummies
    df1 = pd.get_dummies(df, columns=['Crop', 'Season', 'State'], drop_first=False)
    features1 = [col for col in df1.columns if col != 'Yield']
    print(f"Approach 1 (get_dummies, drop_first=False): {len(features1)} features")
    
    # Approach 2: get_dummies with drop_first=True
    df2 = pd.get_dummies(df, columns=['Crop', 'Season', 'State'], drop_first=True)
    features2 = [col for col in df2.columns if col != 'Yield']
    print(f"Approach 2 (get_dummies, drop_first=True): {len(features2)} features")
    
    # Approach 3: Only some columns encoded
    df3 = df.copy()
    # Maybe only some columns were one-hot encoded
    df3 = pd.get_dummies(df3, columns=['Crop'], drop_first=False)
    features3 = [col for col in df3.columns if col not in ['Yield', 'Season', 'State']]
    print(f"Approach 3 (only Crop encoded): {len(features3)} features")
    
    # Check which approach matches
    target_features = model.n_features_in_
    
    if len(features1) == target_features:
        print(f"✅ Approach 1 matches! Using get_dummies with drop_first=False")
        return df1, features1
    elif len(features2) == target_features:
        print(f"✅ Approach 2 matches! Using get_dummies with drop_first=True")
        return df2, features2
    elif len(features3) == target_features:
        print(f"✅ Approach 3 matches! Only Crop column encoded")
        return df3, features3
    else:
        print(f"❌ No approach matches exactly. Need to investigate further.")
        
        # Try to find the exact combination
        print(f"\n🔍 Investigating feature combinations...")
        
        # Maybe some columns were excluded
        numeric_cols = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        
        # Try excluding Production column (as it might be target-related)
        df4 = df.drop('Production', axis=1)
        df4 = pd.get_dummies(df4, columns=['Crop', 'Season', 'State'], drop_first=True)
        features4 = [col for col in df4.columns if col != 'Yield']
        print(f"Approach 4 (exclude Production, drop_first=True): {len(features4)} features")
        
        if len(features4) == target_features:
            print(f"✅ Approach 4 matches! Excluding Production column")
            return df4, features4
        
        # Try different combinations
        for drop_first in [True, False]:
            for exclude_prod in [True, False]:
                df_test = df.copy()
                if exclude_prod:
                    df_test = df_test.drop('Production', axis=1)
                
                df_test = pd.get_dummies(df_test, columns=['Crop', 'Season', 'State'], drop_first=drop_first)
                features_test = [col for col in df_test.columns if col != 'Yield']
                
                if len(features_test) == target_features:
                    print(f"✅ Found match: drop_first={drop_first}, exclude_production={exclude_prod}")
                    return df_test, features_test
        
        print(f"❌ Could not find exact feature combination")
        return None, None

def test_with_correct_features():
    """Test the model with the correct feature encoding"""
    df_encoded, feature_cols = inspect_model_features()
    
    if df_encoded is None:
        print("❌ Could not determine correct feature encoding")
        return
    
    print(f"\n🧪 Testing Model with Correct Features")
    print("=" * 50)
    
    model = joblib.load('ml_models/saved_models/crop_yield_model.joblib')
    
    # Prepare test data
    X = df_encoded[feature_cols]
    y = df_encoded['Yield']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Test predictions on first 10 samples
    print(f"\n📈 Sample Predictions:")
    for i in range(min(10, len(X))):
        sample = X.iloc[i:i+1]
        actual = y.iloc[i]
        
        try:
            prediction = model.predict(sample)[0]
            error = abs(prediction - actual)
            print(f"Sample {i+1:2d}: Pred={prediction:6.3f}, Actual={actual:6.3f}, Error={error:6.3f}")
        except Exception as e:
            print(f"Sample {i+1:2d}: Failed - {e}")
    
    # Calculate performance metrics
    try:
        predictions = model.predict(X[:100])  # Test on first 100 samples
        actuals = y[:100]
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        print(f"\n📊 Performance Metrics (100 samples):")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
        print(f"\n📈 Prediction Statistics:")
        print(f"Pred Min:  {predictions.min():.4f}")
        print(f"Pred Max:  {predictions.max():.4f}")
        print(f"Pred Mean: {predictions.mean():.4f}")
        print(f"Actual Min:  {actuals.min():.4f}")
        print(f"Actual Max:  {actuals.max():.4f}")
        print(f"Actual Mean: {actuals.mean():.4f}")
        
    except Exception as e:
        print(f"❌ Performance calculation failed: {e}")

if __name__ == "__main__":
    test_with_correct_features()