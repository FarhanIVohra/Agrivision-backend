# AgriSmart API Documentation

## Overview

The AgriSmart API is a FastAPI-based backend service for crop yield prediction and recommendation. It uses machine learning models to analyze soil and environmental parameters and recommend the most suitable crops for cultivation.

## Base URL

```
http://localhost:8000
```

## API Endpoints

### 1. Health Check

#### GET `/api/health`

Check the health status of the API and ML model.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "api_version": "1.0.0"
}
```

### 2. Root Endpoint

#### GET `/`

Get basic API information and available endpoints.

**Response:**
```json
{
  "message": "AgriSmart API - Crop Yield Prediction System",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "docs": "/api/docs",
    "crop_prediction": "/api/crop-prediction"
  }
}
```

## Crop Prediction Endpoints

### 3. Get Crop Recommendations

#### POST `/api/crop-prediction/recommend`

Get multiple crop recommendations based on soil and environmental parameters.

**Request Body:**
```json
{
  "nitrogen": 90.0,
  "phosphorus": 42.0,
  "potassium": 43.0,
  "temperature": 20.87,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.93
}
```

**Parameters:**
- `nitrogen` (float): Nitrogen content in soil (0-200 kg/ha)
- `phosphorus` (float): Phosphorus content in soil (0-150 kg/ha)
- `potassium` (float): Potassium content in soil (0-300 kg/ha)
- `temperature` (float): Average temperature in Celsius (-10 to 50°C)
- `humidity` (float): Relative humidity percentage (0-100%)
- `ph` (float): Soil pH level (3.0-10.0)
- `rainfall` (float): Annual rainfall in mm (0-3000mm)

**Response:**
```json
{
  "success": true,
  "message": "Crop recommendations generated successfully",
  "input_parameters": {
    "nitrogen": 90.0,
    "phosphorus": 42.0,
    "potassium": 43.0,
    "temperature": 20.87,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.93
  },
  "recommendations": [
    {
      "crop_id": 0,
      "crop_name": "Rice",
      "confidence": 0.85,
      "suitability_score": 85.2
    },
    {
      "crop_id": 1,
      "crop_name": "Maize",
      "confidence": 0.72,
      "suitability_score": 72.1
    }
  ],
  "total_crops": 2,
  "prediction_metadata": {
    "model_version": "1.0",
    "prediction_time": "2024-01-01T12:00:00Z",
    "feature_count": 7,
    "available_crops": 22
  }
}
```

### 4. Get Single Best Crop Recommendation

#### POST `/api/crop-prediction/predict-single`

Get the single most suitable crop recommendation.

**Request Body:** Same as `/recommend` endpoint

**Response:**
```json
{
  "success": true,
  "message": "Single crop prediction completed successfully",
  "input_parameters": {
    "nitrogen": 90.0,
    "phosphorus": 42.0,
    "potassium": 43.0,
    "temperature": 20.87,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.93
  },
  "prediction": {
    "crop_id": 0,
    "crop_name": "Rice",
    "confidence": 0.85,
    "suitability_score": 85.2
  },
  "prediction_metadata": {
    "model_version": "1.0",
    "prediction_time": "2024-01-01T12:00:00Z",
    "feature_count": 7,
    "available_crops": 22
  }
}
```

### 5. Get Available Crops

#### GET `/api/crop-prediction/crops`

Get a list of all crops that can be predicted by the model.

**Response:**
```json
{
  "success": true,
  "message": "Available crops retrieved successfully",
  "total_crops": 22,
  "crops": [
    {"crop_id": 0, "crop_name": "Rice"},
    {"crop_id": 1, "crop_name": "Maize"},
    {"crop_id": 2, "crop_name": "Chickpea"},
    {"crop_id": 3, "crop_name": "Kidney Beans"},
    {"crop_id": 4, "crop_name": "Pigeon Peas"},
    {"crop_id": 5, "crop_name": "Moth Beans"},
    {"crop_id": 6, "crop_name": "Mung Bean"},
    {"crop_id": 7, "crop_name": "Black Gram"},
    {"crop_id": 8, "crop_name": "Lentil"},
    {"crop_id": 9, "crop_name": "Pomegranate"},
    {"crop_id": 10, "crop_name": "Banana"},
    {"crop_id": 11, "crop_name": "Mango"},
    {"crop_id": 12, "crop_name": "Grapes"},
    {"crop_id": 13, "crop_name": "Watermelon"},
    {"crop_id": 14, "crop_name": "Muskmelon"},
    {"crop_id": 15, "crop_name": "Apple"},
    {"crop_id": 16, "crop_name": "Orange"},
    {"crop_id": 17, "crop_name": "Papaya"},
    {"crop_id": 18, "crop_name": "Coconut"},
    {"crop_id": 19, "crop_name": "Cotton"},
    {"crop_id": 20, "crop_name": "Jute"},
    {"crop_id": 21, "crop_name": "Coffee"}
  ],
  "model_info": {
    "n_features": 7,
    "feature_names": ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"],
    "model_type": "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>"
  }
}
```

### 6. Get Model Information

#### GET `/api/crop-prediction/model-info`

Get detailed information about the loaded ML model.

**Response:**
```json
{
  "success": true,
  "message": "Model information retrieved successfully",
  "model_info": {
    "status": "loaded",
    "model_type": "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>",
    "n_features": 7,
    "n_classes": 22,
    "feature_names": ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"],
    "available_crops": ["Rice", "Maize", "Chickpea", "..."],
    "model_path": "ml_models/saved_models/multiple_Crop_recommandation.joblib"
  }
}
```

## Error Responses

All endpoints return standardized error responses:

```json
{
  "error": true,
  "message": "Error description",
  "status_code": 400,
  "details": {
    "field": "nitrogen",
    "issue": "Value must be between 0 and 200"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid input parameters or validation errors
- `503 Service Unavailable`: ML model is not loaded or available
- `500 Internal Server Error`: Unexpected server errors

## Input Validation

### Parameter Ranges

- **Nitrogen**: 0-200 kg/ha
- **Phosphorus**: 0-150 kg/ha  
- **Potassium**: 0-300 kg/ha
- **Temperature**: -10 to 50°C
- **Humidity**: 0-100%
- **pH**: 3.0-10.0
- **Rainfall**: 0-3000mm

### Validation Rules

1. All parameters are required
2. Values must be numeric (int or float)
3. Values must be within specified ranges
4. No NaN or infinite values allowed
5. Values are rounded to 2 decimal places

## Usage Examples

### Python Example

```python
import requests

# API endpoint
url = "http://localhost:8000/api/crop-prediction/recommend"

# Request data
data = {
    "nitrogen": 90.0,
    "phosphorus": 42.0,
    "potassium": 43.0,
    "temperature": 20.87,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.93
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(f"Recommended crops: {len(result['recommendations'])}")
for crop in result['recommendations']:
    print(f"- {crop['crop_name']}: {crop['suitability_score']:.1f}%")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/crop-prediction/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "nitrogen": 90.0,
       "phosphorus": 42.0,
       "potassium": 43.0,
       "temperature": 20.87,
       "humidity": 82.0,
       "ph": 6.5,
       "rainfall": 202.93
     }'
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/api/crop-prediction/recommend', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    nitrogen: 90.0,
    phosphorus: 42.0,
    potassium: 43.0,
    temperature: 20.87,
    humidity: 82.0,
    ph: 6.5,
    rainfall: 202.93
  })
});

const result = await response.json();
console.log('Recommendations:', result.recommendations);
```

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## CORS Configuration

The API is configured to accept requests from:
- http://localhost:3000
- http://localhost:4028
- http://127.0.0.1:3000
- http://127.0.0.1:4028

## Model Information

- **Model Type**: K-Nearest Neighbors Classifier (scikit-learn)
- **Features**: 7 input parameters (N, P, K, temperature, humidity, pH, rainfall)
- **Classes**: 22 different crop types
- **Training Samples**: 1,760 samples
- **Model File**: `ml_models/saved_models/multiple_Crop_recommandation.joblib`

## Deployment

### Development Server

```bash
cd D:\new-folder1\AI-Crop-Yield-Prediction\agrismart\app
pip install -r requirements.txt
python main.py
```

### Production Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```