# AgriSmart Backend Architecture

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                        │
│                     http://localhost:4028                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP Requests
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                              │
│                  http://localhost:8000                         │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   CORS Layer    │  │  Authentication │  │   API Routes    │ │
│  │   Middleware    │  │   JWT Tokens    │  │   /api/...      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API ENDPOINTS                               │
│                                                                 │
│  /api/auth/          /api/soil-health/    /api/weather/        │
│  ├── login           ├── analyze          ├── current          │
│  ├── register        └── recommendations  ├── forecast         │
│  └── me                                   ├── alerts           │
│                                           └── insights         │
│  /api/crop-prediction/   /api/profitable-crops/               │
│  ├── recommend           ├── predict                          │
│  └── crops               └── analysis                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA LAYER                                   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Supabase      │  │   Mock Data     │  │   ML Models     │ │
│  │   PostgreSQL    │  │   Fallback      │  │   Logic         │ │
│  │   - Users       │  │   - Demo Users  │  │   - Crop Rec    │ │
│  │   - Predictions │  │   - Test Data   │  │   - Soil Score  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow

### 1. User Authentication Flow
```
User Login → Frontend → POST /api/auth/login → Backend
                                              ↓
                                         Validate Credentials
                                              ↓
                                    Check Supabase → Fallback to Mock
                                              ↓
                                        Generate JWT Token
                                              ↓
                                    Return Token + User Info
                                              ↓
Frontend Stores Token → Use in Future Requests
```

### 2. Soil Health Analysis Flow
```
User Input → Frontend Form → POST /api/soil-health/analyze
                                              ↓
                                    Receive Soil Parameters
                                    (N, P, K, pH, moisture)
                                              ↓
                                    Calculate Health Score
                                              ↓
                                    Generate Recommendations
                                              ↓
                                    Return Analysis Results
                                              ↓
                            Frontend Displays Charts & Insights
```

### 3. Weather Data Flow
```
Dashboard Load → GET /api/weather/current → Backend
                                              ↓
                                    Mock Weather Service
                                    (Location: Nadiad, IN)
                                              ↓
                                    Generate Current Conditions
                                              ↓
                                    Calculate Agricultural Insights
                                              ↓
                                    Return Weather + Recommendations
                                              ↓
                            Frontend Updates Weather Cards
```

### 4. Crop Prediction Flow
```
Soil Data → POST /api/crop-prediction/recommend → Backend
                                              ↓
                                    Analyze Input Parameters
                                    (Nutrients, Climate, pH)
                                              ↓
                                    Score Against Crop Database
                                              ↓
                                    Rank Suitable Crops
                                              ↓
                                    Calculate Yield Estimates
                                              ↓
                                    Return Recommendations
                                              ↓
                            Frontend Shows Crop Suggestions
```

## 🛠️ Technical Stack

### Backend Components
- **Framework**: FastAPI (Python)
- **Authentication**: JWT Tokens
- **Database**: Supabase PostgreSQL + Mock Fallback
- **CORS**: Configured for localhost:4028
- **Environment**: python-dotenv for config

### API Structure
```
app/
├── simple_main.py          # Main FastAPI application
├── apis/
│   ├── auth_simple.py      # Authentication endpoints
│   ├── soil_health_simple.py # Soil analysis
│   ├── weather_simple.py   # Weather data
│   └── crop_prediction.py  # ML crop recommendations
└── .env                    # Environment variables
```

### Environment Variables
```
SUPABASE_URL=https://hdyinywsplylrpudcwpe.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIs...
SECRET_KEY=7kxOqreLcj5O3UK53HUEwRo79DW...
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 🔐 Security Features

1. **JWT Authentication**
   - Secure token-based auth
   - 30-minute expiration
   - User session management

2. **CORS Protection**
   - Configured for specific origins
   - Credential support enabled

3. **Input Validation**
   - Pydantic models for all endpoints
   - Type checking and validation

4. **Error Handling**
   - Graceful fallbacks to mock data
   - Comprehensive error responses

## 🚀 Deployment Architecture

### Current Setup (Development)
```
Frontend: Vite Dev Server (Port 4028)
Backend: Uvicorn Server (Port 8000)
Database: Supabase Cloud + Local Mock
```

### Production Ready Features
- Environment-based configuration
- Database connection pooling
- API documentation at `/docs`
- Health check endpoints
- Scalable FastAPI structure

## 📊 Data Models

### User Model
```python
{
    "id": "string",
    "email": "string",
    "full_name": "string", 
    "password_hash": "string",
    "created_at": "datetime"
}
```

### Soil Analysis Model
```python
{
    "nitrogen": float,
    "phosphorus": float,
    "potassium": float,
    "ph": float,
    "organic_matter": float,
    "moisture": float
}
```

### Weather Model
```python
{
    "temperature": float,
    "humidity": float,
    "wind_speed": float,
    "rainfall": float,
    "condition": "string"
}
```

This architecture provides a robust, scalable foundation for the AgriSmart platform with clear separation of concerns and comprehensive API coverage.
