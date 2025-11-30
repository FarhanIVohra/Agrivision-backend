"""
AgriSmart Backend - Main FastAPI Application
Crop Yield Prediction and Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Import routers
from .routers import crop_prediction, profitable_crops, pest_detection, weather, soil_health, predictions, irrigation, dashboard, market, alerts, disease_outbreak, insights_dashboard, disease_database, environmental_monitoring, auth, reports

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for loading ML models on startup"""
    print("🚀 Server started — ML models will be loaded on first request.")
    yield
    # Cleanup on shutdown
    print("🔄 Application shutting down")

# Create FastAPI application with lifespan manager
app = FastAPI(
    title="AgriSmart API",
    description="AI-powered crop yield prediction and recommendation system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4028", "http://localhost:4029", "http://127.0.0.1:3000", "http://127.0.0.1:4028", "http://127.0.0.1:4029"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(crop_prediction.router, prefix="/api/crop-prediction", tags=["Crop Prediction"])
app.include_router(profitable_crops.router, prefix="/api/profitable-crops", tags=["Profitable Crops"])
app.include_router(pest_detection.router, prefix="/api/pest-detection", tags=["Pest Detection"])
app.include_router(weather.router, prefix="/api/weather", tags=["Weather"])
app.include_router(soil_health.router, prefix="/api/soil-health", tags=["Soil Health"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(irrigation.router, prefix="/api/irrigation", tags=["Irrigation"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(disease_outbreak.router, prefix="/api/disease-outbreak", tags=["Disease Outbreak"])
app.include_router(disease_database.router, prefix="/api/disease-database", tags=["Disease Database"])
app.include_router(environmental_monitoring.router, prefix="/api/environmental-monitoring", tags=["Environmental Monitoring"])
app.include_router(insights_dashboard.router, prefix="/api/insights", tags=["Actionable Insights"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AgriSmart API - Crop Yield Prediction System",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/api/docs",
            "crop_prediction": "/api/crop-prediction",
            "profitable_crops": "/api/profitable-crops",
            "pest_detection": "/api/pest-detection",
            "weather": "/api/weather",
            "soil_health": "/api/soil-health",
            "predictions": "/api/predictions",
            "irrigation": "/api/irrigation",
            "dashboard": "/api/dashboard",
            "market": "/api/market",
            "alerts": "/api/alerts",
            "disease_outbreak": "/api/disease-outbreak",
            "disease_database": "/api/disease-database",
            "environmental_monitoring": "/api/environmental-monitoring",
            "insights": "/api/insights",
            "reports": "/api/reports"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_version": "1.0.0"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error occurred",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )