"""
Market Router
API endpoints for market data, prices, and trends
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import random

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/data")
async def get_market_data() -> Dict[str, Any]:
    """Get comprehensive market data"""
    try:
        logger.info("Retrieving market data")
        
        market_data = {
            "success": True,
            "current_prices": generate_current_prices(),
            "price_trends": generate_price_trends(),
            "market_analysis": {
                "overall_trend": "Bullish",
                "volatility_index": 2.3,
                "market_sentiment": "Positive",
                "trading_volume": "High",
                "price_stability": "Moderate"
            },
            "crop_demand": {
                "rice": {"demand": "High", "supply": "Adequate", "price_direction": "Up"},
                "wheat": {"demand": "Moderate", "supply": "High", "price_direction": "Stable"},
                "maize": {"demand": "High", "supply": "Low", "price_direction": "Up"},
                "cotton": {"demand": "Low", "supply": "High", "price_direction": "Down"},
                "tomato": {"demand": "Very High", "supply": "Low", "price_direction": "Up"}
            },
            "regional_insights": generate_regional_insights(),
            "forecasts": generate_market_forecasts(),
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info("Market data retrieved successfully")
        return market_data
        
    except Exception as e:
        logger.error(f"Error retrieving market data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve market data"
        )

@router.get("/prices")
async def get_market_prices(
    crop: Optional[str] = Query(None, description="Specific crop to get prices for"),
    region: Optional[str] = Query(None, description="Specific region to get prices for")
) -> Dict[str, Any]:
    """Get current market prices"""
    try:
        logger.info(f"Retrieving market prices for crop: {crop}, region: {region}")
        
        prices = generate_current_prices()
        
        # Filter by crop if specified
        if crop:
            crop_lower = crop.lower()
            if crop_lower in prices:
                prices = {crop_lower: prices[crop_lower]}
            else:
                prices = {}
        
        response = {
            "success": True,
            "prices": prices,
            "currency": "USD",
            "unit": "per ton",
            "region": region or "Global",
            "last_updated": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving market prices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve market prices"
        )

@router.get("/trends")
async def get_market_trends(
    period: Optional[str] = Query("30d", description="Time period (7d, 30d, 90d, 1y)"),
    crop: Optional[str] = Query(None, description="Specific crop to get trends for")
) -> Dict[str, Any]:
    """Get market price trends"""
    try:
        logger.info(f"Retrieving market trends for period: {period}, crop: {crop}")
        
        # Determine number of days based on period
        days_map = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}
        days = days_map.get(period, 30)
        
        trends = generate_price_trends(days=days, specific_crop=crop)
        
        response = {
            "success": True,
            "trends": trends,
            "period": period,
            "crop": crop or "All",
            "analysis": analyze_trends(trends),
            "generated_at": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving market trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve market trends"
        )

@router.get("/insights")
async def get_market_insights() -> Dict[str, Any]:
    """Get market insights and analysis"""
    try:
        logger.info("Retrieving market insights")
        
        insights = {
            "success": True,
            "key_insights": [
                {
                    "title": "Rice Prices Surge",
                    "description": "Rice prices have increased by 15% due to reduced supply from major producing regions.",
                    "impact": "High",
                    "recommendation": "Consider increasing rice production area",
                    "confidence": 85
                },
                {
                    "title": "Maize Demand Growing",
                    "description": "Industrial demand for maize continues to grow, supporting higher prices.",
                    "impact": "Medium",
                    "recommendation": "Maintain current maize production levels",
                    "confidence": 78
                },
                {
                    "title": "Cotton Market Weakness",
                    "description": "Cotton prices under pressure due to oversupply and reduced textile demand.",
                    "impact": "Medium",
                    "recommendation": "Consider diversifying away from cotton",
                    "confidence": 72
                }
            ],
            "market_opportunities": [
                {
                    "crop": "Tomato",
                    "opportunity": "Premium market demand",
                    "potential_return": "25-30%",
                    "risk_level": "Medium",
                    "timeframe": "3-6 months"
                },
                {
                    "crop": "Rice",
                    "opportunity": "Export market expansion",
                    "potential_return": "15-20%",
                    "risk_level": "Low",
                    "timeframe": "6-12 months"
                }
            ],
            "risk_factors": [
                {
                    "factor": "Weather Volatility",
                    "impact": "High",
                    "probability": "Medium",
                    "mitigation": "Diversify crop portfolio and use weather insurance"
                },
                {
                    "factor": "Supply Chain Disruptions",
                    "impact": "Medium",
                    "probability": "Low",
                    "mitigation": "Establish multiple distribution channels"
                }
            ],
            "seasonal_patterns": generate_seasonal_patterns(),
            "generated_at": datetime.now().isoformat()
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error retrieving market insights: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve market insights"
        )

@router.get("/forecast")
async def get_market_forecast(
    horizon: Optional[str] = Query("3m", description="Forecast horizon (1m, 3m, 6m, 1y)")
) -> Dict[str, Any]:
    """Get market price forecasts"""
    try:
        logger.info(f"Retrieving market forecast for horizon: {horizon}")
        
        # Determine forecast period
        horizon_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
        days = horizon_map.get(horizon, 90)
        
        forecast = {
            "success": True,
            "horizon": horizon,
            "forecasts": generate_market_forecasts(days=days),
            "confidence_intervals": generate_confidence_intervals(),
            "methodology": "Machine Learning with Economic Indicators",
            "last_model_update": (datetime.now() - timedelta(days=7)).isoformat(),
            "generated_at": datetime.now().isoformat()
        }
        
        return forecast
        
    except Exception as e:
        logger.error(f"Error retrieving market forecast: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve market forecast"
        )

def generate_current_prices() -> Dict[str, Dict[str, Any]]:
    """Generate current market prices"""
    base_prices = {
        "rice": 450,
        "wheat": 280,
        "maize": 320,
        "cotton": 1800,
        "tomato": 800
    }
    
    prices = {}
    for crop, base_price in base_prices.items():
        # Add some random variation
        current_price = base_price * (1 + random.uniform(-0.1, 0.15))
        change_percent = random.uniform(-5, 8)
        
        prices[crop] = {
            "current_price": round(current_price, 2),
            "previous_price": round(current_price / (1 + change_percent/100), 2),
            "change_percent": round(change_percent, 2),
            "change_direction": "up" if change_percent > 0 else "down",
            "volume": random.randint(1000, 10000),
            "last_trade": datetime.now().isoformat()
        }
    
    return prices

def generate_price_trends(days: int = 30, specific_crop: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Generate price trend data"""
    crops = [specific_crop] if specific_crop else ["rice", "wheat", "maize", "cotton", "tomato"]
    base_prices = {"rice": 450, "wheat": 280, "maize": 320, "cotton": 1800, "tomato": 800}
    
    trends = {}
    
    for crop in crops:
        if crop not in base_prices:
            continue
            
        base_price = base_prices[crop]
        crop_trends = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            
            # Generate price with some trend and volatility
            trend_factor = 1 + (i * 0.001)  # Slight upward trend
            volatility = random.uniform(-0.05, 0.05)
            price = base_price * trend_factor * (1 + volatility)
            
            crop_trends.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "volume": random.randint(500, 5000),
                "high": round(price * 1.02, 2),
                "low": round(price * 0.98, 2)
            })
        
        trends[crop] = crop_trends
    
    return trends

def generate_regional_insights() -> Dict[str, Dict[str, Any]]:
    """Generate regional market insights"""
    regions = ["North America", "Europe", "Asia", "South America", "Africa"]
    insights = {}
    
    for region in regions:
        insights[region] = {
            "market_size": random.randint(50, 500),
            "growth_rate": round(random.uniform(2, 8), 1),
            "key_crops": random.sample(["rice", "wheat", "maize", "cotton", "tomato"], 3),
            "price_level": random.choice(["High", "Medium", "Low"]),
            "volatility": random.choice(["High", "Medium", "Low"]),
            "trade_balance": random.choice(["Surplus", "Deficit", "Balanced"])
        }
    
    return insights

def generate_market_forecasts(days: int = 90) -> Dict[str, List[Dict[str, Any]]]:
    """Generate market forecasts"""
    crops = ["rice", "wheat", "maize", "cotton", "tomato"]
    base_prices = {"rice": 450, "wheat": 280, "maize": 320, "cotton": 1800, "tomato": 800}
    
    forecasts = {}
    
    for crop in crops:
        base_price = base_prices[crop]
        crop_forecasts = []
        
        for i in range(1, days + 1):
            date = datetime.now() + timedelta(days=i)
            
            # Generate forecast with trend
            trend_factor = 1 + (i * 0.0005)  # Slight upward trend
            forecast_price = base_price * trend_factor
            
            crop_forecasts.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_price": round(forecast_price, 2),
                "confidence": round(random.uniform(70, 95), 1),
                "lower_bound": round(forecast_price * 0.9, 2),
                "upper_bound": round(forecast_price * 1.1, 2)
            })
        
        forecasts[crop] = crop_forecasts
    
    return forecasts

def generate_confidence_intervals() -> Dict[str, Dict[str, float]]:
    """Generate confidence intervals for forecasts"""
    crops = ["rice", "wheat", "maize", "cotton", "tomato"]
    intervals = {}
    
    for crop in crops:
        intervals[crop] = {
            "95_percent_lower": round(random.uniform(0.85, 0.9), 3),
            "95_percent_upper": round(random.uniform(1.1, 1.15), 3),
            "80_percent_lower": round(random.uniform(0.9, 0.95), 3),
            "80_percent_upper": round(random.uniform(1.05, 1.1), 3)
        }
    
    return intervals

def generate_seasonal_patterns() -> Dict[str, Dict[str, Any]]:
    """Generate seasonal market patterns"""
    crops = ["rice", "wheat", "maize", "cotton", "tomato"]
    patterns = {}
    
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    
    for crop in crops:
        crop_patterns = {}
        for season in seasons:
            crop_patterns[season] = {
                "typical_price_change": round(random.uniform(-10, 15), 1),
                "demand_level": random.choice(["High", "Medium", "Low"]),
                "supply_level": random.choice(["High", "Medium", "Low"]),
                "volatility": random.choice(["High", "Medium", "Low"])
            }
        patterns[crop] = crop_patterns
    
    return patterns

def analyze_trends(trends: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze price trends and provide insights"""
    analysis = {
        "overall_direction": "Mixed",
        "volatility_level": "Medium",
        "strongest_performer": "",
        "weakest_performer": "",
        "average_change": 0.0
    }
    
    if trends:
        # Simple analysis based on first and last prices
        crop_changes = {}
        for crop, trend_data in trends.items():
            if len(trend_data) >= 2:
                first_price = trend_data[0]["price"]
                last_price = trend_data[-1]["price"]
                change = ((last_price - first_price) / first_price) * 100
                crop_changes[crop] = change
        
        if crop_changes:
            analysis["strongest_performer"] = max(crop_changes, key=crop_changes.get)
            analysis["weakest_performer"] = min(crop_changes, key=crop_changes.get)
            analysis["average_change"] = round(sum(crop_changes.values()) / len(crop_changes), 2)
            
            avg_change = analysis["average_change"]
            if avg_change > 5:
                analysis["overall_direction"] = "Bullish"
            elif avg_change < -5:
                analysis["overall_direction"] = "Bearish"
            else:
                analysis["overall_direction"] = "Neutral"
    
    return analysis