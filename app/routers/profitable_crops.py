"""
Profitable Crops Router
API endpoints for profitable crop prediction - Frontend compatibility layer
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime

from ..models.schemas import (
    CropPredictionRequest,
    CropPredictionResponse,
    CropRecommendation,
    ErrorResponse
)
from ..services.model_service import ModelService
from ..services.cost_analysis_service import CostAnalysisService
from ..services.market_analysis_service import MarketAnalysisService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
model_service = ModelService()
cost_service = CostAnalysisService()
market_service = MarketAnalysisService()

@router.post(
    "/predict-multiple",
    summary="Predict multiple profitable crops",
    description="Predict the most profitable crops based on soil and environmental parameters"
)
async def predict_multiple_crops(
    request: CropPredictionRequest
) -> Dict[str, Any]:
    """
    Predict multiple profitable crops based on input parameters
    
    Args:
        request: Crop prediction request with soil and environmental data
        
    Returns:
        Dict with profitable crop predictions in frontend-compatible format
    """
    try:
        logger.info(f"Received profitable crops prediction request: {request.dict()}")
        
        # Prepare features from request
        features = model_service.prepare_features_from_request(request.dict())
        
        # Get multiple crop recommendations with enhanced profit analysis
        recommendations = model_service.predict_multiple_crops(
            features, 
            top_k=5, 
            farm_size=request.farm_size
        )
        
        # Convert to frontend-compatible format with enhanced profit analysis
        profitable_crops = []
        
        # Prepare environmental factors for cost and market analysis
        environmental_factors = {
            "temperature": request.temperature,
            "rainfall": request.rainfall,
            "humidity": request.humidity,
            "ph": request.ph,
            "nitrogen": request.nitrogen,
            "phosphorus": request.phosphorus,
            "potassium": request.potassium
        }
        
        for i, rec in enumerate(recommendations):
            crop_name = rec["crop_name"]
            
            # Get comprehensive cost analysis
            cost_breakdown = cost_service.analyze_crop_costs(
                crop_name=crop_name,
                farm_size=request.farm_size,
                region_type="rural",  # Default to rural, could be made configurable
                environmental_factors=environmental_factors
            )
            
            # Get market analysis
            market_analysis = market_service.analyze_crop_market(
                crop_name=crop_name,
                environmental_factors=environmental_factors
            )
            
            # Calculate enhanced profitability metrics
            estimated_yield = rec.get("estimated_yield", 0)
            market_price = market_analysis.current_price
            total_revenue = estimated_yield * market_price * request.farm_size
            total_cost = cost_breakdown.total_cost_per_hectare * request.farm_size
            net_profit = total_revenue - total_cost
            
            # Calculate profit margin and ROI
            profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
            roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
            
            crop_data = {
                "rank": i + 1,
                "crop_name": crop_name,
                "crop_id": rec["crop_id"],
                "suitability_score": rec["suitability_score"],
                "confidence": rec["confidence"],
                "profitability_index": rec.get("profitability_score", rec["suitability_score"] / 100.0),
                
                # Enhanced yield and pricing
                "estimated_yield": round(estimated_yield, 2),
                "market_price": round(market_price, 2),
                "price_trend": market_analysis.price_trend.value,
                "price_volatility": market_analysis.volatility.value,
                
                # Enhanced financial metrics
                "total_revenue": round(total_revenue, 2),
                "total_cost": round(total_cost, 2),
                "net_profit": round(net_profit, 2),
                "profit_margin": round(profit_margin, 1),
                "roi_percentage": round(roi, 1),
                "break_even_yield": round(cost_breakdown.break_even_yield, 2),
                
                # Cost breakdown
                "cost_breakdown": {
                    "seeds": cost_breakdown.cost_categories.get("seeds", 0),
                    "fertilizers": cost_breakdown.cost_categories.get("fertilizers", 0),
                    "pesticides": cost_breakdown.cost_categories.get("pesticides", 0),
                    "labor": cost_breakdown.cost_categories.get("labor", 0),
                    "irrigation": cost_breakdown.cost_categories.get("irrigation", 0),
                    "other": sum([
                        cost_breakdown.cost_categories.get("machinery", 0),
                        cost_breakdown.cost_categories.get("transportation", 0),
                        cost_breakdown.cost_categories.get("storage", 0),
                        cost_breakdown.cost_categories.get("insurance", 0)
                    ])
                },
                
                # Market insights
                "market_insights": {
                    "demand_level": market_analysis.demand_score,
                    "seasonal_factor": 1.0,  # Default seasonal factor
                    "export_potential": market_analysis.export_demand,
                    "risk_level": market_analysis.risk_level
                },
                
                # Optimization potential
                "cost_efficiency_score": cost_breakdown.cost_efficiency_score,
                "optimization_potential": cost_breakdown.optimization_potential,
                
                # Qualitative assessment
                "profit_potential": (
                    "High" if net_profit > 100000 and roi > 50 else
                    "Medium" if net_profit > 50000 and roi > 25 else
                    "Low"
                ),
                "investment_grade": (
                    "Excellent" if roi > 75 and profit_margin > 30 else
                    "Good" if roi > 50 and profit_margin > 20 else
                    "Fair" if roi > 25 and profit_margin > 10 else
                    "Poor"
                )
            }
            profitable_crops.append(crop_data)
        
        # Get prediction metadata
        metadata = model_service.get_prediction_metadata()
        
        # Calculate overall market insights
        if profitable_crops:
            top_crop = profitable_crops[0]
            avg_roi = sum([crop["roi_percentage"] for crop in profitable_crops]) / len(profitable_crops)
            high_profit_crops = [crop for crop in profitable_crops if crop["profit_potential"] == "High"]
            
            # Get cost optimization recommendations for top crop
            top_crop_cost_breakdown = cost_service.analyze_crop_costs(
                crop_name=top_crop["crop_name"],
                farm_size=request.farm_size,
                region_type="rural",
                environmental_factors=environmental_factors
            )
            cost_optimizations = cost_service.get_cost_optimization_recommendations(top_crop_cost_breakdown)
            
            # Enhanced recommendations
            enhanced_recommendations = {
                "top_crop": top_crop,
                "alternative_crops": profitable_crops[1:4] if len(profitable_crops) > 1 else [],
                "high_profit_crops": high_profit_crops,
                "average_roi": round(avg_roi, 1),
                "total_potential_profit": sum([crop["net_profit"] for crop in profitable_crops]),
                
                # Market-based advice
                "seasonal_advice": _generate_seasonal_advice(profitable_crops, environmental_factors),
                "market_timing": _generate_market_timing_advice(profitable_crops),
                "risk_assessment": _assess_overall_risk(profitable_crops),
                
                # Cost optimization insights
                "cost_optimization": {
                    "potential_savings": sum([opt.savings_amount for opt in cost_optimizations[:3]]),
                    "top_optimizations": [
                        {
                            "category": opt.category,
                            "method": opt.optimization_method,
                            "savings": opt.savings_amount,
                            "difficulty": opt.implementation_difficulty
                        }
                        for opt in cost_optimizations[:3]
                    ]
                },
                
                # Investment insights
                "investment_insights": {
                    "best_roi_crop": max(profitable_crops, key=lambda x: x["roi_percentage"])["crop_name"],
                    "lowest_risk_crop": min(profitable_crops, key=lambda x: x.get("market_insights", {}).get("risk_level", "Medium") == "Low")["crop_name"],
                    "diversification_score": len(set([crop["crop_name"] for crop in profitable_crops])) / len(profitable_crops) * 100
                }
            }
        else:
            enhanced_recommendations = {
                "top_crop": None,
                "alternative_crops": [],
                "seasonal_advice": "Consider adjusting soil conditions or exploring different crop varieties",
                "risk_assessment": "High - No suitable crops found for current conditions"
            }
        
        # Enhanced frontend-compatible response format
        response = {
            "success": True,
            "message": "Profitable crops predicted successfully with enhanced market analysis",
            "model_used": "Enhanced ML with Market & Cost Analysis",
            "input_parameters": request.dict(),
            "profitable_crops": profitable_crops,
            "total_crops": len(profitable_crops),
            "prediction_metadata": {
                "model_version": metadata.get("model_version", "2.0"),
                "prediction_time": datetime.utcnow().isoformat() + "Z",
                "feature_count": metadata.get("feature_count", 7),
                "available_crops": metadata.get("available_crops", 22),
                "algorithm": "Enhanced K-NN with Profit Optimization",
                "analysis_components": ["Crop Suitability", "Cost Analysis", "Market Analysis", "Profit Optimization"]
            },
            "recommendations": enhanced_recommendations,
            
            # Additional insights
            "market_summary": {
                "overall_market_condition": "Favorable" if profitable_crops and len(high_profit_crops) > 0 else "Challenging",
                "profit_opportunity_score": min(100, max(0, avg_roi)) if profitable_crops else 0,
                "cost_efficiency_average": sum([crop["cost_efficiency_score"] for crop in profitable_crops]) / len(profitable_crops) if profitable_crops else 0
            }
        }
        
        logger.info(f"Successfully generated {len(profitable_crops)} profitable crop recommendations")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
    except Exception as e:
        logger.error(f"Error in profitable crops prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict profitable crops: {str(e)}"
        )

def _generate_seasonal_advice(profitable_crops: List[Dict], environmental_factors: Dict) -> str:
    """Generate seasonal advice based on crop recommendations and environmental factors"""
    current_month = datetime.now().month
    temperature = environmental_factors.get("temperature", 25)
    rainfall = environmental_factors.get("rainfall", 100)
    
    if current_month in [3, 4, 5]:  # Spring
        if temperature > 30:
            return "Spring planting season - consider heat-tolerant varieties. Monitor irrigation needs."
        else:
            return "Optimal spring planting conditions. Consider early-season crops for better market prices."
    elif current_month in [6, 7, 8]:  # Summer
        if rainfall < 50:
            return "Summer season with low rainfall - prioritize drought-resistant crops and efficient irrigation."
        else:
            return "Good summer growing conditions. Focus on crops with high summer demand."
    elif current_month in [9, 10, 11]:  # Fall
        return "Post-harvest season - consider storage crops and plan for winter varieties."
    else:  # Winter
        return "Winter season - focus on cool-season crops and greenhouse cultivation if applicable."

def _generate_market_timing_advice(profitable_crops: List[Dict]) -> str:
    """Generate market timing advice based on crop characteristics"""
    if not profitable_crops:
        return "No suitable crops identified for current conditions."
    
    high_roi_crops = [crop for crop in profitable_crops if crop["roi_percentage"] > 50]
    if high_roi_crops:
        return f"Excellent timing for high-ROI crops like {high_roi_crops[0]['crop_name']}. Consider immediate planting."
    
    medium_roi_crops = [crop for crop in profitable_crops if 25 <= crop["roi_percentage"] <= 50]
    if medium_roi_crops:
        return f"Moderate market conditions. {medium_roi_crops[0]['crop_name']} shows steady profit potential."
    
    return "Market conditions suggest careful timing. Consider waiting for better price conditions."

def _assess_overall_risk(profitable_crops: List[Dict]) -> str:
    """Assess overall investment risk based on crop portfolio"""
    if not profitable_crops:
        return "High - No profitable crops identified"
    
    avg_roi = sum([crop["roi_percentage"] for crop in profitable_crops]) / len(profitable_crops)
    high_profit_count = len([crop for crop in profitable_crops if crop["profit_potential"] == "High"])
    
    if avg_roi > 60 and high_profit_count >= 2:
        return "Low - Strong profit potential with multiple high-return options"
    elif avg_roi > 40 and high_profit_count >= 1:
        return "Medium - Good profit potential with manageable risk"
    elif avg_roi > 20:
        return "Medium-High - Moderate returns, consider diversification"
    else:
        return "High - Low profit margins, careful planning required"

@router.get(
    "/available-crops",
    summary="Get available profitable crops",
    description="Get a list of all crops available for profitability analysis"
)
async def get_available_crops() -> Dict[str, Any]:
    """
    Get list of all available crops for profitability analysis
    
    Returns:
        Dict with available crops information
    """
    try:
        model_info = model_service.get_model_info()
        if model_info.get("status") != "loaded":
            raise HTTPException(status_code=503, detail="Model not available.")
        
        crops = [
            {"crop_id": i, "crop_name": name}
            for i, name in model_service.crop_mapping.items()
        ]
        
        # Add profitability metadata to each crop
        enhanced_crops = []
        for crop in crops:
            enhanced_crop = {
                **crop,
                "category": "Cash Crop" if crop["crop_id"] in [20, 21, 19, 18] else "Food Crop",
                "growing_season": "Kharif" if crop["crop_id"] in [0, 1, 2, 3] else "Rabi",
                "market_demand": "High" if crop["crop_id"] in [0, 1, 20, 21] else "Medium"
            }
            enhanced_crops.append(enhanced_crop)
        
        return {
            "success": True,
            "message": "Available crops retrieved successfully",
            "crops": enhanced_crops,
            "total_crops": len(enhanced_crops),
            "categories": {
                "cash_crops": len([c for c in enhanced_crops if c["category"] == "Cash Crop"]),
                "food_crops": len([c for c in enhanced_crops if c["category"] == "Food Crop"])
            }
        }
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="ML model file not found.")
    except Exception as e:
        logger.error(f"Error retrieving available crops: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve available crops: {str(e)}"
        )

@router.get(
    "/market-trends",
    summary="Get market trends for profitable crops",
    description="Get current market trends and pricing information"
)
async def get_market_trends() -> Dict[str, Any]:
    """
    Get market trends and pricing information for crops
    
    Returns:
        Dict with market trends data
    """
    try:
        # Mock market trends data - in production, this would come from a market data API
        market_trends = {
            "success": True,
            "message": "Market trends retrieved successfully",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "trending_crops": [
                {"crop_name": "Rice", "price_trend": "up", "demand": "high", "price_change": "+5.2%"},
                {"crop_name": "Wheat", "price_trend": "stable", "demand": "medium", "price_change": "+1.1%"},
                {"crop_name": "Cotton", "price_trend": "up", "demand": "high", "price_change": "+8.7%"},
                {"crop_name": "Sugarcane", "price_trend": "down", "demand": "medium", "price_change": "-2.3%"},
                {"crop_name": "Maize", "price_trend": "up", "demand": "high", "price_change": "+3.4%"}
            ],
            "market_insights": {
                "best_season": "Current season shows high demand for cash crops",
                "price_volatility": "Low to medium volatility expected",
                "export_opportunities": "Cotton and rice showing strong export potential"
            }
        }
        
        return market_trends
        
    except Exception as e:
        logger.error(f"Error retrieving market trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve market trends: {str(e)}"
        )