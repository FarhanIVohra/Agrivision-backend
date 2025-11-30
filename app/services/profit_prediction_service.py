"""
Advanced Profit Prediction Service
Focuses on profitability analysis rather than yield prediction
Integrates market conditions, input costs, and profit optimization
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class MarketTrend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    STABLE = "stable"
    VOLATILE = "volatile"

class SeasonalDemand(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PEAK = "peak"

@dataclass
class MarketConditions:
    """Market conditions affecting crop profitability"""
    price_trend: MarketTrend
    demand_level: SeasonalDemand
    supply_stability: float  # 0.0 to 1.0
    export_potential: float  # 0.0 to 1.0
    competition_level: float  # 0.0 to 1.0
    seasonal_multiplier: float  # Price adjustment for season

@dataclass
class InputCosts:
    """Comprehensive input cost analysis"""
    seed_cost_per_hectare: float
    fertilizer_cost_per_hectare: float
    pesticide_cost_per_hectare: float
    irrigation_cost_per_hectare: float
    labor_cost_per_hectare: float
    machinery_cost_per_hectare: float
    transportation_cost_per_hectare: float
    storage_cost_per_hectare: float
    insurance_cost_per_hectare: float
    land_preparation_cost: float

@dataclass
class ProfitabilityMetrics:
    """Comprehensive profitability analysis"""
    gross_revenue_per_hectare: float
    total_costs_per_hectare: float
    net_profit_per_hectare: float
    profit_margin_percentage: float
    roi_percentage: float
    break_even_price: float
    risk_adjusted_profit: float
    profitability_score: float  # 0-100
    investment_recovery_months: int

class ProfitPredictionService:
    """Advanced service for profit-focused crop recommendations"""
    
    def __init__(self):
        self.market_data = self._initialize_market_data()
        self.cost_structures = self._initialize_cost_structures()
        self.seasonal_factors = self._initialize_seasonal_factors()
        
    def _initialize_market_data(self) -> Dict[str, MarketConditions]:
        """Initialize market conditions for each crop"""
        return {
            "Rice": MarketConditions(
                price_trend=MarketTrend.STABLE,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.8,
                export_potential=0.6,
                competition_level=0.7,
                seasonal_multiplier=1.0
            ),
            "Maize": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.7,
                export_potential=0.8,
                competition_level=0.6,
                seasonal_multiplier=1.1
            ),
            "Chickpea": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.MEDIUM,
                supply_stability=0.6,
                export_potential=0.9,
                competition_level=0.5,
                seasonal_multiplier=1.3
            ),
            "Kidney Beans": MarketConditions(
                price_trend=MarketTrend.STABLE,
                demand_level=SeasonalDemand.MEDIUM,
                supply_stability=0.7,
                export_potential=0.8,
                competition_level=0.5,
                seasonal_multiplier=1.2
            ),
            "Pigeon Peas": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.MEDIUM,
                supply_stability=0.6,
                export_potential=0.7,
                competition_level=0.6,
                seasonal_multiplier=1.1
            ),
            "Moth Beans": MarketConditions(
                price_trend=MarketTrend.VOLATILE,
                demand_level=SeasonalDemand.LOW,
                supply_stability=0.5,
                export_potential=0.4,
                competition_level=0.8,
                seasonal_multiplier=0.9
            ),
            "Mung Bean": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.6,
                export_potential=0.9,
                competition_level=0.4,
                seasonal_multiplier=1.4
            ),
            "Black Gram": MarketConditions(
                price_trend=MarketTrend.STABLE,
                demand_level=SeasonalDemand.MEDIUM,
                supply_stability=0.7,
                export_potential=0.6,
                competition_level=0.6,
                seasonal_multiplier=1.2
            ),
            "Lentil": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.6,
                export_potential=0.8,
                competition_level=0.5,
                seasonal_multiplier=1.3
            ),
            "Pomegranate": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.PEAK,
                supply_stability=0.5,
                export_potential=0.9,
                competition_level=0.3,
                seasonal_multiplier=1.6
            ),
            "Banana": MarketConditions(
                price_trend=MarketTrend.STABLE,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.8,
                export_potential=0.7,
                competition_level=0.7,
                seasonal_multiplier=1.0
            ),
            "Mango": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.PEAK,
                supply_stability=0.6,
                export_potential=0.9,
                competition_level=0.4,
                seasonal_multiplier=1.5
            ),
            "Grapes": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.7,
                export_potential=0.8,
                competition_level=0.5,
                seasonal_multiplier=1.3
            ),
            "Watermelon": MarketConditions(
                price_trend=MarketTrend.VOLATILE,
                demand_level=SeasonalDemand.PEAK,
                supply_stability=0.5,
                export_potential=0.3,
                competition_level=0.8,
                seasonal_multiplier=1.2
            ),
            "Muskmelon": MarketConditions(
                price_trend=MarketTrend.VOLATILE,
                demand_level=SeasonalDemand.MEDIUM,
                supply_stability=0.5,
                export_potential=0.4,
                competition_level=0.7,
                seasonal_multiplier=1.1
            ),
            "Apple": MarketConditions(
                price_trend=MarketTrend.STABLE,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.8,
                export_potential=0.6,
                competition_level=0.6,
                seasonal_multiplier=1.1
            ),
            "Orange": MarketConditions(
                price_trend=MarketTrend.STABLE,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.7,
                export_potential=0.7,
                competition_level=0.6,
                seasonal_multiplier=1.0
            ),
            "Papaya": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.6,
                export_potential=0.8,
                competition_level=0.5,
                seasonal_multiplier=1.2
            ),
            "Coconut": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.8,
                export_potential=0.9,
                competition_level=0.4,
                seasonal_multiplier=1.3
            ),
            "Cotton": MarketConditions(
                price_trend=MarketTrend.VOLATILE,
                demand_level=SeasonalDemand.MEDIUM,
                supply_stability=0.6,
                export_potential=0.8,
                competition_level=0.7,
                seasonal_multiplier=1.0
            ),
            "Jute": MarketConditions(
                price_trend=MarketTrend.BEARISH,
                demand_level=SeasonalDemand.LOW,
                supply_stability=0.7,
                export_potential=0.5,
                competition_level=0.8,
                seasonal_multiplier=0.8
            ),
            "Coffee": MarketConditions(
                price_trend=MarketTrend.BULLISH,
                demand_level=SeasonalDemand.HIGH,
                supply_stability=0.6,
                export_potential=0.9,
                competition_level=0.3,
                seasonal_multiplier=1.4
            )
        }
    
    def _initialize_cost_structures(self) -> Dict[str, InputCosts]:
        """Initialize detailed cost structures for each crop"""
        return {
            "Rice": InputCosts(
                seed_cost_per_hectare=3000, fertilizer_cost_per_hectare=8000,
                pesticide_cost_per_hectare=4000, irrigation_cost_per_hectare=6000,
                labor_cost_per_hectare=15000, machinery_cost_per_hectare=8000,
                transportation_cost_per_hectare=2000, storage_cost_per_hectare=1500,
                insurance_cost_per_hectare=2000, land_preparation_cost=5000
            ),
            "Maize": InputCosts(
                seed_cost_per_hectare=4000, fertilizer_cost_per_hectare=10000,
                pesticide_cost_per_hectare=5000, irrigation_cost_per_hectare=4000,
                labor_cost_per_hectare=12000, machinery_cost_per_hectare=10000,
                transportation_cost_per_hectare=2500, storage_cost_per_hectare=2000,
                insurance_cost_per_hectare=2500, land_preparation_cost=6000
            ),
            "Chickpea": InputCosts(
                seed_cost_per_hectare=5000, fertilizer_cost_per_hectare=6000,
                pesticide_cost_per_hectare=3000, irrigation_cost_per_hectare=3000,
                labor_cost_per_hectare=10000, machinery_cost_per_hectare=6000,
                transportation_cost_per_hectare=2000, storage_cost_per_hectare=3000,
                insurance_cost_per_hectare=2000, land_preparation_cost=4000
            ),
            # Add more crops with detailed cost structures...
            "Coffee": InputCosts(
                seed_cost_per_hectare=15000, fertilizer_cost_per_hectare=20000,
                pesticide_cost_per_hectare=8000, irrigation_cost_per_hectare=12000,
                labor_cost_per_hectare=25000, machinery_cost_per_hectare=15000,
                transportation_cost_per_hectare=5000, storage_cost_per_hectare=8000,
                insurance_cost_per_hectare=5000, land_preparation_cost=10000
            )
        }
    
    def _initialize_seasonal_factors(self) -> Dict[str, Dict[int, float]]:
        """Initialize seasonal price multipliers by month for each crop"""
        return {
            "Rice": {1: 1.1, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.9, 6: 1.0, 
                    7: 1.1, 8: 1.2, 9: 1.3, 10: 1.2, 11: 1.1, 12: 1.0},
            "Coffee": {1: 1.3, 2: 1.2, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.8,
                      7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.4},
            # Add seasonal patterns for all crops...
        }
    
    def calculate_profitability_metrics(
        self, 
        crop_name: str, 
        suitability_score: float,
        environmental_factors: Dict[str, float],
        farm_size: float,
        current_month: int = None
    ) -> ProfitabilityMetrics:
        """
        Calculate comprehensive profitability metrics for a crop
        
        Args:
            crop_name: Name of the crop
            suitability_score: ML model suitability score (0-100)
            environmental_factors: Dict with temp, rainfall, nutrients, etc.
            farm_size: Farm size in hectares
            current_month: Current month for seasonal adjustments
            
        Returns:
            ProfitabilityMetrics object with detailed profit analysis
        """
        if current_month is None:
            current_month = datetime.now().month
            
        market_conditions = self.market_data.get(crop_name)
        cost_structure = self.cost_structures.get(crop_name)
        
        if not market_conditions or not cost_structure:
            # Default values for unknown crops
            market_conditions = MarketConditions(
                MarketTrend.STABLE, SeasonalDemand.MEDIUM, 0.7, 0.5, 0.6, 1.0
            )
            cost_structure = InputCosts(
                4000, 8000, 4000, 5000, 12000, 8000, 2500, 2000, 2500, 5000
            )
        
        # Calculate market-adjusted price
        base_prices = {
            "Rice": 1200, "Maize": 800, "Chickpea": 2500, "Coffee": 8000,
            # Add all crop base prices...
        }
        base_price = base_prices.get(crop_name, 2000)
        
        # Apply market condition adjustments
        price_trend_multiplier = {
            MarketTrend.BULLISH: 1.2,
            MarketTrend.BEARISH: 0.8,
            MarketTrend.STABLE: 1.0,
            MarketTrend.VOLATILE: 0.95
        }[market_conditions.price_trend]
        
        demand_multiplier = {
            SeasonalDemand.PEAK: 1.3,
            SeasonalDemand.HIGH: 1.1,
            SeasonalDemand.MEDIUM: 1.0,
            SeasonalDemand.LOW: 0.8
        }[market_conditions.demand_level]
        
        # Seasonal adjustment
        seasonal_factors = self.seasonal_factors.get(crop_name, {month: 1.0 for month in range(1, 13)})
        seasonal_multiplier = seasonal_factors.get(current_month, 1.0)
        
        # Calculate market-adjusted price
        market_price = (base_price * price_trend_multiplier * 
                       demand_multiplier * seasonal_multiplier * 
                       market_conditions.seasonal_multiplier)
        
        # Estimate production based on suitability (not yield-focused)
        # Focus on quality and market readiness rather than quantity
        production_efficiency = suitability_score / 100.0
        quality_premium = 1.0 + (production_efficiency - 0.5) * 0.3  # Quality affects price
        
        # Calculate revenue per hectare (quality-focused)
        base_production_units = {
            "Rice": 4000, "Maize": 5000, "Coffee": 800,  # Units per hectare
            # Add production estimates for all crops...
        }
        production_per_hectare = base_production_units.get(crop_name, 3000) * production_efficiency
        gross_revenue_per_hectare = production_per_hectare * market_price * quality_premium
        
        # Calculate total costs per hectare
        total_costs_per_hectare = (
            cost_structure.seed_cost_per_hectare +
            cost_structure.fertilizer_cost_per_hectare +
            cost_structure.pesticide_cost_per_hectare +
            cost_structure.irrigation_cost_per_hectare +
            cost_structure.labor_cost_per_hectare +
            cost_structure.machinery_cost_per_hectare +
            cost_structure.transportation_cost_per_hectare +
            cost_structure.storage_cost_per_hectare +
            cost_structure.insurance_cost_per_hectare +
            cost_structure.land_preparation_cost
        )
        
        # Apply efficiency adjustments to costs
        cost_efficiency = 1.0 - (production_efficiency - 0.5) * 0.2  # Better suitability = lower relative costs
        adjusted_costs = total_costs_per_hectare * cost_efficiency
        
        # Calculate profit metrics
        net_profit_per_hectare = gross_revenue_per_hectare - adjusted_costs
        profit_margin = (net_profit_per_hectare / gross_revenue_per_hectare) * 100 if gross_revenue_per_hectare > 0 else 0
        roi = (net_profit_per_hectare / adjusted_costs) * 100 if adjusted_costs > 0 else 0
        break_even_price = adjusted_costs / production_per_hectare if production_per_hectare > 0 else 0
        
        # Risk adjustment based on market conditions
        risk_factor = (
            (1.0 - market_conditions.supply_stability) * 0.3 +
            (1.0 - market_conditions.competition_level) * 0.2 +
            (market_conditions.export_potential) * 0.1
        )
        risk_adjusted_profit = net_profit_per_hectare * (1.0 - risk_factor)
        
        # Calculate profitability score (0-100)
        profitability_score = min(100, max(0, (
            (profit_margin / 50.0) * 30 +  # Profit margin component
            (roi / 100.0) * 25 +           # ROI component
            (suitability_score / 100.0) * 20 +  # Suitability component
            (market_conditions.export_potential) * 15 +  # Export potential
            (market_conditions.supply_stability) * 10    # Market stability
        )))
        
        # Investment recovery time
        monthly_profit = net_profit_per_hectare / 12
        investment_recovery_months = int(adjusted_costs / monthly_profit) if monthly_profit > 0 else 999
        
        return ProfitabilityMetrics(
            gross_revenue_per_hectare=gross_revenue_per_hectare,
            total_costs_per_hectare=adjusted_costs,
            net_profit_per_hectare=net_profit_per_hectare,
            profit_margin_percentage=profit_margin,
            roi_percentage=roi,
            break_even_price=break_even_price,
            risk_adjusted_profit=risk_adjusted_profit,
            profitability_score=profitability_score,
            investment_recovery_months=investment_recovery_months
        )
    
    def rank_crops_by_profitability(
        self, 
        crop_recommendations: List[Dict[str, Any]],
        environmental_factors: Dict[str, float],
        farm_size: float
    ) -> List[Dict[str, Any]]:
        """
        Rank crops by profitability metrics rather than yield
        
        Args:
            crop_recommendations: List of crop recommendations from ML model
            environmental_factors: Environmental conditions
            farm_size: Farm size in hectares
            
        Returns:
            List of crops ranked by profitability with detailed metrics
        """
        enhanced_recommendations = []
        
        for rec in crop_recommendations:
            crop_name = rec["crop_name"]
            suitability_score = rec["suitability_score"]
            
            # Calculate profitability metrics
            profit_metrics = self.calculate_profitability_metrics(
                crop_name, suitability_score, environmental_factors, farm_size
            )
            
            # Get market conditions
            market_conditions = self.market_data.get(crop_name)
            
            # Enhanced recommendation with profit focus
            enhanced_rec = {
                **rec,  # Include original ML model data
                "profitability_score": round(profit_metrics.profitability_score, 2),
                "net_profit_per_hectare": round(profit_metrics.net_profit_per_hectare, 0),
                "total_farm_profit": round(profit_metrics.net_profit_per_hectare * farm_size, 0),
                "profit_margin_percentage": round(profit_metrics.profit_margin_percentage, 2),
                "roi_percentage": round(profit_metrics.roi_percentage, 2),
                "risk_adjusted_profit": round(profit_metrics.risk_adjusted_profit, 0),
                "break_even_price": round(profit_metrics.break_even_price, 2),
                "investment_recovery_months": profit_metrics.investment_recovery_months,
                "market_trend": market_conditions.price_trend.value if market_conditions else "stable",
                "demand_level": market_conditions.demand_level.value if market_conditions else "medium",
                "export_potential": market_conditions.export_potential if market_conditions else 0.5,
                "gross_revenue_per_hectare": round(profit_metrics.gross_revenue_per_hectare, 0),
                "total_costs_per_hectare": round(profit_metrics.total_costs_per_hectare, 0),
                "profit_category": self._categorize_profit(profit_metrics.profitability_score)
            }
            
            enhanced_recommendations.append(enhanced_rec)
        
        # Sort by profitability score (primary) and risk-adjusted profit (secondary)
        enhanced_recommendations.sort(
            key=lambda x: (x["profitability_score"], x["risk_adjusted_profit"]), 
            reverse=True
        )
        
        # Update ranks based on profitability
        for i, rec in enumerate(enhanced_recommendations):
            rec["rank"] = i + 1
            rec["profitability_rank"] = i + 1
        
        return enhanced_recommendations
    
    def _categorize_profit(self, profitability_score: float) -> str:
        """Categorize profit potential based on profitability score"""
        if profitability_score >= 80:
            return "Excellent"
        elif profitability_score >= 65:
            return "High"
        elif profitability_score >= 50:
            return "Good"
        elif profitability_score >= 35:
            return "Moderate"
        else:
            return "Low"
    
    def get_market_insights(self, crop_name: str) -> Dict[str, Any]:
        """Get detailed market insights for a specific crop"""
        market_conditions = self.market_data.get(crop_name)
        if not market_conditions:
            return {"error": "Market data not available for this crop"}
        
        return {
            "price_trend": market_conditions.price_trend.value,
            "demand_level": market_conditions.demand_level.value,
            "supply_stability": market_conditions.supply_stability,
            "export_potential": market_conditions.export_potential,
            "competition_level": market_conditions.competition_level,
            "seasonal_multiplier": market_conditions.seasonal_multiplier,
            "market_recommendation": self._generate_market_recommendation(market_conditions)
        }
    
    def _generate_market_recommendation(self, market_conditions: MarketConditions) -> str:
        """Generate market-based recommendation"""
        if (market_conditions.price_trend == MarketTrend.BULLISH and 
            market_conditions.demand_level in [SeasonalDemand.HIGH, SeasonalDemand.PEAK]):
            return "Excellent market conditions - High profit potential"
        elif market_conditions.price_trend == MarketTrend.BEARISH:
            return "Challenging market conditions - Consider alternatives"
        elif market_conditions.export_potential > 0.8:
            return "Strong export market - Good for large-scale production"
        else:
            return "Stable market conditions - Moderate profit potential"