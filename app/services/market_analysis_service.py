"""
Market Analysis Service
Provides comprehensive market analysis for crop profitability
Includes price trends, demand analysis, and seasonal insights
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class PriceTrend(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    STABLE = "stable"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class MarketVolatility(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class MarketAnalysis:
    """Comprehensive market analysis for a crop"""
    crop_name: str
    current_price: float
    price_trend: PriceTrend
    price_change_percentage: float
    volatility: MarketVolatility
    demand_score: float  # 0-100
    supply_score: float  # 0-100
    export_demand: float  # 0-100
    seasonal_peak_months: List[int]
    seasonal_low_months: List[int]
    market_recommendation: str
    risk_level: str
    opportunity_score: float  # 0-100

@dataclass
class PriceHistory:
    """Historical price data for trend analysis"""
    dates: List[datetime]
    prices: List[float]
    volumes: List[float]

class MarketAnalysisService:
    """Service for comprehensive market analysis and insights"""
    
    def __init__(self):
        self.historical_data = self._initialize_historical_data()
        self.market_indicators = self._initialize_market_indicators()
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        self.export_markets = self._initialize_export_markets()
        
    def _initialize_historical_data(self) -> Dict[str, PriceHistory]:
        """Initialize simulated historical price data for crops"""
        # In a real implementation, this would connect to market data APIs
        base_date = datetime.now() - timedelta(days=365)
        dates = [base_date + timedelta(days=i) for i in range(365)]
        
        historical_data = {}
        
        # Generate realistic price patterns for each crop
        crops_base_prices = {
            "Rice": 1200, "Maize": 800, "Chickpea": 2500, "Kidney Beans": 3000,
            "Pigeon Peas": 2800, "Moth Beans": 2200, "Mung Bean": 3500,
            "Black Gram": 3200, "Lentil": 2800, "Pomegranate": 8000,
            "Banana": 1500, "Mango": 4000, "Grapes": 6000, "Watermelon": 800,
            "Muskmelon": 1200, "Apple": 5000, "Orange": 2500, "Papaya": 2000,
            "Coconut": 1800, "Cotton": 4500, "Jute": 3000, "Coffee": 8000
        }
        
        for crop, base_price in crops_base_prices.items():
            prices = []
            volumes = []
            
            for i, date in enumerate(dates):
                # Add seasonal variation
                seasonal_factor = 1 + 0.3 * math.sin(2 * math.pi * i / 365)
                
                # Add trend (some crops trending up, others down)
                trend_crops = ["Coffee", "Mango", "Pomegranate", "Coconut", "Lentil"]
                trend_factor = 1 + (0.2 * i / 365) if crop in trend_crops else 1 - (0.1 * i / 365)
                
                # Add random variation
                random_factor = 1 + np.random.normal(0, 0.1)
                
                price = base_price * seasonal_factor * trend_factor * random_factor
                volume = np.random.normal(1000, 200)  # Simulated volume
                
                prices.append(max(price, base_price * 0.5))  # Minimum price floor
                volumes.append(max(volume, 100))  # Minimum volume
            
            historical_data[crop] = PriceHistory(dates, prices, volumes)
        
        return historical_data
    
    def _initialize_market_indicators(self) -> Dict[str, Dict[str, float]]:
        """Initialize market indicators for each crop"""
        return {
            "Rice": {
                "demand_growth": 0.05,  # 5% annual growth
                "supply_stability": 0.8,
                "export_potential": 0.6,
                "processing_demand": 0.7,
                "government_support": 0.9
            },
            "Coffee": {
                "demand_growth": 0.12,  # 12% annual growth
                "supply_stability": 0.6,
                "export_potential": 0.95,
                "processing_demand": 0.9,
                "government_support": 0.7
            },
            "Mango": {
                "demand_growth": 0.08,
                "supply_stability": 0.7,
                "export_potential": 0.85,
                "processing_demand": 0.6,
                "government_support": 0.6
            },
            # Add indicators for all crops...
        }
    
    def _initialize_seasonal_patterns(self) -> Dict[str, Dict[str, List[int]]]:
        """Initialize seasonal patterns for each crop"""
        return {
            "Rice": {
                "peak_months": [10, 11, 12],  # Post-harvest demand
                "low_months": [6, 7, 8],     # Pre-harvest surplus
                "planting_months": [6, 7],
                "harvest_months": [10, 11]
            },
            "Coffee": {
                "peak_months": [12, 1, 2],   # Winter demand peak
                "low_months": [6, 7, 8],     # Summer low
                "planting_months": [6, 7],
                "harvest_months": [11, 12, 1]
            },
            "Mango": {
                "peak_months": [4, 5, 6],    # Summer season
                "low_months": [10, 11, 12],  # Off-season
                "planting_months": [7, 8],
                "harvest_months": [3, 4, 5, 6]
            },
            # Add seasonal patterns for all crops...
        }
    
    def _initialize_export_markets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize export market data"""
        return {
            "Coffee": {
                "major_markets": ["USA", "Europe", "Japan"],
                "export_price_premium": 1.3,
                "export_volume_potential": 0.4,
                "quality_requirements": "high"
            },
            "Mango": {
                "major_markets": ["Middle East", "Europe", "USA"],
                "export_price_premium": 1.5,
                "export_volume_potential": 0.3,
                "quality_requirements": "premium"
            },
            # Add export data for all crops...
        }
    
    def analyze_crop_market(self, crop_name: str, current_month: int = None, environmental_factors: Dict = None) -> MarketAnalysis:
        """
        Perform comprehensive market analysis for a specific crop
        
        Args:
            crop_name: Name of the crop to analyze
            current_month: Current month (1-12), defaults to current month
            
        Returns:
            MarketAnalysis object with comprehensive market insights
        """
        if current_month is None:
            current_month = datetime.now().month
        
        # Get historical data
        price_history = self.historical_data.get(crop_name)
        if not price_history:
            return self._create_default_analysis(crop_name)
        
        # Calculate current price and trends
        current_price = price_history.prices[-1]
        price_30_days_ago = price_history.prices[-30] if len(price_history.prices) >= 30 else price_history.prices[0]
        price_change_percentage = ((current_price - price_30_days_ago) / price_30_days_ago) * 100
        
        # Determine price trend
        price_trend = self._determine_price_trend(price_change_percentage)
        
        # Calculate volatility
        recent_prices = price_history.prices[-30:] if len(price_history.prices) >= 30 else price_history.prices
        volatility = self._calculate_volatility(recent_prices)
        
        # Get market indicators
        indicators = self.market_indicators.get(crop_name, {})
        demand_score = self._calculate_demand_score(crop_name, indicators, current_month)
        supply_score = self._calculate_supply_score(crop_name, indicators, current_month)
        export_demand = indicators.get("export_potential", 0.5) * 100
        
        # Get seasonal patterns
        seasonal_data = self.seasonal_patterns.get(crop_name, {})
        seasonal_peak_months = seasonal_data.get("peak_months", [])
        seasonal_low_months = seasonal_data.get("low_months", [])
        
        # Generate market recommendation
        market_recommendation = self._generate_market_recommendation(
            price_trend, demand_score, supply_score, current_month, seasonal_peak_months
        )
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(volatility, supply_score, price_trend)
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(
            demand_score, export_demand, price_trend, current_month, seasonal_peak_months
        )
        
        return MarketAnalysis(
            crop_name=crop_name,
            current_price=round(current_price, 2),
            price_trend=price_trend,
            price_change_percentage=round(price_change_percentage, 2),
            volatility=volatility,
            demand_score=round(demand_score, 1),
            supply_score=round(supply_score, 1),
            export_demand=round(export_demand, 1),
            seasonal_peak_months=seasonal_peak_months,
            seasonal_low_months=seasonal_low_months,
            market_recommendation=market_recommendation,
            risk_level=risk_level,
            opportunity_score=round(opportunity_score, 1)
        )
    
    def _determine_price_trend(self, price_change_percentage: float) -> PriceTrend:
        """Determine price trend based on percentage change"""
        if price_change_percentage > 15:
            return PriceTrend.STRONG_BULLISH
        elif price_change_percentage > 5:
            return PriceTrend.BULLISH
        elif price_change_percentage > -5:
            return PriceTrend.STABLE
        elif price_change_percentage > -15:
            return PriceTrend.BEARISH
        else:
            return PriceTrend.STRONG_BEARISH
    
    def _calculate_volatility(self, prices: List[float]) -> MarketVolatility:
        """Calculate market volatility based on price variations"""
        if len(prices) < 2:
            return MarketVolatility.LOW
        
        # Calculate coefficient of variation
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        cv = (std_price / mean_price) * 100 if mean_price > 0 else 0
        
        if cv > 20:
            return MarketVolatility.EXTREME
        elif cv > 15:
            return MarketVolatility.HIGH
        elif cv > 10:
            return MarketVolatility.MODERATE
        else:
            return MarketVolatility.LOW
    
    def _calculate_demand_score(self, crop_name: str, indicators: Dict[str, float], current_month: int) -> float:
        """Calculate demand score based on various factors"""
        base_demand = 60  # Base demand score
        
        # Add growth factor
        demand_growth = indicators.get("demand_growth", 0.05)
        growth_score = min(demand_growth * 200, 20)  # Max 20 points for growth
        
        # Add processing demand
        processing_demand = indicators.get("processing_demand", 0.5)
        processing_score = processing_demand * 15  # Max 15 points
        
        # Add seasonal adjustment
        seasonal_data = self.seasonal_patterns.get(crop_name, {})
        peak_months = seasonal_data.get("peak_months", [])
        seasonal_adjustment = 10 if current_month in peak_months else 0
        
        total_score = base_demand + growth_score + processing_score + seasonal_adjustment
        return min(total_score, 100)  # Cap at 100
    
    def _calculate_supply_score(self, crop_name: str, indicators: Dict[str, float], current_month: int) -> float:
        """Calculate supply stability score"""
        base_supply = 50  # Base supply score
        
        # Add stability factor
        supply_stability = indicators.get("supply_stability", 0.7)
        stability_score = supply_stability * 30  # Max 30 points
        
        # Add government support
        government_support = indicators.get("government_support", 0.5)
        support_score = government_support * 20  # Max 20 points
        
        total_score = base_supply + stability_score + support_score
        return min(total_score, 100)  # Cap at 100
    
    def _generate_market_recommendation(
        self, 
        price_trend: PriceTrend, 
        demand_score: float, 
        supply_score: float,
        current_month: int,
        peak_months: List[int]
    ) -> str:
        """Generate market recommendation based on analysis"""
        if price_trend in [PriceTrend.STRONG_BULLISH, PriceTrend.BULLISH] and demand_score > 70:
            return "Strong Buy - Excellent market conditions with rising prices and high demand"
        elif current_month in peak_months and demand_score > 60:
            return "Buy - Seasonal peak period with good demand"
        elif price_trend == PriceTrend.STABLE and supply_score > 70:
            return "Hold - Stable market with reliable supply"
        elif price_trend in [PriceTrend.BEARISH, PriceTrend.STRONG_BEARISH]:
            return "Caution - Declining prices, consider alternatives"
        else:
            return "Neutral - Mixed market signals, monitor closely"
    
    def _calculate_risk_level(self, volatility: MarketVolatility, supply_score: float, price_trend: PriceTrend) -> str:
        """Calculate overall risk level"""
        risk_points = 0
        
        # Volatility risk
        volatility_risk = {
            MarketVolatility.LOW: 0,
            MarketVolatility.MODERATE: 1,
            MarketVolatility.HIGH: 2,
            MarketVolatility.EXTREME: 3
        }[volatility]
        risk_points += volatility_risk
        
        # Supply risk
        if supply_score < 50:
            risk_points += 2
        elif supply_score < 70:
            risk_points += 1
        
        # Price trend risk
        if price_trend in [PriceTrend.BEARISH, PriceTrend.STRONG_BEARISH]:
            risk_points += 2
        
        if risk_points >= 5:
            return "High"
        elif risk_points >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_opportunity_score(
        self, 
        demand_score: float, 
        export_demand: float, 
        price_trend: PriceTrend,
        current_month: int,
        peak_months: List[int]
    ) -> float:
        """Calculate opportunity score for investment"""
        base_score = 40
        
        # Demand opportunity
        demand_points = (demand_score / 100) * 25
        
        # Export opportunity
        export_points = (export_demand / 100) * 20
        
        # Price trend opportunity
        trend_points = {
            PriceTrend.STRONG_BULLISH: 15,
            PriceTrend.BULLISH: 10,
            PriceTrend.STABLE: 5,
            PriceTrend.BEARISH: 0,
            PriceTrend.STRONG_BEARISH: -5
        }[price_trend]
        
        # Seasonal opportunity
        seasonal_points = 10 if current_month in peak_months else 0
        
        total_score = base_score + demand_points + export_points + trend_points + seasonal_points
        return max(0, min(total_score, 100))  # Ensure 0-100 range
    
    def _create_default_analysis(self, crop_name: str) -> MarketAnalysis:
        """Create default analysis for unknown crops"""
        return MarketAnalysis(
            crop_name=crop_name,
            current_price=2000.0,
            price_trend=PriceTrend.STABLE,
            price_change_percentage=0.0,
            volatility=MarketVolatility.MODERATE,
            demand_score=60.0,
            supply_score=60.0,
            export_demand=50.0,
            seasonal_peak_months=[],
            seasonal_low_months=[],
            market_recommendation="Neutral - Limited market data available",
            risk_level="Medium",
            opportunity_score=50.0
        )
    
    def get_market_summary(self, crops: List[str]) -> Dict[str, Any]:
        """Get market summary for multiple crops"""
        current_month = datetime.now().month
        analyses = []
        
        for crop in crops:
            analysis = self.analyze_crop_market(crop, current_month)
            analyses.append({
                "crop_name": analysis.crop_name,
                "opportunity_score": analysis.opportunity_score,
                "risk_level": analysis.risk_level,
                "price_trend": analysis.price_trend.value,
                "market_recommendation": analysis.market_recommendation
            })
        
        # Sort by opportunity score
        analyses.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        return {
            "market_summary": analyses,
            "analysis_date": datetime.now().isoformat(),
            "market_outlook": self._generate_overall_market_outlook(analyses)
        }
    
    def _generate_overall_market_outlook(self, analyses: List[Dict[str, Any]]) -> str:
        """Generate overall market outlook"""
        high_opportunity_crops = [a for a in analyses if a["opportunity_score"] > 70]
        bullish_trends = [a for a in analyses if "bullish" in a["price_trend"]]
        
        if len(high_opportunity_crops) > len(analyses) * 0.6:
            return "Positive - Multiple crops showing strong market opportunities"
        elif len(bullish_trends) > len(analyses) * 0.5:
            return "Optimistic - Rising price trends across several crops"
        else:
            return "Mixed - Varied market conditions across different crops"