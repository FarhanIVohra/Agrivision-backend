"""
Cost Analysis Service
Provides comprehensive input cost analysis for crop production
Includes regional variations, seasonal adjustments, and cost optimization
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class CostCategory(Enum):
    SEEDS = "seeds"
    FERTILIZERS = "fertilizers"
    PESTICIDES = "pesticides"
    IRRIGATION = "irrigation"
    LABOR = "labor"
    MACHINERY = "machinery"
    TRANSPORTATION = "transportation"
    STORAGE = "storage"
    INSURANCE = "insurance"
    LAND_PREPARATION = "land_preparation"
    UTILITIES = "utilities"
    CERTIFICATION = "certification"

class RegionType(Enum):
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    REMOTE = "remote"

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a crop"""
    crop_name: str
    total_cost_per_hectare: float
    cost_categories: Dict[str, float]
    regional_multiplier: float
    seasonal_adjustments: Dict[str, float]
    optimization_potential: float  # Percentage cost reduction possible
    cost_efficiency_score: float  # 0-100
    break_even_yield: float
    cost_per_unit_production: float

@dataclass
class CostOptimization:
    """Cost optimization recommendations"""
    category: str
    current_cost: float
    optimized_cost: float
    savings_amount: float
    savings_percentage: float
    optimization_method: str
    implementation_difficulty: str  # Easy, Medium, Hard
    payback_period_months: int

class CostAnalysisService:
    """Service for comprehensive cost analysis and optimization"""
    
    def __init__(self):
        self.base_costs = self._initialize_base_costs()
        self.regional_multipliers = self._initialize_regional_multipliers()
        self.seasonal_factors = self._initialize_seasonal_factors()
        self.optimization_strategies = self._initialize_optimization_strategies()
        self.input_suppliers = self._initialize_supplier_data()
        
    def _initialize_base_costs(self) -> Dict[str, Dict[str, float]]:
        """Initialize base costs per hectare for each crop and category"""
        return {
            "Rice": {
                "seeds": 3000, "fertilizers": 8000, "pesticides": 4000,
                "irrigation": 6000, "labor": 15000, "machinery": 8000,
                "transportation": 2000, "storage": 1500, "insurance": 2000,
                "land_preparation": 5000, "utilities": 1000, "certification": 500
            },
            "Maize": {
                "seeds": 4000, "fertilizers": 10000, "pesticides": 5000,
                "irrigation": 4000, "labor": 12000, "machinery": 10000,
                "transportation": 2500, "storage": 2000, "insurance": 2500,
                "land_preparation": 6000, "utilities": 800, "certification": 300
            },
            "Chickpea": {
                "seeds": 5000, "fertilizers": 6000, "pesticides": 3000,
                "irrigation": 3000, "labor": 10000, "machinery": 6000,
                "transportation": 2000, "storage": 3000, "insurance": 2000,
                "land_preparation": 4000, "utilities": 600, "certification": 400
            },
            "Coffee": {
                "seeds": 15000, "fertilizers": 20000, "pesticides": 8000,
                "irrigation": 12000, "labor": 25000, "machinery": 15000,
                "transportation": 5000, "storage": 8000, "insurance": 5000,
                "land_preparation": 10000, "utilities": 2000, "certification": 3000
            },
            "Mango": {
                "seeds": 12000, "fertilizers": 15000, "pesticides": 6000,
                "irrigation": 10000, "labor": 20000, "machinery": 12000,
                "transportation": 4000, "storage": 6000, "insurance": 4000,
                "land_preparation": 8000, "utilities": 1500, "certification": 2000
            },
            "Coconut": {
                "seeds": 8000, "fertilizers": 12000, "pesticides": 4000,
                "irrigation": 8000, "labor": 18000, "machinery": 10000,
                "transportation": 3000, "storage": 4000, "insurance": 3000,
                "land_preparation": 6000, "utilities": 1200, "certification": 1500
            },
            # Add more crops with detailed cost structures...
        }
    
    def _initialize_regional_multipliers(self) -> Dict[str, Dict[str, float]]:
        """Initialize regional cost multipliers"""
        return {
            "urban": {
                "seeds": 1.2, "fertilizers": 1.1, "pesticides": 1.1,
                "irrigation": 1.3, "labor": 1.5, "machinery": 1.2,
                "transportation": 0.9, "storage": 1.4, "insurance": 1.1,
                "land_preparation": 1.3, "utilities": 1.2, "certification": 1.0
            },
            "suburban": {
                "seeds": 1.1, "fertilizers": 1.05, "pesticides": 1.05,
                "irrigation": 1.1, "labor": 1.2, "machinery": 1.1,
                "transportation": 1.0, "storage": 1.2, "insurance": 1.05,
                "land_preparation": 1.1, "utilities": 1.1, "certification": 1.0
            },
            "rural": {
                "seeds": 1.0, "fertilizers": 1.0, "pesticides": 1.0,
                "irrigation": 1.0, "labor": 1.0, "machinery": 1.0,
                "transportation": 1.1, "storage": 1.0, "insurance": 1.0,
                "land_preparation": 1.0, "utilities": 1.0, "certification": 1.0
            },
            "remote": {
                "seeds": 1.3, "fertilizers": 1.4, "pesticides": 1.3,
                "irrigation": 1.2, "labor": 0.8, "machinery": 1.5,
                "transportation": 1.5, "storage": 1.3, "insurance": 1.2,
                "land_preparation": 0.9, "utilities": 1.4, "certification": 1.2
            }
        }
    
    def _initialize_seasonal_factors(self) -> Dict[int, Dict[str, float]]:
        """Initialize seasonal cost adjustment factors by month"""
        return {
            1: {"seeds": 1.0, "fertilizers": 1.1, "labor": 0.9, "machinery": 1.0},
            2: {"seeds": 1.0, "fertilizers": 1.1, "labor": 0.9, "machinery": 1.0},
            3: {"seeds": 1.2, "fertilizers": 1.2, "labor": 1.1, "machinery": 1.1},
            4: {"seeds": 1.3, "fertilizers": 1.3, "labor": 1.2, "machinery": 1.2},
            5: {"seeds": 1.2, "fertilizers": 1.2, "labor": 1.1, "machinery": 1.1},
            6: {"seeds": 1.4, "fertilizers": 1.4, "labor": 1.3, "machinery": 1.3},
            7: {"seeds": 1.3, "fertilizers": 1.3, "labor": 1.2, "machinery": 1.2},
            8: {"seeds": 1.1, "fertilizers": 1.1, "labor": 1.0, "machinery": 1.0},
            9: {"seeds": 1.0, "fertilizers": 1.0, "labor": 1.0, "machinery": 1.0},
            10: {"seeds": 1.1, "fertilizers": 1.2, "labor": 1.1, "machinery": 1.1},
            11: {"seeds": 1.0, "fertilizers": 1.1, "labor": 1.0, "machinery": 1.0},
            12: {"seeds": 1.0, "fertilizers": 1.0, "labor": 0.9, "machinery": 0.9}
        }
    
    def _initialize_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cost optimization strategies"""
        return {
            "seeds": {
                "bulk_purchase": {"savings": 0.15, "difficulty": "Easy", "payback": 1},
                "certified_varieties": {"savings": 0.10, "difficulty": "Medium", "payback": 2},
                "seed_treatment": {"savings": 0.08, "difficulty": "Easy", "payback": 1}
            },
            "fertilizers": {
                "soil_testing": {"savings": 0.20, "difficulty": "Medium", "payback": 2},
                "organic_alternatives": {"savings": 0.25, "difficulty": "Hard", "payback": 6},
                "precision_application": {"savings": 0.18, "difficulty": "Medium", "payback": 3}
            },
            "pesticides": {
                "integrated_pest_management": {"savings": 0.30, "difficulty": "Hard", "payback": 4},
                "biological_control": {"savings": 0.25, "difficulty": "Medium", "payback": 3},
                "targeted_spraying": {"savings": 0.15, "difficulty": "Medium", "payback": 2}
            },
            "irrigation": {
                "drip_irrigation": {"savings": 0.35, "difficulty": "Hard", "payback": 12},
                "water_scheduling": {"savings": 0.20, "difficulty": "Medium", "payback": 3},
                "mulching": {"savings": 0.15, "difficulty": "Easy", "payback": 1}
            },
            "labor": {
                "mechanization": {"savings": 0.40, "difficulty": "Hard", "payback": 18},
                "training_programs": {"savings": 0.15, "difficulty": "Medium", "payback": 6},
                "seasonal_planning": {"savings": 0.10, "difficulty": "Easy", "payback": 2}
            },
            "machinery": {
                "cooperative_sharing": {"savings": 0.30, "difficulty": "Medium", "payback": 6},
                "maintenance_scheduling": {"savings": 0.15, "difficulty": "Easy", "payback": 3},
                "fuel_efficiency": {"savings": 0.20, "difficulty": "Medium", "payback": 4}
            }
        }
    
    def _initialize_supplier_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize supplier data for cost comparison"""
        return {
            "fertilizers": [
                {"name": "AgriCorp", "price_factor": 1.0, "quality": "high", "reliability": 0.9},
                {"name": "FarmSupply", "price_factor": 0.85, "quality": "medium", "reliability": 0.8},
                {"name": "BulkFert", "price_factor": 0.75, "quality": "medium", "reliability": 0.7}
            ],
            "seeds": [
                {"name": "SeedTech", "price_factor": 1.1, "quality": "premium", "reliability": 0.95},
                {"name": "LocalSeeds", "price_factor": 0.9, "quality": "good", "reliability": 0.85},
                {"name": "BudgetSeeds", "price_factor": 0.7, "quality": "basic", "reliability": 0.7}
            ]
        }
    
    def analyze_crop_costs(
        self, 
        crop_name: str, 
        farm_size: float,
        region_type: str = "rural",
        current_month: int = None,
        environmental_factors: Dict[str, float] = None
    ) -> CostBreakdown:
        """
        Perform comprehensive cost analysis for a specific crop
        
        Args:
            crop_name: Name of the crop
            farm_size: Farm size in hectares
            region_type: Type of region (urban, suburban, rural, remote)
            current_month: Current month for seasonal adjustments
            environmental_factors: Environmental conditions affecting costs
            
        Returns:
            CostBreakdown object with detailed cost analysis
        """
        if current_month is None:
            current_month = datetime.now().month
        
        if environmental_factors is None:
            environmental_factors = {}
        
        # Get base costs
        base_costs = self.base_costs.get(crop_name)
        if not base_costs:
            base_costs = self._get_default_costs()
        
        # Apply regional multipliers
        regional_multipliers = self.regional_multipliers.get(region_type, self.regional_multipliers["rural"])
        
        # Apply seasonal factors
        seasonal_factors = self.seasonal_factors.get(current_month, {})
        
        # Calculate adjusted costs
        adjusted_costs = {}
        total_cost = 0
        
        for category, base_cost in base_costs.items():
            # Apply regional adjustment
            regional_factor = regional_multipliers.get(category, 1.0)
            
            # Apply seasonal adjustment
            seasonal_factor = seasonal_factors.get(category, 1.0)
            
            # Apply environmental adjustments
            environmental_factor = self._calculate_environmental_factor(category, environmental_factors)
            
            # Calculate final cost
            adjusted_cost = base_cost * regional_factor * seasonal_factor * environmental_factor
            adjusted_costs[category] = round(adjusted_cost, 2)
            total_cost += adjusted_cost
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(adjusted_costs)
        
        # Calculate cost efficiency score
        cost_efficiency_score = self._calculate_cost_efficiency_score(
            crop_name, adjusted_costs, environmental_factors
        )
        
        # Estimate break-even yield and cost per unit
        break_even_yield, cost_per_unit = self._calculate_production_metrics(
            crop_name, total_cost, environmental_factors
        )
        
        return CostBreakdown(
            crop_name=crop_name,
            total_cost_per_hectare=round(total_cost, 2),
            cost_categories=adjusted_costs,
            regional_multiplier=self._calculate_average_regional_multiplier(regional_multipliers),
            seasonal_adjustments=seasonal_factors,
            optimization_potential=round(optimization_potential, 1),
            cost_efficiency_score=round(cost_efficiency_score, 1),
            break_even_yield=round(break_even_yield, 2),
            cost_per_unit_production=round(cost_per_unit, 2)
        )
    
    def _get_default_costs(self) -> Dict[str, float]:
        """Get default cost structure for unknown crops"""
        return {
            "seeds": 4000, "fertilizers": 8000, "pesticides": 4000,
            "irrigation": 5000, "labor": 12000, "machinery": 8000,
            "transportation": 2500, "storage": 2000, "insurance": 2500,
            "land_preparation": 5000, "utilities": 1000, "certification": 500
        }
    
    def _calculate_environmental_factor(self, category: str, environmental_factors: Dict[str, float]) -> float:
        """Calculate environmental adjustment factor for cost category"""
        base_factor = 1.0
        
        # Get environmental values
        temperature = environmental_factors.get("temperature", 25)
        rainfall = environmental_factors.get("rainfall", 100)
        humidity = environmental_factors.get("humidity", 60)
        ph = environmental_factors.get("ph", 6.5)
        
        if category == "irrigation":
            # Lower rainfall increases irrigation costs
            if rainfall < 50:
                base_factor *= 1.5
            elif rainfall < 100:
                base_factor *= 1.2
            elif rainfall > 200:
                base_factor *= 0.8
        
        elif category == "pesticides":
            # High humidity increases pest pressure
            if humidity > 80:
                base_factor *= 1.3
            elif humidity > 70:
                base_factor *= 1.1
        
        elif category == "fertilizers":
            # pH affects fertilizer efficiency
            if ph < 5.5 or ph > 8.0:
                base_factor *= 1.2  # Need more fertilizer for poor pH
        
        elif category == "labor":
            # Extreme temperatures affect labor costs
            if temperature > 40 or temperature < 10:
                base_factor *= 1.2
        
        return base_factor
    
    def _calculate_optimization_potential(self, costs: Dict[str, float]) -> float:
        """Calculate overall cost optimization potential"""
        total_savings = 0
        total_cost = sum(costs.values())
        
        for category, cost in costs.items():
            if category in self.optimization_strategies:
                strategies = self.optimization_strategies[category]
                # Take the best optimization strategy for each category
                best_savings = max([s["savings"] for s in strategies.values()])
                total_savings += cost * best_savings
        
        return (total_savings / total_cost) * 100 if total_cost > 0 else 0
    
    def _calculate_cost_efficiency_score(
        self, 
        crop_name: str, 
        costs: Dict[str, float], 
        environmental_factors: Dict[str, float]
    ) -> float:
        """Calculate cost efficiency score (0-100)"""
        base_score = 60
        
        # Efficiency based on cost distribution
        total_cost = sum(costs.values())
        
        # Penalize if any single category is too dominant
        max_category_percentage = max([cost/total_cost for cost in costs.values()]) * 100
        if max_category_percentage > 40:
            base_score -= 10
        
        # Reward balanced cost distribution
        cost_variance = np.var(list(costs.values()))
        if cost_variance < np.mean(list(costs.values())) * 0.5:
            base_score += 10
        
        # Environmental suitability bonus
        suitability_bonus = self._calculate_environmental_suitability_bonus(environmental_factors)
        base_score += suitability_bonus
        
        # Technology adoption bonus (lower machinery costs relative to labor)
        machinery_cost = costs.get("machinery", 0)
        labor_cost = costs.get("labor", 0)
        if machinery_cost > 0 and labor_cost > 0:
            automation_ratio = machinery_cost / (machinery_cost + labor_cost)
            if 0.3 < automation_ratio < 0.7:  # Optimal balance
                base_score += 5
        
        return max(0, min(base_score, 100))
    
    def _calculate_environmental_suitability_bonus(self, environmental_factors: Dict[str, float]) -> float:
        """Calculate bonus points for environmental suitability"""
        bonus = 0
        
        temperature = environmental_factors.get("temperature", 25)
        rainfall = environmental_factors.get("rainfall", 100)
        ph = environmental_factors.get("ph", 6.5)
        
        # Optimal ranges give bonus points
        if 20 <= temperature <= 30:
            bonus += 5
        if 75 <= rainfall <= 150:
            bonus += 5
        if 6.0 <= ph <= 7.5:
            bonus += 5
        
        return bonus
    
    def _calculate_production_metrics(
        self, 
        crop_name: str, 
        total_cost: float, 
        environmental_factors: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate break-even yield and cost per unit production"""
        # Estimated market prices (per kg)
        market_prices = {
            "Rice": 25, "Maize": 20, "Chickpea": 60, "Coffee": 200,
            "Mango": 40, "Coconut": 30, "Cotton": 45
        }
        
        # Estimated yields (kg per hectare)
        base_yields = {
            "Rice": 4000, "Maize": 5000, "Chickpea": 1500, "Coffee": 800,
            "Mango": 8000, "Coconut": 2000, "Cotton": 1200
        }
        
        market_price = market_prices.get(crop_name, 30)
        base_yield = base_yields.get(crop_name, 3000)
        
        # Adjust yield based on environmental factors
        environmental_yield_factor = self._calculate_yield_factor(environmental_factors)
        estimated_yield = base_yield * environmental_yield_factor
        
        # Calculate metrics
        break_even_yield = total_cost / market_price if market_price > 0 else 0
        cost_per_unit = total_cost / estimated_yield if estimated_yield > 0 else 0
        
        return break_even_yield, cost_per_unit
    
    def _calculate_yield_factor(self, environmental_factors: Dict[str, float]) -> float:
        """Calculate yield adjustment factor based on environmental conditions"""
        base_factor = 1.0
        
        temperature = environmental_factors.get("temperature", 25)
        rainfall = environmental_factors.get("rainfall", 100)
        humidity = environmental_factors.get("humidity", 60)
        
        # Temperature factor
        if 20 <= temperature <= 30:
            temp_factor = 1.0
        elif 15 <= temperature <= 35:
            temp_factor = 0.9
        else:
            temp_factor = 0.7
        
        # Rainfall factor
        if 75 <= rainfall <= 150:
            rain_factor = 1.0
        elif 50 <= rainfall <= 200:
            rain_factor = 0.9
        else:
            rain_factor = 0.8
        
        return base_factor * temp_factor * rain_factor
    
    def _calculate_average_regional_multiplier(self, multipliers: Dict[str, float]) -> float:
        """Calculate average regional multiplier"""
        return round(np.mean(list(multipliers.values())), 2)
    
    def get_cost_optimization_recommendations(
        self, 
        cost_breakdown: CostBreakdown
    ) -> List[CostOptimization]:
        """Get specific cost optimization recommendations"""
        recommendations = []
        
        for category, current_cost in cost_breakdown.cost_categories.items():
            if category in self.optimization_strategies:
                strategies = self.optimization_strategies[category]
                
                for strategy_name, strategy_data in strategies.items():
                    savings_percentage = strategy_data["savings"]
                    savings_amount = current_cost * savings_percentage
                    optimized_cost = current_cost - savings_amount
                    
                    recommendation = CostOptimization(
                        category=category,
                        current_cost=current_cost,
                        optimized_cost=round(optimized_cost, 2),
                        savings_amount=round(savings_amount, 2),
                        savings_percentage=round(savings_percentage * 100, 1),
                        optimization_method=strategy_name.replace("_", " ").title(),
                        implementation_difficulty=strategy_data["difficulty"],
                        payback_period_months=strategy_data["payback"]
                    )
                    
                    recommendations.append(recommendation)
        
        # Sort by savings amount (highest first)
        recommendations.sort(key=lambda x: x.savings_amount, reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def compare_regional_costs(
        self, 
        crop_name: str, 
        farm_size: float
    ) -> Dict[str, Any]:
        """Compare costs across different regions"""
        regions = ["urban", "suburban", "rural", "remote"]
        comparisons = {}
        
        for region in regions:
            cost_breakdown = self.analyze_crop_costs(crop_name, farm_size, region)
            comparisons[region] = {
                "total_cost": cost_breakdown.total_cost_per_hectare,
                "cost_efficiency_score": cost_breakdown.cost_efficiency_score,
                "optimization_potential": cost_breakdown.optimization_potential
            }
        
        # Find best and worst regions
        best_region = min(comparisons.keys(), key=lambda x: comparisons[x]["total_cost"])
        worst_region = max(comparisons.keys(), key=lambda x: comparisons[x]["total_cost"])
        
        return {
            "regional_comparison": comparisons,
            "best_region": best_region,
            "worst_region": worst_region,
            "cost_difference": round(
                comparisons[worst_region]["total_cost"] - comparisons[best_region]["total_cost"], 2
            ),
            "recommendation": f"Consider {best_region} region for optimal cost efficiency"
        }