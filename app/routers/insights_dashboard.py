"""
Actionable Insights Dashboard Router
API endpoints for providing farmers with clear, actionable recommendations
Based on disease predictions, environmental data, and crop management insights
Aligns with AI-Driven Crop Disease Prediction and Management System requirements
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
from enum import Enum

from ..services.disease_management_service import DiseaseManagementService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global service instance
disease_management_service = None

def get_disease_management_service() -> DiseaseManagementService:
    """Dependency to get disease management service instance"""
    global disease_management_service
    if disease_management_service is None:
        disease_management_service = DiseaseManagementService()
    return disease_management_service

# Enums for better type safety
class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionCategory(str, Enum):
    IMMEDIATE = "immediate"
    PREVENTIVE = "preventive"
    MONITORING = "monitoring"
    PLANNING = "planning"

# Pydantic models
class FarmProfile(BaseModel):
    """Farm profile information for personalized insights"""
    farm_id: str = Field(..., description="Unique farm identifier")
    crop_types: List[str] = Field(..., description="Types of crops grown")
    farm_size: float = Field(..., description="Farm size in acres", gt=0)
    location: str = Field(..., description="Farm location/region")
    farming_experience: int = Field(default=5, description="Years of farming experience", ge=0)
    technology_adoption: str = Field(default="medium", description="Technology adoption level: low, medium, high")
    budget_category: str = Field(default="medium", description="Budget category: low, medium, high")

class EnvironmentalContext(BaseModel):
    """Current environmental context for insights"""
    temperature: float = Field(..., description="Current temperature in Celsius")
    humidity: float = Field(..., description="Current humidity percentage", ge=0, le=100)
    rainfall: float = Field(default=0, description="Recent rainfall in mm", ge=0)
    wind_speed: float = Field(default=5.0, description="Wind speed in km/h", ge=0)
    soil_moisture: Optional[float] = Field(default=None, description="Soil moisture percentage")
    growth_stage: str = Field(default="vegetative", description="Current crop growth stage")

class ActionableInsight(BaseModel):
    """Individual actionable insight"""
    insight_id: str
    title: str
    description: str
    priority: PriorityLevel
    category: ActionCategory
    estimated_cost: Optional[str] = None
    time_to_implement: str
    expected_benefit: str
    confidence_score: float
    supporting_data: Dict[str, Any]

@router.post(
    "/personalized-insights",
    response_model=Dict[str, Any],
    summary="Get personalized actionable insights",
    description="Generate personalized actionable insights based on farm profile and current conditions"
)
async def get_personalized_insights(
    farm_profile: FarmProfile,
    environmental_context: EnvironmentalContext,
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Generate personalized actionable insights for farmers
    
    Args:
        farm_profile: Farm profile information
        environmental_context: Current environmental conditions
        service: Disease management service instance
        
    Returns:
        Dictionary with personalized insights and recommendations
    """
    try:
        logger.info(f"Generating personalized insights for farm: {farm_profile.farm_id}")
        
        insights = []
        
        # Generate disease risk insights
        for crop_type in farm_profile.crop_types:
            env_data = environmental_context.dict()
            env_data["crop_type"] = crop_type
            
            # Get disease risk assessment
            risk_assessment = service.predict_disease_outbreak_risk(env_data)
            
            # Generate insights based on risk level
            if risk_assessment.get("overall_risk", 0) >= 0.7:
                insights.append(_create_high_risk_insight(crop_type, risk_assessment, farm_profile))
            elif risk_assessment.get("overall_risk", 0) >= 0.4:
                insights.append(_create_moderate_risk_insight(crop_type, risk_assessment, farm_profile))
            else:
                insights.append(_create_low_risk_insight(crop_type, risk_assessment, farm_profile))
        
        # Generate environmental insights
        env_insights = _generate_environmental_insights(environmental_context, farm_profile)
        insights.extend(env_insights)
        
        # Generate management insights
        mgmt_insights = _generate_management_insights(farm_profile, environmental_context)
        insights.extend(mgmt_insights)
        
        # Sort insights by priority and confidence
        insights.sort(key=lambda x: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[x["priority"]],
            x["confidence_score"]
        ), reverse=True)
        
        # Generate summary statistics
        summary = _generate_insights_summary(insights, farm_profile)
        
        return {
            "farm_id": farm_profile.farm_id,
            "generated_at": datetime.now().isoformat(),
            "total_insights": len(insights),
            "insights": insights[:10],  # Top 10 insights
            "summary": summary,
            "next_review_date": (datetime.now() + timedelta(days=3)).isoformat(),
            "emergency_contacts": _get_emergency_contacts(farm_profile.location)
        }
        
    except Exception as e:
        logger.error(f"Error generating personalized insights: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate personalized insights: {str(e)}"
        )

@router.get(
    "/daily-recommendations",
    response_model=Dict[str, Any],
    summary="Get daily recommendations",
    description="Get daily actionable recommendations based on current conditions"
)
async def get_daily_recommendations(
    crop_type: str = Query(..., description="Primary crop type"),
    location: str = Query(..., description="Farm location"),
    farm_size: float = Query(default=10.0, description="Farm size in acres"),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Get daily actionable recommendations for farmers
    
    Args:
        crop_type: Primary crop type
        location: Farm location
        farm_size: Farm size in acres
        service: Disease management service instance
        
    Returns:
        Dictionary with daily recommendations
    """
    try:
        # Generate mock environmental data (replace with actual weather API)
        current_conditions = {
            "temperature": 25.0,
            "humidity": 75.0,
            "rainfall": 5.0,
            "crop_type": crop_type
        }
        
        # Get risk assessment
        risk_assessment = service.predict_disease_outbreak_risk(current_conditions)
        
        # Generate daily recommendations
        recommendations = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "crop_type": crop_type,
            "location": location,
            "weather_summary": {
                "temperature": current_conditions["temperature"],
                "humidity": current_conditions["humidity"],
                "rainfall": current_conditions["rainfall"],
                "conditions": "Moderate risk conditions"
            },
            "priority_actions": _get_priority_actions(risk_assessment, crop_type),
            "monitoring_checklist": _get_monitoring_checklist(crop_type, risk_assessment),
            "treatment_schedule": _get_treatment_schedule(risk_assessment),
            "resource_requirements": _estimate_resource_requirements(farm_size, risk_assessment),
            "cost_estimates": _get_cost_estimates(farm_size, risk_assessment),
            "success_metrics": _define_success_metrics(crop_type, risk_assessment)
        }
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting daily recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get daily recommendations: {str(e)}"
        )

@router.get(
    "/action-plan",
    response_model=Dict[str, Any],
    summary="Generate comprehensive action plan",
    description="Generate a comprehensive action plan based on current farm conditions"
)
async def generate_action_plan(
    crop_types: str = Query(..., description="Comma-separated list of crop types"),
    risk_tolerance: str = Query(default="medium", description="Risk tolerance: low, medium, high"),
    budget_range: str = Query(default="medium", description="Budget range: low, medium, high"),
    time_horizon: int = Query(default=30, description="Planning time horizon in days"),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Generate comprehensive action plan for farm management
    
    Args:
        crop_types: Comma-separated list of crop types
        risk_tolerance: Risk tolerance level
        budget_range: Budget range
        time_horizon: Planning time horizon in days
        service: Disease management service instance
        
    Returns:
        Dictionary with comprehensive action plan
    """
    try:
        crops = [crop.strip() for crop in crop_types.split(",")]
        
        action_plan = {
            "plan_id": f"PLAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "time_horizon": time_horizon,
            "crops_covered": crops,
            "risk_tolerance": risk_tolerance,
            "budget_range": budget_range,
            "phases": []
        }
        
        # Generate phases based on time horizon
        phase_duration = max(7, time_horizon // 4)  # At least 7 days per phase
        
        for phase_num in range(1, 5):  # 4 phases
            phase_start = datetime.now() + timedelta(days=(phase_num - 1) * phase_duration)
            phase_end = phase_start + timedelta(days=phase_duration)
            
            phase = {
                "phase_number": phase_num,
                "phase_name": _get_phase_name(phase_num),
                "start_date": phase_start.strftime("%Y-%m-%d"),
                "end_date": phase_end.strftime("%Y-%m-%d"),
                "objectives": _get_phase_objectives(phase_num, crops),
                "key_activities": _get_phase_activities(phase_num, crops, risk_tolerance),
                "resource_requirements": _get_phase_resources(phase_num, budget_range),
                "success_indicators": _get_phase_indicators(phase_num, crops),
                "contingency_plans": _get_contingency_plans(phase_num, risk_tolerance)
            }
            
            action_plan["phases"].append(phase)
        
        # Add overall plan summary
        action_plan["summary"] = {
            "total_activities": sum(len(phase["key_activities"]) for phase in action_plan["phases"]),
            "estimated_cost": _estimate_total_cost(budget_range, len(crops)),
            "expected_outcomes": _define_expected_outcomes(crops, risk_tolerance),
            "risk_mitigation": _get_risk_mitigation_strategies(risk_tolerance)
        }
        
        return action_plan
        
    except Exception as e:
        logger.error(f"Error generating action plan: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate action plan: {str(e)}"
        )

@router.get(
    "/performance-metrics",
    response_model=Dict[str, Any],
    summary="Get performance metrics and KPIs",
    description="Get key performance indicators and metrics for farm management"
)
async def get_performance_metrics(
    farm_id: str = Query(..., description="Farm identifier"),
    period: str = Query(default="monthly", description="Reporting period: weekly, monthly, seasonal"),
    service: DiseaseManagementService = Depends(get_disease_management_service)
) -> Dict[str, Any]:
    """
    Get performance metrics and KPIs for farm management
    
    Args:
        farm_id: Farm identifier
        period: Reporting period
        service: Disease management service instance
        
    Returns:
        Dictionary with performance metrics and KPIs
    """
    try:
        # Generate mock performance data (replace with actual data from database)
        metrics = {
            "farm_id": farm_id,
            "reporting_period": period,
            "generated_at": datetime.now().isoformat(),
            "disease_management": {
                "early_detection_rate": 85.0,
                "treatment_success_rate": 92.0,
                "prevention_effectiveness": 78.0,
                "outbreak_prevention": 95.0
            },
            "operational_efficiency": {
                "monitoring_compliance": 88.0,
                "treatment_timeliness": 91.0,
                "resource_utilization": 82.0,
                "cost_effectiveness": 87.0
            },
            "crop_health": {
                "overall_health_score": 89.0,
                "disease_incidence": 12.0,
                "yield_protection": 94.0,
                "quality_maintenance": 91.0
            },
            "financial_impact": {
                "cost_savings": "$2,450",
                "yield_protection_value": "$8,900",
                "roi_on_treatments": "340%",
                "total_benefit": "$11,350"
            },
            "recommendations": [
                "Continue current monitoring schedule",
                "Consider increasing preventive treatments during high-risk periods",
                "Implement automated monitoring for better efficiency",
                "Focus on early detection training for farm workers"
            ],
            "trends": {
                "disease_pressure": "Decreasing",
                "treatment_costs": "Stable",
                "effectiveness": "Improving",
                "farmer_satisfaction": "High"
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

# Helper functions
def _create_high_risk_insight(crop_type: str, risk_assessment: Dict, farm_profile: FarmProfile) -> Dict:
    """Create high-risk insight"""
    return {
        "insight_id": f"HIGH_RISK_{crop_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": f"Critical Disease Risk Alert for {crop_type.title()}",
        "description": f"High disease outbreak risk detected. Immediate action required to prevent significant crop loss.",
        "priority": "critical",
        "category": "immediate",
        "estimated_cost": "$200-500 per acre",
        "time_to_implement": "Within 24 hours",
        "expected_benefit": "Prevent 70-90% potential crop loss",
        "confidence_score": 0.9,
        "supporting_data": {
            "risk_level": risk_assessment.get("risk_level", "High"),
            "probability": risk_assessment.get("overall_risk", 0),
            "recommendations": risk_assessment.get("recommendations", [])
        }
    }

def _create_moderate_risk_insight(crop_type: str, risk_assessment: Dict, farm_profile: FarmProfile) -> Dict:
    """Create moderate-risk insight"""
    return {
        "insight_id": f"MOD_RISK_{crop_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": f"Preventive Action Recommended for {crop_type.title()}",
        "description": f"Moderate disease risk detected. Preventive measures recommended to avoid potential issues.",
        "priority": "high",
        "category": "preventive",
        "estimated_cost": "$50-150 per acre",
        "time_to_implement": "Within 2-3 days",
        "expected_benefit": "Prevent 50-70% potential crop loss",
        "confidence_score": 0.75,
        "supporting_data": {
            "risk_level": risk_assessment.get("risk_level", "Moderate"),
            "probability": risk_assessment.get("overall_risk", 0),
            "recommendations": risk_assessment.get("recommendations", [])
        }
    }

def _create_low_risk_insight(crop_type: str, risk_assessment: Dict, farm_profile: FarmProfile) -> Dict:
    """Create low-risk insight"""
    return {
        "insight_id": f"LOW_RISK_{crop_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": f"Continue Monitoring {crop_type.title()}",
        "description": f"Low disease risk. Continue regular monitoring and maintain preventive practices.",
        "priority": "medium",
        "category": "monitoring",
        "estimated_cost": "$10-30 per acre",
        "time_to_implement": "Ongoing",
        "expected_benefit": "Maintain crop health and early detection",
        "confidence_score": 0.8,
        "supporting_data": {
            "risk_level": risk_assessment.get("risk_level", "Low"),
            "probability": risk_assessment.get("overall_risk", 0),
            "recommendations": risk_assessment.get("recommendations", [])
        }
    }

def _generate_environmental_insights(env_context: EnvironmentalContext, farm_profile: FarmProfile) -> List[Dict]:
    """Generate environmental insights"""
    insights = []
    
    # Temperature insights
    if env_context.temperature > 30:
        insights.append({
            "insight_id": f"TEMP_HIGH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": "High Temperature Alert",
            "description": "High temperatures may stress crops and increase disease susceptibility",
            "priority": "medium",
            "category": "immediate",
            "estimated_cost": "$20-50 per acre",
            "time_to_implement": "Today",
            "expected_benefit": "Reduce heat stress and disease risk",
            "confidence_score": 0.85,
            "supporting_data": {"temperature": env_context.temperature}
        })
    
    # Humidity insights
    if env_context.humidity > 80:
        insights.append({
            "insight_id": f"HUMID_HIGH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": "High Humidity Warning",
            "description": "High humidity creates favorable conditions for fungal diseases",
            "priority": "high",
            "category": "preventive",
            "estimated_cost": "$30-80 per acre",
            "time_to_implement": "Within 24 hours",
            "expected_benefit": "Prevent fungal disease outbreaks",
            "confidence_score": 0.9,
            "supporting_data": {"humidity": env_context.humidity}
        })
    
    return insights

def _generate_management_insights(farm_profile: FarmProfile, env_context: EnvironmentalContext) -> List[Dict]:
    """Generate management insights"""
    insights = []
    
    # Technology adoption insight
    if farm_profile.technology_adoption == "low":
        insights.append({
            "insight_id": f"TECH_UPGRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": "Consider Technology Upgrades",
            "description": "Implementing modern monitoring tools could improve disease detection",
            "priority": "low",
            "category": "planning",
            "estimated_cost": "$500-2000 one-time",
            "time_to_implement": "1-2 weeks",
            "expected_benefit": "Improve early detection by 40-60%",
            "confidence_score": 0.7,
            "supporting_data": {"current_level": farm_profile.technology_adoption}
        })
    
    return insights

def _generate_insights_summary(insights: List[Dict], farm_profile: FarmProfile) -> Dict:
    """Generate summary of insights"""
    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    category_counts = {"immediate": 0, "preventive": 0, "monitoring": 0, "planning": 0}
    
    for insight in insights:
        priority_counts[insight["priority"]] += 1
        category_counts[insight["category"]] += 1
    
    return {
        "priority_breakdown": priority_counts,
        "category_breakdown": category_counts,
        "average_confidence": sum(insight["confidence_score"] for insight in insights) / len(insights) if insights else 0,
        "immediate_actions_required": priority_counts["critical"] + priority_counts["high"],
        "total_estimated_cost": "Contact for detailed estimate",
        "expected_roi": "200-400% based on crop protection"
    }

def _get_emergency_contacts(location: str) -> List[Dict]:
    """Get emergency contacts for the location"""
    return [
        {"type": "Agricultural Extension", "phone": "+91-1800-XXX-XXXX", "available": "24/7"},
        {"type": "Veterinary Services", "phone": "+91-1800-XXX-YYYY", "available": "9 AM - 6 PM"},
        {"type": "Emergency Helpline", "phone": "+91-1800-XXX-ZZZZ", "available": "24/7"}
    ]

def _get_priority_actions(risk_assessment: Dict, crop_type: str) -> List[Dict]:
    """Get priority actions based on risk assessment"""
    actions = []
    risk_level = risk_assessment.get("risk_level", "Low")
    
    if risk_level in ["High", "Very High"]:
        actions.extend([
            {"action": "Apply emergency fungicide treatment", "priority": "critical", "deadline": "Today"},
            {"action": "Increase monitoring frequency", "priority": "high", "deadline": "Immediately"},
            {"action": "Prepare isolation areas", "priority": "high", "deadline": "Today"}
        ])
    elif risk_level == "Moderate":
        actions.extend([
            {"action": "Apply preventive treatment", "priority": "high", "deadline": "Within 2 days"},
            {"action": "Monitor weather conditions", "priority": "medium", "deadline": "Daily"},
            {"action": "Check equipment readiness", "priority": "medium", "deadline": "This week"}
        ])
    else:
        actions.extend([
            {"action": "Continue routine monitoring", "priority": "medium", "deadline": "Weekly"},
            {"action": "Maintain preventive practices", "priority": "low", "deadline": "Ongoing"},
            {"action": "Review treatment supplies", "priority": "low", "deadline": "Monthly"}
        ])
    
    return actions

def _get_monitoring_checklist(crop_type: str, risk_assessment: Dict) -> List[Dict]:
    """Get monitoring checklist"""
    return [
        {"item": "Check for disease symptoms on leaves", "frequency": "Daily", "critical": True},
        {"item": "Monitor weather conditions", "frequency": "Daily", "critical": True},
        {"item": "Inspect irrigation systems", "frequency": "Weekly", "critical": False},
        {"item": "Check soil moisture levels", "frequency": "Every 2 days", "critical": True},
        {"item": "Document any changes", "frequency": "Daily", "critical": True}
    ]

def _get_treatment_schedule(risk_assessment: Dict) -> List[Dict]:
    """Get treatment schedule"""
    risk_level = risk_assessment.get("risk_level", "Low")
    
    if risk_level in ["High", "Very High"]:
        return [
            {"treatment": "Emergency fungicide application", "timing": "Immediately", "repeat": "As needed"},
            {"treatment": "Foliar spray", "timing": "Tomorrow", "repeat": "Weekly"},
            {"treatment": "Soil treatment", "timing": "Within 3 days", "repeat": "Monthly"}
        ]
    else:
        return [
            {"treatment": "Preventive spray", "timing": "This week", "repeat": "Bi-weekly"},
            {"treatment": "Soil amendment", "timing": "Next week", "repeat": "Monthly"},
            {"treatment": "Organic treatment", "timing": "As needed", "repeat": "Weekly"}
        ]

def _estimate_resource_requirements(farm_size: float, risk_assessment: Dict) -> Dict:
    """Estimate resource requirements"""
    risk_level = risk_assessment.get("risk_level", "Low")
    multiplier = {"Very High": 2.0, "High": 1.5, "Moderate": 1.2, "Low": 1.0, "Very Low": 0.8}
    
    base_requirements = {
        "labor_hours": farm_size * 2,
        "equipment_hours": farm_size * 0.5,
        "materials_cost": farm_size * 50
    }
    
    factor = multiplier.get(risk_level, 1.0)
    
    return {
        "labor_hours": base_requirements["labor_hours"] * factor,
        "equipment_hours": base_requirements["equipment_hours"] * factor,
        "materials_cost": f"${base_requirements['materials_cost'] * factor:.0f}",
        "total_time": f"{(base_requirements['labor_hours'] + base_requirements['equipment_hours']) * factor:.1f} hours"
    }

def _get_cost_estimates(farm_size: float, risk_assessment: Dict) -> Dict:
    """Get cost estimates"""
    risk_level = risk_assessment.get("risk_level", "Low")
    base_cost_per_acre = {"Very High": 200, "High": 150, "Moderate": 100, "Low": 50, "Very Low": 25}
    
    cost_per_acre = base_cost_per_acre.get(risk_level, 50)
    total_cost = farm_size * cost_per_acre
    
    return {
        "cost_per_acre": f"${cost_per_acre}",
        "total_estimated_cost": f"${total_cost:.0f}",
        "cost_breakdown": {
            "materials": f"${total_cost * 0.6:.0f}",
            "labor": f"${total_cost * 0.3:.0f}",
            "equipment": f"${total_cost * 0.1:.0f}"
        },
        "potential_savings": f"${total_cost * 3:.0f} (crop loss prevention)"
    }

def _define_success_metrics(crop_type: str, risk_assessment: Dict) -> List[Dict]:
    """Define success metrics"""
    return [
        {"metric": "Disease incidence reduction", "target": "< 5%", "measurement": "Weekly surveys"},
        {"metric": "Crop health score", "target": "> 85%", "measurement": "Visual assessment"},
        {"metric": "Yield protection", "target": "> 90%", "measurement": "Harvest comparison"},
        {"metric": "Treatment effectiveness", "target": "> 80%", "measurement": "Follow-up monitoring"},
        {"metric": "Cost efficiency", "target": "ROI > 200%", "measurement": "Financial analysis"}
    ]

def _get_phase_name(phase_num: int) -> str:
    """Get phase name"""
    names = {1: "Assessment & Preparation", 2: "Implementation", 3: "Monitoring & Adjustment", 4: "Evaluation & Planning"}
    return names.get(phase_num, f"Phase {phase_num}")

def _get_phase_objectives(phase_num: int, crops: List[str]) -> List[str]:
    """Get phase objectives"""
    objectives_map = {
        1: ["Assess current crop health", "Prepare treatment plans", "Gather necessary resources"],
        2: ["Implement treatment protocols", "Begin intensive monitoring", "Apply preventive measures"],
        3: ["Monitor treatment effectiveness", "Adjust protocols as needed", "Document progress"],
        4: ["Evaluate overall success", "Plan for next season", "Update protocols"]
    }
    return objectives_map.get(phase_num, [])

def _get_phase_activities(phase_num: int, crops: List[str], risk_tolerance: str) -> List[str]:
    """Get phase activities"""
    activities_map = {
        1: ["Conduct field surveys", "Test soil conditions", "Check equipment", "Order supplies"],
        2: ["Apply treatments", "Set up monitoring systems", "Train staff", "Begin data collection"],
        3: ["Daily monitoring", "Weekly assessments", "Adjust treatments", "Record observations"],
        4: ["Final assessment", "Data analysis", "Report preparation", "Next season planning"]
    }
    return activities_map.get(phase_num, [])

def _get_phase_resources(phase_num: int, budget_range: str) -> Dict:
    """Get phase resource requirements"""
    budget_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.5}
    multiplier = budget_multiplier.get(budget_range, 1.0)
    
    base_resources = {
        1: {"labor": 20, "materials": 500, "equipment": 100},
        2: {"labor": 40, "materials": 1000, "equipment": 200},
        3: {"labor": 30, "materials": 300, "equipment": 150},
        4: {"labor": 15, "materials": 100, "equipment": 50}
    }
    
    resources = base_resources.get(phase_num, {"labor": 20, "materials": 200, "equipment": 100})
    
    return {
        "labor_hours": resources["labor"] * multiplier,
        "materials_cost": f"${resources['materials'] * multiplier:.0f}",
        "equipment_cost": f"${resources['equipment'] * multiplier:.0f}"
    }

def _get_phase_indicators(phase_num: int, crops: List[str]) -> List[str]:
    """Get phase success indicators"""
    indicators_map = {
        1: ["All assessments completed", "Resources secured", "Plans approved"],
        2: ["Treatments applied on schedule", "Monitoring systems active", "Staff trained"],
        3: ["Data collection on track", "No major issues", "Adjustments made as needed"],
        4: ["Objectives met", "Documentation complete", "Next season plan ready"]
    }
    return indicators_map.get(phase_num, [])

def _get_contingency_plans(phase_num: int, risk_tolerance: str) -> List[str]:
    """Get contingency plans"""
    return [
        "Emergency treatment protocols ready",
        "Alternative suppliers identified",
        "Weather contingency plans in place",
        "Expert consultation available"
    ]

def _estimate_total_cost(budget_range: str, num_crops: int) -> str:
    """Estimate total cost"""
    base_cost = {"low": 1000, "medium": 2000, "high": 4000}
    cost = base_cost.get(budget_range, 2000) * num_crops
    return f"${cost:,}"

def _define_expected_outcomes(crops: List[str], risk_tolerance: str) -> List[str]:
    """Define expected outcomes"""
    return [
        f"Protect {len(crops)} crop types from disease outbreaks",
        "Achieve 85-95% disease prevention rate",
        "Maintain crop quality and yield",
        "Reduce overall treatment costs by 20-30%",
        "Improve early detection capabilities"
    ]

def _get_risk_mitigation_strategies(risk_tolerance: str) -> List[str]:
    """Get risk mitigation strategies"""
    strategies_map = {
        "low": [
            "Conservative treatment approach",
            "Multiple backup plans",
            "Frequent monitoring",
            "Early intervention protocols"
        ],
        "medium": [
            "Balanced approach to treatments",
            "Regular monitoring schedule",
            "Contingency plans ready",
            "Cost-effective solutions"
        ],
        "high": [
            "Aggressive prevention strategy",
            "Advanced monitoring tools",
            "Quick response protocols",
            "Investment in technology"
        ]
    }
    return strategies_map.get(risk_tolerance, strategies_map["medium"])