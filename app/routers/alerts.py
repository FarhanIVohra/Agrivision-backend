"""
Alerts Router
API endpoints for alert management and notifications
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta
import random
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models
class AlertCreate(BaseModel):
    title: str
    message: str
    type: str  # critical, warning, info
    crop: Optional[str] = None
    field: Optional[str] = None
    priority: Optional[int] = 1

class AlertUpdate(BaseModel):
    status: Optional[str] = None  # active, acknowledged, resolved
    notes: Optional[str] = None

@router.get("/")
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by status (active, acknowledged, resolved)"),
    type: Optional[str] = Query(None, description="Filter by type (critical, warning, info)"),
    crop: Optional[str] = Query(None, description="Filter by crop"),
    limit: Optional[int] = Query(50, description="Maximum number of alerts to return")
) -> Dict[str, Any]:
    """Get all alerts with optional filtering"""
    try:
        logger.info(f"Retrieving alerts with filters - status: {status}, type: {type}, crop: {crop}")
        
        # Generate sample alerts
        all_alerts = generate_sample_alerts()
        
        # Apply filters
        filtered_alerts = all_alerts
        
        if status:
            filtered_alerts = [alert for alert in filtered_alerts if alert["status"] == status]
        
        if type:
            filtered_alerts = [alert for alert in filtered_alerts if alert["type"] == type]
        
        if crop:
            filtered_alerts = [alert for alert in filtered_alerts if alert.get("crop", "").lower() == crop.lower()]
        
        # Apply limit
        if limit:
            filtered_alerts = filtered_alerts[:limit]
        
        # Calculate summary
        summary = calculate_alert_summary(all_alerts)
        
        response = {
            "success": True,
            "alerts": filtered_alerts,
            "total_count": len(all_alerts),
            "filtered_count": len(filtered_alerts),
            "summary": summary,
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Retrieved {len(filtered_alerts)} alerts")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve alerts"
        )

@router.get("/active")
async def get_active_alerts() -> Dict[str, Any]:
    """Get only active alerts"""
    try:
        logger.info("Retrieving active alerts")
        
        all_alerts = generate_sample_alerts()
        active_alerts = [alert for alert in all_alerts if alert["status"] == "active"]
        
        # Sort by priority and timestamp
        active_alerts.sort(key=lambda x: (
            {"critical": 0, "warning": 1, "info": 2}.get(x["type"], 3),
            x["timestamp"]
        ), reverse=True)
        
        response = {
            "success": True,
            "alerts": active_alerts,
            "count": len(active_alerts),
            "critical_count": len([a for a in active_alerts if a["type"] == "critical"]),
            "warning_count": len([a for a in active_alerts if a["type"] == "warning"]),
            "info_count": len([a for a in active_alerts if a["type"] == "info"]),
            "last_updated": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving active alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve active alerts"
        )

@router.get("/summary")
async def get_alerts_summary() -> Dict[str, Any]:
    """Get alerts summary and statistics"""
    try:
        logger.info("Retrieving alerts summary")
        
        all_alerts = generate_sample_alerts()
        summary = calculate_alert_summary(all_alerts)
        
        # Add trend analysis
        trend_analysis = analyze_alert_trends(all_alerts)
        
        response = {
            "success": True,
            "summary": summary,
            "trends": trend_analysis,
            "recent_activity": get_recent_alert_activity(all_alerts),
            "top_issues": get_top_issues(all_alerts),
            "generated_at": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving alerts summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve alerts summary"
        )

@router.post("/")
async def create_alert(alert_data: AlertCreate) -> Dict[str, Any]:
    """Create a new alert"""
    try:
        logger.info(f"Creating new alert: {alert_data.title}")
        
        # Generate alert ID
        alert_id = f"ALT_{uuid.uuid4().hex[:8].upper()}"
        
        # Create alert object
        new_alert = {
            "id": alert_id,
            "title": alert_data.title,
            "message": alert_data.message,
            "type": alert_data.type,
            "crop": alert_data.crop,
            "field": alert_data.field,
            "priority": alert_data.priority or 1,
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "created_by": "system",
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None,
            "notes": []
        }
        
        # In a real implementation, you would save this to a database
        logger.info(f"Alert created successfully with ID: {alert_id}")
        
        response = {
            "success": True,
            "alert": new_alert,
            "message": "Alert created successfully"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create alert"
        )

@router.put("/{alert_id}")
async def update_alert(alert_id: str, update_data: AlertUpdate) -> Dict[str, Any]:
    """Update an existing alert"""
    try:
        logger.info(f"Updating alert: {alert_id}")
        
        # In a real implementation, you would fetch the alert from database
        # For now, we'll simulate the update
        
        updated_fields = {}
        if update_data.status:
            updated_fields["status"] = update_data.status
            if update_data.status == "acknowledged":
                updated_fields["acknowledged_at"] = datetime.now().isoformat()
                updated_fields["acknowledged_by"] = "user"
            elif update_data.status == "resolved":
                updated_fields["resolved_at"] = datetime.now().isoformat()
        
        if update_data.notes:
            updated_fields["notes"] = update_data.notes
        
        response = {
            "success": True,
            "alert_id": alert_id,
            "updated_fields": updated_fields,
            "message": "Alert updated successfully"
        }
        
        logger.info(f"Alert {alert_id} updated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error updating alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update alert"
        )

@router.delete("/{alert_id}")
async def delete_alert(alert_id: str) -> Dict[str, Any]:
    """Delete an alert"""
    try:
        logger.info(f"Deleting alert: {alert_id}")
        
        # In a real implementation, you would delete from database
        
        response = {
            "success": True,
            "alert_id": alert_id,
            "message": "Alert deleted successfully"
        }
        
        logger.info(f"Alert {alert_id} deleted successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error deleting alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete alert"
        )

@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
    """Acknowledge an alert"""
    try:
        logger.info(f"Acknowledging alert: {alert_id}")
        
        response = {
            "success": True,
            "alert_id": alert_id,
            "acknowledged_at": datetime.now().isoformat(),
            "acknowledged_by": "user",
            "message": "Alert acknowledged successfully"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to acknowledge alert"
        )

@router.post("/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """Resolve an alert"""
    try:
        logger.info(f"Resolving alert: {alert_id}")
        
        response = {
            "success": True,
            "alert_id": alert_id,
            "resolved_at": datetime.now().isoformat(),
            "resolved_by": "user",
            "message": "Alert resolved successfully"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to resolve alert"
        )

def generate_sample_alerts() -> List[Dict[str, Any]]:
    """Generate sample alerts for testing"""
    alerts = [
        {
            "id": "ALT_001",
            "type": "critical",
            "title": "High Pest Risk Detected",
            "message": "Brown planthopper risk is high in rice fields. Immediate action recommended.",
            "crop": "Rice",
            "field": "Field A",
            "priority": 1,
            "status": "active",
            "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
            "created_by": "system",
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None
        },
        {
            "id": "ALT_002",
            "type": "critical",
            "title": "Low Soil Moisture",
            "message": "Soil moisture in wheat field has dropped below critical level (15%).",
            "crop": "Wheat",
            "field": "North Field",
            "priority": 1,
            "status": "active",
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "created_by": "system",
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None
        },
        {
            "id": "ALT_003",
            "type": "warning",
            "title": "Weather Alert",
            "message": "Heavy rainfall expected in next 48 hours. Adjust irrigation schedule.",
            "crop": "All",
            "field": "All Fields",
            "priority": 2,
            "status": "active",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "created_by": "weather_service",
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None
        },
        {
            "id": "ALT_004",
            "type": "warning",
            "title": "Nutrient Deficiency",
            "message": "Nitrogen levels are below optimal in maize field.",
            "crop": "Maize",
            "field": "South Field",
            "priority": 2,
            "status": "acknowledged",
            "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
            "created_by": "system",
            "acknowledged_by": "farmer_john",
            "acknowledged_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "resolved_at": None
        },
        {
            "id": "ALT_005",
            "type": "info",
            "title": "Harvest Reminder",
            "message": "Tomato crop in greenhouse is ready for harvest.",
            "crop": "Tomato",
            "field": "Greenhouse 1",
            "priority": 3,
            "status": "active",
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "created_by": "system",
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None
        },
        {
            "id": "ALT_006",
            "type": "warning",
            "title": "Equipment Maintenance",
            "message": "Irrigation pump in Field B requires maintenance check.",
            "crop": None,
            "field": "Field B",
            "priority": 2,
            "status": "active",
            "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
            "created_by": "maintenance_system",
            "acknowledged_by": None,
            "acknowledged_at": None,
            "resolved_at": None
        },
        {
            "id": "ALT_007",
            "type": "info",
            "title": "Market Price Update",
            "message": "Rice prices have increased by 8% in the local market.",
            "crop": "Rice",
            "field": None,
            "priority": 3,
            "status": "resolved",
            "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
            "created_by": "market_monitor",
            "acknowledged_by": "farmer_john",
            "acknowledged_at": (datetime.now() - timedelta(hours=10)).isoformat(),
            "resolved_at": (datetime.now() - timedelta(hours=8)).isoformat()
        }
    ]
    
    return alerts

def calculate_alert_summary(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate alert summary statistics"""
    total = len(alerts)
    active = len([a for a in alerts if a["status"] == "active"])
    acknowledged = len([a for a in alerts if a["status"] == "acknowledged"])
    resolved = len([a for a in alerts if a["status"] == "resolved"])
    
    critical = len([a for a in alerts if a["type"] == "critical"])
    warning = len([a for a in alerts if a["type"] == "warning"])
    info = len([a for a in alerts if a["type"] == "info"])
    
    return {
        "total": total,
        "active": active,
        "acknowledged": acknowledged,
        "resolved": resolved,
        "critical": critical,
        "warning": warning,
        "info": info,
        "resolution_rate": round((resolved / total * 100) if total > 0 else 0, 1)
    }

def analyze_alert_trends(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze alert trends over time"""
    # Simple trend analysis
    now = datetime.now()
    last_24h = [a for a in alerts if datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00').replace('+00:00', '')) > now - timedelta(hours=24)]
    last_week = [a for a in alerts if datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00').replace('+00:00', '')) > now - timedelta(days=7)]
    
    return {
        "last_24h_count": len(last_24h),
        "last_week_count": len(last_week),
        "trend": "increasing" if len(last_24h) > 3 else "stable",
        "most_common_type": "warning",  # Simplified
        "most_affected_crop": "Rice"     # Simplified
    }

def get_recent_alert_activity(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get recent alert activity"""
    # Sort by timestamp and get recent ones
    sorted_alerts = sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    recent = sorted_alerts[:5]
    
    activity = []
    for alert in recent:
        activity.append({
            "alert_id": alert["id"],
            "title": alert["title"],
            "type": alert["type"],
            "status": alert["status"],
            "timestamp": alert["timestamp"]
        })
    
    return activity

def get_top_issues(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get top issues by frequency"""
    # Simplified top issues
    return [
        {"issue": "Pest Risk", "count": 3, "severity": "High"},
        {"issue": "Soil Moisture", "count": 2, "severity": "Medium"},
        {"issue": "Weather Alerts", "count": 2, "severity": "Medium"},
        {"issue": "Equipment Maintenance", "count": 1, "severity": "Low"}
    ]