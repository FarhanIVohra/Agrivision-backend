"""
Reports Router
API endpoints for generating and managing reports
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
import base64
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO

from .auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/generate")
async def generate_report(
    report_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate a PDF report with farm summary data

    Args:
        report_data: Dictionary containing summary data and metadata
        current_user: Current authenticated user

    Returns:
        Dictionary with base64 encoded PDF
    """
    try:
        logger.info(f"Generating report for user: {current_user.get('email', 'unknown')}")

        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph("AgriSmart Farm Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Generated date
        generated_at = report_data.get('generatedAt', datetime.now().isoformat())
        date_str = datetime.fromisoformat(generated_at.replace('Z', '+00:00')).strftime('%B %d, %Y %H:%M UTC')
        date_para = Paragraph(f"Generated on: {date_str}", styles['Normal'])
        story.append(date_para)
        story.append(Spacer(1, 12))

        # User info
        user = report_data.get('user', 'Farm Manager')
        user_para = Paragraph(f"Report for: {user}", styles['Normal'])
        story.append(user_para)
        story.append(Spacer(1, 24))

        # Summary section
        summary = report_data.get('summary', {})

        summary_title = Paragraph("Farm Summary", styles['Heading2'])
        story.append(summary_title)
        story.append(Spacer(1, 12))

        # Best crop recommendation
        best_crop = summary.get('bestCrop', 'Rice')
        crop_para = Paragraph(f"<b>Recommended Crop:</b> {best_crop}", styles['Normal'])
        story.append(crop_para)
        story.append(Spacer(1, 6))

        # Soil suggestions
        soil_suggestions = summary.get('soilSuggestions', 'Maintain current soil management')
        soil_para = Paragraph(f"<b>Soil Management:</b> {soil_suggestions}", styles['Normal'])
        story.append(soil_para)
        story.append(Spacer(1, 6))

        # Irrigation alert
        irrigation_alert = summary.get('irrigationAlert', 'Check irrigation status')
        irrigation_para = Paragraph(f"<b>Irrigation Status:</b> {irrigation_alert}", styles['Normal'])
        story.append(irrigation_para)
        story.append(Spacer(1, 6))

        # Pest detection
        pest_detection = summary.get('pestDetection', 'Check pest status')
        pest_para = Paragraph(f"<b>Pest Detection:</b> {pest_detection}", styles['Normal'])
        story.append(pest_para)
        story.append(Spacer(1, 6))

        # Expected ROI
        expected_roi = summary.get('expectedROI', '15-20%')
        roi_para = Paragraph(f"<b>Expected ROI:</b> {expected_roi}", styles['Normal'])
        story.append(roi_para)
        story.append(Spacer(1, 24))

        # Recommendations section
        recommendations_title = Paragraph("Recommendations", styles['Heading2'])
        story.append(recommendations_title)
        story.append(Spacer(1, 12))

        recommendations = [
            "Monitor soil moisture levels regularly",
            "Apply appropriate fertilizers based on soil analysis",
            "Implement integrated pest management practices",
            "Track weather patterns for optimal irrigation timing",
            "Maintain detailed records of crop performance"
        ]

        for rec in recommendations:
            rec_para = Paragraph(f"• {rec}", styles['Normal'])
            story.append(rec_para)
            story.append(Spacer(1, 3))

        # Build PDF
        doc.build(story)

        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()

        # Convert to base64
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')

        logger.info("Report generated successfully")

        return {
            "success": True,
            "message": "Report generated successfully",
            "pdf_base64": pdf_base64,
            "filename": f"farm-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        }

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )

@router.get("/templates")
async def get_report_templates(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get available report templates

    Args:
        current_user: Current authenticated user

    Returns:
        Dictionary with available templates
    """
    try:
        templates = [
            {
                "id": "farm_summary",
                "name": "Farm Summary Report",
                "description": "Comprehensive overview of farm performance and recommendations",
                "sections": ["crop_prediction", "soil_health", "irrigation", "pest_detection", "market_data"]
            },
            {
                "id": "crop_performance",
                "name": "Crop Performance Report",
                "description": "Detailed analysis of crop yields and health metrics",
                "sections": ["crop_prediction", "soil_health", "weather"]
            },
            {
                "id": "financial_analysis",
                "name": "Financial Analysis Report",
                "description": "Cost-benefit analysis and ROI projections",
                "sections": ["market_data", "crop_prediction", "cost_analysis"]
            }
        ]

        return {
            "success": True,
            "message": "Report templates retrieved successfully",
            "templates": templates
        }

    except Exception as e:
        logger.error(f"Error retrieving report templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve report templates"
        )
