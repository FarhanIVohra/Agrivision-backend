"""
Disease Database Service
Comprehensive database service for crop disease information, symptoms, and management strategies
Aligns with AI-Driven Crop Disease Prediction and Management System requirements
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

class DiseaseInformationService:
    """Service for managing comprehensive disease information database"""
    
    def __init__(self):
        """Initialize the disease information service"""
        self.disease_database = self._initialize_disease_database()
        self.symptom_database = self._initialize_symptom_database()
        self.treatment_database = self._initialize_treatment_database()
        self.prevention_database = self._initialize_prevention_database()
        logger.info("Disease Information Service initialized successfully")
    
    def _initialize_disease_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive disease database"""
        return {
            "bacterial_blight": {
                "name": "Bacterial Blight",
                "scientific_name": "Xanthomonas oryzae pv. oryzae",
                "crop_types": ["rice", "wheat", "barley"],
                "severity": "high",
                "description": "A serious bacterial disease that affects rice and other cereals, causing leaf blight and yield reduction.",
                "pathogen_type": "bacteria",
                "favorable_conditions": {
                    "temperature_range": [25, 35],
                    "humidity_range": [80, 95],
                    "rainfall": "high",
                    "wind": "moderate_to_high"
                },
                "economic_impact": "high",
                "distribution": "worldwide",
                "host_range": ["rice", "wheat", "barley", "oats"],
                "transmission": ["wind", "rain", "infected_seeds", "contaminated_tools"],
                "survival": "Can survive in crop residues and alternative hosts",
                "detection_methods": ["visual_inspection", "laboratory_testing", "molecular_diagnostics"]
            },
            "blast_disease": {
                "name": "Blast Disease",
                "scientific_name": "Magnaporthe oryzae",
                "crop_types": ["rice", "wheat", "barley"],
                "severity": "very_high",
                "description": "One of the most destructive diseases of rice, causing significant yield losses worldwide.",
                "pathogen_type": "fungus",
                "favorable_conditions": {
                    "temperature_range": [20, 30],
                    "humidity_range": [85, 95],
                    "rainfall": "moderate_to_high",
                    "wind": "low_to_moderate"
                },
                "economic_impact": "very_high",
                "distribution": "worldwide",
                "host_range": ["rice", "wheat", "barley", "finger_millet"],
                "transmission": ["airborne_spores", "infected_seeds", "crop_residues"],
                "survival": "Survives in infected plant debris and alternative hosts",
                "detection_methods": ["visual_inspection", "spore_trapping", "molecular_diagnostics"]
            },
            "brown_spot": {
                "name": "Brown Spot",
                "scientific_name": "Bipolaris oryzae",
                "crop_types": ["rice"],
                "severity": "moderate",
                "description": "A fungal disease causing brown spots on rice leaves, affecting photosynthesis and grain quality.",
                "pathogen_type": "fungus",
                "favorable_conditions": {
                    "temperature_range": [25, 30],
                    "humidity_range": [75, 90],
                    "rainfall": "moderate",
                    "wind": "low"
                },
                "economic_impact": "moderate",
                "distribution": "tropical_and_subtropical",
                "host_range": ["rice"],
                "transmission": ["airborne_spores", "infected_seeds"],
                "survival": "Survives in infected seeds and crop residues",
                "detection_methods": ["visual_inspection", "microscopic_examination"]
            },
            "sheath_blight": {
                "name": "Sheath Blight",
                "scientific_name": "Rhizoctonia solani",
                "crop_types": ["rice", "maize", "wheat"],
                "severity": "high",
                "description": "A soil-borne fungal disease affecting the sheath and stem of rice plants.",
                "pathogen_type": "fungus",
                "favorable_conditions": {
                    "temperature_range": [28, 32],
                    "humidity_range": [80, 95],
                    "rainfall": "high",
                    "wind": "low"
                },
                "economic_impact": "high",
                "distribution": "worldwide",
                "host_range": ["rice", "maize", "wheat", "soybean"],
                "transmission": ["soil_borne", "water", "infected_plant_material"],
                "survival": "Survives as sclerotia in soil and plant debris",
                "detection_methods": ["visual_inspection", "soil_testing", "laboratory_culture"]
            },
            "downy_mildew": {
                "name": "Downy Mildew",
                "scientific_name": "Peronosclerospora spp.",
                "crop_types": ["maize", "sorghum", "pearl_millet"],
                "severity": "high",
                "description": "A destructive disease of cereals causing stunting and yield reduction.",
                "pathogen_type": "oomycete",
                "favorable_conditions": {
                    "temperature_range": [20, 25],
                    "humidity_range": [85, 95],
                    "rainfall": "high",
                    "wind": "moderate"
                },
                "economic_impact": "high",
                "distribution": "tropical_and_subtropical",
                "host_range": ["maize", "sorghum", "pearl_millet", "sugarcane"],
                "transmission": ["airborne_spores", "soil_borne", "infected_seeds"],
                "survival": "Survives as oospores in soil and infected plant material",
                "detection_methods": ["visual_inspection", "microscopic_examination", "molecular_diagnostics"]
            },
            "powdery_mildew": {
                "name": "Powdery Mildew",
                "scientific_name": "Erysiphe graminis",
                "crop_types": ["wheat", "barley", "oats"],
                "severity": "moderate",
                "description": "A fungal disease causing white powdery growth on leaves and stems.",
                "pathogen_type": "fungus",
                "favorable_conditions": {
                    "temperature_range": [15, 25],
                    "humidity_range": [60, 80],
                    "rainfall": "low_to_moderate",
                    "wind": "moderate"
                },
                "economic_impact": "moderate",
                "distribution": "temperate_regions",
                "host_range": ["wheat", "barley", "oats", "rye"],
                "transmission": ["airborne_spores"],
                "survival": "Survives as cleistothecia on crop residues",
                "detection_methods": ["visual_inspection", "microscopic_examination"]
            },
            "rust_diseases": {
                "name": "Rust Diseases",
                "scientific_name": "Puccinia spp.",
                "crop_types": ["wheat", "barley", "oats", "maize"],
                "severity": "high",
                "description": "A group of fungal diseases causing rust-colored pustules on leaves and stems.",
                "pathogen_type": "fungus",
                "favorable_conditions": {
                    "temperature_range": [15, 25],
                    "humidity_range": [70, 90],
                    "rainfall": "moderate",
                    "wind": "moderate_to_high"
                },
                "economic_impact": "high",
                "distribution": "worldwide",
                "host_range": ["wheat", "barley", "oats", "maize", "rye"],
                "transmission": ["airborne_spores", "wind"],
                "survival": "Survives on alternative hosts and crop residues",
                "detection_methods": ["visual_inspection", "spore_identification", "molecular_diagnostics"]
            },
            "late_blight": {
                "name": "Late Blight",
                "scientific_name": "Phytophthora infestans",
                "crop_types": ["potato", "tomato"],
                "severity": "very_high",
                "description": "A devastating disease of potato and tomato causing rapid plant death.",
                "pathogen_type": "oomycete",
                "favorable_conditions": {
                    "temperature_range": [15, 20],
                    "humidity_range": [85, 95],
                    "rainfall": "high",
                    "wind": "moderate"
                },
                "economic_impact": "very_high",
                "distribution": "worldwide",
                "host_range": ["potato", "tomato", "eggplant"],
                "transmission": ["airborne_spores", "infected_tubers", "water"],
                "survival": "Survives in infected tubers and plant debris",
                "detection_methods": ["visual_inspection", "laboratory_testing", "molecular_diagnostics"]
            }
        }
    
    def _initialize_symptom_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize symptom database with detailed descriptions"""
        return {
            "bacterial_blight": {
                "early_symptoms": [
                    "Small, water-soaked lesions on leaf tips and margins",
                    "Yellowing of leaf edges",
                    "Wilting of young leaves"
                ],
                "advanced_symptoms": [
                    "Large, irregular brown lesions with yellow halos",
                    "Leaf blight progressing from tips to base",
                    "Bacterial ooze from cut stems",
                    "Stunted plant growth",
                    "Reduced tillering"
                ],
                "diagnostic_features": [
                    "Water-soaked appearance of lesions",
                    "Yellow halos around lesions",
                    "Bacterial streaming from cut tissues",
                    "Systemic wilting in severe cases"
                ],
                "affected_parts": ["leaves", "stems", "panicles"],
                "confusion_with": ["blast_disease", "brown_spot"],
                "seasonal_variation": "More severe during monsoon season"
            },
            "blast_disease": {
                "early_symptoms": [
                    "Small, diamond-shaped lesions with gray centers",
                    "Brown borders around lesions",
                    "Lesions on leaves, nodes, and panicles"
                ],
                "advanced_symptoms": [
                    "Large, spindle-shaped lesions on leaves",
                    "Node infection causing stem breakage",
                    "Neck blast causing panicle death",
                    "Complete plant collapse in severe cases"
                ],
                "diagnostic_features": [
                    "Diamond or spindle-shaped lesions",
                    "Gray centers with brown borders",
                    "Lesions at nodes causing breakage",
                    "Abundant sporulation under humid conditions"
                ],
                "affected_parts": ["leaves", "nodes", "panicles", "stems"],
                "confusion_with": ["bacterial_blight", "brown_spot"],
                "seasonal_variation": "Peak during cool, humid weather"
            },
            "brown_spot": {
                "early_symptoms": [
                    "Small, circular brown spots on leaves",
                    "Spots with light brown centers",
                    "Dark brown borders around spots"
                ],
                "advanced_symptoms": [
                    "Numerous spots coalescing into large patches",
                    "Premature leaf senescence",
                    "Reduced photosynthetic area",
                    "Poor grain filling"
                ],
                "diagnostic_features": [
                    "Circular to oval brown spots",
                    "Light centers with dark borders",
                    "Concentric rings in mature lesions",
                    "Spots on both leaf surfaces"
                ],
                "affected_parts": ["leaves", "leaf_sheaths", "panicles"],
                "confusion_with": ["blast_disease", "bacterial_blight"],
                "seasonal_variation": "Common during warm, humid conditions"
            },
            "sheath_blight": {
                "early_symptoms": [
                    "Water-soaked lesions on leaf sheaths near water line",
                    "Oval to irregular shaped lesions",
                    "Lesions with white centers and brown borders"
                ],
                "advanced_symptoms": [
                    "Lesions expanding upward on plant",
                    "Sclerotia formation on infected tissues",
                    "Lodging of heavily infected plants",
                    "Reduced grain weight and quality"
                ],
                "diagnostic_features": [
                    "Characteristic sclerotia (small, dark, round structures)",
                    "Lesions starting near water line",
                    "White mycelial growth under humid conditions",
                    "Irregular lesion shapes"
                ],
                "affected_parts": ["leaf_sheaths", "stems", "panicles"],
                "confusion_with": ["bacterial_blight"],
                "seasonal_variation": "Severe during warm, humid weather with dense canopy"
            }
        }
    
    def _initialize_treatment_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize treatment database with detailed management strategies"""
        return {
            "bacterial_blight": {
                "chemical_control": {
                    "bactericides": [
                        {
                            "name": "Copper oxychloride",
                            "concentration": "0.2%",
                            "application_timing": "Early symptoms or preventive",
                            "frequency": "7-10 days interval",
                            "notes": "Most effective when applied early"
                        },
                        {
                            "name": "Streptomycin sulfate",
                            "concentration": "100-200 ppm",
                            "application_timing": "Early infection stage",
                            "frequency": "10-14 days interval",
                            "notes": "Systemic action, use with caution"
                        }
                    ],
                    "application_method": "Foliar spray",
                    "precautions": ["Avoid spraying during flowering", "Use protective equipment"]
                },
                "biological_control": {
                    "agents": [
                        {
                            "name": "Pseudomonas fluorescens",
                            "application": "Seed treatment and foliar spray",
                            "effectiveness": "70-80%",
                            "notes": "Environmentally safe, promotes plant growth"
                        },
                        {
                            "name": "Bacillus subtilis",
                            "application": "Foliar spray",
                            "effectiveness": "60-75%",
                            "notes": "Compatible with other treatments"
                        }
                    ]
                },
                "cultural_control": [
                    "Use resistant varieties",
                    "Avoid excessive nitrogen fertilization",
                    "Maintain proper plant spacing",
                    "Remove infected plant debris",
                    "Avoid overhead irrigation"
                ],
                "integrated_approach": "Combine resistant varieties with biological agents and judicious use of bactericides"
            },
            "blast_disease": {
                "chemical_control": {
                    "fungicides": [
                        {
                            "name": "Tricyclazole",
                            "concentration": "0.06%",
                            "application_timing": "Preventive or early symptoms",
                            "frequency": "10-15 days interval",
                            "notes": "Highly effective against blast"
                        },
                        {
                            "name": "Carbendazim",
                            "concentration": "0.1%",
                            "application_timing": "Early infection",
                            "frequency": "14 days interval",
                            "notes": "Systemic fungicide"
                        },
                        {
                            "name": "Propiconazole",
                            "concentration": "0.1%",
                            "application_timing": "Preventive",
                            "frequency": "15-20 days interval",
                            "notes": "Broad spectrum activity"
                        }
                    ],
                    "application_method": "Foliar spray",
                    "resistance_management": "Rotate fungicides with different modes of action"
                },
                "biological_control": {
                    "agents": [
                        {
                            "name": "Trichoderma viride",
                            "application": "Seed treatment and soil application",
                            "effectiveness": "65-75%",
                            "notes": "Enhances plant immunity"
                        }
                    ]
                },
                "cultural_control": [
                    "Use blast-resistant varieties",
                    "Balanced fertilization (avoid excess nitrogen)",
                    "Proper water management",
                    "Remove infected stubble",
                    "Adjust planting dates"
                ],
                "host_resistance": "Deploy varieties with multiple resistance genes"
            },
            "brown_spot": {
                "chemical_control": {
                    "fungicides": [
                        {
                            "name": "Mancozeb",
                            "concentration": "0.2%",
                            "application_timing": "Early symptoms",
                            "frequency": "10-14 days interval",
                            "notes": "Protective fungicide"
                        },
                        {
                            "name": "Propiconazole",
                            "concentration": "0.1%",
                            "application_timing": "Early infection",
                            "frequency": "15 days interval",
                            "notes": "Systemic action"
                        }
                    ]
                },
                "cultural_control": [
                    "Use healthy, certified seeds",
                    "Maintain proper nutrition (especially potassium)",
                    "Avoid water stress",
                    "Remove infected plant debris",
                    "Crop rotation"
                ],
                "nutritional_management": "Ensure adequate potassium and silicon nutrition"
            },
            "sheath_blight": {
                "chemical_control": {
                    "fungicides": [
                        {
                            "name": "Validamycin",
                            "concentration": "0.2%",
                            "application_timing": "Early symptoms near water line",
                            "frequency": "15-20 days interval",
                            "notes": "Specific for Rhizoctonia"
                        },
                        {
                            "name": "Hexaconazole",
                            "concentration": "0.1%",
                            "application_timing": "Preventive or early symptoms",
                            "frequency": "15 days interval",
                            "notes": "Systemic fungicide"
                        }
                    ]
                },
                "biological_control": {
                    "agents": [
                        {
                            "name": "Trichoderma harzianum",
                            "application": "Soil application",
                            "effectiveness": "60-70%",
                            "notes": "Competes with pathogen in soil"
                        }
                    ]
                },
                "cultural_control": [
                    "Avoid excessive nitrogen fertilization",
                    "Maintain proper plant spacing",
                    "Drain fields periodically",
                    "Remove infected stubble",
                    "Use silicon fertilizers"
                ]
            }
        }
    
    def _initialize_prevention_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize prevention database with comprehensive strategies"""
        return {
            "general_prevention": {
                "seed_treatment": [
                    "Use certified, disease-free seeds",
                    "Treat seeds with appropriate fungicides/bactericides",
                    "Hot water treatment for some diseases",
                    "Biological seed treatment with beneficial microorganisms"
                ],
                "field_sanitation": [
                    "Remove and destroy infected plant debris",
                    "Clean farm equipment between fields",
                    "Control alternative hosts and weeds",
                    "Proper disposal of infected materials"
                ],
                "crop_management": [
                    "Use resistant/tolerant varieties",
                    "Follow recommended planting dates",
                    "Maintain proper plant spacing",
                    "Implement crop rotation",
                    "Balanced fertilization"
                ],
                "water_management": [
                    "Avoid overhead irrigation when possible",
                    "Ensure proper drainage",
                    "Irrigate during morning hours",
                    "Maintain appropriate water levels"
                ],
                "monitoring": [
                    "Regular field scouting",
                    "Weather monitoring",
                    "Disease forecasting systems",
                    "Early detection protocols"
                ]
            },
            "bacterial_blight": {
                "specific_prevention": [
                    "Use bacterial blight resistant varieties",
                    "Avoid mechanical damage to plants",
                    "Control insect vectors",
                    "Quarantine measures for new planting material",
                    "Avoid working in wet fields"
                ],
                "resistant_varieties": [
                    "IR64", "Swarna", "MTU1010", "BPT5204"
                ]
            },
            "blast_disease": {
                "specific_prevention": [
                    "Deploy blast-resistant varieties",
                    "Avoid excessive nitrogen fertilization",
                    "Maintain proper plant nutrition",
                    "Monitor weather conditions",
                    "Use disease forecasting models"
                ],
                "resistant_varieties": [
                    "Pusa Basmati 1121", "IR64", "Swarna", "Samba Mahsuri"
                ]
            },
            "environmental_management": {
                "microclimate_modification": [
                    "Improve air circulation",
                    "Reduce humidity around plants",
                    "Manage plant density",
                    "Use mulching appropriately"
                ],
                "soil_health": [
                    "Maintain soil pH",
                    "Improve soil organic matter",
                    "Ensure proper drainage",
                    "Use beneficial microorganisms"
                ]
            }
        }
    
    def get_disease_information(self, disease_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific disease
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Dictionary with comprehensive disease information
        """
        try:
            disease_key = disease_name.lower().replace(" ", "_")
            
            if disease_key not in self.disease_database:
                # Try to find partial matches
                matches = [key for key in self.disease_database.keys() if disease_key in key or key in disease_key]
                if matches:
                    disease_key = matches[0]
                else:
                    return {"error": f"Disease '{disease_name}' not found in database"}
            
            disease_info = self.disease_database[disease_key].copy()
            
            # Add symptoms if available
            if disease_key in self.symptom_database:
                disease_info["symptoms"] = self.symptom_database[disease_key]
            
            # Add treatment information if available
            if disease_key in self.treatment_database:
                disease_info["treatment"] = self.treatment_database[disease_key]
            
            # Add prevention information
            prevention_info = self.prevention_database.get("general_prevention", {})
            if disease_key in self.prevention_database:
                prevention_info.update(self.prevention_database[disease_key])
            disease_info["prevention"] = prevention_info
            
            # Add metadata
            disease_info["last_updated"] = datetime.now().isoformat()
            disease_info["database_version"] = "1.0"
            
            return disease_info
            
        except Exception as e:
            logger.error(f"Error retrieving disease information: {str(e)}")
            return {"error": f"Failed to retrieve disease information: {str(e)}"}
    
    def search_diseases_by_symptoms(self, symptoms: List[str], crop_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for diseases based on observed symptoms
        
        Args:
            symptoms: List of observed symptoms
            crop_type: Optional crop type to filter results
            
        Returns:
            List of possible diseases with match scores
        """
        try:
            results = []
            
            for disease_key, disease_info in self.disease_database.items():
                # Filter by crop type if specified
                if crop_type and crop_type.lower() not in [crop.lower() for crop in disease_info.get("crop_types", [])]:
                    continue
                
                # Calculate symptom match score
                match_score = 0
                total_symptoms = 0
                
                if disease_key in self.symptom_database:
                    symptom_info = self.symptom_database[disease_key]
                    all_disease_symptoms = []
                    
                    # Collect all symptoms for this disease
                    for symptom_category in ["early_symptoms", "advanced_symptoms", "diagnostic_features"]:
                        all_disease_symptoms.extend(symptom_info.get(symptom_category, []))
                    
                    total_symptoms = len(all_disease_symptoms)
                    
                    # Check for symptom matches
                    for observed_symptom in symptoms:
                        for disease_symptom in all_disease_symptoms:
                            if self._symptoms_match(observed_symptom.lower(), disease_symptom.lower()):
                                match_score += 1
                                break
                
                # Calculate match percentage
                match_percentage = (match_score / max(len(symptoms), 1)) * 100 if total_symptoms > 0 else 0
                
                if match_percentage > 20:  # Only include diseases with >20% match
                    results.append({
                        "disease_name": disease_info["name"],
                        "scientific_name": disease_info.get("scientific_name", ""),
                        "match_percentage": round(match_percentage, 1),
                        "severity": disease_info.get("severity", "unknown"),
                        "crop_types": disease_info.get("crop_types", []),
                        "key_symptoms": self.symptom_database.get(disease_key, {}).get("diagnostic_features", [])[:3]
                    })
            
            # Sort by match percentage
            results.sort(key=lambda x: x["match_percentage"], reverse=True)
            
            return results[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.error(f"Error searching diseases by symptoms: {str(e)}")
            return []
    
    def get_treatment_recommendations(self, disease_name: str, severity: str = "moderate", 
                                   crop_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed treatment recommendations for a specific disease
        
        Args:
            disease_name: Name of the disease
            severity: Severity level (mild, moderate, severe)
            crop_type: Optional crop type for specific recommendations
            
        Returns:
            Dictionary with treatment recommendations
        """
        try:
            disease_key = disease_name.lower().replace(" ", "_")
            
            if disease_key not in self.treatment_database:
                # Try to find partial matches
                matches = [key for key in self.treatment_database.keys() if disease_key in key or key in disease_key]
                if matches:
                    disease_key = matches[0]
                else:
                    return {"error": f"Treatment information for '{disease_name}' not found"}
            
            treatment_info = self.treatment_database[disease_key].copy()
            
            # Adjust recommendations based on severity
            if severity.lower() == "severe":
                # Prioritize chemical control for severe cases
                treatment_info["priority"] = "chemical_control"
                treatment_info["urgency"] = "immediate"
            elif severity.lower() == "mild":
                # Prioritize biological and cultural control for mild cases
                treatment_info["priority"] = "biological_control"
                treatment_info["urgency"] = "within_week"
            else:
                # Balanced approach for moderate cases
                treatment_info["priority"] = "integrated_approach"
                treatment_info["urgency"] = "within_3_days"
            
            # Add general recommendations
            treatment_info["general_recommendations"] = [
                "Monitor weather conditions",
                "Document treatment applications",
                "Follow up with field inspections",
                "Maintain treatment records"
            ]
            
            # Add cost estimates (mock data)
            treatment_info["cost_estimates"] = {
                "chemical_control": "$50-150 per acre",
                "biological_control": "$30-80 per acre",
                "cultural_control": "$20-50 per acre"
            }
            
            return treatment_info
            
        except Exception as e:
            logger.error(f"Error getting treatment recommendations: {str(e)}")
            return {"error": f"Failed to get treatment recommendations: {str(e)}"}
    
    def get_prevention_strategies(self, crop_type: str, region: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive prevention strategies for a specific crop
        
        Args:
            crop_type: Type of crop
            region: Optional region for location-specific recommendations
            
        Returns:
            Dictionary with prevention strategies
        """
        try:
            # Get general prevention strategies
            prevention_strategies = self.prevention_database.get("general_prevention", {}).copy()
            
            # Add crop-specific strategies
            crop_diseases = []
            for disease_key, disease_info in self.disease_database.items():
                if crop_type.lower() in [crop.lower() for crop in disease_info.get("crop_types", [])]:
                    crop_diseases.append({
                        "disease": disease_info["name"],
                        "severity": disease_info.get("severity", "unknown"),
                        "prevention": self.prevention_database.get(disease_key, {}).get("specific_prevention", [])
                    })
            
            prevention_strategies["crop_specific"] = {
                "crop_type": crop_type,
                "common_diseases": crop_diseases,
                "resistant_varieties": self._get_resistant_varieties(crop_type),
                "seasonal_calendar": self._get_prevention_calendar(crop_type)
            }
            
            # Add region-specific recommendations if available
            if region:
                prevention_strategies["regional_considerations"] = self._get_regional_recommendations(region)
            
            return prevention_strategies
            
        except Exception as e:
            logger.error(f"Error getting prevention strategies: {str(e)}")
            return {"error": f"Failed to get prevention strategies: {str(e)}"}
    
    def get_disease_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the disease database"""
        try:
            stats = {
                "total_diseases": len(self.disease_database),
                "diseases_by_pathogen": {},
                "diseases_by_severity": {},
                "diseases_by_crop": {},
                "database_coverage": {
                    "diseases_with_symptoms": len(self.symptom_database),
                    "diseases_with_treatments": len(self.treatment_database),
                    "diseases_with_prevention": len([k for k in self.prevention_database.keys() if k != "general_prevention"])
                }
            }
            
            # Count by pathogen type
            for disease_info in self.disease_database.values():
                pathogen = disease_info.get("pathogen_type", "unknown")
                stats["diseases_by_pathogen"][pathogen] = stats["diseases_by_pathogen"].get(pathogen, 0) + 1
            
            # Count by severity
            for disease_info in self.disease_database.values():
                severity = disease_info.get("severity", "unknown")
                stats["diseases_by_severity"][severity] = stats["diseases_by_severity"].get(severity, 0) + 1
            
            # Count by crop type
            for disease_info in self.disease_database.values():
                for crop in disease_info.get("crop_types", []):
                    stats["diseases_by_crop"][crop] = stats["diseases_by_crop"].get(crop, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": f"Failed to get database stats: {str(e)}"}
    
    def _symptoms_match(self, observed: str, database: str) -> bool:
        """Check if observed symptom matches database symptom"""
        # Simple keyword matching (can be improved with NLP)
        observed_words = set(observed.split())
        database_words = set(database.split())
        
        # Check for common words (at least 2 words match or 50% overlap)
        common_words = observed_words.intersection(database_words)
        return len(common_words) >= 2 or len(common_words) / max(len(observed_words), 1) >= 0.5
    
    def _get_resistant_varieties(self, crop_type: str) -> List[str]:
        """Get resistant varieties for a crop type"""
        varieties_map = {
            "rice": ["IR64", "Swarna", "Pusa Basmati 1121", "Samba Mahsuri", "MTU1010"],
            "wheat": ["HD2967", "PBW343", "WH147", "DBW17", "HD3086"],
            "maize": ["DHM117", "HQPM1", "Vivek QPM9", "HQPM5", "DHM121"],
            "potato": ["Kufri Jyoti", "Kufri Chipsona", "Kufri Badshah", "Kufri Pukhraj"],
            "tomato": ["Arka Rakshak", "Arka Samrat", "Pusa Ruby", "Hisar Arun"]
        }
        return varieties_map.get(crop_type.lower(), [])
    
    def _get_prevention_calendar(self, crop_type: str) -> Dict[str, List[str]]:
        """Get seasonal prevention calendar"""
        return {
            "pre_planting": [
                "Select resistant varieties",
                "Treat seeds with fungicides/bactericides",
                "Prepare disease-free nursery",
                "Clean and disinfect equipment"
            ],
            "planting": [
                "Follow recommended spacing",
                "Use certified seeds",
                "Apply soil amendments",
                "Install monitoring systems"
            ],
            "vegetative_stage": [
                "Regular field scouting",
                "Maintain proper nutrition",
                "Monitor weather conditions",
                "Apply preventive treatments if needed"
            ],
            "reproductive_stage": [
                "Intensive monitoring",
                "Adjust irrigation practices",
                "Apply protective treatments",
                "Document any disease incidents"
            ],
            "post_harvest": [
                "Remove crop residues",
                "Clean storage facilities",
                "Plan crop rotation",
                "Analyze season's disease data"
            ]
        }
    
    def _get_regional_recommendations(self, region: str) -> List[str]:
        """Get region-specific recommendations"""
        regional_map = {
            "tropical": [
                "Focus on fungal disease prevention during monsoon",
                "Improve drainage systems",
                "Use heat-tolerant varieties",
                "Monitor for vector-borne diseases"
            ],
            "temperate": [
                "Prepare for frost damage",
                "Monitor for cold-weather diseases",
                "Adjust planting dates",
                "Use cold-tolerant varieties"
            ],
            "arid": [
                "Focus on water management",
                "Monitor for drought stress diseases",
                "Use drought-tolerant varieties",
                "Implement efficient irrigation"
            ]
        }
        return regional_map.get(region.lower(), [
            "Follow local agricultural extension recommendations",
            "Monitor local disease outbreaks",
            "Adapt practices to local conditions",
            "Consult with local experts"
        ])