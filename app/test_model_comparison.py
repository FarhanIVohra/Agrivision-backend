"""
Pest Detection Model Comparison Test
Compares crop_disease_model.h5 (current) vs pest_disease_model.h5 (new)
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare two pest detection models"""
    
    def __init__(self):
        self.models_dir = Path("ml_models/saved_models")
        self.current_model_path = self.models_dir / "crop_disease_model.h5"
        self.new_model_path = self.models_dir / "pest_disease_model.h5"
        
    def load_and_analyze_model(self, model_path: Path, model_name: str):
        """Load and analyze a model"""
        try:
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return None
                
            logger.info(f"\n=== Analyzing {model_name} ===")
            logger.info(f"Path: {model_path}")
            
            # Get file info
            stat = model_path.stat()
            logger.info(f"File size: {stat.st_size / (1024*1024):.2f} MB")
            logger.info(f"Modified: {datetime.fromtimestamp(stat.st_mtime)}")
            
            # Load model
            model = tf.keras.models.load_model(str(model_path))
            
            # Model architecture info
            logger.info(f"Input shape: {model.input_shape}")
            logger.info(f"Output shape: {model.output_shape}")
            logger.info(f"Total parameters: {model.count_params():,}")
            
            # Layer information
            logger.info(f"Total layers: {len(model.layers)}")
            
            # Get model summary
            logger.info("\nModel Summary:")
            model.summary(print_fn=logger.info)
            
            return {
                'model': model,
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'total_params': model.count_params(),
                'total_layers': len(model.layers),
                'file_size_mb': stat.st_size / (1024*1024),
                'modified_date': datetime.fromtimestamp(stat.st_mtime)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {model_name}: {str(e)}")
            return None
    
    def compare_models(self):
        """Compare both models"""
        logger.info("Starting model comparison...")
        
        # Analyze current model
        current_info = self.load_and_analyze_model(self.current_model_path, "Current Model (crop_disease_model.h5)")
        
        # Analyze new model
        new_info = self.load_and_analyze_model(self.new_model_path, "New Model (pest_disease_model.h5)")
        
        if not current_info or not new_info:
            logger.error("Could not load one or both models for comparison")
            return
        
        # Comparison summary
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        logger.info(f"{'Metric':<25} {'Current Model':<20} {'New Model':<20}")
        logger.info("-" * 65)
        logger.info(f"{'File Size (MB)':<25} {current_info['file_size_mb']:<20.2f} {new_info['file_size_mb']:<20.2f}")
        logger.info(f"{'Input Shape':<25} {str(current_info['input_shape']):<20} {str(new_info['input_shape']):<20}")
        logger.info(f"{'Output Classes':<25} {current_info['output_shape'][1]:<20} {new_info['output_shape'][1]:<20}")
        logger.info(f"{'Total Parameters':<25} {current_info['total_params']:<20,} {new_info['total_params']:<20,}")
        logger.info(f"{'Total Layers':<25} {current_info['total_layers']:<20} {new_info['total_layers']:<20}")
        logger.info(f"{'Modified Date':<25} {current_info['modified_date'].strftime('%Y-%m-%d'):<20} {new_info['modified_date'].strftime('%Y-%m-%d'):<20}")
        
        # Analysis
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS")
        logger.info("="*60)
        
        # Size comparison
        size_diff = new_info['file_size_mb'] - current_info['file_size_mb']
        if size_diff > 0:
            logger.info(f"✓ New model is {size_diff:.2f} MB larger (potentially more complex)")
        else:
            logger.info(f"✓ New model is {abs(size_diff):.2f} MB smaller (more efficient)")
        
        # Parameter comparison
        param_diff = new_info['total_params'] - current_info['total_params']
        if param_diff > 0:
            logger.info(f"✓ New model has {param_diff:,} more parameters")
        else:
            logger.info(f"✓ New model has {abs(param_diff):,} fewer parameters")
        
        # Output classes comparison
        current_classes = current_info['output_shape'][1]
        new_classes = new_info['output_shape'][1]
        if new_classes > current_classes:
            logger.info(f"✓ New model supports {new_classes - current_classes} more disease classes")
        elif new_classes < current_classes:
            logger.info(f"⚠ New model supports {current_classes - new_classes} fewer disease classes")
        else:
            logger.info(f"✓ Both models support the same number of classes ({current_classes})")
        
        # Recommendations
        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATIONS")
        logger.info("="*60)
        
        if new_info['file_size_mb'] < current_info['file_size_mb'] and new_classes >= current_classes:
            logger.info("🎯 RECOMMENDED: Use new model - smaller size with same or better coverage")
        elif new_classes > current_classes:
            logger.info("🎯 RECOMMENDED: Use new model - supports more disease classes")
        elif new_info['total_params'] < current_info['total_params']:
            logger.info("🎯 RECOMMENDED: Use new model - more efficient (fewer parameters)")
        else:
            logger.info("🤔 EVALUATION NEEDED: Run performance tests to determine best model")
        
        return current_info, new_info

def main():
    """Main function to run model comparison"""
    comparator = ModelComparator()
    comparator.compare_models()

if __name__ == "__main__":
    main()