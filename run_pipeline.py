"""
Main pipeline runner for Startup Growth Analytics.
Executes all stages of the MLOps pipeline in sequence.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_stage(stage_name: str, module_path: str):
    """
    Run a single pipeline stage.
    
    Args:
        stage_name: Name of the stage
        module_path: Path to the module to execute
    """
    logger.info("=" * 80)
    logger.info(f"STARTING STAGE: {stage_name}")
    logger.info("=" * 80)
    
    try:
        # Import and run the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(stage_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run the main function
        if hasattr(module, 'main'):
            module.main()
        
        logger.info(f"✓ Stage '{stage_name}' completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Stage '{stage_name}' failed: {e}")
        return False

def main():
    """Main function to run the complete pipeline."""
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("STARTUP GROWTH ANALYTICS - MLOps PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Pipeline started at: {start_time}")
    logger.info("")
    
    # Define pipeline stages
    stages = [
        ("Data Ingestion", "src/data_ingestion.py"),
        ("Feature Engineering", "src/feature_engineering.py"),
        ("Success Scoring", "src/success_scoring.py"),
        ("Model Training", "src/model_training.py"),
        ("Evaluation", "src/evaluation.py"),
        ("Deployment", "src/deployment.py")
    ]
    
    # Track results
    results = {}
    
    # Run each stage
    for stage_name, module_path in stages:
        success = run_stage(stage_name, module_path)
        results[stage_name] = success
        
        if not success:
            logger.error(f"Pipeline stopped due to failure in stage: {stage_name}")
            break
        
        logger.info("")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    for stage_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{stage_name}: {status}")
    
    logger.info("")
    logger.info(f"Pipeline completed at: {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info("=" * 80)
    
    # Exit with appropriate code
    if all(results.values()):
        logger.info("🎉 All stages completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
