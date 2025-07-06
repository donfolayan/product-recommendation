"""
Example script demonstrating the modular pipeline usage.

This script shows how to:
1. Create pipeline configurations
2. Initialize and run the pipeline
3. Handle different scenarios
"""

from pathlib import Path
from src.pipeline import Pipeline, PipelineConfig
from src.utils.logging_utils import setup_logger

def main():
    """Demonstrate pipeline usage with different configurations."""
    
    # Setup logging
    logger = setup_logger('example_pipeline', log_dir=Path('logs'))
    logger.info("Starting pipeline example")
    
    # Get project root
    project_root = Path(__file__).parent
    logger.info(f"Project root: {project_root}")
    
    # Example 1: Basic configuration
    logger.info("=== Example 1: Basic Configuration ===")
    basic_config = PipelineConfig(
        batch_size=32,
        num_epochs=5,
        learning_rate=0.001,
        min_samples_per_class=10
    )
    
    try:
        pipeline = Pipeline(basic_config, project_root=str(project_root))
        logger.info("Basic pipeline initialized successfully")
        
        # Load data
        data = pipeline.load_data()
        logger.info(f"Loaded {len(data)} samples")
        
        # Prepare data loaders
        train_loader, val_loader = pipeline.prepare_data_loaders()
        logger.info(f"Created train loader with {len(train_loader)} batches")
        logger.info(f"Created validation loader with {len(val_loader)} batches")
        
        # Create model
        model = pipeline.create_model()
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        logger.error(f"Basic pipeline failed: {e}")
    
    # Example 2: Custom configuration for small dataset
    logger.info("\n=== Example 2: Small Dataset Configuration ===")
    small_config = PipelineConfig(
        batch_size=8,
        num_epochs=2,
        learning_rate=0.01,
        min_samples_per_class=2,
        num_workers=0,  # Avoid multiprocessing issues
        test_size=0.3
    )
    
    try:
        pipeline = Pipeline(small_config, project_root=str(project_root))
        logger.info("Small dataset pipeline initialized successfully")
        
        # Load data
        data = pipeline.load_data()
        logger.info(f"Loaded {len(data)} samples")
        
        # Prepare data loaders
        train_loader, val_loader = pipeline.prepare_data_loaders()
        logger.info(f"Created train loader with {len(train_loader)} batches")
        logger.info(f"Created validation loader with {len(val_loader)} batches")
        
        # Create model
        model = pipeline.create_model()
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        logger.error(f"Small dataset pipeline failed: {e}")
    
    # Example 3: Training pipeline
    logger.info("\n=== Example 3: Training Pipeline ===")
    training_config = PipelineConfig(
        batch_size=16,
        num_epochs=3,
        learning_rate=0.001,
        min_samples_per_class=5,
        num_workers=0,
        test_size=0.2
    )
    
    try:
        pipeline = Pipeline(training_config, project_root=str(project_root))
        logger.info("Training pipeline initialized successfully")
        
        # Load data
        data = pipeline.load_data()
        logger.info(f"Loaded {len(data)} samples")
        
        # Prepare data loaders
        train_loader, val_loader = pipeline.prepare_data_loaders()
        logger.info(f"Created train loader with {len(train_loader)} batches")
        logger.info(f"Created validation loader with {len(val_loader)} batches")
        
        # Create model
        model = pipeline.create_model()
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model (commented out to avoid long execution)
        # logger.info("Starting training...")
        # results = pipeline.train()
        # logger.info(f"Training completed. Best accuracy: {results.get('best_accuracy', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
    
    logger.info("\n=== Pipeline Examples Completed ===")
    logger.info("Check the logs directory for detailed execution logs")

if __name__ == "__main__":
    main() 