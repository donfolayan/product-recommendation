#!/usr/bin/env python3
"""
Example script demonstrating the modular pipeline usage.
"""

import sys
from pathlib import Path
import argparse
from src.utils.project_utils import setup_project_path

# Add project root to path
setup_project_path()
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pipeline import Pipeline, PipelineConfig


def main():
    """Run the modular pipeline with command line arguments."""
    parser = argparse.ArgumentParser(description="Run the modular ML pipeline")
    parser.add_argument("project_root", type=str, help="Path to project root")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    config = PipelineConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Create and run pipeline
    pipeline = Pipeline(config, project_root=args.project_root)
    
    try:
        results = pipeline.run()
        print("Pipeline completed successfully!")
        print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
        print(f"Best epoch: {results['best_epoch']}")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 