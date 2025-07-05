"""
Pipeline module for the product recommendation and image detection system.

This module provides a modular pipeline for:
- Data ingestion and preprocessing
- Model training and evaluation
- Inference and prediction
- Feature engineering and analysis

Usage:
    from src.pipeline import Pipeline, PipelineConfig
    config = PipelineConfig(batch_size=32, num_epochs=50)
    pipeline = Pipeline(config)
    results = pipeline.run()
"""

from .data_ingestion import load_csv, build_image_label_df
from .preprocessing import preprocess_data
from .feature_engineering import generate_embeddings
from .model_training import train_model, run_training_loop
from .evaluation import evaluate_model
from .inference import predict
from .datasets import ProductDataset
from .pipeline import Pipeline, PipelineConfig

__all__ = [
    'load_csv',
    'build_image_label_df', 
    'preprocess_data',
    'generate_embeddings',
    'train_model',
    'run_training_loop',
    'evaluate_model',
    'predict',
    'ProductDataset',
    'Pipeline',
    'PipelineConfig'
]
