import torch
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from .model_training import run_training_loop
from .evaluation import evaluate_model
from .inference import predict
from .datasets import ProductDataset
from src.models.cnn_model import CNNModel
from src.utils.logging_utils import setup_logger


class PipelineConfig:
    """Configuration class for pipeline parameters."""
    
    def __init__(self, **kwargs):
        # Data parameters
        self.data_file = kwargs.get('data_file', 'src/data/dataset/final_cnn_training_data.csv')
        self.images_dir = kwargs.get('images_dir', 'static/images')
        self.min_samples_per_class = kwargs.get('min_samples_per_class', 2)
        
        # Model parameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.test_size = kwargs.get('test_size', 0.2)
        self.random_state = kwargs.get('random_state', 42)
        
        # Training parameters
        self.device = kwargs.get('device', 'auto')  # 'auto', 'cpu', 'cuda'
        self.num_workers = kwargs.get('num_workers', 4)
        self.save_checkpoints = kwargs.get('save_checkpoints', True)
        
        # Output parameters
        self.model_dir = kwargs.get('model_dir', 'models')
        self.log_dir = kwargs.get('log_dir', 'logs/pipeline')
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.min_samples_per_class <= 0:
            raise ValueError("min_samples_per_class must be positive")
        
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        
        if self.device not in ['auto', 'cpu', 'cuda']:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        else:
            return torch.device(self.device)


class Pipeline:
    """
    Main pipeline class for orchestrating the complete ML workflow.
    
    This class provides a modular interface for:
    - Data loading and preprocessing
    - Model training and evaluation
    - Inference and prediction
    - Pipeline state management
    """
    
    def __init__(self, config: PipelineConfig, project_root: str):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            project_root: Path to project root directory
        """
        if not project_root or project_root.strip() == "":
            raise ValueError("project_root cannot be empty")
        
        self.config = config
        self.project_root = Path(project_root)
        self.device = config.get_device()
        
        # Create directories
        self.model_dir = self.project_root / config.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_dir = self.project_root / config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('pipeline', log_dir=log_dir)
        
        self.data = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
        self.logger.info(f"Pipeline initialized with device: {self.device}")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Model directory: {self.model_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare training data.
        
        Returns:
            DataFrame with image paths and labels
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data file is empty or has invalid format
        """
        self.logger.info("Loading training data...")
        
        # Load CSV data
        data_file = self.project_root / self.config.data_file
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        try:
            self.data = pd.read_csv(data_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate CSV format
        required_columns = ['image_path', 'label']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        if len(self.data) == 0:
            raise ValueError("No data found in CSV file")
        
        # Validate data types and convert labels to integers
        try:
            self.data['label'] = self.data['label'].astype(int)
        except (ValueError, TypeError):
            raise ValueError("Labels must be convertible to integers")
        
        # Filter classes with enough samples
        class_counts = self.data['label'].value_counts()
        valid_classes = class_counts[class_counts >= self.config.min_samples_per_class].index
        
        if len(valid_classes) == 0:
            raise ValueError("No classes have enough samples for training")
        
        self.data = self.data[self.data['label'].isin(valid_classes)]
        
        self.logger.info(f"After filtering, using {len(self.data)} images from {len(valid_classes)} classes")
        
        return self.data
    
    def prepare_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Prepare train and validation data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.logger.info("Preparing data loaders...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            self.data, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            stratify=self.data['label']
        )
        
        # Create datasets
        train_dataset = ProductDataset(
            train_df['image_path'].values, 
            train_df['label'].values, 
            project_root=self.project_root
        )
        val_dataset = ProductDataset(
            val_df['image_path'].values, 
            val_df['label'].values, 
            project_root=self.project_root
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers
        )
        
        self.logger.info(f"Created train loader with {len(train_dataset)} samples")
        self.logger.info(f"Created validation loader with {len(val_dataset)} samples")
        
        return self.train_loader, self.val_loader
    
    def create_model(self) -> CNNModel:
        """
        Create and initialize the model.
        
        Returns:
            Initialized CNN model
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        num_classes = len(self.data['label'].unique())
        self.model = CNNModel(num_classes).to(self.device)
        
        self.logger.info(f"Created model with {num_classes} output classes")
        return self.model
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model using the pipeline configuration.
        
        Returns:
            Dictionary with training results
        """
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders not prepared. Call prepare_data_loaders() first.")
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.logger.info("Starting model training...")
        
        # Setup training components
        import torch.nn as nn
        import torch.optim as optim
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Run training
        best_val_acc, best_epoch, history = run_training_loop(
            self.model, 
            self.train_loader, 
            self.val_loader, 
            criterion, 
            optimizer, 
            self.config.num_epochs, 
            self.device, 
            self.model_dir, 
            self.logger
        )
        
        # Save mapping
        stock_code_to_label = {}
        for _, row in self.data.iterrows():
            stock_code = Path(row['image_path']).parent.name
            stock_code_to_label[stock_code] = row['label']
        
        mapping_path = self.model_dir / 'stockcode_to_index.json'
        with open(mapping_path, 'w') as f:
            json.dump(stock_code_to_label, f)
        
        self.logger.info(f"Saved StockCode-to-index mapping to {mapping_path}")
        
        return {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "num_epochs": self.config.num_epochs,
            "history": history
        }
    
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            test_data: Test data loader or dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.logger.info("Evaluating model...")
        
        accuracy, metrics = evaluate_model(self.model, test_data, None)
        
        self.logger.info(f"Model accuracy: {accuracy:.4f}")
        
        return metrics
    
    def predict(self, input_data: Any) -> Any:
        """
        Make predictions using the trained model.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return predict(self.model, input_data)
    
    def save_model(self, filename: str = "best_model.pth") -> None:
        """
        Save the trained model.
        
        Args:
            filename: Name of the model file
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path = self.model_dir / filename
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str = "best_model.pth") -> None:
        """
        Load a trained model.
        
        Args:
            filename: Name of the model file
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        model_path = self.model_dir / filename
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.logger.info(f"Model loaded from {model_path}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to training.
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting complete pipeline...")
        
        # Load data
        self.load_data()
        
        # Prepare data loaders
        self.prepare_data_loaders()
        
        # Create model
        self.create_model()
        
        # Train model
        results = self.train()
        
        # Save model
        self.save_model()
        
        self.logger.info("Pipeline completed successfully!")
        
        return results 