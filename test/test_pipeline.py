#!/usr/bin/env python3
"""
Tests for the modular pipeline components.
"""

import sys
import tempfile
import shutil
import pandas as pd
import torch
import logging
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import Pipeline, PipelineConfig, ProductDataset
from src.models.cnn_model import CNNModel


class TestPipelineConfig(unittest.TestCase):
    """Test the PipelineConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_epochs, 50)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.device, 'auto')
        self.assertEqual(config.min_samples_per_class, 2)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            batch_size=64,
            num_epochs=100,
            learning_rate=0.01,
            device='cpu'
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_epochs, 100)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.device, 'cpu')
    
    def test_get_device(self):
        """Test device selection logic."""
        config = PipelineConfig(device='cpu')
        device = config.get_device()
        self.assertEqual(device, torch.device('cpu'))
        
        config = PipelineConfig(device='auto')
        device = config.get_device()
        # Should return either cpu or cuda
        self.assertIn(device, [torch.device('cpu'), torch.device('cuda')])


class TestProductDataset(unittest.TestCase):
    """Test the ProductDataset class."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create test image directory structure
        test_dir = self.project_root / 'static' / 'images' / 'test_class'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy image files
        for i in range(3):
            img_path = test_dir / f'test_image_{i}.jpg'
            img_path.write_bytes(b'dummy image data')
    
    def tearDown(self):
        """Clean up test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test ProductDataset creation with project_root."""
        image_paths = ['static\\images\\test_class\\test_image_0.jpg']
        labels = [0]
        
        dataset = ProductDataset(
            image_paths, 
            labels, 
            project_root=self.project_root
        )
        
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.image_paths, image_paths)
        self.assertEqual(dataset.labels, labels)
    
    def test_dataset_getitem(self):
        """Test ProductDataset __getitem__ method."""
        image_paths = ['static\\images\\test_class\\test_image_0.jpg']
        labels = [0]
        
        dataset = ProductDataset(
            image_paths, 
            labels, 
            project_root=self.project_root
        )
        
        # Mock PIL.Image.open to avoid actual image loading
        with patch('PIL.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image
            
            # Test with transform
            with patch('torchvision.transforms.Compose') as mock_transform:
                mock_transform.return_value = lambda x: x
                dataset = ProductDataset(
                    image_paths, 
                    labels, 
                    project_root=self.project_root
                )
                
                image, label = dataset[0]
                self.assertEqual(label, 0)


class TestPipeline(unittest.TestCase):
    """Test the Pipeline class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create test directory structure
        (self.project_root / 'src' / 'data' / 'dataset').mkdir(parents=True, exist_ok=True)
        (self.project_root / 'static' / 'images' / 'test_class').mkdir(parents=True, exist_ok=True)
        (self.project_root / 'static' / 'images' / 'test_class2').mkdir(parents=True, exist_ok=True)
        (self.project_root / 'models').mkdir(parents=True, exist_ok=True)
        (self.project_root / 'logs').mkdir(parents=True, exist_ok=True)
        
        # Create test CSV data with at least 4 samples per class
        test_data = pd.DataFrame({
            'image_path': [
                'static\\images\\test_class\\test_image_0.jpg',
                'static\\images\\test_class\\test_image_1.jpg',
                'static\\images\\test_class\\test_image_2.jpg',
                'static\\images\\test_class\\test_image_3.jpg',
                'static\\images\\test_class2\\test_image_4.jpg',
                'static\\images\\test_class2\\test_image_5.jpg',
                'static\\images\\test_class2\\test_image_6.jpg',
                'static\\images\\test_class2\\test_image_7.jpg'
            ],
            'label': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        test_data.to_csv(self.project_root / 'src' / 'data' / 'dataset' / 'final_cnn_training_data.csv', index=False)
        
        # Create dummy image files
        for i in range(4):
            img_path = self.project_root / 'static' / 'images' / 'test_class' / f'test_image_{i}.jpg'
            img_path.write_bytes(b'dummy image data')
        for i in range(4, 8):
            img_path = self.project_root / 'static' / 'images' / 'test_class2' / f'test_image_{i}.jpg'
            img_path.write_bytes(b'dummy image data')
    
    def tearDown(self):
        """Clean up test environment and close log handlers."""
        # Close all handlers for the pipeline's logger and root logger
        loggers = [logging.getLogger(), getattr(self, 'pipeline_logger', None)]
        for logger in loggers:
            if logger is not None:
                handlers = getattr(logger, 'handlers', [])
                for handler in handlers[:]:
                    try:
                        handler.close()
                    except Exception:
                        pass
                    logger.removeHandler(handler)
        logging.shutdown()
        # Disable loggers to avoid file lock
        for logger in loggers:
            if logger is not None:
                logger.disabled = True
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test Pipeline initialization."""
        config = PipelineConfig(
            batch_size=16,
            num_epochs=1,
            min_samples_per_class=1
        )
        
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        self.assertEqual(pipeline.config, config)
        self.assertEqual(pipeline.project_root, self.project_root)
        self.assertIsNone(pipeline.data)
        self.assertIsNone(pipeline.model)
    
    def test_load_data(self):
        """Test data loading functionality."""
        config = PipelineConfig(
            min_samples_per_class=1
        )
        
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Mock the logger to avoid file system issues
        with patch.object(pipeline, 'logger'):
            data = pipeline.load_data()
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 8)
            self.assertIn('image_path', data.columns)
            self.assertIn('label', data.columns)
    
    def test_prepare_data_loaders(self):
        """Test data loader preparation."""
        config = PipelineConfig(
            batch_size=16,
            min_samples_per_class=1
        )
        
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Load data first
        with patch.object(pipeline, 'logger'):
            pipeline.load_data()
            
            # Mock ProductDataset to avoid image loading issues
            with patch('src.pipeline.datasets.ProductDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                mock_dataset.return_value.__len__ = lambda self: 8
                
                train_loader, val_loader = pipeline.prepare_data_loaders()
                
                self.assertIsNotNone(train_loader)
                self.assertIsNotNone(val_loader)
    
    def test_create_model(self):
        """Test model creation."""
        config = PipelineConfig(
            min_samples_per_class=1
        )
        
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Load data first
        with patch.object(pipeline, 'logger'):
            pipeline.load_data()
            
            # Patch only the 'to' method of the CNNModel instance after creation
            with patch.object(CNNModel, 'to', return_value=MagicMock(spec=CNNModel)) as mock_to:
                pipeline.create_model()
                mock_to.assert_called()
    
    @patch('src.pipeline.model_training.run_training_loop')
    def test_train(self, mock_training_loop):
        """Test training functionality."""
        config = PipelineConfig(
            batch_size=16,
            num_epochs=1,
            min_samples_per_class=1,
            num_workers=0
        )
        
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Mock all dependencies
        with patch.object(pipeline, 'logger'):
            pipeline.load_data()
            
            with patch('src.pipeline.datasets.ProductDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                mock_dataset.return_value.__len__ = lambda self: 8
                
                pipeline.prepare_data_loaders()
                pipeline.create_model = MagicMock()
                pipeline.model = MagicMock()  # Ensure model is set so train() does not raise
                pipeline.model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
                pipeline.model.side_effect = lambda x: torch.randn(x.size(0), 2, requires_grad=True)  # batch_size x num_classes
                pipeline.model.state_dict.return_value = {'layer1.weight': torch.randn(2, 2), 'layer1.bias': torch.randn(2)}

                # Mock training loop
                mock_training_loop.return_value = (0.85, 1, {'train_loss': [0.5]})
                
                results = pipeline.train()
                
                self.assertIn('best_val_acc', results)
                self.assertIn('best_epoch', results)
                self.assertIn('num_epochs', results)
                self.assertIn('history', results)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the pipeline."""
    
    def test_pipeline_imports(self):
        """Test that all pipeline components can be imported."""
        try:
            self.assertTrue(True)  # If we get here, imports work
        except ImportError as e:
            self.fail(f"Failed to import pipeline components: {e}")
    
    def test_pipeline_config_creation(self):
        """Test PipelineConfig creation with various parameters."""
        config = PipelineConfig(
            batch_size=64,
            num_epochs=25,
            learning_rate=0.01,
            device='cpu',
            min_samples_per_class=5
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_epochs, 25)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.min_samples_per_class, 5)


if __name__ == '__main__':
    unittest.main() 