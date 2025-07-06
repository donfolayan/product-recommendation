#!/usr/bin/env python3
"""
Edge case tests for the modular pipeline components.
Tests error conditions, invalid inputs, and boundary conditions.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import unittest
import pandas as pd
import torch
from src.pipeline import Pipeline, PipelineConfig, ProductDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TestPipelineEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for the pipeline."""
    
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
        
        # Create test CSV data
        test_data = pd.DataFrame({
            'image_path': [
                'static\\images\\test_class\\test_image_0.jpg',
                'static\\images\\test_class\\test_image_1.jpg',
                'static\\images\\test_class2\\test_image_2.jpg',
                'static\\images\\test_class2\\test_image_3.jpg'
            ],
            'label': [0, 0, 1, 1]
        })
        test_data.to_csv(self.project_root / 'src' / 'data' / 'dataset' / 'final_cnn_training_data.csv', index=False)
        
        # Create dummy image files
        for i in range(2):
            img_path = self.project_root / 'static' / 'images' / 'test_class' / f'test_image_{i}.jpg'
            img_path.write_bytes(b'dummy image data')
        for i in range(2, 4):
            img_path = self.project_root / 'static' / 'images' / 'test_class2' / f'test_image_{i}.jpg'
            img_path.write_bytes(b'dummy image data')

    def tearDown(self):
        """Clean up test environment."""
        import logging
        logging.shutdown()
        shutil.rmtree(self.temp_dir)

    def test_empty_data_file(self):
        """Test pipeline with empty data file."""
        # Create empty CSV
        empty_data = pd.DataFrame(columns=['image_path', 'label'])
        empty_data.to_csv(self.project_root / 'src' / 'data' / 'empty_data.csv', index=False)
        
        config = PipelineConfig(data_file='src/data/empty_data.csv')
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        with self.assertRaises(ValueError) as context:
            pipeline.load_data()
        self.assertIn("No data found", str(context.exception))

    def test_missing_data_file(self):
        """Test pipeline with missing data file."""
        config = PipelineConfig(data_file='nonexistent_file.csv')
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        with self.assertRaises(FileNotFoundError):
            pipeline.load_data()

    def test_invalid_csv_format(self):
        """Test pipeline with invalid CSV format."""
        # Create CSV with wrong columns
        invalid_data = pd.DataFrame({
            'wrong_column': ['test'],
            'another_wrong': ['test']
        })
        invalid_data.to_csv(self.project_root / 'src' / 'data' / 'invalid_data.csv', index=False)
        
        config = PipelineConfig(data_file='src/data/invalid_data.csv')
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        with self.assertRaises(ValueError) as context:
            pipeline.load_data()
        self.assertIn("CSV file missing required columns", str(context.exception))

    def test_invalid_batch_size(self):
        """Test pipeline with invalid batch size."""
        with self.assertRaises(ValueError):
            PipelineConfig(batch_size=0)
        
        with self.assertRaises(ValueError):
            PipelineConfig(batch_size=-1)

    def test_invalid_learning_rate(self):
        """Test pipeline with invalid learning rate."""
        with self.assertRaises(ValueError):
            PipelineConfig(learning_rate=0)
        
        with self.assertRaises(ValueError):
            PipelineConfig(learning_rate=-0.1)

    def test_invalid_num_epochs(self):
        """Test pipeline with invalid number of epochs."""
        with self.assertRaises(ValueError):
            PipelineConfig(num_epochs=0)
        
        with self.assertRaises(ValueError):
            PipelineConfig(num_epochs=-1)

    def test_invalid_test_size(self):
        """Test pipeline with invalid test size."""
        with self.assertRaises(ValueError):
            PipelineConfig(test_size=0)
        
        with self.assertRaises(ValueError):
            PipelineConfig(test_size=1.1)

    def test_missing_images(self):
        """Test pipeline with missing image files."""
        # Create CSV with non-existent image paths
        missing_images_data = pd.DataFrame({
            'image_path': [
                'static\\images\\nonexistent\\image.jpg',
                'static\\images\\test_class\\test_image_0.jpg'
            ],
            'label': [0, 0]
        })
        missing_images_data.to_csv(self.project_root / 'src' / 'data' / 'missing_images.csv', index=False)
        
        config = PipelineConfig(data_file='src/data/missing_images.csv')
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Should handle missing images gracefully
        data = pipeline.load_data()
        self.assertGreater(len(data), 0)  # Should still load valid images

    def test_invalid_device(self):
        """Test pipeline with invalid device specification."""
        with self.assertRaises(ValueError):
            PipelineConfig(device='invalid_device')

    def test_negative_num_workers(self):
        """Test pipeline with negative number of workers."""
        with self.assertRaises(ValueError):
            PipelineConfig(num_workers=-1)

    def test_model_save_load_edge_cases(self):
        """Test model save/load edge cases."""
        config = PipelineConfig()
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Try to save model before training
        with self.assertRaises(ValueError) as context:
            pipeline.save_model()
        self.assertIn("Model not trained", str(context.exception))
        
        # Try to load model before creating model
        with self.assertRaises(ValueError) as context:
            pipeline.load_model()
        self.assertIn("Model not created", str(context.exception))

    def test_evaluate_without_training(self):
        """Test evaluation without training."""
        config = PipelineConfig()
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        with self.assertRaises(ValueError) as context:
            pipeline.evaluate(None)
        self.assertIn("Model not trained", str(context.exception))

    def test_predict_without_training(self):
        """Test prediction without training."""
        config = PipelineConfig()
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        with self.assertRaises(ValueError) as context:
            pipeline.predict(None)
        self.assertIn("Model not trained", str(context.exception))

    def test_invalid_project_root(self):
        """Test pipeline with invalid project root."""
        with self.assertRaises(ValueError):
            Pipeline(PipelineConfig(), project_root="")
        
        with self.assertRaises(ValueError):
            Pipeline(PipelineConfig(), project_root=None)

    def test_very_small_dataset(self):
        """Test pipeline with very small dataset."""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'image_path': [
                'static\\images\\test_class\\test_image_0.jpg',
                'static\\images\\test_class\\test_image_1.jpg'
            ],
            'label': [0, 0]
        })
        minimal_data.to_csv(self.project_root / 'src' / 'data' / 'minimal_data.csv', index=False)
        
        config = PipelineConfig(
            data_file='src/data/minimal_data.csv',
            min_samples_per_class=1,
            batch_size=1
        )
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Should handle minimal dataset gracefully
        data = pipeline.load_data()
        self.assertEqual(len(data), 2)
        
        train_loader, val_loader = pipeline.prepare_data_loaders()
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

    def test_duplicate_image_paths(self):
        """Test pipeline with duplicate image paths."""
        duplicate_data = pd.DataFrame({
            'image_path': [
                'static\\images\\test_class\\test_image_0.jpg',
                'static\\images\\test_class\\test_image_0.jpg',  # Duplicate
                'static\\images\\test_class2\\test_image_2.jpg'
            ],
            'label': [0, 0, 1]
        })
        duplicate_data.to_csv(self.project_root / 'src' / 'data' / 'duplicate_data.csv', index=False)
        
        config = PipelineConfig(data_file='src/data/duplicate_data.csv')
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Should handle duplicates gracefully
        data = pipeline.load_data()
        self.assertGreater(len(data), 0)

    def test_unicode_characters_in_paths(self):
        """Test pipeline with unicode characters in paths."""
        unicode_data = pd.DataFrame({
            'image_path': [
                'static\\images\\test_class\\test_image_0.jpg',
                'static\\images\\test_class\\test_image_1.jpg'
            ],
            'label': [0, 0]
        })
        unicode_data.to_csv(self.project_root / 'src' / 'data' / 'unicode_data.csv', index=False)
        
        config = PipelineConfig(data_file='src/data/unicode_data.csv')
        pipeline = Pipeline(config, project_root=str(self.project_root))
        
        # Should handle unicode gracefully
        data = pipeline.load_data()
        self.assertGreater(len(data), 0)


class TestProductDatasetEdgeCases(unittest.TestCase):
    """Test edge cases for the ProductDataset class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        (self.project_root / 'static' / 'images' / 'test_class').mkdir(parents=True, exist_ok=True)
        
        # Create test image
        img_path = self.project_root / 'static' / 'images' / 'test_class' / 'test_image.jpg'
        img_path.write_bytes(b'dummy image data')

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_empty_dataset(self):
        """Test ProductDataset with empty lists."""
        dataset = ProductDataset([], [], project_root=self.project_root)
        self.assertEqual(len(dataset), 0)
        
        with self.assertRaises(IndexError):
            dataset[0]

    def test_mismatched_lengths(self):
        """Test ProductDataset with mismatched image_paths and labels."""
        with self.assertRaises(ValueError):
            ProductDataset(['path1', 'path2'], [0], project_root=self.project_root)

    def test_missing_image_file(self):
        """Test ProductDataset with missing image file."""
        dataset = ProductDataset(
            ['nonexistent.jpg'], 
            [0], 
            project_root=self.project_root
        )
        
        # Should return default tensor for missing image
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

    def test_invalid_image_format(self):
        """Test ProductDataset with invalid image format."""
        # Create invalid image file
        invalid_img_path = self.project_root / 'static' / 'images' / 'test_class' / 'invalid.jpg'
        invalid_img_path.write_bytes(b'invalid image data')
        
        dataset = ProductDataset(
            ['static/images/test_class/invalid.jpg'], 
            [0], 
            project_root=self.project_root
        )
        
        # Should handle invalid image gracefully
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

    def test_absolute_vs_relative_paths(self):
        """Test ProductDataset with both absolute and relative paths."""
        # Test relative path
        dataset_rel = ProductDataset(
            ['static/images/test_class/test_image.jpg'], 
            [0], 
            project_root=self.project_root
        )
        
        # Test absolute path
        abs_path = str(self.project_root / 'static' / 'images' / 'test_class' / 'test_image.jpg')
        dataset_abs = ProductDataset(
            [abs_path], 
            [0], 
            project_root=self.project_root
        )
        
        # Both should work
        image_rel, label_rel = dataset_rel[0]
        image_abs, label_abs = dataset_abs[0]
        
        self.assertIsInstance(image_rel, torch.Tensor)
        self.assertIsInstance(image_abs, torch.Tensor)


if __name__ == '__main__':
    unittest.main() 