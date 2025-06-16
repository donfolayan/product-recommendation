import os
import sys
from pathlib import Path
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import traceback
import random

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.cnn_model import CNNModel
from src.utils.logging_utils import setup_logger

# Set up logging
def setup_logging(log_dir):
    return setup_logger(__name__, log_dir)

class ProductDataset(Dataset):
    """Dataset for product images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return a blank image if there's an error
            return torch.zeros(3, 224, 224), self.labels[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_dir, logger):
    """Train the model"""
    try:
        # Create model directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize best validation accuracy
        best_val_acc = 0.0
        start_epoch = 0
        
        # Try to load the last checkpoint
        checkpoint_files = list(model_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoint_files:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint = torch.load(latest_checkpoint)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            
            logger.info(f"Resuming training from epoch {start_epoch}")
            logger.info(f"Previous best validation accuracy: {best_val_acc:.2f}%")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{train_loss/train_total:.1f}',
                    'acc': f'{100.*train_correct/train_total:.1f}'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Update statistics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f'{val_loss/val_total:.1f}',
                        'acc': f'{100.*val_correct/val_total:.1f}'
                    })
            
            # Calculate epoch statistics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Log epoch statistics
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_dir / 'best_model.pth')
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, model_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main(project_root_str, batch_size=32, num_epochs=50, learning_rate=0.001):
    """Main function to train the CNN model"""
    try:
        # Convert project root to Path object
        project_root = Path(project_root_str)
        
        # Set up logging
        log_dir = project_root / 'logs'
        logger = setup_logging(log_dir)
        
        # Load and preprocess data
        data_file = project_root / 'src' / 'data' / 'final_cnn_training_data.csv'
        df = pd.read_csv(data_file)
        
        # Create a mapping of stock codes to labels from the image paths
        stock_code_to_label = {}
        for _, row in df.iterrows():
            # Extract stock code from image path (e.g., "22384" from "static/images/22384/22384_1.jpg")
            stock_code = Path(row['image_path']).parent.name
            stock_code_to_label[stock_code] = row['label']
        
        # Find all images in the static/images directory
        images_dir = project_root / 'static' / 'images'
        existing_images = []
        
        # Process each stock code directory
        for stock_code_dir in images_dir.iterdir():
            if not stock_code_dir.is_dir():
                continue
                
            stock_code = stock_code_dir.name
            if stock_code not in stock_code_to_label:
                logger.warning(f"Skipping directory {stock_code} - no label found")
                continue
                
            label = stock_code_to_label[stock_code]
            
            # Add all images in this directory
            for img_path in stock_code_dir.glob('*.jpg'):
                existing_images.append({
                    'image_path': str(img_path),
                    'label': label
                })
        
        df = pd.DataFrame(existing_images)
        logger.info(f"Loaded {len(df)} images for training")
        
        # Check class distribution
        class_counts = df['label'].value_counts()
        logger.info("Class distribution:")
        for label, count in class_counts.items():
            logger.info(f"Class {label}: {count} images")
        
        # Remove classes with too few samples
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index
        df = df[df['label'].isin(valid_classes)]
        
        if len(df) == 0:
            raise ValueError("No classes have enough samples for training")
        
        logger.info(f"After filtering, using {len(df)} images from {len(valid_classes)} classes")
        
        # Split data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Create datasets
        train_dataset = ProductDataset(train_df['image_path'].values, train_df['label'].values)
        val_dataset = ProductDataset(val_df['image_path'].values, val_df['label'].values, transform=None)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = len(valid_classes)
        model = CNNModel(num_classes).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create model directory
        model_dir = project_root / 'models'
        
        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_dir, logger)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN model from scratch using scraped product images."
    )
    parser.add_argument(
        "project_root",
        type=str,
        help="The absolute path to the project root directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for training"
    )
    
    args = parser.parse_args()
    main(args.project_root, args.batch_size, args.num_epochs, args.learning_rate) 