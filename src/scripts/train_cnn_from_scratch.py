from src.utils.project_utils import setup_project_path
setup_project_path()
from pathlib import Path
from src.utils.logging_utils import setup_logger
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import traceback
from typing import Optional, Any, Tuple, Dict



from src.models.cnn_model import CNNModel
from src.pipeline.datasets import ProductDataset

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    num_epochs: int, 
    device: torch.device, 
    model_dir: Path, 
    logger: Any
) -> Tuple[float, Optional[int]]:
    """Train the model"""
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        best_val_acc = 0.0
        best_epoch = None
        start_epoch = 0
        
        for epoch in range(start_epoch, num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch in train_pbar:
                inputs, labels = batch
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                train_pbar.set_postfix({
                    'loss': f'{train_loss/train_total:.1f}',
                    'acc': f'{100.*train_correct/train_total:.1f}'
                })
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for batch in val_pbar:
                    inputs, labels = batch
                    if isinstance(inputs, tuple):
                        inputs = inputs[0]
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_pbar.set_postfix({
                        'loss': f'{val_loss/val_total:.1f}',
                        'acc': f'{100.*val_correct/val_total:.1f}'
                    })
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_dir / 'best_model.pth')
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}% (epoch {best_epoch})')
        return best_val_acc, best_epoch
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main(project_root_str: str, batch_size: int = 32, num_epochs: int = 50, learning_rate: float = 0.001) -> Dict[str, Any]:
    """Main function to train the CNN model"""
    try:
        project_root = Path(project_root_str)
        log_dir = project_root / 'logs/training'
        logger = setup_logger(__name__, log_dir)
        

        data_file = project_root / 'src' / 'data' / 'dataset' / 'final_cnn_training_data.csv'
        if not data_file.exists():
            raise FileNotFoundError(f"Training data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} images from CSV")
        

        missing_files = []
        for _, row in df.iterrows():
            img_path = project_root / row['image_path']
            if not img_path.exists():
                missing_files.append(row['image_path'])
        
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} image files. First few: {missing_files[:5]}")
            # Remove missing files from dataset
            df = df[~df['image_path'].isin(missing_files)]
            logger.info(f"After removing missing files: {len(df)} images remaining")
        
        if len(df) == 0:
            raise ValueError("No valid images found for training")
        

        class_counts = df['label'].value_counts()
        logger.info("Class distribution:")
        for label, count in class_counts.items():
            logger.info(f"Class {label}: {count} images")
        

        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index
        df = df[df['label'].isin(valid_classes)]
        
        if len(df) == 0:
            raise ValueError("No classes have enough samples for training")
        
        logger.info(f"After filtering, using {len(df)} images from {len(valid_classes)} classes")
        

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        train_df['label'] = train_df['label'].astype(int)
        val_df['label'] = val_df['label'].astype(int)
        
        train_dataset = ProductDataset(train_df['image_path'].values, train_df['label'].values, project_root=project_root)
        val_dataset = ProductDataset(val_df['image_path'].values, val_df['label'].values, transform=None, project_root=project_root)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = len(valid_classes)
        model = CNNModel(num_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model_dir = project_root / 'models'
        
        best_val_acc, best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_dir, logger)
        
        logger.info("Training completed successfully")
        
        label_to_index = {label: idx for idx, label in enumerate(sorted(valid_classes))}
        mapping_path = model_dir / 'label_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(label_to_index, f, indent=2)
        logger.info(f"Saved label mapping to {mapping_path}")
        
        return {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "num_epochs": num_epochs,
            "num_classes": num_classes,
            "total_images": len(df)
        }
        
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