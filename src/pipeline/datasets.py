import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import List, Union, Optional, Any
from pathlib import Path

class ProductDataset(Dataset):
    """Dataset for product images"""
    def __init__(self, image_paths: List[str], labels: List[Union[int, str]], transform: Optional[Any] = None, project_root: Optional[Path] = None) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(f"Number of image paths ({len(image_paths)}) must match number of labels ({len(labels)})")
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.project_root = project_root
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            # Handle both absolute and relative paths
            img_path = self.image_paths[idx]
            if self.project_root and not Path(img_path).is_absolute():
                # Convert Windows backslashes to forward slashes and resolve relative to project root
                img_path = str(self.project_root / img_path.replace('\\', '/'))
            else:
                img_path = str(Path(img_path))
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            if not torch.is_tensor(label):
                label = torch.tensor(int(label))
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            return torch.zeros(3, 224, 224), torch.tensor(0) 