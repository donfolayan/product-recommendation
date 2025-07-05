from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

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
            label = self.labels[idx]
            if not torch.is_tensor(label):
                label = torch.tensor(int(label))
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return a blank image if there's an error
            return torch.zeros(3, 224, 224), torch.tensor(0) 