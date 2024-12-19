from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import os
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlbumentationsDataset(Dataset):
    def __init__(self, rootDir, transform=None, class_map=None):
        # Initialize dataset using ImageFolder
        self.dataset = datasets.ImageFolder(root=rootDir)
        self.transform = transform
        # Define custom class mapping
        self.class_map = class_map if class_map else {i: i for i in range(len(self.dataset.classes))}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        # Map the label using the custom class mapping
        mapped_label = self.class_map[label]
        return img, mapped_label
    
    @property
    def classes(self):
        return [self.class_map[i] for i in range(len(self.dataset.classes))]

    # MVD with TCR -> 0
    # MVD without TCR -> 1
    # Normal Heart -> 2

def get_dataloader(rootDir, transforms, batchSize, class_map=None, shuffle=True):
    # Pass the class_map to AlbumentationsDataset
    ds = AlbumentationsDataset(rootDir, transform=transforms, class_map=class_map)
    loader = DataLoader(ds, batch_size=batchSize, shuffle=shuffle, num_workers=0, pin_memory=True if DEVICE == "cuda" else False)
    return ds, loader
