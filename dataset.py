import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import color


class ColorizationDataset(Dataset):
    
    def __init__(self, image_paths, img_size=256):
        self.image_paths = image_paths
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
            img = np.array(img)
        except Exception as e:
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
        
        lab = color.rgb2lab(img).astype(np.float32)
        L = lab[:, :, 0:1] / 50.0 - 1.0
        ab = lab[:, :, 1:] / 128.0
        L = torch.from_numpy(L.transpose(2, 0, 1))
        ab = torch.from_numpy(ab.transpose(2, 0, 1))
        return L, ab


def get_data_loaders(image_dir, batch_size=16, img_size=256, 
                     train_split=0.8, num_workers=4, max_images=None):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(extensions):
                image_paths.append(os.path.join(root, f))
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    np.random.seed(42)
    np.random.shuffle(image_paths)
    
    split_idx = int(train_split * len(image_paths))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    train_dataset = ColorizationDataset(train_paths, img_size=img_size)
    val_dataset = ColorizationDataset(val_paths, img_size=img_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset
