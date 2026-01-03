import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from scipy.ndimage import label, binary_erosion
from pathlib import Path

class HSIDataset(Dataset):
    """
    Custom Dataset for Hyperspectral Image (HSI) processing.
    Handles loading .mat files and extracting spatial-spectral patches.
    """
    def __init__(self, image_path, gt_path, patch_size=1, mask=None):
        # Load .mat files
        self.data = sio.loadmat(image_path)
        # Handle common HSI keys (e.g., 'indian_pines_corrected' or 'salinas_corrected')
        data_key = [k for k in self.data.keys() if not k.startswith('__')][0]
        self.image = self.data[data_key].astype(np.float32)
        
        self.gt = sio.loadmat(gt_path)
        gt_key = [k for k in self.gt.keys() if not k.startswith('__')][0]
        self.labels = self.gt[gt_key].astype(np.int64)
        
        self.patch_size = patch_size
        
        # Normalize Data (Spectral-wise)
        self.image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        
        # Filter indices based on the provided spatial mask (Train/Test)
        if mask is not None:
            self.indices = np.argwhere((self.labels > 0) & (mask > 0))
        else:
            self.indices = np.argwhere(self.labels > 0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        r, c = self.indices[i]
        margin = self.patch_size // 2
        
        # Pad image to handle edges for spatial patches
        padded_image = np.pad(self.image, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
        
        # Extract patch
        patch = padded_image[r:r + self.patch_size, c:c + self.patch_size, :]
        label = self.labels[r, c] - 1  # 0-indexed for CrossEntropyLoss
        
        # Convert to Torch Tensors
        return torch.from_numpy(patch).permute(2, 0, 1), torch.tensor(label)

def get_spatially_disjoint_masks(gt_labels, strategy="block", block_size=16, buffer_size=3):
    """
    Implements advanced splitting to prevent information leakage.
    """
    h, w = gt_labels.shape
    train_mask = np.zeros_like(gt_labels)
    test_mask = np.zeros_like(gt_labels)

    if strategy == "block":
        # Create checkerboard pattern of blocks
        for r in range(0, h, block_size):
            for c in range(0, w, block_size):
                if ((r // block_size) + (c // block_size)) % 2 == 0:
                    train_mask[r:r+block_size, c:c+block_size] = 1
                else:
                    test_mask[r:r+block_size, c:c+block_size] = 1

    elif strategy == "polygon":
        # Ensure entire connected crop fields are assigned together
        labeled_array, num_features = label(gt_labels > 0)
        polygons = np.arange(1, num_features + 1)
        np.random.shuffle(polygons)
        
        split = int(len(polygons) * 0.7)
        train_polys = polygons[:split]
        
        train_mask = np.isin(labeled_array, train_polys)
        test_mask = (gt_labels > 0) & (~train_mask)

    # Apply Buffer Zones to strictly separate Receptive Fields
    if buffer_size > 0:
        train_mask = binary_erosion(train_mask, iterations=buffer_size)
        test_mask = binary_erosion(test_mask, iterations=buffer_size)

    return train_mask, test_mask