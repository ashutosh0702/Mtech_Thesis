from version_1.dataset import HSIDataset, get_spatially_disjoint_masks
from torch.utils.data import DataLoader

# 1. Define Paths (Example for Indian Pines)
img_path = 'data/raw/Indian_pines_corrected.mat'
gt_path = 'data/raw/Indian_pines_gt.mat'

# 2. Generate the Spatially Disjoint Masks
# Using Block-Based splitting with a 3-pixel buffer to ensure no leakage
train_m, test_m = get_spatially_disjoint_masks(
    gt_labels=sio.loadmat(gt_path)['indian_pines_gt'], 
    strategy="block", 
    block_size=16, 
    buffer_size=3
)

# 3. Create Datasets
train_ds = HSIDataset(img_path, gt_path, patch_size=7, mask=train_m)
test_ds = HSIDataset(img_path, gt_path, patch_size=7, mask=test_m)

# 4. Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)