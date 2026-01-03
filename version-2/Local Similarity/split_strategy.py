import numpy as np

def block_split(image_shape, block_size=(4, 4), train_ratio=0.5):
    """
    Implements a checkerboard/block-based split.
    """
    h, w = image_shape[:2]
    rows, cols = h // block_size[0], w // block_size[1]
    
    # Create a grid of block indices
    grid = np.indices((rows, cols))
    # Simple checkerboard: (row + col) % 2
    checkerboard = (grid[0] + grid[1]) % 2
    
    # Expand checkerboard back to pixel-wise mask
    mask = np.repeat(np.repeat(checkerboard, block_size[0], axis=0), 
                     block_size[1], axis=1)
    
    # Pad mask if image size isn't perfectly divisible
    pad_h, pad_w = h - mask.shape[0], w - mask.shape[1]
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='edge')
    
    return mask  # 0 for Train, 1 for Test

def polygon_aware_split(gt_labels, train_ratio=0.7):
    """
    Ensures entire connected components (polygons) are assigned 
    to either train or test exclusively.
    """
    from scipy.ndimage import label
    
    # Identify unique polygons in the ground truth
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(gt_labels > 0, structure=structure)
    
    unique_polygons = np.arange(1, num_features + 1)
    np.random.shuffle(unique_polygons)
    
    split_idx = int(len(unique_polygons) * train_ratio)
    train_polygons = unique_polygons[:split_idx]
    
    train_mask = np.isin(labeled_array, train_polygons)
    test_mask = (gt_labels > 0) & (~train_mask)
    
    return train_mask, test_mask