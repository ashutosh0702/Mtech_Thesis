import torch
import torchgeo
import torch_geometric
import wandb

# 1. Initialize WandB
wandb.init(project="geospatial-test", name="stack-check")

# 2. Check TorchGeo (e.g., a simple dataset metadata)
print(f"TorchGeo Version: {torchgeo.__version__}")

# 3. Check PyG (e.g., create a simple graph)
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(f"PyG Data Object: {data}")

# 4. Log a test metric
wandb.log({"status": "ready", "cuda_available": torch.cuda.is_available()})
wandb.finish()