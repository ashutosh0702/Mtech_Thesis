# import torch
# import torchgeo
# import torch_geometric
# import mlflow
# import mlflow.pytorch

# # 1. Initialize MLflow Experiment
# # This creates a local 'mlruns' directory if it doesn't exist.
# mlflow.set_tracking_uri("http://mlflow-ui:5000")
# mlflow.set_experiment("geospatial-test")

# # Enable automatic logging for PyTorch (optional but recommended)
# # This captures parameters and metrics without manual log statements.
# mlflow.pytorch.autolog()

# with mlflow.start_run(run_name="stack-check"):
#     # 2. Check TorchGeo Metadata
#     print(f"TorchGeo Version: {torchgeo.__version__}")
#     mlflow.log_param("torchgeo_version", torchgeo.__version__)

#     # 3. Check PyG (create a simple graph)
#     from torch_geometric.data import Data
#     edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
#     x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#     data = Data(x=x, edge_index=edge_index)
#     print(f"PyG Data Object: {data}")
    
#     # Log PyG metadata as a parameter
#     mlflow.log_param("pyg_version", torch_geometric.__version__)

#     # 4. Log hardware acceleration status
# # Check for NVIDIA (CUDA) and Apple Silicon (MPS)
#     cuda_available = torch.cuda.is_available()
#     mps_available = torch.backends.mps.is_available()

#     # Determine the primary device being used
#     if cuda_available:
#         device_type = "cuda"
#     elif mps_available:
#         device_type = "mps"
#     else:
#         device_type = "cpu"

#     # Log individual flags and the active device name to MLflow
#     mlflow.log_param("active_device", device_type)
#     mlflow.log_metric("cuda_available", 1 if cuda_available else 0)
#     mlflow.log_metric("mps_available", 1 if mps_available else 0)

#     # Log a consolidated "accelerator_ready" metric for easy filtering
#     mlflow.log_metric("accelerator_ready", 1 if (cuda_available or mps_available) else 0)

#     print(f"Hardware Check: CUDA={cuda_available}, MPS={mps_available}. Using: {device_type}")

# # Note: Unlike WandB, you don't need 'run.finish()' if using a context manager.
# print("View your results by running 'mlflow ui' in your terminal and visiting http://localhost:5000")