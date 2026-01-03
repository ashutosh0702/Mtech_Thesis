import torch
import torchgeo
import os
from importlib.metadata import version

def run_health_check():
    print("=== RESEARCH ENVIRONMENT HEALTH CHECK ===\n")
    
    # 1. Hardware Check
    print(f"PyTorch Version: {torch.__version__}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device Recommendation: {device} (Note: Docker on Mac uses CPU)")

    # 2. Geospatial Stack Check
    print(f"\nTorchGeo Version: {torchgeo.__version__}")
    try:
        import rasterio
        print(f"GDAL/Rasterio Version: {rasterio.__gdal_version__}")
    except:
        print("✘ Rasterio/GDAL failed to load.")

    # 3. Critical Data Check (Based on your project structure)
    data_files = [
        "data/raw/Indian_Pines/Indian_pines_corrected.mat",
        "data/raw/Salinas/Salinas_gt.mat"
    ]
    print("\n--- Data Health ---")
    for f in data_files:
        status = "✔ FOUND" if os.path.exists(f) else "✘ MISSING"
        print(f"{f}: {status}")

    # 4. MLflow Tracking Check
    ml_path = "/app/mlruns"
    print(f"\nTracking Directory ({ml_path}): {'✔ READY' if os.path.exists(ml_path) or os.access('/app', os.W_OK) else '✘ ERROR'}")

if __name__ == "__main__":
    run_health_check()