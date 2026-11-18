"""Small shared helpers: device selection and seeding."""
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed: int) -> None:
    """Seed torch exactly as the notebook does (CPU, plus all GPUs when present)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_environment(device: torch.device) -> None:
    import numpy as np
    import pandas as pd
    import sklearn
    import torch_geometric

    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"GPU name          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"PyTorch Geometric : {torch_geometric.__version__}")
    print(f"NumPy / pandas    : {np.__version__} / {pd.__version__}")
    print(f"scikit-learn      : {sklearn.__version__}")
    print(f"Device selected   : {device}")
