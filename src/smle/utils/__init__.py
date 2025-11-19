import random
import numpy as np

try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "Torch is required by smle."
        "Install it yourself (GPU/CPU) or use 'pip install smle[torch]'."
    ) from e

def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False