import random

import numpy as np
import torch


def set_dataset(root_read_path):
    if "0_geolife" in root_read_path:
        train_set, query_set, base_set = 3000, 1000, 13386 - 4000
    elif "porto" in root_read_path:
        train_set, query_set, base_set = 3000, 1000, 1601579 - 4000
    else:
        raise Exception("root_read_path error")
    return train_set, query_set, base_set


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_torch_device(device=None) -> torch.device:
    """Resolve a requested device string to an actually available torch.device."""
    if isinstance(device, torch.device):
        device_str = str(device)
    elif device is None:
        device_str = "auto"
    else:
        device_str = str(device).strip() or "auto"

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            if ":" not in device_str:
                return torch.device("cuda:0")
            try:
                index = int(device_str.split(":", 1)[1])
            except ValueError:
                return torch.device("cuda:0")
            if 0 <= index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}")
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if device_str == "cpu":
        return torch.device("cpu")

    return torch.device(device_str)

if __name__ == '__main__':
    pass
