import os

import torch


def checkpoint_path(save_name: str, root_dir: str = "models") -> str:
    return os.path.join(root_dir, f"{save_name}.pt")


def save_model_checkpoint(model, save_name: str, metadata=None, root_dir: str = "models") -> str:
    os.makedirs(root_dir, exist_ok=True)
    path = checkpoint_path(save_name, root_dir=root_dir)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_model_checkpoint(path: str, map_location=None):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("metadata", {})
    return payload, {}
