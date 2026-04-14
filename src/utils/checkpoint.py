import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
):
    """
    Saves a training checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)

    print(f" Checkpoint saved to {path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str,
):
    """
    Load model (and optionally optimizer) state from checkpoint.

    Returns:
        start_epoch (int) if optimizer is provided, else None
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Always load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch = None

    # Only load optimizer if it exists (training case)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1

    print(f" Loaded checkpoint from {checkpoint_path}")
    return start_epoch
