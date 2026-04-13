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

    print(f"💾 Checkpoint saved to {path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str,
):
    """
    Loads a training checkpoint.

    Returns:
        start_epoch (int)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"✅ Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return start_epoch