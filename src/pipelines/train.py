"""
Multimodal biometric training script.

This script trains a multimodal deep learning model using iris and fingerprint
biometric data stored in a Parquet file. Each modality is encoded separately,
fused via concatenation, and optimized end-to-end for multi-class classification.

Key features:
- Reproducible training via deterministic seeding
- Modular dataset design (Iris, Fingerprint, and MultiModal datasets)
- Pluggable encoder and fusion architectures
- TensorBoard logging for training metrics
- Profiling hooks for forward/backward timing
- Checkpointing with resume support

Typical usage:
    python train.py              # start training from scratch
    python train.py --resume     # resume from latest checkpoint

Artifacts:
- TensorBoard logs: runs/
- Checkpoints: checkpoints/latest.pt
"""

import os
import torch
from torch.utils.data import DataLoader

from src.datasets.iris import IrisDataset
from src.datasets.fingerprint import FingerprintDataset
from src.datasets.multimodal import MultiModalDataset

from src.models.encoders import SimpleEncoder
from src.models.fusion import ConcatFusion
from src.models.model import MultiModalModel

from src.utils.seed import set_seed
from src.utils.tensorboard import create_tensorboard_writer
from src.utils.profiling import log_timing
from src.utils.checkpoint import save_checkpoint, load_checkpoint


PARQUET_PATH = "data/processed/biometric.parquet"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")


def train_one_epoch(model, loader, optimizer, writer, epoch):

    """
        Run one training epoch over the dataset.

        This function performs forward and backward passes for each batch,
        updates model parameters, logs training loss to TensorBoard, and
        measures execution time for profiling purposes.

        Args:
            model (torch.nn.Module):
                The multimodal model containing modality-specific encoders
                and a fusion/classification head.
            loader (torch.utils.data.DataLoader):
                DataLoader providing batches from the MultiModalDataset.
            optimizer (torch.optim.Optimizer):
                Optimizer used to update model parameters.
            writer (torch.utils.tensorboard.SummaryWriter):
                TensorBoard writer for logging training metrics.
            epoch (int):
                Current epoch index (0-based), used for global step calculation.

        Notes:
            - Only the first 10 batches are processed per epoch to enable
            fast debugging and profiling.
            - The model return a dictionary containing a
            scalar `loss` entry.
        """

    model.train()

    for batch_idx, batch in enumerate(loader):
        with log_timing("forward_backward"):
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar("train/loss", loss.item(), global_step)

        if batch_idx >= 10:
            break


def main(resume: bool = False):

    """
        Entry point for multimodal biometric model training.

        This function sets up reproducibility, logging, datasets, model,
        optimizer, and the training loop. It optionally resumes training
        from a previously saved checkpoint.

        Args:
            resume (bool, optional):
                If True, attempts to load model and optimizer state from
                the latest checkpoint before continuing training.
                Defaults to False.

        Workflow:
            1. Set random seeds for reproducibility
            2. Initialize TensorBoard logging
            3. Load iris and fingerprint datasets from Parquet storage
            4. Construct multimodal dataset and DataLoader
            5. Initialize encoders, fusion module, and full model
            6. Resume from checkpoint if enabled
            7. Train for a fixed number of epochs
            8. Save checkpoints after each epoch

        """

    # -----------------------
    # Reproducibility
    # -----------------------
    set_seed(42)

    # -----------------------
    # TensorBoard
    # -----------------------
    writer, log_dir = create_tensorboard_writer(
        base_dir="runs",
        run_name="multimodal_biometrics",
    )

    # -----------------------
    # Dataset
    # -----------------------
    iris = IrisDataset(PARQUET_PATH)
    fp = FingerprintDataset(PARQUET_PATH)
    dataset = MultiModalDataset(iris, fp)

    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )

    # -----------------------
    # Model
    # -----------------------
    model = MultiModalModel(
        iris_enc=SimpleEncoder(128),
        fp_enc=SimpleEncoder(128),
        fusion=ConcatFusion(128, num_classes=45),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------
    # Resume from checkpoint (optional)
    # -----------------------
    start_epoch = 0
    if resume and os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_checkpoint(
            model,
            optimizer,
            CHECKPOINT_PATH,
        )

    # -----------------------
    # Training loop
    # -----------------------
    num_epochs = 3

    for epoch in range(start_epoch, num_epochs):
        print(f" Epoch {epoch}")
        train_one_epoch(model, loader, optimizer, writer, epoch)

        # Save checkpoint after every epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_dir=CHECKPOINT_DIR,
            filename="latest.pt",
        )

    writer.close()
    print("Training finished")


if __name__ == "__main__":
    main(resume=True)