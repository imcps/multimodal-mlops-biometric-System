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

        if batch_idx >= 10:  # keep runtime small
            break


def main(resume: bool = False):
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
        print(f"🚀 Epoch {epoch}")
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
    print("✅ Training finished")


if __name__ == "__main__":
    # Set resume=True to continue training
    main(resume=True)