"""
Multimodal biometric inference script.

This script loads a trained multimodal biometric model and performs
inference on iris and fingerprint data stored in a Parquet file.
The model uses modality-specific encoders followed by a fusion module
to produce classification logits.

Key features:
- Deterministic evaluation via `torch.no_grad()`
- Support for loading trained weights from a checkpoint
- Batched inference using PyTorch DataLoader
- Early stopping after a fixed number of predictions for quick testing

Typical usage:
    python inference.py

Inputs:
- Parquet dataset containing iris and fingerprint features

Outputs:
- Printed sample predictions from the trained (or randomly initialized) model
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
from src.utils.checkpoint import load_checkpoint


PARQUET_PATH = "data/processed/biometric.parquet"
CHECKPOINT_PATH = "checkpoints/latest.pt"


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    preds = []

    for batch in loader:
        out = model(batch)
        logits = out["logits"]
        preds.extend(torch.argmax(logits, dim=1).tolist())

        if len(preds) >= 50:
            break

    return preds


def main():
    iris = IrisDataset(PARQUET_PATH)
    fp = FingerprintDataset(PARQUET_PATH)
    dataset = MultiModalDataset(iris, fp)

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = MultiModalModel(
        iris_enc=SimpleEncoder(128),
        fp_enc=SimpleEncoder(128),
        fusion=ConcatFusion(128, num_classes=45),
    )

    # Load trained weights
    if os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(model, optimizer=None, checkpoint_path=CHECKPOINT_PATH)
    else:
        print("No checkpoint found, running with random weights")

    preds = run_inference(model, loader)
    print("Inference finished")
    print("Sample predictions:", preds[:10])


if __name__ == "__main__":
    main()