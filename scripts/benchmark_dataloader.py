import time
from torch.utils.data import DataLoader

from src.datasets.iris import IrisDataset
from src.datasets.fingerprint import FingerprintDataset
from src.datasets.multimodal import MultiModalDataset

PARQUET_PATH = "data/processed/biometric.parquet"


def benchmark_dataloader(num_workers: int, batch_size: int, num_batches: int = 50):
    iris = IrisDataset(PARQUET_PATH)
    fp = FingerprintDataset(PARQUET_PATH)
    dataset = MultiModalDataset(iris, fp)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    start = time.time()
    for i, _ in enumerate(loader):
        if i >= num_batches:
            break
    elapsed = time.time() - start

    print(
        f"✅ num_workers={num_workers}, "
        f"batch_size={batch_size}, "
        f"time={elapsed:.2f}s"
    )


def main():
    print("📊 DataLoader Benchmarking\n")

    for num_workers in [0, 2, 4]:
        benchmark_dataloader(
            num_workers=num_workers,
            batch_size=32,
        )


if __name__ == "__main__":
    main()