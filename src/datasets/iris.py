import pyarrow.parquet as pq
from src.utils.io import load_image
from .base import BaseImageDataset


class IrisDataset(BaseImageDataset):
    """
    Iris modality dataset.

    Reads metadata from Parquet and loads images lazily.
    """

    def __init__(self, parquet_path: str):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Filter only iris modality samples
        self.df = df[df["modality"] == "iris"].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = load_image(row.image_path)
        label = int(row.person_id)

        return {
            "image": image,
            "label": label,
        }
