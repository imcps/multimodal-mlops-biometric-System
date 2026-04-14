from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseImageDataset(Dataset, ABC):
    """
    Abstract base class for all image-based biometric datasets.

    Enforces a consistent interface across modalities:
    - __len__ must return dataset size
    - __getitem__ must return a dict with:
        - "image": torch.Tensor
        - "label": int
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample.

        Expected output format:
        {
            "image": torch.Tensor,
            "label": int
        }
        """
        pass
