import torch.nn as nn


class SimpleEncoder(nn.Module):
    """
    Simple encoder for biometric images.

    This intentionally avoids complex architectures.
    The goal is explainability and clean abstraction,
    not model performance.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 1, H, W)

        Returns:
            Tensor of shape (B, embedding_dim)
        """
        return self.net(x)
