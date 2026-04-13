import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Feature-level fusion by concatenation.

    This is a standard and widely-used multimodal
    fusion technique due to simplicity and stability.
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()

        self.classifier = nn.Linear(embedding_dim * 2, num_classes)

    def forward(self, z_iris, z_fp):
        """
        Args:
            z_iris: Tensor (B, embedding_dim)
            z_fp:   Tensor (B, embedding_dim)

        Returns:
            dict with:
              - logits: Tensor (B, num_classes)
              - loss:   Scalar tensor (dummy loss for demo)
        """
        z = torch.cat([z_iris, z_fp], dim=1)
        logits = self.classifier(z)

        # NOTE: Loss is intentionally simplistic
        loss = logits.mean()

        return {
            "logits": logits,
            "loss": loss,
        }
``