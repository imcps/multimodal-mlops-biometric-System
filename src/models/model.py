import torch.nn as nn


class MultiModalModel(nn.Module):
    """
    Full multimodal model combining multiple encoders
    and a fusion module.
    """

    def __init__(self, iris_enc, fp_enc, fusion):
        super().__init__()

        self.iris_encoder = iris_enc
        self.fp_encoder = fp_enc
        self.fusion = fusion

    def forward(self, batch):
        """
        Args:
            batch: dict with keys
              - iris
              - fingerprint
              - label (not used here)

        Returns:
            dict returned by fusion module
        """
        z_iris = self.iris_encoder(batch["iris"])
        z_fp = self.fp_encoder(batch["fingerprint"])

        return self.fusion(z_iris, z_fp)