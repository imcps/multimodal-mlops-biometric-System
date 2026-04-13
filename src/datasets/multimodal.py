from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    Dataset that combines multiple biometric modalities
    (e.g., iris + fingerprint).

    Assumes datasets are aligned by index.
    """

    def __init__(self, iris_dataset, fingerprint_dataset):
        assert len(iris_dataset) == len(
            fingerprint_dataset
        ), "Datasets must be aligned and of equal length"

        self.iris_dataset = iris_dataset
        self.fingerprint_dataset = fingerprint_dataset

    def __len__(self):
        return len(self.iris_dataset)

    def __getitem__(self, idx):
        iris_sample = self.iris_dataset[idx]
        fp_sample = self.fingerprint_dataset[idx]

        return {
            "iris": iris_sample["image"],
            "fingerprint": fp_sample["image"],
            "label": iris_sample["label"],
        }