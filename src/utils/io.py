from PIL import Image
from src.utils.transform import get_image_transform


# Default transform (can be overridden later if needed)
_DEFAULT_TRANSFORM = get_image_transform()


def load_image(path: str):
    """
    Loads an image from disk and applies preprocessing.

    Args:
        path (str): Path to image file

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    with Image.open(path) as img:
        return _DEFAULT_TRANSFORM(img)