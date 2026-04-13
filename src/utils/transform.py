import torchvision.transforms as T


def get_image_transform(image_size: int = 64):
    """
    Returns a deterministic image transform pipeline.
    """
    return T.Compose(
        [
            T.Grayscale(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )