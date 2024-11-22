import PIL
import numpy as np
import torch
from typing import Union, List


def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images
