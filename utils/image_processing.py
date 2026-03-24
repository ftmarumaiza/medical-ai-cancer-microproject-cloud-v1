"""
Image preprocessing: resize and normalize for inference.
"""

import io
from typing import Tuple, Union

import numpy as np
from PIL import Image


def preprocess_for_inference(
    image: np.ndarray,
    max_dim: int = 640,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image so longest side is max_dim, preserving aspect ratio.
    Keeps RGB order; no normalization (YOLO handles internally).

    Args:
        image: BGR or RGB numpy array (H, W, C).
        max_dim: Maximum dimension (e.g. 640 for YOLO).

    Returns:
        (resized_image, (new_width, new_height))
    """
    h, w = image.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w == w and new_h == h:
        return image, (w, h)
    import cv2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (new_w, new_h)


def load_image(path_or_bytes: Union[str, bytes]) -> Image.Image:
    """
    Load image from file path or bytes into PIL Image (RGB).

    Args:
        path_or_bytes: Path string or bytes (uploaded file).

    Returns:
        PIL Image in RGB mode.
    """
    if isinstance(path_or_bytes, bytes):
        return Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    return Image.open(path_or_bytes).convert("RGB")
