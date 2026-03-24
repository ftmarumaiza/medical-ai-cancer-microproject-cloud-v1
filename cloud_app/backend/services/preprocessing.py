"""
Image preprocessing utilities for upload requests.
"""

from io import BytesIO
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from utils.image_processing import preprocess_for_inference


def preprocess_upload_image(file_bytes: bytes, max_dim: int) -> Tuple[Image.Image, Dict[str, int]]:
    """
    Convert uploaded bytes into model-ready PIL image.

    Returns:
        (preprocessed_image, metadata)
    """
    pil_image = Image.open(BytesIO(file_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    resized_np, resized_size = preprocess_for_inference(np_image, max_dim=max_dim)
    resized_pil = Image.fromarray(resized_np)

    metadata = {
        "original_width": pil_image.width,
        "original_height": pil_image.height,
        "processed_width": resized_size[0],
        "processed_height": resized_size[1],
    }
    return resized_pil, metadata
