"""
YOLOv8 model loading with caching for fast repeated inference.
Loads the model from config.MODEL_NAME; relative paths are resolved from project root.
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

import config


def load_detection_model(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> YOLO:
    """
    Load YOLOv8 detection model using the configured path (or override).

    Uses config.MODEL_NAME by default. Relative paths (e.g. "best.pt") are resolved
    from project root so the model is always loaded from the project folder regardless
    of current working directory; absolute paths are used as-is.

    Args:
        weights_path: Override path to .pt weights; None uses config.MODEL_NAME.
        device: 'cpu' or 'cuda'; None uses config.DEVICE.

    Returns:
        Loaded YOLO model instance.
    """
    weights = weights_path or config.MODEL_NAME
    dev = device or config.DEVICE

    # Relative path → resolve from project root; absolute path → use as-is
    p = Path(weights)
    if not p.is_absolute():
        weights = str(config.PROJECT_ROOT / p)

    # If configured weights file is missing, fall back to pretrained yolov8n (downloads if needed)
    if not Path(weights).exists():
        weights = "yolov8n.pt"

    model = YOLO(weights)
    model.to(dev)
    return model
