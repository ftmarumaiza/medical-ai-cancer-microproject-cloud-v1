"""
Detection module: YOLOv8 model loading, inference, and pseudo-label generation.
"""

from .model_loader import load_detection_model
from .inference import run_detection
from .pseudo_label import generate_and_save_pseudo_labels

__all__ = [
    "load_detection_model",
    "run_detection",
    "generate_and_save_pseudo_labels",
]
