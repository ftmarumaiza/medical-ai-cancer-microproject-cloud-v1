"""
Model loading and inference wrapper for the API.
Keeps existing detector logic intact by reusing detector/* modules.
"""

from functools import lru_cache
from typing import Any, Dict

from detector.inference import parse_detections, run_detection
from detector.model_loader import load_detection_model
from utils.metrics import compute_detection_metrics

from ..config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    IOU_THRESHOLD,
    MAX_IMAGE_DIM,
    MODEL_PATH,
)


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load the trained model once and reuse it for all requests."""
    return load_detection_model(weights_path=MODEL_PATH, device=DEVICE)


def predict_cancer(image) -> Dict[str, Any]:
    """
    Run inference and convert detections to a binary cancer presence output.
    """
    model = get_model()

    results, processing_time, _ = run_detection(
        model,
        image,
        conf_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        imgsz=MAX_IMAGE_DIM,
    )
    detections = parse_detections(results)
    metrics = compute_detection_metrics(detections, processing_time)

    max_confidence = max((d["confidence"] for d in detections), default=0.0)
    cancer_present = len(detections) > 0

    return {
        "cancer_present": cancer_present,
        "result": "Cancer Detected" if cancer_present else "No Cancer Detected",
        "confidence": round(max_confidence, 4),
        "detections": detections,
        "metrics": metrics,
    }
