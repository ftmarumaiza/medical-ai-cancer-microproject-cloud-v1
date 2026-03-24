"""
Model loading and inference wrapper for the API.
Keeps existing detector logic intact by reusing detector/* modules.
"""

from functools import lru_cache
from typing import Any, Dict, List

from detector.inference import parse_detections, run_detection
from detector.model_loader import load_detection_model
from utils.metrics import compute_detection_metrics

from ..config import (
    CANCER_DECISION_CONFIDENCE,
    CANCER_LABEL_KEYWORDS,
    CANCER_POSITIVE_LABELS,
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


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def _is_cancer_label(label: str) -> bool:
    normalized = _normalize_label(label)
    if not normalized:
        return False

    if CANCER_POSITIVE_LABELS and any(token in normalized for token in CANCER_POSITIVE_LABELS):
        return True

    return any(keyword in normalized for keyword in CANCER_LABEL_KEYWORDS)


def _extract_cancer_evidence(detections: List[dict]) -> List[dict]:
    evidence: List[dict] = []
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        label = str(det.get("label", ""))
        if conf < CANCER_DECISION_CONFIDENCE:
            continue
        if _is_cancer_label(label):
            evidence.append(det)
    return evidence


def predict_cancer(image) -> Dict[str, Any]:
    """
    Run inference and convert detections to a binary cancer presence output.

    Important: this binary decision is based on medical label matching,
    not on 'any object detected'.
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

    cancer_evidence = _extract_cancer_evidence(detections)
    cancer_present = len(cancer_evidence) > 0

    top_detection = max(detections, key=lambda d: float(d.get("confidence", 0.0)), default=None)
    top_label = top_detection.get("label") if top_detection else None

    max_confidence = max((d["confidence"] for d in cancer_evidence), default=0.0)

    if cancer_present:
        result_text = "Cancer Detected"
    elif top_label:
        result_text = f"No Cancer Detected (Detected: {top_label})"
    else:
        result_text = "No Cancer Detected"

    return {
        "cancer_present": cancer_present,
        "result": result_text,
        "confidence": round(max_confidence, 4),
        "top_detection": top_detection,
        "cancer_evidence": cancer_evidence,
        "detections": detections,
        "decision": {
            "keywords": list(CANCER_LABEL_KEYWORDS),
            "positive_labels": list(CANCER_POSITIVE_LABELS),
            "min_confidence": CANCER_DECISION_CONFIDENCE,
        },
        "metrics": metrics,
    }
