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


def _matches_cancer_label(label: str) -> bool:
    normalized = (label or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in CANCER_LABEL_KEYWORDS)


def _filter_cancer_evidence(detections: List[dict]) -> List[dict]:
    evidence: List[dict] = []
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        label = str(det.get("label", ""))
        if conf < CANCER_DECISION_CONFIDENCE:
            continue
        if not _matches_cancer_label(label):
            continue
        evidence.append(det)
    return evidence


def predict_cancer(image) -> Dict[str, Any]:
    """
    Run inference and convert detections to a binary cancer presence output.

    Cancer is considered present only when:
    1) detection confidence >= CANCER_DECISION_CONFIDENCE
    2) detection label matches medical cancer keywords
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

    cancer_evidence = _filter_cancer_evidence(detections)
    cancer_present = len(cancer_evidence) > 0
    max_confidence = max((d["confidence"] for d in cancer_evidence), default=0.0)

    return {
        "cancer_present": cancer_present,
        "result": "Cancer Detected" if cancer_present else "No Cancer Detected",
        "confidence": round(max_confidence, 4),
        "detections": detections,
        "cancer_evidence": cancer_evidence,
        "decision": {
            "keywords": list(CANCER_LABEL_KEYWORDS),
            "min_confidence": CANCER_DECISION_CONFIDENCE,
        },
        "metrics": metrics,
    }
