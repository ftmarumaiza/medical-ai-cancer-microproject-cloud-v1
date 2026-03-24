"""
Model loading and inference wrapper for the API.
Keeps existing detector logic intact by reusing detector/* modules.
"""

from functools import lru_cache
from typing import Any, Dict, List, Tuple

from detector.inference import parse_detections, run_detection
from detector.model_loader import load_detection_model
from utils.metrics import compute_detection_metrics

from ..config import (
    CANCER_DECISION_CONFIDENCE,
    CANCER_LABEL_KEYWORDS,
    CANCER_POSITIVE_LABELS,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    ENABLE_NONCOCO_FALLBACK,
    IOU_THRESHOLD,
    MAX_IMAGE_DIM,
    MODEL_PATH,
)


COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
}


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load the trained model once and reuse it for all requests."""
    return load_detection_model(weights_path=MODEL_PATH, device=DEVICE)


def _normalize(label: str) -> str:
    return (label or "").strip().lower()


def _matches_cancer_label(label: str) -> bool:
    normalized = _normalize(label)
    if not normalized:
        return False

    if CANCER_POSITIVE_LABELS:
        return any(token in normalized for token in CANCER_POSITIVE_LABELS)

    return any(keyword in normalized for keyword in CANCER_LABEL_KEYWORDS)


def _is_coco_label(label: str) -> bool:
    return _normalize(label) in COCO_LABELS


def _filter_cancer_evidence(detections: List[dict]) -> Tuple[List[dict], str]:
    keyword_evidence: List[dict] = []

    for det in detections:
        conf = float(det.get("confidence", 0.0))
        label = str(det.get("label", ""))
        if conf < CANCER_DECISION_CONFIDENCE:
            continue
        if _matches_cancer_label(label):
            keyword_evidence.append(det)

    if keyword_evidence:
        return keyword_evidence, "label_match"

    # Fallback for custom-trained medical models with non-descriptive labels.
    # Prevent false positives from COCO classes such as dog/bed/tie.
    if not ENABLE_NONCOCO_FALLBACK:
        return [], "none"

    fallback_evidence: List[dict] = []
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        label = str(det.get("label", ""))
        if conf < CANCER_DECISION_CONFIDENCE:
            continue
        if _is_coco_label(label):
            continue
        fallback_evidence.append(det)

    if fallback_evidence:
        return fallback_evidence, "non_coco_fallback"

    return [], "none"


def predict_cancer(image) -> Dict[str, Any]:
    """
    Run inference and convert detections to a binary cancer presence output.

    Decision logic:
    1) Keyword/label match with confidence threshold.
    2) Optional fallback for non-COCO labels (custom medical classes like "0").
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

    cancer_evidence, decision_mode = _filter_cancer_evidence(detections)
    cancer_present = len(cancer_evidence) > 0
    max_confidence = max((d["confidence"] for d in cancer_evidence), default=0.0)

    return {
        "cancer_present": cancer_present,
        "result": "Cancer Detected" if cancer_present else "No Cancer Detected",
        "confidence": round(max_confidence, 4),
        "detections": detections,
        "cancer_evidence": cancer_evidence,
        "decision": {
            "mode": decision_mode,
            "keywords": list(CANCER_LABEL_KEYWORDS),
            "positive_labels": list(CANCER_POSITIVE_LABELS),
            "min_confidence": CANCER_DECISION_CONFIDENCE,
            "non_coco_fallback": ENABLE_NONCOCO_FALLBACK,
        },
        "metrics": metrics,
    }
