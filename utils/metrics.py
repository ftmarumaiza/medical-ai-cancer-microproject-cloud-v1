"""
Metrics for detection results: count, average confidence, processing time.
"""

from typing import List


def compute_detection_metrics(
    detections: List[dict],
    processing_time_seconds: float,
) -> dict:
    """
    Compute dashboard metrics from detection list and timing.

    Args:
        detections: List of {"label", "confidence", "bbox"}.
        processing_time_seconds: Elapsed inference time.

    Returns:
        {
            "total_objects": int,
            "average_confidence": float (0–1),
            "processing_time_ms": float,
        }
    """
    n = len(detections)
    if n == 0:
        avg_conf = 0.0
    else:
        avg_conf = sum(d.get("confidence", 0) for d in detections) / n
    return {
        "total_objects": n,
        "average_confidence": round(avg_conf, 4),
        "processing_time_ms": round(processing_time_seconds * 1000, 2),
    }
