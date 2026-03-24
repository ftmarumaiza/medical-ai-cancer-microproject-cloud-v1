"""
YOLOv8 inference: run detection and return bounding boxes with confidence.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

import config


def run_detection(
    model: YOLO,
    image: Image.Image,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    imgsz: Optional[int] = None,
) -> Tuple[Results, float, Image.Image]:
    """
    Run YOLOv8 detection on an image. Model resizes internally.

    Args:
        model: Loaded YOLO model.
        image: PIL Image (RGB).
        conf_threshold: Minimum confidence; None uses config.
        iou_threshold: NMS IoU; None uses config.
        imgsz: Inference size; None uses config.MAX_IMAGE_DIM.

    Returns:
        (results, processing_time_seconds, annotated_image)
    """
    import time

    conf = conf_threshold if conf_threshold is not None else config.CONFIDENCE_THRESHOLD
    iou = iou_threshold if iou_threshold is not None else config.IOU_THRESHOLD
    sz = imgsz or config.MAX_IMAGE_DIM

    img_arr = np.array(image)

    t0 = time.perf_counter()
    results = model.predict(
        img_arr,
        conf=conf,
        iou=iou,
        imgsz=sz,
        device=config.DEVICE,
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - t0

    annotated = results.plot()
    annotated_pil = Image.fromarray(annotated)

    return results, elapsed, annotated_pil


def parse_detections(results: Results) -> List[dict]:
    """
    Parse Ultralytics results into list of detection dicts.

    Returns:
        List of {"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
    """
    detections: List[dict] = []
    if results.boxes is None:
        return detections

    names = results.names or {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        label = names.get(cls_id, "object")
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": xyxy,
        })
    return detections
