"""
Pseudo-label generation for semi-supervised simulation.
Saves high-confidence predictions as JSON for future training.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config


def generate_and_save_pseudo_labels(
    image_name: str,
    detections: List[dict],
    confidence_threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Filter detections by confidence and save as pseudo-labels (JSON).
    Used to simulate semi-supervised pipeline for demonstration.

    Args:
        image_name: Original image filename.
        detections: List of {"label", "confidence", "bbox"}.
        confidence_threshold: Min confidence; None uses config.
        output_dir: Where to save JSON; None uses config.PSEUDO_LABELS_DIR.

    Returns:
        List of pseudo-labels that were saved (for display).
    """

    thresh = confidence_threshold or config.PSEUDO_LABEL_CONFIDENCE_THRESHOLD
    out_dir = output_dir or config.PSEUDO_LABELS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo = []
    for d in detections:
        conf = d.get("confidence", 0.0)
        if conf < thresh:
            continue
        label = d.get("label", config.PSEUDO_LABEL_CLASS_NAME)
        pseudo.append({
            "image": image_name,
            "label": label,
            "confidence": round(conf, 4),
        })

    if not pseudo:
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(image_name).stem
    out_path = out_dir / f"pseudo_{safe_name}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"image": image_name, "pseudo_labels": pseudo, "timestamp": timestamp},
            f,
            indent=2,
        )
    return pseudo
