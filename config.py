"""
Configuration for the Medical Image Detection System.
Centralizes paths, thresholds, and model settings.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
PSEUDO_LABELS_DIR = PROJECT_ROOT / "pseudo_labels"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Ensure directories exist on import
for _dir in (DATA_DIR, PSEUDO_LABELS_DIR, UPLOADS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Single setting for which model is loaded. Use a filename (e.g. "best.pt") to load
# from project root, or an absolute path (e.g. "C:/models/polyp.pt"). Placing a .pt
# file in the project root and setting MODEL_NAME to that filename makes the app use
# it. For better confidence on medical images, use a fine-tuned model trained on
# similar data (e.g. Kvasir for polyp detection).
MODEL_NAME = "best.pt"
MODEL_SIZE = "nano"  # nano, small, medium, large, xlarge
DEVICE = "cpu"  # Force CPU for portability; set "cuda" if GPU available

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
MAX_IMAGE_DIM = 640  # Resize longer side before inference (YOLO default)
CONFIDENCE_THRESHOLD = 0.25  # Min confidence for YOLO to return a detection (low so we see results on medical images)
IOU_THRESHOLD = 0.45  # NMS IoU threshold

# Display bar: we treat detections ≥ this as "high confidence" in UI and pseudo-labels
HIGH_CONFIDENCE_DISPLAY_THRESHOLD = 0.80  # ≥ 80% = reliable / save as pseudo-label

# -----------------------------------------------------------------------------
# Pseudo-label (semi-supervised simulation)
# -----------------------------------------------------------------------------
PSEUDO_LABEL_CONFIDENCE_THRESHOLD = 0.80  # Save pseudo-label only if conf ≥ 80%
PSEUDO_LABEL_CLASS_NAME = "polyp"  # Default class for medical polyp detection

# -----------------------------------------------------------------------------
# Risk level mapping (for clinical-style dashboard)
# -----------------------------------------------------------------------------
RISK_HIGH_CONFIDENCE = 0.80   # confidence ≥ 80% → HIGH
RISK_MEDIUM_CONFIDENCE = 0.50  # 50–80% → MEDIUM, < 60% → LOW
