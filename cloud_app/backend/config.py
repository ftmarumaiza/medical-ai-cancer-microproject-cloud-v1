"""
Configuration for the cloud-ready FastAPI application.
"""

from pathlib import Path
import os


CLOUD_APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLOUD_APP_ROOT.parent
FRONTEND_DIR = CLOUD_APP_ROOT / "frontend"
LOCAL_STORAGE_DIR = CLOUD_APP_ROOT / "cloud_storage" / "uploads"
MODEL_DIR = CLOUD_APP_ROOT / "model"

LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "best.pt",
    PROJECT_ROOT / "yolov8n.pt",
    MODEL_DIR / "best.pt",
    MODEL_DIR / "yolov8n.pt",
]


def _resolve_model_path() -> str:
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        return env_model_path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    return "yolov8n.pt"


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODEL_PATH = _resolve_model_path()
DEVICE = os.getenv("DEVICE", "cpu")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "640"))

# Binary cancer decision settings for detection outputs.
# Preferred path: label-based matching using medical keywords.
CANCER_LABEL_KEYWORDS = tuple(
    part.strip().lower()
    for part in os.getenv(
        "CANCER_LABEL_KEYWORDS",
        "cancer,tumor,tumour,polyp,lesion,malignant,neoplasm,adenoma",
    ).split(",")
    if part.strip()
)

# Optional exact/contains positive labels (example: "polyp,0") for custom-trained models.
CANCER_POSITIVE_LABELS = tuple(
    part.strip().lower()
    for part in os.getenv("CANCER_POSITIVE_LABELS", "").split(",")
    if part.strip()
)

# Minimum confidence for the final cancer/no-cancer decision.
CANCER_DECISION_CONFIDENCE = float(os.getenv("CANCER_DECISION_CONFIDENCE", "0.25"))

# If True, when no keyword match exists we allow non-COCO labels as cancer evidence.
# This helps custom medical models that emit labels like "0" or "class0".
ENABLE_NONCOCO_FALLBACK = _env_bool("ENABLE_NONCOCO_FALLBACK", True)

STORAGE_MODE = os.getenv("STORAGE_MODE", "local").lower()  # local | s3
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_PREFIX = os.getenv("S3_PREFIX", "medical-uploads")
