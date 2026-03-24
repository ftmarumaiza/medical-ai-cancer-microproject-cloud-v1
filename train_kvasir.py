"""
Fine-tune YOLOv8 on Kvasir polyp dataset and copy best.pt to project root.

Prerequisites:
  1. Run dataset/prepare_kvasir_yolo.py so dataset/kvasir_yolo/ exists.
  2. pip install ultralytics (already in requirements.txt).

Usage (from project root):
  python train_kvasir.py

Optional env or edit below: epochs, batch, device.
After training, best.pt is copied to PROJECT_ROOT/best.pt so the Streamlit app uses it.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "kvasir.yaml"
KVASIR_YOLO_DIR = PROJECT_ROOT / "dataset" / "kvasir_yolo"


def main() -> int:
    if not DATA_YAML.exists():
        print(f"Missing {DATA_YAML}. Create it or run from project root.")
        return 1
    if not (KVASIR_YOLO_DIR / "images" / "train").exists():
        print(
            f"Missing prepared dataset at {KVASIR_YOLO_DIR}. "
            "Run: python dataset/prepare_kvasir_yolo.py --kvasir-dir /path/to/Kvasir-SEG"
        )
        return 1

    from ultralytics import YOLO

    # Use path relative to cwd; run from project root
    data_yaml_str = str(DATA_YAML)
    if not Path(data_yaml_str).is_absolute():
        data_yaml_str = str(DATA_YAML.resolve())

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml_str,
        epochs=80,
        imgsz=640,
        batch=16,
        device="cpu",  # or "0" for GPU
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="kvasir_train",
        exist_ok=True,
        pretrained=True,
    )

    best_src = PROJECT_ROOT / "runs" / "detect" / "kvasir_train" / "weights" / "best.pt"
    if not best_src.exists() and hasattr(results, "save_dir"):
        best_src = Path(results.save_dir) / "weights" / "best.pt"
    if not best_src.exists():
        print("Training finished but best.pt not found. Look under runs/detect/kvasir_train/weights/")
        return 0

    best_dest = PROJECT_ROOT / "best.pt"
    shutil.copy2(best_src, best_dest)
    print(f"Copied {best_src} -> {best_dest}. Restart the Streamlit app to use the new model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
