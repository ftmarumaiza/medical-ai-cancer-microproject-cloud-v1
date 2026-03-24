# Kvasir Fine-Tuning: Get best.pt for Higher Confidence

This guide gets you from the Kvasir dataset to a `best.pt` model in your project root so the Streamlit app gives **higher confidence** on Kvasir colonoscopy/polyp images.

---

## Overview

1. **Download** Kvasir-SEG (images + masks).
2. **Convert** to YOLO format (images/train, images/val, labels/train, labels/val) using the provided script.
3. **Train** YOLOv8 on that data with `train_kvasir.py` (or `yolo detect train`).
4. **Use** the produced `best.pt` by placing it in the project root; the app loads it automatically.

---

## Step 1: Download Kvasir-SEG

- Go to: **https://datasets.simula.no/kvasir/**
- Download **Kvasir-SEG** (colonoscopy images with polyp segmentation masks).
- Extract the zip so you have a folder with two subfolders:
  - `images/` — colonoscopy images (e.g. .jpg)
  - `masks/`  — segmentation masks (same filenames; polyp region in white)

Example layout after extract:

```
Kvasir-SEG/
  images/
    cju0qkwl35piu0993l0dewei2.jpg
    ...
  masks/
    cju0qkwl35piu0993l0dewei2.jpg
    ...
```

---

## Step 2: Prepare YOLO Dataset (mask → bbox labels)

From the **project root** (`medical_ai_project/`):

```bash
# Windows (PowerShell or cmd)
python dataset/prepare_kvasir_yolo.py --kvasir-dir "F:\path\to\Kvasir-SEG"

# Example if Kvasir-SEG is on F:\datasets\Kvasir-SEG
python dataset/prepare_kvasir_yolo.py --kvasir-dir "F:\datasets\Kvasir-SEG"
```

- **`--kvasir-dir`**: Path to the Kvasir-SEG root (the folder that contains `images` and `masks`).
- Optional: `--output-dir` (default: `dataset/kvasir_yolo`), `--val-ratio 0.2`, `--seed 42`.

This creates:

- `dataset/kvasir_yolo/images/train/` and `images/val/`
- `dataset/kvasir_yolo/labels/train/` and `labels/val/` (one `.txt` per image: `class_id x_center y_center width height` normalized 0–1).

---

## Step 3: Train YOLOv8

From the **project root**:

```bash
python train_kvasir.py
```

- Uses `dataset/kvasir.yaml` (points to `dataset/kvasir_yolo`).
- Fine-tunes **yolov8n.pt** for 80 epochs by default.
- Saves runs to `runs/detect/kvasir_train/`.
- **At the end**, copies `runs/detect/kvasir_train/weights/best.pt` → **`best.pt`** in the project root.

**Optional (edit `train_kvasir.py`):**

- `epochs=80` → change to 50–100.
- `batch=16` → reduce to 8 or 4 if you run out of memory (CPU/GPU).
- `device="cpu"` → use `device="0"` if you have a GPU.

**Alternative (CLI only):**

```bash
yolo detect train data=dataset/kvasir.yaml model=yolov8n.pt epochs=80 imgsz=640 project=runs/detect name=kvasir_train
```

Then copy the best weights manually:

```bash
copy runs\detect\kvasir_train\weights\best.pt best.pt
```

---

## Step 4: Use best.pt in the App

- After `train_kvasir.py` finishes, **`best.pt`** is already in the project root.
- In **`config.py`** you should have: **`MODEL_NAME = "best.pt"`** (already set).
- **Restart** the Streamlit app:

  ```bash
  streamlit run app.py
  ```

The app will load `best.pt` instead of the COCO fallback, so you should see **higher confidence** on Kvasir images.

---

## Where Things Are

| Item | Location |
|------|----------|
| Conversion script | `dataset/prepare_kvasir_yolo.py` |
| Dataset YAML | `dataset/kvasir.yaml` |
| Prepared YOLO data | `dataset/kvasir_yolo/` (after Step 2) |
| Training script | `train_kvasir.py` |
| Training runs | `runs/detect/kvasir_train/` |
| Best weights (after training) | `runs/detect/kvasir_train/weights/best.pt` → copied to **`best.pt`** (project root) |

---

## Troubleshooting

- **“No image/mask pairs”**: Ensure `--kvasir-dir` points to the folder that contains `images` and `masks`, and that mask filenames match image filenames.
- **Out of memory**: In `train_kvasir.py`, set `batch=4` or `batch=8`.
- **best.pt not found**: Check `runs/detect/kvasir_train/weights/` and copy `best.pt` to the project root manually if the script didn’t.

---

## Reproducing Later

1. Keep (or re-download) Kvasir-SEG.
2. Run: `python dataset/prepare_kvasir_yolo.py --kvasir-dir /path/to/Kvasir-SEG`
3. Run: `python train_kvasir.py`
4. Restart the app; it will use the new `best.pt` in the project root.
