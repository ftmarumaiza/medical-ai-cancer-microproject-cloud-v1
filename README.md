# Semi-Supervised Medical Image Detection System

**AI-powered medical image detection with an interactive dashboard.**  
Deep learning (YOLOv8) + pseudo-label simulation for a clinical-style demo. Suitable for BTech/CSE final-year micro projects and demonstrations.

---

## Overview

This system performs **object detection** on medical images (e.g. colonoscopy for polyp detection), draws **bounding boxes**, and simulates a **semi-supervised pipeline** by saving high-confidence predictions as **pseudo-labels** in JSON. The UI is a **Streamlit** medical dashboard with a dark theme, teal/cyan accents, and clear metrics.

### Features

- **Upload** medical images (drag & drop).
- **YOLOv8 (nano)** detection — lightweight, CPU-friendly.
- **Bounding boxes** and confidence scores.
- **Pseudo-label generation** (confidence > 0.6) stored in `/pseudo_labels/`.
- **Dashboard:** detection result, confidence, risk level (HIGH/MEDIUM/LOW), processing time.
- **Metrics:** total objects, average confidence, processing time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                            │
│  ┌─────────────┐  ┌─────────────────────────────┐  ┌─────────────────┐  │
│  │ Upload      │  │ Main: Original / Detection   │  │ AI Analysis     │  │
│  │ (sidebar)   │  │ View + Metrics               │  │ (sidebar)       │  │
│  └──────┬──────┘  └──────────────┬───────────────┘  └────────┬────────┘  │
└─────────┼────────────────────────┼──────────────────────────┼───────────┘
          │                        │                          │
          ▼                        ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  utils.image_processing    detector.inference      detector.pseudo_label │
│  (load, resize)            (YOLOv8 predict)        (save JSON)           │
└─────────────────────────────────────────────────────────────────────────┘
          │                        │
          ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐
│  detector.model_loader │  │  config.py          │
│  (YOLOv8n cached)    │  │  (paths, thresholds) │
└─────────────────────┘  └─────────────────────┘
```

**Workflow:** Upload → Preprocess → YOLO inference → Bounding boxes → Pseudo-label (if conf > 0.6) → Dashboard.

---

## Installation

### Requirements

- **Python 3.10+**
- **pip**

### Steps

1. **Clone or copy** the project into a folder, e.g. `medical_ai_project/`.

2. **Create a virtual environment (recommended):**
   ```bash
   cd medical_ai_project
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This installs PyTorch, Ultralytics (YOLOv8), Streamlit, OpenCV, Pillow, etc. First run may download YOLOv8n weights automatically.

---

## How to Run

From the project root (`medical_ai_project/`):

```bash
streamlit run app.py
```

- App opens in the browser (default: `http://localhost:8501`).
- Use the **left sidebar** to upload an image (PNG, JPG, JPEG, BMP).
- Wait for **“Running AI Analysis…”** to finish.
- View **Original** / **Detection** in the main area and **AI Analysis** + metrics in the sidebar.

---

## Dataset

- **In-app:** No dataset is required; you can upload any image. The pretrained YOLOv8 model (COCO) will detect 80 classes; for a **medical** demo, use medical images.
- **Recommended:** [Kvasir Polyp Dataset](https://datasets.simula.no/kvasir/) — colonoscopy images, ideal for polyp detection and for fine-tuning.
- See **`dataset/README.md`** for download instructions and optional folder layout.
- To **fine-tune YOLOv8 on Kvasir** and get higher confidence in the app, see **[TRAINING.md](TRAINING.md)** (dataset preparation script, `kvasir.yaml`, training script, and copy of `best.pt` to project root).

---

## Project Structure

```
medical_ai_project/
├── app.py                 # Streamlit entry point
├── config.py              # Paths, thresholds, model settings
├── detector/
│   ├── model_loader.py    # YOLOv8 load + cache
│   ├── inference.py       # Run detection, parse boxes
│   ├── pseudo_label.py    # Save pseudo-labels (JSON)
├── ui/
│   ├── components.py      # Header, upload, analysis panel, metrics, toggle
│   ├── theme.py           # Dark medical dashboard CSS
├── utils/
│   ├── image_processing.py # Load, resize
│   ├── metrics.py         # Total objects, avg confidence, time
├── dataset/
│   ├── README.md          # Dataset links and download
│   ├── kvasir.yaml        # YOLO dataset config for Kvasir
│   └── prepare_kvasir_yolo.py  # Convert Kvasir-SEG to YOLO format
├── pseudo_labels/         # Generated pseudo-label JSONs (auto-created)
├── train_kvasir.py        # Fine-tune YOLOv8 on Kvasir, copy best.pt to root
├── TRAINING.md            # Step-by-step: Kvasir → best.pt
├── requirements.txt
└── README.md              # This file
```

---

## Pseudo-Label Format

Saved under `pseudo_labels/` when confidence > 0.6:

```json
{
  "image": "img1.png",
  "pseudo_labels": [
    { "label": "polyp", "confidence": 0.82 }
  ],
  "timestamp": "20250109_143022"
}
```

---

## Configuration

Edit **`config.py`** to change:

- `MODEL_NAME` — which model weights to load (see below)
- `PSEUDO_LABEL_CONFIDENCE_THRESHOLD` (default 0.8)
- `RISK_HIGH_CONFIDENCE` / `RISK_MEDIUM_CONFIDENCE` for risk levels
- `MAX_IMAGE_DIM`, `CONFIDENCE_THRESHOLD`, `DEVICE` (e.g. `"cuda"` if GPU available)

---

## Using a different fine-tuned model

To use a different YOLO model that gives **better confidence** on medical images:

1. **Place the `.pt` weights file** in the project root (e.g. `medical_ai_project/my_model.pt`), or put it anywhere and use an absolute path.
2. **Set `MODEL_NAME` in `config.py`** to that path:
   - For a file in project root: `MODEL_NAME = "my_model.pt"`
   - For an absolute path: `MODEL_NAME = "C:/models/polyp.pt"` (Windows) or `MODEL_NAME = "/path/to/model.pt"` (Linux/macOS).
3. **Restart the Streamlit app**; it will load the new model.

**Notes:**

- The model must be a **YOLO-compatible `.pt` file** (e.g. Ultralytics YOLOv8 format).
- Better confidence on medical images typically comes from a model **trained or fine-tuned on medical data** similar to the images you analyze (e.g. Kvasir for polyp detection).

---

## Screenshots

*(Add screenshots of the dashboard, upload area, and detection result here for a GitHub-ready README.)*

- Screenshot 1: Main dashboard with upload and detection view.
- Screenshot 2: AI Analysis panel with risk level and confidence.
- Screenshot 3: Metrics panel (total objects, avg confidence, time).

---

## Future Improvements

- **True semi-supervised training:** Use pseudo-labels to retrain or fine-tune the model.
- **Transformer / attention:** Add attention modules or use a transformer-based detector.
- **Hospital deployment:** REST API, auth, audit logs, DICOM support.
- **Edge AI:** Export to ONNX/TFLite for edge or mobile.
- **Mobile app:** Companion app for capture and quick review.
- **Segmentation:** Optional U-Net (or similar) branch for polyp segmentation.

---

## License

Use for educational and project demonstration. For medical use, ensure compliance with clinical and data regulations.
