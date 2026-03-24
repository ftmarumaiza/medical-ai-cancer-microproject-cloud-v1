# Cloud Deployment Version (Student Microproject)

This folder converts your existing local cancer detection project into a cloud-ready web application without changing the core model inference logic in `detector/`.

## Flow

User uploads image -> image sent to FastAPI server -> model runs inference -> prediction returned to user.

## Folder Structure

```text
cloud_app/
|-- backend/
|   |-- main.py                    # FastAPI app and /predict endpoint
|   |-- config.py                  # Cloud app settings (model path, storage mode)
|   `-- services/
|       |-- preprocessing.py       # Image preprocessing for API requests
|       |-- model_service.py       # Model loading + prediction wrapper
|       `-- storage_service.py     # Local storage simulation + optional S3
|-- frontend/
|   |-- index.html                 # Upload UI
|   |-- styles.css                 # Basic styling
|   `-- app.js                     # Calls /predict and shows response
|-- model/
|   `-- README.md                  # Where to place trained model files
|-- cloud_storage/uploads/         # Simulated cloud storage for uploaded images
|-- requirements-cloud.txt
|-- Procfile
|-- render.yaml
|-- railway.json
`-- Dockerfile
```

## Core Model Logic (Reused)

The cloud backend directly reuses existing modules:

- `detector/model_loader.py`
- `detector/inference.py`
- `utils/image_processing.py`
- `utils/metrics.py`

No change is required in your original deep learning logic.

## Sample Code

### 1. API endpoint (`/predict`) - `backend/main.py`

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    storage_uri = storage_service.save_image(file_bytes, file.filename, file.content_type)
    preprocessed_image, preprocessing_meta = preprocess_upload_image(file_bytes, max_dim=MAX_IMAGE_DIM)
    prediction = predict_cancer(preprocessed_image)

    return {
        "filename": file.filename,
        "storage_uri": storage_uri,
        "preprocessing": preprocessing_meta,
        **prediction,
    }
```

### 2. Image preprocessing - `backend/services/preprocessing.py`

```python
def preprocess_upload_image(file_bytes: bytes, max_dim: int):
    pil_image = Image.open(BytesIO(file_bytes)).convert("RGB")
    np_image = np.array(pil_image)
    resized_np, resized_size = preprocess_for_inference(np_image, max_dim=max_dim)
    return Image.fromarray(resized_np), {
        "processed_width": resized_size[0],
        "processed_height": resized_size[1],
    }
```

### 3. Model loading + prediction - `backend/services/model_service.py`

```python
@lru_cache(maxsize=1)
def get_model():
    return load_detection_model(weights_path=MODEL_PATH, device=DEVICE)


def predict_cancer(image):
    model = get_model()
    results, processing_time, _ = run_detection(
        model,
        image,
        conf_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        imgsz=MAX_IMAGE_DIM,
    )
    detections = parse_detections(results)
    return {
        "cancer_present": len(detections) > 0,
        "detections": detections,
    }
```

## Local Run

From `medical_ai_project/`:

```bash
pip install -r cloud_app/requirements-cloud.txt
uvicorn cloud_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://localhost:8000`

## Storage Options

### Option A: Simulated cloud storage (default)

- Set `STORAGE_MODE=local`
- Uploaded files are saved in `cloud_app/cloud_storage/uploads/`

### Option B: AWS S3

Set environment variables:

```bash
STORAGE_MODE=s3
S3_BUCKET_NAME=your-bucket-name
S3_REGION=us-east-1
S3_PREFIX=medical-uploads
```

Also configure AWS credentials in your hosting platform.

## Deployment Steps

### Deploy on Render

1. Push project to GitHub.
2. Create new **Web Service** on Render.
3. Set **Root Directory** to `medical_ai_project/cloud_app`.
4. Build command:
   `pip install -r requirements-cloud.txt`
5. Start command:
   `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add env vars:
   - `STORAGE_MODE=local`
   - `MODEL_PATH=../yolov8n.pt` (or `../best.pt`)
   - `DEVICE=cpu`
7. Deploy and open the service URL.

### Deploy on Railway

1. Create a new Railway project from your GitHub repo.
2. Set root directory to `medical_ai_project/cloud_app`.
3. Set start command:
   `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add same environment variables as Render.
5. Deploy and open generated URL.

### Deploy on AWS EC2 (Docker)

From `medical_ai_project/`:

```bash
docker build -f cloud_app/Dockerfile -t medical-ai-cloud .
docker run -d -p 80:8000 --name medical-ai-app medical-ai-cloud
```

Then open: `http://<EC2_PUBLIC_IP>/`

## Notes for Your Existing Trained Model

- If you already have `best.pt`, place it in `medical_ai_project/` and set:
  `MODEL_PATH=../best.pt`
- If not provided, app falls back to `yolov8n.pt`.

## Quick API Test

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

## Cancer Decision Rules

To avoid false positives from non-medical objects (dog, tie, bed, etc.), binary cancer output now uses:

- `CANCER_LABEL_KEYWORDS` (default: `cancer,tumor,tumour,polyp,lesion,malignant,neoplasm,adenoma`)
- `CANCER_DECISION_CONFIDENCE` (default: `0.50`)

Only detections matching these medical keywords and minimum confidence are counted as cancer evidence.
