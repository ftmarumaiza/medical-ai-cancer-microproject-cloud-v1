"""
FastAPI server for cloud-based medical image prediction.
"""

from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure original project modules (detector/utils/config.py) are importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config import FRONTEND_DIR, MAX_IMAGE_DIM, MODEL_PATH, STORAGE_MODE
from .services.model_service import predict_cancer
from .services.preprocessing import preprocess_upload_image
from .services.storage_service import CloudStorageService


storage_service = CloudStorageService()


app = FastAPI(
    title="Medical Cancer Detection API",
    description="Upload image -> run deep learning inference -> receive cancer prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def read_home():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "storage_mode": STORAGE_MODE,
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        storage_uri = storage_service.save_image(file_bytes, file.filename, file.content_type)
        preprocessed_image, preprocessing_meta = preprocess_upload_image(
            file_bytes,
            max_dim=MAX_IMAGE_DIM,
        )
        prediction = predict_cancer(preprocessed_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "filename": file.filename,
        "storage_uri": storage_uri,
        "preprocessing": preprocessing_meta,
        **prediction,
    }
