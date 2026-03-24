"""
Storage service for uploaded images.
Supports local simulation and optional AWS S3.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import (
    LOCAL_STORAGE_DIR,
    S3_BUCKET_NAME,
    S3_PREFIX,
    S3_REGION,
    STORAGE_MODE,
)


class CloudStorageService:
    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or STORAGE_MODE).lower()

    @staticmethod
    def _safe_filename(filename: str) -> str:
        original = Path(filename or "upload.jpg")
        suffix = original.suffix.lower() or ".jpg"
        stem = "".join(c for c in original.stem if c.isalnum() or c in ("-", "_"))
        stem = stem or "medical_image"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{stem}_{timestamp}_{unique_id}{suffix}"

    def _save_local(self, file_bytes: bytes, filename: str) -> str:
        safe_name = self._safe_filename(filename)
        output_path = LOCAL_STORAGE_DIR / safe_name
        output_path.write_bytes(file_bytes)
        return f"local://cloud_storage/uploads/{safe_name}"

    def _save_s3(self, file_bytes: bytes, filename: str, content_type: Optional[str]) -> str:
        if not S3_BUCKET_NAME:
            raise ValueError("S3_BUCKET_NAME is required when STORAGE_MODE=s3")

        try:
            import boto3
        except Exception as exc:
            raise RuntimeError("boto3 is required for S3 mode") from exc

        safe_name = self._safe_filename(filename)
        key = f"{S3_PREFIX.strip('/')}/{safe_name}"

        client = boto3.client("s3", region_name=S3_REGION)
        client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType=content_type or "application/octet-stream",
        )
        return f"s3://{S3_BUCKET_NAME}/{key}"

    def save_image(self, file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> str:
        if self.mode == "s3":
            return self._save_s3(file_bytes, filename, content_type)
        return self._save_local(file_bytes, filename)
