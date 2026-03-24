FROM python:3.11-slim

WORKDIR /app

COPY cloud_app/requirements-cloud.txt /app/requirements-cloud.txt
RUN pip install --no-cache-dir -r /app/requirements-cloud.txt

COPY detector /app/detector
COPY utils /app/utils
COPY config.py /app/config.py
COPY yolov8n.pt /app/yolov8n.pt
COPY cloud_app /app/cloud_app

WORKDIR /app/cloud_app

ENV STORAGE_MODE=local
ENV DEVICE=cpu
ENV MODEL_PATH=/app/yolov8n.pt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
