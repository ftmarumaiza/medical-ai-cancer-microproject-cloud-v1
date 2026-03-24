Place your trained model file here if you want a self-contained cloud_app package.

Recommended:
- `best.pt` (your fine-tuned medical model)

Model loading priority in this cloud setup:
1. `MODEL_PATH` environment variable
2. `../best.pt`
3. `../yolov8n.pt`
4. `model/best.pt`
5. `model/yolov8n.pt`
