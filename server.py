import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import uvicorn

MODEL_PATH = "yolov8n.pt"
TARGET_CLASS = "dog"

app = FastAPI()

# Allow frontend (e.g., Vercel domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO once
model = YOLO(MODEL_PATH)

# Resolve class id
DOG_CLASS_ID = None
for idx, name in model.names.items():
    if name == TARGET_CLASS:
        DOG_CLASS_ID = idx
        break
if DOG_CLASS_ID is None:
    raise ValueError(f"{TARGET_CLASS} not found in model.names")

@app.post("/detect")
async def detect_dog(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run YOLO (no tracking here, just detection)
    results = model.predict(image, classes=[DOG_CLASS_ID], conf=0.35, verbose=False)

    detections = []
    if results and results[0].boxes is not None:
        r = results[0]
        for box, score in zip(r.boxes.xyxy, r.boxes.conf):
            x1, y1, x2, y2 = map(float, box)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "class_id": int(DOG_CLASS_ID),
                    "class_name": TARGET_CLASS,
                }
            )

    return JSONResponse({"detections": detections})


if __name__ == "__main__":
    # Run locally: python server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
