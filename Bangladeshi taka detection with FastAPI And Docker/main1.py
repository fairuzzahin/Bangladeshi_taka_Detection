from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import os
import uuid

# ==========================================================
# FastAPI Application
# ==========================================================
app = FastAPI(
    title="Bangladeshi Taka Detection API",
    description="Detect Bangladeshi Taka notes using a trained YOLO model.",
    version="1.0"
)

# ==========================================================
# Load Trained YOLO Model
# ==========================================================
# Place your trained model (best.pt) inside the project folder
# Example:
# project/
# ├── main.py
# ├── best.pt
# ├── uploads/
# └── ...
 
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# ==========================================================
# Create Upload Directory
# ==========================================================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================================
# Taka Detection Endpoint
# ==========================================================
@app.post("/detect")
async def detect_taka(file: UploadFile = File(...)):
    """
    Upload an image and detect Bangladeshi Taka notes.

    Returns:
        {
            "result": "DETECTED",
            "notes": ["100_taka", "500_taka"]
        }

        or

        {
            "result": "NONE",
            "notes": []
        }
    """

    # ------------------------------------------------------
    # Save uploaded image
    # ------------------------------------------------------
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_extension}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # --------------------------------------------------
        # Open image
        # --------------------------------------------------
        image = Image.open(filepath).convert("RGB")

        # --------------------------------------------------
        # Run YOLO Inference
        # --------------------------------------------------
        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        detected_notes = []

        # --------------------------------------------------
        # Process detections
        # --------------------------------------------------
        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                detected_notes.append(class_name)

        # Remove duplicate detections
        detected_notes = list(set(detected_notes))

        # --------------------------------------------------
        # Return Response
        # --------------------------------------------------
        if detected_notes:
            return JSONResponse(
                status_code=200,
                content={
                    "result": "DETECTED",
                    "notes": detected_notes
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "result": "NONE",
                "notes": []
            }
        )

    finally:
        # --------------------------------------------------
        # Delete uploaded image
        # --------------------------------------------------
        if os.path.exists(filepath):
            os.remove(filepath)


# ==========================================================
# Root Endpoint
# ==========================================================
@app.get("/")
def home():
    return {
        "message": "Bangladeshi Taka Detection API is running.",
        "endpoint": "/detect"
    }