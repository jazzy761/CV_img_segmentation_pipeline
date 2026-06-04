from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import torch
import uuid
import os

from Model.inference_core import get_model, run_pipeline, compute_dimensions

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once (important for performance)
model = get_model()
model.load_state_dict(torch.load("Model/best_model.pth", map_location=device))
model.to(device)
model.eval()

K = np.load("Calibration/script_results/camera_matrix.npy")
dist = np.load("Calibration/script_results/dist_coeffs.npy")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>CV Measurement System</h2>
    <form action="/predict/" enctype="multipart/form-data" method="post">
        <input type="file" name="file">
        <input type="submit">
    </form>
    """


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed_img, pred = run_pipeline(image, model, device, K, dist)

    scores = pred["scores"].cpu().numpy()
    boxes = pred["boxes"].cpu().numpy()
    masks = pred["masks"].cpu().numpy()

    results = []

    for score, box, mask in zip(scores, boxes, masks):

        if score < 0.7:
            continue

        mask = mask[0] > 0.5

        w_mm, h_mm = compute_dimensions(mask)

        results.append({
            "confidence": float(score),
            "width_mm": float(w_mm),
            "height_mm": float(h_mm)
        })

    filename = f"{uuid.uuid4().hex}.jpg"
    out_path = f"API/static/outputs/{filename}"

    cv2.imwrite(out_path, processed_img)

    return {
        "results": results,
        "output_image": out_path
    }