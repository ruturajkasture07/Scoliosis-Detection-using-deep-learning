from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from math_utils import cobb_angle_cal
import os
import json

from model_keypoint_rcnn import run_kprcnn_inference
from model_residual_unet import run_residual_unet_inference
from model_yolov8 import run_yolov8_inference
from model_vertenet_wrapper import run_vertenet_inference

app = FastAPI(title="ScolioVis Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "Models")
    models = {
        "keypoint_rcnn": os.path.join(models_dir, "keypointsrcnn_weights.pt"),
        "yolov8_pose": os.path.join(models_dir, "yolo_best.pt"),
        "vertenet": os.path.join(models_dir, "Vertenet_best.pth"),
        "residual_unet": os.path.join(models_dir, "residual-unet_best_model_v2.pth"),
    }
    return models

models_paths = load_models()

def mock_inference(image, num_vertebrae=17):
    # This is a fallback mock function if a specific model isn't fully implemented yet
    width, height = image.size
    boxes = []
    keypoints = []
    start_y = height * 0.1
    end_y = height * 0.9
    step_y = (end_y - start_y) / num_vertebrae
    for i in range(num_vertebrae):
        cy = start_y + i * step_y
        cx = width / 2 + np.sin(i / num_vertebrae * np.pi) * 50
        bw, bh = 80, step_y * 0.8
        x1, y1 = cx - bw/2, cy - bh/2
        x2, y2 = cx + bw/2, cy + bh/2
        boxes.append([x1, y1, x2, y2])
        keypoints.extend([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    return boxes, keypoints

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form("keypoint_rcnn")):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    boxes = []
    keypoints = []
    
    try:
        if model_type == "keypoint_rcnn":
            boxes, keypoints = run_kprcnn_inference(image, models_paths["keypoint_rcnn"])
        elif model_type == "residual_unet":
            boxes, keypoints = run_residual_unet_inference(image, models_paths["residual_unet"])
        elif model_type == "yolov8_pose":
            boxes, keypoints = run_yolov8_inference(image, models_paths["yolov8_pose"])
        elif model_type == "vertenet":
            boxes, keypoints = run_vertenet_inference(image, models_paths["vertenet"])
        else:
            print(f"Model {model_type} not natively implemented yet. Using mock inference.")
            boxes, keypoints = mock_inference(image)
    except Exception as e:
        print(f"Error during inference with {model_type}: {e}")
        # Fallback
        boxes, keypoints = mock_inference(image)
        
    cobb_angles, angles_with_pos, curve_type, midpoint_lines, standardized_kp = cobb_angle_cal(keypoints, image.size)
    
    return {
        "detections": boxes,
        "landmarks": standardized_kp,
        "cobb_angles_list": cobb_angles,
        "angles_with_pos": angles_with_pos,
        "curve_type": curve_type,
        "midpoint_lines": midpoint_lines
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
