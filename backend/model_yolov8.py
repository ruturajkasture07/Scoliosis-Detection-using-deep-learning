from ultralytics import YOLO
import numpy as np
from PIL import Image

def run_yolov8_inference(image: Image.Image, model_path: str):
    model = YOLO(model_path)
    # ultralytics predict can take PIL images directly
    results = model.predict(source=image, conf=0.5, save=False)[0]
    
    boxes_out = []
    keypoints_out = []
    
    if len(results.keypoints) == 0 or len(results.keypoints.xy) == 0:
        return boxes_out, keypoints_out
        
    keypoints = results.keypoints.xy.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    
    # Sort vertebrae top-to-bottom
    sorted_indices = np.argsort([np.mean(kpts[:, 1]) for kpts in keypoints])
    keypoints = keypoints[sorted_indices]
    boxes = boxes[sorted_indices]
    
    for i, kpts in enumerate(keypoints):
        # YOLOv8 pose returns 4 keypoints per instance
        # Ensure order is TL, TR, BL, BR
        sorted_by_y = kpts[np.argsort(kpts[:, 1])]
        top_pts = sorted_by_y[:2]
        bottom_pts = sorted_by_y[2:]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
        
        ordered_kpts = [
            [float(top_pts[0][0]), float(top_pts[0][1])],
            [float(top_pts[1][0]), float(top_pts[1][1])],
            [float(bottom_pts[0][0]), float(bottom_pts[0][1])],
            [float(bottom_pts[1][0]), float(bottom_pts[1][1])]
        ]
        keypoints_out.append(ordered_kpts)
        
        box = boxes[i]
        boxes_out.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        
    flat_keypoints_x = []
    flat_keypoints_y = []
    for kp_group in keypoints_out:
        for kp in kp_group:
            flat_keypoints_x.append(kp[0])
            flat_keypoints_y.append(kp[1])
            
    return boxes_out, flat_keypoints_x + flat_keypoints_y
