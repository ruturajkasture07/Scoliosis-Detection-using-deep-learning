# ==============================================================================
# FINAL PRODUCTION INFERENCE CELL (YOLOv8-Pose)
# ==============================================================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==============================================================================
# 1. LOAD THE TRAINED YOLOv8-POSE MODEL
# ==============================================================================
# Loading the weights from your successful baseline run
MODEL_WEIGHTS = "/kaggle/working/runs/pose/Scoliosis_Detection/v1_yolov8n_baseline/weights/best.pt"
model = YOLO(MODEL_WEIGHTS)

# ==============================================================================
# 2. RUN INFERENCE ON TEST IMAGE
# ==============================================================================
TEST_IMAGE_PATH = "/kaggle/input/datasets/adwayne/yolo-data/scoliosis_fixed/images/val/001521.jpg"

if os.path.exists(TEST_IMAGE_PATH):
    # Run Inference
    results = model.predict(source=TEST_IMAGE_PATH, conf=0.5, save=False)[0]
    img_bgr = cv2.imread(TEST_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    # ==============================================================================
    # 3. EXTRACT AND FORMAT KEYPOINTS
    # ==============================================================================
    # Extract tensor shape: [N, 4, 2]
    keypoints = results.keypoints.xy.cpu().numpy()
    
    # Sort vertebrae top-to-bottom (based on average Y coordinate)
    sorted_indices = np.argsort([np.mean(kpts[:, 1]) for kpts in keypoints])
    keypoints = keypoints[sorted_indices]

    all_x = []
    all_y = []
    vertebrae_coords_text = []

    for i, kpts in enumerate(keypoints):
        v_text = f"Vertebra {i+1} Coordinates:"
        
        # Convex Hull for drawing clean bounding polygons
        pts = np.array(kpts, np.int32)
        hull = cv2.convexHull(pts)
        cv2.polylines(img_rgb, [hull], True, (0, 255, 0), 2)
        
        for x, y in kpts:
            x_int, y_int = round(float(x)), round(float(y))
            
            # Append to flat lists for the landmark_xy format
            all_x.append(x_int)
            all_y.append(y_int)
            
            # Append to text block for detailed printing
            v_text += f"\n  -> x: {x_int}, y: {y_int}"
            
            # Draw keypoint dots
            cv2.circle(img_rgb, (x_int, y_int), 4, (255, 0, 0), -1)
            
        vertebrae_coords_text.append(v_text)

    # ==============================================================================
    # 4. PRINT FINAL OUTPUT FORMAT
    # ==============================================================================
    print("\n" + "="*80)
    print(f"landmark_xy: {all_x + all_y}")
    print(f"image_shape: ({h}, {w}, {c})")
    print("="*80 + "\n")
    
    print(f"Detected {len(keypoints)} vertebrae.")
    for report in vertebrae_coords_text:
        print(report)

    # ==============================================================================
    # 5. VISUALIZE
    # ==============================================================================
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title(f"Inference Result: {len(keypoints)} Vertebrae Identified", fontsize=14)
    plt.axis('off')
    plt.show()
else:
    print("❌ Error: Test image not found. Please check the path.")