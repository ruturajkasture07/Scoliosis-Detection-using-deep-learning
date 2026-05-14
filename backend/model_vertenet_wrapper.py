import os
import numpy as np
from PIL import Image
import tempfile
from model_vertenet import VerteNetInference

def run_vertenet_inference(image: Image.Image, model_path: str):
    # Vertenet predict method expects an image path
    # We will save the PIL image to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_vertenet_input.jpg")
    image.save(temp_path)
    
    # Initialize the model (using strip mode for vertical detection, 17 vertebrae)
    model = VerteNetInference(weights_path=model_path, head_conv=512, topk=17, mode='strip', conf_thresh=0.05)
    result = model.predict(temp_path)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    boxes_out = []
    keypoints_out = []
    
    # Extract
    for v in result['_raw']['vertebrae_list']:
        corners = v['corners'] # shape (4, 2) in order: TL, TR, BL, BR
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])
        
        boxes_out.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        
        # corners are guaranteed to be TL, TR, BL, BR by Vertenet's decode_detections
        ordered_kpts = [
            [float(corners[0][0]), float(corners[0][1])],
            [float(corners[1][0]), float(corners[1][1])],
            [float(corners[2][0]), float(corners[2][1])],
            [float(corners[3][0]), float(corners[3][1])]
        ]
        keypoints_out.append(ordered_kpts)
        
    flat_keypoints_x = []
    flat_keypoints_y = []
    for kp_group in keypoints_out:
        for kp in kp_group:
            flat_keypoints_x.append(kp[0])
            flat_keypoints_y.append(kp[1])
            
    return boxes_out, flat_keypoints_x + flat_keypoints_y
