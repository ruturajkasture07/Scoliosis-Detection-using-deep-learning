import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Global model cache
_unet_model = None

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def get_unet_model(model_path, device):
    try:
        model = smp.Unet(
            encoder_name="efficientnet-b3", 
            encoder_weights=None,
            in_channels=3,                  
            classes=1,                      
        )
        # Try loading without weights_only first for compatibility
        state_dict = torch.load(model_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading UNet model: {e}")
        raise e

def run_residual_unet_inference(image, model_path):
    global _unet_model
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = (768, 768)
    
    if _unet_model is None:
        _unet_model = get_unet_model(model_path, DEVICE)

    img_np = np.array(image.convert('RGB'))
    img_rgb = img_np
    
    orig_h, orig_w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    
    tensor_img = val_transform(image=img_resized)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Using more standard autocast syntax
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                pred_mask = _unet_model(tensor_img)
        else:
            pred_mask = _unet_model(tensor_img)
            
    pred_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
    binary_mask = (pred_prob > 0.5).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scale_x = orig_w / IMAGE_SIZE[0]
    scale_y = orig_h / IMAGE_SIZE[1]
    
    bboxes = []
    keypoints = []
    
    min_area = 200
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
            
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        box[:, 0] = box[:, 0] * scale_x
        box[:, 1] = box[:, 1] * scale_y
        
        # Sort corners to approximate top-left, top-right, bottom-left, bottom-right
        # Actually for Cobb angles, math_utils expects specific order.
        # But rect corners from minAreaRect are sorted clockwise starting from lowest y
        # We'll just return the 4 corners. Math_utils might need adjustment or expects flat.
        
        box_int = np.int32(box)
        x_min, y_min = np.min(box_int[:, 0]), np.min(box_int[:, 1])
        x_max, y_max = np.max(box_int[:, 0]), np.max(box_int[:, 1])
        bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        
        # We need top-left, top-right, bottom-left, bottom-right.
        # Sort by Y first
        sorted_by_y = box[np.argsort(box[:, 1])]
        top_pts = sorted_by_y[:2]
        bottom_pts = sorted_by_y[2:]
        
        top_pts = top_pts[np.argsort(top_pts[:, 0])] # Left, Right
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])] # Left, Right
        
        # Keypoints: top-left, top-right, bottom-left, bottom-right
        keypoints.append([
            [float(top_pts[0][0]), float(top_pts[0][1])],
            [float(top_pts[1][0]), float(top_pts[1][1])],
            [float(bottom_pts[0][0]), float(bottom_pts[0][1])],
            [float(bottom_pts[1][0]), float(bottom_pts[1][1])]
        ])

    # Sort vertebrae by centroid y-coordinate (top to bottom)
    if len(bboxes) > 0:
        sorted_indices = np.argsort([np.mean([pt[1] for pt in kp_group]) for kp_group in keypoints])
        bboxes = [bboxes[i] for i in sorted_indices]
        keypoints = [keypoints[i] for i in sorted_indices]

    flat_keypoints_x = []
    flat_keypoints_y = []
    for kp_group in keypoints:
        for kp in kp_group:
            flat_keypoints_x.append(kp[0])
            flat_keypoints_y.append(kp[1])
            
    return bboxes, flat_keypoints_x + flat_keypoints_y
