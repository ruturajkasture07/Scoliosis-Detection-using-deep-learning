import requests
import numpy as np
from PIL import Image
from io import BytesIO

def test_model(model_type):
    print(f"Testing model: {model_type}...")
    # Create a dummy spine-like image (white column on black background)
    img_array = np.zeros((1000, 500, 3), dtype=np.uint8)
    img_array[100:900, 200:300, :] = 200 # A "spine"
    
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    files = {'file': ('test.jpg', buf, 'image/jpeg')}
    data = {'model_type': model_type}
    
    try:
        response = requests.post('http://localhost:8000/api/predict', files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()
            print(f"SUCCESS: Received response from {model_type}")
            print(f"Cobb Angles: {res_json.get('cobb_angles_list')}")
            print(f"Curve Type: {res_json.get('curve_type')}")
            return True
        else:
            print(f"FAILED: {model_type} returned status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"ERROR: Could not connect to backend for {model_type}: {e}")
        return False

if __name__ == "__main__":
    kprcnn_ok = test_model('keypoint_rcnn')
    unet_ok = test_model('residual_unet')
    vertenet_ok = test_model('vertenet')
    yolo_ok = test_model('yolov8_pose')
    
    if kprcnn_ok and unet_ok and vertenet_ok and yolo_ok:
        print("\nCONCLUSION: All models are properly integrated and responding.")
    else:
        print("\nCONCLUSION: One or more models failed. Check backend logs.")
