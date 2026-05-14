import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np

def get_kprcnn_model(model_path):
    num_keypoints = 4
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    return model

def _filter_output(output):
    # 1. Get Scores
    scores = output['scores'].detach().cpu().numpy()

    # 2. Get Indices of Scores over Threshold
    high_scores_idxs = np.where(scores > 0.5)[0].tolist()

    if len(high_scores_idxs) == 0:
        return [], [], []

    # 3. Get Indices after Non-max Suppression
    post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], 0.3).cpu().numpy()

    # 4. Get final `bboxes` and `keypoints` and `scores` based on indices
    np_keypoints = output['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
    np_bboxes = output['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
    np_scores = output['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()

    # 5. Get the Top 17 Scores
    sorted_scores_idxs = np.argsort(-1*np_scores) # descending

    np_scores = np_scores[sorted_scores_idxs][:18]
    np_keypoints = np.array([np_keypoints[idx] for idx in sorted_scores_idxs])[:18]
    np_bboxes = np.array([np_bboxes[idx] for idx in sorted_scores_idxs])[:18]

    # 6. Sort by centroid y
    ymins = np.array([np.mean([kp[1] for kp in kps]) for kps in np_keypoints])

    sorted_ymin_idxs = np.argsort(ymins) # ascending
    
    np_scores = np.array([np_scores[idx] for idx in sorted_ymin_idxs])
    np_keypoints = np.array([np_keypoints[idx] for idx in sorted_ymin_idxs])
    np_bboxes = np.array([np_bboxes[idx] for idx in sorted_ymin_idxs])
    
    # 7. Convert everything to List Instead of Numpy
    keypoints_list = []
    for kps in np_keypoints:
        keypoints_list.append([list(map(float, kp[:2])) for kp in kps])

    bboxes_list = []
    for bbox in np_bboxes:
        bboxes_list.append(list(map(int, bbox.tolist())))
      
    scores_list = np_scores.tolist()

    return bboxes_list, keypoints_list, scores_list

# Global model cache to avoid reloading on every request
_kprcnn_model = None

def run_kprcnn_inference(image, model_path):
    global _kprcnn_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if _kprcnn_model is None:
        _kprcnn_model = get_kprcnn_model(model_path)
        _kprcnn_model.to(device)
        _kprcnn_model.eval()

    # Ensure image is in RGB format for F.to_tensor
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        
    image_input = F.to_tensor(image).to(device)

    with torch.no_grad():
        outputs = _kprcnn_model([image_input])

    bboxes, keypoints, scores = _filter_output(outputs[0])
    
    # Convert Keypoints output to a flat format [x1..xn, y1..yn]
    flat_keypoints_x = []
    flat_keypoints_y = []
    for kp_group in keypoints:
        for kp in kp_group:
            flat_keypoints_x.append(float(kp[0]))
            flat_keypoints_y.append(float(kp[1]))
            
    return bboxes, flat_keypoints_x + flat_keypoints_y
