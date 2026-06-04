import cv2
import torch
import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model():
    model = maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, 2
    )

    return model


def compute_dimensions(mask, px_to_mm=0.25):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return 0, 0

    w_px = xs.max() - xs.min()
    h_px = ys.max() - ys.min()

    return w_px * px_to_mm, h_px * px_to_mm


def run_pipeline(image, model, device, K, dist):

    # undistort
    h, w = image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    image = cv2.undistort(image, K, dist, None, new_K)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        pred = model([tensor.to(device)])[0]

    return image, pred