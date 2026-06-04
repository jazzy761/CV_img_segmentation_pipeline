import os
import cv2
import torch
import argparse
import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



# Model

def get_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes
    )

    return model



# Pixel → MM measurement

def compute_dimensions(mask, px_to_mm):
    """
    mask: binary mask (H,W)
    returns width_mm, height_mm
    """

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0

    width_px = xs.max() - xs.min()
    height_px = ys.max() - ys.min()

    return width_px * px_to_mm, height_px * px_to_mm



# Main pipeline

def run(image_path):

    os.makedirs("output", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Load image
    
    image = cv2.imread(image_path)

    
    # Undistortion (Phase 2)
    
    K = np.load(r"D:\Cv_img_segmentation_pipeline\Calibration\script_results\camera_matrix.npy")
    dist = np.load(r"D:\Cv_img_segmentation_pipeline\Calibration\script_results\dist_coeffs.npy")

    h, w = image.shape[:2]

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    image = cv2.undistort(image, K, dist, None, new_K)

    
    # Model
    
    model = get_model()
    model.load_state_dict(torch.load("Model/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    
    # Inference
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        pred = model([tensor.to(device)])[0]

    
    # Outputs
    
    scores = pred["scores"].cpu().numpy()
    boxes = pred["boxes"].cpu().numpy()
    masks = pred["masks"].cpu().numpy()

    PX_TO_MM = 0.25  # replace with calibrated value later

    result = image.copy()

    for score, box, mask in zip(scores, boxes, masks):

        if score < 0.7:
            continue

        mask = mask[0] > 0.5

        
        # Measurement (Phase 3 core)
        
        width_mm, height_mm = compute_dimensions(mask, PX_TO_MM)

        print("Confidence:", score)
        print("Width (mm):", width_mm)
        print("Height (mm):", height_mm)

        
        # Overlay mask
        
        result[mask] = (result[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            result,
            f"{score:.2f} | {width_mm:.1f}mm x {height_mm:.1f}mm",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    
    # Save outputs
    
    cv2.imwrite("output/overlay.jpg", result)

    print("\nSaved outputs:")
    print("output/overlay.jpg")



# CLI

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)

    args = parser.parse_args()

    run(args.image)