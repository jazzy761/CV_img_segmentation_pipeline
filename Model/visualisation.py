import os
import cv2
import torch
import numpy as np

from dataset import CoinDataset

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


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DIR = r"D:\Cv_img_segmentation_pipeline\Dataset\test"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_model.pth"
)

os.makedirs("results", exist_ok=True)



# Load model
model = get_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()



# Dataset


dataset = CoinDataset(TEST_DIR)
print("Test Images:", len(dataset))



# Predict & Save
for idx in range(min(10, len(dataset))):

    image_tensor, target = dataset[idx]

    with torch.no_grad():

        prediction = model(
            [image_tensor.to(DEVICE)]
        )[0]

    image = (
        image_tensor.permute(1, 2, 0)
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    scores = prediction["scores"].cpu().numpy()
    boxes = prediction["boxes"].cpu().numpy()
    masks = prediction["masks"].cpu().numpy()

    for score, box, mask in zip(scores, boxes, masks):

        if score < 0.7:
            continue

        x1, y1, x2, y2 = box.astype(int)

        mask = mask[0] > 0.5

        image[mask] = (
            image[mask] * 0.5 +
            np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    output_path = f"results/pred_{idx+1}.jpg"

    cv2.imwrite(output_path, image)

    print("Saved:", output_path)

print("Done.")