import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import CoinDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval




def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes
    )

    return model



# Training loop

def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)



# Evaluation (COCO mAP)

def evaluate(model, loader, dataset, device):
    model.eval()

    coco_gt = dataset.coco
    coco_dt_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"].item()

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box

                    coco_dt_list.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    coco_dt = coco_gt.loadRes(coco_dt_list)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@0.5:0.95



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CoinDataset(os.path.join(args.data, "train"))
    val_ds = CoinDataset(os.path.join(args.data, "valid"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model(num_classes=2)  # background + coin
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    scaler = torch.cuda.amp.GradScaler()

    best_map = 0

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler
        )

        map50_95 = evaluate(
            model,
            val_loader,
            val_ds,
            device
        )

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"mAP@0.5:0.95: {map50_95:.4f}")

        if map50_95 > best_map:
            best_map = map50_95
            torch.save(model.state_dict(), r"D:\Cv_img_segmentation_pipeline\Model\best_model.pth")
            print("Saved best model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)

    args = parser.parse_args()

    main(args)


#python Model/train.py --data Dataset --epochs 10 --batch_size 2 --lr 0.005