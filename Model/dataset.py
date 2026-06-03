import os
import torch
import numpy as np

from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from torch.utils.data import Dataset


class CoinDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.coco = COCO(
            os.path.join(root_dir, "_annotations.coco.json")
        )

        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(
            self.root_dir,
            image_info["file_name"]
        )

        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:

            x, y, w, h = ann["bbox"]

            boxes.append(
                [x, y, x + w, y + h]
            )

            masks.append(
                self.coco.annToMask(ann)
            )

            labels.append(ann["category_id"])    # coin class

            areas.append(ann["area"])

            iscrowd.append(
                ann.get("iscrowd", 0)
            )

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        masks = torch.as_tensor(
            np.array(masks),
            dtype=torch.uint8
        )

        areas = torch.as_tensor(
            areas,
            dtype=torch.float32
        )

        iscrowd = torch.as_tensor(
            iscrowd,
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        image = torch.from_numpy(
            np.array(image)
        ).permute(2, 0, 1).float() / 255.0

        return image, target
    


if __name__ == "__main__":

    ds = CoinDataset(r"D:\Cv_img_segmentation_pipeline\Dataset\train")
    #"D:\Cv_img_segmentation_pipeline\Dataset\train"

    print("Images:", len(ds))

    img, target = ds[0]

    print(img.shape)
    print(target.keys())
    print(target["masks"].shape)