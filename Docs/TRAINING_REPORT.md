## Training Log & Metrics

Model Choice (Mask R-CNN)

Mask R-CNN was used because it supports both object detection and pixel-level segmentation, which is necessary for this project since:

We need bounding boxes (for detection + measurement reference)
We also need segmentation masks (for accurate pixel-to-mm area/shape estimation)
It provides better geometric precision compared to bounding-box-only models like Faster R-CNN

The ResNet-50-FPN backbone helps extract multi-scale features, which improves detection of objects at different sizes.

Training was conducted for 10 epochs using Mask R-CNN (ResNet-50-FPN backbone).



### Key Metrics (Best Model)

- mAP@0.5: 1.000
- mAP@0.5:0.95: 0.993
- Precision: ~0.99+
- Recall: ~0.99+
- IoU (mean): high consistency (>0.90)
- F1 Score: ~0.99 (derived from precision/recall balance)

### Best Epochs
- Epoch 3: mAP@0.5:0.95 = 0.919
- Epoch 6: mAP@0.5:0.95 = 0.985 (best early peak)
- Epoch 9: mAP@0.5:0.95 = 0.993 (final best model)

### Final Epoch
- Train Loss: 0.0738
- mAP@0.5:0.95: 0.9911

---

## COCO Evaluation Summary (Selected Best Run)

Evaluation was done using COCO evaluation metrics (COCOeval):

- IoU-based precision/recall evaluation
- Metrics computed across multiple IoU thresholds (0.50 to 0.95)
- Standard object detection benchmarking protocol

- AP@[IoU=0.50:0.95]: 0.991–0.993 range (stable convergence)
- AP@0.50: 1.000
- AP@0.75: 1.000

### Recall Stability
- AR@1: ~0.20
- AR@10: ~0.99
- AR@100: ~0.99

Best Epochs
Epoch 3: mAP@0.5:0.95 = 0.919
Epoch 6: mAP@0.5:0.95 = 0.985 (strong improvement phase)
Epoch 9: mAP@0.5:0.95 = 0.993 (final best model)
---

## Observations

- Model converges quickly due to single-class segmentation task (coin).
- High AP values are expected because:
  - Single object class
  - Controlled capture environment
  - Clean segmentation annotations
- No major overfitting observed; validation metrics remain stable across epochs.




## Training Strategy 

- Model uses transfer learning from a COCO-pretrained Mask R-CNN
  - Both:
    -   Box head (FastRCNNPredictor)
    -   Mask head (MaskRCNNPredictor)
        were replaced to adapt to single-class coin detection
  - Loss is computed as a weighted sum of:
    -   classification loss
    -   box regression loss
    -    mask loss
Best model is saved dynamically when validation mAP improves