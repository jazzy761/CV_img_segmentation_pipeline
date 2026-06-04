## Training Log & Metrics

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

- AP@[IoU=0.50:0.95]: 0.991–0.993 range (stable convergence)
- AP@0.50: 1.000
- AP@0.75: 1.000

### Recall Stability
- AR@1: ~0.20
- AR@10: ~0.99
- AR@100: ~0.99

---

## Observations

- Model converges quickly due to single-class segmentation task (coin).
- High AP values are expected because:
  - Single object class
  - Controlled capture environment
  - Clean segmentation annotations
- No major overfitting observed; validation metrics remain stable across epochs.