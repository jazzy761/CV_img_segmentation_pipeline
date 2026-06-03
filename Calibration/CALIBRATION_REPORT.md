# Calibration Report

## Object Selection

**Target Object:** Pakistani 1 Rupee Coin

The 1 Rupee coin was selected because it is easily available, has a well-defined circular shape, and is straightforward to annotate for image segmentation tasks. Its fixed dimensions make it suitable for later real-world measurement experiments.

---

## Camera Calibration

### Method

Intrinsic camera calibration was performed using OpenCV's chessboard calibration approach. A set of 21 chessboard images was captured from different viewpoints and orientations. Corner points were detected using `cv2.findChessboardCorners()` and camera parameters were estimated using `cv2.calibrateCamera()`.

### Calibration Images

* Total calibration images captured: 21
* Valid detections: 20

### Camera Parameters

#### Camera Matrix

```text
[[1232.18523,    0.00000, 798.70765],
 [   0.00000, 1232.24786, 459.26723],
 [   0.00000,    0.00000,   1.00000]]
```

#### Distortion Coefficients

```text
[0.02001483, 0.09007609, 0.00119459, 0.00396159, -0.57435930]
```

### Reprojection Error

The mean reprojection error obtained during calibration was **0.18 pixels**.

A reprojection error below 0.5 pixels is generally considered excellent for camera calibration. This indicates that the estimated intrinsic parameters accurately model the camera geometry and lens distortion, making the calibration suitable for image undistortion and real-world measurement tasks.


### Image Undistortion

All dataset images can be corrected using OpenCV's `cv2.undistort()` function with the estimated camera matrix and distortion coefficients.

### Reprojection Error

The calibration reprojection error was computed from the detected chessboard corners and was found to be low enough for accurate image correction and measurement tasks.

---

# Dataset Card

## Dataset Summary

* Object Class: Coin
* Annotation Type: Instance Segmentation
* Total Images: 84
* Total Classes: 1

### Class Distribution

| Class | Count      |
| ----- | ---------- |
| Coin  | 84 Images |

### Image Variations

The dataset contains:

* Single coin images
* Multiple coin images
* Different backgrounds
* Different viewing angles
* Different lighting conditions
* Different object positions and scales

### Dataset Split

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 20%        |
| Test       | 10%        |

Annotations were created using Roboflow and exported for segmentation model training.

---

# Setup Guide

1. Capture calibration images using a chessboard pattern.
2. Estimate camera intrinsic parameters using OpenCV calibration.
3. Undistort images using the computed camera matrix and distortion coefficients.
4. Capture coin images using the calibrated camera.
5. Annotate coin masks using Roboflow.
6. Export the dataset in segmentation format.
7. Train a segmentation model on the labelled dataset.
8. Use segmented object masks and calibrated camera parameters for real-world measurement in millimetres.
