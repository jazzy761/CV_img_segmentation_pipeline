## Camera Calibration

### Calibration Methodology

Camera calibration was performed using OpenCV's standard chessboard-based intrinsic calibration procedure.

A calibration target consisting of a **9 × 6 internal corner chessboard pattern** was captured from multiple viewpoints, orientations, and distances to provide sufficient geometric variation for parameter estimation.

For each calibration image:

1. The image was converted to grayscale.
2. Chessboard corners were detected using `cv2.findChessboardCorners()`.
3. Corresponding 3D world coordinates and 2D image coordinates were stored.
4. Images with unsuccessful corner detection were discarded.

A total of **21 calibration images** were collected, of which **20 images produced valid corner detections**.

### Object and Image Points

The calibration process requires correspondences between:

* **3D object points** representing known chessboard corner locations in the real world.
* **2D image points** representing detected chessboard corners in the image plane.

The 3D object points were generated using a planar grid:

```python
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
```

These points define the physical geometry of the calibration target.

### Intrinsic Parameter Estimation

Camera intrinsic parameters were estimated using OpenCV's `cv2.calibrateCamera()` function:

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)
```

The calibration process estimates:

* Camera focal lengths
* Principal point coordinates
* Radial lens distortion coefficients
* Tangential lens distortion coefficients
* Camera pose for each calibration image

### Calibration Images

| Metric                  | Value |
| ----------------------- | ----- |
| Total Images Captured   | 21    |
| Valid Corner Detections | 20    |
| Chessboard Pattern      | 9 × 6 |

### Camera Matrix

```text
[[1232.18523,    0.00000, 798.70765],
 [   0.00000, 1232.24786, 459.26723],
 [   0.00000,    0.00000,   1.00000]]
```

The camera matrix contains the intrinsic parameters of the imaging system, including focal length and principal point coordinates.

### Distortion Coefficients

```text
[0.02001483, 0.09007609, 0.00119459, 0.00396159, -0.57435930]
```

These coefficients model radial and tangential lens distortion introduced by the camera optics.

### Reprojection Error Analysis

Calibration quality was evaluated using reprojection error.

For each calibration image:

1. Known 3D chessboard points were projected back into the image plane using the estimated camera parameters.
2. The projected locations were compared against the detected corner locations.
3. The average Euclidean distance between these points was computed.

The resulting mean reprojection error was:

**0.1899 pixels**

A reprojection error below 0.5 pixels is generally considered excellent and indicates that the estimated camera parameters accurately model the imaging geometry.

### Calibration Validation

To validate the calibration results, image undistortion was performed using:

```python
cv2.getOptimalNewCameraMatrix()
cv2.undistort()
```

The original image and corrected image were saved for visual inspection.

The validation confirmed that lens distortion effects were successfully removed and that the calibration parameters were suitable for downstream segmentation and real-world measurement tasks.

### Calibration Outputs

The following calibration artifacts were saved for later use in the measurement pipeline:

| File              | Purpose                      |
| ----------------- | ---------------------------- |
| camera_matrix.npy | Intrinsic camera matrix      |
| dist_coeffs.npy   | Lens distortion coefficients |

These calibration parameters are loaded during inference and measurement to ensure all dimensional estimates are computed from undistorted images.

### Importance for Measurement

Accurate real-world measurement requires geometrically correct imagery.

Lens distortion causes object dimensions to vary across the image plane, resulting in incorrect pixel measurements. Therefore, every image used in the measurement pipeline is first undistorted using the estimated intrinsic parameters before segmentation and pixel-to-millimetre conversion are performed.
