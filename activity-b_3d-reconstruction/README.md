# Sparse 3D Reconstruction: A Two-View Framework
**AGAST + FREAK + Levenberg-Marquardt Bundle Adjustment**
*Computer Vision Final Project — Prof. Luca Lombardi, May 2026*

---

## Overview

This project implements a complete **two-view Structure from Motion (SfM)** pipeline that estimates a sparse 3D point cloud from a stereo image pair. Starting from two 2D photographs of the same scene, the script recovers the underlying 3D geometry through feature matching, epipolar pose estimation, algebraic triangulation, and non-linear optimization.

---

## Project Structure

```
activity-b_3d-reconstruction/
    ├── reconstruct.py            # Two-view SfM pipeline
    ├── requirements.txt          # Python dependencies
    ├── images
    ├── cloud_000_002.np
    ├── cloud_000_004.npy
    ├── cloud_000_006.npy
    ├── cloud_000_008.npy
    ├── cloud.png
    ├── computer_vision_project.pdf
    ├── test.py
    ├── merged_cloud.py
    ├── dino_Ps.mat
    ├── dino_cloud.npy
    └── README.md

---

## Algorithm: Two-View SfM Pipeline

### 1. Feature Detection — AGAST

The Adaptive and Generic Accelerated Segment Test (AGAST) detects corner keypoints in each image. It improves on the classic FAST detector by optimizing decision trees for corner classification, offering high repeatability without sacrificing speed.

### 2. Feature Description — FREAK

Detected keypoints are described using the Fast Retina Keypoint (FREAK) descriptor, inspired by the topology of the human retina. Its concentric sampling pattern mimics the eye's foveal structure. Because FREAK produces **binary strings**, matches are computed using **Hamming distance**.

### 3. Matching — Brute-Force + Cross-Check

A Brute-Force Matcher finds the best pair for each descriptor. Cross-checking ensures a match is only kept if it is the best match in both directions, significantly reducing false positives.

### 4. Pose Estimation — Essential Matrix + RANSAC

The Essential Matrix E encodes the rigid transformation (R, t) between the two cameras via the epipolar constraint:

```
x2ᵀ E x1 = 0
```

RANSAC robustly estimates E from the noisy match set, discarding outliers. The relative rotation R and translation t are then recovered via chirality check (ensuring reconstructed points have positive depth in both views).

### 5. Triangulation — SVD

With the projection matrices P1 = K[I|0] and P2 = K[R|t] known, each matched point pair is lifted to 3D by solving the homogeneous system AX = 0 via Singular Value Decomposition. The solution corresponds to the right singular vector of the smallest singular value.

### 6. Outlier Filtering — Z-score

Before bundle adjustment, points further than `outlier-thresh` standard deviations from the centroid are removed. This prevents geometric outliers from corrupting the non-linear optimization.

### 7. Bundle Adjustment — Levenberg-Marquardt

The triangulated cloud is refined by minimizing the **reprojection error** — the geometric distance between observed 2D features and the projection of the estimated 3D points:

```
E(R, t, X) = Σ [ ‖u1,i − π(K, I, 0, Xi)‖² + ‖u2,i − π(K, R, t, Xi)‖² ]
```

The Levenberg-Marquardt algorithm solves this non-linear least-squares problem by interpolating between gradient descent (large damping λ) and Gauss-Newton (small λ), converging rapidly near the solution.

---

## Requirements

```bash
pip install -r requirements.txt
```

```
matplotlib            3.10.9
numpy                 2.4.4
opencv-contrib-python 4.13.0.92
scipy                 1.17.1
```

> **Important:** `opencv-contrib-python` is required — not plain `opencv-python`. AGAST and FREAK are part of OpenCV's `xfeatures2d` contrib module and are not included in the standard package.

---

## Usage

```bash
python reconstruct.py --img1 <image1> --img2 <image2> --K <fx fy cx cy> [options]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--img1` | Yes | — | Path to the first (left) image |
| `--img2` | Yes | — | Path to the second (right) image |
| `--K` | Yes | — | Camera intrinsics: `fx fy cx cy` in pixels |
| `--ransac-thresh` | No | `1.0` | RANSAC reprojection threshold in pixels |
| `--outlier-thresh` | No | `2.0` | Z-score threshold for outlier removal (0 = disabled) |
| `--save` | No | — | Save final 3D point cloud to a `.npy` file |
| `--plot` | No | — | Show interactive 3D scatter plot after reconstruction |
| `--save-plot` | No | — | Save plot to a `.png` file (use on headless systems) |
| `--debug` | No | — | Enable verbose DEBUG-level logging |

### Examples

Basic reconstruction with interactive plot:
```bash
python reconstruct.py --img1 images/viff.000.ppm --img2 images/viff.006.ppm \
    --K 2360 2360 384 288 --plot
```

Save point cloud and plot to disk (no display):
```bash
python reconstruct.py --img1 images/viff.000.ppm --img2 images/viff.006.ppm \
    --K 2360 2360 384 288 --save cloud.npy --save-plot cloud.png
```

Stricter outlier filtering with a tighter RANSAC threshold:
```bash
python reconstruct.py --img1 images/viff.000.ppm --img2 images/viff.006.ppm \
    --K 2360 2360 384 288 --ransac-thresh 0.5 --outlier-thresh 1.5 --save cloud.npy
```

---

## Expected Output

The script logs a detailed trace to standard output at each stage of the pipeline:

```
[INFO] Intrinsic matrix K: ...
[INFO] Loading images ...
[INFO]   image 1: 432 keypoints | image 2: 398 keypoints
[INFO]   284 raw matches found
[INFO] Estimating Essential Matrix with RANSAC (threshold=1.00 px) ...
[INFO]   201 inliers after RANSAC (70.8%)
[INFO]   Pose recovered — rotation angle: 12.34 deg
[INFO] Triangulating 201 point pairs ...
[INFO] Pre-BA filter: 201 -> 188 points
[INFO] Running Levenberg-Marquardt bundle adjustment ...
[INFO]   Optimization converged | final cost: 0.0312
[INFO] Reconstruction complete -- 188 3-D points.
```
