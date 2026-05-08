# Computer Vision — Course Projects
**Prof. Luca Lombardi**
*Andrea, May 2026*

---

## Repository Overview

This repository contains the two projects developed for the Computer Vision course final evaluation. Both are command-line tools that implement concepts from the course, ranging from foundational image processing to full 3D scene reconstruction.

```
computer-vision-projects/
│
├── activity-a_harris-corner-detector/
│   ├── detector.py               # Harris Corner Detector (from scratch)
│   ├── detected_corners.png      # Sample output
│   ├── README.md
│   └── images
│
└── activity-b_3d-reconstruction/
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
```

---

## Activity A — Foundational Algorithm: Harris Corner Detector

### Goal

Implement a classic computer vision algorithm **from scratch**, using only standard Python and NumPy — no high-level OpenCV filtering calls.

### What it does

Given a grayscale image, the script detects corner keypoints by analyzing local intensity changes using the Harris response function. Detected corners are visualized as green dots overlaid on the original image.

### Pipeline

| Step | Description |
|------|-------------|
| Gaussian smoothing | Custom kernel + manual 2D convolution to reduce noise |
| Sobel gradients | Custom Sobel kernels applied via `apply_convolution` to compute Ix and Iy |
| Structure tensor | Box filtering of Ix², Iy², IxIy over a local window |
| Harris response | `R = det(M) - k * trace(M)²`, with k = 0.04 |
| Thresholding | Keeps pixels above `threshold × max(R)` |
| NMS | Manual local-maxima extraction to isolate exact corner coordinates |

### Usage

```bash
python detector.py --image <path_to_image> [options]
```

| Argument | Default | Description |
|---|---|---|
| `--image` | required | Path to input image |
| `--k` | `0.04` | Harris sensitivity constant |
| `--threshold` | `0.1` | NMS threshold as fraction of max R |
| `--window` | `3` | Window size for structure tensor and NMS |
| `--output` | `detected_corners.png` | Output image path |

**Examples:**
```bash
# Basic
python detector.py --image images/viff.000.ppm

# Tuned sensitivity
python detector.py --image images/viff.000.ppm --k 0.06 --threshold 0.05 --output results/corners.png
```

### Dependencies

```bash
pip install numpy opencv-python matplotlib
```

---

## Activity B — Complex Problem: Sparse 3D Reconstruction

### Goal

Solve a real computer vision problem using standard libraries (OpenCV, SciPy). Starting from a stereo image pair, estimate a sparse **3D point cloud** of the scene using a full Structure from Motion (SfM) pipeline.

### What it does

The script takes two images of the same scene from slightly different viewpoints and reconstructs the 3D geometry by detecting and matching features, estimating camera pose, triangulating points in 3D, and refining them with bundle adjustment.

### Pipeline

| Step | Tool | Description |
|------|------|-------------|
| Feature detection | AGAST | Detects corner keypoints in each image |
| Feature description | FREAK | Encodes keypoints as binary strings |
| Matching | BFMatcher + Hamming | Finds best cross-checked pairs between the two views |
| Pose estimation | Essential Matrix + RANSAC | Robustly estimates relative camera rotation R and translation t |
| Triangulation | SVD | Lifts 2D correspondences to 3D points using projection matrices |
| Outlier filtering | Z-score | Removes points beyond a configurable number of std deviations |
| Bundle adjustment | Levenberg-Marquardt | Non-linear refinement minimizing reprojection error |

### Usage

```bash
python reconstruct.py --img1 <image1> --img2 <image2> --K <fx fy cx cy> [options]
```

| Argument | Default | Description |
|---|---|---|
| `--img1` | required | Path to first (left) image |
| `--img2` | required | Path to second (right) image |
| `--K` | required | Camera intrinsics: `fx fy cx cy` in pixels |
| `--ransac-thresh` | `1.0` | RANSAC reprojection threshold in pixels |
| `--outlier-thresh` | `2.0` | Max std deviations from centroid (0 = disabled) |
| `--save` | — | Save 3D point cloud to `.npy` file |
| `--plot` | — | Show interactive 3D scatter plot |
| `--save-plot` | — | Save plot to `.png` (for headless systems) |
| `--debug` | — | Enable verbose DEBUG-level logging |

**Examples:**
```bash
# Basic reconstruction with interactive plot
python reconstruct.py --img1 images/viff.000.ppm --img2 images/viff.006.ppm \
    --K 2360 2360 384 288 --plot

# Save point cloud and plot, no display
python reconstruct.py --img1 images/viff.000.ppm --img2 images/viff.006.ppm \
    --K 2360 2360 384 288 --save cloud.npy --save-plot cloud.png
```

### Dependencies

```bash
pip install -r requirements.txt
```

```
matplotlib            3.10.9
numpy                 2.4.4
opencv-contrib-python 4.13.0.92
scipy                 1.17.1
```

> **Note:** `opencv-contrib-python` is required (not plain `opencv-python`) because AGAST and FREAK live in OpenCV's `xfeatures2d` contrib module.

---

## Course Info

| | |
|---|---|
| **Course** | Computer Vision |
| **Professor** | Luca Lombardi |
| **Academic Year** | 2025/2026 |
| **Student** | Andrea |
