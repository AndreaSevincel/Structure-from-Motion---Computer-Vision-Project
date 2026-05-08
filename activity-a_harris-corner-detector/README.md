# Computer Vision Final Project
**Foundational Feature Extraction — Harris Corner Detector**
*Andrea, May 2026*

---

## Overview

This project implements the **Harris Corner Detector** from scratch using Python and NumPy, as part of the Computer Vision final exam. The goal is to demonstrate a rigorous understanding of continuous mathematics translated into discrete array operations, without relying on high-level filtering abstractions.

---

## Project Structure

```
computer-vision-project/
├── detector.py               # Main Harris Corner Detector implementation
├── detected_corners.png      # Output image with detected corners visualized
├── images  
└── README.md
```

---

## Algorithm: Harris Corner Detector

The Harris Corner Detector identifies distinctive keypoints in an image by analyzing local intensity changes using the **structure tensor** M:

```
M = Σ w(x,y) * [ Ix²    IxIy ]
                [ IxIy   Iy²  ]
```

The **Harris response** R is computed as:

```
R = det(M) - k * trace(M)²
```

where `k = 0.04` (empirical constant), `det(M) = λ₁λ₂`, and `trace(M) = λ₁ + λ₂`.

### Implementation Steps

1. **Gaussian smoothing** — Pre-blur the image using a custom Gaussian kernel to reduce noise.
2. **Sobel gradients** — Compute `Ix` and `Iy` using a custom 2D convolution engine with Sobel operators.
3. **Structure tensor** — Build the M matrix using a custom box filter over the configured window.
4. **Response map R** — Compute the Harris response at each pixel.
5. **Thresholding** — Keep only responses above `threshold × max(R)`.
6. **Non-Maximal Suppression (NMS)** — Manually iterate over the response map to retain only local maxima, isolating exact corner coordinates.

> All filtering steps (Gaussian blur, Sobel, box filter) are implemented from scratch using `apply_convolution` — no `cv2` filtering functions are used inside the pipeline.

---

## Requirements

- Python 3.x
- NumPy
- OpenCV (`cv2`)
- Matplotlib

Install dependencies with:

```bash
pip install numpy opencv-python matplotlib
```

---

## Usage

Run the detector as a command-line script:

```bash
python detector.py --image <path_to_image> [options]
```

### Arguments

| Argument      | Required | Default                | Description                                        |
|---------------|----------|------------------------|----------------------------------------------------|
| `--image`     | Yes      | —                      | Path to the input image (e.g. images/viff.000.ppm) |
| `--k`         | No       | 0.04                   | Harris sensitivity constant                        |
| `--threshold` | No       | 0.1                    | NMS threshold as a fraction of max R               |
| `--window`    | No       | 3                      | Window size for structure tensor and NMS           |
| `--output`    | No       | detected_corners.png   | Path for the output image                          |

### Examples

Basic usage:
```bash
python detector.py --image images/viff.000.ppm
```

Custom sensitivity and output path:
```bash
python detector.py --image images/viff.000.ppm --k 0.06 --output results/corners.png
```

Stricter threshold with a larger window:
```bash
python detector.py --image images/viff.000.ppm --threshold 0.05 --window 5
```

---

## Output

The script prints the number of detected corners and saves the result image to the path specified by `--output`. Corners are drawn as green filled circles (radius 3px) over the grayscale input image.

```
Corners detected: 142
Output saved to: detected_corners.png
```
