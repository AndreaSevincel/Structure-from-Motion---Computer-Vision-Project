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
└── README.md
```

---

## Algorithm: Harris Corner Detector

The Harris Corner Detector identifies distinctive keypoints in an image by analyzing local intensity changes using the **structure tensor** M:

$$M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

The **Harris response** R is computed as:

```
R = det(M) - k * trace(M)²
```

where `k = 0.04` (empirical constant), `det(M) = λ₁λ₂`, and `trace(M) = λ₁ + λ₂`.

### Implementation Steps

1. **Gaussian smoothing** — Pre-blur the image to reduce noise.
2. **Sobel gradients** — Compute `Ix` and `Iy` using Sobel operators (custom convolution engine included).
3. **Structure tensor** — Build the M matrix using box filtering over a 3×3 window.
4. **Response map R** — Compute the Harris response at each pixel.
5. **Thresholding** — Keep only responses above 10% of the maximum R value.
6. **Non-Maximal Suppression (NMS)** — Manually iterate over the response map to retain only local maxima, isolating exact corner coordinates.

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

Run the detector from the command line:

```bash
python detector.py
```

By default, the script reads the image from the hardcoded path at the top of `detector.py`. To use a different image, edit the `img_path` variable:

```python
img_path = "/path/to/your/image.ppm"
```

The output is saved as `detected_corners.png` in the current working directory — a copy of the input image with detected corners marked in green.

---

## Output

Detected corners are drawn as green filled circles (radius 3px) over a grayscale version of the input image and saved to `detected_corners.png`.

---

## Notes

- The custom `apply_convolution` and `create_gaussian_kernel` functions are included for reference and to satisfy the requirement of avoiding high-level filtering calls in the foundational implementation.
- The `harris_corner_detector` function uses OpenCV utilities (`GaussianBlur`, `Sobel`, `boxFilter`) for efficiency in the final version, while the manual convolution engine remains available.
- No external packages beyond those listed above are required.
