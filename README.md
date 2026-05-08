# Sparse 3D Reconstruction: A Two-View Framework

This repository contains a complete Structure from Motion (SfM) pipeline that estimates a sparse 3D point cloud from a stereoscopic image pair. This project was developed as the final evaluation for the Computer Vision course under Prof. Luca Lombardi.

## Project Overview

The application is a command-line tool that sequentially processes two 2D images to recover their underlying 3D geometry. The mathematical pipeline consists of:

* **Feature Detection:** AGAST (Adaptive and Generic Accelerated Segment Test)
* **Feature Description:** FREAK (Fast Retina Keypoint)
* **Matching:** Brute-Force Matcher utilizing Hamming distance with cross-checking
* **Robust Pose Estimation:** Essential Matrix estimation via RANSAC
* **3D Triangulation:** Linear algebraic triangulation using Singular Value Decomposition (SVD)
* **Non-Linear Optimization:** Levenberg-Marquardt Bundle Adjustment minimizing reprojection error

## Environment Setup

The pipeline relies on standard scientific computing and computer vision libraries. Ensure your environment is set up with the required dependencies.

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```text
matplotlib            3.10.9
numpy                 2.4.4
opencv-contrib-python 4.13.0.92
scipy                 1.17.1
```

## Command-Line Usage

The program is executed via the command line and accepts several arguments to define the input data and execution parameters. 

```bash
python reconstruct.py --img1 images/(IMAGE_NAME1) --img2 images/(IMAGE_NAME2) --K 3217 2292 289 288 --outlier-thresh 2.5 --save (PC_NAME).npy
```

**Arguments:**
* `--img1`: Path to the first image.
* `--img2`: Path to the second image.
* `--K`: Path to the text file containing the $3 \times 3$ intrinsic camera matrix.
* `--plot`: Optional flag to render an interactive 3D Matplotlib plot upon completion.
* `--save`: Optional flag to output the final 3D point cloud as a `.npy` file.

## Expected Output

During execution, the script generates a detailed logging trace to standard output. This trace details the funneling of data through the pipeline, including:

1.  The initial count of detected AGAST keypoints for each view.
2.  The number of raw FREAK matches.
3.  The strict inlier count and survival percentage following RANSAC filtration.
4.  The recovered rotation angle between the two camera views.
5.  The final geometric cost (reprojection error) after the Levenberg-Marquardt optimization converges.
