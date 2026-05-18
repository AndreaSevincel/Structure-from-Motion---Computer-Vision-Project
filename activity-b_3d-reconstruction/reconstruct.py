#!/usr/bin/env python3
"""
Sparse 3D Reconstruction CLI
Two-view Structure from Motion using AGAST + FREAK + LM Bundle Adjustment.

Usage:
    python reconstruct.py --img1 viff.000.ppm --img2 viff.006.ppm \
        --K 2360 2360 384 288 --save cloud.npy --save-plot cloud.png
"""

import argparse
import logging
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#full transcparency, log EVERYTHING
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def extract_and_match(img1: np.ndarray, img2: np.ndarray): #images as n-dimensional arrays
    detector = cv2.AgastFeatureDetector_create(threshold=5) #imported the agast feature detector... freak as well
    descriptor = cv2.xfeatures2d.FREAK_create() 

    kp1 = detector.detect(img1, None) #detecting points of interest
    kp2 = detector.detect(img2, None)
    log.info("  image 1: %d keypoints | image 2: %d keypoints", len(kp1), len(kp2)) 

    kp1, des1 = descriptor.compute(img1, kp1) #freak describing the points of interest
    kp2, des2 = descriptor.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #hamming distance since we use FREAK, binary strings. BFMatcher = bruteforce matcher
    matches = bf.match(des1, des2) #finding the best pairs
    matches = sorted(matches, key=lambda x: x.distance) #ranking by similarity
    log.info("  %d raw matches found", len(matches))

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]) #m containts indices, queryIdx is the point's row in Img 1
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]) #trainIdx points row in Img 2
    return pts1, pts2 
    #loop through the matches, looks up the (x, y) 
    # coordinate for each index in the original keypoint lists (kp1, kp2), and stores them as np array


def estimate_pose(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, ransac_thresh: float):
    #estimate essential matrix via RANSAC and recover relative pose (R, t).
    log.info("Estimating Essential Matrix with RANSAC (threshold=%.2f px) ...", ransac_thresh)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, ransac_thresh) #E=essential matrix, mask=column vector, 1 inlier 0 outlier

    inlier_mask = mask.ravel() == 1 #converting into True/False values
    pts1_in = pts1[inlier_mask] #keeping only inliers
    pts2_in = pts2[inlier_mask]
    log.info("  %d inliers after RANSAC (%.1f%%)", inlier_mask.sum(),
             100.0 * inlier_mask.sum() / len(mask))

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K) #R=rotation t=translation, _=we do not need, placeholder to say "we ignore this part"
    log.info("  Pose recovered — rotation angle: %.2f deg",
             np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return R, t, pts1_in, pts2_in


def triangulate(pts1: np.ndarray, pts2: np.ndarray,
                R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Linear triangulation via SVD to obtain an initial 3-D point cloud."""
    log.info("Triangulating %d point pairs ...", len(pts1))
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1)))) #first camera, placed at origin, no rotation
    P2 = K @ np.hstack((R, t)) #second camera, uses the R & t we got from before. hstack=horizontal stacking of arrays

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)# two projection matrices and the matched 2D points. uses SVD to find the 3D point
    points_3d = (points_4d[:3, :] / points_4d[3, :]).T #points_4d is X, Y, Z, W
    log.info("  Cloud shape: %s", points_3d.shape)
    return points_3d


def project_points(points_3d: np.ndarray, rvec: np.ndarray,
                   tvec: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None) #rotates and translates the 3D points into the camera's local coordinate system
    return pts2d.reshape(-1, 2) #It then flattens those 3D coordinates onto a 2D plane based on the geometry defined in the k matrix
#flattened into clean nx2 array

def reprojection_residuals(params: np.ndarray, n_points: int,
                           pts1: np.ndarray, pts2: np.ndarray,
                           K: np.ndarray) -> np.ndarray:
    rvec  = params[:3] #the first 3 $,$,$,_,_,_
    tvec  = params[3:6] # _,_,_,$,$,$
    pts3d = params[6:].reshape((n_points, 3)) #3 x n_points, clever trick: could have used pts3d = params[6:].reshape((-1, 3)) if we did not know n_points, telling Numpy to figure out 
    #the number of rows we need

    proj1 = project_points(pts3d, np.zeros(3), np.zeros(3), K) #where the 3D points should appear in the first image
    proj2 = project_points(pts3d, rvec, tvec, K) #where 3D points should appear in the second image according to your current model

    return np.hstack(((proj1 - pts1).ravel(), (proj2 - pts2).ravel())) #stacking of the reprojection residuals


def bundle_adjust(pts1: np.ndarray, pts2: np.ndarray,
                  points_3d: np.ndarray, R: np.ndarray,
                  t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Non-linear refinement via Levenberg-Marquardt bundle adjustment."""
    log.info("Running Levenberg-Marquardt bundle adjustment ...")
    rvec, _ = cv2.Rodrigues(R) #rodrigues rotation, reduces the number of parameters from 9 to 3
    initial_params = np.hstack((rvec.ravel(), t.ravel(), points_3d.ravel())) #initial guess, contains the rotation vector, the translation vector, and the coordinates for every 3D point in the cloud
    n_points = points_3d.shape[0]

    res = least_squares( #least squares used as numerical solver
        reprojection_residuals,
        initial_params,
        method="lm", #the optimizer i use is the levenberg marquardt. 
        args=(n_points, pts1, pts2, K),
    )
    log.info("  Optimization %s | final cost: %.4f",
             "converged" if res.success else "did not converge", res.cost)

    return res.x[6:].reshape((n_points, 3)) #2D array containing the refined 3D coordinates of my points, [6:] since pts3d the same.


def visualize(points_3d: np.ndarray, title: str = "Sparse 3D Point Cloud",
              save_path: str = None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               s=2, c="crimson", marker=".")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved to %s", save_path)
    else:
        plt.show()



def build_parser() -> argparse.ArgumentParser: #command-line interface, and generates a --help menu
    p = argparse.ArgumentParser(
        description="Two-view sparse 3D reconstruction (AGAST + FREAK + LM BA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--img1", required=True, help="Path to first (left) image")
    p.add_argument("--img2", required=True, help="Path to second (right) image")
    p.add_argument(
        "--K", nargs=4, type=float, metavar=("fx", "fy", "cx", "cy"),
        required=True,
        help="Camera intrinsics: focal lengths and principal point in pixels",
    )
    p.add_argument("--ransac-thresh", type=float, default=1.0,
                   help="RANSAC reprojection threshold in pixels")
    p.add_argument("--outlier-thresh", type=float, default=2.0,
                   help="Outlier filter: max std deviations from centroid (0 = disabled)")
    p.add_argument("--save", metavar="FILE.npy",
                   help="Save optimised 3-D points to a NumPy .npy file")
    p.add_argument("--plot", action="store_true",
                   help="Show interactive 3-D scatter plot after reconstruction")
    p.add_argument("--save-plot", metavar="FILE.png",
                   help="Save plot to image file (use instead of --plot on headless systems)")
    p.add_argument("--debug", action="store_true",
                   help="Enable DEBUG-level logging")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # the camera parameters, put them on a camera calibration matrix.
    fx, fy, cx, cy = args.K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    #This matrix maps 3D coordinates in the camera's view to 2D pixel coordinates on the image sensor.
    log.info("Intrinsic matrix K:\n%s", K)

    # Load images
    log.info("Loading images ...")
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        log.error("Could not load one or both images -- check paths.")
        sys.exit(1)
    log.info("  img1: %s  |  img2: %s", img1.shape, img2.shape)

    pts1, pts2= extract_and_match(img1, img2) #finding distinct keypoints (like corners or distinct textures) in both images and pairing them up
    R, t, pts1_in, pts2_in = estimate_pose(pts1, pts2, K, args.ransac_thresh) #how the camera moved between the two shots
    #RANSAC to filter out the bad matches (outliers), returning only the valid, verified matching points (pts1_in, pts2_in).
    points_3d= triangulate(pts1_in, pts2_in, R, t, K) #projecting lines out from the camera positions through the 2D matching pixels.
    # The intersection of these lines in 3D space determines the (X, Y, Z) coordinates of each point.

    #RANSAC cleans up mistakes in the 2D image matching, while this cleans up geometry mistakes in the 3D space.
    if args.outlier_thresh > 0:
        mean = np.mean(points_3d, axis=0)
        std  = np.std(points_3d, axis=0)
        std[std < 1e-6] = 1e-6
        mask = np.all(np.abs(points_3d - mean) < args.outlier_thresh * std, axis=1)
        points_3d = points_3d[mask]
        pts1_in   = pts1_in[mask]
        pts2_in   = pts2_in[mask]
        log.info("Pre-BA filter: %d -> %d points", mask.size, mask.sum())

    optimised_3d= bundle_adjust(pts1_in, pts2_in, points_3d, R, t, K) #Levenberg-Marquardt solver
    log.info("Reconstruction complete -- %d 3-D points.", len(optimised_3d))

    # Save point cloud
    if args.save:
        np.save(args.save, optimised_3d)
        log.info("Point cloud saved to %s", args.save)

    # Plot
    if args.save_plot:
        visualize(optimised_3d, save_path=args.save_plot)
    elif args.plot:
        visualize(optimised_3d)


if __name__ == "__main__":
    main()
