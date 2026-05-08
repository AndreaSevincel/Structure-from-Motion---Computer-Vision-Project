import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description="Harris Corner Detector — implemented from scratch with NumPy."
)
parser.add_argument(
    "--image",
    required=True,
    help="Path to the input image (e.g. images/viff.000.ppm)"
)
parser.add_argument(
    "--k",
    type=float,
    default=0.04,
    help="Harris sensitivity constant k (default: 0.04)"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.1,
    help="NMS threshold as a fraction of the maximum R value (default: 0.1)"
)
parser.add_argument(
    "--window",
    type=int,
    default=3,
    help="Window size for structure tensor and NMS (default: 3)"
)
parser.add_argument(
    "--output",
    type=str,
    default="detected_corners.png",
    help="Path for the output image (default: detected_corners.png)"
)
args = parser.parse_args()

def apply_convolution(image, kernel):
    #2D convolution manually using NumPy (no cv2 filter calls)
    k_height, k_width = kernel.shape
    img_height, img_width = image.shape

    pad_y = k_height // 2
    pad_x = k_width // 2

    padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    output = np.zeros((img_height, img_width), dtype=np.float64)

    for y in range(img_height):
        for x in range(img_width):
            region = padded_image[y:y + k_height, x:x + k_width]
            output[y, x] = np.sum(region * kernel)

    return output


def create_gaussian_kernel(size, sigma=1.0):
    #Build a normalised Gaussian kernel of the given size and sigma
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)

    for y in range(size):
        for x in range(size):
            diff_y = y - center
            diff_x = x - center
            kernel[y, x] = np.exp(-(diff_x ** 2 + diff_y ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)


def compute_sobel_gradients(image):
    #Compute Ix and Iy using custom Sobel kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    grad_x = apply_convolution(image, sobel_x)
    grad_y = apply_convolution(image, sobel_y)

    return grad_x, grad_y


def apply_box_filter(image, size):
    #uniform box filter — equivalent to cv2.boxFilter with normalize=False
    kernel = np.ones((size, size), dtype=np.float64)
    return apply_convolution(image, kernel)


def harris_corner_detector(img, k=0.04, threshold=0.1, window_size=3):
    #gaussian smoothing
    gaussian_kernel = create_gaussian_kernel(3, sigma=1.0)
    img = apply_convolution(img.astype(np.float64), gaussian_kernel)

    #sobel gradients (float64 preserves negative values)
    ix, iy = compute_sobel_gradients(img)

    # structure tensor elements via box filtering
    m11 = apply_box_filter(ix * ix, window_size)
    m22 = apply_box_filter(iy * iy, window_size)
    m12 = apply_box_filter(ix * iy, window_size)

    #harris response R = det(M) - k * trace(M)²
    det = (m11 * m22) - (m12 ** 2)
    tr  = m11 + m22
    R   = det - (k * (tr ** 2))

    #non-Maximal suppression
    trd = threshold * R.max()
    height, width = R.shape
    offset = window_size // 2
    nmsOutput = np.zeros_like(R)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window = R[y - offset:y + offset + 1, x - offset:x + offset + 1]
            center_value = R[y, x]
            if center_value > trd and center_value == np.max(window):
                nmsOutput[y, x] = center_value

    return nmsOutput


img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read image: {args.image}")

nms_output = harris_corner_detector(
    img,
    k=args.k,
    threshold=args.threshold,
    window_size=args.window
)

y_coords, x_coords = np.where(nms_output > 0)
print(f"Corners detected: {len(x_coords)}")

vis_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for x, y in zip(x_coords, y_coords):
    cv2.circle(vis_img, (x, y), 3, [0, 255, 0], -1)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Output saved to: {args.output}")
