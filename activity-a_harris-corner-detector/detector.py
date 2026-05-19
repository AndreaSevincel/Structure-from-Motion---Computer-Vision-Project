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
    #2d convolution manual using NumPy
    k_height, k_width = kernel.shape #filter dimensions
    img_height, img_width = image.shape

    pad_y = k_height // 2 #radius of kernel
    pad_x = k_width // 2

    padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect') #np.pad adds extra pixels around the border, using mode='reflect' means the border pixels are mirrored outward, preventing harsh artificial edges.
    output = np.zeros((img_height, img_width), dtype=np.float64) #store the results here

    for y in range(img_height):
        for x in range(img_width):
            region = padded_image[y:y + k_height, x:x + k_width] #for each coordinate, it slices out a sub-matrix from the padded_image that matches the exact shape of the kernel, centered over the current pixel.
            output[y, x] = np.sum(region * kernel) #region * kernel element-wise multiplication. The top-left pixel of the region multiplies by the top-left pixel of the kernel, and so on.
            #np.sum adds all those multiplied values together into a single number.
    return output


def create_gaussian_kernel(size, sigma=1.0):
    #build a normalised gaussian kernel of the given size and predefined sigma = 1.0
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)

    for y in range(size):
        for x in range(size):
            diff_y = y - center
            diff_x = x - center
            kernel[y, x] = np.exp(-(diff_x ** 2 + diff_y ** 2) / (2 * sigma ** 2)) #gauss

    return kernel / np.sum(kernel)


def compute_sobel_gradients(image):
    #Compute Ix and Iy using custom Sobel kernels
    sobel_x = np.array([ #horizontal gradient, does not look at the pixel itself, since there is 0 in the center. Vertical edges
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_y = np.array([ #horizontal edges
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    grad_x = apply_convolution(image, sobel_x)
    grad_y = apply_convolution(image, sobel_y)

    return grad_x, grad_y #convoluted images


def apply_box_filter(image, size):
    #uniform box filter — equivalent to cv2.boxFilter with normalize=False
    kernel = np.ones((size, size), dtype=np.float64)
    return apply_convolution(image, kernel) #the easiest convolution apart from the identity, I guess


def harris_corner_detector(img, k=0.04, threshold=0.1, window_size=3):
    #gaussian smoothing
    gaussian_kernel = create_gaussian_kernel(3, sigma=1.0)
    img = apply_convolution(img.astype(np.float64), gaussian_kernel)

    #sobel gradients (float64 preserves negative values)
    ix, iy = compute_sobel_gradients(img) #"directional derivatives"

    # structure tensor elements via box filtering
    m11 = apply_box_filter(ix * ix, window_size)
    m22 = apply_box_filter(iy * iy, window_size)
    m12 = apply_box_filter(ix * iy, window_size)
    #for each pixel, Harris builds a matrix called the Structure Tensor M, which captures the distribution of gradients in the local neighborhood
    #M = [m11, m12]
    #    [m12, m21]
    #we dont want averages, we want the total sum of gradient energies within that window_size neighborhood.

    #harris response R = det(M) - k * trace(M)²
    det = (m11 * m22) - (m12 ** 2)
    tr  = m11 + m22
    R   = det - (k * (tr ** 2))

    #non-Maximal suppression
    trd = threshold * R.max() #standard to use 0.1 * max(R) (what i have read, at least)
    height, width = R.shape
    offset = window_size // 2 #if window 3x3, offset 1
    nmsOutput = np.zeros_like(R)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window = R[y - offset:y + offset + 1, x - offset:x + offset + 1] #standard Python slice
            center_value = R[y, x]
            if center_value > trd and center_value == np.max(window): #2 conditions, pixel value must be larger than its neighborhood, pixel value must be larger than trd
                nmsOutput[y, x] = center_value
    #since the default matrix is 0, if pixel doesnt meet the condition it gets to 0
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
#np.where(nms_output > 0) scans the non-maximal suppression matrix, looks for any pixel that survived (any value greater than 0) and gets its coordinates
print(f"Corners detected: {len(x_coords)}")

vis_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for x, y in zip(x_coords, y_coords):
    cv2.circle(vis_img, (x, y), 3, [0, 255, 0], -1) #[0, 255, 0] represents pure Green in BGR color space
    #-1 to fill the circle completely, painting a solid green dot with a radius of 3 pixels right on top of each detected corner.

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Output saved to: {args.output}")
