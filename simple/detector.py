import numpy as np
import cv2

img_path = "/home/andrea/Downloads/computer vision project/simple/images/viff.000.ppm"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
import numpy as np

def apply_convolution(image, kernel): #for sobel
    k_height, k_width = kernel.shape
    img_height, img_width = image.shape
    
    pad_y = k_height // 2
    pad_x = k_width // 2
    
    padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    output = np.zeros((img_height, img_width), dtype=np.float64)
    
    for y in range(img_height):
        for x in range(img_width):
            region = padded_image[y:y+k_height, x:x+k_width]
            output[y, x] = np.sum(region * kernel)
            
    return output

def compute_sobel_gradients(image): #for the gradients of the images
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

def create_gaussian_kernel(size, sigma=1.0):
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    
    for y in range(size):
        for x in range(size):
            diff_y = y - center
            diff_x = x - center
            kernel[y, x] = np.exp(-(diff_x**2 + diff_y**2) / (2 * sigma**2))
            
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_box_filter(image, size):
    # Uniform kernel of ones — equivalent to cv2.boxFilter with normalize=False
    kernel = np.ones((size, size), dtype=np.float64)
    return apply_convolution(image, kernel)

def harris_corner_detector(img, k=0.04):
    
    # Gaussian blur using custom kernel + convolution
    gaussian_kernel = create_gaussian_kernel(3, sigma=1.0)
    img = apply_convolution(img.astype(np.float64), gaussian_kernel)

    # Sobel gradients — float64, so we don't lose negative gradient values
    # instead of using gaussian derivatives, I use sobel
    ix, iy = compute_sobel_gradients(img)
    
    window_size = 3 #3x3 window

    m11 = apply_box_filter(ix*ix, window_size) #structure tensor
    m22 = apply_box_filter(iy*iy, window_size)
    m12 = apply_box_filter(ix*iy, window_size)

    #determinant = m11m22 - m12^2
    det = (m11 * m22) - (m12 ** 2)

    #trace = m11+ m22
    tr = m11 + m22


    R= det - (k * (tr**2))

    # A common heuristic is 1% or 0.1% of the maximum R value
    trd = 0.1 * R.max()

    #now for the hard part, non maximal suppression
    
    height, width = R.shape
    offset = window_size // 2
    nmsOutput = np.zeros_like(R) #same matrix as R, but only zeros (initially)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            #local window
            window = R[y-offset:y+offset+1, x-offset:x+offset+1]
            
            center_value = R[y, x]
            
            #check if current pixel is the max in the window and above threshold
            if center_value > trd and center_value == np.max(window):
                nmsOutput[y, x] = center_value
    return nmsOutput


import matplotlib.pyplot as plt
nms_output = harris_corner_detector(img)

y_coords, x_coords = np.where(nms_output > 0)

vis_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

for x, y in zip(x_coords, y_coords):
    cv2.circle(vis_img, (x, y), 3, [0, 255, 0], -1)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.axis('off') # Hide axes

# THIS IS THE KEY CHANGE
# Swap plt.show() for plt.savefig()
plt.savefig('detected_corners.png', bbox_inches='tight', pad_inches=0)
print("Image saved as detected_corners.png")

# Optional: Close the plot to free memory if processing many images
plt.close()