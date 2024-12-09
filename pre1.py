import numpy as np
import cv2
from skimage import io, img_as_float
from skimage.restoration import denoise_bilateral
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import disk
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import os

# Load SAR Image
sar_image_path = '/Users/risapandey/Desktop/College/sih/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png'
sar_image = io.imread(sar_image_path)

# Convert SAR Image to Float
sar_image_float = img_as_float(sar_image)

# Normalize the image to [0, 1] range
sar_image_normalized = rescale_intensity(sar_image_float, in_range='image', out_range=(0, 1))

# Apply Bilateral Filter for Denoising
sar_image_denoised = denoise_bilateral(sar_image_normalized, sigma_color=0.05, sigma_spatial=15)

# Apply Gaussian Blur
sar_image_blurred = gaussian(sar_image_denoised, sigma=1)

# Apply Histogram Equalization (CLAHE)
sar_image_equalized = equalize_adapthist(sar_image_blurred, clip_limit=0.03)

# Apply Fourier Transform
f_transform = np.fft.fft2(sar_image_equalized)
f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency component to center
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Apply High-Pass Filter
rows, cols = sar_image_equalized.shape
crow, ccol = rows // 2 , cols // 2   # Center of the image
mask = np.ones((rows, cols), np.uint8)
r = 30  # Radius of the mask; tweak this value
mask[crow-r:crow+r, ccol-r:ccol+r] = 0
f_transform_shifted = f_transform_shifted * mask
f_ishift = np.fft.ifftshift(f_transform_shifted)
sar_image_filtered = np.fft.ifft2(f_ishift)
sar_image_filtered = np.abs(sar_image_filtered)

# Apply Edge Detection (Canny)
edges = cv2.Canny((sar_image_filtered * 255).astype(np.uint8), 100, 200)

# Apply Morphological Operations (Dilation)
kernel = np.ones((3,3), np.uint8)
sar_image_morph = cv2.dilate(edges, kernel, iterations=1)

# Define the directory and file path for saving
output_directory = '/Users/risapandey/Desktop/College/sih/v_2'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
preprocessed_image_path = os.path.join(output_directory, 'preprocessed_image_fourier1.tiff')

# Save the preprocessed image
io.imsave(preprocessed_image_path, sar_image_morph)

# Display the images for comparison (Optional)
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title('Original SAR Image')
plt.imshow(sar_image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Denoised SAR Image')
plt.imshow(sar_image_denoised, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Gaussian Blurred Image')
plt.imshow(sar_image_blurred, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Histogram Equalized Image')
plt.imshow(sar_image_equalized, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Magnitude Spectrum (Fourier Transform)')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(2, 3, 6)
plt.title('Final Processed Image')
plt.imshow(sar_image_morph, cmap='gray')

plt.show()
