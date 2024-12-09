import numpy as np
import cv2
from skimage import io, img_as_float
from skimage.restoration import denoise_bilateral
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import os

# Load SAR Image
sar_image_path = '/Users/risapandey/Desktop/College/sih/v_2/agri/ROIs1868_summer_s1_59_p2.tiff'
sar_image = io.imread(sar_image_path)

# Convert SAR Image to Float
sar_image_float = img_as_float(sar_image)

# Normalize the image to [0, 1] range
sar_image_normalized = rescale_intensity(sar_image_float, in_range='image', out_range=(0, 1))

# Apply Bilateral Filter for Denoising
sar_image_denoised = denoise_bilateral(sar_image_normalized, sigma_color=0.05, sigma_spatial=15)

# Define the directory and file path for saving
output_directory = '/Users/risapandey/Desktop/College/sih/v_2'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
preprocessed_image_path = os.path.join(output_directory, 'preprocessed_image.tiff')

# Save the preprocessed image
io.imsave(preprocessed_image_path, sar_image_denoised)

# Display the images for comparison (Optional)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original SAR Image')
plt.imshow(sar_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Preprocessed SAR Image')
plt.imshow(sar_image_denoised, cmap='gray')

plt.show()
