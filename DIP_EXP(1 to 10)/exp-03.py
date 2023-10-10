import numpy as np
import skimage.io as io
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Load the image
img01 = io.imread('cameraman.tif')

# Apply Gaussian filtering
img02 = ndi.gaussian_filter(img01, 1, truncate=1)

# Traditional Laplacian filter
laplacian_filter = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply Laplacian filtering
img03 = ndi.convolve(img01, laplacian_filter, mode='constant')

# Plot the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img01, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(img02, cmap='gray')
plt.title('Gaussian Filtered Image')

plt.subplot(1, 3, 3)
plt.imshow(img03, cmap='gray')
plt.title('Laplacian Filtered Image')

plt.tight_layout()
plt.show()
