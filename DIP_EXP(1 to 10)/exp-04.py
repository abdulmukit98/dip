import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import pywt

img = io.imread('cameraman.tif')

# Perform FFT of grayscale image
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
dft_result = 20 * np.log(np.abs(dft_shift))

# Plot original image and its DFT
plt.subplot(1,2,1)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img,cmap='gray')
plt.subplot(1,2,2)
plt.axis('off')
plt.title('FFT of the Image')
plt.imshow(dft_result,cmap='gray')

# Perform DWT of grayscale image
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

# Display the original image and the DWT components
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(cA, cmap='gray')
plt.title('Approximation (LL)')
plt.subplot(2, 2, 3)
plt.imshow(cH, cmap='gray')
plt.title('Horizontal Detail (LH)')
plt.subplot(2, 2, 4)
plt.imshow(cV, cmap='gray')
plt.title('Vertical Detail (HL)')

plt.tight_layout()
plt.show()

