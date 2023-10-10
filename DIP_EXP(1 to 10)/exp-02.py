import cv2
import numpy as np
import skimage.io as io
import skimage.exposure as ex
import matplotlib.pyplot as plt
from skimage import color

# Grayscale image
img1 = io.imread('cameraman.tif')
eq_img1 = cv2.equalizeHist(img1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title('Grayscale Image')
plt.subplot(1, 3, 2)
plt.hist(img1.flatten(), bins=256)
plt.title('Histogram')
plt.subplot(1, 3, 3)
plt.hist(eq_img1.flatten(), bins=256)
plt.title('Equalized Histogram')
plt.tight_layout()
plt.show()

# RGB image
img2 = io.imread('color.tiff')
# Separate RGB channels
r_channel = img2[:, :, 0]
g_channel = img2[:, :, 1]
b_channel = img2[:, :, 2]

plt.figure(figsize=(20, 5))
plt.subplot(1, 5, 1)
plt.imshow(img2)
plt.title('RGB Image')
plt.subplot(1, 5, 2)
plt.hist(r_channel.flatten(), bins=256, color='red')
plt.title("R Channel Histogram")
plt.subplot(1, 5, 3)
plt.hist(g_channel.flatten(), bins=256, color='green')
plt.title("G Channel Histogram")
plt.subplot(1, 5, 4)
plt.hist(b_channel.flatten(), bins=256, color='blue')
plt.title("B Channel Histogram")

# Histogram equalization of the RGB image
Ihsv = color.rgb2hsv(img2)
V = ex.equalize_hist(Ihsv[:, :, 2])
Ihsv[:, :, 2] = V
eq_img2 = color.hsv2rgb(Ihsv)

plt.subplot(1, 5, 5)
plt.hist(eq_img2.flatten(), bins=256)
plt.title('Equalized Histogram')
plt.tight_layout()
plt.show()
