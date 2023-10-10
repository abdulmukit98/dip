import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters 

img1 = io.imread('cameraman.tif')

# Get image dimensions
r, c = img1.shape

# Create the x, y grid
x, y = np.mgrid[0:r, 0:c].astype(float)

# Calculate the adaptive thresholding value
p2 = 255.0 - img1 + y / 2

plt.subplot(1,3,1)
plt.imshow((img1>40),cmap='gray')
plt.title('Single thresholding')
plt.subplot(1,3,2)
plt.imshow(((img1>40) & (img1<80)),cmap='gray')
plt.title('Double thresholding')
plt.subplot(1,3,3)
plt.imshow(p2,cmap='gray')
plt.title('Adaptive thresholding')
plt.show()


# edge detection
#prewitt filter
edge_p = filters.prewitt(img1)
#roberts filter
edge_r = filters.roberts(img1)
#sobel filter
edge_s = filters.sobel(img1)

plt.subplot(1,3,1)
plt.imshow(edge_p,cmap='gray')
plt.title('Prewitt edge detection')
plt.subplot(1,3,2)
plt.imshow(edge_r,cmap='gray')
plt.title('Roberts edge detection')
plt.subplot(1,3,3)
plt.imshow(edge_s,cmap='gray')
plt.title('Sobel edge detection')
plt.show()


