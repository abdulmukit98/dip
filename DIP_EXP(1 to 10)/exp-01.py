import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

gray_img = io.imread('cameraman.tif')
color_img = io.imread('color.tiff')
indexed_img = io.imread('indexed.png')

plt.subplot(1,3,1)
plt.imshow(gray_img,cmap='gray')
plt.title('Grayscale image')
plt.subplot(1,3,2)
plt.imshow(color_img)
plt.title('Color image')
plt.subplot(1,3,3)
plt.imshow(indexed_img)
plt.title('Indexed image')
plt.show()
