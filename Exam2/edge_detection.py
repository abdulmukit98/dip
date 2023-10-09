import skimage.filters as filters
import skimage.io as io
import matplotlib.pyplot as plt

img = io.imread('D:\\Dataset\\5.1.12.tiff')

plt.subplot(1,3,1)
io.imshow(img)
plt.title('original')

edge_sobel = filters.sobel(img)

plt.subplot(1,3,2)
io.imshow(edge_sobel)
plt.title('Sobel Filtered image')

plt.subplot(1,3,3)
edge_prewitt = filters.prewitt(img)
io.imshow(edge_prewitt)
plt.title('prewitt filtered image')

io.show()
