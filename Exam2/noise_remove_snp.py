import skimage.io as io
import skimage.color as color
import matplotlib.pyplot as plt
import skimage.util.noise as noise
import scipy.ndimage as ndi

img = io.imread('D:\\Dataset\\5.1.12.tiff')

img_snp = noise.random_noise(img, mode='s&p', amount=0.05)

img_fl = ndi.median_filter(img_snp, 3)

plt.subplot(1,3,1)
io.imshow(img)
plt.title('original')

plt.subplot(1,3,2)
io.imshow(img_snp)
plt.title('Salt and Pepper noisy img')

plt.subplot(1,3,3)
io.imshow(img_fl)
plt.title('Median filtered image')

io.show()
