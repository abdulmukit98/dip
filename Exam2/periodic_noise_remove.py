import skimage.io as io
import matplotlib.pyplot as plt
import skimage.util.noise as noise
import scipy.ndimage as ndi
import numpy as np
import skimage.util as util
import numpy.fft as fft

img = io.imread('D:\\Dataset\\5.1.12.tiff')
io.imshow(img)
plt.title('original')
io.show()

r, c = img.shape
x, y = np.mgrid[0:r, 0:c].astype('float32')
p = np.sin(x/3 + y/3) + 1.0

img_n = (2 * util.img_as_float(img) + p/2) /3 

img_r = fft.fftshift(fft.fft2(img_n))

#denoised = np.abs(np.fft.ifft2(img_r))
#img_fl = ndi.median_filter(img_r, 3)


io.imshow(img_n)
plt.title('Periodic noisy image')


io.show()
