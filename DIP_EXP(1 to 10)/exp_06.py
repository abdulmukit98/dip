import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.util.noise as noise
import skimage.util as util
import scipy.ndimage as ndi
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import skimage.exposure as ex

# Read the input image
img = io.imread('5.3.01.tiff')

# Add salt and pepper noise to the image
sp = noise.random_noise(img, mode='s&p', amount=0.3)

# Add periodic noise
r, c = img.shape
x, y = np.mgrid[0:r, 0:c].astype('float32')
p = np.sin(x / 3 + y / 3) + 1.0
gp = (2 * util.img_as_float(img) + p / 2) / 3

# Display images with salt and pepper noise and periodic noise
plt.subplot(1, 2, 1)
plt.title('Image with salt and pepper noise')
plt.imshow(sp, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Image with Periodic Noise')
plt.imshow(gp, cmap='gray')
plt.show()

# Apply median filtering to remove salt and pepper noise
img_med = ndi.median_filter(sp, 3)

# Apply the band-reject filter to the image with periodic noise
gf=fftshift(fft2(gp))
mg_spectrum=np.abs(gf)
temp= ex.rescale_intensity(abs(gf),out_range=(0,1))
gf2=util.img_as_ubyte(temp)
gf2[512,512]=0  #center of dft
i,j=np.where(gf2==gf2.max())
z=np.sqrt((x-512)**2+(y-512)**2)
k=1 
d=np.sqrt(5832)
br=(z<np.floor(d-k))|(z>np.ceil(d+k))
gfr=gf*br
filtered_gp = np.real(ifft2(ifftshift(gfr)))

# Display the images with removed salt and pepper noise and periodic noise
plt.subplot(1, 2, 1)
plt.title('Removing salt and pepper noise')
plt.imshow(img_med, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Removing Periodic Noise')
plt.imshow(filtered_gp, cmap='gray')
plt.show()
