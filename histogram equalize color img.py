import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import skimage.exposure as ex
import skimage.color as color

img01 = io.imread('D:\\Dataset\\4.1.04.tiff')
io.imshow(img01)
io.show()

#eq_img = ex.equalize_hist(img01)
#io.imshow(eq_img)
#io.show()


Ihsv = color.rgb2hsv(img01)
v = ex.equalize_hist(Ihsv[:,:,2])
Ihsv[:,:,2] = v

eq_img2 = color.hsv2rgb(Ihsv)
io.imshow(eq_img2)
io.show()

plt.hist(eq_img2.flatten(), bins = 256)
plt.show()
