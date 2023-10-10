import skimage.io as io
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import binary_erosion, binary_dilation

bw= io.imread('text.png')

se1 = np.ones((4, 4), dtype=bool)  # 4 by 4 square
se2 = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]], dtype=bool)  # Line, length 5, angle 45 degrees

bw1 = binary_dilation(bw, structure=se1)  # Dilate image
bw2 = binary_erosion(bw, structure=se2)    # Erode image

plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Dilated Image')
plt.imshow(bw1, cmap='gray')

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Erosion of Binary Image')
plt.imshow(bw2, cmap='gray')

plt.show()
