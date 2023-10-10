import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def rgb_to_hsi(rgb_image):
    r, g, b = rgb_image[:, :, 0] / 255.0, rgb_image[:, :, 1] / 255.0, rgb_image[:, :, 2] / 255.0
    intensity = (r + g + b) / 3.0
    minimum = np.minimum(r, np.minimum(g, b))
    saturation = 1.0 - (3.0 / (r + g + b + 1e-6)) * minimum
    angle_numerator = 0.5 * ((r - g) + (r - b))
    angle_denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    angle_denominator[angle_denominator == 0] = 1e-6  # Avoid division by zero
    hue = np.arccos(np.clip(angle_numerator / angle_denominator, -1, 1))
    hue[b > g] = 2 * np.pi - hue[b > g]
    hue /= (2 * np.pi)
    return np.stack([hue, saturation, intensity], axis=2)

# Load a color image
image = io.imread('color.tiff')
io.imshow(image)
io.show()

hsv_image = color.rgb2hsv(image)
hsi_image = rgb_to_hsi(image)
lab_image = color.rgb2lab(image)

# Separate color channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Plot original image and its components
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(hsv_image)
plt.title('HSV Image')
plt.subplot(2, 3, 2)
plt.imshow(hsi_image, vmin=0, vmax=1)
plt.title('HSI Image')
plt.subplot(2, 3, 3)
plt.imshow(lab_image)
plt.title('CIE L*a*b* Image')
plt.subplot(2, 3, 4)
plt.imshow(red_channel)
plt.title('Red Channel')
plt.subplot(2, 3, 5)
plt.imshow(green_channel)
plt.title('Green Channel')
plt.subplot(2, 3, 6)
plt.imshow(blue_channel)
plt.title('Blue Channel')

plt.tight_layout()
plt.show()
