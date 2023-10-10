import numpy as np
import skimage.io as io
import skimage.color as co
import matplotlib.pyplot as plt

def polar2im(image, r, theta):
    rows, cols = image.shape
    x, y = polar2cart(r, theta)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    x = clip(x, 0, cols - 1)
    y = clip(y, 0, rows - 1)
    indices = np.ravel_multi_index((y, x), (rows, cols))
    ripple_image = image.flat[indices].reshape((rows, cols))
    return ripple_image

def clip(x, y, z):
    x[x < y] = y
    x[x > z] = z
    return x

# Function to convert Cartesian to polar coordinates
def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# Load the image
f = io.imread('iris1.jpg')

# Convert to grayscale
fg = co.rgb2gray(f)

rows, cols = fg.shape

# Vertical ripples
y, x = np.mgrid[0:cols, 0:rows]
x2 = clip(x + x % 32, 0, rows - 1)
ripple1 = np.reshape(fg[x2.ravel(), y.ravel()], (rows, cols)).T

# Horizontal ripples
y2 = clip(y + y % 32, 0, cols - 1)
ripple2 = np.reshape(fg[x.ravel(), y2.ravel()], (rows, cols)).T

# Both vertical and horizontal ripples
ripple3 = np.reshape(fg[x2.ravel(), y2.ravel()], (rows, cols)).T

# Ripples on pond 
r, theta = cart2polar(x, y)
r2 = r + np.mod(r, 30)
ripple4 = polar2im(fg,r2,theta)

# Display the images
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.axis('off')
plt.imshow(ripple1, cmap='gray')
plt.title('Vertical Ripple')

plt.subplot(2, 2, 2)
plt.axis('off')
plt.imshow(ripple2, cmap='gray')
plt.title('Horizontal Ripple')

plt.subplot(2, 2, 3)
plt.axis('off')
plt.imshow(ripple3, cmap='gray')
plt.title('Vertical and Horizontal Ripple')

plt.subplot(2, 2, 4)
plt.axis('off')
plt.imshow(ripple4, cmap='gray')
plt.title('Radial Ripple')

plt.tight_layout()
plt.show()
