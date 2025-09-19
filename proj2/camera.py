import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2
from matplotlib.colors import hsv_to_rgb
os.makedirs("results", exist_ok=True)
img = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not read 'cameraman.png'")
img_f = img.astype(float)

Dx = np.array([[1, 0, -1]], dtype=float)
Dy = np.array([[1], [0], [-1]], dtype=float)

Ix = convolve2d(img_f, Dx, mode="same", boundary="symm")
Iy = convolve2d(img_f, Dy, mode="same", boundary="symm")
grad = np.hypot(Ix, Iy)
THR = np.percentile(grad, 90)
edges = (grad >= THR).astype(np.uint8) * 255

plt.imsave("results/10_plain_Ix.png", Ix, cmap="gray")
plt.imsave("results/11_plain_Iy.png", Iy, cmap="gray")
plt.imsave("results/12_plain_gradmag.png", grad, cmap="gray")
plt.imsave("results/13_plain_edges.png", edges, cmap="gray")




ksize = 9
sigma = 1.5
g1 = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
G = g1.dot(g1.T)

img_blur = convolve2d(img_f, G, mode="same", boundary="symm")
Ix_blur = convolve2d(img_blur, Dx, mode="same", boundary="symm")
Iy_blur = convolve2d(img_blur, Dy, mode="same", boundary="symm")
grad_blur = np.hypot(Ix_blur, Iy_blur)
THR_blur = np.percentile(grad_blur, 85)
edges_blur = (grad_blur >= THR_blur).astype(np.uint8) * 255

plt.imsave("results/20_blur_img.png", img_blur, cmap="gray")
plt.imsave("results/21_blur_Ix.png", Ix_blur, cmap="gray")
plt.imsave("results/22_blur_Iy.png", Iy_blur, cmap="gray")
plt.imsave("results/23_blur_gradmag.png", grad_blur, cmap="gray")
plt.imsave("results/24_blur_edges.png", edges_blur, cmap="gray")

DoGx = convolve2d(G, Dx, mode="full", boundary="fill")
DoGy = convolve2d(G, Dy, mode="full", boundary="fill")

plt.imsave("results/30_DoGx_filter.png", DoGx, cmap="gray")
plt.imsave("results/31_DoGy_filter.png", DoGy, cmap="gray")

Ix_dog = convolve2d(img_f, DoGx, mode="same", boundary="symm")
Iy_dog = convolve2d(img_f, DoGy, mode="same", boundary="symm")
grad_dog = np.hypot(Ix_dog, Iy_dog)
THR_dog = np.percentile(grad_dog, 85)
edges_dog = (grad_dog >= THR_dog).astype(np.uint8) * 255

plt.imsave("results/32_dog_Ix.png", Ix_dog, cmap="gray")
plt.imsave("results/33_dog_Iy.png", Iy_dog, cmap="gray")
plt.imsave("results/34_dog_gradmag.png", grad_dog, cmap="gray")
plt.imsave("results/35_dog_edges.png", edges_dog, cmap="gray")

mag = np.hypot(Ix, Iy)

theta = np.arctan2(Iy, Ix)
hue = (theta + np.pi) / (2.0 * np.pi)
vmax = np.percentile(mag, 99.0) or 1.0
val = np.clip(mag / vmax, 0, 1)
sat = np.ones_like(val)

hsv = np.stack([hue, sat, val], axis=-1)
rgb = hsv_to_rgb(hsv)

os.makedirs("results", exist_ok=True)
plt.imsave("results/orientation_hsv.png", rgb)
plt.imsave("results/gradient_magnitude.png", mag, cmap="gray")
