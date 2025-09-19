import numpy as np
import cv2 #for reading images only
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d
def convolve(img, k):
    img = np.asarray(img, dtype=float)
    k = np.asarray(k, dtype=float)
    kx, ky = k.shape
    imgx,imgy = img.shape
    padx,pady = kx // 2, ky // 2
    img_padded = np.pad(img, ((padx, padx), (pady, pady)))
    out = np.zeros_like(img, dtype=float)
    for i in range(imgx):
        for j in range(imgy):
            s = 0.0
            for m in range(kx):
                for n in range(ky):
                    s += img_padded[i + m, j + n] * k[kx - 1 - m, ky - 1 - n]
            out[i, j] += s
    return out

def convolve_2loops(img, k):
    img = np.asarray(img, dtype=float)
    k = np.asarray(k, dtype=float)
    kx, ky = k.shape
    imgx,imgy = img.shape
    padx,pady = kx // 2, ky // 2
    img_padded = np.pad(img, ((padx, padx), (pady, pady)))
    out = np.zeros_like(img, dtype=float)
    for i in range(imgx):
        for j in range(imgy):
            patch = img_padded[i:i+kx, j:j+ky]
            out[i,j] = np.sum(patch * kernel[::-1,::-1])
    return out 

if __name__ == "__main__":
    img = cv2.imread("my_face.jpg", cv2.IMREAD_GRAYSCALE)
    
    img = img[::8, ::8]
    kernel = np.ones((10, 10), dtype=float) / 100.0
    result = convolve(img, kernel)
    plt.imsave("results/box_convolve.png", result, cmap="gray")

    result2 = convolve_2loops(img, kernel)
    plt.imsave("results/box_convolve_2loops.png", result2, cmap="gray")
    result_scipy = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    plt.imsave("results/box_convolve_scipy.png", result_scipy, cmap="gray")
    # --- vertical edge filter (Dy) ---
    kernel = np.array([[1], [0], [-1]])
    result = convolve_2loops(img, kernel)
    plt.imsave("results/edges_vertical.png", result, cmap="gray")

    # --- horizontal edge filter (Dx) ---
    kernel = np.array([[1, 0, -1]])
    result = convolve_2loops(img, kernel)
    plt.imsave("results/edges_horizontal.png", result, cmap="gray")

    

        
