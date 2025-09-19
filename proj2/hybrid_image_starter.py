import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.transform import rescale
from align_image_code import align_images
from skimage.transform import pyramid_gaussian, resize

def lowpass(im, sigma):
    return gaussian(im, sigma=sigma, channel_axis=-1, preserve_range=True)

def highpass(im, sigma):
    return im - lowpass(im, sigma)

def hybrid_image(im_hi, im_lo, sigma_hi, sigma_lo):
    lo = lowpass(im_lo, sigma_lo)
    hi = highpass(im_hi, sigma_hi)
    hybrid = np.clip(lo + hi, 0.0, 1.0)
    return hybrid, hi, lo

def fft_logmag(gray):
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mag = np.log(np.abs(Fshift) + 1e-8)
    return mag

def show_frequency_analysis(im1, im2, hi, lo, hybrid):
    g1 = rgb2gray(im1)
    g2 = rgb2gray(im2)
    ghi = rgb2gray(np.clip(hi + 0.5, 0, 1))  # recenters hi for visualization
    glo = rgb2gray(lo)
    ghy = rgb2gray(hybrid)

    figs = [
        ("Input 1 (hi-pass source)", g1),
        ("Input 2 (lo-pass source)", g2),
        ("High-pass (from input 1)", ghi),
        ("Low-pass (from input 2)", glo),
        ("Hybrid", ghy),
    ]
    plt.figure(figsize=(12, 8))
    for i, (title, g) in enumerate(figs, 1):
        plt.subplot(2, 5, i)
        plt.imshow(g, cmap='gray')
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.subplot(2, 5, i + 5)
        plt.imshow(fft_logmag(g), cmap='gray')
        plt.title("FFT log|·|", fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    im2 = plt.imread('./my_face.jpg')/255.
    im2 = im2[::3, ::3]
    im1 = plt.imread('./bo2.jpg')/255
    im1_aligned, im2_aligned = align_images(im1, im2)

    sigma1 = 8  # for high-pass on image 1
    sigma2 = 10  # for low-pass on image 2

    hybrid, hi, lo = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    # Show hybrid
    plt.figure(figsize=(6, 6))
    plt.imshow(hybrid)
    plt.title(f"Hybrid (sigma_hi={sigma1}, sigma_lo={sigma2})")
    plt.axis('off')
    plt.show()


    g1 = rgb2gray(im1)
    g2 = rgb2gray(im2)
    ghi = rgb2gray(np.clip(hi + 0.5, 0, 1))
    glo = rgb2gray(lo)
    ghy = rgb2gray(hybrid)

    figs = [
        ("Input 1 (hi-pass source)", g1),
        ("Input 2 (lo-pass source)", g2),
        ("High-pass (from input 1)", ghi),
        ("Low-pass (from input 2)", glo),
        ("Hybrid", ghy),
    ]
    plt.figure(figsize=(12, 8))
    for i, (title, g) in enumerate(figs, 1):
        plt.subplot(2, 5, i)
        plt.imshow(g, cmap='gray')
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.subplot(2, 5, i + 5)
        plt.imshow(fft_logmag(g), cmap='gray')
        plt.title("FFT log|·|", fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # # Pyramids of the hybrid
    # N = 5
    # pyramids(hybrid, N)