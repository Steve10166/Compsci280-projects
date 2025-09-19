import cv2
import numpy as np
from skimage.io import imsave
import os

os.makedirs("results", exist_ok=True)

def save_u8(path, arr01):
    imsave(path, (np.clip(arr01, 0, 1) * 255).astype(np.uint8))

taj = cv2.cvtColor(cv2.imread("taj.jpg"), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
other = cv2.cvtColor(cv2.imread("my_face.jpg"), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

sigma = 3.0
amounts = [0.5, 1.0, 1.5, 2.5]

blur_taj = cv2.GaussianBlur(taj, (0,0), sigma)
hi_taj = taj - blur_taj
sharp_taj = np.clip(taj + 1.5*hi_taj, 0, 1)

save_u8("results/taj_blurred.png", blur_taj)
save_u8("results/taj_highfreq.png", 0.5 + 0.5*hi_taj)
save_u8("results/taj_sharpened_a1.5.png", sharp_taj)

for a in amounts:
    out = np.clip(taj + a*(taj - cv2.GaussianBlur(taj, (0,0), sigma)), 0, 1)
    save_u8(f"results/taj_sharpened_a{a}.png", out)

blur_oth = cv2.GaussianBlur(other, (0,0), sigma)
hi_oth = other - blur_oth
sharp_oth = np.clip(other + 1.5*hi_oth, 0, 1)

save_u8("results/other_blurred.png", blur_oth)
save_u8("results/other_highfreq.png", 0.5 + 0.5*hi_oth)
save_u8("results/other_sharpened_a1.5.png", sharp_oth)

for a in amounts:
    out = np.clip(other + a*(other - cv2.GaussianBlur(other, (0,0), sigma)), 0, 1)
    save_u8(f"results/other_sharpened_a{a}.png", out)