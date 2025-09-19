import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave

img = cv2.imread("taj.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blur = cv2.GaussianBlur(img, (9,9), 3) #blur

sharp = cv2.addWeighted(img, 3, blur, -2, 0)

imsave("results/original.png", img)
imsave("results/blurred.png", blur)
imsave("results/sharpened.png", sharp)