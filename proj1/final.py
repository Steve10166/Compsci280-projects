import os, glob
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift
from skimage.color import rgb2gray
from skimage import img_as_ubyte, exposure
import matplotlib.pyplot as plt
import cv2

# ----------------- Alignment -----------------
def align(ref, mov):
    s = [0.125, 0.25, 0.5, 1.0]
    tot = np.array([0.0, 0.0])
    for sc in s:
        rs = rescale(ref, sc, anti_aliasing=True, preserve_range=True)
        ms = rescale(mov, sc, anti_aliasing=True, preserve_range=True)
        ms = nd_shift(ms, tot * sc, order=1, mode='constant', cval=0.0)
        d, _, _ = phase_cross_correlation(rs, ms, upsample_factor=20)
        tot += d / sc
    return tot

# ----------------- Cropping -----------------
def trim(rgb):
    rgb = np.clip(rgb, 0, 1)
    h, w, _ = rgb.shape

    g  = rgb2gray(rgb)
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)

    t0, t1 = int(0.10*h), int(0.90*h)
    l0, l1 = int(0.10*w), int(0.90*w)
    ig, isat = g[t0:t1, l0:l1], sat[t0:t1, l0:l1]

    dark   = np.quantile(ig, 0.05)
    bright = np.quantile(ig, 0.98)               
    sref   = np.median(isat) + 0.28

    rim = (g < dark) | (g > bright) | (sat > sref)

    def scan_rows(start, step, thr, win=22):
        i, keep = start, False
        while 0 <= i < h:
            i0 = max(0, i - win) if step > 0 else i
            i1 = min(h, i + win + 1) if step > 0 else min(h, i + 1)
            p = rim[i0:i1, :].mean()
            t = thr - (0.10 if keep else 0.0)
            if p >= t:
                keep, i = True, i + step
            else:
                break
        return i

    def scan_cols(start, step, thr, win=22):
        j, keep = start, False
        while 0 <= j < w:
            j0 = max(0, j - win) if step > 0 else j
            j1 = min(w, j + win + 1) if step > 0 else min(w, j + 1)
            p = rim[:, j0:j1].mean()
            t = thr - (0.10 if keep else 0.0)
            if p >= t:
                keep, j = True, j + step
            else:
                break
        return j

    top = scan_rows(0,     +1, 0.60)
    bot = scan_rows(h - 1, -1, 0.60)
    lef = scan_cols(0,     +1, 0.60)
    rig = scan_cols(w - 1, -1, 0.60)

    top = min(max(top + 2, 0), h - 1)
    lef = min(max(lef + 2, 0), w - 1)
    bot = max(min(bot - 1, h), top + 1)
    rig = max(min(rig - 1, w), lef + 1)

    return slice(top, bot), slice(lef, rig)

def auto_white_balance(rgb, method="grayworld", percentile=95):
    img = np.clip(rgb.astype(np.float32), 0, 1)
    if method == "grayworld":
        means = img.reshape(-1, 3).mean(axis=0)
        gains = means.mean() / (means + 1e-6)
    elif method == "whitepatch":
        vals = np.percentile(img.reshape(-1, 3), percentile, axis=0)
        gains = 1.0 / (vals + 1e-6)
        gains = gains / gains[1]
    else:
        raise ValueError("method must be 'grayworld' or 'whitepatch'")
    balanced = np.clip(img * gains.reshape(1, 1, 3), 0, 1)
    return balanced
def rgb_correction(rgb, M):
    h, w, _ = rgb.shape
    out = rgb.reshape(-1, 3) @ M.T
    out = out.reshape(h, w, 3)
    return np.clip(out, 0, 1)
# ----------------- Main -----------------
os.makedirs("results", exist_ok=True)
paths = sorted(glob.glob("data/*.tif")) + sorted(glob.glob("data/*.png"))

for p in paths:
    im = skio.imread(p)
    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height:2*height]
    r = im[2*height:3*height]
    sg = align(b, g)
    sr = align(b, r)
    g2 = nd_shift(g, sg, mode='constant', cval=0.0, order=1)
    r2 = nd_shift(r, sr, mode='constant', cval=0.0, order=1)
    rgb = np.dstack([r2, g2, b])
    margin = 175
    rgb = rgb[margin:-margin, margin:-margin]
    rs, cs = trim(rgb)
    rgb = rgb[rs, cs]
    rgb = exposure.rescale_intensity(rgb, in_range='image', out_range=(0,1))
    M = np.array([
        [1.20, -0.10, -0.10],
        [-0.05, 1.10, -0.05],
        [-0.05, -0.10, 1.10]
    ])
    rgb = rgb_correction(rgb, M)
    rgb = auto_white_balance(rgb, method="whitepatch")
    
    out = os.path.join("results", os.path.splitext(os.path.basename(p))[0] + ".png") 
    print(out)
    skio.imsave(out, img_as_ubyte(rgb))