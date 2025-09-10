import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from skimage.filters import sobel
from skimage.morphology import binary_opening, binary_closing, remove_small_holes, remove_small_objects, square
from skimage.measure import label, regionprops

def overlap_pair(a, b, dy, dx):
    H, W = a.shape
    y0a = max(0,  dy); y1a = min(H, H + dy)
    x0a = max(0,  dx); x1a = min(W, W + dx)
    y0b = max(0, -dy); y1b = min(H, H - dy)
    x0b = max(0, -dx); x1b = min(W, W - dx)
    if y1a <= y0a or x1a <= x0a: return None, None
    return a[y0a:y1a, x0a:x1a], b[y0b:y1b, x0b:x1b]

def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return -np.inf
    return float((a.ravel() @ b.ravel()) / (na*nb))

def local_search(ref, mov, cx, cy, radius):
    best = (-1e9, cx, cy)
    for ddy in range(-radius, radius+1):
        for ddx in range(-radius, radius+1):
            dy, dx = cy + ddy, cx + ddx
            A, B = overlap_pair(ref, mov, dy, dx)
            if A is None: continue
            s = ncc(A, B)
            if s > best[0]:
                best = (s, dx, dy)
    return best[1], best[2]

def shift_no_wrap(im, dx, dy):
    H, W = im.shape
    out = np.zeros_like(im)
    A, B = overlap_pair(out, im, dy, dx)
    if B is None: return out
    y0b = max(0, -dy); x0b = max(0, -dx)
    y0o = max(0,  dy); x0o = max(0,  dx)
    out[y0o:y0o+B.shape[0], x0o:x0o+B.shape[1]] = B
    return out

def pyramid_align(ref, mov, base_radius=15, scale=0.5, min_size=256, feature="gradient"):
    def feat(x): return sobel(x) if feature=="gradient" else x
    pyr_ref = [feat(ref.astype(np.float32))]
    pyr_mov = [feat(mov.astype(np.float32))]
    while min(pyr_ref[-1].shape) > min_size:
        pyr_ref.append(rescale(pyr_ref[-1], scale, anti_aliasing=True, channel_axis=None))
        pyr_mov.append(rescale(pyr_mov[-1], scale, anti_aliasing=True, channel_axis=None))
    dx, dy = 0, 0
    for lvl in range(len(pyr_ref)-1, -1, -1):
        r = pyr_ref[lvl]; m = pyr_mov[lvl]
        if lvl != len(pyr_ref)-1:
            dx = int(round(dx / scale))
            dy = int(round(dy / scale))
        dx, dy = local_search(r, m, dx, dy, base_radius)
    aligned = shift_no_wrap(mov, dx, dy)
    return dx, dy, aligned

def content_mask(img):
    H, W = img.shape
    fw = max(5, int(min(H, W)*0.02))
    frame = np.concatenate([img[:fw,:].ravel(), img[-fw:,:].ravel(), img[:, :fw].ravel(), img[:, -fw:].ravel()])
    med = np.median(frame)
    mad = np.median(np.abs(frame - med)) + 1e-6
    t = 3.5*mad + 0.02
    m = np.abs(img - med) > t
    m = binary_opening(m, square(3))
    m = binary_closing(m, square(5))
    m = remove_small_holes(m, area_threshold=max(256, (H*W)//500))
    m = remove_small_objects(m, min_size=max(256, (H*W)//500))
    lbl = label(m)
    if lbl.max()==0: return m
    regs = regionprops(lbl)
    amax = 0; bb = (0,H,0,W)
    for r in regs:
        if r.area > amax:
            amax = r.area
            minr, minc, maxr, maxc = r.bbox
            bb = (minr, maxr, minc, maxc)
    out = np.zeros_like(m, dtype=bool)
    out[bb[0]:bb[1], bb[2]:bb[3]] = True
    return out

def bbox_from_mask(m):
    ys = np.where(m.any(axis=1))[0]
    xs = np.where(m.any(axis=0))[0]
    if ys.size==0 or xs.size==0: return (0, m.shape[0], 0, m.shape[1])
    return (ys[0], ys[-1]+1, xs[0], xs[-1]+1)

def crop_to_bbox(img, bbox):
    y0,y1,x0,x1 = bbox
    return img[y0:y1, x0:x1]

def robust_contrast(img, q=1.0):
    lo = np.percentile(img, q)
    hi = np.percentile(img, 100-q)
    if hi<=lo: return np.clip(img,0,1)
    return np.clip((img-lo)/(hi-lo), 0, 1)

def gray_world_white_balance(rgb):
    means = rgb.reshape(-1,3).mean(axis=0)
    g = means.mean()
    gains = np.where(means>0, g/means, 1.0)
    out = rgb * gains
    mx = out.max()
    if mx>0: out = out/mx
    return out

def apply_color_matrix(rgb, M=None):
    if M is None: return rgb
    H,W,_ = rgb.shape
    x = rgb.reshape(-1,3)
    y = x @ M.T
    y = np.clip(y, 0, 1)
    return y.reshape(H,W,3)
imname = 'cs180 proj1 data/monastery.jpg'
im = skio.imread(imname)
im = sk.img_as_float(im)
height = np.floor(im.shape[0] / 3.0).astype(int)
b = im[:height]
g = im[height:2*height]
r = im[2*height:3*height]
mb = content_mask(b)
mg = content_mask(g)
mr = content_mask(r)
m = mb | mg | mr
bbox = bbox_from_mask(m)
b = crop_to_bbox(b, bbox)
g = crop_to_bbox(g, bbox)
r = crop_to_bbox(r, bbox)
gdx, gdy, ag = pyramid_align(b, g, base_radius=25, feature="gradient")
rdx, rdy, ar = pyramid_align(b, r, base_radius=25, feature="gradient")
print("G to B:", gdx, gdy)
print("R to B:", rdx, rdy)
rgb = np.dstack([ar, ag, b])
rgb = np.dstack([robust_contrast(rgb[...,0], q=1.0),
                 robust_contrast(rgb[...,1], q=1.0),
                 robust_contrast(rgb[...,2], q=1.0)])
rgb = gray_world_white_balance(rgb)
rgb = apply_color_matrix(rgb, M=None)
out = np.clip(rgb, 0, 1)
out_u8 = np.ascontiguousarray((out*255).round().astype(np.uint8))
skio.imsave('out.jpg', out_u8)