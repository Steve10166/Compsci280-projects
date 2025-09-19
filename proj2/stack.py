import cv2, numpy as np

def gstack(x, n, sigma=2.0, grow=1.6):
    s=[x.astype(np.float32)]
    sig=sigma
    for _ in range(n-1):
        s.append(cv2.GaussianBlur(s[-1], (0,0), sig))
        sig*=grow
    return s

def lstack(x, n, sigma=2.0, grow=1.6):
    g=gstack(x,n,sigma,grow)
    l=[g[i]-g[i+1] for i in range(n-1)]
    l.append(g[-1])
    return l,g

def blend(L1,L2,GM):
    y=[]
    for i in range(len(L1)):
        m=GM[i][...,None]
        y.append(L1[i]*m+L2[i]*(1-m))
    return y

def recon(L):
    y=L[-1]
    for i in range(len(L)-2,-1,-1):
        y=y+L[i]
    return np.clip(y,0,1)

def to01(x):
    m=x.min(); M=x.max()
    return (x-m)/(M-m+1e-8)

a=cv2.imread('Pasted Graphic 1.jpg').astype(np.float32)/255.0
o=cv2.imread('photo-1521747116042-5a810fda9664.jpeg').astype(np.float32)/255.0
h,w=a.shape[:2]; o=cv2.resize(o,(w,h))
m=np.zeros((h,w),np.float32); m[:, :w//2]=1.0
# m = np.zeros((h,w), np.float32)
# m[:h//2, :] = 1.0
n=6
la,ga=lstack(a,n,sigma=1.5,grow=1.4)
lo,go=lstack(o,n,sigma=1.5,grow=1.4)
gm=gstack(m,n,sigma=12.0,grow=1.8)
lr=blend(la,lo,gm)
r=recon(lr)
cv2.imwrite('sunmoon.jpeg',(r*255).astype(np.uint8))

# GA=np.hstack([to01(x) for x in ga])
# GO=np.hstack([to01(x) for x in go])
# LA=np.hstack([to01(x) for x in la])
# LO=np.hstack([to01(x) for x in lo])
# GM=np.hstack([to01(x)[...,None].repeat(3,2) for x in gm])
# LB=np.hstack([to01(x) for x in lr])

# cv2.imwrite('stack_gaussian_apple.png',(GA*255).astype(np.uint8))
# cv2.imwrite('stack_gaussian_orange.png',(GO*255).astype(np.uint8))
# cv2.imwrite('stack_laplacian_apple.png',(LA*255).astype(np.uint8))
# cv2.imwrite('stack_laplacian_orange.png',(LO*255).astype(np.uint8))
# cv2.imwrite('stack_mask.png',(GM*255).astype(np.uint8))
# cv2.imwrite('stack_laplacian_blended.png',(LB*255).astype(np.uint8))