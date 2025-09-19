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

a=cv2.imread('sun.png').astype(np.float32)/255.0
o=cv2.imread('moon.png').astype(np.float32)/255.0
h,w=a.shape[:2]; o=cv2.resize(o,(w,h))

m=np.zeros((h,w),np.float32)
r=int(0.3*min(h,w))
cv2.circle(m,(w//2,h//2),r,1.0,-1)

n=6
la,ga=lstack(a,n,sigma=1.5,grow=1.4)
lo,go=lstack(o,n,sigma=1.5,grow=1.4)
gm=gstack(m,n,sigma=12.0,grow=1.8)
lr=blend(la,lo,gm)
rimg=recon(lr)
cv2.imwrite('sun_moon_circle_blend.png',(rimg*255).astype(np.uint8))