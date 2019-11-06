# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:19:01 2019

@author: panxiang
"""

import cv2
import numpy as np

img1=cv2.imread("C:\\quanjing1.jpg")

cv2.imshow("img1",img1)

img1zuo=img1[:,:600]
img1you=img1[:,400:]

center=(img1you.shape[1]/2,img1you.shape[0]/2)
angle=10
scale=1
M=cv2.getRotationMatrix2D(center,angle,scale)
img1you=cv2.warpAffine(img1you,M,(img1you.shape[1],img1you.shape[0]))





cv2.imshow("img1zuo",img1zuo)
cv2.imshow("img1you",img1you)




cv2.imwrite("C:\\img1zuo.jpg",img1zuo)
cv2.imwrite("C:\\img1you.jpg",img1you)




#SURF角点检测
img1zuo=cv2.imread("C:\\img1zuo.jpg")
img1you=cv2.imread("C:\\img1you.jpg")
img1zuojiaodian=img1zuo.copy()
img1youjiaodian=img1you.copy()


gray = cv2.cvtColor(img1zuo,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img1you,cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(5000)
keypoints, descriptor = surf.detectAndCompute(gray, None)
img1zuojiaodian = cv2.drawKeypoints(img1zuo, keypoints, img1zuojiaodian,(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

surf2 = cv2.xfeatures2d.SURF_create(5000)
keypoints2, descriptor2 = surf2.detectAndCompute(gray2, None)
img1youjiaodian = cv2.drawKeypoints(img1you, keypoints2, img1youjiaodian,(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("img1zuojiaodian",img1zuojiaodian)
cv2.imshow("img1youjiaodian",img1youjiaodian)


#暴力匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptor,descriptor2, k=2)

#选取距离较近的点
good = []
src_pts=[]
dst_pts=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        src_pts.append(keypoints[m.queryIdx].pt)
        dst_pts.append(keypoints2[m.trainIdx].pt)
        
src_pts=np.float32(src_pts)        
dst_pts=np.float32(dst_pts)          




img3 = cv2.drawMatchesKnn(img1zuo,keypoints,img1you,keypoints2,good,None,flags=2)


M, mask = cv2.findHomography( dst_pts,src_pts,cv2.RANSAC,2.0)
imageTransform = cv2.warpPerspective(img1you,M,(img1.shape[1],img1.shape[0]))
imageTransform[:img1zuo.shape[0],:img1zuo.shape[1]]=img1zuo
#cv2.imshow("img3",img3)
cv2.imshow("img1zuo",img1zuo)
cv2.imshow("img1you",img1you)
cv2.imshow("dst",imageTransform)
cv2.imwrite("C:\\result2.jpg",imageTransform)
key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()



































