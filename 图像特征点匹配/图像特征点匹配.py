# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:48:33 2019

@author: panxiang
"""

import cv2
import numpy as np
print(cv2.__version__)




#读取图片
img1=cv2.imread("C:\\box.png")
img2=cv2.imread("C:\\box_in_scene.png")
cv2.imshow("img1",img1)
cv2.imshow("img2",img2)

    
#灰度转换    
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

img1jiaodian=gray1.copy()
img1jiaodian=gray2.copy()


#角点检测
surf = cv2.xfeatures2d.SURF_create()
keypoints1, descriptor1 = surf.detectAndCompute(gray1, None)
surf2 = cv2.xfeatures2d.SURF_create()
keypoints2, descriptor2 = surf.detectAndCompute(gray2, None)

img1jiaodian = cv2.drawKeypoints(gray1, keypoints1, gray1,(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img2jiaodian = cv2.drawKeypoints(gray2, keypoints2, gray2,(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.imshow("img1",img1jiaodian)
cv2.imshow("img2",img2jiaodian)





#角点匹配

#bf = cv2.BFMatcher()
#matches = bf.knnMatch(descriptor1,descriptor2, k=2)
flann_index_kdtree = 0
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptor1, descriptor2, k=2)



good = []
#src_pts=[]
#dst_pts=[]
#筛选距离近的点
for m,n in matches:
    if m.distance < 0.75*n.distance:
        #good.append([m])
        good.append(m)
       # src_pts.append(keypoints1[m.queryIdx].pt)
       # dst_pts.append(keypoints2[m.trainIdx].pt)
        
#src_pts=np.float32(src_pts)        
#dst_pts=np.float32(dst_pts)    

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,2.0)

#img3 = cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,matches,None,flags=2)
#img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good[: 60],img2,flags=2)



h, w = gray1.shape
point1 = np.float32([[0, 0],[0, h -1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst=cv2.perspectiveTransform(point1,M)
img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)



#matchesMask = mask.ravel().tolist()
#draw_params = dict(matchColor=(0, 255, 0),
#                   singlePointColor=None,
#                   matchesMask=matchesMask,
#                   flags=2)
#img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good, None, **draw_params)
img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good[: 30],None,flags=2)



cv2.imshow("img3",img3)
print(m)

key=cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()










    
    
    