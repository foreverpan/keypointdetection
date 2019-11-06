# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:42:09 2019

@author: panxiang
"""

import cv2
import numpy as np


#读取视频选取ROI
cap = cv2.VideoCapture("C:\\vtest.avi")
ret, frame = cap.read()
track=cv2.selectROI("track",frame)
x,y,w,h=track
roi=frame[y:y+h,x:x+w]
cv2.imshow("roi",roi)
print(x)
print(y)
print(w)
print(h)


#ROI角点检测
gray1 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
surf = cv2.xfeatures2d.SURF_create()
keypoints1, descriptor1 = surf.detectAndCompute(gray1, None)
roijiaodian = cv2.drawKeypoints(gray1, keypoints1, gray1,(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imshow("roijiaodian",roijiaodian)






#对每一帧进行角点检测，并与ROI的角点进行匹配
surf2 = cv2.xfeatures2d.SURF_create()

#bf = cv2.BFMatcher()

flann_index_kdtree = 0
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)


while 1:
    ret, frame = cap.read()
    gray2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    keypoints2, descriptor2 = surf2.detectAndCompute(gray2, None)
    jiaodian2 = cv2.drawKeypoints(gray2, keypoints2, gray2,(0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)
    #matches = bf.knnMatch(descriptor1,descriptor2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
   

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,2.0)
    
    
    
    point1 = np.float32([[x, y],[x, y+h], [x+w, y+h], [x+w, y]]).reshape(-1, 1, 2)
    dst=cv2.perspectiveTransform(point1,M)
    frame = cv2.polylines(frame, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
    
    
    
    
    cv2.imshow("jiaodian2", jiaodian2)
    cv2.imshow("cap", frame)  
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()