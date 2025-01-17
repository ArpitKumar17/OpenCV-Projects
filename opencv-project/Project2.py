import os
import time

import cv2
import mediapipe as mp
import numpy as np

import HandDetectionModule as HDM

wCam,hCam =640,480
tipIds = [4,8,12,16,20]
detector = HDM.handDetector(detectionCon= 0.75)
cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
while True:
  success,img = cap.read()
  img = detector.findHands(img)
  lmList = detector.findLandMark(img,draw=False)
  if len(lmList)!=0:
    fingersUp = []
    if lmList[tipIds[0]][1]>lmList[tipIds[0]-1][1]:
      fingersUp.append(1)
    else:
        fingersUp.append(0)
    for id in range(1,5):
      if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
        fingersUp.append(1)
      else:
        fingersUp.append(0)
    totalFingers =  fingersUp.count(1)
    cv2.rectangle(img,(20,225),(170,425),(0,255,0),-1)
    cv2.putText(img, str(int(totalFingers)), (45, 375),
              cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

  cTime = time.time()
  fps = 1/(cTime- pTime)
  pTime = cTime
  cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
              cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  cv2.imshow("Image",img)
  cv2.waitKey(1)
